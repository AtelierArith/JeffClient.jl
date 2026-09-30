# パフォーマンスの知見

計測環境: Apple M4、Julia 1.13.1、Metal 1.11.1、PyTorch 2.14.0。
モデルは `mstrasser/Jeff-Qwen3.5-0.8B` の revision
`0f212b3e72acb4dde3f7da61e925d6ab7f819990`。

## Metal 推論での slice と broadcast の割当

- `x[:, range]` などの通常の slice はデータをコピーした配列を作る。層・head・chunk のループ内で繰り返すと、中間配列とその管理オブジェクトの割当が累積する。参照だけでよい箇所は `view` / `@views` を検討する。ただし view 自体にもラッパーがあり、GPU 演算が非連続な view を扱えるかは確認が必要。
- `y = f.(x)` のような broadcast は結果配列を作る。別々の文で broadcast を繰り返すと、各段階で中間配列ができる。ドット演算を一つの式にまとめれば融合できる場合があり、既存バッファへの `y .= ...` / `broadcast!` で出力の割当を抑えられる。broadcast が常に中間配列を作るわけではない。
- Metal では、小さい slice・broadcast を多数実行すると、GPU 配列のラッパー、カーネル引数などのホスト側 heap allocation に加え、GPU カーネルの起動回数も増える。型が安定していても、この割当と起動のコストは残る。
- MPS 行列積にも、feed/result 辞書、tensor-data、Objective-C ラッパーなどの割当がある。今回の割当を slice・broadcast だけのせいにしない。GPU バッファを再利用しても、ホスト側の管理オブジェクトの割当は別に残る。
- 最適化では、head・chunk ごとの細かい演算をまとめる、作業バッファを再利用する、複数の演算を専用カーネルに融合することを検討する。変更前後のウォームアップ済み実行時間と割当を測り、数値結果も比較する。

### 今回の測定から

- 型不安定性の修正だけでは割当はほとんど減らなかった。
- MPS のバッチ内エンコード、バッファ再利用、mask の共有、アップロード改善後も、1 推論あたり約 265 万回・136 MB の Julia heap allocation が残った。同じ測定条件で Metal の中央値は約 2.49 秒から約 1.59 秒に改善した。
- 上記は Float32、batch 1、系列長 256 の測定値。Julia の heap allocation と GPU メモリの割当・使用量は区別して扱う。

## MPS とバッファ再利用

- Laya.jl と Metal 1.11.1 の汎用 MPSGraph 行列積は `alpha*(A*B) + beta*C` を構築する。beta=0 でも C を読み、`0*NaN` が結果に混ざり得るため、当初は出力を毎回ゼロ初期化した。
- 積専用の graph は `A*B` だけを構築し、C を入力に含めないため、この初期化と alpha/beta の演算を除ける。`tools/verify_metal_primitives.jl` で、出力を NaN で埋めた通常・バッチ・転置の 8 組合せが一致することを確認した（最大誤差 `1.1920929e-7`）。実モデル 12 ケース × 3 回も最大 logit 誤差 `3.3408403e-5` で通った。
- 積専用 graph の 20 回計測は中央値 0.272 秒、60,105 回 / 5,951,776 bytes。直前の 0.278 秒とは範囲が重なるため、約 2% の中央値の差を確実な速度改善とは扱わない。割当は 68,045 回から約 12% 減った。
- MPSGraph を Metal の現在の command batch にエンコードすると、行列積ごとの個別 commit を減らせる。配列だけでなく feed/result と Objective-C オブジェクトも、GPU 完了まで queue の roots に保持する必要がある。
- private バッファの再利用と shared アップロードの再利用は寿命条件が異なる。shared の CPU 書き込みは GPU 完了を確認してから行う。実装は Metal 1.11.1 の内部 API に依存している。

## 演算をまとめた結果

- DeltaNet を head・chunk ごとの配列演算から、一つの recurrent Metal カーネルに変更した。Float32 の 0.8B モデル、batch 1、系列長 256、101 active tokens、10 回のウォーム計測で、中央値は約 1.586 秒から 0.693 秒になった。Julia heap allocation は 2,647,320 回 / 135,908,480 bytes から 116,725 回 / 8,786,080 bytes に減った。
- full attention の行列積を head ごとから 3 次元配列のバッチ行列積にし、causal depthwise convolution の slice・加算・SiLU を一つのカーネルに融合した。20 回のウォーム計測で中央値 0.571 秒、67,858 回 / 6,281,696 bytes。各値は GPU 完了とスコアの CPU 返却を含み、読み込み・コンパイルを除く。
- この時点の実モデル検証は 12 ケース × 3 回、最大 logit 誤差 `3.3140182e-5`。小さい fixture の一致だけでなく、実モデルの重みでも確認した。
- 同期を挟んで単独ステージを測ると、DeltaNet attention 約 28.8 ms、full attention 約 12.2 ms、MLP 約 3.89 ms だった。単独計測は command batching が完全な推論と異なるため、足し合わせて推論時間を推定しない。
- 上記 DeltaNet カーネルは key 成分の reduction に token ごとの threadgroup barrier を使っていた。SIMD group ごとに value 行を担当させ、lane 内の固定長 tuple に状態を保持する方式で barrier を除いた。20 回の計測で中央値 0.278 秒、68,045 回 / 6,284,688 bytes。実モデル 12 ケース × 3 回の最大 logit 誤差は `3.3408403e-5`。
- 同じ Float32・入力・20 回で original Python MPS を再計測すると中央値 0.350 秒だった。この比較は batch 1、系列長 256 の準備済み入力に限る。異なる精度や batch size の結果に一般化しない。
- RMS/L2 を SIMD reduction に融合すると、20 回の中央値 0.254 秒、45,665 回 / 5,113,120 bytes になった。実モデルの最大 logit 誤差は `3.361702e-5`。
- masked softmax も scale・causal/padding mask・maximum・exp・sum・正規化を一つのカーネルにした。dense な CPU mask が不要になり、中央値 0.253 秒、44,097 回 / 3,460,368 bytes。直前と速度の差は小さいが、heap bytes は約 32% 減った。実モデル 12 ケース × 3 回の最大 logit 誤差は `3.361702e-5`。
- 上記 softmax 版の pool は計測終了時に約 3.41 GB の private バッファを保持していた。heap の削減は、GPU の保持メモリが同じ割合で減ることを意味しない。明示的 release と作業バッファの再利用は次の改善対象。

## RoPE と head 配置変換の融合

- Laya の `split_rope` を参考に、Q/K の centered RMS、partial RoPE、grouped KV head の展開、MPS 用 `(head_dim, sequence, heads)` 配置への書き込みを専用カーネルにまとめた。K の処理と同時に V を配置し、出力側の配置変換と sigmoid gate も一つにまとめた。通常の slice、`cat`、`permutedims`、KV index の GPU アップロードが full attention から消えた。
- RoPE の cos/sin は queue ごとに現在の一組だけをキャッシュする。キーは rotary width・系列長・Float32 の base。CPU 実装と同じ Float32 の周波数計算を使う。テーブルを置き換えても queued kernel が配列を保持するため、使用中の shared バッファを CPU が書き換えない。
- 単体検証では、head width 4/7/256、full/partial/no RoPE、KV の共有あり・なし、rotary width と base の変更、cache hit、出力 gate を CPU 実装と比較した。小さい fixture は最大誤差 `2.3841858e-7`。実モデル 12 ケース × 3 回の最大 logit 誤差は引き続き `3.361702e-5`。
- 同じ batch 1・長さ 256・101 active tokens・Float32、20 回の計測で中央値 **234.060896 ms**、**33,202 回 / 1,640,432 bytes** になった。直前の 253.363667 ms に対して中央値は約 7.6% 短く、割当数は約 25%、heap bytes は約 53% 減った。最小 217.6 ms、p95 312.3 ms、最大 382.3 ms でばらつきがあり、tail latency の改善とは扱わない。
- 計測終了時の private free pool は約 1.57 GB。保持量は GC・系列長・試行の履歴で変わるため、これをピーク GPU メモリの測定値とは扱わない。
- `@code_warntype` / JET の 6 対象（logits、delta/full layer、DeltaNet、linear、Metal matmul）はすべて報告なし。10% の Profile.Allocs では 3,651 サンプルを取得し、MPS の feed/result・tensor-data の 3 箇所が 1,229 サンプル（約 34%）だった。RoPE の slice 等は上位から消え、residual・MLP の個別 broadcast とカーネル起動の管理オブジェクトが残った。サンプルの割合は実行時間の割合ではない。

## Laya を再読して分かった点

- 正規化で `sum(map(abs2, values))` を使うと、`values` が 32 要素の register tuple（hidden width 1024）になったところで LLVM の `CallAnalyzer::analyze` が Bus error になった。幅 8・128・256 は通った。2 次元 grid と `@inbounds` だけでは解消しなかった。
- Laya の LayerNorm は tuple の成分を明示的な accumulator loop で足している。この形に合わせると、幅 1024 の centered/noncentered RMS も独立 CPU 参照と一致して通った。小さい fixture は、実モデルの幅に固有のコンパイラ問題を検出できない。
- keyword の Bool をそのまま `Val(centered)` にすると、型の分からない `Val` が残り、JET が fused normalization の runtime dispatch を検出した。`if centered` の各枝で `Val(true)` / `Val(false)` を渡し、呼び先の型を確定させる。
- Laya は中間配列を演算の後で明示的に release し、同じ queue の後続 GPU 演算で private バッファを再利用する。JeffClient の従来のプールは最終 owner の GC/完了待ちに依存するため、1 推論内の再利用が不足していた。明示的 release は GPU の順序と物理バッファの寿命を守って実装する必要がある。shared バッファの CPU 書き換えには、依然として GPU 完了が必要。
- Laya は RoPE の cos/sin テーブルをキャッシュし、head の分割・RoPE・レイアウト変換を一つのカーネルにまとめている。JeffClient は各 full attention 層でテーブルの生成・アップロード、slice、cat、permutedims を繰り返している。
- Laya は residual と normalization、MLP の gate をそれぞれ融合している。JeffClient の RMS と SiLU でも同様の融合を検討できる。Laya の Flash Attention は head dim 64 専用で、Jeff 0.8B の head dim 256 へそのまま適用できない。タイル幅・register 数・threadgroup memory・occupancy を再設計して測る必要がある。

## Cthulhu / TypedSyntax で MPS の割当元を追った結果

2026-10-01、Julia 1.13.1、Cthulhu 3.0.2、TypedSyntax 1.5.4、Metal 1.11.1。
診断用の `tools` 環境へ追加し、`tools/inspect_typed_source.jl` を作った。

### 「Cthulhu で解決した」の範囲

Cthulhu で確認できたのは、MPS の shape 変換が型安定でも毎回オブジェクトを生成すること。Profile.Allocs で対象を絞り、Cthulhu で呼び出し先と型推論結果を調べ、shape のキャッシュを実装して、その割当を削減した。型不安定性の修正ではなく、具体型のオブジェクト生成を繰り返す処理の修正だった。

shape キャッシュによる割当数の削減は **33,202 → 28,426 回**。その後の feed/result 辞書の修正で **24,048 回**まで減ったが、こちらは別の変更である。heap allocation 全体はまだ解消していない。Cthulhu は原因調査に役立った診断ツールであり、改善の効果は変更ごとの実測で判断する。

現時点の結論は「MPS shape の繰り返し生成という一つの原因を特定し、キャッシュで改善した」。後続の融合・in-place 化・FFI 変換の削減・重み tensor-data 再利用まで含む最新測定は **13,992 allocations / 645,648 bytes / 中央値 198.9896875 ms** だが、これを Cthulhu 単独の効果とはしない。残る主な割当元は pooled buffer に付随する DataRef/MtlArray の管理ラッパー、kernel 起動、MPS submission のコンテナ・tensor-data。型が安定していてもこれらは生成されるため、今後も Profile.Allocs と変更前後の測定が必要。

- Cthulhu は型推論結果を見ながら呼び出し先へ降りるために使った。TypedSyntax は結果を元のソースに対応づける表示に使った。これらがコードを自動修正したわけではなく、見つけた割当元をもとに実装を変更した。
- RoPE 融合後の JET 6 対象は既に報告なしで、Profile.Allocs の MPS feed/result・tensor-data 関連 3 箇所は約 34% を占めていた。この実測を出発点に、Cthulhu の対話的 descent で `MPSGraphTensorData(::MtlArray)` → `convert(MPSShape, reverse(size(matrix)))` へ降りた。
- Metal の shape 変換は `NSArray(NSNumber.(collect(tuple)))`。戻り値は具体的な `NSArray` に推論されていたが、呼び出しごとに `Vector{Int}`、`Vector{NSNumber}`、Objective-C の array を作っていた。型が確定していても、こうした明示的なオブジェクト生成の割当は残る。
- 修正では `ProductGraph` に A/B/C の immutable な shape を保持し、`graph_tensor_data(matrix, cached_shape)` から buffer・既存 shape・dtype を受け取る MPS コンストラクタを呼ぶようにした。shape は graph key のサイズと一致するので、通常・転置・バッチ行列積で共有できる。tensor-data と feed/result 辞書の生成そのものは残っている。
- 最初の shape キャッシュでは実モデル検証が `NSInvalidArgumentException`（解放済みの array に `count` を送信）で落ちた。ObjectiveC.jl の `NSArray` は `managed=false` の autoreleased wrapper であり、Julia のフィールド参照だけでは pool 終了後の Objective-C オブジェクトの生存を保証しなかった。
- shape ごとに明示的な `retain` を行い、mutable な `ProductGraph` の finalizer で対応する `release` を行う形に修正した。修正後の実モデル **12 ケース × 3 回**は GC を挟んで通り、最大 logit 誤差は **`3.361702e-5`**。単体 verifier も同じ graph を GC/pool 終了後に再利用する検証へ拡張した。
- 同じ Float32・batch 1・長さ 256・101 active tokens、20 回のウォーム計測で、shape 再利用前後の割当は **33,202 回 → 28,426 回（約 14.4% 減）**、heap bytes は **1,640,432 → 1,449,392（約 11.6% 減）**。中央値は **234.060896 → 218.520229 ms** だった。最新の min/p95/max は 216.2/291.5/365.0 ms で、前後の時間の分布は重なる。速度の差を確定的なものとは扱わず、割当削減を確認できた改善として記録する。Cthulhu の導入だけで速度が上がったとは扱わない。
- 修正後も JET の 6 対象はすべて報告なし。10% の割当プロファイルは 3,099 サンプルで、MPS feed/result・encode の 3 箇所は 687 サンプル（約 22%）だった。TypedSyntax の全 7 対象も表示でき、修正後の `graph_tensor_data` は既存 `NSArray` を受け取って具体的な `MPSGraphTensorData` を返すことを確認した。
- TypedSyntax のソース対応づけは macro・closure で不完全になり得る。実際、`@autoreleasepool` のある `batched_matmul!` では、`key = MatmulGraphKey(...)` に生成された closure の型が対応づけられた。表示だけを根拠に型の問題と判断せず、Cthulhu の `[T]yped` 表示や `code_warntype` の IR と照合する。

再利用するコマンド（REFERENCE は従来の JSON 配列・拡張 document の両方に対応）:

```sh
# 非対話: TypedSyntax のソース表示
julia --project=tools tools/inspect_typed_source.jl CHECKPOINT REFERENCE all source
# 非対話: macro 展開後の型付き IR
julia --project=tools tools/inspect_typed_source.jl CHECKPOINT REFERENCE submit typed
# 対話: 元の MPS コンストラクタから shape 変換へ降りる
julia --project=tools tools/inspect_typed_source.jl CHECKPOINT REFERENCE tensor descend
# キャッシュした shape を使う修正後のコンストラクタ
julia --project=tools tools/inspect_typed_source.jl CHECKPOINT REFERENCE cached-tensor source
```

`descend` は terminal と明示的な指定がある場合だけ起動する。
バッチ診断では `source` / `typed` を使い、メニュー入力待ちで停止させない。
参照: [Cthulhu README](https://github.com/JuliaDebug/Cthulhu.jl)、
[TypedSyntax README](https://github.com/JuliaDebug/Cthulhu.jl/blob/master/TypedSyntax/README.md)。

## MPS feed/result の Julia Dict と変換コピーを除いた結果

- shape 再利用後も、`NSDictionary(::Dict)` は `collect(keys)` / `collect(values)` とそれぞれの `NSArray` を作る。feed/result のキーは graph ごとに固定なので、`ProductGraph` に key array を保持し、値の `Vector{MPSGraphTensorData}` だけを作って Objective-C の dictionary factory へ渡す形にした。Julia Dict の backing storage と key/value のコピーが不要になった。
- キャッシュした key array は shape と同じ unmanaged な `NSArray` なので、明示的な retain/release を行う。変更ごとに作る値の vector は queue roots に残し、managed な tensor-data と GPU バッファの寿命を保つ。autoreleased な NSDictionary は、従来同様、pool 内で `encode!` に渡す。
- 通常・転置・バッチの 8 組合せ × 2 回（NaN 出力・GC 後の再利用）、RMS/RoPE の単体検証、実モデル **12 ケース × 3 回**が通った。実モデルの最大 logit 誤差は引き続き **`3.361702e-5`**。
- 同じ M4・Float32・batch 1・長さ 256・101 active tokens、20 回の計測で **24,048 回 / 1,172,384 bytes**。直前の 28,426 回 / 1,449,392 bytes に対して、割当数は約 **15.4%**、heap bytes は約 **19.1%** 減った。中央値は **218.8108335 ms**（直前 218.520229 ms）で、速度改善は確認できなかった。
- private free pool は同じ約 1.57 GB。今回の変更はホスト管理オブジェクトの削減であり、GPU の保持量は減っていない。tensor-data、値の vector、Objective-C 辞書・カーネル起動の管理オブジェクトは依然として生成する。
- 修正後の JET 6 対象はすべて報告なし。10% の割当プロファイルは 2,653 サンプルで、RMS launch は 338、residual/MLP の 4 箇所は計 400 サンプルだった（合わせて約 28%）。`tensor_dictionary` は 164 サンプル。これは割当の構成比であり、GPU 実行時間の構成比ではない。次の候補は residual + RMS と MLP gate の融合、および中間バッファの明示的な再利用。

## 最終正規化と mask コピーの削減

- 各層に渡す同じ mask 行のコピーを、各 batch 行につき一度だけ作る形へ変更した。
- 最終 RMS は列ごとに独立し、readout は末尾列のみ使うため、末尾列を取り出してから正規化する形へ変更した。全系列の最終正規化バッファを作らずに済む。
- Metal の独立参照検証は系列長 1〜512 と混合言語・padding を含む 12 ケース × 3 回で数値一致を確認。最大 logit 誤差は `3.361702e-5`。
- M4・Float32・batch 1・長さ 256・101 active tokens の同期付き 20 回計測で、割当は **24,048 → 23,979 回**、heap bytes は **1,172,384 → 1,123,808**（約 4.1% 減）。中央値は **218.8108335 → 219.297833 ms**、新しい min/p95/max は 218.1/231.3/355.1 ms。速度改善は確認できない。記録は `artifacts/metal-validation/benchmark-metal-last-rms.json`。
- Laya の `residual_norm` は residual 出力と正規化出力を同じ reduction kernel で生成する。Jeff でも post-attention の加算と RMS の融合を次の候補とする。Laya の LayerNorm と Jeff の RMS は統計計算が異なるため、カーネルをそのまま移植せず RMS の数値順序を維持する。

## residual と RMS の融合

- `native_residual_rms` を追加し、Metal では既存の RMS カーネル内で residual を加算・保存してから正規化する。二つの出力バッファは必要だが、独立した加算 broadcast の起動を各層で一回減らす。
- 通常の RMS/L2 は `Val(false)`、融合版は `Val(true)` で分岐をコンパイル時に確定する。幅 4096 を超える入力は汎用実装に戻す。
- 実モデルの独立参照検証（12 ケース × 3 回）は通り、最大 logit 誤差 `3.361702e-5`。ログは `/private/tmp/jeff-residual-rms-validation.log`。
- 同一条件の 20 回計測で **23,979 → 23,103 allocations**、**1,123,808 → 1,093,344 heap bytes**。中央値は **219.297833 → 219.795417 ms** で速度改善は確認できない。min/p95/max は 218.5/232.1/356.6 ms。記録は `artifacts/metal-validation/benchmark-metal-residual-rms.json`。
- 一方、private free pool は **1,565,655,040 → 3,270,836,224 bytes** に増加した。従来の generic broadcast 出力だった residual も private pool 出力へ変わり、forward 中の中間配列を早期返却していない。ホスト割当の削減だけでは保持 GPU メモリの改善にならない。最終採用の判断には中間バッファの寿命短縮と再計測が必要。

## 層内の private pool 配列の早期返却

- Laya の `release_one!` と Metal 1.11.1 の `record_operation!` を確認。queue roots は配列オブジェクトを保持するが、別の DataRef 所有権を取得するわけではない。自前 pool は物理 MTLBuffer を保持し、同じ queue の後続演算だけに再利用する。
- `native_release_temporary` は自前 `ReturnBuffer` と現在の queue が一致する場合だけ `Metal.unsafe_free!` を呼ぶ。共有アップロード、重み、外部の Metal 配列は返却しない。汎用 CPU 実装では何もしない。
- 層内の正規化出力、attention 出力、gate/up 射影、residual、down 射影を最後の消費処理の投入後に返却する。呼び出し元の入力 x はこの関数で返却しない。
- 独立参照の 12 ケース × 3 回は GC を挟んで通り、最大 logit 誤差は `3.361702e-5`。ログは `/private/tmp/jeff-early-return-validation.log`。保持量・割当・速度の変更後の計測は未実施。
- その後の 20 回計測では **23,439 allocations / 1,101,408 bytes / 中央値 222.1264165 ms**。pool misses は 2352 → 1796 に減ったが、free bytes は **3,270,836,224 → 4,246,929,408** に増加。目的の保持量削減に失敗し、この早期返却変更は撤回した。記録は `artifacts/metal-validation/benchmark-metal-early-return.json`。pool のサイズ別保持と GC/試行境界を含む寿命設計が必要で、返却追加だけでメモリ改善を主張できない。

## MLP の SiLU と up 乗算の融合

- `native_mlp_gate` で `native_silu.(gate) .* up` を一回の broadcast に融合した。Metal は pooled output に書き込み、SiLU 単独の中間配列と起動を除いた。CPU の汎用実装も融合した式を使う。
- 実モデル 12 ケース × 3 回は通り、最大 logit 誤差は `3.361702e-5`。同じ M4/F32/B1/L256/101 active tokens の 20 回計測は **21,943 allocations / 1,038,928 bytes / 中央値 214.692375 ms**（min/p95/max 212.6/223.8/346.0 ms）。直前の residual+RMS 版から割当数は約 5.0% 減った。中央値は約 2.3% 減ったが、別試行の時間差であり再現性確認は必要。
- private free pool は **1,749,647,360 bytes**。サイズ別内訳を `metal_pool_stats().free_buckets` に追加した。2 MiB × 247 個、3.5 MiB × 135 個、6 MiB × 70 個、1 MiB × 226 個で保持量の大部分を占める。割当のサイズだけでなく保持する個数を制御する必要がある。試行終了後の値は peak/resident GPU メモリを直接表さず、GC と回収タイミングにも依存する。
- 計測結果は `artifacts/metal-validation/benchmark-metal-mlp-gate.json`、検証ログは `/private/tmp/jeff-mlp-gate-validation.log`。
- 融合後の Profile.Allocs 10% は 2,551 サンプル。pool の生成 3 箇所で 541、通常 RMS launch 270、residual RMS launch 79、MLP gate launch 68、tensor dictionary 178、DeltaNet の decay 式 234 サンプル。これは割当比率であって実行時間比率ではない。ログは `/private/tmp/jeff-mlp-gate-profile.log`。JET は引き続き 6 対象で報告なし。
- 通常 RMS/L2 の起動では residual 用の二つのダミー MtlArray 引数を `nothing` に変更した。`Val(false)` で該当分岐は除かれるため配列の引数変換・保持は不要。変更後の効果は別計測で確認する。
- 不要な引数を除いた 20 回計測は **21,943 allocations / 1,027,552 bytes / 中央値 214.029875 ms**。割当数は変わらず、heap bytes は 11,376 bytes 減った。min/p95/max は 212.8/224.9/356.7 ms、private free pool は同じ 1,749,647,360 bytes。記録は `artifacts/metal-validation/benchmark-metal-rms-args.json`。この引数変更による速度改善は確認できない。
- 最終状態でも実モデル 12 ケース × 3 回が通り、最大 logit 誤差は `3.361702e-5`。既存の MPS/RMS/RoPE プリミティブ検証も通った。実モデル検証ログは `/private/tmp/jeff-rms-args-validation.log`、プリミティブは `/private/tmp/jeff-norm-unused-primitives.log`。

## DeltaNet decay の配列単項マイナス

- `-exp.(attention.a_log)` は dot のない配列単項マイナスが broadcast 融合を切る。`-1.0f0 .* exp.(attention.a_log) .* native_softplus.(...)` へ変更して符号反転・exp・softplus・乗算を一つの broadcast にした。
- 独立参照 12 ケース × 3 回は通り、最大 logit 誤差 `3.361702e-5`。同条件 20 回で **20,489 allocations / 974,768 bytes / 中央値 217.053292 ms**。直前の 21,943 / 1,027,552 に比べ割当数約 6.6%、bytes 約 5.1% 減。min/p95/max は 215.4/224.5/369.5 ms。private free pool は同じ 1,749,647,360 bytes。
- 直前中央値 214.029875 ms から速度改善は確認できない。融合した broadcast では固定の `exp(a_log)` も列ごとに再評価される。固定係数の事前計算・保存を次に検討する。記録は `artifacts/metal-validation/benchmark-metal-decay.json`、検証ログは `/private/tmp/jeff-decay-validation.log`。

## 固定 decay 係数の事前計算

- モデル読み込み時に各 backend 上で `-1.0f0 .* exp.(A_log)` を一回計算し、attention の `a_decay` に保持する。CPU と Metal はそれぞれの Float32 exp を使い、推論中には係数の再計算を行わない。読み込み後の推論重みは固定として扱う。
- CPU fixture の 5 入力で最大誤差 `3.874302e-7`、Metal 実モデル 12 ケース × 3 回で最大誤差 `3.361702e-5`。どちらも独立参照と一致した。Metal 検証ログは `/private/tmp/jeff-decay-cache-validation.log`。
- 同条件の 20 回計測は **20,489 allocations / 972,176 bytes / 中央値 216.182792 ms**。直前と割当数は同じで bytes は 2,592 減った。min/p95/max は 215.0/223.7/350.2 ms、pool free bytes は同じ 1,749,647,360。中央値 217.053292 → 216.182792 ms の差を速度改善の確証とはしない。記録は `artifacts/metal-validation/benchmark-metal-decay-cache.json`。

## attention mask の GPU 転送の共有

- `native_prepare_mask` と `PreparedMetalMask` を追加した。各入力行で Float32 mask を一度だけアップロードし、全 attention 層で同じ device vector を使う。元の host mask も保持し、未対応幅で汎用 attention へ戻るときは host mask を渡す。単独の attention 呼び出しで host mask を渡す従来経路も使える。
- 独立参照 12 ケース × 3 回（GC と padding を含む）は通り、最大 logit 誤差は `3.361702e-5`。ログは `/private/tmp/jeff-mask-reuse-validation.log`。
- 同条件 20 回で **20,318 allocations / 942,416 bytes / 中央値 213.3363335 ms**。min/p95/max は 211.9/220.4/348.4 ms。直前の 20,489 / 972,176 から割当数 171、bytes 29,760 減。private free pool は同じ 1,749,647,360 bytes。
- trial 全体の shared upload misses/reuses は **152/425 → 14/34**。これは一推論当たりの値ではなく、ロード・warmup・BenchmarkTools の試行を含む累計。繰り返し転送の削減は確認できたが、216.2 → 213.3 ms の差の再現性は追加計測が必要。記録は `artifacts/metal-validation/benchmark-metal-mask-reuse.json`。
- 再プロファイルの JET 6 対象は報告なし。Profile.Allocs 10% は 2,374 サンプルで、pool 生成 3 箇所計 586、通常 RMS launch 250、tensor dictionary 173、Q/K/V slice 3 箇所計 352。decay 式は前回 234 → 72 サンプルに減った。サンプル比率は時間比率ではなく、抽出の揺らぎもある。次は packed QKV の切り出しと L2 正規化の融合を検討する。ログは `/private/tmp/jeff-mask-reuse-profile.log`。

## packed QKV からの Q/K 読み取りと L2 正規化

- `packed_qk_kernel!` は convolution 出力の (packed channels, tokens) を直接読み、head ごとの平方和を SIMD reduction で求めて正規化済み Q/K を出力する。Q/K の slice コピーと reshape を除いた。平方和、epsilon、sqrt、factor、除算の順序は従来の L2 kernel と同じ。V の slice はまだ残る。
- 実モデル 12 ケース × 3 回は通り、最大 logit 誤差 `3.361702e-5`。検証ログは `/private/tmp/jeff-packed-qk-validation.log`。
- 同条件 20 回で **18,092 allocations / 824,464 bytes / 中央値 203.837229 ms**。直前の 20,318 / 942,416 から約 11.0% / 12.5% 減。min/p95/max は 202.2/211.5/336.6 ms。中央値 213.3363335 → 203.837229 ms の再現性は確認が必要。記録は `artifacts/metal-validation/benchmark-metal-packed-qk.json`。
- 一方 private free pool は **6,376,996,864 bytes** を記録した。最終スナップショットは GC/試行の回収タイミングに依存するが、保持量の増加を見落とさない。ピークメモリを直接測っていないため、メモリ全体の改善はまだ主張できない。次は同期・GC 後の比較と pool の保持制御を検討する。
- ベンチマークに `metal_pool_after_gc` を追加し、trial の計測外で full GC・GPU 同期・upload 回収後の統計も残すようにした。再計測は **203.94675 ms / 18,092 allocations / 824,464 bytes**。pool は直後 **6,376,996,864**、GC 後 **6,478,266,368 bytes**。GC 後にも同程度の保持が残り、単なる回収前のスナップショット差だけでは説明できない。記録は `artifacts/metal-validation/benchmark-metal-packed-qk-gc.json`。
- 現在の private pool は allocation miss のときだけ上限超過を見て全消去する。既存サイズで reuse が続く場合、保持量を縮小する機会がない。`limit_bytes` も統計へ追加した。GPU 完了後の縮小と、forward 内の必要数を限定する設計を検討する。
- M4 の recommended working set は **19,069,665,280 bytes**、従来の pool 上限はその 1/4 の **4,767,416,320 bytes**。GC 後の保持量はこの上限を超えていた。
- `trim_completed_buffer_pool!` を追加し、`native_host` の `Array(input)` が GPU 完了を待った後、free pool の上限を超えるバッファを大きいサイズから解放する。通常は bytes 比較だけ行い、上限以下なら終了する。GPU 完了前には呼ばない。上限は live buffer と peak/resident memory の上限ではない。
- この縮小を加えた実モデル 12 ケース × 3 回は通り、最大 logit 誤差は `3.361702e-5`。ログは `/private/tmp/jeff-pool-trim-validation.log`。縮小後の保持量と速度は次に計測する。
- 縮小後の 20 回計測は **205.7833125 ms / 18,093 allocations / 824,496 bytes**。trial 直後の pool は **2,669,461,504**、GC 後は **5,547,130,880 bytes**。直前の GC 後 6,478,266,368 から減ったが、GC で遅れて返る配列は次の完了時縮小まで上限を超え得る。厳密な常時メモリ上限を実装したわけではない。記録は `artifacts/metal-validation/benchmark-metal-pool-trim.json`。
- ベンチマークには GC 後の明示的縮小を行った第三の `metal_pool_after_trim` も追加した。これらの統計処理はすべて latency/heap trial の外で行う。post-GC と post-trim を混同せず、遅延返却の影響と cache 制御の効果を分けて判断する。
- 第三の統計を含む再計測では trial/GC/trim 後がそれぞれ **2,669,461,504 / 5,547,130,880 / 4,766,990,336 bytes**。明示的縮小後に上限 **4,767,416,320 bytes** 以下になることを確認した。中央値 **208.4717085 ms**、**18,093 allocations / 824,496 bytes**。JET 6 対象も報告なし。記録は `artifacts/metal-validation/benchmark-metal-pool-trim-gc.json`、型診断は `/private/tmp/jeff-packed-qk-trim-types.log`。

## DeltaNet の packed V の直接読み取り

- recurrent kernel に V の開始行を渡し、convolution 出力から `start + (head-1)*value_dim + row` の行を直接読む形にした。V の slice コピーと reshape が不要になった。Q/K/V 切り出し用のコピー配列はすべて除去したが、正規化済み Q/K と recurrent 出力の配列は残る。
- 実モデル 12 ケース × 3 回は通り、最大 logit 誤差 `3.361702e-5`。ログは `/private/tmp/jeff-packed-v-validation.log`。
- 同条件 20 回は **17,012 allocations / 763,040 bytes / 中央値 200.6846455 ms**。直前の pool 縮小版 18,093 / 824,496 から約 6.0% / 7.5% 減。min/p95/max は 198.4/268.6/347.9 ms で tail の改善は確認できない。記録は `artifacts/metal-validation/benchmark-metal-packed-v.json`。
- pool の trial/GC/trim 後は **4,064,804,864 / 5,058,494,464 / 4,766,990,336 bytes**。縮小後は上限内に収まるが、GC 後まで常時上限を守る実装ではない。
- プリミティブ検証に幅 7/128/256 と系列長 1/9/65 の 9 組合せを追加した。packed Q/K を CPU L2 参照、V 直接読み取りを CPU の行列形式 recurrent 更新と比較し、全組合せが通った。head 比率 2、value 幅 3/7/9、ゼロ入力列、GC を含む。ログは `/private/tmp/jeff-packed-qkv-primitives.log`。
- V の直接読み取りを含む最終コードでも JET 6 対象はすべて報告なし。ログは `/private/tmp/jeff-packed-v-types.log`。

## batch 2・長さ 512 の original Python 比較

- 拡張参照 case 12（active tokens 512/256）、M4、Float32、同期・readout・CPU score return 込みで双方 20 回を順に計測。Julia は batch 行を順次処理、original Python は元の forward の batch 処理を使う。
- Julia Metal 中央値 **806.1352295 ms**、original Python MPS F32 **1338.6191045 ms**。この条件では Julia の中央値が約 39.8% 短い。original Python は FLA/causal-conv1d がなく PyTorch 参照実装を使う条件の比較であり、他の実装や入力へ一般化しない。
- Julia は **34,546 allocations / 1,545,760 bytes**、min/p95/max **780.5/988.8/1001.7 ms**。Python は **1294.4/1344.2/1422.5 ms**。数値誤差も参照許容範囲内（Python 最大 `1.7881393e-5`）。Julia pool は GC 後 **4,911,136,768**、縮小後 **4,763,287,552 bytes**。
- 記録は `artifacts/metal-validation/benchmark-metal-b2-l512.json` と `benchmark-python-b2-l512.json`。長い系列でも改善を確認したが、Julia の実 batch 化と tail latency には改良の余地がある。

## DeltaNet 出力の RMS と SiLU gate の融合

- 正規化カーネルに `Val{GATED}` specialization を追加し、RMS の weight 乗算後に `native_silu(gate)` を掛けて保存する。正規化だけの中間出力と後続 broadcast 起動を除いた。通常 RMS/L2 と residual RMS では gate に `nothing` を渡し `Val(false)` で分岐を除く。
- 実モデル 12 ケース × 3 回は通り、最大 logit 誤差は引き続き `3.361702e-5`。ログは `/private/tmp/jeff-rms-gate-validation.log`。
- 同条件 20 回で **16,227 allocations / 722,768 bytes / 中央値 199.3218955 ms**。直前の 17,012 / 763,040 から約 4.6% / 5.3% 減。min/p95/max は 197.4/228.2/337.7 ms。中央値 200.6846455 → 199.3218955 ms の差を速度改善の確証とはしない。記録は `artifacts/metal-validation/benchmark-metal-rms-gate.json`。
- pool の trial/GC/trim 後は **4,067,426,304 / 5,100,437,504 / 4,766,990,336 bytes**。保持 GPU メモリの改善は確認できない。
- RMS+gate の単体検証に幅 8/128/256/1024、3D head/token 配列、ゼロ入力列、正負の gate（約 ±12）、正負の weight、GC を挟む 2 回実行を追加した。CPU RMS と SiLU の式に一致し、既存プリミティブ検証も通った。ログは `/private/tmp/jeff-rms-gate-primitives.log`。
- 融合後も JET 6 対象は報告なし。Profile.Allocs 10% は 1,876 サンプルで、pool の DataRef/MtlArray 生成 3 箇所は計 523、tensor dictionary は 183、packed Q/K 起動 117、層末尾 residual broadcast 96、RMS gate 起動 68。管理ラッパーと MPS submission が依然として残る。サンプル比率は実行時間の比率ではない。次は層内で所有する residual への in-place 加算と reusable workspace を検討する。ログは `/private/tmp/jeff-rms-gate-profile.log`。

## 層末尾の residual 加算を in-place 化

- 層内で新規生成した residual へ `residual .+= mlp` と書き込み、同じ配列を返す形にした。呼び出し元の入力と重みは変更しない。正規化・MLP の先行読み取りを同じ queue へ投入した後の書き込みなので、処理順序を保持する。
- CPU fixture 5 入力と Metal 実モデル 12 ケース × 3 回は通った。最大 logit 誤差は CPU `3.874302e-7`、Metal `3.361702e-5`。Metal ログは `/private/tmp/jeff-residual-inplace-validation.log`。
- 同条件 20 回は **15,939 allocations / 714,320 bytes / 中央値 203.5682295 ms**。直前 16,227 / 722,768 から 288 allocations / 8,448 bytes 減。min/p95/max は 198.5/273.7/342.1 ms。直前中央値 199.3218955 ms に対して速度改善は確認できない。記録は `artifacts/metal-validation/benchmark-metal-residual-inplace.json`。
- pool の trial/GC/trim 後は **673,808,384 / 6,335,201,280 / 4,765,483,008 bytes**。snapshot に大きな差が出るため、trial 直後の値だけを GPU メモリ改善の証拠にしない。常時保持量と peak の改善は未確認。

## MLP gate 射影配列の in-place 再利用

- `native_mlp_gate!` は層内の gate 射影へ SiLU と up 乗算の結果を書き込み、同じ配列を down 射影へ渡す。従来の活性化結果用 pooled output は不要になる。関数名の `!` で gate を変更する契約を明示した。入力 hidden とモデル重みは変更しない。
- Metal 実モデル 12 ケース × 3 回は通り、最大 logit 誤差 `3.361702e-5`。ログは `/private/tmp/jeff-mlp-inplace-validation.log`。変更後の速度・割当と CPU 検証は次に行う。
- CPU fixture の 5 入力も通り、最大 logit 誤差 `3.874302e-7`。同条件の Metal 20 回計測は **15,771 allocations / 708,944 bytes / 中央値 202.657625 ms**。直前の 15,939 / 714,320 から 168 allocations / 5,376 bytes 減。min/p95/max は 199.4/216.5/433.4 ms。速度と tail latency の改善は確認できない。記録は `artifacts/metal-validation/benchmark-metal-mlp-inplace.json`。
- pool の trial/GC/trim 後は **1,712,029,696 / 5,996,544,000 / 4,765,515,776 bytes**。配列一つの生成を省いても、ホストの kernel 起動・MPS submission と GC 依存の保持量は残る。
- in-place residual/MLP を含む最終状態でも JET 6 対象は報告なし。ログは `/private/tmp/jeff-mlp-inplace-types.log`。

## NSArray の値配列変換に残るポインタ Vector

- `tensor_dictionary` の `NSArray(values)` を ObjectiveC.jl の実装で追跡した。`foundation.jl` の `NSArray(::Vector{<:Object})` は `arrayWithObjects:count:` を呼び、`syntax.jl` の `Base.cconvert(::Type{<:id}, ::Vector{<:Object})` が `idArray([pointer(obj) for obj in objs], objs)` を生成する。
- このため既存の `Vector{MPSGraphTensorData}` に加え、Objective-C ポインタの Vector が feed と result ごとに一つ生成される。具体型でも変換用コンテナの割当は消えない。MPS の値配列・tensor-data の生成をすべて除けたわけではない。
- feed 2 個・result 1 個を固定長のポインタ領域で渡す方法を次に検討する。C 呼び出し中はポインタ領域と元の managed tensor-data を GC から保護し、GPU 完了までは元の値を queue roots に保持する。ポインタ保持領域の寿命と GPU オブジェクトの寿命は別に扱う。実装変更の効果はまだ未計測。
- `Val(2)` / `Val(1)` の固定長 tuple を Ref に保持し、`GC.@preserve values storage` 内で `arrayWithObjects:count:` に渡す実装へ変更した。元の値 vector は queue roots に残し、変換用ポインタ Vector のみ除いた。固定長 Ref も必ず無割当になるとは主張せず、全体の実測で判断する。
- 既存プリミティブ検証と実モデル 12 ケース × 3 回は通り、最大 logit 誤差は `3.361702e-5`。ログは `/private/tmp/jeff-fixed-pointer-primitives.log` と `/private/tmp/jeff-fixed-pointer-validation.log`。
- 同条件 20 回で **14,179 allocations / 651,632 bytes / 中央値 202.7504165 ms**。直前の 15,771 / 708,944 から割当数約 10.1%、bytes 約 8.1% 減。min/p95/max は 198.3/217.4/369.2 ms。中央値は直前 202.657625 ms と同程度で、速度改善は確認できない。記録は `artifacts/metal-validation/benchmark-metal-fixed-pointer.json`。
- 続いて元の managed tensor-data の feed/result 値 Vector も、2 個/1 個の tuple に置き換えた。`tensor_dictionary` は `NTuple{N,MPSGraphTensorData}` を受け取る。元の tuple を queue roots に保持し、ポインタ Ref は C 呼び出し中だけ `GC.@preserve` で保護する。tuple の boxing と tensor-data 生成まで無割当になるとは主張しない。
- tuple 版の既存プリミティブ検証と実モデル 12 ケース × 3 回は通り、最大 logit 誤差は `3.361702e-5`。ログは `/private/tmp/jeff-tuple-feed-primitives.log` と `/private/tmp/jeff-tuple-feed-validation.log`。tuple 版の速度・割当は次に計測する。
- tuple 版は **14,776 allocations / 696,208 bytes / 中央値 203.4739585 ms**。直前の値 Vector＋固定長ポインタ領域版 14,179 / 651,632 より割当が増えたため、値の保持を Vector に戻した。変換用ポインタ Vector の削減は維持する。tuple 化すれば常に割当が減るわけではなく、FFI・GC 保護・queue roots を含む経路で確認する必要がある。増加の正確な内訳は未確定。
- tuple 版でも JET 6 対象は報告なし。記録は `artifacts/metal-validation/benchmark-metal-tuple-feed.json` と `/private/tmp/jeff-tuple-feed-profile.log`。型安定性だけでは割当の退行を検出できない。
- 採用する値 Vector＋固定長ポインタ領域版に戻した状態でも JET 6 対象は報告なし。ログは `/private/tmp/jeff-fixed-pointer-types.log`。公開性能表は採用版の 14,179 allocations / 651,632 bytes / 202.7504165 ms に更新した。
- 次の変更では ProductGraph に固定キーの Ref ポインタ領域を保持し、`NSDictionary dictionaryWithObjects:forKeys:count:` へ値・キーのポインタを直接渡す。値の NSArray 作成を省く。固定キーの元の NSArray は明示的 retain/release を維持し、managed tensor-data の値 Vector も従来どおり queue roots に残す。
- この直接辞書版は、GC 後の graph 再利用を含む既存プリミティブ検証を通った。ログは `/private/tmp/jeff-direct-dictionary-primitives.log`。実モデル・割当・速度の検証はまだ必要。
- 直接辞書版の実モデル 12 ケース × 3 回も通り、最大 logit 誤差は `3.361702e-5`。同条件 20 回は **14,179 allocations / 651,632 bytes / 中央値 203.7404165 ms**。min/p95/max は 201.1/241.9/511.6 ms。直前の値 NSArray 経由版と Julia allocation/bytes は同じで、速度改善も確認できない。
- ソース上は値 NSArray の作成を省いたが、Objective-C 側の割当量は Julia の BenchmarkTools では直接測れない。Julia heap が減った、あるいは native heap の実測値が減ったとは主張しない。pool の trial/GC/trim 後は **1,712,029,696 / 5,996,544,000 / 4,765,515,776 bytes** で直前と同じ。記録は `artifacts/metal-validation/benchmark-metal-direct-dictionary.json` と `/private/tmp/jeff-direct-dictionary-validation.log`。

## 重みの MPS tensor-data の再利用

- Metal の `native_linear` に重み専用の経路を追加し、固定の buffer/shape/dtype を持つ重みの MPSGraphTensorData を再利用する。activation と出力の tensor-data は引き続き毎回生成する。系列長が変わっても重みの物理 shape は変わらない。
- cache は array objectid をキーにし、所有者の WeakRef と managed tensor-data を保持する。objectid の一致だけでなく所有者の identity を確認する。所有配列の finalizer でエントリを削除し、別の所有者のエントリを誤って消さないよう確認する。
- 実モデル 12 ケース × 3 回は通り、最大 logit 誤差 `3.361702e-5`。ログは `/private/tmp/jeff-weight-tensor-validation.log`。一時的な重みで linear の数値を確認し、配列が scope を抜けた後に full GC/synchronize/full GC を行って cache エントリが消えることも確認した。
- 変更後の割当・速度と JET は次に測定する。native tensor-data は buffer のネイティブ所有権を持つため、WeakRef だけでなくエントリ削除まで確認する必要がある。
- 同条件 20 回で **13,992 allocations / 645,648 bytes / 中央値 198.9896875 ms**。直前の直接辞書版 14,179 / 651,632 から 187 allocations / 5,984 bytes 減で、重み linear 187 回の tensor-data wrapper 生成を省いた数と一致する。min/p95/max は 197.5/216.8/376.0 ms。速度差の再現性は未確認。記録は `artifacts/metal-validation/benchmark-metal-weight-tensor.json`。
- 再利用版でも JET 6 対象はすべて報告なし。ログは `/private/tmp/jeff-weight-tensor-types.log`。型が確定した wrapper を再利用することで減った割当であり、新たな型不安定性の修正ではない。
- `tools/verify_metal_primitives.jl` に継続的な検証を追加した。同じ重みで系列長 1/9/65/1 の linear を CPU 参照と比較し、GC を挟んでも同一 tensor-data を再利用することを確認する。重みが関数 scope を抜けた後には full GC・同期・full GC を行い、cache エントリの削除を確認する。追加した検証と既存プリミティブ検証はすべて通った。ログは `/private/tmp/jeff-weight-cache-primitives.log`。
- 検証追加後の再診断でも JET 6 対象は報告なし。Profile.Allocs 10% は 1,693 サンプルで、`pooled_array` の DataRef/MtlArray 生成 3 箇所が計 513（約 30.3%）、packed Q/K 起動が 141、residual RMS 起動が 85。型別では DataRef 86、RefCounted 51、Atomic 48、MPS tensor-data の Vector 46、tensor-data wrapper 45 サンプルだった。物理 GPU buffer の再利用だけでは管理オブジェクトの生成は消えない。次は同時に生存する中間配列の区別と GPU 完了条件を守る reusable workspace を検討する。サンプル比率は時間比率ではない。ログは `/private/tmp/jeff-weight-cache-profile.log`。

## 中間配列 workspace の試作

- `JEFF_METAL_WORKSPACE=1` で有効になる試作を追加した。task-local な workspace に各 `pooled_array` 呼び出し位置の配列を保持する。同じ shape の異なる位置には別スロットを割り当て、同時に生存する Q/K・residual などを上書きしない。型・shape が一致する次回呼び出しでは MtlArray/DataRef の生成を省く。
- 一入力行の scope は CPU score の readback まで含み、次行のスロット再利用は GPU 完了後に始める。例外時には同期してから scope を終了する。queue が変われば workspace を交換し、実行経路が短くなれば余った末尾スロットを削除する。CPU と通常の単体 kernel 呼び出しは従来経路を使う。
- workspace 自体が配列を保持するため、その GPU メモリは free pool 統計に含まれない。割当の減少だけでメモリ全体の改善を判断してはいけない。型診断・速度・保持メモリの測定前には既定で有効にしない。独立参照 12 ケース × 3 回の検証を `/private/tmp/jeff-workspace-validation.log` に実行中。
- 試作の独立参照 12 ケース × 3 回は完了し、最大 logit 誤差は従来と同じ `3.361702e-5`。GC と系列長変更を含む数値検証は通った。20 回の性能測定は `/private/tmp/jeff-workspace-benchmark.log`、結果の保存先は `artifacts/metal-validation/benchmark-metal-workspace.json`。
- 同じ B1/L256/F32/101 active tokens、同期・readout・CPU返却込み20回で **11,250 allocations / 556,176 bytes / 中央値 204.7517085 ms**。直前の重み tensor-data 再利用版 13,992 / 645,648 から **2,742 allocations（約19.6%）/ 89,472 bytes（約13.9%）減**。min/p95/max は 197.7/218.0/338.2 ms。直前中央値 198.9896875 ms より速度改善は確認できない。
- この測定では private pool misses は392、reuses/free bytesは0。配列が workspace に保持されるため、free bytes=0 は GPU メモリを使っていない意味ではない。`metal_pool_stats` に `workspace_arrays` と16KB単位で確保された `workspace_bytes` を追加した。初回測定は追加前なのでこれらの値は未記録。モデル重み・workspace外の配列・native MPS資源・peak/resident memoryはこの値に含まれない。
- workspace を有効にした状態でも既存 JET 6 対象はすべて報告なし。ログは `/private/tmp/jeff-workspace-types.log`。保持量の統計を追加した再測定は `artifacts/metal-validation/benchmark-metal-workspace-memory.json` に保存する。
- 再測定は **11,250 allocations / 556,176 bytes / 中央値198.5395415 ms**、min/p95/max196.8/199.3/334.8 ms。workspace は392配列・837,386,240 bytesを保持し、GC/trim後も同じ。free poolは0。速度改善の確証はなく、管理オブジェクトの割当削減は再現した。

### Profile による workspace 版のボトルネック確認

- `tools/inspect_native.jl` の CPU Profile をウォーム推論20回へ拡張し、call treeも出力するようにした（`JEFF_PROFILE_ITERATIONS` で変更可）。`JEFF_ALLOC_SAMPLE_RATE=1` で一推論の全割当を採取した。ログは `/private/tmp/jeff-workspace-profile.log`。JET6対象は報告なし。ウォーム単発のGC時間は0秒で、GC支配を示す証拠はない。
- 全割当11,359件（profiling自体の影響を含む）は、packed Q/K起動1,219、通常RMS起動803、residual RMS起動770、MLP gate781、residual加算771など。型別ではMPS tensor-data410、値Vector398、VectorのMemory398、MPSCommandBuffer199。workspace導入前に目立ったpooled DataRef/MtlArray生成は上位から消えた。kernel起動・broadcast・MPS submissionに残る管理オブジェクトが次の削減対象。
- CPU Profileのmain呼び出しは268サンプル、そのうちembedding gather104、`wait_cmdbuf!`101、queueのinflight制限・cleanup待ち95。gatherにはGPUArraysのbounds checkとbroadcastが含まれる。スタックは重なり、各値は加算不可。別スレッドの`__psynch_cvwait`/`kevent`が多数あり、全6,810サンプルをGPU演算時間の割合へ換算しない。Julia/LLVMのコンパイルスタックも一部残り、純粋な定常CPUコストを断定するには再採取が必要。ProfileはGPU kernel内部を計測しない。
- この結果を受け、workspaceが所有する行列積出力のMPS tensor-dataもcacheする試作を追加した。既存エントリは入力側でも利用できる。buffer/physical shapeが固定のslotだけを登録し、一時reshape wrapperは登録しない。slot交換・末尾削除時には対応するtensor-dataを削除する。数値検証は `/private/tmp/jeff-workspace-tensor-validation.log` に実行中で、割当削減量は未測定。
- tensor-data再利用版も独立参照12ケース×3回に合格し、最大logit誤差は `3.361702e-5`。系列長変更・GCを挟んだ再利用でも数値が一致した。`metal_pool_stats` に保持する `workspace_tensor_data` 個数を追加した。20回の性能測定を `artifacts/metal-validation/benchmark-metal-workspace-tensor.json` に保存する。
- tensor-data再利用版の同条件20回は **11,027 allocations / 549,040 bytes / 中央値199.2508335 ms**。workspaceのみの11,250 / 556,176から223 allocations / 7,136 bytes減。199個の出力tensor-dataを保持し、一部は後続入力としても再利用される。min/p95/max197.5/203.9/334.0 msで、直前中央値198.5395415 msに対する速度改善は確認できない。workspace配列は392個・837,386,240 bytesのまま。native tensor-data自体の保持bytesとresident/peak memoryは未測定。
- Metal 1.11.1の `lib/mpsgraphs/tensor.jl:36` は `initWithMTLBuffer:shape:dataType:` でmanaged MPSGraphTensorDataを生成する。workspace slotが存続する間はbuffer/shape/dtypeが固定なので、内容の更新ごとにwrapperを作り直す必要はない。slot交換・削除時にcacheも削除する。これは固定重みだけでなく、同期条件を守る再利用activationにも適用できる。
- tensor-data再利用版のJET6対象はすべて報告なし。ログは `/private/tmp/jeff-workspace-tensor-types.log`。

## embedding gather の GPU bounds check

- Profileで目立ったembedding gatherをGPUArraysの `src/host/indexing.jl` で追跡した。vectorized indexingの `checkbounds` はGPU indexに対して `all(broadcast(checkindex,...))` を実行する。CPU側でtoken IDの範囲を検証済みでも、従来経路はGPUへ転送したindexを再検証していた。
- Float32 Metal embeddingとCPUの整数Vectorに専用gatherを追加した。ID範囲はCPUで検証し、一つのMetal kernelでembeddingを読み取ってpooled outputへ書く。workspace有効時は出力wrapperも再利用する。GPU配列のscalar indexingは使わない。空のID Vectorではkernelを起動せず、無効IDはArgumentErrorにする。
- 独立参照12ケース×3回を `/private/tmp/jeff-gather-validation.log` に実行中。速度・割当改善はまだ未測定で、bounds checkの除去だけから速度改善を断定しない。
- 専用gatherの実モデル検証は完了し、12ケース×3回で最大logit誤差 `3.361702e-5`。プリミティブverifierにも幅7/128/1024、空入力、先頭/末尾/重複ID、負のID・vocabulary上限・typemax(Int64)の拒否を追加した。ログは `/private/tmp/jeff-gather-primitives.log`。
- 追加したgather単体検証と既存プリミティブ検証はすべて通った。20回の同条件性能測定を `artifacts/metal-validation/benchmark-metal-gather.json` に保存する。
- 同条件20回で **10,877 allocations / 541,856 bytes / 中央値198.9514165 ms**。直前のtensor-data再利用版11,027 / 549,040から150 allocations / 7,184 bytes減。min/p95/max197.5/199.6/200.4 ms。中央値199.2508335 msとの差から速度改善は断定しない。workspaceは393配列・838,434,816 bytes、tensor-data199個を保持する。gather出力を保持するため、workspaceのbytesは1,048,576増えた。
- 再プロファイルは `/private/tmp/jeff-gather-profile.log`。採取用 `profile_forwards` 自体を3回warmupしてから同じ関数で20回採取するよう修正し、Profile間隔1ms・flat出力のthread別表示を追加した。これにより採取ループの初回コンパイルと別threadの待機を切り分けやすくする。出力の実測確認前にbounds-checkスタックが消えたとは主張しない。
- 再採取は完了し、JET6対象は報告なし。CPU flat出力（mincount=10）には旧gatherのGPUArrays `checkbounds`/`checkindex`経路が現れなくなった。main推論スタック164サンプル中、`wait_cmdbuf!`100、inflight制限/cleanup待ち91、MPS encode29、Metal kernel launch41。これらは重なる呼び出しで加算不可。thread1のkevent3,086、thread2の条件変数待ち3,286を別表示できた。ProfileはGPU kernel内部を測らず、待機を特定のGPU演算へ帰属させることはできない。
- 全割当プロファイル10,977件ではpacked Q/K起動1,219、RMS起動803、MLP gate781、residual加算771などが残る。tensor-dataは **410→187** 件となり、workspace cacheによる223件削減と一致する。feed/result値Vector398件とMemory398件、MPSCommandBuffer199件は残る。単発GC時間は0秒。次はkernel起動に伴う管理処理・broadcast起動とMPS feed/result容器の再利用が候補。

## workspace の MPS result 値 Vector 再利用

- workspaceのtensor-data cacheを1要素の `Vector{MPSGraphTensorData}` を保持する形へ変更した。同じ出力slotのMPS submissionでは、tensor-dataに加えてresult値Vectorを再利用する。Vectorを推論中に変更せず、従来どおりqueue rootsにも保持する。feed側は毎回生成する。resultはworkspaceが所有する出力だけを保持し、モデル重みをworkspaceへ追加保持しない。
- 独立参照12ケース×3回は通り、最大logit誤差 `3.361702e-5`。ログは `/private/tmp/jeff-result-vector-validation.log`。単体verifierへ系列長9/9/1/1/65/65・GC・同じ配列/Vectorのidentity・cacheの古いエントリ削除の確認を追加した。ログは `/private/tmp/jeff-result-vector-primitives.log`。割当と速度はまだ未測定。
- 追加したworkspace検証と既存プリミティブ検証はすべて通った。20回の同条件性能測定を `artifacts/metal-validation/benchmark-metal-result-vector.json` に保存する。
- 同条件20回は **10,479 allocations / 529,120 bytes / 中央値204.7490835 ms**。専用gather版10,877 / 541,856から398 allocations / 12,736 bytes減で、199個のresult VectorとそのMemoryを省いた数に一致する。min/p95/max197.5/216.5/230.6 ms。速度改善は確認できない。workspaceは393配列・838,434,816 bytes・tensor-data199個で直前と同じ。型診断は `/private/tmp/jeff-result-vector-types.log`。
- result Vector再利用版でもJET6対象はすべて報告なし。
- forward scopeを導入したCPU経路もfixture5入力に合格し、最大誤差 `3.874302e-7`。ログは `/private/tmp/jeff-workspace-cpu-validation.log`。workspace無効の通常Metal経路は `/private/tmp/jeff-default-gather-validation.log` で独立参照検証を実行する。
- workspace無効の通常Metal経路も12ケース×3回に合格し、最大logit誤差 `3.361702e-5`。通常・workspace両方の数値検証を確認した状態で今回の変更をまとめる。workspaceは引き続きopt-inで、既定有効化や全goalの完了を意味しない。
- 通常設定の同条件20回は **13,850 allocations / 638,720 bytes / 中央値200.2005 ms**。min/p95/max198.0/207.2/243.1 ms。poolはtrial/GC/trim後 **1,706,786,816 / 5,996,544,000 / 4,765,515,776 bytes**、workspace配列0。結果は `artifacts/metal-validation/benchmark-metal-default-gather.json`。workspace採用版10,479 / 529,120に比べ、workspaceは3,371 allocations / 109,600 bytesを省くが、両測定から速度改善は確定できない。
- 段階別測定toolのMLPを現在のin-place gateへ合わせ、attentionには準備済みmaskを渡すよう修正した。段階ごとに同期するため、各時間はfull forwardへ単純加算できない。結果の保存先は `artifacts/metal-validation/benchmark-stages-current.json`。
- 段階別測定（M4/F32/L256、各10回、個別同期）は DeltaNet attention **11.170042 ms**、full attention **3.632208 ms**、MLP **4.314021 ms**、RMS **0.4515415 ms**、depthwise convolution **0.8551875 ms**、QKV projection **1.570125 ms**。DeltaNetが測定した単独attentionの中で重く、18層で使われる。full forwardはcommandをまとめるため、層数を掛けて総時間や割合を算出しない。次はDeltaNet内のrecurrentと射影・正規化を分けて測る。
- 分解測定 `benchmark-stages-delta.json` はrecurrent **4.6479995 ms**、入力4射影 **2.1765625 ms**、RMS/SiLU gate **0.49375 ms**。DeltaNet全体8.97775 ms。別々に同期するため加算不可で、今回QKV単独3.380333 msが4射影より長いなど測定間の揺らぎもある。順位の手掛かりとして扱い、変更の効果はfull forwardの同条件測定で確認する。
- recurrent threadgroupの行数1/4/8/16を比較するtoolを追加した。各設定で同じ入力の出力が一致することを確認してから同期込み測定を行う。製品側のrows=8はまだ変更していない。結果の保存先は `artifacts/metal-validation/benchmark-stages-recurrent-rows.json`。
- 行数比較は数値一致の確認を通り、各10回のrecurrent中央値は rows=8 **4.7519165 ms**、rows=1 **4.576896 ms**、rows=4 **4.901875 ms**、rows=16 **4.6105625 ms**。各設定の割当は50/2,240 bytesで同じ。小さな差かつ個別同期の結果なので速度改善を確定しない。rows=1を候補として製品側へ仮適用し、実モデル検証を `/private/tmp/jeff-recurrent-row1-validation.log` に実行する。full forwardで改善を確認できなければrows=8へ戻す。
- rows=1候補の独立参照12ケース×3回は通り、最大logit誤差 `3.361702e-5`。同期・readout・CPU返却込みの20回測定を `artifacts/metal-validation/benchmark-metal-recurrent-row1.json` に保存する。workspace有効の条件で直前のrows=8・result Vector再利用版と比較する。
- rows=1のfull forward20回は **199.2289795 ms / 10,479 allocations / 529,120 bytes**。min/p95/max197.4/203.7/203.9 ms。直前rows=8の204.7490835 msより短いが、過去のrows=8でも約199 msが出ていたため再現性は未確定。いったん製品側をrows=8に戻し、同じ最終実装の再測定を `artifacts/metal-validation/benchmark-metal-recurrent-row8-repeat.json` に保存する。単独stageの小さな差だけでは採用しない。
- rows=8再測定は **201.9345415 ms / 10,479 allocations / 529,120 bytes**、min/p95198.5/203.6 ms。rows=1との差は約1.3%で、測定分布は重なる。明確な改善とせず、製品側のrows=8を維持する。

## recurrent decay の exp 再評価の削減候補

- recurrent kernelはhead/tokenで同じ `exp(decay)` をvalue行/laneごとに評価していた。既存のdecay broadcastを `exp.(a_decay .* softplus.(...))` にし、head/tokenあたり一つのfactorを保存する候補を追加した。中間配列とbroadcast起動数は増やさず、recurrentはfactorを読む。
- kernelには `Val{PRECOMPUTED}` を追加し、従来のlog-decayを受ける単体検証・段階別toolも引き続き使えるようにした。製品側は `Val(true)` でexpを省く。数値検証は `/private/tmp/jeff-decay-factor-validation.log`。割当・速度への効果はまだ未測定。
- factor事前計算版は独立参照12ケース×3回に合格し、最大logit誤差 `3.361702e-5`。単体verifierはdecayを-12〜0へ広げ、従来のlog-decayと事前exp factorの両方をCPU recurrent参照と比較する。ログは `/private/tmp/jeff-decay-factor-primitives.log`。
- factor事前計算と従来のlog-decayを含むプリミティブ検証はすべて通った。workspace有効・同条件20回の測定を `artifacts/metal-validation/benchmark-metal-decay-factor.json` に保存する。
- factor事前計算版のfull forwardは **201.7068335 ms / 10,479 allocations / 529,120 bytes**。min/p95/max201.0/203.1/203.9 ms。直前rows=8の201.9345415 msとほぼ同じで、速度・割当改善は確認できない。事前expとkernel内expを同じ準備済み入力で比較するstageを追加し、`artifacts/metal-validation/benchmark-stages-decay-factor.json` に測定する。採用判断はまだ保留。
- 単独stageは従来kernel内exp **5.1464585 ms**、事前factor **4.830125 ms**（各10回）。単体で約6.1%短いがfull forwardへ改善が出なかったため、製品側は従来のlog-decay経路へ戻した。`Val{PRECOMPUTED}` の比較経路と単体検証は診断用に残す。exp評価回数の削減だけでは推論全体の高速化を保証しない。

## workspace の明示的な参照解放

- extension内の `clear_forward_workspace!()` を追加した。現在taskのworkspaceがactiveなら拒否し、GPU同期後にtensor-data/値Vectorと配列の参照を外してtask-local entryを削除する。モデル重みや他taskのworkspaceは変更しない。配列の物理bufferを直接freeせず、既存DataRef/queue rootsとGCによる回収を使うため、呼び出し直後のresident memory減少は保証しない。
- プリミティブverifierへ、forward途中で例外を起こしてactiveが解除されることと、明示clear後にslot/tensor-data/task-local参照が残らないことを追加した。ログは `/private/tmp/jeff-workspace-clear-primitives.log`。
- 例外・clear・GC・再利用を含む追加検証と既存プリミティブ検証はすべて通った。

## workspace の MPS feed 値 Vector 再利用候補

- 出力slotごとに2要素のfeed値Vectorを保持する候補を追加した。各submissionでleft/rightのtensor-dataを設定し、GPU完了までqueue rootsに保持する。同一scope内で別の出力slotは別Vectorを使う。
- scope終了時は完了済みfeedの2要素をそのslotが所有する出力tensor-dataへ置き換える。これによりVectorの記憶領域を再利用しながら、入力・モデル重みのnative bufferをworkspaceに残さない。slot交換・末尾削除・明示clearではfeed entryも削除する。入力bindingを書き換えるのはnormal readbackまたは例外時同期の後に限る。
- 単体verifierへfeed Vectorのidentity、scope終了後のinput binding解除、長さ変更後の古いentry削除とclearを追加した。実モデル検証は `/private/tmp/jeff-feed-vector-validation.log`。効果は未測定。
- feed Vector再利用候補の独立参照12ケース×3回は合格し、最大logit誤差 `3.361702e-5`。追加した単体検証は `/private/tmp/jeff-feed-vector-primitives.log` に実行する。
- feed Vectorのidentity・完了後のinput binding解除・系列長変更・GC・clearと既存プリミティブ検証はすべて通った。同条件20回の性能測定を `artifacts/metal-validation/benchmark-metal-feed-vector.json` に保存する。
- feed Vector再利用版は **10,081 allocations / 513,200 bytes / 中央値198.687604 ms**。直前result Vector再利用版10,479 / 529,120から398 allocations / 15,920 bytes減。199個の2要素feed VectorとそのMemoryの再利用による削減と一致する。min/p95/max197.6/199.7/200.0 ms。速度差は測定揺らぎの範囲で、追加の速度改善は確定しない。workspace配列393個・838,434,816 bytes、tensor-data199個・feed Vector199個を保持する。型診断は `/private/tmp/jeff-feed-vector-types.log`。
- feed Vector再利用版もJET6対象はすべて報告なし。

## packed Q/K 正規化の起動を一回にまとめる候補

- `packed_qk_pair_kernel!` 内で従来のQ/K正規化を順に実行し、別々のquery/key出力へ保存する。平方和・SIMD reduction・epsilon・factor・除算の式を共有し、出力配列の数は変えず、GPU起動数を2→1へ減らす。`packed_qk` の単独経路も残す。
- `packed_qk_pair` は2個のpooled配列を確保し、workspaceではそれぞれ別slotを再利用する。kernelがまとまってもquery/keyのbufferをaliasしない。
- 実モデル検証は `/private/tmp/jeff-qk-pair-validation.log`。単体verifierにも幅7/128/256・長さ1/9/65のCPU参照比較を追加した。速度・割当の効果はまだ未測定。
- Q/K一回起動版の独立参照12ケース×3回は合格し、最大logit誤差 `3.361702e-5`。追加の単体検証は `/private/tmp/jeff-qk-pair-primitives.log` に実行する。
- Q/K一回起動版と既存プリミティブ検証はすべて通った。性能はまだ未測定で、feed Vector再利用版の10,081 allocations / 513,200 bytes / 198.687604 msをこの変更後の測定値として扱わない。
- Q/K一回起動版の同条件20回は **9,444 allocations / 495,824 bytes / 中央値198.1489375 ms**。直前feed Vector再利用版10,081 / 513,200から637 allocations（約6.3%）/ 17,376 bytes（約3.4%）減。min/p95/max197.5/199.2/199.3 ms。中央値198.687604→198.1489375 msの差から追加速度改善を確定しない。結果は `artifacts/metal-validation/benchmark-metal-qk-pair.json`。型・20回CPU Profile・全割当再採取は `/private/tmp/jeff-qk-pair-profile.log`。
- Q/K統合後もJET6対象は報告なし。全割当プロファイル9,545件ではpaired Q/K起動584（旧separate 1,219）、mask broadcast821、beta799、decay793、RMS806、residual RMS780、MLP gate776、residual加算769が上位。MPS値Vector/Memoryは上位型リストから消え、tensor-data187件・command buffer199件は残る。kernel/broadcast起動管理が次の削減対象。プロファイル値はBenchmarkTools totalsとは別に扱う。
- 通常設定の最新値は `artifacts/metal-validation/benchmark-metal-default-qk-pair.json` に測定する。workspace有効時の9,444 allocationsを既定値として報告しない。
- Q/K統合後の通常設定は **13,213 allocations / 621,344 bytes / 中央値199.7354375 ms**、min/p95197.7/207.3 ms。workspace版との差は3,769 allocations / 125,520 bytes。通常版とworkspace版の速度差は確定しない。

## DeltaNet beta/decay の起動融合候補

- betaのsigmoidとdecayの `a_decay * softplus(a + dt_bias)` を単一の1D Metal kernelで計算する `delta_gates` を追加した。式は従来のscalar helperを共有する。b/a射影は別々のままで、beta/decay出力も別pooled配列を保持する。workspaceでは両出力の管理wrapperも再利用する。
- 実モデル検証は `/private/tmp/jeff-delta-gates-validation.log`。単体verifierにhead数1/3/16、長さ1/9/65、入力±100、dt_biasとa_decayの変化、GCを含むCPU式比較を追加した。割当・速度の効果はまだ未測定。
- beta/decay融合候補の独立参照12ケース×3回は合格し、最大logit誤差 `3.361702e-5`。単体検証は `/private/tmp/jeff-delta-gates-primitives.log` に実行する。
- beta/decay融合のhead数・長さ・入力±100・GCを含む単体検証と既存プリミティブ検証はすべて通った。同条件20回の性能測定を `artifacts/metal-validation/benchmark-metal-delta-gates.json` に保存する。
- beta/decay融合版は **8,196 allocations / 421,824 bytes / 中央値198.396083 ms**。直前Q/K一回起動版9,444 / 495,824から1,248 allocations（約13.2%）/ 74,000 bytes（約14.9%）減。min/p95/max197.6/199.3/199.6 ms。中央値198.1489375 msに対する追加の速度改善は確認できない。workspace保持bytesは838,434,816→839,024,640（beta/decayのpooled出力を保持するため589,824増）。保持量とheap削減を別指標として扱う。型診断は `/private/tmp/jeff-delta-gates-types.log`。
- beta/decay融合版もJET6対象はすべて報告なし。20回CPU Profileと全割当の再採取は `/private/tmp/jeff-delta-gates-profile.log`。実行中のプロファイルが現在のkernelを採取できるよう、終了まで次の製品コード変更は行わない。
- 再採取は完了し、全割当8,261件。旧beta799+decay793に対して融合delta_gatesは308件。残る上位はmask814、RMS803、residual RMS785、residual加算772、MLP gate769、paired Q/K577など。計測間に小さな揺らぎはあるが、狙った2箇所の起動管理割当の削減を確認した。

## DeltaNet mask 乗算の workspace 出力候補

- mask乗算のgeneric broadcastを `delta_masked_input` へ変更する候補を追加した。列ごとのdevice maskを一つのMetal kernelで読み、pooled outputへ保存する。workspace有効時はこの出力のMtlArray/DataRefも再利用する。CPU maskのアップロードは既存PreparedMetalMaskを共有する。
- 実モデル検証は `/private/tmp/jeff-delta-mask-validation.log`。単体verifierに幅7/128/1024・長さ1/9/65・交互mask・GCを含むCPU乗算との完全一致を追加した。割当・速度と増えるworkspace保持量はまだ未測定。
- mask専用kernelは独立参照12ケース×3回に合格し、最大logit誤差 `3.361702e-5`。単体検証は `/private/tmp/jeff-delta-mask-primitives.log` に実行する。
- mask専用kernelを含む単体検証もすべて合格した（同ログ）。mask変更後の性能は未測定であり、READMEのworkspace測定値8,196 allocations / 421,824 bytesは直前のbeta/decay融合版の値として扱う。
- mask専用kernel版のB1/L256/active101/F32・workspace有効・同期CPU返却込み20回は **7,710 allocations / 391,008 bytes / 中央値197.896167 ms**。直前8,196 / 421,824から486 allocations / 30,816 bytes減。min/p95/max196.358417/198.52825/198.910041 ms。直前中央値198.396083 msとの小差から追加速度改善を確定しない。447配列が857,899,008 device-buffer bytesを保持し、直前839,024,640から18,874,368 bytes増えた。TD/feed Vectorは各199。結果は `artifacts/metal-validation/benchmark-metal-delta-mask.json`。型・CPU Profile・全割当の再採取は `/private/tmp/jeff-delta-mask-profile.log`。
- 再採取は完了し、JET6対象すべて報告なし。全割当プロファイル7,757件ではmask起動310件（直前generic broadcast814件）となり、狙った箇所の削減を確認した。残る上位はRMS803、residual RMS785、residual加算772、MLP gate769、MPS submit597、paired Q/K577、RMS gate564。kernel launchの引数tuple・KernelState・pipeline Ref等が残る。CPU samplingのkevent待機はGPU実行待ちを含み、kernel内部の遅さや転送時間の内訳はこのProfileから確定できない。

## MLP gate 専用起動候補

- ProfileのMLP gate769件を対象に、既存in-place broadcastをFloat32 MtlMatrix専用kernelへ置き換える候補を追加した。SiLUのscalar helperと既存destinationを共有し、新しいGPU出力配列は作らない。単体verifierに幅7/128/3584、長さ0/1/9/65、入力±100程度、destination identityとGCを含むCPU参照比較を追加。実モデル12ケース×3回の検証ログは `/private/tmp/jeff-mlp-kernel-validation.log`。性能・数値検証はまだ完了していない。
- 専用MLP gateの実モデル12ケース×3回は合格し、最大logit誤差 `3.361702e-5`。単体検証は `/private/tmp/jeff-mlp-kernel-primitives.log` に実行する。速度・割当の効果はまだ未測定。
- 単体検証はすべて合格。同条件workspace B1/L256/active101/F32・20回は **7,374 allocations / 357,600 bytes / 中央値193.2450415 ms**。直前mask版7,710 / 391,008から336 allocations / 33,408 bytes減。min/p95/max192.567416/193.940625/194.16175 ms。直前中央値197.896167 msから約2.35%短縮したが、独立反復による再確認はまだない。447配列/857,899,008 bytesのworkspace保持量は増えていない。結果は `artifacts/metal-validation/benchmark-metal-mlp-kernel.json`。JETとProfile再採取は `/private/tmp/jeff-mlp-kernel-profile.log`。
- JET6対象はすべて報告なし。全割当プロファイルは7,397件で、MLP gate起動409件（旧broadcast769件）に減った。残る上位はRMS803、residual RMS785、residual加算772、MPS submit597。Metal 1.11.1の `src/compiler/execution.jl` は `@nospecialize(args::Tuple)` を渡す起動境界と、毎回作るKernelState・引数Ref・queue操作を持つ。これらの型安定な管理割当はGPU出力配列の再利用だけでは消えず、起動数の融合や引数削減を別に評価する必要がある。
- 独立プロセスで同条件20回を反復し、**7,374 allocations / 357,600 bytes / 中央値193.361146 ms**、min/p95/max192.701375/194.2895/194.579833 msを得た。最初の193.2450415 msを再現した。結果は `artifacts/metal-validation/benchmark-metal-mlp-kernel-repeat.json`。直前mask版197.896167 msより両測定とも約2.3%短いが、この改善を他の入力・機種に一般化しない。

## 残差加算の専用起動候補

- Profileのresidual加算772件を対象に、`native_residual_add!` のgeneric in-place broadcastとMetal専用kernelを分岐する候補を追加した。既存residualを更新し、新しい出力配列は作らない。単体verifierに幅7/128/1024、長さ0/1/9/65、destination identity、GC、同一配列を両入力に渡すaliasケースを追加した。実モデル検証ログは `/private/tmp/jeff-residual-kernel-validation.log`。性能・検証はまだ完了していない。
- 実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`。generic hook変更のCPU fixture5件も合格、最大 `3.874302e-7`（`/private/tmp/jeff-residual-cpu-validation.log`）。単体検証は `/private/tmp/jeff-residual-kernel-primitives.log` に実行する。性能はまだ未測定。
- 単体検証も合格。同条件workspace B1/L256/active101/F32・20回は **7,038 allocations / 324,192 bytes / 中央値193.4229585 ms**。直前MLP版7,374 / 357,600から336 allocations / 33,408 bytes減。min/p95/max192.651208/194.013417/194.424167 ms。速度は直前193.2–193.4 msと同程度で、追加改善は確認できない。workspace保持量447配列/857,899,008 bytesは変わらない。結果は `artifacts/metal-validation/benchmark-metal-residual-kernel.json`。JETとProfile再採取は `/private/tmp/jeff-residual-kernel-profile.log`。
- JET6対象は報告なし。全割当プロファイル7,037件ではresidual加算412件（旧broadcast772件）となり、狙った起動の削減を確認した。残る最大はRMS803、residual RMS785。専用起動でも引数tupleとMetalの起動状態割当は残る。通常設定の再測定ログは `/private/tmp/jeff-default-residual-kernel.log`。
- 現在の通常設定workspace無効も同条件20回を測定し、**11,185 allocations / 461,808 bytes / 中央値194.878646 ms**、min/p95/max193.15975/201.134125/243.408917 ms。load2.065312667秒、初回forward10.495385417秒（package import除外）。free poolのtrial/GC/trimは1,732,640,768/6,110,314,496/4,765,515,776 bytesで、workspace保持0。結果は `artifacts/metal-validation/benchmark-metal-default-residual-kernel.json`。workspace版7,038 / 324,192と混同しない。Python F32既存測定349.829792 msに対してこの条件では約44%短いが、最適化済みPython kernelや他機種との比較ではない。

## RMS 起動引数をまとめる候補

- RMS/residual RMS/gated RMSの共通kernelは16引数を渡していた。起動境界の引数tuple/scalar boxingを対象に、eps/factor/width/columnsを16-byte isbits `NormalizationConfig` へまとめ、6個のcompile-time flagを型パラメータに移す候補を追加した。kernelの算術・配列所有・起動数は変えない。引数は7個となるが、config生成自体も割当を起こし得るので性能改善は測定前に断定しない。単体検証ログは `/private/tmp/jeff-rms-config-primitives.log`。
- 単体検証と実モデル12ケース×3回はすべて合格、最大logit誤差 `3.361702e-5`。実モデルログは `/private/tmp/jeff-rms-config-validation.log`。同条件20回測定を `artifacts/metal-validation/benchmark-metal-rms-config.json` に保存する。
- B1/L256/active101/F32・workspace有効・同期CPU返却込み20回は **6,837 allocations / 324,192 bytes / 中央値193.186729 ms**。直前残差版7,038から201件減ったが、heap bytesは同じ324,192のまま。min/p95/max192.41025/193.784125/193.943209 ms。速度は直前193.4229585 msと同程度で追加改善は確定しない。割当数と割当容量は別指標であり、引数集約だけでは容量削減を保証しない。JET・Profile再採取ログは `/private/tmp/jeff-rms-config-profile.log`。
- JET6対象はすべて報告なし。全割当プロファイル6,836件ではRMS起動628（旧803）、residual RMS641（旧785）、gated RMS456（旧564）となった一方、config生成226件が新たに記録され、差し引き201件減となる。Float32の記録件数は456→188、Int32は441→355。配列・起動数・GPU保持量は変えていない。configを作るだけで無割当になったとは解釈しない。
- 通常設定workspace無効の同条件20回も **10,984 allocations / 461,808 bytes / 中央値194.860208 ms**、min/p95/max192.709292/202.687916/249.350792 ms。load2.066119084秒、初回forward10.451615709秒。直前通常設定11,185 / 461,808から201件減で、容量は同じ。free poolのtrial/GC/trim値も直前と同じ。結果は `artifacts/metal-validation/benchmark-metal-default-rms-config.json`。公開表の通常値とworkspace値を別々に更新した。

## 層間の residual/input RMS 融合候補

- `native_layer_outputs` で層末尾のresidualとMLP出力を別々に返し、Metal forwardでは次層input RMSのkernel内で加算とresidual更新を行う候補を追加した。同じqueue上で前層MLPの読み取り後に書き戻す。normalized出力だけ新たに必要で、大きなresidual配列を追加しない。最終層はreadout列だけ加算・RMSを行う。24層モデルの独立加算24起動を省く狙い。generic forwardは従来どおり各層を加算して返し、hidden幅4096超はgenericへfallbackする。単体verifierにresidual identityとCPU式比較を追加した。実モデル検証は `/private/tmp/jeff-residual-rms-fusion-validation.log`。性能・精度はまだ未確認。
- 実モデル12ケース×3回は合格し最大logit誤差 `3.361702e-5`。CPU fixture5件も合格、最大 `3.874302e-7`（`/private/tmp/jeff-residual-rms-fusion-cpu.log`）。単体検証は `/private/tmp/jeff-residual-rms-fusion-primitives.log` に実行する。性能は未測定。
- 単体検証も合格。同条件workspace B1/L256/active101/F32・20回は **6,482 allocations / 325,632 bytes / 中央値193.061562 ms**。直前6,837 / 324,192から355件減ったが、bytesは1,440増えた。min/p95/max192.408916/193.631542/193.652167 ms。速度は193.186729 msと同程度で追加改善は未確定。workspace447配列/857,899,008 bytesも変わらない。結果は `artifacts/metal-validation/benchmark-metal-residual-rms-fusion.json`。JET・Profile再採取は `/private/tmp/jeff-residual-rms-fusion-profile.log`。
- JET6対象は報告なし。全割当プロファイル6,482件でresidual/input RMS626、post RMS631、KernelState208件（直前231）となり、独立residual加算は上位から消えた。最終列の2つのsliceコピーをviewへ変更する追加候補を用意した。Metalのcontiguous derived arrayはDataRefを共有しoffsetを保持するが、view wrapperも割当を作るので改めて検証・測定する。
- 最終列view版も実モデル12ケース×3回に合格、最大logit誤差 `3.361702e-5`（`/private/tmp/jeff-residual-rms-view-validation.log`）。単体verifierには最後の列のview更新が親配列の他列を変更しないことと、GPU起動直後のGCを追加した。単体ログは `/private/tmp/jeff-residual-rms-view-primitives.log`。性能はまだ未測定。
- view版の単体検証も合格。同条件workspace B1/L256/active101/F32・20回は **6,366 allocations / 319,616 bytes / 中央値193.213771 ms**。copy版6,482 / 325,632から116件 / 6,016 bytes減り、融合前6,837 / 324,192に対して471件 / 4,576 bytes減。min/p95/max192.486541/193.712291/193.889292 ms。速度は193 ms付近で追加改善は未確認。workspace447配列/857,899,008 bytesは変わらない。結果は `artifacts/metal-validation/benchmark-metal-residual-rms-view.json`。JETとProfile再採取は `/private/tmp/jeff-residual-rms-view-profile.log`。
- view版JET6対象はすべて報告なし。全割当プロファイルも6,366件、起動管理やMPS submitが残る。通常設定の再測定は `/private/tmp/jeff-default-residual-rms-view.log` に保存する。
- 通常設定workspace無効の同条件20回は **10,513 allocations / 457,232 bytes / 中央値194.962271 ms**、min/p95/max192.368625/200.986459/213.083459 ms。load2.0807445秒、初回forward10.297229917秒。free poolのtrial/GC/trimは1,732,575,232/6,133,907,456/4,765,515,776 bytes。直前通常設定10,984 / 461,808から471件 / 4,576 bytes減で、速度は195 ms付近のまま。結果は `artifacts/metal-validation/benchmark-metal-default-residual-rms-view.json`。GPU内部実行が支配的かをこのCPU Profileだけで断定しない。

## 同じ open command buffer 内での MPS wrapper 再利用候補

- ProfileのMPSCommandBuffer199件を対象に、現在taskの最新MTLCommandBuffer ownerとMPS wrapper一組を保持する候補を追加した。`Metal.ensure_cmdbuf!` が同じJulia ownerを返す間だけwrapperを再利用し、flushでownerが切り替わると作り直す。pointer値だけで判定しない。各encodeのqueue rootsにもwrapperを記録するため、cache更新後も未完了GPUのwrapperは保持される。スコアreadback後とworkspace例外cleanup後にcacheのowner/command参照を外す。
- Metal 1.11.1の `src/command_batching.jl` はflush時にopen cmdbufをnothingに戻す。`lib/mps/command_buf.jl` のwrapperは外部MTLCommandBufferを包むconstructorで、明示的commitAndContinueは内部bufferを切り替えるが本実装では呼ばない。Layaも同じopen bufferへencodeするがwrapper再利用はしていない。既存プリミティブ検証は合格（`/private/tmp/jeff-mps-command-primitives.log`）。identity・GC・submit後の非再利用・readback後clearの追加検証は `/private/tmp/jeff-mps-command-identity.log`。実モデルと性能は未検証。
- identity等の追加検証と実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`（`/private/tmp/jeff-mps-command-validation.log`）。workspaceでMPSをencodeした直後に例外を投げた場合も参照解除されるassertを追加し、`/private/tmp/jeff-mps-command-exception.log` に単体検証を再実行する。性能は未測定。
- 例外cleanupを含む単体検証も合格。同条件workspace B1/L256/active101/F32・20回は **6,180 allocations / 313,664 bytes / 中央値193.21225 ms**。直前view版6,366 / 319,616から186件 / 5,952 bytes減（32 bytes×186）。min/p95/max192.188833/193.627917/193.855709 ms。中央値193.213771 msに対する追加速度改善は確認できない。workspace保持量は447配列/857,899,008 bytesのまま。結果は `artifacts/metal-validation/benchmark-metal-mps-command.json`。JET・Profile再採取は `/private/tmp/jeff-mps-command-profile.log`。
- JET6対象は報告なし、全割当Profile6,180件。MPSCommandBufferは上位15型の一覧から外れたため、inspect_nativeにMPSCommandBuffer/TD/KernelStateの件数を常に出す診断を追加し、`/private/tmp/jeff-mps-command-focused-profile.log` に再採取する。「上位から消えた」だけで無割当と判断しない。
- 通常設定workspace無効の同条件20回は **10,327 allocations / 451,280 bytes / 中央値194.6985205 ms**、min/p95/max192.619791/201.891458/206.169833 ms。load2.062734708秒、初回forward10.242352042秒。直前通常版10,513 / 457,232から186件 / 5,952 bytes減。pool保持量は直前と同じ。結果は `artifacts/metal-validation/benchmark-metal-default-mps-command.json`。
- focused再採取も完了しJET6対象は報告なし。ProfileでMPSCommandBuffer **13件 / 416 bytes**、MPSGraphTensorData187件 / 5,984 bytes、KernelState206件 / 9,888 bytesを直接確認した。旧wrapper199件 / 6,368 bytesから差186件 / 5,952 bytesで、BenchmarkToolsの削減量と一致する。生成0ではなく、各open batchごとに必要なwrapperが残る。

## 小さな kernel の重複サイズ引数を省く候補

- delta gateのhead数/length、maskのwidth/length、MLP gateとresidual addのlengthを独立scalar引数として渡す代わりに、GPU側MtlDeviceArrayのsize/lengthから取得する候補を追加した。Int32への変換と演算順・出力所有は従来と同じ。配列descriptorがすでに同じ情報を持つため、起動引数tupleとscalar boxingを減らす狙いだが、GPU側サイズ取得やcodegenの影響も測定する。単体検証はすべて合格（`/private/tmp/jeff-kernel-dims-primitives.log`）。実モデル検証は `/private/tmp/jeff-kernel-dims-validation.log`。性能は未測定。
- 実モデル12ケース×3回も合格、最大logit誤差 `3.361702e-5`。同条件20回測定を `artifacts/metal-validation/benchmark-metal-kernel-dims.json` に保存する。先に手順を更新した `jeff-metal-performance` skillもquick_validateで合格した。PythonCallを単発コマンドで使う場合も `include("tools/python_env.jl")` をusingより先に実行し、既存extern/jeff interpreterとNull Conda backendを選ぶ。
- 同条件workspace B1/L256/active101/F32・20回は **6,102 allocations / 310,688 bytes / 中央値193.227979 ms**。直前MPS wrapper版6,180 / 313,664から78件 / 2,976 bytes減。min/p95/max192.606333/193.85775/193.962208 ms。中央値193.21225 msに対する追加速度改善はない。workspace447配列/857,899,008 bytesは変わらない。結果は上記JSON。JET・Profile再採取は `/private/tmp/jeff-kernel-dims-profile.log`。
- JET6対象は報告なし、全割当Profileも6,102件。Int32の記録は331→253で78件減り、BenchmarkToolsの割当数差と一致する。MLP gate391（旧415）、delta gate293（旧311）、mask275（旧311）。MPS wrapper13、TD187、KernelState206は変わらない。不要なサイズscalarのboxingを省いた効果であり、起動数を減らした結果ではない。
- 通常設定workspace無効の同条件20回は **10,249 allocations / 448,304 bytes / 中央値195.248833 ms**、min/p95/max192.721667/202.966541/209.459875 ms。load2.031935125秒、初回forward10.189492083秒。直前通常設定10,327 / 451,280から78件 / 2,976 bytes減。pool保持量は直前と同じ。結果は `artifacts/metal-validation/benchmark-metal-default-kernel-dims.json`。通常・workspaceとも速度は従来と同程度。現在の実装をB2/L512へ広げた測定は `/private/tmp/jeff-current-b2-l512.log` に保存する。
- 現在のworkspace有効版を独立参照case12のB2/L512/active512・256/F32で20回測定し、**12,524 allocations / 637,824 bytes / 中央値767.3309375 ms**、min/p95/max756.940125/773.256583/773.753208 ms。447配列が1,766,096,896 bytesを保持する。旧workspace導入前806.1352295 msより約4.8%短いが、複数変更と設定差を含むため最後のサイズ引数変更の効果と扱わない。GPU上でもbatchは行ごとの逐次処理。結果は `artifacts/metal-validation/benchmark-metal-current-b2-l512.json`。original Pythonの同一case12/F32/20回の再測定は `/private/tmp/jeff-python-current-b2-l512.log`。
- original Python MPS F32の同一case12・20回は **中央値1300.984021 ms**、min/p95/max1291.082625/1327.641999997/1333.523667003 ms、最大active logit誤差 `1.788139343e-5`。今回Julia767.3309375 msは約41.0%短い（約1.70倍throughput）。結果は `artifacts/metal-validation/benchmark-python-current-b2-l512.json`。Pythonは実際のbatch forward、Juliaは行ごとのforwardであり、同じ入力・F32・CPU返却込みの比較。Python側は引き続きFLA/causal-conv1d未導入の参照fallbackなので、最適化済みPython CUDAなどとの一般的な優劣は示さない。
- 残るpipeline Ref割当の原因として、Metal 1.11.1 `src/compiler/execution.jl:209` のmtlfunctionがcompiled resultをcacheから取得した後でも `Ref{MTLComputePipelineState}()` を毎回作ることを確認した。HostKernel自体は後段のkernel_instancesで再利用するが、その手前のRefは残る。compiled kernel handleを再利用する候補を評価するときはdevice・argument types・method更新の無効化とqueue rootsを守り、追加cache lookupの割当も比較する。

## 送信バッチと kernel handle cache の比較候補

- 製品コードを変えず、workspace B1/L256/F32・20回を `JULIA_METAL_COMMAND_BATCHING_OPS=128` / `JULIA_METAL_COMMAND_BATCHING_BYTES=268435456` で測定した。**5,992 allocations / 296,144 bytes / 中央値193.238667 ms**、min/p95/max192.610042/194.055542/194.961375 ms。既定32ops版6,102 / 310,688 / 193.227979 msから110件 / 14,544 bytes減ったが速度は同程度。結果は `artifacts/metal-validation/benchmark-metal-batch128.json`。両設定を同時に変更したため各設定の単独効果は分からず、既定値は変更しない。
- `ext/metal_kernels.jl` にdefault compiler optionsのsingleton kernel専用handle cacheを試作し、まずMLP gateの24起動だけに適用する。device identity・GPU引数型・Julia world counterで判定し、method更新時に全handleを破棄する。closure等のstateful functionは通常mtlfunctionへfallbackし、実際の起動はMetal HostKernel callableを使うのでqueue roots/encoder/synchronizationを迂回しない。GC preserveも通常macroと同様に行う。追加cache lookupの費用も含めて比較する。単体検証ログは `/private/tmp/jeff-kernel-handles-primitives.log`。
- 単体検証は合格し、追加のGC後handle identityとGPU関数methodを書き換えて出力1→2へ変わる検証も合格（`/private/tmp/jeff-kernel-handles-world.log`）。実モデル検証は `/private/tmp/jeff-kernel-handles-validation.log`。キャッシュがmethod更新を無視して古いGPUコードを起動しないことを直接確認した。性能は未測定。
- MLP handle cache版の実モデル12ケース×3回も合格、最大logit誤差 `3.361702e-5`。20回の性能比較を `artifacts/metal-validation/benchmark-metal-kernel-handles-mlp.json` に保存する。送信設定は既定32ops/64MBで、先の128ops実験と混同しない。
- 最初のMLP handle cache候補は **6,366 allocations / 323,744 bytes / 中央値193.2442085 ms**。直前6,102 / 310,688から264件 / 13,056 bytes増えた。速度も変わらない。cache lookupと呼び出し境界の追加費用がpipeline Ref削減を上回っており、この値を改善として採用しない。追加Profileは `/private/tmp/jeff-kernel-handles-mlp-profile.log`。
- 最初の案もJET6対象は報告なし。Profileは6,366件でpipeline Ref206→182、Core.Box208→184と狙った24起動分は減ったが、新しいlaunch helper側が511件を作った。helperへ直接code_warntypeを行ってもBody::Nothingだった（`/private/tmp/jeff-kernel-handles-launch-types.log`）。型推論結果だけでは呼び出し境界の割当増加を判断できない。
- helperを `f::F, args::Vararg{Any,N} where {F,N}` としてFunction型・Vararg数の特殊化を明示すると、同条件20回で **6,054 allocations / 309,920 bytes / 中央値193.105792 ms** となった。最初の非明示版から312件 / 13,824 bytes減り、従来macro版6,102 / 310,688より48件 / 768 bytes減。min/p95/max192.47925/193.471375/193.624 ms。追加速度改善は未確認。Function/Varargを転送するhelperではJuliaの特殊化heuristicを測定で確認する必要がある。結果は `artifacts/metal-validation/benchmark-metal-kernel-handles-specialized.json`。再採取は `/private/tmp/jeff-kernel-handles-specialized-profile.log`。
- 特殊化明示版もJET6対象は報告なし、全割当Profile6,054件。MLP launch helperは343件で従来macro側391件より48件減った。pipeline RefとCore.Box各24件の削減に対応する。次の候補ではdelta gateとmaskにもhandle cacheを適用し、同じclosure型でcapture値3/4を変えた場合のfallbackを単体verifierへ追加する。ログは `/private/tmp/jeff-kernel-handles-simple-primitives.log`。
- MLP/delta gate/maskへ適用した候補も単体検証・capture値3/4のclosure fallback・method更新と、実モデル12ケース×3回に合格（最大logit誤差 `3.361702e-5`、`/private/tmp/jeff-kernel-handles-simple-validation.log`）。同条件20回測定を `artifacts/metal-validation/benchmark-metal-kernel-handles-simple.json` に保存する。
- 3種類のhandle cache版は **5,982 allocations / 308,768 bytes / 中央値193.034521 ms**。従来macro版6,102 / 310,688から120件 / 1,920 bytes減（60起動×pipeline Ref/Core.Box各1件）。min/p95/max192.13775/193.436375/193.585792 ms。速度は従来と同程度。workspace447配列/857,899,008 bytesは変わらない。JET・Profile再採取は `/private/tmp/jeff-kernel-handles-simple-profile.log`。
- 3種類適用後もJET6対象は報告なし、全割当Profile5,982件。pipeline Ref206→146、Core.Box208→148で各60件の削減を直接確認した。KernelState206、MPS wrapper13、TD187は変わらない。GPU起動処理全体を省いたのではなく、コンパイル済みhandleを取得する手前の管理オブジェクトを減らした。
- 通常設定workspace無効の同条件20回は **10,129 allocations / 446,384 bytes / 中央値194.8515 ms**、min/p95/max192.92525/201.833834/217.379125 ms。load2.036597708秒、初回forward10.264693334秒。直前通常設定10,249 / 448,304から120件 / 1,920 bytes減。pool保持量は直前と同じ。結果は `artifacts/metal-validation/benchmark-metal-default-kernel-handles-simple.json`。現在のB2/L512再測定ログは `/private/tmp/jeff-kernel-handles-b2-l512.log`。
- B2/L512/active512・256/F32/workspace有効・20回も **12,284 allocations / 633,984 bytes / 中央値762.7117295 ms**、min/p95/max757.364042/768.880959/769.938375 ms。直前12,524 / 637,824から240件 / 3,840 bytes減（B1の2倍）。workspace447配列/1,766,096,896 bytesは同じ。767.3309375 msとの分布は重なるため、この小差を追加速度改善と確定しない。結果は `artifacts/metal-validation/benchmark-metal-kernel-handles-b2-l512.json`。GPU handle/MPS wrapper再利用とFunction/Vararg転送の検証手順を既存jeff-metal-performance skillにも追加し、quick_validateは合格した。測定値はskillへ複写していない。

## RMS 系への kernel handle cache 適用候補

- shared normalization kernelの4起動箇所（通常RMS、post residual RMS、gated RMS、次層input residual RMS）にも既存handle cacheを適用する候補を追加した。device/argument-type/worldによる判定と実際のMetal HostKernel起動を共有し、GPU算術・threadgroup配置・配列所有は変えない。NormalizationConfigの型パラメータもGPU引数型keyに含まれる。単体検証ログは `/private/tmp/jeff-kernel-handles-rms-primitives.log`。性能・実モデル検証は未完了。
- 直接置換の単体検証は合格したが、JETで4対象にruntime dispatchを検出した（`/private/tmp/jeff-kernel-handles-rms-types.log`）。原因は `NormalizationConfig{_A,...} where _A` のPARTSが実行時widthから決まり、private cached launchの呼び出し型が確定しないこと。既存Metal macroの外部呼び出しに隠れていた境界がprivate helperに現れた。報告対象moduleを除外して隠さず、主要幅1024/128/256と32以下でVal(PARTS)を明示し、ほかの幅は従来macro起動へ進む候補に修正した。任意幅でPARTSを大きく丸めることはせず、GPU算術を変えない。再JETログは `/private/tmp/jeff-kernel-handles-rms-parts-types.log`。
- 主要幅のPARTSを確定した版はJET6対象すべて報告なし。単体verifierには幅33/64/129/257/512のRMS・gated RMSも追加し、通常macroへ進む幅の数値とGCも検証する。ログは `/private/tmp/jeff-kernel-handles-rms-parts-primitives.log`。GPU算術の変更によって型問題を避けたのではなく、host側の主要幅の分岐で型パラメータを確定している。
- 増やした幅を含む単体検証と、実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`（`/private/tmp/jeff-kernel-handles-rms-parts-validation.log`）。20回の性能比較は `artifacts/metal-validation/benchmark-metal-kernel-handles-rms.json` に保存する。callbackの型分岐の費用も含めて従来と比較する。
- 型分岐を含む版のB1/L256/active101/F32/workspace有効・20回は **5,043 allocations / 262,704 bytes / 中央値192.809625 ms**、min/p95/max192.233834/193.489541/193.54125 ms。直前5,982 / 308,768から939件 / 46,064 bytes減。workspace447配列 / 857,899,008 bytes、tensor-data/feed vector各199は同じ。直前中央値193.034521 msとの差は小さく、追加の速度改善とは確定しない。この版の通常設定・B2再測定と全割当Profileは未実施であり、既存の公開比較値は前コミットの測定値を維持する。
- コミット3b70d0cを全割当Profileで再検査し、5,043件、JET6対象すべて報告なしを確認した（`/private/tmp/jeff-kernel-handles-rms-profile.log`）。Core.Box148→81、KernelState206・MPS wrapper13・TD187は変わらない。残る起動サイトはpacked QK pair582件、recurrent487件、causal depthwise309件、query/key RoPE168/167件、softmax109件、merge gate98件。cached launchの1,841件は複数kernelの合計であり、独立した演算の費用ではない。

## Attention の固定引数 kernel handle cache 候補

- RMS版の全割当Profileに基づき、causal depthwise・masked softmax・merge gateの3起動を既存cached launchへ移す候補を追加した。GPU算術・scalar引数・配置は維持する。実行時Valを含むQK/RoPE/recurrentは今回の変更に含めない。単体検証は `/private/tmp/jeff-kernel-handles-attention-primitives.log`。性能と実モデル検証は未完了。
- 単体検証と実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`（`/private/tmp/jeff-kernel-handles-attention-validation.log`）。同条件20回の性能測定を `artifacts/metal-validation/benchmark-metal-kernel-handles-attention.json` に保存する。型検査と割当Profileはこの候補について未実施。
- B1/L256/active101/F32/workspace有効・20回は **4,983 allocations / 261,744 bytes / 中央値192.805833 ms**。RMS版5,043 / 262,704から60件 / 960 bytes減。min/p95/max192.342959/193.511709/193.91575 ms。18回のdepthwiseと各6回のsoftmax/merge gate、計30起動に対して2件ずつ減る結果で、追加の速度改善はない。JET・全割当Profileは `/private/tmp/jeff-kernel-handles-attention-profile.log` で実行中。
- 上記Profileは完了し、JET6対象すべて報告なし、全割当記録4,983件を確認した。KernelState206、MPS wrapper13、TD187は変わらない。起動時管理オブジェクトを減らしてもGPU演算・起動数を変えない限り、この条件の全体latencyはほぼ変わらない。

## Attention helper の重複サイズ引数削減候補

- depthwiseのchannels/length/kernel、merge gateのwidth/heads/length、softmaxのlength/columnsを独立scalar引数で渡す代わりに、GPU側のinput/weight/values/scoresのdescriptorから取得する候補を追加した。Int32変換、算術順、配置、配列所有は維持する。単体検証ログは `/private/tmp/jeff-attention-dims-primitives.log`。性能・実モデル・型検査は未完了。
- 単体検証と実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`（`/private/tmp/jeff-attention-dims-validation.log`）。同条件20回を `artifacts/metal-validation/benchmark-metal-attention-dims.json` へ保存する。型検査と全割当Profileは未実施。
- 同条件20回は **4,959 allocations / 259,056 bytes / 中央値193.5626875 ms**、min/p95/max192.756375/194.103042/194.303541 ms。直前4,983 / 261,744から24件 / 2,688 bytes減。中央値は直前192.805833 msより約0.4%長く、分布は重なる。割当削減を速度改善と扱わず、型検査・全割当Profile（`/private/tmp/jeff-attention-dims-profile.log`）後に再測定して採否を判断する。
- JET6対象すべて報告なし、全割当Profile4,959件。Int32は186→162件で総割当数の減少24件と一致し、KernelState206・MPS wrapper13・TD187は変わらない。配列descriptorの再利用によりサイズscalarのboxingと引数tupleのbytesを減らした結果。再測定を `artifacts/metal-validation/benchmark-metal-attention-dims-repeat.json` に保存する。
- 再測定も4,959 / 259,056、中央値194.1110625 ms、min/p95/max193.263291/194.55525/194.640625 ms。候補を退避して直前コミット648a1e1を再測定すると4,983 / 261,744、中央値193.3279165 ms、min/p95/max192.541667/193.974083/193.977167 ms（`artifacts/metal-validation/benchmark-metal-attention-handles-control.json`）。測定順は候補→候補→controlで無作為交互比較ではなく、分布も重なるため確定的なGPU速度退行とはしない。ただし候補の中央値は2回ともcontrolより約0.4%長く、割当削減24件のために採用する根拠は不足。3ファイルを648a1e1へ戻し、サイズ引数削減は不採用とした。前のdelta gate/mask/MLPのサイズ引数削減まで撤回する根拠ではない。

## 採用済み RMS/attention handle cache の通常設定再測定

- workspace無効、B1/L256/active101/F32・20回は **9,130 allocations / 399,360 bytes / 中央値194.887542 ms**、min/p95/max192.139209/202.468625/233.617334 ms。直前通常設定10,129 / 446,384から999件 / 47,024 bytes減。load後初回forward12.142814792秒はsteady-stateに含めない。free poolはtrial/GC/trimで1,732,575,232 / 6,133,907,456 / 4,765,515,776 bytesと変わらない。結果は `artifacts/metal-validation/benchmark-metal-default-attention-handles.json`。速度は従来と同程度。README/PLANの通常・workspace B1の値を更新し、古い10% allocation sampleの割合を現在のfull profileの実数に置き換えた。
- workspace有効、B2/L512/active512・256/F32・20回は **10,190 allocations / 538,400 bytes / 中央値765.487021 ms**、min/p95/max757.707667/773.58825/774.093208 ms、最大logit誤差 `1.9311905e-5`。直前12,284 / 633,984から2,094件 / 95,584 bytes減。workspace447配列 / 1,766,096,896 device-buffer bytesは同じ。前の762.7117295 msとの分布は重なり、追加速度改善はない。結果は `artifacts/metal-validation/benchmark-metal-attention-handles-b2-l512.json`。同じcaseのoriginal Python MPS F32参照1,300.984021 msより約41.2%短いが、FLA/causal-conv1d未導入fallback、Juliaの行逐次処理という比較条件を維持する。README/PLANのB2値も更新した。
