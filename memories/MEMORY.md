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

shape キャッシュによる割当数の削減は **33,202 → 28,426 回**。その後の feed/result 辞書の修正で **24,048 回**まで減ったが、こちらは別の変更である。heap allocation 全体はまだ解消しておらず、最新の辞書修正では推論時間の改善も確認できていない。Cthulhu は原因調査に役立った診断ツールであり、改善の効果は変更ごとの実測で判断する。

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
