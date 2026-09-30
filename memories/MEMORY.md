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
