# パフォーマンスの知見

## 2026-10-01: Issue 4/5/6 CPU 2倍高速化の新ベースライン

- 現在の実機は Intel Core i9-9900K / x86_64 macOS / Julia 1.13.1。過去の Apple M4 の時間をこの作業のベースラインとして流用しない。
- pinned Jeff 0.8B revision `0f212b3e72acb4dde3f7da61e925d6ab7f819990` を取得。parcel B1/L256/active101、Float32、readout込み、ロード/コンパイル/tokenization除外。
- 現行高速設定: Julia 8 workers、BLAS 8、AppleAccelerate 0.7.0（8 threads報告）、parallel heads / MLP workspace / vector math / leading padding trim 有効、in-place Delta RMS 無効。
- 20回ウォーム測定: median 495.254279 ms / min 482.833361 / p95 511.029933 / max 521.144273、392,752,624 heap bytes / 22,671 allocations。独立参照最大 logit 誤差 9.536743e-6。生データ `artifacts/cpu-tuning/baseline-fast.json`。
- 2倍の達成基準は同条件で median <=247.6271395 ms。短いfixtureや古い標準設定との比較で達成扱いにしない。Profile/JET/Profile.Allocs の実行ログは `/private/tmp/jeff-cpu-baseline-profile.log`。

### worker scratch と型・割当診断

- `JEFF_CPU_DELTA_WORKSPACE=1` のforward-local worker workspaceを試作。state/full/tailはworker番号で所有し、層の@sync完了後に再利用する。headごとにstateをzero resetし、同時forwardで共有しない。畳み込みchannelループのSIMDも追加。
- 同条件20回: median488.698334ms / p95519.214219ms、325,403,424bytes /14,153allocations、maxerror9.536743e-6。割当は減ったが速度改善は未確定。Accelerate singleは1403.1929285msと遅く、採用しない。データ `artifacts/cpu-tuning/workspace-{simd,accelerate-single}.json`。
- fixture独立参照18件とscratch所有100件合格。workspace/MLP/並列flag組合せ、同backendの2同時forward、GC後、入力保持、返却済みscore保持を確認。実モデルの多形状・内部mask穴・同時実行の検証はまだ必要。
- Cthulhu3.0.2で実モデルMLP→3引数mul!→5引数mul!→_mul!へ対話descent。配列と戻り値は具体型、transposeフラグはT/N。未使用mul!のトップ表示Anyだけで型不安定と判断しない。
- workspace追加後JETが層loopのruntime dispatchを1件検出。optional MLP/Delta workspaceのUnionの組合せをforward入口で絞り込むhelperへ修正し、再診断中。
- AllocCheck0.2.6はGPUCompiler<=1.23、Metal1.11.1は>=2.8.1を要求するためtools環境へ追加できない。`/private/tmp/jeff-alloccheck-env`にAllocCheck/AppleAccelerateのみを導入し、`tools/inspect_cpu_allocations.jl`を実行した。
- AllocCheck静的検出/ウォームheap: projection mul! 0件/0bytes、owned gate7件/32bytes、worker workspace4件/495,248bytes、logits472件/325,403,424bytes。静的件数は実行回数ではない。gateの検出にはbroadcastのaliasコピー分岐が含まれ、非alias実測で巨大コピーが起きた証拠ではない。
- Profile.Allocs 5%ではworker scratch生成が上位から消え、matmul、conv output、RoPE、RMSの配列が残る。native BLAS内部メモリはJulia heap測定に含まれない。ログ `/private/tmp/jeff-cpu-{alloccheck,workspace-profile,workspace-specialized-profile,types}.log`。
- 分岐を絞ったhelper修正後、Profile/JETはexit0・No errors detected。warm単発483.421682ms /325,400,480bytes /GC5.495422ms（性能確定には別BenchmarkTools repeatを使う）。型付きIRではhidden/layer/MLP/gateのBodyがMatrix{Float32}、workspaceは意図した小Union。修正後fixture18+所有100件も合格。ログを `artifacts/cpu-tuning/{alloccheck,profile,types}.log` に保存した。
- Intel profileの畳み込みweightのReinterpret/Reshape scalar accessを受け、tapごとに係数を連続Vectorへコピーするtrial。同じ20回F32/設定のparcel中央値430.2262715ms /p95455.427254ms /min400.624397 /max464.369811、327,174,560bytes /14,323allocations、maxerror9.536743e-6。基準495.254279msより約13.1%短いが2倍は未達。係数コピーの追加heapを認め、ロード時の小さいconv重みの配置変更で除去する案を次に評価する。生データ `artifacts/cpu-tuning/conv-coefficients.json`。
- 同trialのOpenBLAS8/Julia8/scalar vector fallbackは20回median720.920916ms /p95747.074693ms /maxerror1.335144e-5。Accelerateを置き換える高速化として採用しない。`artifacts/cpu-tuning/openblas8.json`。CPU convだけをロード時にchannel-contiguousなMatrixのTransposeへ変更し、tapをviewで参照する版の検証・測定を続行中。
- ユーザー指定によりMKLは使用しない。toolsのMKL/MKL_jll直接依存とbenchmark内のMKL分岐を除去。Apple Accelerateは別backendであり、現行比較に使用する。
- load-time conv packing版20回はmedian435.8153635ms /p95449.114716ms、325,393,568heap bytes /14,107allocations、maxerror9.536743e-6（`conv-packed.json`）。
- optional SIMD.jl 3.7 extensionを追加。8-wide Float32のconv、tap順序保持、fastmath/FMAなし、scalar tail、GC.@preserve付き。幅1/7/8/9/128/6144、tokens0/1/9、kernel1/4の独立ordered scalar比較72件合格。LoopVectorization0.12.174の@turbo smoke testはJulia1.13で動作したが、推論への採用・高速化は未確認。
- `JEFF_CPU_FINAL_QUERY=1` は最後のfull attentionのQ/gateのみを最終tokenへ限定し、K/Vは全contextを保持するopt-in。n1/9/65とmask穴、fixture参照を含む11件追加、計129件をSIMD有無で通過。
- SIMD+final query+scratchの20回はmedian441.457928ms /p95481.478997ms、310,815,216heap bytes /13,856allocations、maxerror1.04904175e-5（`simd-final-query.json`）。baseline比約1.12倍で、2倍は未達。scalar-packed版との速度差は改善の証拠にならず、SIMDのdefault有効化はしない。
- `final-query-no-simd.json` の最初のrunはProfileプロセスの起動と重なったため比較から除外する。再計測は独立実行する。
- `tools/compare_cpu_convolution.jl` に実幅6144×101×kernel4のmicro比較を追加（各100 samples、evals1）。medianは通常@simd407.211µs、explicit SIMD392.9815µs（完全一致）、@turbo477.856µs（最大誤差4.7683716e-7）。@turboはこの条件で遅く推論へ採用しない。microの差をend-to-endの改善と同一視しない。ログ `artifacts/cpu-tuning/convolution-comparison.log`。
- final query/scalar SIMDの最新Profile/JETは指摘なし。warm単発399.812391ms /310,814,640heap bytes /GC7.614192ms。main threadではBLAS GEMM・libdispatch待ちが大きく、畳み込みだけの改善では2倍へ到達しない。Profileのnative内部symbol名はunwind上の表示であり、表示されたdouble/complex演算を実際に呼んだと断定しない。ログ `artifacts/cpu-tuning/profile-final-query.log`。
- profile終了後の独立20回（final query / scratch有効、explicit SIMD無効）はmedian417.5303145ms /p95441.452236ms /min397.09657 /max465.09941、310,814,640heap bytes /13,838allocations、maxerror1.04904175e-5。baseline比1.186倍、heap bytes約20.9%減、2倍未達。`artifacts/cpu-tuning/final-query-no-simd-independent.json`。explicit SIMDの有効化は全体速度の優位を確認できずoffのまま。

### chunk / worker / GEMM の追加切り分け

- 前turnはSIMD/LV単体比較・独立full-forward repeat・Profile/JETでprogress。今回はissue4/5/6を再読し、`JEFF_CPU_DELTA_WORKERS`（available threads/value headsへ上限制約）と`JEFF_CPU_DELTA_CHUNK_SIZE`（正整数、既定64）を試験用に追加。workspace full/tailと実行spanは同じchunkサイズを使う。既定値は変えない。
- `tools/sweep_cpu_delta.jl` はreal0.8B/parcel/F32、Accelerate8、Julia8、scratch/final query/MLP/vector/trim有効、chunk16/32/64/128×workers1/2/4/8を各5samplesでscreenする。全16設定の独立参照logit guard通過、maxerror<=1.04904175e-5。chunk64のworkers1/2/4/8 median619.74/528.30/495.66/456.02ms、chunk32の8workers444.29ms、chunk16の8workers454.83ms、chunk128の8workers561.57ms。5samplesのp95はtail確定に弱く、earlier417.53msとの時系列差もあるためchunk32を採用する根拠としない。ログ `artifacts/cpu-tuning/delta-sweep.jsonl`。
- chunk1/3/16/32/64/128×workers1/2/4/8のfixture5件と不正値0の例外2件を追加、122件通過。既存129件も通過、計251件。ログ `artifacts/cpu-tuning/chunk-worker-tests.log`。
- LoopVectorizationベースのpure-Julia GEMM候補Octavian0.3.29をtoolsの診断用依存に追加し、`tools/compare_cpu_gemm.jl`でreal first-layerのgate/downを20samples、101tokensで比較。gateはAccelerate1.55370ms /Octavian threaded1.84101ms /serial11.91360ms、downは1.67611/1.86755/12.10002ms、全てJulia heap0bytes。最大差<=9.536743e-7。threadedでも遅く、推論実装には導入しない。これはfirst-layer microであり全24層のweight sweepやfull-forwardではない。ログ `artifacts/cpu-tuning/gemm-comparison.log`。
- Intel trialの条件・未達2倍・opt-in所有・chunkによる丸め変化・heapとpeakの違いを`docs/src/performance.md`へ追加。Apple M4の既存比較とは分離して記載した。
- Accelerate4指定の2回のfull-forwardはJSON実値が8（median441.242223/441.4668055ms）だったため4thread測定ではない。AppleAccelerate0.7.0のローカルsourceを確認し、set_num_threadsは1ならsingle、1以外ならautomatic multiを選ぶだけと判明。独立smokeは要求1/2/4/8に対し報告1/8/8/8。4固定のbenchmarkと呼ばない。benchmarkは明示overrideを一般BLAS設定後に適用し、この制約をコメントへ記載した。古いsingle trialのJSONは1を記録しているため一律に無効化しない。

### 層別phaseとworker-local小行列積 / recurrent試作

- 前turnはchunk/worker screen、251 tests、Octavian大行列の不採用、Accelerateのthread toggle確認でprogress。`tools/time_cpu_phases.jl`はreal0.8B各層の実activationsを次層へ渡し、5 full passesの時間とheapを7 phaseに分ける診断。各passは独立parcel logit guardを通過。total402.54〜415.99ms、phase合計の中央値はpre RMS1.21 /attention251.02（delta202.82/full46.58）/post RMS2.39 /gate+up85.18 /activation21.85 /down41.03 /residual0.63ms。カテゴリーごとの中央値は足して厳密totalにならない。instrumentation自体も通常BenchmarkToolsとは異なる。ログ `artifacts/cpu-tuning/phases.jsonl`。
- `tools/compare_cpu_delta_gemm.jl`で実際と同じhead-strided RHS/Float32、100samples、n37/64を比較。state×queryはAccelerate32.11/24.96µs、Octavian serial10.23/16.74µs。一方systemは7.11/16.38 vs11.28/21.54µs、state updateは19.85/18.99 vs20.15/41.08µs。全面置換の根拠ではない。
- `JEFF_CPU_OCTAVIAN_DELTA=1`のoptional extensionでstate×RHSの2箇所だけworker-local serial Octavianへ試験切替。state各辺<=256/RHS列<=128を上限とし、他はBLASへfallback。alpha/beta、NaN destination(beta0)、head-strided/contiguous/transposed RHS、入力保持を含む144 tests通過、@code_warntypeはMatrix{Float32} body。ログ `artifacts/cpu-tuning/octavian-state-tests.log`。
- 最初の`octavian-delta.json`はweakdep追加後のManifest metadata未解決によりBase.get_extension=nothingのままBLASで測定されていた（434.98ms）。候補評価から除外。Pkg.resolve後にextensionの登録を確認し、benchmark/verificationへextension存在assertを追加。SIMD extensionは同じ検査で実際に存在した。
- extension有効の20samplesはmedian410.989238ms /p95439.118232ms /310,852,720heap bytes /15,028allocations、maxerror1.04904175e-5（`octavian-delta-active.json`）。earlier417.53msとの差は独立repeatなしで改善確定としない。2倍は未達、default off。
- chunkの小BLAS/TRSMを省く別候補`JEFF_CPU_RECURRENT_DELTA=1`を試作。Float32のstate×key→rank-one state更新＋state×queryをrow方向@simdで融合。worker所有stateをheadごとresetし、tokenごとprojection/resultをresetする。forward scratchはn1の既存bufferとstateだけを保持しchunk scratchを不要にする（永続cacheなし）。chunkの丸め順序とは異なるため参照を再検証する。
- recurrent有効で既存251件＋n1/9/65/129/256×mask穴×parallel有無のchunked比較20件=271件通過（`recurrent-tests-fixed.log`）。最初はrecurrent scratch縮小時に不正chunk0の検証を迂回するテスト1件失敗があり、scratch選択前に必ず正整数検証するよう修正した。実モデルfull-forward計測は別に行う。
- recurrentのreal0.8B/parcel20samplesはmedian408.828974ms /p95449.153710ms /min384.642854 /max450.465069、291,769,968heap bytes /5,068allocations、maxerror9.536743e-6。baseline495.254279msから約1.21倍だが2倍未達。件数削減ほど時間は短縮しない。以前の417.53msとの差は独立repeatなしで確定としない。`artifacts/cpu-tuning/recurrent-delta.json`。
- recurrentの独立50samples repeatはmedian402.390980ms /p95455.791675ms /min377.902317 /max481.109223、heap291,769,968bytes /5,068件、maxerror9.536743e-6。初期495.25msの約1.23倍（time約18.75%減）、heap約25.7%減、件数約77.6%減。2倍基準247.63ms未達。`artifacts/cpu-tuning/recurrent-delta-repeat50.json`。
- recurrent有効のProfile/JETはNo errors detected、warm単発367.860271ms /291,769,968bytes /GC7.114798ms。main threadでは引き続きprojection GEMMとlibdispatchが多く、サンプル配列生成としてmatmulやQ/Kコピーが残る。ログ `artifacts/cpu-tuning/profile-recurrent.log`。AllocCheck診断にrecurrent kernel単独を追加し、全forward allocationと分離して再確認する。
- isolated AllocCheck0.2.6の再診断: recurrent kernelは静的指摘0 /ウォームheap0bytes、projection mul!も0/0。worker scratchは静的5 /70,032bytes（recurrent用state+n1 scratchを新規作成）、owned gate7 /32bytes、全logits536 /291,769,968bytes。静的件数は未実行のchunk/例外分岐も含み実行allocation件数ではない。kernelの0割当は重み投影・Q/Kコピー・forward workspace生成を含まない。ログ `artifacts/cpu-tuning/alloccheck-recurrent.log`。

### Intel の投影配置 / BLIS micro比較

- 前turnはrecurrent SIMD実装・271 tests・独立50 samples・JET/Profile/AllocCheckでprogress。今回はmicroを物理transposeとright-product+copyへ拡張。Intel/Accelerate8でgate通常1.61411ms→packed1.35877ms、down1.79932ms→packed1.58713ms（各20samples、F32/101tokens、全Julia heap0）。right+copyは2.11322/1.97171msと遅い。M4ではtransposeMLPに差がなかったが、別CPUの今回のmicroはfull-forward trialを評価する根拠になる。まだ全forward改善は未証明。`gemm-orientation-accelerate-fixed.log`。最初のrunはpermutedims!のperm引数欠落で途中失敗し、(2,1)指定後に全対象を再測定した。
- 非MKL候補BLISBLAS0.2.0 /blis_jll2.0.0+2をtoolsの診断用依存に追加。macOS/x86_64のartifact利用とBLIS8 threadsを確認。gate通常2.30516ms /packed2.50571ms、down1.44855 /packed2.62688ms。gateはAccelerateより遅く、全面採用しない。source上はLP64/ILP64 BLISとLAPACKをstartup時にLBTへforwardする。推論中のglobal backend変更は行わない。`gemm-orientation-blis.log`。
- ユーザーの「改善ごとにcommit and push」指示を受け、recurrentを含む現段階をcheckpointにする。`julia --threads=8 --project=test test/runtests.jl`は全testset成功、`git diff --check`成功。2倍達成とは扱わず、既定offのtrialと測定条件をdocsに明記してコミットする。ログ `artifacts/cpu-tuning/checkpoint-tests.log`。

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

## Packed Q/K の主要幅での kernel handle cache 候補

- 全割当Profileで582件を記録したpaired Q/K起動について、実モデルのkey_dim128に限ってVal(4)を明示し既存cached launchへ進む候補を追加した。他の幅は従来macroで正確なcld値を使う。RMSで確認した実行時Valによるruntime dispatchを避け、GPU算術・配置・所有は変更しない。単体検証ログは `/private/tmp/jeff-qk-handles-primitives.log`。性能・実モデル・型検査は未完了。
- 単体検証と実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`（`/private/tmp/jeff-qk-handles-validation.log`）。同条件20回を `artifacts/metal-validation/benchmark-metal-qk-handles.json` へ保存する。型検査・全割当Profileは未実施。
- B1/L256/active101/F32/workspace有効・20回は **4,695 allocations / 249,360 bytes / 中央値193.334437 ms**、min/p95/max192.480833/193.81275/193.899708 ms。直前4,983 / 261,744から288件 / 12,384 bytes減。control再測定193.3279165 msと同程度で、追加の速度改善はない。pipeline管理だけを減らしたと断定せず、Val型確定の効果も全割当Profile（`/private/tmp/jeff-qk-handles-profile.log`）で確認する。
- 上記Profileは完了し、JET6対象報告なし、全割当4,695件。Int32は186→114、Float32は96→60、Tuple{Int64,Int64}は657→621。KernelState206・MPS wrapper13・TD187は不変。18起動のpipeline管理だけで説明できる36件より大きい288件減であり、主要幅をhost分岐で確定したことによる引数管理の削減も含む。

## Recurrent 起動の主要幅での kernel handle cache 候補

- 最新全割当Profileで487件を記録したrecurrent起動にも、key_dim128に限りVal(4)/Val(8)を確定して既存cached launchを適用する候補を追加した。rows8、GPU演算、scalar引数、配置、所有は維持し、他のkey幅は従来macroを使う。単体検証は `/private/tmp/jeff-recurrent-handles-primitives.log`。性能・実モデル・型検査は未完了。
- 単体検証と実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`（`/private/tmp/jeff-recurrent-handles-validation.log`）。同条件20回を `artifacts/metal-validation/benchmark-metal-recurrent-handles.json` へ保存する。型検査・全割当Profileは未実施。
- B1/L256/active101/F32/workspace有効・20回は **4,479 allocations / 237,264 bytes / 中央値193.2585835 ms**、min/p95/max192.815375/193.747042/194.055833 ms。直前4,695 / 249,360から216件 / 12,096 bytes減。中央値193.334437 msと同程度で、追加の速度改善はない。型検査・全割当Profileは `/private/tmp/jeff-recurrent-handles-profile.log` で実行する。
- 上記Profileは完了し、JET6対象すべて報告なし、全割当4,479件。KernelState206・MPS wrapper13・TD187は変わらない。主要幅のhost型分岐とhandle再利用で割当を減らし、GPU起動や演算を省いた結果ではない。

## RoPE 起動の主要幅での kernel handle cache 候補

- 最新全割当Profileでquery/key起動は168/167件を記録した。head_dim256に限りVal(8)とqueryフラグを確定したhelperで既存cached launchを使う候補を追加し、他の幅は従来macroを使う。GPU算術・配置・所有は変更しない。単体検証は `/private/tmp/jeff-rope-handles-primitives.log`。性能・実モデル・型検査は未完了。
- 単体検証と実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`（`/private/tmp/jeff-rope-handles-validation.log`）。同条件20回を `artifacts/metal-validation/benchmark-metal-rope-handles.json` へ保存する。型検査・全割当Profileは未実施。
- B1/L256/active101/F32/workspace有効・20回は **4,335 allocations / 228,624 bytes / 中央値193.601583 ms**、min/p95/max192.9335/194.211375/194.252416 ms。直前4,479 / 237,264から144件 / 8,640 bytes減。中央値193.2585835 msとの差は小さく分布も重なり、追加速度改善はない。型検査・全割当Profileは `/private/tmp/jeff-rope-handles-profile.log` で実行する。
- 上記Profileは完了し、JET6対象すべて報告なし、全割当4,335件。KernelState206・MPS wrapper13・TD187は変わらない。次のTD再利用候補を調べると、graph_tensor_dataは未登録arrayを「現在のcursor位置のslotと同一」のときだけ登録しており、過去slotのnormalization出力などはworkspace所有でも初回登録されない。所有slotのidentityを検証できる索引と、形状変更・clear・例外・GCの検証が必要。reshape wrapperを無条件に登録すると別形状のmetadataや一時ownerを誤って保持するため、その経路は分けて検討する。

## Workspace 過去 slot の MPS tensor-data 再利用候補

- ForwardWorkspaceにobjectid→slot番号の索引を追加し、graph_tensor_dataが現在cursor以前のowned slotで配列identityも一致するときに登録する候補を追加した。slot置換・末尾削除・clearで索引を削除する。通常設定と一時reshape wrapperの経路は維持し、GPU buffer所有や同期条件は変えない。単体verifierに過去slotの入力配列/TD identity、length9/9/1/1/65/65、GC、索引一致、例外後の末尾削除、clearを追加した。検証ログは `/private/tmp/jeff-slot-td-primitives.log`。性能・実モデル・型検査は未完了。
- 追加した所有/identity検証を含む単体検証と、workspace有効の実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`（`/private/tmp/jeff-slot-td-validation.log`）。同条件20回を `artifacts/metal-validation/benchmark-metal-slot-td.json` に保存する。型検査・全割当Profileは未実施。
- B1/L256/active101/F32/workspace有効・20回は **4,166 allocations / 223,216 bytes / 中央値193.6943125 ms**、min/p95/max193.1335/194.110166/194.878167 ms。直前4,335 / 228,624から169件 / 5,408 bytes減。保持TD199→278、feed Vector199、配列447 / 857,899,008 device-buffer bytesは同じ。保持TD増加はhost/native metadataの再利用であり、GPU bufferを79個増やした結果ではない。前中央値193.601583 msと同程度で、追加速度改善はない。型検査・全割当Profileは `/private/tmp/jeff-slot-td-profile.log` で実行する。
- 上記Profileは完了し、JET6対象報告なし、全割当4,166件。TDの記録は **187→18件 / 5,984→576 bytes** で、総割当差169件 / 5,408 bytesと一致した。79個の過去slotのTDを保持することで、forward内で同じ入力が複数回使われる分も含めて169回の再構築を省けた。KernelState206・MPS command wrapper13は不変。残る18 TDは所有slot以外のwrapper経路をさらに調査する対象であり、shape/offset/所有を確認せずbufferだけでcacheしない。

## DeltaNet gate の直接行列出力候補

- 残る18 TDと、18 DeltaNet層のgated出力reshape→out projection経路が対応する候補を調べた。normalization kernelはoutputに線形添字で書き込み、幅と列数はinputから決めるため、同じ要素数の行列outputを直接確保できる。rms_silu_gateにoutput_dimsを追加し、DeltaNetは(value_dim*value_heads,length)のowned slotへ書く候補に変更した。通常呼び出しは元の形状を維持し、GPU算術・入力RMS幅・配列所有は変えない。verifierに各幅の直接行列出力を追加。単体ログは `/private/tmp/jeff-flat-gate-primitives.log`。実モデル・型検査・性能は未完了。
- 単体検証とworkspace有効の実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`（`/private/tmp/jeff-flat-gate-validation.log`）。同条件20回を `artifacts/metal-validation/benchmark-metal-flat-gate.json` に保存する。型検査・全割当Profileは未実施。
- B1/L256/active101/F32/workspace有効・20回は **4,112 allocations / 221,200 bytes / 中央値194.250959 ms**、min/p95/max193.081875/194.7955/194.849875 ms。直前4,166 / 223,216から54件 / 2,016 bytes減。保持TD278→296、配列447 / 857,899,008 device-buffer bytes・feed Vector199は同じ。前中央値193.6943125 msとの分布は重なり、追加速度改善はない。型検査・全割当Profileは `/private/tmp/jeff-flat-gate-profile.log` で実行する。速度差の再現性も確認して採否を決める。
- 上記Profileは完了し、JET6対象すべて報告なし、全割当4,112件。**MPSGraphTensorDataの割当は18→0件**となった。KernelState206・MPS command wrapper13は変わらない。残る18件がDeltaNet gated reshapeのmetadataだったことを変更前後の割当記録で確認した。GPU起動数を省く変更ではない。再測定を `artifacts/metal-validation/benchmark-metal-flat-gate-repeat.json` に保存する。
- 再測定は同じ4,112 / 221,200、中央値193.272063 ms、min/p95/max192.872375/194.479084/194.511208 ms。初回の小さな中央値増加は繰り返されず、速度改善とはしないが、直接出力により不要なwrapperとTD構築を省く変更として採用した。
- Metal 1.11.1 `src/compiler/execution.jl:467` のKernelStateはlaunchごとにRandom.rand(UInt32)のseed、malloc/exception buffer GPU address、kernel relocation table addressから作られる。206件のstateを単に固定値で共有するとseedやkernel固有addressの意味を変えるため、その再利用は行わない。残る管理割当とGPU実行時間は別に扱い、速度改善には演算融合・行列積のまとめ方なども測定する必要がある。

## Packed MLP projection の段階測定候補

- Layaの `ext/LayaMetalExt.jl:398` のgelu_gate_kernelはpacked projectionの前半・後半を読み、別の半幅outputへgate結果を出す。我々はSiLUなので算術は既存native_siluを使い、同じ配置でgate/up weightを一度結合し1回のMPS積と専用gate kernelで処理する診断候補を `tools/benchmark_stages.jl` に追加した。weight CPU readback/結合/uploadは測定外、候補は追加weight copyを保持しpacked projection＋半幅gate outputの一時bufferを使う。通常実装は変更していない。既存MLPと2e-4許容で数値比較してから同期込み10回の段階測定を行う。結果は `artifacts/metal-validation/stages-packed-mlp.json`、ログは `/private/tmp/jeff-stages-packed-mlp.log`。段階測定は全体forward速度の証明ではない。
- B1/L256/active101/F32、workspace無効・同期込み10回のMLP単体は従来 **4.3575625 ms / 81 allocations / 3,424 bytes**、packed候補 **3.729979 ms / 75 allocations / 2,640 bytes**で約14.4%短い。embedding直後のhiddenを使う単体比較で、正規化後の実層入力や全24層forwardでは未確認。24倍して全体短縮量と主張しない。実装候補をモデル読み込み時のweight packingと全体forwardへ広げ、独立参照と保持メモリを検証する価値がある。

## モデル全体の Packed MLP 実装候補

- coreのnative_mlp_weights/native_mlpでbackend固有の重み配置と演算をdispatchする構成を追加した。Metalでは `JEFF_METAL_PACKED_MLP=1` のときだけ読み込み時にgate/upを結合し、PackedMLPにはgate_up/downだけを保持する。元のgate/upはbackendに残さないが、読み込み中の一時CPU/GPUコピーとGC前のbuffer保持は生じる。forwardはpacked投影、半幅gate output、down投影を使い、従来よりMLPの一時buffer総要素数が増える。通常設定・CPUは既存gate/upのまま。全体の独立参照検証は `/private/tmp/jeff-packed-mlp-validation.log`、性能・型・所有/primitive検証は未完了。
- packed MLP/workspace有効の実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`。CPU算術を参照する幅7/128/3584・length1/9のpacked MLP単体検証もverifierへ追加した（未実行）。全体20回の結果を `artifacts/metal-validation/benchmark-metal-packed-mlp.json` へ保存する。
- 全体B1/L256/active101/F32、packed MLP/workspace有効・20回は **4,001 allocations / 203,760 bytes / 中央値192.2585835 ms**、min/p95/max191.905917/192.973459/193.322625 ms。直前非packed再測定4,112 / 221,200 / 193.272063 msに対して111件 / 17,440 bytes減、中央値差は約0.5%。MLP単体の約14%短縮を全体へ外挿できない。workspace447配列は変わらないがbuffer保持857,899,008→945,979,392 bytes（88,080,384増）、TD296・feed Vector175。load2.533058167秒、初回12.106032166秒はsteady-state外。追加メモリに対する速度差の再現性・型検査・単体/所有検証を確認して採否を決める。単体verifierは `/private/tmp/jeff-packed-mlp-primitives.log` で実行中。
- 追加したpacked MLPのCPU算術参照・GC後再実行を含む単体verifierは合格した。全体のJET・Profileは `/private/tmp/jeff-packed-mlp-profile.log` で実行する。
- 上記Profileは完了し、JET6対象報告なし、全割当記録4,002件（BenchmarkTools中央値4,001件とは1件差）。profileとbenchmarkの件数を同一と扱わない。再測定を `artifacts/metal-validation/benchmark-metal-packed-mlp-repeat.json` へ保存する。
- packed再測定は **4,001 / 203,760 / 中央値192.5496875 ms**、min/p95/max191.830167/193.358916/193.663041 ms。JET/ProfileでTD新規割当0、MPS command wrapper12（従来13）、KernelState206。coreのMLP dispatch変更を含む非packedcontrolを `artifacts/metal-validation/benchmark-metal-mlp-unpacked-control.json` へ測定し、直近比較で採否を判断する。
- 非packedcontrolは **4,112 / 221,200 / 中央値193.359875 ms**、min/p95/max192.539583/194.081833/194.580292 ms。packed再測定との中央値差は約0.4%で分布は重なる。B2/L512のpacked版を `artifacts/metal-validation/benchmark-metal-packed-mlp-b2-l512.json` へ測定する。benchmark_inferenceには今後の結果へworkspace/packed MLP設定をboolで保存する変更も追加した（開始済みの測定は旧toolを読み込んでいる可能性があるため記録条件で補う）。
- B2/L512/active512・256/F32、packed MLP/workspace有効・20回は **8,106 allocations / 420,512 bytes / 中央値771.81 ms**、min/p95/max758.75075/777.114417/778.897083 ms。workspace447配列 / 1,942,257,664 bytes、TD296/feed175。旧attention handle版765.487021 msより速くないが、その後の複数変更を含むため直接効果とはしない。現行非packedcontrolを `artifacts/metal-validation/benchmark-metal-unpacked-control-b2-l512.json` へ測定して比較する。
- 現行B2非packedcontrolは **8,328 / 455,392 / 中央値776.1274165 ms**、min/p95/max761.969125/785.229583/788.595208 ms。packedは中央値約0.6%短く222件 / 34,880 bytes少ないが、buffer保持は176,160,768 bytes増える。順序がpacked→controlで分布も重なるため速度改善を確定しない。B1/B2ともhost割当削減は再現したので、既定は非packedのまま、追加bufferと割当削減のtradeoffを選べる実験用opt-inとして残す候補とする。CPU・workspace identity/cleanup・通常設定検証と文書化を済ませてから確定する。
- native_mlp dispatch導入後のCPU fixture verifierは合格、最大logit誤差 `3.874302e-7`（`/private/tmp/jeff-packed-refactor-cpu.log`）。packed MLPにはworkspace3 slot/TD identity、length9/9/1/1/65/65、GC、completed feed input解除、例外cleanup、clearの検証を追加した。ログは `/private/tmp/jeff-packed-mlp-workspace-primitives.log`。通常設定のpacked測定は未実施。
- 追加workspace ownership検証を含む単体verifierも合格。workspace無効のpacked版を `artifacts/metal-validation/benchmark-metal-packed-mlp-workspace0.json` へ測定する。
- workspace無効のpacked B1/L256/active101/F32・20回は **8,191 allocations / 342,080 bytes / 中央値194.116375 ms**、min/p95/max192.506834/209.653084/211.6505 ms。free pool trial/GC/trim543,326,208 / 6,370,426,880 / 4,761,911,296 bytes。JSONに設定boolを保存した。workspace有効の約192.5 msを既定設定の値と扱わない。非packed/非workspaceの現行既定を `artifacts/metal-validation/benchmark-metal-default-current.json` へ測定する。READMEにはopt-inの割当とbuffer保持のtradeoffを追記した。
- 現行既定（packed/workspaceとも無効）の同条件20回は **8,446 allocations / 364,512 bytes / 中央値195.213854 ms**、min/p95/max192.465292/207.947167/214.76925 ms。free pool trial/GC/trim1,651,802,112 / 6,053,167,104 / 4,765,515,776 bytes。通常packedとの差255件 / 22,432 bytesも実測したが、latencyの範囲は重なる。README/PLANの既定・workspace・B2値を現行結果へ更新した。packed MLPは既定無効の実験用opt-inとして採用し、14%の段階測定を全体速度の宣伝に使わない。

## 先頭 padding の計算省略候補

- B1/L256ではactive101以外の先頭155 tokenも全層で処理している。`JEFF_METAL_TRIM_PADDING=1` のMetal候補を追加し、先頭の連続0 maskだけを省いてgather/forwardする。系列内の0は残し、ID/maskの全入力検証と最終位置active要件は維持する。CPU/通常設定は省略しない。biasなしDeltaNetの先頭masked入力/stateは0、causal convolutionの境界は0相当、full attentionのactive位置間RoPE相対差は保存されるという根拠があるが、位置shiftのFloat32丸めは変わるため独立参照検証が必要。実モデルログは `/private/tmp/jeff-trim-padding-validation.log`。性能・型・interior mask/所有/設定組合せは未完了。B2で行の実長が変わるとworkspace slotの形状置換が増える点も測定する。
- workspace有効・packed無効・trim有効の実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`。一部caseの誤差は位置shiftで変わるが許容内。同条件B1の20回を `artifacts/metal-validation/benchmark-metal-trim-padding.json` に保存する。interior mask・各設定組合せ・型検査は未完了。
- B1/L256/active101/F32、workspace/trim有効・packed無効・20回は **4,111 allocations / 217,584 bytes / 中央値86.121625 ms**、min/p95/max85.907417/86.357208/86.389375 ms、最大logit誤差 `7.6293945e-6`。非trimの直近193.359875 msより約55.5%短い。同じ論理入力を渡しているが、実際の計算系列長は256→101となる最適化であり、同じGPU演算量の比較ではない。workspace447配列のbuffer保持857,899,008→336,756,736 bytesと減る。型検査・全割当Profileは `/private/tmp/jeff-trim-padding-profile.log`。系列に先頭paddingがない場合の速度改善を示す結果ではない。
- 上記Profileは完了し、JET6対象報告なし、全割当4,111件。MPS command wrapper13、TD新規割当0、KernelState206。速度短縮はGPU起動数削減ではなく、各起動の演算要素数を減らした結果。独立参照generatorにlength9/65/129・B3の3ケースを追加した。各ケースにはprefix+interior holes、最後の1 tokenだけactive、先頭activeでinterior holesの3行を含む。PythonCall経由のoriginal PyTorch CPU参照を `artifacts/metal-validation/reference-with-mask-holes.json` へ生成する（`/private/tmp/jeff-mask-holes-reference.log`）。既存12ケースの順序は維持した。
- 独立参照15ケースの生成は完了した。trim有効・workspace有効・packed無効で各3回の検証を `/private/tmp/jeff-trim-padding-mask-validation.log` で実行する。benchmarkには論理sequence_lengthと区別して実際のmetal_computed_sequence_lengthsを保存する変更を加えた。
- 上記15ケース×3回は合格、最大logit誤差 `3.540516e-5`。interior holesを持つ行と最後の1 tokenだけactiveの行も独立PyTorch参照に一致した。trim有効・packed有効・workspace無効の組合せも `/private/tmp/jeff-trim-padding-packed-workspace0-validation.log` で各3回検証する。primitive verifierには先頭0だけを省く選択、先頭active/最後だけactive、CPUとtrim無効の開始位置維持を追加した（未実行）。
- trim有効・packed有効・workspace無効も15ケース×3回合格、最大logit誤差 `3.540516e-5`。B2/L512/active512・256でtrim有効・packed無効・workspace有効の20回測定を `artifacts/metal-validation/benchmark-metal-trim-padding-b2-l512.json` へ保存する。論理長512とは別に各行の計算長512/256を記録する。
- 上記B2・20回は **17,114 allocations / 1,039,200 bytes / 中央値583.9950625 ms**、min/p95/max573.550292/4,822.468167/9,204.780084 ms。非trim現行776.1274165 msより中央値は約24.8%短いが、異常に長いtailがあり安定した速度改善とは扱わない。行ごとのshape512/256で単一workspaceのslotを毎回置換するため、非trim8,328件から割当が増える。終了時workspaceは最後の行の447配列 / 857,899,008 bytesを保持する。この値は途中の512-token rowやfree poolを含むpeakではない。形状切替の割当とtailを調査してから既定採用を検討する。
- primitive verifierの追加trim選択・CPU/無効時維持を含む検証は合格（`/private/tmp/jeff-trim-padding-primitives.log`）。B2のpool misses9,098 / reuses11,374、trial free0からfull GC後free25,717,506,048 bytesへ増え、trim後4,765,908,992 bytes。GPU bufferの参照解放がGCまで遅れることが大量のfresh allocationを引き起こす証拠であり、9秒tailの原因と断定するには追加測定が必要。shape別のworkspace再利用と、完了後のbuffer圧力/GC処理を評価する。
- Metal 1.11.1 `src/memory_pressure.jl:77` のmaybe_collectはallocated/recommended working setの75%（待機時50%）以上でrate limit付きGC(false)を行う。`src/pool.jl:66` はallocation失敗時GC(true)+device同期でretryする。我々のReturnBufferはGCでbufferをfree poolへ返し、完了後trimで物理解放するため、GCと物理解放は同時ではない。大きなworking setとold-generationの参照を含むshape置換では小さいJulia heapだけを見たGC制御では足りない可能性がある。まず繰り返しshapeのworkspaceを保持して置換自体を減らす候補を評価し、buffer圧力による安全な完了後回収も別に検討する。
- trim有効・packed無効・workspace無効も15ケース×3回合格、最大logit誤差 `3.540516e-5`（`/private/tmp/jeff-trim-padding-workspace0-validation.log`）。残るpacked/workspaceとも有効の組合せを `/private/tmp/jeff-trim-padding-packed-workspace1-validation.log` で実行する。
- trim/packed/workspaceとも有効も15ケース×3回合格、最大logit誤差 `3.540516e-5`。trimの4組合せは確認済み。

## 計算長別の最大2 workspace 再利用候補

- `JEFF_METAL_SHAPE_WORKSPACES=1` とworkspace有効のとき、native_forward_scopeの3引数版で計算長ごとのworkspaceを選ぶ候補を追加した。2組をLRUで保持し、完了後に合計buffer保持がrecommended working setの1/4を超えたら古い組を除去する（単一の現在workspaceが上限超の場合はそのまま）。既存2引数/nested scopeは選択中workspaceを進め、clearは全組を解除する。workspace自身のbytesをslotの置換/削除/clearで追跡し、pool statsは全組の配列/TD/feed/bytesと保持長を報告する。入力wrapperや未完了bufferの所有・同期条件は変えない。
- trim/shape workspace有効・packed無効で独立参照15ケース×3回は合格、最大logit誤差 `3.540516e-5`（`/private/tmp/jeff-shape-workspaces-validation.log`）。B2/L512の20回は `artifacts/metal-validation/benchmark-metal-shape-workspaces-b2-l512.json` へ保存する。LRU/byte上限/GC/例外/clearのprimitive検証とJET/Profileは未完了。
- primitive verifierにshape9/1/9/1/65で配列/TD identityとLRU、1 pageのbyte上限によるeviction、nested scopeで選択維持、例外後feed input解除、全組clearを追加した。ログは `/private/tmp/jeff-shape-workspaces-primitives.log`。
- B2/L512/active512・256/F32、trim/shape workspace有効・packed無効・20回は **8,334 allocations / 749,024 bytes / 中央値574.837729 ms**、min/p95/max571.196625/577.362709/579.664291 ms。単一workspace trim版17,114 / 1,039,200から8,780件 / 290,176 bytes減り、この20回では秒単位のtailが出なくなった。非trim776.1274165 msより中央値約25.9%短い。保持2長512/256、894配列 / 2,623,995,904 bytes（片方だけではない）、pool misses894でwarm後のGPU fresh allocation増加を避けた。通常nontrim1組の保持1,766,096,896 bytesより857,899,008 bytes多い。最小/中央値を超える一般的な安定性は長時間測定が必要。JET/Profileとownership verifierは未完了。
- ownership primitive verifierは完了し合格した。形状切替での配列/TD再利用、LRU、byte上限による除去、nested scope、例外後の解除、全workspace clearを確認済み。shape workspace版のJET/Profileと長時間測定は未実施であり、trim/shape workspaceは引き続き既定無効のopt-inとする。
- shape workspace版B2/L512 case12のJET/CPU Profile/全割当記録は完了（`/private/tmp/jeff-shape-workspaces-b2-profile.log`）。JET6対象報告なし、warm @timed 569.969417 ms / 750,800 bytes / GC時間0。全割当Profileは8,364件（BenchmarkToolsの8,334件とは別測定）、KernelState412 / MPS command wrapper26 / TD0。cached kernel起動サイト6,165件 / 261,168 bytesが最多。RoPE表生成だけでもcos/sinのサイト16件 / 197,088 bytesが残り、1組cacheでは計算長512/256で毎forward表を作り直していた。これは非trimよりhost bytesが大きい理由の一つである。`inspect_native.jl` にexpanded参照・case番号・設定の表示を追加した。
- RoPE表をqueueごとの最大2組LRUにする候補を追加した。keyはrotary_dim/計算長/rope_thetaで、表を書き換えず、除去後もqueued kernelのrootによる寿命を維持する。primitive検証に長さ切替後のGC/identity、異なる長さの分離、2組上限、LRU除去を追加した。ログは `/private/tmp/jeff-rope-two-tables-primitives.log`。数値・型・性能の再検証は進行中で、改善量は未確定。
- 上記primitive verifierは合格した。初回は引数lengthがBase.lengthを隠してMethodErrorとなり、sequence_lengthへ改名して修正した。既存の実幅RMS/RoPE/行列積とGC後再利用、追加LRU検証を通過した。B2・50回benchmarkを `/private/tmp/jeff-shape-workspaces-rope-b2-benchmark.log` で実行する。
- RoPE 2組cache後のB2/L512/active512・256/F32・workspace/trim/shape有効・packed無効・50回は **8,274 allocations / 448,608 bytes / 中央値580.947313 ms**、min/p95/max572.132917/589.669875/591.444166 ms（`artifacts/metal-validation/benchmark-metal-shape-workspaces-rope-b2-l512.json`）。変更前20回の8,334 / 749,024から60件 / 300,416 bytes減った。中央値は約1.1%増えており速度改善の証拠とはしないが、50回でも秒単位のtailは観測されなかった。workspace GPU保持は894配列 / 2,623,995,904 bytesで同じ。RoPE shared表の保持はworkspace統計に含まれない。logit誤差1.9311905e-5。実モデル15ケース×3回の再検証は `/private/tmp/jeff-rope-two-tables-validation.log` で実行する。
- 実モデル15ケース×3回の再検証は合格、最大誤差3.540516e-5。変更後JET6対象も報告なし（`/private/tmp/jeff-shape-workspaces-rope-b2-profile.log`）。warm @timed579.705625 ms / 449,760 bytes / GC時間0。全割当Profile8,277件で、kernel起動サイト6,165件 / 261,168 bytes（件数の約74%）、MPS command wrapper26 / TD0 / KernelState412。RoPE cos/sin表生成サイトは消えた。型別には引数tuple、device lookupのtuple/boxing、encoder、KernelState、HostKernelのboxing等が残る。Metal1.11.1のHostKernelはautoreleasepoolを経てnospecializeなlaunchへ渡し、GPU引数のencodeとqueued-operation rootsを保つため、表をcacheするだけでこの割当をなくせるわけではない。次の候補は起動回数の削減と、所有条件を維持した引数管理の改善。性能スキルを現行4flag/計算長/case選択/LRU/遅延分布の手順へ更新し、PythonCall経由のquick_validateに合格した。

## workspace queue 明示の候補

- Metal依存内の最初のstack frameも集計するようinspect_nativeを拡張した。B2全割当ログ `/private/tmp/jeff-metal-dependency-allocation-profile.log` ではencode_arguments_nospec!1,250件 / 43,744 bytes、record_operation!277行1,246 / 58,480、278行948 / 111,328、global_queue838 / 26,816、launch_with_queue824 / 56,960等。これらは先のJeffClient caller別集計と同じ記録の別分類であり、足し合わせない。record_operation!はGPU完了までのrootsを保持するため省略できない。
- active workspaceにMetal.BatchedCommandQueue自身も保持し、cached kernel起動のqueue keywordに同じqueueを渡す候補を追加した。各scopeは既存のqueue identity検査を継続し、kernel.deviceとqueue.deviceが一致する場合だけ使い、非workspace/異なるdeviceは通常経路へ戻る。HostKernelのautoreleasepool、引数encode、record_operation!、batch flushはMetal通常経路のまま。primitive verifierは合格（`/private/tmp/jeff-workspace-launch-queue-primitives.log`）。B2/50回測定は `/private/tmp/jeff-workspace-launch-queue-b2-benchmark.log` で実行する。実モデル全ケース・JET・割当再Profileは未完了。
- B2/L512/active512・256/F32・workspace/trim/shape有効・packed無効・50回は **7,864 allocations / 435,488 bytes / 中央値581.4081245 ms**、min/p95/max572.152292/591.514458/593.695083 ms。直前の8,274 / 448,608から410件 / 13,120 bytes減った（workspace scope経由のqueue検索を省略）。中央値はほぼ同じで速度改善の証拠とは扱わない。GPU保持894配列 / 2,623,995,904 bytesも同じ、最大logit誤差1.9311905e-5。primitive verifierにactive workspace内のcaptured kernel起動・task queue同一性・readback後の出力検証を追加し再実行する。全ケース/JET/Profileは引き続き未完了。
- 追加kernel/queue/readback検証を含むprimitive verifierは合格した。実モデル15ケース×3回は `/private/tmp/jeff-workspace-launch-queue-validation.log` で実行する。
- 実モデル15ケース×3回は合格、最大誤差3.540516e-5。JET/全割当の再検査は `/private/tmp/jeff-workspace-launch-queue-b2-profile.log` で実行する。次のGPU起動削減候補として、DeltaNetの入力maskを直前のRMS正規化に融合できるか検討する。maskはqkv/a/b/zの全projectionに必要なので、convolutionへの単純移動は意味を変える。full layerや最終readoutの正規化にmaskを適用してはいけない。層間residual入力正規化の融合経路と直接native_layer経路の両方を考慮する必要がある。
- queue明示後のJET6対象は報告なし、MPS command wrapper26 / TD0 / KernelState412を維持。通常HostKernel経路のままqueue検索を削減した候補を採用する。設定はworkspace有効時に限り、workspaceのGPU保持量は変わらない。
- 全割当Profile7,868件、warm @timed573.741 ms / 436,640 bytes / GC0。global_queueサイトは838件 / 26,816 bytesから428件 / 13,696 bytesへ減り、BenchmarkToolsの410件削減と一致した。`ed3db06` としてコミットした。

## DeltaNet入力maskと正規化の融合候補

- `JEFF_METAL_FUSED_DELTA_MASK=1` の候補を追加した。RMSの出力destinationをGPUへadaptするMaskedNormalizationOutputで包み、setindex!時に列maskを掛ける。正規化の平均・重み・residual更新は元のkernelのままで、normalizedだけをmaskedにする。Metal native_hidden_forwardではDeltaNetかつkey_dim<=256の入力正規化/層間residual入力正規化に限って使う。full attention、post norm、final readoutの正規化は変更しない。coreにpremasked Val hookを追加し、DeltaNet側はすでにmask済みのprojection入力にdelta_masked_inputを重ねない。直接native_layerや未対応幅は元の処理を継続する。
- fixture metal5ケースは合格、最大誤差2.3841858e-7（`/private/tmp/jeff-fused-delta-mask-fixture.log`）。primitive verifierにmask付きRMSとmask付きresidual入力RMSを追加し、幅8〜1024、centered/noncentered、interior holes、residual自体をmaskしないことを検証する。ログは `/private/tmp/jeff-fused-delta-mask-primitives.log`。実モデル・JET・Profile・全体性能は未完了。
- 追加masked RMS / residual RMSを含むprimitive verifierは合格した。destination生成ではmask列数・正の幅も検査する。実モデル15ケース×3回の検証は `/private/tmp/jeff-fused-delta-mask-validation.log` で実行する。GPUのmask乗算は既存のnormalization kernel出力書き込み時に行い、追加の整数列index計算とmask読出しがあるため速度改善は実測で判断する。
- 実モデル15ケース×3回は合格、最大logit誤差3.540516e-5。B2/L512・50回の全forward測定は `/private/tmp/jeff-fused-delta-mask-b2-benchmark.log` で実行する。数値一致は確認済みだが割当・速度の効果と型検査は未確定。
- 融合有効B2/L512/active512・256/F32・workspace/trim/shape有効・packed無効・50回は **7,516 allocations / 429,152 bytes / 中央値583.471562 ms**、min/p95/max572.948291/598.175375/598.574041 ms。非融合直前7,864 / 435,488から348件 / 6,336 bytes減る。保持858配列 / 2,567,372,800 bytesで、36配列 / 56,623,104 device bytes減った。中央値は581.4081245→583.471562 ms（約0.35%増）で改善の証拠ではない。KernelState/引数wrapper/型の再検査は `/private/tmp/jeff-fused-delta-mask-b2-profile.log` で実行する。候補は既定無効のまま。実幅1024でのmask列index除算をcompile-time幅で省く余地と、wrapper管理費用を調べる。
- 上記Profile/JETは完了した。JET6対象報告なし、warm @timed578.419416 ms / 430,304 bytes / GC0、全割当7,516件、MPS command wrapper26 / TD0 / KernelState376（非融合412から36個減）。正規化kernel自身が持つ列番号をstore_normalization!へ渡す版に変更し、MaskedNormalizationOutputの幅fieldと追加整数除算を除去した。GPU出力型によるdispatchでmask付き/なしを書き分け、他のnormalization kernelも同じhelperを使う。列番号再利用版のprimitive検証は `/private/tmp/jeff-fused-delta-mask-column-primitives.log` で実行する。
- 列番号再利用版のprimitive verifierは合格した。実モデル15ケース×3回は `/private/tmp/jeff-fused-delta-mask-column-validation.log` で実行する。別の候補として、trim後に全token activeとなる行はmask乗算自体を省略できる。ただしslot順序で管理するworkspaceでは、同じ計算長のall-active行とmask穴のある行で配列生成位置が変わるとshape置換が増えるため、単純なearly returnだけを採用してはいけない。割当順序の維持かworkspace key/operation管理の変更と併せて測定する必要がある。
- 列番号再利用版も実モデル15ケース×3回合格、最大誤差3.540516e-5。CPU fixture5ケースも合格、最大3.874302e-7（`/private/tmp/jeff-fused-delta-mask-cpu-fixture.log`）。B2の50回測定は `/private/tmp/jeff-fused-delta-mask-column-b2-benchmark.log` で実行する。slot順序を持つworkspaceの入力依存分岐に関する検証原則をAGENTS.mdへ追加した。
- 列番号再利用版B2・同設定50回は **7,516 allocations / 429,152 bytes / 中央値593.5079375 ms**、min/p95/max578.032209/603.181042/604.668041 ms、保持858配列 / 2,567,372,800 bytes。前の整数除算版583.471562 msより遅いため、除算削除によるlatency改善を主張しない。測定条件の時間変化の可能性はあるが原因の証拠ではない。現行コードでfusion無効controlと有効repeatを測って判断する。現行JET/Profileは `/private/tmp/jeff-fused-delta-mask-column-b2-profile.log` で実行する。
- 列番号再利用版のJET6対象は報告なし。warm @timed577.656166 ms / 430,304 bytes / GC0、全割当Profile7,515件（BenchmarkToolsとは1件差）、MPS command wrapper26 / TD0 / KernelState376。現行コードのfusion無効・同条件50回controlを `/private/tmp/jeff-fused-delta-mask-column-control-b2-benchmark.log` で実行する。
- 現行fusion無効control・B2同条件50回は **7,866 allocations / 435,552 bytes / 中央値585.829396 ms**、min/p95/max574.36925/599.809542/602.011333 ms。候補導入前7,864件との差2件 / 64 bytesには各rowの設定確認が含まれる。融合有効は350件 / 6,400 bytesと56,623,104 device bytes少ないが、593.5対585.8 msの分布は重なり速度改善は確認できない。有効repeatを `/private/tmp/jeff-fused-delta-mask-column-repeat-b2-benchmark.log` で測定する。
- 有効repeat50回は7,516 / 429,152を再現、中央値580.224729 ms、min/p95/max570.341875/588.542292/590.682542 ms。593.5→580.2 msと測定間で変わるため対照とのlatency差を因果的な改善としない。融合有効時、PreparedMetalMask.hostがすべて1ならnormalization destinationを通常配列に戻し、mask乗算/MaskedNormalizationOutput自体も省く候補を追加した。premasked Valはtrueのままで独立delta mask slotを生成せず、all-active/穴あり両方の生成slot数は同じ。maskがPreparedMetalMaskでない場合はall-activeと推測しない。実モデル検証は `/private/tmp/jeff-fused-delta-mask-all-active-validation.log` で実行する。
- all-active省略版も実モデル15ケース×3回合格、最大誤差3.540516e-5。mask穴を持つ行とall-active行の混在を含む。B2/50回測定は `/private/tmp/jeff-fused-delta-mask-all-active-b2-benchmark.log` で実行する。
- all-active省略版B2・同条件50回は **7,480 allocations / 424,544 bytes / 中央値581.5682295 ms**、min/p95/max571.012791/590.594458/596.004084 ms。wrapper付き版7,516 / 429,152から36件 / 4,608 bytes減った。対照7,866 / 435,552からは386件 / 11,008 bytes減り、GPU保持は858配列 / 2,567,372,800 bytes。速度差は測定範囲が重なるため確定しない。同じlength9でall-active/interior/prefix maskを切り替え、GC後にもworkspace全slotの配列identityが保たれる専用fixture probeをprimitive verifierへ追加した。ログは `/private/tmp/jeff-fused-delta-mask-all-active-primitives.log`。
- 同長mask切替とGCのslot全identity検証は合格した。最新JET6対象も報告なし、warm @timed574.8355 ms / 425,696 bytes / GC0、全割当Profile7,479件、MPS command wrapper26 / TD0 / KernelState376（`/private/tmp/jeff-fused-delta-mask-all-active-b2-profile.log`）。workspace無効・packed有効の組合せは `/private/tmp/jeff-fused-delta-mask-packed-workspace0-validation.log` で実モデル15ケース×3回を検証する。READMEへ既定無効・割当/GPU保持の削減・速度改善未確認を追記し、性能スキルも5flagと同長mask切替/割当分類の手順へ更新した。
- workspace無効・packed有効の融合版も15ケース×3回合格、最大誤差3.540516e-5。workspace/packedとも有効の組合せを `/private/tmp/jeff-fused-delta-mask-packed-workspace1-validation.log` で検証する。更新した性能スキルはPythonCall経由quick_validateに合格した。
- workspace/packedとも有効の融合版も15ケース×3回合格、最大誤差3.540516e-5。残るworkspace無効・packed無効の組合せを `/private/tmp/jeff-fused-delta-mask-unpacked-workspace0-validation.log` で検証する。これらの実モデル検証はtrim有効時の結果であり、全設定の組合せを検証したとはしない。
- workspace無効・packed無効も15ケース×3回合格、最大誤差3.540516e-5。trim有効時のworkspace/packedの4組合せを確認した。融合はhost割当とGPU保持量を減らす実験用opt-inとして採用し、latency改善は未確認のまま記録する。

## 真のbatch推論の実装着手

- mask融合は `8777e03` としてmainへpushした。次は行ごとのforwardをまとめるため、まずprivateなbatched_causal_depthwiseを追加した。inputはchannels×(sequence_length*batch)で、各sampleのtokenを連続列へ配置する。kernelは各列のsample開始位置を計算し、それより前のtapを0として扱う。既存単独行kernelと通常forwardは変更しない。
- Layaのsplit_rope_kernel/merge_heads_kernelとattention_unfusedでは、head*batchをMPSGraphのbatch軸へまとめ、投影用layoutへ戻している。Jeffもこのlayoutを利用できるが、DeltaNetの畳み込みとrecurrent stateはsampleごとに独立させる必要がある。今回の畳み込みだけではbatch推論や速度改善は成立しない。
- primitive verifierへchannels7/128、系列長1/3/9/65、batch1/2/3、tap1/4の48組合せを追加した。CPUの各行独立処理と2回比較しGCを挟む。さらに最初のsampleを100へ変更して後続sampleへの影響がないことを確認する。全primitive実行ログは `/private/tmp/jeff-batched-convolution-primitives.log`。exit0で完了し、新しいbatch境界/GC検証と既存primitive検証に合格した。モデル全体への接続・性能測定・型検査は未実施。
- batch DeltaNet recurrent kernelを追加した。gridの第2軸をvalue_heads*batchとし、sample/headを分解してflat token offsetを計算する。stateは各head/sampleのthreadgroupごとに0から始め、query/key/value/beta/decay/outputは同じsampleの列だけを参照する。既存単独行kernelは変更しない。host側は完全な系列列数、head比、packed channel数とQ/K/gate形状を検査する。key_dim<=256の範囲を対象とする。
- 幅7/128/256、系列長1/9/65、batch1/2/3の27組合せを、各sample独立のCPU state更新と比較した。2回実行とGC、最初のsampleのpacked value変更後にも後続sample結果が変わらないことを検証し、`/private/tmp/jeff-batched-recurrent-primitives.log` はexit0で合格した。新しい `tools/verify_metal_primitives.jl batch` でbatch kernelだけを検証できる。decay factorをtuple mapの前に一度計算する最終版は `/private/tmp/jeff-batched-recurrent-final-primitives.log` で再検証中。full attentionと推論本体はまだbatch対応していないため、全体速度改善は未確認。
- 最終版もexit0で完了し、batch畳み込みとrecurrentのCPU一致・sample境界・GC検証に合格した。次はfull attentionのRoPE/head layout、per-sample mask softmax、merge gateをbatch対応し、モデル全体の独立参照検証と計測へ進む。
- batch softmaxを追加した。scoresは(key, query, heads*batch)、maskはsample-contiguous token列とし、columnからsampleを計算して該当行のmaskだけを参照する。因果条件はsample内query/key番号で比較し、全masked queryは従来と同じ-floatmaxを使う。通常softmaxとforwardは変更しない。系列長1/9/65、heads1/3、batch1/2/3の18組合せに対し独立CPU softmax、2回実行/GC、他行mask変更による影響なしを検証する。ログ `/private/tmp/jeff-batched-softmax-primitives.log` は実行中。全体のbatch推論はRoPE/layout/mergeとforwardへの接続が未完了で、速度・割当の改善量は未測定。
- 上記batch verifierはexit0で完了し、畳み込み/recurrent/softmaxの3検証とも合格した。softmaxの18組合せには先頭padding、interior mask穴、有効keyがないqueryも含む。
- `ext/metal_batch_attention.jl` にbatch用RMS/RoPE/head変換、merge gateとfull attentionを追加した。projection列はsampleごとの連続token、attention配列は(head_dim, sequence_length, heads*batch)。RoPE tableの参照にはsample内token、projectionにはsample offset込みtokenを使う。KV head複製とgate sourceもsample内head/位置から計算する。head_matmulは既存のMPSGraph batch積を再利用する。通常full attention/forwardは変更しない。
- 幅4/7/256、grouped/non-grouped KV、partial/full RoPEを持つ3構成で、各行独立のJulia CPU full_attentionとbatch GPU結果を比較する。GC後の2回目と、sample逆順に対するoutputの同じ並べ替えも検証する。ログ `/private/tmp/jeff-batched-attention-primitives.log` は実行中。実モデル全体の検証と速度/割当計測はforward接続後に行う。
- 上記batch verifierはexit0で完了し、batch full attentionのCPU一致・sample逆順・GC検証に合格した。既存batch畳み込み/recurrent/softmax検証も合格した。次はvalidated入力からのbatch forward hook、DeltaNetのまとめたprojection、各行最後のcolumnのreadoutを実装する。
- `JEFF_METAL_BATCHED=1` でbatch>1のMetal backendがvalidated入力からbatch forwardへ入るprivate hookを実装した。共通trim開始位置は各行の開始位置の最小値なので、短い行のpaddingも残す。ID/maskはsample-contiguousにflattenし、projection/MLPをまとめ、畳み込み/recurrent/full attentionだけsample境界を分離する。最後は各sample最終columnのresidual+MLPをH×Bへgatherし、final RMS/readoutとCPU返却を一度行う。既定無効、B1/CPU/未対応幅は従来経路。batch版は入力maskを独立kernelで適用し、既存mask融合flagはまだbatchには適用しない。
- batch prototypeは単一workspaceを再利用し、inactiveな従来shape bankはbatch移行時にclearする。activeなnested scopeはclearしない。異なるB/Lでは形状置換があり、batch用shape bankは未実装。benchmarkにはbatch requested/executionと共通計算長を記録する。
- trim/workspace/shape有効、packed/mask融合無効の実モデル15ケース×3回はexit0で合格、最大logit誤差3.361702e-5（`/private/tmp/jeff-batched-model-validation.log`）。B2/L512/active512・256の20回測定を `artifacts/metal-validation/benchmark-metal-batched-b2-l512.json` へ保存する。batchでは計算長512/512、従来row trimは512/256なので演算量は同じでない。JET/Profileと他設定の組合せ・モデル全体の所有/slot検証は未完了。
- 上記batch B2/L512の20回は **4,152 allocations / 258,208 host bytes / 中央値792.445 ms**、min/p95/max771.24225/799.361792/800.467416 ms。workspace448配列 / 3,532,177,408 bytes、TD296/feed199。row trimの直近非融合control7,866 /435,552/585.829396 ms（50回）に比べhost割当は約47%少ないがlatencyは約35%長く、buffer保持も約0.91 GB多い。row側の計算長512/256に対しbatch512/512なので、演算量増加を含む比較でありGPU管理費用だけの影響ではない。現段階ではrow trimの置き換えに採用しない。短い/同長batchの比較、型とProfile、workspace ownership検証を続ける。original Python F32の1300.984021 msよりは短いが、我々のrow版より速いとは主張しない。
- batch初回Profile（`/private/tmp/jeff-batched-b2-profile.log`）はwarm @timed791.826334 ms /259,360 bytes /GC0、全割当記録4,147件。cached kernel起動2,943件 /125,344 bytes、TD新規0、MPS command wrapper13、KernelState207。row版412 launchesに対して約半分だが、latency短縮につながっていない。CPU samplingはThread1の6,847 snapshots中6,472がkevent、GPU batch flush/limit_inflightの待ちstackも含む。CPU ProfileからGPU kernel別の時間や転送単体時間は導けない。
- JETはlogitsのbatch calleesでruntime dispatch3件を検出した。batched_prepare_headsのquery/KV2経路とbatched_delta_recurrentで、実行時幅からVal(cld(width,32))を作りcached launcherへ渡していた。実モデルhead_dim256/key_dim128では明示Val(8)/Val(4)でcached起動し、それ以外の幅は既存単独行と同じMetal.@metal経路へ分けた。JET/割当の再検査は `/private/tmp/jeff-batched-specialized-b2-profile.log` で実行中。
- 修正後batch primitive verifierはexit0で合格（`/private/tmp/jeff-batched-specialized-primitives.log`）。追加したtiny model batch forward probeでは同じB2/L9でmask値を変えてGCしても全workspace配列とtensor-data tupleのidentityが維持され、B3/B2へのshape変更後もCPU参照と一致した。終了時workspace.active=false、clearとENV復元も確認した。実モデル再検証・比較benchmarkは引き続き必要。
- 定数Val経路のJET/Profileはexit0で完了。JET6対象すべて報告なし、warm @timed789.35225 ms /256,960 bytes /GC0、全割当記録4,057件（修正前4,147件から90減）。MPS command wrapper13、TD新規0、KernelState207は同じ。typed dispatchの修正で管理割当は減ったが、約0.79秒の全体速度改善の証拠にはならない。READMEへbatch実験の共通padding/単一workspace/非融合maskと初回速度悪化を追記した。同長B2/L1の50回batch測定を `artifacts/metal-validation/benchmark-metal-batched-b2-l1.json` へ実行する（`/private/tmp/jeff-batched-b2-l1-benchmark.log`）。
- 同長B2/L1のF32/workspace/trim/shape有効・packed/mask融合無効で50回比較した。batchは **3,947 allocations /213,168 bytes /中央値22.776521 ms**、min/p95/max22.403791/25.911417/32.454625 ms。現行row controlは **7,751 /415,296 /43.8900205 ms**、min/p95/max43.228208/47.263541/50.776041 ms（`artifacts/metal-validation/benchmark-metal-row-control-b2-l1.json`）。計算長は両方1/1、batch→row順に測定し、batchは中央値約48.1%短い（約1.93倍）、割当3,804件 /202,128 bytes少ない。workspace保持はbatch9,404,416、row7,913,472 bytes。最大logit誤差はいずれも3.361702e-5。短い同長batchで有効な証拠だが、長系列やmixed paddingへ一般化せず、中間長・再測定も続ける。
- B2/L65・同じF32/flag設定の50回比較ではbatch **3,971 allocations /218,736 bytes /中央値112.488584 ms**、min/p95/max111.669958/113.5605/113.850333 ms。row control **7,793 /418,272 /105.4916875 ms**、min/p95/max104.779666/106.119125/114.781458 ms。host割当3,822件 /199,536 bytes少ないがbatch中央値は約6.6%長い。workspace保持batch428,310,528、row329,613,312 bytes。結果は `benchmark-metal-batched-b2-l65.json` と `benchmark-metal-row-control-b2-l65.json`。mixed paddingを含むため、次に元の独立参照case5の第1行を2行へ複製した `reference-b2-equal-l65.json` を作った。入力と対応するPyTorch logitを両方複製しており、新しく計算した参照ではない。同長版のbatch50回測定を `/private/tmp/jeff-batched-b2-equal-l65-benchmark.log` で実行する。
- ユーザーの区切り・結論依頼に従い、残件をGitHub Issueへ登録した。#1 batchの適用条件と設定/所有検証、#2 層間bufferとtensor-data再利用によるGPU保持量削減、#3 kernel引数encode/boxing/rootsの残存heap割当削減。同長L65 batchの測定は完了したが対応row比較は未実施であり、Issue #1へ引き継ぐ。未確認の条件でbatch高速化を主張せず既定無効を維持する。
- 同長B2/L65 batchの50回中央値112.428438 ms、3,971 allocations /218,736 bytes、最大logit誤差1.4781952e-5（`benchmark-metal-batched-b2-equal-l65.json`）。同じ入力のrow結果がないため速度比は未確定。

## 2026-10-01: README の実際の0.8Bデモを再測定

- `mstrasser/Jeff-Qwen3.5-0.8B` revision `0f212b3e72acb4dde3f7da61e925d6ab7f819990` の safetensors を使用。tiny fixture/ONNX graphではない。README parcelデモと旧独立PyTorch参照case1の入力が同一であることを確認し、入力・参照logitsを `examples/data/parcel_reference.json` に保存した。
- Apple M4、Julia1.13.1/Metal1.11.1/PyTorch2.14.0、Float32、B1/L256/active101、各20回、CPU8threads、バックエンドを順に独立実行。load/compile/tokenization/calibrationを除外し、readoutとCPUスコア返却・GPU同期を含む。PythonはGPU入力を事前準備、JuliaはCPU入力のuploadをforward内に含む。
- 中央値/p95(ms): Julia CPU1893.106/2062.085、Python CPU3015.735/3023.922（1.59倍）、Julia Metal既定197.007/204.068、Python MPS-F32364.013/375.429（1.85倍）、Julia workspace+trim89.379/93.190（Python比4.07倍）。任意設定はworkspace/trimのみ1、他flagは0。trimは計算長101、既定は256。
- 最大絶対logit誤差: CPU1.049e-5、Metal既定6.676e-6、trim7.629e-6。先頭mask0のみ除去し有効tokenは残す。今回と既存の参照検証の数値一致は確認したが、大規模な分類精度評価とは区別する。
- Julia CPU loading0.780s/first forward5.349s、Metal既定2.247s/15.290s。package import/downloadはこの時間に含まれない。
- 元Jeff commit `f06788292874c21a5b5c41549ac220dd9e15da7f`、FLA/causal-conv1dなしのPyTorch fallback。MLX比較ではない。生JSONはignored `artifacts/metal-validation/demo-0.8b-{cpu,metal,metal-trim,python-cpu,python-mps}.json`。公開表・再実行手順は `docs/src/performance.md`。

## 2026-10-01: GPUの知見をCPUへ適用

- CPU専用 `tools/profile_native_cpu.jl` を追加。@code_warntype/JET、warm @timed、5 forwardsのProfile、5%のProfile.Allocsを記録。変更前JETはNo errors detected。warm1.856s/3,066,053,744 bytes/GC0.0168s。CPU8,350 snapshotsの中にBLAS GEMM、triangular solve、exp、sliceコピーがある。サンプルbyte上位はcausal_depthwiseのslice broadcast（native.jl旧241、244）、次にmatmul。sampled bytesを総割当や時間比率と混同しない。
- `src/native_cpu.jl` にMatrix{Float32}専用畳み込みを追加。tap順を保持してcolumn-major loopで既存outputへ直接加算しSiLUをin-place適用。channel数を検査してから@inboundsを使う。GPUのgeneric methodは保持。
- CPUのhidden forwardでは最後の層のAttentionまでは全系列を計算し、最後のresidual/RMS/MLPだけ最終columnにする。MLPは位置ごとに独立でreadoutが最終columnしか消費しない。Metalで使った知見をCPUに移した。CPU重みはReinterpret/ReshapeでもProfileでは既にBLASへ到達しており、重量の形式だけを変える必要は確認されていない。
- `JEFF_CPU_TRIM_PADDING=1` を追加（既定0）。先頭mask0のみ除去、interior holesは残す。Metalの同名でないflagと独立。benchmarkはCPU計算長とtrim flagを記録し、`JEFF_BLAS_THREADS`（既定8）でCPU threadを変えられる。
- real0.8B parcel、B1/L256/active101、Float32、各20回、8BLASthreads:変更前中央値1893.106ms/3,060,843,632 bytes/126,111allocations→変更後1757.048ms/2,034,107,424 bytes/125,408allocations。約7.2%latency減、約33.5%heap bytes減。p951863.412ms、最大logit誤差1.1444e-5。
- 同変更+CPUtrim（計算長101）中央値673.920ms/p95695.165ms/840,342,656 bytes/66,679allocations、maxerror1.2398e-5。変更前の約2.81倍、元Python CPU3015.735msの約4.47倍。Python側はL256計算なのでpadding演算削減も含む。BLAS1threadsは1195.544ms/p951216.943msで8threadsより遅く、単スレッド化は採用しない。
- 生データ `artifacts/metal-validation/demo-0.8b-cpu-{fused,trim,trim-blas1}.json`、profile変更前 `profile-cpu-demo.log`、変更後 `profile-cpu-demo-after.log`。CPUscratch再利用/DeltaNet中間配列削減は残る。新しいCPU変更の全入力・モデルへの一般化はこのデモの測定だけで主張しない。
- 変更後のCPU profilerもexit0。JETはNo errors detected、warm1.768s/2,034,107,424 bytes/GC0.0125s。convolutionの旧slice/broadcast割当は消え、残りの主な割当はmatmulとDeltaNet chunk中間配列。samplingはThread1の7,896 snapshots、Thread2はidleであり二重計上しない。

## 2026-10-01: CPU DeltaNet chunkとApple Accelerate

- CPUの三角解法だけをBLAS.trsm!へ差し替えたtrialはtrim中央値662.218ms（20回）、以前673.920msと範囲が重なるため単独の高速化とは主張しない。その専用delta_solve methodは最終版から除き、CPU chunk内の所有RHSだけを直接trsm!で更新する。
- CPU専用delta_attentionを実装。共有Q/Kの正規化をvalue headごとに繰り返さず一度行い、chunkはviewで参照。system/intraのdecay broadcastをin-place化し、triangular RHSは直接埋めてsolve、corrections/result/state更新をmul!のalpha/betaで融合。stateはheadごとにzero resetする。pair_decayの式は元のcumulative[i]-cumulative[j]であり、初期trialの符号違いは独立参照guardで検出して修正した。GPU処理は既存methodを使う。
- OpenBLAS8threads/real0.8B/parcel/20回でuntrim中央値1529.238ms/p951593.114ms/1,277,654,112 bytes/56,036allocations。直前1757.048ms/2.03GBより短い。trimは671.481ms/p95868.609ms/504,848,576 bytes/30,103allocations、maxerror1.2398e-5。直前trim673.920msに対し中央値ほぼ同じ・tail悪化もあり、割当低減だけで速度改善を主張しない。
- Laya.jlのext/LayaAppleAccelerateExt.jlとsrc/backends.jlを再読。optional AppleAccelerate importでprocess-wideにLBTをAccelerateへforwardする知見を採用。tools依存にAppleAccelerate=0.7.0を追加し、native CPU demo/benchmark/profilerでJEFF_CPU_ACCELERATE=1のときだけimport。macOS13.4以上でforwardを確認し、他OS/unsupported macOSは明示error。コアruntime依存は増やさず、Metal/Python/ONNXにCPUを委譲しない。
- AccelerateはOpenBLASとthread APIが異なる。LBT/BLAS.get_num_threads()は8だがAppleAccelerate.get_num_threads()は10（framework-managed）。同一8threadsのライブラリ比較とは主張しない。benchmark JSONはblas_config/accelerate_version/accelerate_threadsを保存する。
- 同real0.8B/Float32/B1L256active101、20回: Accelerate untrim中央値533.254ms/p95572.821ms、trim254.157ms/p95266.436ms、独立repeat trim257.283ms/p95273.999ms。trimは計算長101、最大logit誤差1.1444e-5、割当504,848,576 bytes/30,103件。公開表はrepeat257msを使う。前回trim674msの約2.62倍、元Python CPU3015.735ms比約11.72倍（fallback実装/異なるCPU math kernels/trim演算量削減も含む）。
- Accelerate+trimの独立reference6caseで各3回benchmarkし、初回reference guardを通過した。英日混在B3、B2L1/65/512、B3interior masksL65/129。最大誤差3.6001205e-5。これは検証入力の数値一致であり分類精度dataset評価ではない。artifact demo-0.8b-cpu-accelerate-case-{1,2,5,12,14,15}.json、benchmark-cpu-accelerate-cases.log。その他主要artifactはdemo-0.8b-cpu-{chunk,chunk-no-trim,accelerate,accelerate-no-trim,accelerate-repeat}.json。
- 変更後CPU profiler（Accelerate+trim）はexit0、JET No errors detected、warm267.703ms/504,848,576 bytes/GC2.25ms、Profile thread1 1,032snapshots、thread2idle。profile-cpu-accelerate.log。CPU scratch再利用、normalization/activationのvector化は次の候補。
- optional fast CPU exampleは引数なしHF_HUB_OFFLINE=1で実行済み、Device:cpu/delivery0.996566/confidence0.993133。docs/READMEに任意設定と実測結果を記載し、Documenter build exit0。

## CPU極限チューニング: chunk scratchの層内再利用（作業中）

- active goal CPU実装を極限まで高速化。前turnはCPU chunk改良/Accelerate導入、独立参照と計測、commit7fc0082のためprogressとして扱う。
- cpu_delta_buffersにpair/system/intra、weighted/ending/scaled_query、RHS2種、corrections/resultをまとめた。full64とtail用の2組を層のforward内に所有し、head/chunk間で再利用。task共有cacheではないので他forwardとの競合はない。beta0のmul!で出力全域を上書きし、RHSとcorrectionsも利用前に完全上書きする。
- Accelerate+trimのreal0.8B parcel20回は中央値256.270ms/p95265.992ms/385,530,176 bytes/13,903allocations/maxerror1.1444e-5。直前repeat257.283ms/504,848,576 bytes/30,103allocations。速度は範囲が重なるがheap bytes約23.6%、割当数約53.8%減。生JSON demo-0.8b-cpu-chunk-buffers.json。
- 新scratch経路の15ケース各3回benchmark/reference guardは cpu-buffers-cases.log に実行中。完了確認後にまとめる。変更後のProfile/JETとvector mathの検討は未完了。AppleAccelerate0.7 array.jlにexp!(out::Array,input::Array)があり、次の候補はscalarexp/SiLUを任意のvForce経路へ置き換えること。追加scratchコストと全forwardの数値/速度を測定して判断する。
- scratch層内再利用の15ケース×各3回benchmarkはexit0、最大logit誤差3.6001205e-5（cpu-buffers-case-{1..15}.json）。active goal前turnはsource変更/20回計測でprogress。今回もvector経路の実装と計測でprogress。
- AppleAccelerateのoptional extensionを追加（weakdep、compat=0.7.0）。JEFF_CPU_VECTOR_MATH=1のみvForce exp!を使う。最初のSiLU/MLP temp版20回は235.755ms/p95333.474ms/463,913,888 bytes/14,071allocations/maxerror1.1921e-5。中央値は下がるが追加tempとtail増加を認め、既定は無効。
- MLPの所有gate/upを両方消費するprivate cpu_owned_mlp_gate!へ分離。up*=gateしてgateを負数expへ上書きし、gate=up/(1+exp)を作る。公開でない既存native_mlp_gate!のup保持の挙動は変更しない。これで20回216.671ms/p95223.387ms/430,359,584 bytes/13,999allocations/maxerror1.2398e-5（demo-0.8b-cpu-vector-owned.json）。
- さらにowned qkv projectionを畳み込み完了後のSiLU exp scratchとして再利用するprivate cpu_owned_causal_depthwiseを追加。通常causal_depthwiseは入力を上書きしない。JEFF_CPU_VECTOR_MATH=0では所有inputもそのままで元のscalar処理。新しいall-owned版50回はdemo-0.8b-cpu-vector-all-owned.jsonに実行中。全ケースでのvector経路の検証・Profile/JET・default無効時の再計測は未完了のためまだcommitしない。
- all-owned vForce版50回はexit0、中央値218.832ms/p95237.897ms/min214.374ms/max243.410ms/385,531,520 bytes/13,945allocations/maxerror1.2398e-5。vector無効256.270msから中央値約14.6%減、追加scratchのheap増加を解消。50samplesと20samplesの比較なので今後同条件repeatも行う。
- vector有効15ケース×3回benchmark/reference guardを cpu-vector-owned-cases.log に開始した。Profile/JETとscalar fallback確認はまだ必要。optional flag既定0のまま、目標完了は未証明。
- all-owned vector mathの15ケース×3回benchmarkはexit0、最大誤差3.3974648e-5（cpu-vector-owned-case-{1..15}.json）。前goal turnは所有配列再利用/50回計測/15case起動でprogress、今回はRMS in-place/Accelerate threading比較/Profilerでprogress。
- JEFF_CPU_INPLACE_DELTA_RMS=1のprototypeを追加（既定0）。state更新済みで以後raw resultを使わない地点のRMSを、同じ結果bufferへ書く。通常native_rmsの破壊的挙動は変えない。weights幅を検査し、列ごとのsum(abs2,view)とscale/weight乗算。50回は218.046ms/p95236.072ms/369,828,032 bytes/9,931allocations/maxerror1.2398e-5（demo-0.8b-cpu-inplace-rms.json）。既存218.832msと範囲が重なるので速度改善とは主張せず、alloc削減として扱う。RMS1の全ケース確認は未完了。
- benchmarkにJEFF_CPU_ACCELERATE_THREADSを追加。AppleAccelerate.set_num_threadsは1=single、>1=framework-managedであり厳密n本の指定ではない。all-owned/vector/RMS/trim有効でAccelerate singleの50回中央値219.573ms/p95236.404ms、10thread auto218.046msと同等。単スレッドを高速版として採用しない。
- profile_native_cpuのJET targetにoptional AppleAccelerate extensionも含めた。初回はJET No errors detected、warm251.278ms/369,959,104 bytes/GC0.938ms。ただしJET依存Reviseのfolder monitorがsoft FD256を超えてbackground EMFILE errorを出した。モデル推論のerrorではないがきれいなprofileにするため、その子shellだけulimit -n4096として再実行中（profile-cpu-vector-owned-fd4096.log）。JULIA_REVISE=manualではfolder watch自体を止めないため代替にしない。
- child shellのulimit4096再実行もexit0/No errors detected/warm224.139msだったがRevise background EMFILEは残ったため、FD上限だけの回避は不十分と判明。支持されるJULIA_REVISE_POLL=1でfolder monitorを使わない再実行をprofile-cpu-vector-owned-poll.logに開始。pollingによるbackground activityはプロファイル時に区別する。
- JULIA_REVISE_POLL=1のprofile再実行はexit0、EMFILE/Unhandled task errorなし。core+AppleAccelerate extensionを対象にJET No errors detected。warm225.160ms/369,959,104 bytes/GC0.704ms（profile-cpu-vector-owned-poll.log）。Revise polling background activityとモデルstackを区別し、時間の確定値は別BenchmarkTools trialを使う。
- RMS in-place1＋vector1＋Accelerate＋trim1の15ケース各3回benchmark/reference guardをcpu-inplace-rms-cases.logに開始。次のturnは同じ実行handleを確認し、terminalと15JSONのmaxerrorを検査する。CPU極限goalは引き続きactive、workspace再利用と行列積のpacking/threading等の候補をまだ監査していない。

## CPU tuning handoff (2026-10-01)

- 現状commit/push・残件Issue化の指示で今回の作業をまとめる。未計測のMLP packing試作は除外。極限最適化が完了したとは扱わない。
- RMS/vector/Accelerate/trimの15ケース各3回benchmarkはexit0、15JSONを確認、最大logit誤差3.3974648e-5。vector/RMSは既定無効。初回guardでありGC/alias/任意入力の網羅的証明ではない。
- 公開値はvector版50回中央値218.832ms/p95237.897ms、385,531,520bytes/13,945allocations。元Python CPU3015.735ms比13.78倍。RMS218.046msの追加速度差は未証明。
- 残件: [#4 workspace](https://github.com/AtelierArith/JeffClient.jl/issues/4)、[#5 MLP配置と並列化](https://github.com/AtelierArith/JeffClient.jl/issues/5)、[#6 vector/RMS検証](https://github.com/AtelierArith/JeffClient.jl/issues/6)。owned MLP先行up*gateの極端値overflowは既定採用前に評価する。

## CPU vector gating overflow guard trial

- commit abc3750後の継続調査でgate=[-100,-90,-10,10], up=floatmax(Float32)をJuliaで実行。元のSiLU*upは[-0,-0,-1.5448093e35,Inf]、先行up*gate版は[NaN,NaN,-Inf,Inf]となり問題を再現した。
- 所有配列を書き換える前に有限入力の積overflowを走査し、該当すれば元のscalar式へinvokeでfallbackする試作をextensionへ追加。同じ入力で元の結果への一致を実行確認。追加走査の全forward性能は未確定であり未commit。
- real0.8B parcel、vector/Accelerate/trim、50回をcpu-vector-overflow-guard.jsonへ計測中。実行handle93463を継続して確認し、219ms基準と比較する。単一極端値の確認だけで任意入力の数値安全性が解決したとは扱わない。

- handle93463はexit0。50回中央値222.027ms/p95238.991ms/385,531,520bytes/13,945allocations、maxerror1.2398e-5。基準218.832msに対し約1.5%増、分布は重なるが保護走査コストを否定できない。heap増加なし。採用判断にはrepeatと代替式比較が必要。

## CPU MLP packing full-forward trial

- 前goal turnはoverflow再現・保護試作・50回計測でprogress。今回はnative_mlp_weightsのCPU専用packing試作を実装し実際の0.8Bで比較した。gate/upをhcatで結合し、transpose(x)*packedでtoken-major投影、gate半分copy・up半分view、down*transpose(gate)を計算。
- vector/Accelerate/trimとoverflow guard有効、parcel active101、50回: median219.006ms/p95243.861ms、418,707,072heap bytes/13,945allocations、maxerror1.2398e-5、load0.965s（cpu-packed-mlp-trial.json）。guard有効の非packing222.027ms/p95238.991ms、385,531,520bytesと比べ速度改善は未証明、約33.2MB追加heapとtail増。packing試作は除去した。フラグは実行コマンドで1を指定したがJSONにはpacking設定のfieldなし。
- 既存ProfileはAccelerate GEMMを主要サンプルとして示す。単純なgate/up結合だけで大きく改善すると推測しない。次はコピーを増やさない投影mul!とforward workspaceを検討する。overflow guardは未commitのまま残る。

## CPU forward-local MLP workspace trial

- JEFF_CPU_MLP_WORKSPACE=1（既定無効）でgate/up投影のMatrixをforward内に所有し、全層のmul!で再利用する試作。最終層は最後のtokenだけのため別1列bufferを所有。層のMLP widthが異なれば通常経路にfallback。永続cacheやtask共有は使わず、入力とresidualを上書きしない。
- 実0.8B parcel/vector/Accelerate/trim/overflow guard有効、50回median222.239ms/p95224.234ms、321,368,336heap bytes/13,815allocations、maxerror1.2398e-5（cpu-mlp-workspace-trial.json）。guard有効の直前222.027ms/385,531,520bytes/13,945件と比べ速度改善は未証明だが約64.2MB/16.6%のheap削減。
- このtrialではworkspace設定はコマンドで1を指定し、後からbenchmark JSONにcpu_mlp_workspace_enabledを追加。全ケース・GC/所有・異なるshapeの検証は未完了。JET/Profileをprofile-cpu-mlp-workspace.logに起動、handleは本turnの実行結果を参照して継続確認する。

- 前goal turnはMLP workspace実装/50回計測/Profile起動でprogress。handle27367はexit0、JET No errors detected、warm226.745ms/321,499,408bytes/GC0s。Thread1 886 snapshots、APL_sgemm_QRに468サンプル（inclusiveで重複しうるため他行へ加算しない）。
- 同backendで15ケースと[1,15,2,1]再訪、各2回（2回目GC.gc(true)後）、logit guard/finite/input不変確認は全て通過、maxerror3.3974648e-5。ただし一時検証スクリプトがfinallyでNativeBackendに存在しないcloseを呼びexit1。推論の失敗ではないがclean exitを得るためclose除去してrepeat起動（cpu-mlp-workspace-validation-repeat.log）。同じhandleを次turnで確認。

## CPU workspace Cthulhu type audit

- ユーザー指定のCthulhuを実行。Julia1.13.1/Cthulhu3.0.2/TypedSyntax1.5.4、real0.8B parcel型、MLP workspace/vector/Accelerate有効。tools/inspect_native_cpu_types.jlを追加しtyped/source/対話descendを再現可能にした。
- 対話descentでnative_mlp→3引数mul!→5引数mul!→_mul!を辿った。配列型、transpose wrapper、BLAS flag、戻り値は具体型/定数。トップメニューの未使用mul!戻り値::Anyだけで型不安定と判断しない。typed IRではその値を束縛せず、降りた先の戻り値はMatrix{Float32}。
- 5対象（hidden_forward/workspace/layer/MLP/gate）のtyped IRはcpu-workspace-type-ir.log。hidden/layer/MLP/gateのBodyはMatrix{Float32}。workspaceのみUnion{Nothing,具体NamedTuple}で設定fallbackの小Union、利用branchで絞る。JET core+extension No errors detectedと整合。対象以外の任意型・全入力まで型安定を証明したとは扱わない。
- MLP workspace数値/所有検証repeat handle82653はexit0、15ケースと[1,15,2,1]再訪、各2回/GC後/input保持を通過、maxerror3.3974648e-5。永続workspaceではなくforward-localである点も含め記録。

- gateもCthulhu対話表示でextension methodのFloat32演算/Matrix戻り値を確認。TypedSyntax sourceはCore.Const(ENV)を環境変数内容まで展開するため、診断ツールのsource/descendでは子プロセスのENVを必要な設定に絞ってから表示する。型確認のdefaultはtyped IR。
- GEMMのサンプルが多いことだけでBLAS内部packingが原因とは断定しない。Float32を維持した次の候補はロード時の重みmaterialization/配置比較であり、ロード時メモリと全forward時間を測る。

## CPU weight layout trials

- safetensors由来ReshapedArray/ReinterpretArrayの行列をロード時にMatrixへmaterializeする試作はreal0.8B parcel/workspace/vector/Accelerate/trim/guard、50回median222.695ms/p95243.434ms、321,368,144bytes/13,814allocations、maxerror1.2398e-5、load0.984s（cpu-materialized-weights-trial.json）。元workspace222.239ms/load0.782sより速度改善なし。物理配置は同じなのでBLAS packingの改善を証明しない。試作除去。
- 次にMLPだけpermutedimsをロード時に行いtranspose wrapperで論理weight形状を維持、native_linearのtransposeがunwrapされBLAS N指定となるJEFF_CPU_TRANSPOSE_MLP=1試作を起動。cpu-transposed-mlp-trial.json、実行handleは本turn出力を参照。まだ検証/採用未確定。

- transposeMLP trial handle62284はexit0、50回median222.213ms/p95264.525ms/max363.308ms、321,368,336bytes/13,815allocations、maxerror9.059906e-6、load1.209s。元workspace222.239msと中央値同等、tailとload悪化。transpose配置だけではGEMM改善を証明できず、試作除去。ロード時Matrix化とtransposeMLPのどちらも既定にしない。次は形状別GEMM単独測定とhead/batch分割の費用を定量化する。

## CPU projection and layer-weight sweep benchmarks

- tools/benchmark_cpu_projections.jlを追加。real0.8B/Float32/Accelerate/Apple M4/parcel active101、最初のfull/delta層の実activationsを使ってmul!を各50回測定、出力はpreallocate。モデル全forwardやactivationを含まない。cpu-projections-101.jsonとcpu-projections-101-sweeps.json。
- 単独GEMM median: delta.qkv1.167ms(1024x6144)、full.q0.879ms(1024x4096)、MLP gate/up各0.809〜0.824ms(1024x3584)、down0.780ms(3584x1024)、delta.z0.439ms、full.k/v0.111ms、delta.a/b0.014ms。全て0Julia heap bytes/0allocations。BLAS内部native allocationsはこの指標では測れない。
- 24層の異なるMLP weightsを同一zero input101tokensで順に投影するsweepを各50回測定。同一weight連続のcache有利なmicrobenchを補うが、全forwardのキャッシュ状態を再現する証明ではない。gate21.007ms/24（平均0.8753ms）、up20.866ms/24（0.8694ms）、down19.886ms/24（0.8286ms）。全て0heap/0alloc。実forwardの最終層は1tokenだがsweepは全層101tokensなのでそのまま足してforward時間と比較しない。
- 前turnは重みmaterialization/transposeMLP実測と不採用でprogress、今回も測定ツール追加と投影別/異なる重みsweepでprogress。GEMMラッパーの型やJulia heap割当をこれ以上削るより、DeltaNet headの並列化（scratch所有分離とBLAS oversubscription検証）を次に評価する。

## CPU DeltaNet head parallel trial

- cpu_delta_heads!を抽出し、複数headをworkerごとの範囲で処理する。state/full/tail scratchはhelper callが所有、headの出力領域は重ならず、q/k/v/z/weightsはread-only。threadid依存のcacheは使わず、@syncで全worker完了後にout projectionへ進む。JEFF_CPU_PARALLEL_HEADS=1かつdefault worker>1でのみ有効、既定0。
- real0.8B parcel/vector/Accelerate/trim/MLP workspace/overflow guard、Julia --threads=4、50回median190.265ms/p95198.349ms、348,846,704heap bytes/17,775allocations、maxerror1.2398e-5（cpu-parallel-heads-4.json）。実行handle11760 exit0。最初のJSONにはparallel設定/worker数fieldがまだなく、後からbenchmarkへ追加。Accelerate auto10、LBT8。
- 直前workspace222.239msより約14.4%短いが、同worker数の逐次baseline/8worker/Accelerate single/全ケースをまだ比較していない。同--threads4でparallel0の50回をcpu-serial-heads-4.jsonに起動。両helper経路のJET/所有/数値検証も未完了。

- --threads4/parallel0 baseline handle72877はexit0、50回median225.306ms/p95240.565ms、321,368,912bytes/13,833allocations/maxerror1.2398e-5。parallel4の190.265msは同worker設定の逐次版から中央値約15.6%短い。次に--threads8/parallel1の50回をcpu-parallel-heads-8.jsonへ起動し同handleで確認する。

- head並列8workerのhandle84910はexit0。50回median185.748ms/p95201.600ms、385,451,504heap bytes/22,671allocations、maxerror1.2398e-5（cpu-parallel-heads-8.json）。4worker190.265msとの差は約2.4%、heap/件数は増える。autoAccelerate10とJulia8の競合を評価するためAccelerate singleを同8worker/50回で比較開始（cpu-parallel-heads-8-accelerate-single.json）。

- Accelerate single+8worker handle1555はexit0、50回median188.755ms/p95194.849ms、385,451,504bytes/22,671allocations/maxerror1.2398e-5。auto185.748msより中央値改善なし。singleを最速と主張せずautoを維持。
- tools/validate_native_cpu.jlを追加し、一時スクリプトの検証を再現可能にした。15ケースと[1,15,2,1]再訪、各2回/2回目GC、入力保持と以前返したscore保持を確認する。parallel8設定で起動し、成功後に同設定のProfile/JETを逐次実行する（cpu-parallel-heads-validation.log / profile-cpu-parallel-heads.log）。同じsessionを次turnで確認し、計測を並列に起動しない。

## Parallel CPU commit checkpoint

- 検証+profile session99822はexit0。15ケースと再訪、各2回/GC/input保持/以前返したscores保持は通過、maxerror3.3974648e-5。JET core+extension No errors detected。warm186.076ms/385,451,840bytes/GC0.853ms。worker snapshotsではhead処理が複数workerへ分散。
- ユーザーの現状commit/push指示に従い、検証済みMLP workspace・head並列化・overflow guardと診断ツールをまとめる。scratch層間再利用の案は未実装。極限最適化完了とは扱わない。READMEのoptional高速CPU設定は8worker+各flag、docsに4/8worker時間・heapのtradeoffを記録。

## Intel portable CPU vector math and projection trial (2026-10-01)

- Intel i9-9900Kでの現行比較基準は495.254ms（Apple Accelerate、Float32、Julia/BLAS8、parallel heads/MLP workspace/vector math/trim有効）。2倍目標は247.627ms以下。既存recurrent版402.391ms（50回）も未達。上のApple M4測定と混同しない。
- LLVM/native inspectionではrecurrent state loopは既に8-wide Float32 SIMD、通常SiLUはscalar exp呼出し。inline/inboundsを追加するだけで指数関数のベクトル化を保証しない。tools/inspect_cpu_assembly.jl、artifacts/cpu-tuning/assembly-openblas.log。
- MLPロード時物理transposeのIntel再試験も不採用。20回control401.366ms/trial406.642ms、load2.165→3.292s、peak RSS5.203→5.955GB。実装と専用テストを除去。pack-mlp-{control,trial}20.json。
- optional LoopVectorization extensionを追加。JEFF_CPU_PORTABLE_VECTOR_MATH=1でのみ所有Matrix{Float32}のSiLU/MLP gateを@turboで融合。有限gate範囲(-20,80)、abs(gate)>1e-12、有限upでabs(up)が(1e-12,1e12)の範囲を全配列走査し、範囲外/alias/不正形状は変更前にfalseを返してscalar fallback。極端値やsigned zeroにfast-mathを適用しない。既定0。通常値でも丸めの一致ではなく許容誤差の確認が必要。
- verify_cpu_vector_math.jlは126 activation/guard/ownership検証と294 native検証にexit0（8 threads）。NaN/Inf/overflow/underflow/±0、空/shape/alias、独立tiny PyTorch参照、GC後/同backend同時forward、chunk/worker/mask/length検証。任意checkpointの分類精度保証ではない。
- 3584×101 SiLU micro 50回scalar2334.123us/vector191.692us、6144×101は3995.916us/373.822us（各32heap bytes）。入力はbounded合成値、copyはsetupで時間外。モデル全体の12倍化とは扱わない。
- JEFF_CPU_PARALLEL_PROJECTIONS=1はJulia workerごとに出力行を分割、single-threaded BLASのみ利用。最低256 output rows/1e6 multiply-elements、8worker時も小行列は逐次fallback。入力/重みはreadonly、出力aliasは一時結果へ逃がす。forward中にBLAS global設定を変更しない。単独gate/down GEMMはOpenBLAS1の7.293/7.538ms→8Juliaworkers1.250/1.226ms、task heap3680bytes（parallel-gemm-openblas.log）。Linux/macOS対応ライブラリだがLinux性能は未測定。
- real0.8B同parcel input B1/L256/active101、recurrent/head/MLP/trim/delta-workspace/final-query有効、OpenBLAS1＋Julia8 projection、20回scalar475.591ms/p95509.220msとportable vector365.620ms/p95391.088ms（parallel-scalar-repeat.json/parallel-portable-vector.json）。最大参照誤差1.05e-5、vector292332400heap bytes/12894alloc、model retained3.010637944GB/peak RSS5.346942976GB。RSSはロード・コンパイルを含むプロセス高水位、推論のscratchだけではない。
- 同vector有効/同機能でprojection分割なしOpenBLAS8は513.775ms/p95537.363ms（openblas8-portable-vector.json）。parallel vectorは約1.405倍このcontrolより速い。元495.254ms比約1.355倍で、要求2倍には未達。
- ユーザー指定でFlux/NNlibのmaster実装を読んだ。FluxはNNlibをreexportし、swishはinline x*sigmoid_fast(x)。sigmoid_fastは局所fastmath exp(-abs(x))、符号でinvまたはt/(1+t)、x>40/<-80は1/0へ飽和する。参考 https://github.com/FluxML/NNlib.jl/blob/master/src/activations.jl 。負の極端値で元式と異なるためそのまま置換しない。compare_cpu_vector_math.jlのnnlib_style実験は式を参考にしたもの、実NNlib実行の測定と混同しない。

- portable vector/head/projection設定のProfile/JETはexit0、core＋LoopVectorization extensionにNo errors detected。warm342.329ms/292332400heap bytes/GC6.638ms。workerではOpenBLAS sgemm kernel/input packingが主、時間の確定値は別BenchmarkTools trialを用いる。Allocsは5%サンプリングでforward投影配列が残る（profile-portable-vector.log）。
- NNlib-style式のSiLU micro再試験は3584×101 scalar2329.556us/NNlib-style2149.341us/vector190.114us、6144×101は3990.844/3759.258/376.539us。式だけの変更は約6〜8%短縮に留まり、指数関数の実ベクトル化が大きい。NNlib本体を導入・実行した結果ではない。
- full test suiteはexit0（portable-checkpoint-tests.log）。別プロセスのvector/parallel20回repeatは382.683ms/p95462.790ms、同heap12894件/292332400bytes、最大参照誤差1.05e-5（parallel-portable-vector-repeat.json）。中央値の改善は再現するがtailは不安定、普遍的高速化や2倍達成は主張しない。極端値alias guardはBase.mightaliasで同じMatrix object以外の共有storageも除外する。
- 最終alias guard版の1Julia worker fallback検証もexit0、126 activation＋294 native checks（vector-math-single-worker.log）。8worker版と1worker版の両方を確認。設定は既定offのまま、改善をcheckpointとしてcommit/pushし、目標はactiveを維持する。

## CPU forward thread policy and full-attention head trial

- 前goal turnはportable vector/parallel projectionの実装・参照検証・20回比較・repeat・commit/pushでprogress。現行d864d57を処理別に再測定した。time_cpu_phases.jlがAccelerateを無条件importしgate/upに直接mul!していたため、optional importとcpu_projection!へ修正しOpenBLAS1設定を反映した。
- 同parcel/Julia8/OpenBLAS1/portable vector/parallel projection/recurrent設定、5warm passesのinstrumented totals338.211〜353.485ms。phase medians: pre RMS1.399、attention211.783（Delta169.017/full42.630）、post RMS3.223、gate/up80.762、activation7.405、down34.360、residual0.799ms。別phase中央値を合計してforward中央値とはしない（phases-portable-vector.jsonl）。
- BLAS.get_num_threads単独50samples median58.596us対task_local_storage lookup6.853ns。BLAS thread判定を小行列threshold後に移し、JEFF_CPU_PROJECTION_THREAD_SCOPE=1でBLAS thread snapshotをforwardのtask-localなimmutable値として保持する候補。共有global cache/threadid cacheなし、Baseが例外時にも旧TLS値を復元。forward中のBLAS設定変更は禁止条件を維持する。
- scope候補単独20回は356.961ms/p95384.571ms/292332464heap bytes/12896allocs（projection-thread-scope.json）。直前365.620〜382.683msより短いが独立repeat前なのでscope単独の速度改善確定値とはしない。
- 通常full-attention headを分離するnative_full_heads! hookを追加。CPU opt JEFF_CPU_PARALLEL_FULL_HEADS=1は16tokens以上、複数Julia workers、single BLASでのみ有効。Q/K/V/gate/mask readonly、workerのout head row域は分離、各workerのscores/probabilities/valuesはprivate、@sync後にout projection。CPU viewでQ/K/Vのheadコピーを避け、generic/GPU/flag0経路は元の計算を維持する。既定0、最終queryだけのattentionは変更しない。
- 8worker検証は126 activation＋335 native checksにexit0（full-heads-validation.log）。scopeの成功/例外/nested TLS旧値/同時task復元9checks、full attentionのn1/9/65/129×maskholes、同時呼び出し/GC/input保持32checksを追加した。実モデルの全forward速度とJETはまだ検証中。
- full-heads-parallel.jsonの20回は319.038ms/p95345.544ms/271369904heap bytes/12671allocs。独立30回repeatは318.656ms/p95333.545ms、同heap/alloc、最大参照誤差1.05e-5、model retained3.010637944GB、peak RSS5.312884736GB（full-heads-parallel-repeat.json）。直前365.620〜382.683msから改善を再現し、元495.254ms比約1.554倍。2倍目標247.627msは未達。
- full test suiteはexit0（full-heads-checkpoint-tests.log）。real modelのcore+portable extension JETはNo errors detected、warm300.779ms/271369904bytes/GC5.474ms、5forward時間Profile＋5%Allocsはprofile-full-heads.logに保存。scope単独でなく今回のhead並列/コピー削減を組み合わせた改善として報告する。
- 1workerの全新flag fallbackも126 activation＋335 native checksにexit0（full-heads-single-worker.log）。JuliaFormatter整形済み。今回の改善をcommit/pushし、目標activeを維持する。次候補はDeltaNetの残るprojection/state/gating段階別計測とprojection融合。model精度Float32を維持し、過去不採用MLP transposeを根拠なく再導入しない。

## DeltaNet normalization and blockwise vector activation trial

- 前turnはfull attention/head-copy削減、task-local BLAS policy、全test/JET/Profile/独立repeat/commit-pushでprogress。現行c5af264から段階別のdiagnostic tools/time_cpu_delta_stages.jlを追加した。real modelの実activationsを層順に流し、毎passのlogitsを保存済み独立参照と照合する。instrumented phase測定でありBenchmarkTools中央値の代替ではない。
- 最初の5passesのstage合計中央値: mask0.747ms/qkv56.497/conv+SiLU27.229/QK prepare15.462/z-beta-decay22.184/state-RMS-gate18.925/out projection18.203。ステージを分けたdomain調査はconv12.151/SiLU13.630/QK15.288ms。domain走査の時間はphase外だがpass全体には含まれる。
- domain調査の最初の試作はconvolution出力similarを0初期化せず参照guardでexit1（delta-stages-domain.jsonl）。これは計測toolの不具合でproductionコードは変更していない。そのデータは採用せず、zerosを使う修正版5passesがexit0/参照guard通過（delta-stages-domain-corrected.jsonl）。
- 第1Delta層のSiLU前値はmin-20.083/max10.065、-20以下が1要素、abs<=1e-12が1414要素。第2層はmin-26.428/max7.896、-20以下4要素。それ以外の16Delta層は既存eligible範囲内。少数の値で62万要素のwhole-array SIMDがfallbackする費用を確認した。
- JEFF_CPU_VECTOR_MATH_BLOCKS=1をportable SiLU callbackに追加。whole-array eligibleなら従来通り、範囲外を含むと256element spanごとに同じfinite/domain guardを評価。safe spanだけ@turbo、他spanは元のnative_silu式をscalarで評価する。NaN/Inf/±0/underflow/極端値にfastmathを適用しない。MLP gateのwhole-array条件/alias/input保持は変えない。既定0、portable flag1＋extension import必須。
- JEFF_CPU_DELTA_NORM_LOOP=1は所有Q/Kの列優先SIMD平方和とdivisionに置換し中間reduction/broadcast配列を減らす。fastmathや低精度化なし。summation orderは変わり得るため参照検証が必要。既定0。helperのnormal/empty/extreme value102checksが通過。
- block activationの18496 element checks、従来activation126checks、native437checksが8workersでexit0（delta-norm-block-validation.log）。chunk boundaries1/255/256/257/513/1024、NaN/Inf/floatmax/±0/subnormalを検証。
- 同model/input/Julia8/OpenBLAS1/currentflags＋両新flagの20回は302.631ms/p95324.113ms/271128624heap bytes/12601allocs、max logit error1.1444e-5（delta-norm-block.json）。直前318.656msより短いが独立repeat/JET/fullsuite検証は進行中。元495.254msの2倍基準247.627msは未達。
- 独立30回repeatは304.157ms/p95328.888ms/同heapとalloc/maxerror。model retained3.010637944GB、peak RSS5.344727040GB（delta-norm-block-repeat.json）。元495.254ms比約1.628倍、直前318.656msから約4.6%短縮。JET core+portable extension No errors detected、warm286.468ms/271128624bytes/GC5.628ms（profile-delta-norm-block.log）。full test suiteはdelta-norm-block-checkpoint-tests.logでexit0。
- 最終版は1workerでもactivation126＋block18496＋native437checksがexit0（delta-norm-block-single-worker.log）。JuliaFormatter整形済み。実測改善をcheckpointとしてcommit/pushしgoalはactive維持。次は大きな費用を占める投影の融合/worker分割とメモリtrafficを検討する。

## Projection worker and fusion follow-up

- 前turnはnorm/block実装、全test/JET/Profile/独立repeat/commit-pushでprogress。現在4dd4a40からworker数と融合を実測した。各試験は参照guard通過、既存最速の設定は変更していない。
- 16Julia threads/OpenBLAS1のGEMM microでraw gate 8/12/16 workers=1.332/1.651/1.345ms、down1.368/1.572/1.237ms、qkv2.889/2.881/2.927ms（projection-workers-16-micro.log）。特に12/16への増加を一般的高速化とは扱わない。
- 全forwardはDelta workerを8に固定し、Julia12/16（projectionは12/16、full attentionはcfg.headsまで）各20回を比較。12workers328.823ms/p95348.922ms/271414832bytes/15770allocs、16workers305.098ms/p95323.004ms/271654832bytes/18770allocs。既存8workers304.157ms/12601allocsに対して改善なし（projection-workers{12,16}.json）。worker設定は8を維持する。
- tools/compare_cpu_projection_fusion.jlで同じ入力を共有するgate/upまたはqkv/zをロード後hcatして1回投影するmicroを追加。combined結果を別々のMatrixへ戻すcopy込みは101tokens gate/up2.867→3.422ms、qkv/z3.893→4.418msと遅いためproductionへ導入しない。
- copyなしの直接combined outputは101tokens gate/up2.867→2.825ms、qkv/z3.893→3.647ms、256tokens5.650→5.592ms/7.995→7.680ms。次の候補だが新しいview/ownership/loader契約とwhole-forwardの確認が必要。task heap7424→3456bytesだけで全forward高速化を主張しない。microはwarm/synthetic sin input、実モデル全層cache条件と同じとは限らない（projection-fusion-direct-micro.log）。
- BLISBLASの公式実装とblis_jll配布定義を確認した。macOS/Linux（x86_64/aarch64）の配布あり、LBTのBLAS forwardingで使用可能。MKLではない。https://github.com/JuliaLinearAlgebra/BLISBLAS.jl および https://github.com/JuliaBinaryWrappers/blis_jll.jl/blob/main/Artifacts.toml 。まず隔離temp Julia環境へ導入して比較し、root/toolsの依存はまだ変更しない。
- 隔離環境は/private/tmp/jeff-blis-htwsuz（BLISBLAS0.2.0、blis/blis32_jll2.0.0+2、LAPACK3.12.1、LLVMOpenMP23.1.1）。最初のJulia mktempdir既定cleanup環境はプロセス終了時に自動削除されたため、再利用用はcleanup=falseで作った。共有depotのpackage/artifactは削除しない。BLIS/LBT getter両方1threadsとloaded libraryのblisを確認し、プロセス開始時のみforwardingを変更した。
- BLIS single/Julia8のraw gate/down/qkv microは2.294/1.222/4.739ms、同OpenBLAS1は1.332/1.368/2.889ms。downだけ有利だが全体switchには根拠不足。full forward10-call screenは375.005ms（p95415.116ms、10samplesのtailはscreen statistic）、同heap271128624bytes/12601alloc（blis-full-screen.json）、保存済み独立logit guard通過。既存OpenBLAS304.157msから改善なし、依存・デフォルト変更はしない。
- 今回はworker増加、copy込み融合、BLIS全体切替を不採用とする測定結果が得られた。production実装の速度改善はまだないため、その改善を主張するcommit/pushは行わない。診断toolとjournal変更は次checkpointへ引き継ぐ。2倍目標はactiveを維持し、次はforward内の投影workspace/配列生成と、copyなしfusionの契約を検討する。

## Delta projection workspace checkpoint

- JEFF_CPU_DELTA_PROJECTION_WORKSPACE=1でmasked/QKV/convolution/Q/K/Z/beta/decay/head output/out projectionをforward-localに確保し、18 Delta層で再利用する。既定off、実入力101tokensの配列payload9,114,240bytes。convolutionは必ずfill!(mixed,0)後、projectionは上書き。backendに共有cacheを置かず、同時forwardと以前返したscoresの所有を維持する。
- 同real0.8B/Float32/parcel B1/padded256/active101、Julia8/OpenBLAS1、portable vector/block/norm/recurrent/head/full-head/MLP/trim/final-query/projection-scope有効。20samples median290.285ms/p95312.535ms、独立30samples296.796ms/p95321.881ms、113,553,680heap bytes/11,895allocations、maxerror1.1444e-5。直前304.157ms/271,128,624bytesからrepeat中央値約2.4%短縮、heap約58.1%削減。固定基準495.254ms比約1.669倍、247.627ms目標は未達。
- activation126＋block18496＋native597checksが8worker/1worker両方exit0、160checks追加（n1/9/65/129、mask再訪、NaNでscratchを汚して上書き確認、recurrent/parallel組合せ、GC、入力保持）。全test suite exit0、core+portable extension JET No errors detected、Profile/Allocs5%取得。warm299.665ms、同heap、GC0。ログdelta-projection-buffer-{validation,single-worker,checkpoint-tests}.log、profile-delta-projection-buffers.log、delta-projection-buffers{,-repeat}.json。
- 最新Allocsではnative_residual_rms/native_rms/native_rope/native_linearが残る。heap0は目標にしない。既存logitsのfresh score、forward-local workspace、spawn taskには割当が必要。caller-owned logits!/永続scratch/worker方式変更は別の所有契約と検証を要する。
- tools/compare_cpu_projection_kernels.jlは同8worker行分割でOpenBLAS/Octavian serial/直接turboを比較。101tokens gate1.260/1.854/4.136ms、down1.231/1.859/3.873、qkv2.873/3.836/13.215。参照guard通過、Matrix materializationでも逆転なし。不採用。150個の大きなweightのexact zero fractionは全て0で、ゼロ行/列pruningの根拠なし（weight-zero-structure.jsonl）。
- phase toolはprojection workspaceを実際に渡す。手書きDelta stage toolはworkspace path未対応のためflag1を明示的に拒否し、誤った比較を避ける。次候補はRMS/残差融合、MLP down出力再利用、copyなしQKV/Z融合。時間改善は必ず全forwardで再確認する。

## CPU refresh after pull: 5f2d0e9 on Apple M4

- ユーザーのpull後に現状CPU benchmark/docsを再取得。source5f2d0e9、測定時tracked worktree clean。既存未追跡assetsには直前4dd4a40の測定JSONが残っていたため比較snapshotとして保存し、新規JSONはsourcehash付き別名。
- real0.8B pinnedrevision/Float32/parcelB1L256active101/Julia1.13.1/AppleM4、8workers/各30warmforward、process逐次/override除去。
- default: median1503.129ms/p951545.248ms/960896160heap bytes/22016allocs/maxerror1.144409e-05。
- readme-accelerate: median183.340ms/p95189.787ms/385445744heap bytes/22707allocs/maxerror1.239777e-05。
- portable-control: median351.977ms/p95360.786ms/265559088heap bytes/12601allocs/maxerror1.049042e-05。
- portable-projection-workspace: median359.101ms/p95363.926ms/109062416heap bytes/11895allocs/maxerror1.049042e-05。
- accelerate-projection-workspace: median205.757ms/p95213.276ms/126843856heap bytes/4380allocs/maxerror1.049042e-05。
- projectionWorkspaceのportable matched比較でheap58.9%減、M4中央値の速度改善は未確認。Intel結果をM4へ一般化しない。全5runs参照guard通過。
- docs/src/performance.md冒頭に現行source5設定表、前snapshot比較、全reproducecommandsを追加。raw集約docs/src/assets/benchmarks/cpu-2026-10-01-5f2d0e9.json。今回はPython再測定なし。

## CPU RMS / residual fusion probe (not adopted)

- 前turnはFlux/NNlibの公式source読解、workspaceの記録更新でprogress。今回5f2d0e9でDelta projection workspaceと診断toolをcommit/push済み。
- 所有Float32 Arrayのnative_rmsに列毎@simd平方和/scale/outputを追加し、native_residual_rmsでは残差保存と平方和を融合するJEFF_CPU_RMS_LOOP試作を評価。明示fastmathなし、両出力はfresh配列、入力不変、GPU dispatch不変更。
- 81追加checks（幅1/7/128、tokens0/1/9、centered/noncentered、Array3、±0/NaN/Inf/floatmax/subnormal、shape拒否）と既存activation/native/所有テスト、全test suiteがexit0。JET No errors detected。Profile/Allocs取得、warm292.899ms/113,242,240heap bytes/GC0（profile-rms-loop.log）。
- real0.8B parcel/current flags/Julia8/OpenBLAS1、30samples試作296.982ms/p95322.099ms/113,242,240bytes/11,531allocs/maxerror1.2398e-5（rms-loop.json）。直後の同試作flagoff control30samples298.307ms/p95315.610ms/113,556,368bytes/11,979allocs/maxerror1.1444e-5（rms-loop-control.json）。元5f2d0e9 repeat296.796ms。0.4%差は十分な全体速度改善の証拠と扱わず、不採用。flagoffにもENV判定分の割当が加わるためproduction methods/追加tests/benchmark flagを除去し、worktreeを5f2d0e9のsourceに戻した。
- 試作の再現patchはignored artifacts/cpu-tuning/rms-loop-trial.patch、他ログrms-loop-validation.log/rms-loop-checkpoint-tests.log。割当0のための変更は追わず、次はworkspace後のphaseを再測定し、投影/MLP down/コピーなし融合を優先する。2倍目標は未達、active。
- 5f2d0e9の実workspace経路でtime_cpu_phasesを5warm passes実行、exit0/各pass参照guard通過（phases-delta-projection-workspace.jsonl）。instrumented total287.178〜298.388ms、stage合計の中央値preRMS1.420958/attention163.613085/postRMS3.310502/gate-up75.454604/activation7.604631/down32.651782/residual0.834017ms。Delta attention135.716339/full27.690589ms。異なるphase中央値を足してforward中央値とは扱わない。RMSよりGEMMが大きい。次はdown projectionをBLAS beta=1で所有residualに直接累積し、down中間配列と別add走査を省く候補を検証する。

## CPU MLP down / residual accumulation checkpoint

- JEFF_CPU_MLP_RESIDUAL_FUSION=1（既定off）で層が所有するresidualへdown projectionをBLAS beta=1で直接累積し、down中間配列と別add走査を省く。parallel projectionはworkerごとにdisjoint行領域、aliasは先に独立productを作ってから累積する。既存beta=0経路は従来の3引数mul!を保持。
- 同Intel i9-9900K/Float32/real0.8B/parcel B1L256active101/Julia8/OpenBLAS1/current flags、30samples294.338ms/p95325.456ms、独立30samples293.934ms/p95313.750ms、103,750,032heap bytes/11,823allocs/maxerror1.2398e-5。matched flagoff30samples297.079ms/p95322.429ms/113,554,448bytes/11,919allocs。速度差約1%、heap約9.8MB減、普遍的速度改善とは主張しない。固定495.254ms基準比約1.685倍、247.627ms目標未達。
- 追加46checksと既存activation126/block18496/native597checksが8worker/1worker両方exit0。projectionのinput/weight alias、parallel/serial、tokens1/9/65、MLP workspaceなし/あり、入力保持を検証。全test suite exit0、JET core+portable extension No errors detected、Profile/Allocs5%取得。warm285.602ms/同heap/GC0。
- mlp-residual-fusion{,-repeat,-control}.json、mlp-residual-fusion-{validation,single-worker,checkpoint-tests}.log、profile-mlp-residual-fusion.log。phase toolもbeta=1の経路を測定するよう更新。リモート9ded04aのM4 docs/journalと手元の記録は両方保持してff pull済み、実装差はなく計測条件を混同しない。
