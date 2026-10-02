# パフォーマンスの知見

## 2026-10-01: 自動スレッド CPU 比較を標準 driver の既定へ変更

- ユーザー指定で `tools/mac-M-series.sh` の `--include-auto-cpu` とAUTO_CPU分岐を廃止。PyTorch8とJulia8/Accelerate自動の比較はCPU1対1・GPU比較とともに毎回実行する。スレッド予算が異なる比較としての区別、既定30samples×2fresh processes、逐次実行は維持。追加実験だったJulia1/Accelerate自動はこのフラグの旧機能に含まれておらず、driverへ追加していない。
- README/performance/profilingとhelpを更新。過去の測定記録中の旧フラグは当時の再現コマンドとして保持する。今後はフラグなしで実行し、旧フラグを渡すとunknown option/exit2となる。
- `bash -n`、help表示・旧フラグ拒否・diff checkを確認。実機でフラグなし `--samples 3 --repeats 1 --checkpoint <pinned Scratch path>` を実行しexit0、全6groupsのJSONとsummary guardを確認。ignored `artifacts/benchmarks/mac-M-series-20261001T083530Z/`。3samplesは変更の動作確認であり新しい性能ベースラインとして扱わない。

## 2026-10-01: M2 Max / CPU thread budget 比較

- ユーザー依頼で `./tools/mac-M-series.sh --include-auto-cpu --checkpoint <pinned Scratch path>` を実行しexit0。追加依頼により、終了後に `julia --threads=1 --startup-file=no --project=tools tools/benchmark_inference.jl <checkpoint> cpu examples/data/parcel_reference.json 1 30 <output> --python-reference` も2fresh process逐次実行/exit0。source7867e7c、開始時tracked worktree clean。
- 同M2 Max/12cores/96GiB/macOS26.5.2/Julia1.13.1/PyTorch2.14.0/Accelerate0.7.0、AC接続。F32/real0.8B/parcelB1L256active101、全256列、readout/CPU score返却込み、load/compile/tokenization除外、30samples×2fresh processes、host非隔離/affinityなし。新source最適化は行っていない。
- CPU PyTorch1 median583.570/588.584ms（p95601.821/613.187ms）、PyTorch8 median3904.427/3977.236ms（p954062.408/12103.612ms、run2max14881.207ms）。このhost/inputでは8が遅い。run2の外れ値原因は未調査で、PyTorch一般のthread scalingと扱わない。未調整runtime情報でもPyTorch intra-op8/inter-op12、OMP_NUM_THREADS未設定。「スレッド無指定＝1」ではない。
- Julia1/Accelerate1 median659.679/781.668ms（p95713.166/931.495ms）。Julia8/Accelerate自動 median402.669/411.071ms（p95425.608/998.798ms）。追加Julia1/Accelerate自動 median552.929/550.296ms（p95834.346/971.668ms）。この条件ではJulia8側が速く、Julia1＋BLAS並列だけが最善とは言えない。両自動設定でLBT報告8/Accelerate報告12、各演算の実稼働thread数は未計測。PyTorch8とequal-thread-budget比較ではない。
- Julia8自動のwarm heap847,294,512bytes/28,367allocations、Julia1自動803,619,600bytes/22,337allocations。両run maxerror6.67572e-6で独立参照guard通過。追加groupもshape/mask/full compute length/sample/profile/BLAS/Accelerate設定が元Julia8と一致することをassertした。これはJulia heapでありnative BLAS内部allocationの計測ではない。
- 再測定GPU MPS medians264.776/263.450ms、Metal93.773/94.096msも完了。全7groups×2runs×30samplesを確認。raw JSON/log/runtime/hash/hardware/summaryはignored `artifacts/benchmarks/mac-M-series-20261001T081746Z/`。追加Julia1自動は `cpu-one-auto-julia-run{1,2}.{json,log}`、guard/summary追記scriptは `add_cpu_one_auto.jl`。既存driver/reportのsourceは変更せず、summaryJSON/Markdownへ追加groupを保存した。

## 2026-10-01: M2 Max の MPSCommandBuffer 再利用を修正

- `ext/JeffClientMetalExt.jl` のtask-local MPS wrapper cacheを削除し、Layaと同様にencodeごとに `MPS.MPSCommandBuffer(Metal.ensure_cmdbuf!(queue))` を作る。Metalのcommand batchingと、GPU完了まで配列/tensor-data/commandを `record_operation!` で保持する処理は維持。不要なreadback/例外cleanupのcache clearと古いcache identity検証も削除。
- `tools/verify_metal_command_buffers.jl` は実際の `native_matmul` とbroadcastを16回queueへ投入し、系列長256→9→256、処理未完了時GC、完了後GC、入力/weight保持、異なる値の過去出力保持を検証。修正前同assertion/exit134、修正後exit0。既存primitive verifierからも呼ぶ。古いCPU padding-disabled前提はtask-local `with_cpu_settings(:trim_padding=>false)` で明示するよう修正。
- `Pkg.test()`、全 `verify_metal_primitives.jl`、独立PyTorch参照15cases×2passesのfull-sequence adapter実モデル照合がexit0。padding/length1–512/interior mask/batch1–3/GCを含み、max logit error3.445754e-5。Metal scalar indexingは無効。変更したmatmul helperの `@code_warntype` はMtlMatrix{Float32,PrivateStorage}、JETはNo errors detected。preallocated64×64/64×9のsubmission＋同期でJulia heap1072bytes（GPU buffer割当数とは別）。JuliaFormatter適用、読み取り専用reviewの重大指摘なし。
- 標準 `tools/mac-M-series.sh --checkpoint <pinned Scratch path>` が既定30samples×2processでexit0、summary guard通過。Apple M2 Max/12cores/96GiB/macOS26.5.2/Julia1.13.1/Metal1.11.1/PyTorch2.14.0、F32、parcelB1/L256/active101、全256token/readout/CPU score返却/GPU upload＋完了込み。ロード/compile/tokenization除外、AC接続、host非隔離/affinityなし。
- CPU1thread PyTorch medians568.902/567.711ms、Julia/Accelerate1 medians638.369/639.512ms。GPU MPSF32 medians264.487/264.122ms、Metalfull-sequence93.738/93.761ms（p9594.267/101.296ms）。修正前Metalはabortしたため、cache削除の速度差は測定できない。旧M4結果と直接比較しない。
- Metal warm Julia heap373,520/466,496bytes、8,692/11,821allocations。GPU buffer割当回数の計測ではない。両processのpool保持はGC後26,626,867,200bytes/trim後20,870,184,960bytesで、warm Julia heap・model保持量とは区別する。
- benchmark JSON/log/runtime/hashes/summaryはignored `artifacts/benchmarks/mac-M-series-20261001T080357Z/`。regression before/after、primitives、Pkg.test、独立参照JSON/生成log/実モデル照合、型/heap診断、source.patch/hashes/新規source snapshotsはignored `artifacts/metal-fix/`。計測sourceはbase2268210に未commitの本修正を適用したもの。旧MWEは意図的に不正な再利用を再現するコードとして保存している。

## 2026-10-01: Apple M2 Max / mac-M-series.sh 実行失敗

- ユーザー依頼で標準driverを実行。source `2268210df81cc5afccfc805b2eef5a68ea912dde`、開始時tracked worktree clean。Apple M2 Max / 12 cores / 96 GiB / Darwin25.5.0 / Julia1.13.1、AC接続。古いGeneral registryを更新して依存解決し、`uv sync --project extern/jeff --no-default-groups` と pinned checkpoint取得を実施。
- `./tools/mac-M-series.sh --checkpoint /Users/atelierarith/.julia/scratchspaces/99eee0ee-b0c7-48fb-9455-442e76ba5e88/hub/models--mstrasser--Jeff-Qwen3.5-0.8B/snapshots/0f212b3e72acb4dde3f7da61e925d6ab7f819990`。既定30samples×2process、全256token/F32/readout/CPU score返却、GPU upload込み。
- run1のみCPU PyTorch median574.453ms/p95586.953ms、CPU Julia median638.918ms/p95689.267ms、MPS median264.052ms/p95264.463msのJSONを生成。Metal run1が `_status < MTLCommandBufferStatusCommitted` / `-[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]` line323 のnative assertionでsignal6、driver exit1。同じMetal commandの単独再実行も同assertion/exit134。原因は未特定。2回目は未実施、summary未生成のため比較全体の有効な結果として採用しない。
- partial JSON/runtime/hash/hardware/元の失敗logと `gpu-metal-retry.log` はignored `artifacts/benchmarks/mac-M-series-20261001T074311Z/`。準備logは `artifacts/benchmarks/mac-device-setup/`。hostは非隔離で別LanguageServer processのCPU使用も観測。推論sourceは変更していない。

### MPSCommandBuffer 再利用の MWE

- `tools/mwe_metal_mps_command_buffer.jl` はMetalのみimportし、4096×4096と4096×256のFloat32行列積MPSGraphと `c .+= 1f0` を交互に16回投入する。JeffClient/Python/実モデル不要。JuliaFormatter適用済み。
- macOSの製品versionは `sw_vers` で26.5.2/build25F84（Darwin25.5.0）。Metal1.11.1 / Julia1.13.1 / M2 Max。`julia --startup-file=no --project=tools tools/mwe_metal_mps_command_buffer.jl` は6回目のencode後に元queue bufferがCommitted、MPS wrapperの現在bufferはNotEnqueuedになり、次のcompute encoder作成が同assertionでexit134。小行列64×64/100回のprobeは通過。
- 同じMWEへ `--fresh` を付け、encodeごとにMPS wrapperを新規生成すると16回と全要素4097の数値assertが通過/exit0。元wrapperの再利用時、MPSGraphが内部commit/continueしてqueueの元bufferと乖離する挙動が直接観測された。JeffClientの `batched_mps_command_buffer` はowner identityだけでwrapperを再利用しており、この連携が再現条件。Metal.jl単独の公開演算の不具合を立証したものではない。本体の修正はまだしていない。
- 最終MWEの再現log `artifacts/metal-mwe/reproduce.log`、対照log `artifacts/metal-mwe/control-fresh.log`。縮小probe/formatterログも同ignored directory。内部API（ensure_cmdbuf!/end_encoder!/record_operation!/maybe_autoflush!）はMetal1.11.1の `src/command_batching.jl`、MPS wrapperは `lib/mps/command_buf.jl`、graph encodeは `lib/mpsgraphs/execution.jl` を参照。

## 2026-10-01: Apple M4 の旧計測を破棄

ユーザーの指示により、旧 Apple M4 の CPU/Metal/MLX ベンチマークはすべて無効。旧速度比・割当比較を根拠に使わない。docs/src の旧表と M4 JSON、README/PLAN の旧速度記述を削除。実装・所有権・数値検証の知見は性能値とは区別する。新しい比較は1thread/全256token/F32を基準にし、8thread対Accelerate自動10threadとGPU結果は別記する。

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

## MPS とバッファ再利用

- Laya.jl と Metal 1.11.1 の汎用 MPSGraph 行列積は `alpha*(A*B) + beta*C` を構築する。beta=0 でも C を読み、`0*NaN` が結果に混ざり得るため、当初は出力を毎回ゼロ初期化した。
- 積専用の graph は `A*B` だけを構築し、C を入力に含めないため、この初期化と alpha/beta の演算を除ける。`tools/verify_metal_primitives.jl` で、出力を NaN で埋めた通常・バッチ・転置の 8 組合せが一致することを確認した（最大誤差 `1.1920929e-7`）。実モデル 12 ケース × 3 回も最大 logit 誤差 `3.3408403e-5` で通った。
- MPSGraph を Metal の現在の command batch にエンコードすると、行列積ごとの個別 commit を減らせる。配列だけでなく feed/result と Objective-C オブジェクトも、GPU 完了まで queue の roots に保持する必要がある。
- private バッファの再利用と shared アップロードの再利用は寿命条件が異なる。shared の CPU 書き込みは GPU 完了を確認してから行う。実装は Metal 1.11.1 の内部 API に依存している。

## 演算をまとめた結果

- この時点の実モデル検証は 12 ケース × 3 回、最大 logit 誤差 `3.3140182e-5`。小さい fixture の一致だけでなく、実モデルの重みでも確認した。

## RoPE と head 配置変換の融合

- Laya の `split_rope` を参考に、Q/K の centered RMS、partial RoPE、grouped KV head の展開、MPS 用 `(head_dim, sequence, heads)` 配置への書き込みを専用カーネルにまとめた。K の処理と同時に V を配置し、出力側の配置変換と sigmoid gate も一つにまとめた。通常の slice、`cat`、`permutedims`、KV index の GPU アップロードが full attention から消えた。
- RoPE の cos/sin は queue ごとに現在の一組だけをキャッシュする。キーは rotary width・系列長・Float32 の base。CPU 実装と同じ Float32 の周波数計算を使う。テーブルを置き換えても queued kernel が配列を保持するため、使用中の shared バッファを CPU が書き換えない。
- 単体検証では、head width 4/7/256、full/partial/no RoPE、KV の共有あり・なし、rotary width と base の変更、cache hit、出力 gate を CPU 実装と比較した。小さい fixture は最大誤差 `2.3841858e-7`。実モデル 12 ケース × 3 回の最大 logit 誤差は引き続き `3.361702e-5`。
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



- Cthulhu は型推論結果を見ながら呼び出し先へ降りるために使った。TypedSyntax は結果を元のソースに対応づける表示に使った。これらがコードを自動修正したわけではなく、見つけた割当元をもとに実装を変更した。
- RoPE 融合後の JET 6 対象は既に報告なしで、Profile.Allocs の MPS feed/result・tensor-data 関連 3 箇所は約 34% を占めていた。この実測を出発点に、Cthulhu の対話的 descent で `MPSGraphTensorData(::MtlArray)` → `convert(MPSShape, reverse(size(matrix)))` へ降りた。
- Metal の shape 変換は `NSArray(NSNumber.(collect(tuple)))`。戻り値は具体的な `NSArray` に推論されていたが、呼び出しごとに `Vector{Int}`、`Vector{NSNumber}`、Objective-C の array を作っていた。型が確定していても、こうした明示的なオブジェクト生成の割当は残る。
- 修正では `ProductGraph` に A/B/C の immutable な shape を保持し、`graph_tensor_data(matrix, cached_shape)` から buffer・既存 shape・dtype を受け取る MPS コンストラクタを呼ぶようにした。shape は graph key のサイズと一致するので、通常・転置・バッチ行列積で共有できる。tensor-data と feed/result 辞書の生成そのものは残っている。
- 最初の shape キャッシュでは実モデル検証が `NSInvalidArgumentException`（解放済みの array に `count` を送信）で落ちた。ObjectiveC.jl の `NSArray` は `managed=false` の autoreleased wrapper であり、Julia のフィールド参照だけでは pool 終了後の Objective-C オブジェクトの生存を保証しなかった。
- shape ごとに明示的な `retain` を行い、mutable な `ProductGraph` の finalizer で対応する `release` を行う形に修正した。修正後の実モデル **12 ケース × 3 回**は GC を挟んで通り、最大 logit 誤差は **`3.361702e-5`**。単体 verifier も同じ graph を GC/pool 終了後に再利用する検証へ拡張した。
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

## 最終正規化と mask コピーの削減

- 各層に渡す同じ mask 行のコピーを、各 batch 行につき一度だけ作る形へ変更した。
- 最終 RMS は列ごとに独立し、readout は末尾列のみ使うため、末尾列を取り出してから正規化する形へ変更した。全系列の最終正規化バッファを作らずに済む。
- Metal の独立参照検証は系列長 1〜512 と混合言語・padding を含む 12 ケース × 3 回で数値一致を確認。最大 logit 誤差は `3.361702e-5`。
- Laya の `residual_norm` は residual 出力と正規化出力を同じ reduction kernel で生成する。Jeff でも post-attention の加算と RMS の融合を次の候補とする。Laya の LayerNorm と Jeff の RMS は統計計算が異なるため、カーネルをそのまま移植せず RMS の数値順序を維持する。

## residual と RMS の融合

- `native_residual_rms` を追加し、Metal では既存の RMS カーネル内で residual を加算・保存してから正規化する。二つの出力バッファは必要だが、独立した加算 broadcast の起動を各層で一回減らす。
- 通常の RMS/L2 は `Val(false)`、融合版は `Val(true)` で分岐をコンパイル時に確定する。幅 4096 を超える入力は汎用実装に戻す。
- 実モデルの独立参照検証（12 ケース × 3 回）は通り、最大 logit 誤差 `3.361702e-5`。ログは `/private/tmp/jeff-residual-rms-validation.log`。

## 層内の private pool 配列の早期返却

- Laya の `release_one!` と Metal 1.11.1 の `record_operation!` を確認。queue roots は配列オブジェクトを保持するが、別の DataRef 所有権を取得するわけではない。自前 pool は物理 MTLBuffer を保持し、同じ queue の後続演算だけに再利用する。
- `native_release_temporary` は自前 `ReturnBuffer` と現在の queue が一致する場合だけ `Metal.unsafe_free!` を呼ぶ。共有アップロード、重み、外部の Metal 配列は返却しない。汎用 CPU 実装では何もしない。
- 層内の正規化出力、attention 出力、gate/up 射影、residual、down 射影を最後の消費処理の投入後に返却する。呼び出し元の入力 x はこの関数で返却しない。

## MLP の SiLU と up 乗算の融合

- `native_mlp_gate` で `native_silu.(gate) .* up` を一回の broadcast に融合した。Metal は pooled output に書き込み、SiLU 単独の中間配列と起動を除いた。CPU の汎用実装も融合した式を使う。
- 計測結果は `artifacts/metal-validation/benchmark-metal-mlp-gate.json`、検証ログは `/private/tmp/jeff-mlp-gate-validation.log`。
- 通常 RMS/L2 の起動では residual 用の二つのダミー MtlArray 引数を `nothing` に変更した。`Val(false)` で該当分岐は除かれるため配列の引数変換・保持は不要。変更後の効果は別計測で確認する。
- 最終状態でも実モデル 12 ケース × 3 回が通り、最大 logit 誤差は `3.361702e-5`。既存の MPS/RMS/RoPE プリミティブ検証も通った。実モデル検証ログは `/private/tmp/jeff-rms-args-validation.log`、プリミティブは `/private/tmp/jeff-norm-unused-primitives.log`。

## DeltaNet decay の配列単項マイナス

- `-exp.(attention.a_log)` は dot のない配列単項マイナスが broadcast 融合を切る。`-1.0f0 .* exp.(attention.a_log) .* native_softplus.(...)` へ変更して符号反転・exp・softplus・乗算を一つの broadcast にした。

## 固定 decay 係数の事前計算

- モデル読み込み時に各 backend 上で `-1.0f0 .* exp.(A_log)` を一回計算し、attention の `a_decay` に保持する。CPU と Metal はそれぞれの Float32 exp を使い、推論中には係数の再計算を行わない。読み込み後の推論重みは固定として扱う。
- CPU fixture の 5 入力で最大誤差 `3.874302e-7`、Metal 実モデル 12 ケース × 3 回で最大誤差 `3.361702e-5`。どちらも独立参照と一致した。Metal 検証ログは `/private/tmp/jeff-decay-cache-validation.log`。

## attention mask の GPU 転送の共有

- `native_prepare_mask` と `PreparedMetalMask` を追加した。各入力行で Float32 mask を一度だけアップロードし、全 attention 層で同じ device vector を使う。元の host mask も保持し、未対応幅で汎用 attention へ戻るときは host mask を渡す。単独の attention 呼び出しで host mask を渡す従来経路も使える。
- 独立参照 12 ケース × 3 回（GC と padding を含む）は通り、最大 logit 誤差は `3.361702e-5`。ログは `/private/tmp/jeff-mask-reuse-validation.log`。
- 再プロファイルの JET 6 対象は報告なし。Profile.Allocs 10% は 2,374 サンプルで、pool 生成 3 箇所計 586、通常 RMS launch 250、tensor dictionary 173、Q/K/V slice 3 箇所計 352。decay 式は前回 234 → 72 サンプルに減った。サンプル比率は時間比率ではなく、抽出の揺らぎもある。次は packed QKV の切り出しと L2 正規化の融合を検討する。ログは `/private/tmp/jeff-mask-reuse-profile.log`。

## packed QKV からの Q/K 読み取りと L2 正規化

- `packed_qk_kernel!` は convolution 出力の (packed channels, tokens) を直接読み、head ごとの平方和を SIMD reduction で求めて正規化済み Q/K を出力する。Q/K の slice コピーと reshape を除いた。平方和、epsilon、sqrt、factor、除算の順序は従来の L2 kernel と同じ。V の slice はまだ残る。
- 実モデル 12 ケース × 3 回は通り、最大 logit 誤差 `3.361702e-5`。検証ログは `/private/tmp/jeff-packed-qk-validation.log`。
- 現在の private pool は allocation miss のときだけ上限超過を見て全消去する。既存サイズで reuse が続く場合、保持量を縮小する機会がない。`limit_bytes` も統計へ追加した。GPU 完了後の縮小と、forward 内の必要数を限定する設計を検討する。
- `trim_completed_buffer_pool!` を追加し、`native_host` の `Array(input)` が GPU 完了を待った後、free pool の上限を超えるバッファを大きいサイズから解放する。通常は bytes 比較だけ行い、上限以下なら終了する。GPU 完了前には呼ばない。上限は live buffer と peak/resident memory の上限ではない。
- この縮小を加えた実モデル 12 ケース × 3 回は通り、最大 logit 誤差は `3.361702e-5`。ログは `/private/tmp/jeff-pool-trim-validation.log`。縮小後の保持量と速度は次に計測する。
- ベンチマークには GC 後の明示的縮小を行った第三の `metal_pool_after_trim` も追加した。これらの統計処理はすべて latency/heap trial の外で行う。post-GC と post-trim を混同せず、遅延返却の影響と cache 制御の効果を分けて判断する。

## DeltaNet の packed V の直接読み取り

- recurrent kernel に V の開始行を渡し、convolution 出力から `start + (head-1)*value_dim + row` の行を直接読む形にした。V の slice コピーと reshape が不要になった。Q/K/V 切り出し用のコピー配列はすべて除去したが、正規化済み Q/K と recurrent 出力の配列は残る。
- 実モデル 12 ケース × 3 回は通り、最大 logit 誤差 `3.361702e-5`。ログは `/private/tmp/jeff-packed-v-validation.log`。
- プリミティブ検証に幅 7/128/256 と系列長 1/9/65 の 9 組合せを追加した。packed Q/K を CPU L2 参照、V 直接読み取りを CPU の行列形式 recurrent 更新と比較し、全組合せが通った。head 比率 2、value 幅 3/7/9、ゼロ入力列、GC を含む。ログは `/private/tmp/jeff-packed-qkv-primitives.log`。
- V の直接読み取りを含む最終コードでも JET 6 対象はすべて報告なし。ログは `/private/tmp/jeff-packed-v-types.log`。

## batch 2・長さ 512 の original Python 比較

- 拡張参照 case 12（active tokens 512/256）、M4、Float32、同期・readout・CPU score return 込みで双方 20 回を順に計測。Julia は batch 行を順次処理、original Python は元の forward の batch 処理を使う。
- 記録は `artifacts/metal-validation/benchmark-metal-b2-l512.json` と `benchmark-python-b2-l512.json`。長い系列でも改善を確認したが、Julia の実 batch 化と tail latency には改良の余地がある。

## DeltaNet 出力の RMS と SiLU gate の融合

- 正規化カーネルに `Val{GATED}` specialization を追加し、RMS の weight 乗算後に `native_silu(gate)` を掛けて保存する。正規化だけの中間出力と後続 broadcast 起動を除いた。通常 RMS/L2 と residual RMS では gate に `nothing` を渡し `Val(false)` で分岐を除く。
- 実モデル 12 ケース × 3 回は通り、最大 logit 誤差は引き続き `3.361702e-5`。ログは `/private/tmp/jeff-rms-gate-validation.log`。
- RMS+gate の単体検証に幅 8/128/256/1024、3D head/token 配列、ゼロ入力列、正負の gate（約 ±12）、正負の weight、GC を挟む 2 回実行を追加した。CPU RMS と SiLU の式に一致し、既存プリミティブ検証も通った。ログは `/private/tmp/jeff-rms-gate-primitives.log`。
- 融合後も JET 6 対象は報告なし。Profile.Allocs 10% は 1,876 サンプルで、pool の DataRef/MtlArray 生成 3 箇所は計 523、tensor dictionary は 183、packed Q/K 起動 117、層末尾 residual broadcast 96、RMS gate 起動 68。管理ラッパーと MPS submission が依然として残る。サンプル比率は実行時間の比率ではない。次は層内で所有する residual への in-place 加算と reusable workspace を検討する。ログは `/private/tmp/jeff-rms-gate-profile.log`。

## 層末尾の residual 加算を in-place 化

- 層内で新規生成した residual へ `residual .+= mlp` と書き込み、同じ配列を返す形にした。呼び出し元の入力と重みは変更しない。正規化・MLP の先行読み取りを同じ queue へ投入した後の書き込みなので、処理順序を保持する。
- CPU fixture 5 入力と Metal 実モデル 12 ケース × 3 回は通った。最大 logit 誤差は CPU `3.874302e-7`、Metal `3.361702e-5`。Metal ログは `/private/tmp/jeff-residual-inplace-validation.log`。

## MLP gate 射影配列の in-place 再利用

- `native_mlp_gate!` は層内の gate 射影へ SiLU と up 乗算の結果を書き込み、同じ配列を down 射影へ渡す。従来の活性化結果用 pooled output は不要になる。関数名の `!` で gate を変更する契約を明示した。入力 hidden とモデル重みは変更しない。
- in-place residual/MLP を含む最終状態でも JET 6 対象は報告なし。ログは `/private/tmp/jeff-mlp-inplace-types.log`。

## NSArray の値配列変換に残るポインタ Vector

- `tensor_dictionary` の `NSArray(values)` を ObjectiveC.jl の実装で追跡した。`foundation.jl` の `NSArray(::Vector{<:Object})` は `arrayWithObjects:count:` を呼び、`syntax.jl` の `Base.cconvert(::Type{<:id}, ::Vector{<:Object})` が `idArray([pointer(obj) for obj in objs], objs)` を生成する。
- このため既存の `Vector{MPSGraphTensorData}` に加え、Objective-C ポインタの Vector が feed と result ごとに一つ生成される。具体型でも変換用コンテナの割当は消えない。MPS の値配列・tensor-data の生成をすべて除けたわけではない。
- feed 2 個・result 1 個を固定長のポインタ領域で渡す方法を次に検討する。C 呼び出し中はポインタ領域と元の managed tensor-data を GC から保護し、GPU 完了までは元の値を queue roots に保持する。ポインタ保持領域の寿命と GPU オブジェクトの寿命は別に扱う。実装変更の効果はまだ未計測。
- 既存プリミティブ検証と実モデル 12 ケース × 3 回は通り、最大 logit 誤差は `3.361702e-5`。ログは `/private/tmp/jeff-fixed-pointer-primitives.log` と `/private/tmp/jeff-fixed-pointer-validation.log`。
- 次の変更では ProductGraph に固定キーの Ref ポインタ領域を保持し、`NSDictionary dictionaryWithObjects:forKeys:count:` へ値・キーのポインタを直接渡す。値の NSArray 作成を省く。固定キーの元の NSArray は明示的 retain/release を維持し、managed tensor-data の値 Vector も従来どおり queue roots に残す。
- この直接辞書版は、GC 後の graph 再利用を含む既存プリミティブ検証を通った。ログは `/private/tmp/jeff-direct-dictionary-primitives.log`。実モデル・割当・速度の検証はまだ必要。

## 重みの MPS tensor-data の再利用

- Metal の `native_linear` に重み専用の経路を追加し、固定の buffer/shape/dtype を持つ重みの MPSGraphTensorData を再利用する。activation と出力の tensor-data は引き続き毎回生成する。系列長が変わっても重みの物理 shape は変わらない。
- cache は array objectid をキーにし、所有者の WeakRef と managed tensor-data を保持する。objectid の一致だけでなく所有者の identity を確認する。所有配列の finalizer でエントリを削除し、別の所有者のエントリを誤って消さないよう確認する。
- 実モデル 12 ケース × 3 回は通り、最大 logit 誤差 `3.361702e-5`。ログは `/private/tmp/jeff-weight-tensor-validation.log`。一時的な重みで linear の数値を確認し、配列が scope を抜けた後に full GC/synchronize/full GC を行って cache エントリが消えることも確認した。
- 変更後の割当・速度と JET は次に測定する。native tensor-data は buffer のネイティブ所有権を持つため、WeakRef だけでなくエントリ削除まで確認する必要がある。
- `tools/verify_metal_primitives.jl` に継続的な検証を追加した。同じ重みで系列長 1/9/65/1 の linear を CPU 参照と比較し、GC を挟んでも同一 tensor-data を再利用することを確認する。重みが関数 scope を抜けた後には full GC・同期・full GC を行い、cache エントリの削除を確認する。追加した検証と既存プリミティブ検証はすべて通った。ログは `/private/tmp/jeff-weight-cache-primitives.log`。
- 検証追加後の再診断でも JET 6 対象は報告なし。Profile.Allocs 10% は 1,693 サンプルで、`pooled_array` の DataRef/MtlArray 生成 3 箇所が計 513（約 30.3%）、packed Q/K 起動が 141、residual RMS 起動が 85。型別では DataRef 86、RefCounted 51、Atomic 48、MPS tensor-data の Vector 46、tensor-data wrapper 45 サンプルだった。物理 GPU buffer の再利用だけでは管理オブジェクトの生成は消えない。次は同時に生存する中間配列の区別と GPU 完了条件を守る reusable workspace を検討する。サンプル比率は時間比率ではない。ログは `/private/tmp/jeff-weight-cache-profile.log`。

## 中間配列 workspace の試作

- `JEFF_METAL_WORKSPACE=1` で有効になる試作を追加した。task-local な workspace に各 `pooled_array` 呼び出し位置の配列を保持する。同じ shape の異なる位置には別スロットを割り当て、同時に生存する Q/K・residual などを上書きしない。型・shape が一致する次回呼び出しでは MtlArray/DataRef の生成を省く。
- 一入力行の scope は CPU score の readback まで含み、次行のスロット再利用は GPU 完了後に始める。例外時には同期してから scope を終了する。queue が変われば workspace を交換し、実行経路が短くなれば余った末尾スロットを削除する。CPU と通常の単体 kernel 呼び出しは従来経路を使う。
- workspace 自体が配列を保持するため、その GPU メモリは free pool 統計に含まれない。割当の減少だけでメモリ全体の改善を判断してはいけない。型診断・速度・保持メモリの測定前には既定で有効にしない。独立参照 12 ケース × 3 回の検証を `/private/tmp/jeff-workspace-validation.log` に実行中。
- 試作の独立参照 12 ケース × 3 回は完了し、最大 logit 誤差は従来と同じ `3.361702e-5`。GC と系列長変更を含む数値検証は通った。20 回の性能測定は `/private/tmp/jeff-workspace-benchmark.log`、結果の保存先は `artifacts/metal-validation/benchmark-metal-workspace.json`。
- この測定では private pool misses は392、reuses/free bytesは0。配列が workspace に保持されるため、free bytes=0 は GPU メモリを使っていない意味ではない。`metal_pool_stats` に `workspace_arrays` と16KB単位で確保された `workspace_bytes` を追加した。初回測定は追加前なのでこれらの値は未記録。モデル重み・workspace外の配列・native MPS資源・peak/resident memoryはこの値に含まれない。
- workspace を有効にした状態でも既存 JET 6 対象はすべて報告なし。ログは `/private/tmp/jeff-workspace-types.log`。保持量の統計を追加した再測定は `artifacts/metal-validation/benchmark-metal-workspace-memory.json` に保存する。

### Profile による workspace 版のボトルネック確認

- 全割当11,359件（profiling自体の影響を含む）は、packed Q/K起動1,219、通常RMS起動803、residual RMS起動770、MLP gate781、residual加算771など。型別ではMPS tensor-data410、値Vector398、VectorのMemory398、MPSCommandBuffer199。workspace導入前に目立ったpooled DataRef/MtlArray生成は上位から消えた。kernel起動・broadcast・MPS submissionに残る管理オブジェクトが次の削減対象。
- CPU Profileのmain呼び出しは268サンプル、そのうちembedding gather104、`wait_cmdbuf!`101、queueのinflight制限・cleanup待ち95。gatherにはGPUArraysのbounds checkとbroadcastが含まれる。スタックは重なり、各値は加算不可。別スレッドの`__psynch_cvwait`/`kevent`が多数あり、全6,810サンプルをGPU演算時間の割合へ換算しない。Julia/LLVMのコンパイルスタックも一部残り、純粋な定常CPUコストを断定するには再採取が必要。ProfileはGPU kernel内部を計測しない。
- この結果を受け、workspaceが所有する行列積出力のMPS tensor-dataもcacheする試作を追加した。既存エントリは入力側でも利用できる。buffer/physical shapeが固定のslotだけを登録し、一時reshape wrapperは登録しない。slot交換・末尾削除時には対応するtensor-dataを削除する。数値検証は `/private/tmp/jeff-workspace-tensor-validation.log` に実行中で、割当削減量は未測定。
- tensor-data再利用版も独立参照12ケース×3回に合格し、最大logit誤差は `3.361702e-5`。系列長変更・GCを挟んだ再利用でも数値が一致した。`metal_pool_stats` に保持する `workspace_tensor_data` 個数を追加した。20回の性能測定を `artifacts/metal-validation/benchmark-metal-workspace-tensor.json` に保存する。
- Metal 1.11.1の `lib/mpsgraphs/tensor.jl:36` は `initWithMTLBuffer:shape:dataType:` でmanaged MPSGraphTensorDataを生成する。workspace slotが存続する間はbuffer/shape/dtypeが固定なので、内容の更新ごとにwrapperを作り直す必要はない。slot交換・削除時にcacheも削除する。これは固定重みだけでなく、同期条件を守る再利用activationにも適用できる。
- tensor-data再利用版のJET6対象はすべて報告なし。ログは `/private/tmp/jeff-workspace-tensor-types.log`。

## embedding gather の GPU bounds check

- Profileで目立ったembedding gatherをGPUArraysの `src/host/indexing.jl` で追跡した。vectorized indexingの `checkbounds` はGPU indexに対して `all(broadcast(checkindex,...))` を実行する。CPU側でtoken IDの範囲を検証済みでも、従来経路はGPUへ転送したindexを再検証していた。
- Float32 Metal embeddingとCPUの整数Vectorに専用gatherを追加した。ID範囲はCPUで検証し、一つのMetal kernelでembeddingを読み取ってpooled outputへ書く。workspace有効時は出力wrapperも再利用する。GPU配列のscalar indexingは使わない。空のID Vectorではkernelを起動せず、無効IDはArgumentErrorにする。
- 専用gatherの実モデル検証は完了し、12ケース×3回で最大logit誤差 `3.361702e-5`。プリミティブverifierにも幅7/128/1024、空入力、先頭/末尾/重複ID、負のID・vocabulary上限・typemax(Int64)の拒否を追加した。ログは `/private/tmp/jeff-gather-primitives.log`。
- 追加したgather単体検証と既存プリミティブ検証はすべて通った。20回の同条件性能測定を `artifacts/metal-validation/benchmark-metal-gather.json` に保存する。
- 再採取は完了し、JET6対象は報告なし。CPU flat出力（mincount=10）には旧gatherのGPUArrays `checkbounds`/`checkindex`経路が現れなくなった。main推論スタック164サンプル中、`wait_cmdbuf!`100、inflight制限/cleanup待ち91、MPS encode29、Metal kernel launch41。これらは重なる呼び出しで加算不可。thread1のkevent3,086、thread2の条件変数待ち3,286を別表示できた。ProfileはGPU kernel内部を測らず、待機を特定のGPU演算へ帰属させることはできない。

## workspace の MPS result 値 Vector 再利用

- workspaceのtensor-data cacheを1要素の `Vector{MPSGraphTensorData}` を保持する形へ変更した。同じ出力slotのMPS submissionでは、tensor-dataに加えてresult値Vectorを再利用する。Vectorを推論中に変更せず、従来どおりqueue rootsにも保持する。feed側は毎回生成する。resultはworkspaceが所有する出力だけを保持し、モデル重みをworkspaceへ追加保持しない。
- 追加したworkspace検証と既存プリミティブ検証はすべて通った。20回の同条件性能測定を `artifacts/metal-validation/benchmark-metal-result-vector.json` に保存する。
- result Vector再利用版でもJET6対象はすべて報告なし。
- forward scopeを導入したCPU経路もfixture5入力に合格し、最大誤差 `3.874302e-7`。ログは `/private/tmp/jeff-workspace-cpu-validation.log`。workspace無効の通常Metal経路は `/private/tmp/jeff-default-gather-validation.log` で独立参照検証を実行する。
- workspace無効の通常Metal経路も12ケース×3回に合格し、最大logit誤差 `3.361702e-5`。通常・workspace両方の数値検証を確認した状態で今回の変更をまとめる。workspaceは引き続きopt-inで、既定有効化や全goalの完了を意味しない。
- 段階別測定toolのMLPを現在のin-place gateへ合わせ、attentionには準備済みmaskを渡すよう修正した。段階ごとに同期するため、各時間はfull forwardへ単純加算できない。結果の保存先は `artifacts/metal-validation/benchmark-stages-current.json`。
- recurrent threadgroupの行数1/4/8/16を比較するtoolを追加した。各設定で同じ入力の出力が一致することを確認してから同期込み測定を行う。製品側のrows=8はまだ変更していない。結果の保存先は `artifacts/metal-validation/benchmark-stages-recurrent-rows.json`。
- rows=1候補の独立参照12ケース×3回は通り、最大logit誤差 `3.361702e-5`。同期・readout・CPU返却込みの20回測定を `artifacts/metal-validation/benchmark-metal-recurrent-row1.json` に保存する。workspace有効の条件で直前のrows=8・result Vector再利用版と比較する。

## recurrent decay の exp 再評価の削減候補

- recurrent kernelはhead/tokenで同じ `exp(decay)` をvalue行/laneごとに評価していた。既存のdecay broadcastを `exp.(a_decay .* softplus.(...))` にし、head/tokenあたり一つのfactorを保存する候補を追加した。中間配列とbroadcast起動数は増やさず、recurrentはfactorを読む。
- kernelには `Val{PRECOMPUTED}` を追加し、従来のlog-decayを受ける単体検証・段階別toolも引き続き使えるようにした。製品側は `Val(true)` でexpを省く。数値検証は `/private/tmp/jeff-decay-factor-validation.log`。割当・速度への効果はまだ未測定。
- factor事前計算版は独立参照12ケース×3回に合格し、最大logit誤差 `3.361702e-5`。単体verifierはdecayを-12〜0へ広げ、従来のlog-decayと事前exp factorの両方をCPU recurrent参照と比較する。ログは `/private/tmp/jeff-decay-factor-primitives.log`。
- factor事前計算と従来のlog-decayを含むプリミティブ検証はすべて通った。workspace有効・同条件20回の測定を `artifacts/metal-validation/benchmark-metal-decay-factor.json` に保存する。

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
- feed Vector再利用版もJET6対象はすべて報告なし。

## packed Q/K 正規化の起動を一回にまとめる候補

- `packed_qk_pair_kernel!` 内で従来のQ/K正規化を順に実行し、別々のquery/key出力へ保存する。平方和・SIMD reduction・epsilon・factor・除算の式を共有し、出力配列の数は変えず、GPU起動数を2→1へ減らす。`packed_qk` の単独経路も残す。
- `packed_qk_pair` は2個のpooled配列を確保し、workspaceではそれぞれ別slotを再利用する。kernelがまとまってもquery/keyのbufferをaliasしない。
- Q/K一回起動版の独立参照12ケース×3回は合格し、最大logit誤差 `3.361702e-5`。追加の単体検証は `/private/tmp/jeff-qk-pair-primitives.log` に実行する。

## DeltaNet beta/decay の起動融合候補

- betaのsigmoidとdecayの `a_decay * softplus(a + dt_bias)` を単一の1D Metal kernelで計算する `delta_gates` を追加した。式は従来のscalar helperを共有する。b/a射影は別々のままで、beta/decay出力も別pooled配列を保持する。workspaceでは両出力の管理wrapperも再利用する。
- beta/decay融合候補の独立参照12ケース×3回は合格し、最大logit誤差 `3.361702e-5`。単体検証は `/private/tmp/jeff-delta-gates-primitives.log` に実行する。
- beta/decay融合のhead数・長さ・入力±100・GCを含む単体検証と既存プリミティブ検証はすべて通った。同条件20回の性能測定を `artifacts/metal-validation/benchmark-metal-delta-gates.json` に保存する。

## DeltaNet mask 乗算の workspace 出力候補

- mask乗算のgeneric broadcastを `delta_masked_input` へ変更する候補を追加した。列ごとのdevice maskを一つのMetal kernelで読み、pooled outputへ保存する。workspace有効時はこの出力のMtlArray/DataRefも再利用する。CPU maskのアップロードは既存PreparedMetalMaskを共有する。
- mask専用kernelは独立参照12ケース×3回に合格し、最大logit誤差 `3.361702e-5`。単体検証は `/private/tmp/jeff-delta-mask-primitives.log` に実行する。

## MLP gate 専用起動候補

- ProfileのMLP gate769件を対象に、既存in-place broadcastをFloat32 MtlMatrix専用kernelへ置き換える候補を追加した。SiLUのscalar helperと既存destinationを共有し、新しいGPU出力配列は作らない。単体verifierに幅7/128/3584、長さ0/1/9/65、入力±100程度、destination identityとGCを含むCPU参照比較を追加。実モデル12ケース×3回の検証ログは `/private/tmp/jeff-mlp-kernel-validation.log`。性能・数値検証はまだ完了していない。

## 残差加算の専用起動候補

- Profileのresidual加算772件を対象に、`native_residual_add!` のgeneric in-place broadcastとMetal専用kernelを分岐する候補を追加した。既存residualを更新し、新しい出力配列は作らない。単体verifierに幅7/128/1024、長さ0/1/9/65、destination identity、GC、同一配列を両入力に渡すaliasケースを追加した。実モデル検証ログは `/private/tmp/jeff-residual-kernel-validation.log`。性能・検証はまだ完了していない。
- 実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`。generic hook変更のCPU fixture5件も合格、最大 `3.874302e-7`（`/private/tmp/jeff-residual-cpu-validation.log`）。単体検証は `/private/tmp/jeff-residual-kernel-primitives.log` に実行する。性能はまだ未測定。

## RMS 起動引数をまとめる候補

- 単体検証と実モデル12ケース×3回はすべて合格、最大logit誤差 `3.361702e-5`。実モデルログは `/private/tmp/jeff-rms-config-validation.log`。同条件20回測定を `artifacts/metal-validation/benchmark-metal-rms-config.json` に保存する。

## 層間の residual/input RMS 融合候補

- `native_layer_outputs` で層末尾のresidualとMLP出力を別々に返し、Metal forwardでは次層input RMSのkernel内で加算とresidual更新を行う候補を追加した。同じqueue上で前層MLPの読み取り後に書き戻す。normalized出力だけ新たに必要で、大きなresidual配列を追加しない。最終層はreadout列だけ加算・RMSを行う。24層モデルの独立加算24起動を省く狙い。generic forwardは従来どおり各層を加算して返し、hidden幅4096超はgenericへfallbackする。単体verifierにresidual identityとCPU式比較を追加した。実モデル検証は `/private/tmp/jeff-residual-rms-fusion-validation.log`。性能・精度はまだ未確認。
- 実モデル12ケース×3回は合格し最大logit誤差 `3.361702e-5`。CPU fixture5件も合格、最大 `3.874302e-7`（`/private/tmp/jeff-residual-rms-fusion-cpu.log`）。単体検証は `/private/tmp/jeff-residual-rms-fusion-primitives.log` に実行する。性能は未測定。
- 最終列view版も実モデル12ケース×3回に合格、最大logit誤差 `3.361702e-5`（`/private/tmp/jeff-residual-rms-view-validation.log`）。単体verifierには最後の列のview更新が親配列の他列を変更しないことと、GPU起動直後のGCを追加した。単体ログは `/private/tmp/jeff-residual-rms-view-primitives.log`。性能はまだ未測定。

## 同じ open command buffer 内での MPS wrapper 再利用候補

- ProfileのMPSCommandBuffer199件を対象に、現在taskの最新MTLCommandBuffer ownerとMPS wrapper一組を保持する候補を追加した。`Metal.ensure_cmdbuf!` が同じJulia ownerを返す間だけwrapperを再利用し、flushでownerが切り替わると作り直す。pointer値だけで判定しない。各encodeのqueue rootsにもwrapperを記録するため、cache更新後も未完了GPUのwrapperは保持される。スコアreadback後とworkspace例外cleanup後にcacheのowner/command参照を外す。
- Metal 1.11.1の `src/command_batching.jl` はflush時にopen cmdbufをnothingに戻す。`lib/mps/command_buf.jl` のwrapperは外部MTLCommandBufferを包むconstructorで、明示的commitAndContinueは内部bufferを切り替えるが本実装では呼ばない。Layaも同じopen bufferへencodeするがwrapper再利用はしていない。既存プリミティブ検証は合格（`/private/tmp/jeff-mps-command-primitives.log`）。identity・GC・submit後の非再利用・readback後clearの追加検証は `/private/tmp/jeff-mps-command-identity.log`。実モデルと性能は未検証。
- identity等の追加検証と実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`（`/private/tmp/jeff-mps-command-validation.log`）。workspaceでMPSをencodeした直後に例外を投げた場合も参照解除されるassertを追加し、`/private/tmp/jeff-mps-command-exception.log` に単体検証を再実行する。性能は未測定。

## 小さな kernel の重複サイズ引数を省く候補

- delta gateのhead数/length、maskのwidth/length、MLP gateとresidual addのlengthを独立scalar引数として渡す代わりに、GPU側MtlDeviceArrayのsize/lengthから取得する候補を追加した。Int32への変換と演算順・出力所有は従来と同じ。配列descriptorがすでに同じ情報を持つため、起動引数tupleとscalar boxingを減らす狙いだが、GPU側サイズ取得やcodegenの影響も測定する。単体検証はすべて合格（`/private/tmp/jeff-kernel-dims-primitives.log`）。実モデル検証は `/private/tmp/jeff-kernel-dims-validation.log`。性能は未測定。
- 実モデル12ケース×3回も合格、最大logit誤差 `3.361702e-5`。同条件20回測定を `artifacts/metal-validation/benchmark-metal-kernel-dims.json` に保存する。先に手順を更新した `jeff-metal-performance` skillもquick_validateで合格した。PythonCallを単発コマンドで使う場合も `include("tools/python_env.jl")` をusingより先に実行し、既存extern/jeff interpreterとNull Conda backendを選ぶ。

## 送信バッチと kernel handle cache の比較候補

- `ext/metal_kernels.jl` にdefault compiler optionsのsingleton kernel専用handle cacheを試作し、まずMLP gateの24起動だけに適用する。device identity・GPU引数型・Julia world counterで判定し、method更新時に全handleを破棄する。closure等のstateful functionは通常mtlfunctionへfallbackし、実際の起動はMetal HostKernel callableを使うのでqueue roots/encoder/synchronizationを迂回しない。GC preserveも通常macroと同様に行う。追加cache lookupの費用も含めて比較する。単体検証ログは `/private/tmp/jeff-kernel-handles-primitives.log`。
- 単体検証は合格し、追加のGC後handle identityとGPU関数methodを書き換えて出力1→2へ変わる検証も合格（`/private/tmp/jeff-kernel-handles-world.log`）。実モデル検証は `/private/tmp/jeff-kernel-handles-validation.log`。キャッシュがmethod更新を無視して古いGPUコードを起動しないことを直接確認した。性能は未測定。
- MLP/delta gate/maskへ適用した候補も単体検証・capture値3/4のclosure fallback・method更新と、実モデル12ケース×3回に合格（最大logit誤差 `3.361702e-5`、`/private/tmp/jeff-kernel-handles-simple-validation.log`）。同条件20回測定を `artifacts/metal-validation/benchmark-metal-kernel-handles-simple.json` に保存する。

## RMS 系への kernel handle cache 適用候補

- shared normalization kernelの4起動箇所（通常RMS、post residual RMS、gated RMS、次層input residual RMS）にも既存handle cacheを適用する候補を追加した。device/argument-type/worldによる判定と実際のMetal HostKernel起動を共有し、GPU算術・threadgroup配置・配列所有は変えない。NormalizationConfigの型パラメータもGPU引数型keyに含まれる。単体検証ログは `/private/tmp/jeff-kernel-handles-rms-primitives.log`。性能・実モデル検証は未完了。
- 直接置換の単体検証は合格したが、JETで4対象にruntime dispatchを検出した（`/private/tmp/jeff-kernel-handles-rms-types.log`）。原因は `NormalizationConfig{_A,...} where _A` のPARTSが実行時widthから決まり、private cached launchの呼び出し型が確定しないこと。既存Metal macroの外部呼び出しに隠れていた境界がprivate helperに現れた。報告対象moduleを除外して隠さず、主要幅1024/128/256と32以下でVal(PARTS)を明示し、ほかの幅は従来macro起動へ進む候補に修正した。任意幅でPARTSを大きく丸めることはせず、GPU算術を変えない。再JETログは `/private/tmp/jeff-kernel-handles-rms-parts-types.log`。
- 主要幅のPARTSを確定した版はJET6対象すべて報告なし。単体verifierには幅33/64/129/257/512のRMS・gated RMSも追加し、通常macroへ進む幅の数値とGCも検証する。ログは `/private/tmp/jeff-kernel-handles-rms-parts-primitives.log`。GPU算術の変更によって型問題を避けたのではなく、host側の主要幅の分岐で型パラメータを確定している。
- 増やした幅を含む単体検証と、実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`（`/private/tmp/jeff-kernel-handles-rms-parts-validation.log`）。20回の性能比較は `artifacts/metal-validation/benchmark-metal-kernel-handles-rms.json` に保存する。callbackの型分岐の費用も含めて従来と比較する。

## Attention の固定引数 kernel handle cache 候補

- RMS版の全割当Profileに基づき、causal depthwise・masked softmax・merge gateの3起動を既存cached launchへ移す候補を追加した。GPU算術・scalar引数・配置は維持する。実行時Valを含むQK/RoPE/recurrentは今回の変更に含めない。単体検証は `/private/tmp/jeff-kernel-handles-attention-primitives.log`。性能と実モデル検証は未完了。

## Attention helper の重複サイズ引数削減候補

- depthwiseのchannels/length/kernel、merge gateのwidth/heads/length、softmaxのlength/columnsを独立scalar引数で渡す代わりに、GPU側のinput/weight/values/scoresのdescriptorから取得する候補を追加した。Int32変換、算術順、配置、配列所有は維持する。単体検証ログは `/private/tmp/jeff-attention-dims-primitives.log`。性能・実モデル・型検査は未完了。

## 採用済み RMS/attention handle cache の通常設定再測定


## Packed Q/K の主要幅での kernel handle cache 候補

- 全割当Profileで582件を記録したpaired Q/K起動について、実モデルのkey_dim128に限ってVal(4)を明示し既存cached launchへ進む候補を追加した。他の幅は従来macroで正確なcld値を使う。RMSで確認した実行時Valによるruntime dispatchを避け、GPU算術・配置・所有は変更しない。単体検証ログは `/private/tmp/jeff-qk-handles-primitives.log`。性能・実モデル・型検査は未完了。

## Recurrent 起動の主要幅での kernel handle cache 候補

- 最新全割当Profileで487件を記録したrecurrent起動にも、key_dim128に限りVal(4)/Val(8)を確定して既存cached launchを適用する候補を追加した。rows8、GPU演算、scalar引数、配置、所有は維持し、他のkey幅は従来macroを使う。単体検証は `/private/tmp/jeff-recurrent-handles-primitives.log`。性能・実モデル・型検査は未完了。

## RoPE 起動の主要幅での kernel handle cache 候補

- 最新全割当Profileでquery/key起動は168/167件を記録した。head_dim256に限りVal(8)とqueryフラグを確定したhelperで既存cached launchを使う候補を追加し、他の幅は従来macroを使う。GPU算術・配置・所有は変更しない。単体検証は `/private/tmp/jeff-rope-handles-primitives.log`。性能・実モデル・型検査は未完了。

## Workspace 過去 slot の MPS tensor-data 再利用候補

- ForwardWorkspaceにobjectid→slot番号の索引を追加し、graph_tensor_dataが現在cursor以前のowned slotで配列identityも一致するときに登録する候補を追加した。slot置換・末尾削除・clearで索引を削除する。通常設定と一時reshape wrapperの経路は維持し、GPU buffer所有や同期条件は変えない。単体verifierに過去slotの入力配列/TD identity、length9/9/1/1/65/65、GC、索引一致、例外後の末尾削除、clearを追加した。検証ログは `/private/tmp/jeff-slot-td-primitives.log`。性能・実モデル・型検査は未完了。

## DeltaNet gate の直接行列出力候補

- 残る18 TDと、18 DeltaNet層のgated出力reshape→out projection経路が対応する候補を調べた。normalization kernelはoutputに線形添字で書き込み、幅と列数はinputから決めるため、同じ要素数の行列outputを直接確保できる。rms_silu_gateにoutput_dimsを追加し、DeltaNetは(value_dim*value_heads,length)のowned slotへ書く候補に変更した。通常呼び出しは元の形状を維持し、GPU算術・入力RMS幅・配列所有は変えない。verifierに各幅の直接行列出力を追加。単体ログは `/private/tmp/jeff-flat-gate-primitives.log`。実モデル・型検査・性能は未完了。

## Packed MLP projection の段階測定候補

- Layaの `ext/LayaMetalExt.jl:398` のgelu_gate_kernelはpacked projectionの前半・後半を読み、別の半幅outputへgate結果を出す。我々はSiLUなので算術は既存native_siluを使い、同じ配置でgate/up weightを一度結合し1回のMPS積と専用gate kernelで処理する診断候補を `tools/benchmark_stages.jl` に追加した。weight CPU readback/結合/uploadは測定外、候補は追加weight copyを保持しpacked projection＋半幅gate outputの一時bufferを使う。通常実装は変更していない。既存MLPと2e-4許容で数値比較してから同期込み10回の段階測定を行う。結果は `artifacts/metal-validation/stages-packed-mlp.json`、ログは `/private/tmp/jeff-stages-packed-mlp.log`。段階測定は全体forward速度の証明ではない。

## モデル全体の Packed MLP 実装候補

- coreのnative_mlp_weights/native_mlpでbackend固有の重み配置と演算をdispatchする構成を追加した。Metalでは `JEFF_METAL_PACKED_MLP=1` のときだけ読み込み時にgate/upを結合し、PackedMLPにはgate_up/downだけを保持する。元のgate/upはbackendに残さないが、読み込み中の一時CPU/GPUコピーとGC前のbuffer保持は生じる。forwardはpacked投影、半幅gate output、down投影を使い、従来よりMLPの一時buffer総要素数が増える。通常設定・CPUは既存gate/upのまま。全体の独立参照検証は `/private/tmp/jeff-packed-mlp-validation.log`、性能・型・所有/primitive検証は未完了。
- packed MLP/workspace有効の実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`。CPU算術を参照する幅7/128/3584・length1/9のpacked MLP単体検証もverifierへ追加した（未実行）。全体20回の結果を `artifacts/metal-validation/benchmark-metal-packed-mlp.json` へ保存する。
- 追加したpacked MLPのCPU算術参照・GC後再実行を含む単体verifierは合格した。全体のJET・Profileは `/private/tmp/jeff-packed-mlp-profile.log` で実行する。
- native_mlp dispatch導入後のCPU fixture verifierは合格、最大logit誤差 `3.874302e-7`（`/private/tmp/jeff-packed-refactor-cpu.log`）。packed MLPにはworkspace3 slot/TD identity、length9/9/1/1/65/65、GC、completed feed input解除、例外cleanup、clearの検証を追加した。ログは `/private/tmp/jeff-packed-mlp-workspace-primitives.log`。通常設定のpacked測定は未実施。
- 追加workspace ownership検証を含む単体verifierも合格。workspace無効のpacked版を `artifacts/metal-validation/benchmark-metal-packed-mlp-workspace0.json` へ測定する。

## 先頭 padding の計算省略候補

- B1/L256ではactive101以外の先頭155 tokenも全層で処理している。`JEFF_METAL_TRIM_PADDING=1` のMetal候補を追加し、先頭の連続0 maskだけを省いてgather/forwardする。系列内の0は残し、ID/maskの全入力検証と最終位置active要件は維持する。CPU/通常設定は省略しない。biasなしDeltaNetの先頭masked入力/stateは0、causal convolutionの境界は0相当、full attentionのactive位置間RoPE相対差は保存されるという根拠があるが、位置shiftのFloat32丸めは変わるため独立参照検証が必要。実モデルログは `/private/tmp/jeff-trim-padding-validation.log`。性能・型・interior mask/所有/設定組合せは未完了。B2で行の実長が変わるとworkspace slotの形状置換が増える点も測定する。
- workspace有効・packed無効・trim有効の実モデル12ケース×3回は合格、最大logit誤差 `3.361702e-5`。一部caseの誤差は位置shiftで変わるが許容内。同条件B1の20回を `artifacts/metal-validation/benchmark-metal-trim-padding.json` に保存する。interior mask・各設定組合せ・型検査は未完了。
- 独立参照15ケースの生成は完了した。trim有効・workspace有効・packed無効で各3回の検証を `/private/tmp/jeff-trim-padding-mask-validation.log` で実行する。benchmarkには論理sequence_lengthと区別して実際のmetal_computed_sequence_lengthsを保存する変更を加えた。
- 上記15ケース×3回は合格、最大logit誤差 `3.540516e-5`。interior holesを持つ行と最後の1 tokenだけactiveの行も独立PyTorch参照に一致した。trim有効・packed有効・workspace無効の組合せも `/private/tmp/jeff-trim-padding-packed-workspace0-validation.log` で各3回検証する。primitive verifierには先頭0だけを省く選択、先頭active/最後だけactive、CPUとtrim無効の開始位置維持を追加した（未実行）。
- trim有効・packed有効・workspace無効も15ケース×3回合格、最大logit誤差 `3.540516e-5`。B2/L512/active512・256でtrim有効・packed無効・workspace有効の20回測定を `artifacts/metal-validation/benchmark-metal-trim-padding-b2-l512.json` へ保存する。論理長512とは別に各行の計算長512/256を記録する。
- trim有効・packed無効・workspace無効も15ケース×3回合格、最大logit誤差 `3.540516e-5`（`/private/tmp/jeff-trim-padding-workspace0-validation.log`）。残るpacked/workspaceとも有効の組合せを `/private/tmp/jeff-trim-padding-packed-workspace1-validation.log` で実行する。
- trim/packed/workspaceとも有効も15ケース×3回合格、最大logit誤差 `3.540516e-5`。trimの4組合せは確認済み。

## 計算長別の最大2 workspace 再利用候補

- `JEFF_METAL_SHAPE_WORKSPACES=1` とworkspace有効のとき、native_forward_scopeの3引数版で計算長ごとのworkspaceを選ぶ候補を追加した。2組をLRUで保持し、完了後に合計buffer保持がrecommended working setの1/4を超えたら古い組を除去する（単一の現在workspaceが上限超の場合はそのまま）。既存2引数/nested scopeは選択中workspaceを進め、clearは全組を解除する。workspace自身のbytesをslotの置換/削除/clearで追跡し、pool statsは全組の配列/TD/feed/bytesと保持長を報告する。入力wrapperや未完了bufferの所有・同期条件は変えない。
- trim/shape workspace有効・packed無効で独立参照15ケース×3回は合格、最大logit誤差 `3.540516e-5`（`/private/tmp/jeff-shape-workspaces-validation.log`）。B2/L512の20回は `artifacts/metal-validation/benchmark-metal-shape-workspaces-b2-l512.json` へ保存する。LRU/byte上限/GC/例外/clearのprimitive検証とJET/Profileは未完了。
- primitive verifierにshape9/1/9/1/65で配列/TD identityとLRU、1 pageのbyte上限によるeviction、nested scopeで選択維持、例外後feed input解除、全組clearを追加した。ログは `/private/tmp/jeff-shape-workspaces-primitives.log`。
- ownership primitive verifierは完了し合格した。形状切替での配列/TD再利用、LRU、byte上限による除去、nested scope、例外後の解除、全workspace clearを確認済み。shape workspace版のJET/Profileと長時間測定は未実施であり、trim/shape workspaceは引き続き既定無効のopt-inとする。
- RoPE表をqueueごとの最大2組LRUにする候補を追加した。keyはrotary_dim/計算長/rope_thetaで、表を書き換えず、除去後もqueued kernelのrootによる寿命を維持する。primitive検証に長さ切替後のGC/identity、異なる長さの分離、2組上限、LRU除去を追加した。ログは `/private/tmp/jeff-rope-two-tables-primitives.log`。数値・型・性能の再検証は進行中で、改善量は未確定。
- 上記primitive verifierは合格した。初回は引数lengthがBase.lengthを隠してMethodErrorとなり、sequence_lengthへ改名して修正した。既存の実幅RMS/RoPE/行列積とGC後再利用、追加LRU検証を通過した。B2・50回benchmarkを `/private/tmp/jeff-shape-workspaces-rope-b2-benchmark.log` で実行する。

## workspace queue 明示の候補

- 追加kernel/queue/readback検証を含むprimitive verifierは合格した。実モデル15ケース×3回は `/private/tmp/jeff-workspace-launch-queue-validation.log` で実行する。
- queue明示後のJET6対象は報告なし、MPS command wrapper26 / TD0 / KernelState412を維持。通常HostKernel経路のままqueue検索を削減した候補を採用する。設定はworkspace有効時に限り、workspaceのGPU保持量は変わらない。

## DeltaNet入力maskと正規化の融合候補

- `JEFF_METAL_FUSED_DELTA_MASK=1` の候補を追加した。RMSの出力destinationをGPUへadaptするMaskedNormalizationOutputで包み、setindex!時に列maskを掛ける。正規化の平均・重み・residual更新は元のkernelのままで、normalizedだけをmaskedにする。Metal native_hidden_forwardではDeltaNetかつkey_dim<=256の入力正規化/層間residual入力正規化に限って使う。full attention、post norm、final readoutの正規化は変更しない。coreにpremasked Val hookを追加し、DeltaNet側はすでにmask済みのprojection入力にdelta_masked_inputを重ねない。直接native_layerや未対応幅は元の処理を継続する。
- fixture metal5ケースは合格、最大誤差2.3841858e-7（`/private/tmp/jeff-fused-delta-mask-fixture.log`）。primitive verifierにmask付きRMSとmask付きresidual入力RMSを追加し、幅8〜1024、centered/noncentered、interior holes、residual自体をmaskしないことを検証する。ログは `/private/tmp/jeff-fused-delta-mask-primitives.log`。実モデル・JET・Profile・全体性能は未完了。
- 追加masked RMS / residual RMSを含むprimitive verifierは合格した。destination生成ではmask列数・正の幅も検査する。実モデル15ケース×3回の検証は `/private/tmp/jeff-fused-delta-mask-validation.log` で実行する。GPUのmask乗算は既存のnormalization kernel出力書き込み時に行い、追加の整数列index計算とmask読出しがあるため速度改善は実測で判断する。
- 列番号再利用版も実モデル15ケース×3回合格、最大誤差3.540516e-5。CPU fixture5ケースも合格、最大3.874302e-7（`/private/tmp/jeff-fused-delta-mask-cpu-fixture.log`）。B2の50回測定は `/private/tmp/jeff-fused-delta-mask-column-b2-benchmark.log` で実行する。slot順序を持つworkspaceの入力依存分岐に関する検証原則をAGENTS.mdへ追加した。
- all-active省略版も実モデル15ケース×3回合格、最大誤差3.540516e-5。mask穴を持つ行とall-active行の混在を含む。B2/50回測定は `/private/tmp/jeff-fused-delta-mask-all-active-b2-benchmark.log` で実行する。
- workspace無効・packed有効の融合版も15ケース×3回合格、最大誤差3.540516e-5。workspace/packedとも有効の組合せを `/private/tmp/jeff-fused-delta-mask-packed-workspace1-validation.log` で検証する。更新した性能スキルはPythonCall経由quick_validateに合格した。
- workspace/packedとも有効の融合版も15ケース×3回合格、最大誤差3.540516e-5。残るworkspace無効・packed無効の組合せを `/private/tmp/jeff-fused-delta-mask-unpacked-workspace0-validation.log` で検証する。これらの実モデル検証はtrim有効時の結果であり、全設定の組合せを検証したとはしない。

## 真のbatch推論の実装着手

- mask融合は `8777e03` としてmainへpushした。次は行ごとのforwardをまとめるため、まずprivateなbatched_causal_depthwiseを追加した。inputはchannels×(sequence_length*batch)で、各sampleのtokenを連続列へ配置する。kernelは各列のsample開始位置を計算し、それより前のtapを0として扱う。既存単独行kernelと通常forwardは変更しない。
- Layaのsplit_rope_kernel/merge_heads_kernelとattention_unfusedでは、head*batchをMPSGraphのbatch軸へまとめ、投影用layoutへ戻している。Jeffもこのlayoutを利用できるが、DeltaNetの畳み込みとrecurrent stateはsampleごとに独立させる必要がある。今回の畳み込みだけではbatch推論や速度改善は成立しない。
- primitive verifierへchannels7/128、系列長1/3/9/65、batch1/2/3、tap1/4の48組合せを追加した。CPUの各行独立処理と2回比較しGCを挟む。さらに最初のsampleを100へ変更して後続sampleへの影響がないことを確認する。全primitive実行ログは `/private/tmp/jeff-batched-convolution-primitives.log`。exit0で完了し、新しいbatch境界/GC検証と既存primitive検証に合格した。モデル全体への接続・性能測定・型検査は未実施。
- batch DeltaNet recurrent kernelを追加した。gridの第2軸をvalue_heads*batchとし、sample/headを分解してflat token offsetを計算する。stateは各head/sampleのthreadgroupごとに0から始め、query/key/value/beta/decay/outputは同じsampleの列だけを参照する。既存単独行kernelは変更しない。host側は完全な系列列数、head比、packed channel数とQ/K/gate形状を検査する。key_dim<=256の範囲を対象とする。
- 幅7/128/256、系列長1/9/65、batch1/2/3の27組合せを、各sample独立のCPU state更新と比較した。2回実行とGC、最初のsampleのpacked value変更後にも後続sample結果が変わらないことを検証し、`/private/tmp/jeff-batched-recurrent-primitives.log` はexit0で合格した。新しい `tools/verify_metal_primitives.jl batch` でbatch kernelだけを検証できる。decay factorをtuple mapの前に一度計算する最終版は `/private/tmp/jeff-batched-recurrent-final-primitives.log` で再検証中。full attentionと推論本体はまだbatch対応していないため、全体速度改善は未確認。
- 最終版もexit0で完了し、batch畳み込みとrecurrentのCPU一致・sample境界・GC検証に合格した。次はfull attentionのRoPE/head layout、per-sample mask softmax、merge gateをbatch対応し、モデル全体の独立参照検証と計測へ進む。
- 上記batch verifierはexit0で完了し、畳み込み/recurrent/softmaxの3検証とも合格した。softmaxの18組合せには先頭padding、interior mask穴、有効keyがないqueryも含む。
- `ext/metal_batch_attention.jl` にbatch用RMS/RoPE/head変換、merge gateとfull attentionを追加した。projection列はsampleごとの連続token、attention配列は(head_dim, sequence_length, heads*batch)。RoPE tableの参照にはsample内token、projectionにはsample offset込みtokenを使う。KV head複製とgate sourceもsample内head/位置から計算する。head_matmulは既存のMPSGraph batch積を再利用する。通常full attention/forwardは変更しない。
- 上記batch verifierはexit0で完了し、batch full attentionのCPU一致・sample逆順・GC検証に合格した。既存batch畳み込み/recurrent/softmax検証も合格した。次はvalidated入力からのbatch forward hook、DeltaNetのまとめたprojection、各行最後のcolumnのreadoutを実装する。
- `JEFF_METAL_BATCHED=1` でbatch>1のMetal backendがvalidated入力からbatch forwardへ入るprivate hookを実装した。共通trim開始位置は各行の開始位置の最小値なので、短い行のpaddingも残す。ID/maskはsample-contiguousにflattenし、projection/MLPをまとめ、畳み込み/recurrent/full attentionだけsample境界を分離する。最後は各sample最終columnのresidual+MLPをH×Bへgatherし、final RMS/readoutとCPU返却を一度行う。既定無効、B1/CPU/未対応幅は従来経路。batch版は入力maskを独立kernelで適用し、既存mask融合flagはまだbatchには適用しない。
- batch prototypeは単一workspaceを再利用し、inactiveな従来shape bankはbatch移行時にclearする。activeなnested scopeはclearしない。異なるB/Lでは形状置換があり、batch用shape bankは未実装。benchmarkにはbatch requested/executionと共通計算長を記録する。
- trim/workspace/shape有効、packed/mask融合無効の実モデル15ケース×3回はexit0で合格、最大logit誤差3.361702e-5（`/private/tmp/jeff-batched-model-validation.log`）。B2/L512/active512・256の20回測定を `artifacts/metal-validation/benchmark-metal-batched-b2-l512.json` へ保存する。batchでは計算長512/512、従来row trimは512/256なので演算量は同じでない。JET/Profileと他設定の組合せ・モデル全体の所有/slot検証は未完了。
- 修正後batch primitive verifierはexit0で合格（`/private/tmp/jeff-batched-specialized-primitives.log`）。追加したtiny model batch forward probeでは同じB2/L9でmask値を変えてGCしても全workspace配列とtensor-data tupleのidentityが維持され、B3/B2へのshape変更後もCPU参照と一致した。終了時workspace.active=false、clearとENV復元も確認した。実モデル再検証・比較benchmarkは引き続き必要。

## 2026-10-01: README の実際の0.8Bデモを再測定

- `mstrasser/Jeff-Qwen3.5-0.8B` revision `0f212b3e72acb4dde3f7da61e925d6ab7f819990` の safetensors を使用。tiny fixture/ONNX graphではない。README parcelデモと旧独立PyTorch参照case1の入力が同一であることを確認し、入力・参照logitsを `examples/data/parcel_reference.json` に保存した。
- Apple M4、Julia1.13.1/Metal1.11.1/PyTorch2.14.0、Float32、B1/L256/active101、各20回、CPU8threads、バックエンドを順に独立実行。load/compile/tokenization/calibrationを除外し、readoutとCPUスコア返却・GPU同期を含む。PythonはGPU入力を事前準備、JuliaはCPU入力のuploadをforward内に含む。
- 最大絶対logit誤差: CPU1.049e-5、Metal既定6.676e-6、trim7.629e-6。先頭mask0のみ除去し有効tokenは残す。今回と既存の参照検証の数値一致は確認したが、大規模な分類精度評価とは区別する。
- 元Jeff commit `f06788292874c21a5b5c41549ac220dd9e15da7f`、FLA/causal-conv1dなしのPyTorch fallback。MLX比較ではない。生JSONはignored `artifacts/metal-validation/demo-0.8b-{cpu,metal,metal-trim,python-cpu,python-mps}.json`。公開表・再実行手順は `docs/src/performance.md`。

## 2026-10-01: GPUの知見をCPUへ適用

- `src/native_cpu.jl` にMatrix{Float32}専用畳み込みを追加。tap順を保持してcolumn-major loopで既存outputへ直接加算しSiLUをin-place適用。channel数を検査してから@inboundsを使う。GPUのgeneric methodは保持。
- CPUのhidden forwardでは最後の層のAttentionまでは全系列を計算し、最後のresidual/RMS/MLPだけ最終columnにする。MLPは位置ごとに独立でreadoutが最終columnしか消費しない。Metalで使った知見をCPUに移した。CPU重みはReinterpret/ReshapeでもProfileでは既にBLASへ到達しており、重量の形式だけを変える必要は確認されていない。
- `JEFF_CPU_TRIM_PADDING=1` を追加（既定0）。先頭mask0のみ除去、interior holesは残す。Metalの同名でないflagと独立。benchmarkはCPU計算長とtrim flagを記録し、`JEFF_BLAS_THREADS`（既定8）でCPU threadを変えられる。
- 生データ `artifacts/metal-validation/demo-0.8b-cpu-{fused,trim,trim-blas1}.json`、profile変更前 `profile-cpu-demo.log`、変更後 `profile-cpu-demo-after.log`。CPUscratch再利用/DeltaNet中間配列削減は残る。新しいCPU変更の全入力・モデルへの一般化はこのデモの測定だけで主張しない。

## 2026-10-01: CPU DeltaNet chunkとApple Accelerate

- CPU専用delta_attentionを実装。共有Q/Kの正規化をvalue headごとに繰り返さず一度行い、chunkはviewで参照。system/intraのdecay broadcastをin-place化し、triangular RHSは直接埋めてsolve、corrections/result/state更新をmul!のalpha/betaで融合。stateはheadごとにzero resetする。pair_decayの式は元のcumulative[i]-cumulative[j]であり、初期trialの符号違いは独立参照guardで検出して修正した。GPU処理は既存methodを使う。
- Laya.jlのext/LayaAppleAccelerateExt.jlとsrc/backends.jlを再読。optional AppleAccelerate importでprocess-wideにLBTをAccelerateへforwardする知見を採用。tools依存にAppleAccelerate=0.7.0を追加し、native CPU demo/benchmark/profilerでJEFF_CPU_ACCELERATE=1のときだけimport。macOS13.4以上でforwardを確認し、他OS/unsupported macOSは明示error。コアruntime依存は増やさず、Metal/Python/ONNXにCPUを委譲しない。
- AccelerateはOpenBLASとthread APIが異なる。LBT/BLAS.get_num_threads()は8だがAppleAccelerate.get_num_threads()は10（framework-managed）。同一8threadsのライブラリ比較とは主張しない。benchmark JSONはblas_config/accelerate_version/accelerate_threadsを保存する。
- Accelerate+trimの独立reference6caseで各3回benchmarkし、初回reference guardを通過した。英日混在B3、B2L1/65/512、B3interior masksL65/129。最大誤差3.6001205e-5。これは検証入力の数値一致であり分類精度dataset評価ではない。artifact demo-0.8b-cpu-accelerate-case-{1,2,5,12,14,15}.json、benchmark-cpu-accelerate-cases.log。その他主要artifactはdemo-0.8b-cpu-{chunk,chunk-no-trim,accelerate,accelerate-no-trim,accelerate-repeat}.json。
- optional fast CPU exampleは引数なしHF_HUB_OFFLINE=1で実行済み、Device:cpu/delivery0.996566/confidence0.993133。docs/READMEに任意設定と実測結果を記載し、Documenter build exit0。

## CPU極限チューニング: chunk scratchの層内再利用（作業中）

- active goal CPU実装を極限まで高速化。前turnはCPU chunk改良/Accelerate導入、独立参照と計測、commit7fc0082のためprogressとして扱う。
- cpu_delta_buffersにpair/system/intra、weighted/ending/scaled_query、RHS2種、corrections/resultをまとめた。full64とtail用の2組を層のforward内に所有し、head/chunk間で再利用。task共有cacheではないので他forwardとの競合はない。beta0のmul!で出力全域を上書きし、RHSとcorrectionsも利用前に完全上書きする。
- 新scratch経路の15ケース各3回benchmark/reference guardは cpu-buffers-cases.log に実行中。完了確認後にまとめる。変更後のProfile/JETとvector mathの検討は未完了。AppleAccelerate0.7 array.jlにexp!(out::Array,input::Array)があり、次の候補はscalarexp/SiLUを任意のvForce経路へ置き換えること。追加scratchコストと全forwardの数値/速度を測定して判断する。
- scratch層内再利用の15ケース×各3回benchmarkはexit0、最大logit誤差3.6001205e-5（cpu-buffers-case-{1..15}.json）。active goal前turnはsource変更/20回計測でprogress。今回もvector経路の実装と計測でprogress。
- さらにowned qkv projectionを畳み込み完了後のSiLU exp scratchとして再利用するprivate cpu_owned_causal_depthwiseを追加。通常causal_depthwiseは入力を上書きしない。JEFF_CPU_VECTOR_MATH=0では所有inputもそのままで元のscalar処理。新しいall-owned版50回はdemo-0.8b-cpu-vector-all-owned.jsonに実行中。全ケースでのvector経路の検証・Profile/JET・default無効時の再計測は未完了のためまだcommitしない。
- vector有効15ケース×3回benchmark/reference guardを cpu-vector-owned-cases.log に開始した。Profile/JETとscalar fallback確認はまだ必要。optional flag既定0のまま、目標完了は未証明。
- all-owned vector mathの15ケース×3回benchmarkはexit0、最大誤差3.3974648e-5（cpu-vector-owned-case-{1..15}.json）。前goal turnは所有配列再利用/50回計測/15case起動でprogress、今回はRMS in-place/Accelerate threading比較/Profilerでprogress。
- RMS in-place1＋vector1＋Accelerate＋trim1の15ケース各3回benchmark/reference guardをcpu-inplace-rms-cases.logに開始。次のturnは同じ実行handleを確認し、terminalと15JSONのmaxerrorを検査する。CPU極限goalは引き続きactive、workspace再利用と行列積のpacking/threading等の候補をまだ監査していない。

## CPU tuning handoff (2026-10-01)

- 現状commit/push・残件Issue化の指示で今回の作業をまとめる。未計測のMLP packing試作は除外。極限最適化が完了したとは扱わない。
- RMS/vector/Accelerate/trimの15ケース各3回benchmarkはexit0、15JSONを確認、最大logit誤差3.3974648e-5。vector/RMSは既定無効。初回guardでありGC/alias/任意入力の網羅的証明ではない。
- 残件: [#4 workspace](https://github.com/AtelierArith/JeffClient.jl/issues/4)、[#5 MLP配置と並列化](https://github.com/AtelierArith/JeffClient.jl/issues/5)、[#6 vector/RMS検証](https://github.com/AtelierArith/JeffClient.jl/issues/6)。owned MLP先行up*gateの極端値overflowは既定採用前に評価する。

## CPU vector gating overflow guard trial

- commit abc3750後の継続調査でgate=[-100,-90,-10,10], up=floatmax(Float32)をJuliaで実行。元のSiLU*upは[-0,-0,-1.5448093e35,Inf]、先行up*gate版は[NaN,NaN,-Inf,Inf]となり問題を再現した。
- 所有配列を書き換える前に有限入力の積overflowを走査し、該当すれば元のscalar式へinvokeでfallbackする試作をextensionへ追加。同じ入力で元の結果への一致を実行確認。追加走査の全forward性能は未確定であり未commit。


## CPU MLP packing full-forward trial

- 前goal turnはoverflow再現・保護試作・50回計測でprogress。今回はnative_mlp_weightsのCPU専用packing試作を実装し実際の0.8Bで比較した。gate/upをhcatで結合し、transpose(x)*packedでtoken-major投影、gate半分copy・up半分view、down*transpose(gate)を計算。
- 既存ProfileはAccelerate GEMMを主要サンプルとして示す。単純なgate/up結合だけで大きく改善すると推測しない。次はコピーを増やさない投影mul!とforward workspaceを検討する。overflow guardは未commitのまま残る。

## CPU forward-local MLP workspace trial

- JEFF_CPU_MLP_WORKSPACE=1（既定無効）でgate/up投影のMatrixをforward内に所有し、全層のmul!で再利用する試作。最終層は最後のtokenだけのため別1列bufferを所有。層のMLP widthが異なれば通常経路にfallback。永続cacheやtask共有は使わず、入力とresidualを上書きしない。
- このtrialではworkspace設定はコマンドで1を指定し、後からbenchmark JSONにcpu_mlp_workspace_enabledを追加。全ケース・GC/所有・異なるshapeの検証は未完了。JET/Profileをprofile-cpu-mlp-workspace.logに起動、handleは本turnの実行結果を参照して継続確認する。

- 同backendで15ケースと[1,15,2,1]再訪、各2回（2回目GC.gc(true)後）、logit guard/finite/input不変確認は全て通過、maxerror3.3974648e-5。ただし一時検証スクリプトがfinallyでNativeBackendに存在しないcloseを呼びexit1。推論の失敗ではないがclean exitを得るためclose除去してrepeat起動（cpu-mlp-workspace-validation-repeat.log）。同じhandleを次turnで確認。

## CPU workspace Cthulhu type audit

- ユーザー指定のCthulhuを実行。Julia1.13.1/Cthulhu3.0.2/TypedSyntax1.5.4、real0.8B parcel型、MLP workspace/vector/Accelerate有効。tools/inspect_native_cpu_types.jlを追加しtyped/source/対話descendを再現可能にした。
- 対話descentでnative_mlp→3引数mul!→5引数mul!→_mul!を辿った。配列型、transpose wrapper、BLAS flag、戻り値は具体型/定数。トップメニューの未使用mul!戻り値::Anyだけで型不安定と判断しない。typed IRではその値を束縛せず、降りた先の戻り値はMatrix{Float32}。
- 5対象（hidden_forward/workspace/layer/MLP/gate）のtyped IRはcpu-workspace-type-ir.log。hidden/layer/MLP/gateのBodyはMatrix{Float32}。workspaceのみUnion{Nothing,具体NamedTuple}で設定fallbackの小Union、利用branchで絞る。JET core+extension No errors detectedと整合。対象以外の任意型・全入力まで型安定を証明したとは扱わない。
- MLP workspace数値/所有検証repeat handle82653はexit0、15ケースと[1,15,2,1]再訪、各2回/GC後/input保持を通過、maxerror3.3974648e-5。永続workspaceではなくforward-localである点も含め記録。

- gateもCthulhu対話表示でextension methodのFloat32演算/Matrix戻り値を確認。TypedSyntax sourceはCore.Const(ENV)を環境変数内容まで展開するため、診断ツールのsource/descendでは子プロセスのENVを必要な設定に絞ってから表示する。型確認のdefaultはtyped IR。
- GEMMのサンプルが多いことだけでBLAS内部packingが原因とは断定しない。Float32を維持した次の候補はロード時の重みmaterialization/配置比較であり、ロード時メモリと全forward時間を測る。

## CPU weight layout trials

- 次にMLPだけpermutedimsをロード時に行いtranspose wrapperで論理weight形状を維持、native_linearのtransposeがunwrapされBLAS N指定となるJEFF_CPU_TRANSPOSE_MLP=1試作を起動。cpu-transposed-mlp-trial.json、実行handleは本turn出力を参照。まだ検証/採用未確定。


## CPU projection and layer-weight sweep benchmarks

- tools/benchmark_cpu_projections.jlを追加。real0.8B/Float32/Accelerate/Apple M4/parcel active101、最初のfull/delta層の実activationsを使ってmul!を各50回測定、出力はpreallocate。モデル全forwardやactivationを含まない。cpu-projections-101.jsonとcpu-projections-101-sweeps.json。
- 前turnは重みmaterialization/transposeMLP実測と不採用でprogress、今回も測定ツール追加と投影別/異なる重みsweepでprogress。GEMMラッパーの型やJulia heap割当をこれ以上削るより、DeltaNet headの並列化（scratch所有分離とBLAS oversubscription検証）を次に評価する。

## CPU DeltaNet head parallel trial

- cpu_delta_heads!を抽出し、複数headをworkerごとの範囲で処理する。state/full/tail scratchはhelper callが所有、headの出力領域は重ならず、q/k/v/z/weightsはread-only。threadid依存のcacheは使わず、@syncで全worker完了後にout projectionへ進む。JEFF_CPU_PARALLEL_HEADS=1かつdefault worker>1でのみ有効、既定0。



- tools/validate_native_cpu.jlを追加し、一時スクリプトの検証を再現可能にした。15ケースと[1,15,2,1]再訪、各2回/2回目GC、入力保持と以前返したscore保持を確認する。parallel8設定で起動し、成功後に同設定のProfile/JETを逐次実行する（cpu-parallel-heads-validation.log / profile-cpu-parallel-heads.log）。同じsessionを次turnで確認し、計測を並列に起動しない。

## Parallel CPU commit checkpoint

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

## CPU optimization stopping checkpoint / QKV-Z fusion not adopted

- ユーザーの「ここで潮時」「まとめましょう」に従い、新規最適化と進行中の検証を停止。検証済み・push済みの実装7ccfe19を維持する。固定Intel基準495.254ms→293.934ms（約1.685倍）、2倍目標247.627msは未達であり、達成とは扱わない。heap392,752,624→103,750,032bytes（約73.6%減）、Float32/同入力/Julia8、最終OpenBLAS1、MKLなし。M4記録と混同しない。
- QKV/Z fusion試作はCPU load時にhcatし元qkv/zを同じMatrixのviewに置換、forward-localなcombined出力をviewで分割してコピーを省いた。保持modelは約3.010638GBでほぼ不変。元版は追加41checks＋既存checksが1/8worker両方exit0、全test suite exit0、saved independent parcel参照最大誤差1.1444e-5。
- 同Intel/current flags/各30samples、非融合293.576ms/p95317.125ms、融合291.372ms/p95321.907ms、独立repeat292.594ms/p95324.978ms。改善約0.3〜0.8%、heap103,750,288→103,666,176bytes、allocs11,827→10,890。load2.239→2.529s、peak process RSS5.305→5.994GB（load/compileを含む高水位、warm scratch量ではない）。利点が小さくロード/複雑性コストが増えるため不採用。
- 元fusion版のJETはworkspaceの相関が失われたNamedTuple unionにruntime dispatch6件を報告。factoryを分離した版の30sample benchmarkは291.208ms/参照guard通過まで完了したが、その版のJET/Profile/全test再検証はユーザーの停止指示で中断。型警告が解消したとは未確認。途中runは完成済み検証として扱わない。
- 自分の試作diffをignored artifacts/cpu-tuning/qkvz-fusion-trial.patchに退避し、src/native.jl/native_cpu.jl/test/native.jl/tools/benchmark_inference.jlをapply_patchで7ccfe19へ復元。中断したjobの専用process group停止とsession exit143を確認。qkvz-fusion{,-repeat,-control,-typed}.json、qkvz-fusion-{validation,single-worker,checkpoint-tests}.log、profile-qkvz-fusion.logを保持し、再開時の判断材料とする。

## Automatic CPU defaults / removal of environment switches

- ユーザーの新依頼でJEFF_CPU_*実行時切替を廃止し、BLAS自動設定も明示承認済み。2倍探索goalはpausedのまま、この作業は設定整理であって探索再開ではない。src/ext/tools/examplesのCPU環境変数読み取りは0。READMEを環境変数なしへ変更し、performanceの旧測定・旧コマンドはb35e530以前の履歴と明示して保存。
- src/cpu_settings.jlに型の固定されたimmutable policyを定義。Apple Silicon macOSでAccelerate forwardingが有効なら既存readme最速profile（Accelerate/vector math/head parallel/MLP workspace/trim、その他portable試験flagoff）を選択。それ以外は検証済みportable profile（LV/block/norm/recurrent/projection scope/parallel heads/full heads/MLP/Delta/projection workspace/final query/residual fusion/trim）を既定化。QKV/Z fusion、Octavian/SIMD試験、inplace Delta RMSは既定不採用。
- LoopVectorizationとAppleAccelerateをstrong dependenciesへ昇格。LVは自動load、AppleAccelerateはApple Silicon macOSでのみ自動import。Linux/IntelでApple BLASへforwardしない。Julia load後にCPU policyを実際のBLAS状態から確定し、portable複数JuliaworkerならBLAS1、portable単一workerならmin(8,Sys.CPU_THREADS)、Apple profileならLBT8相当でframework-managed threadingを選ぶ。process-wide変更が他BLAS利用者へ影響する旨README/docsへ記載。forward中の変更やthread pool再設定はしない。
- 比較用切替は内部with_cpu_settingsのtask-localなscopeへ置換（公開設定APIではない）。Function/VarargはF/Nで特殊化。workerにはimmutable policyを明示伝播し、chunk/recurrent/parallel参照比較が実際に異なる経路を通ることを維持する。古い変数を設定しても無視されるregression testとscope成功/例外復元8checks追加。
- CPU environmentなし、Intel i9-9900K/Julia1.13.1/Float32/real0.8B/parcel B1L256active101/Julia8/OpenBLAS1の30warm samples median293.433ms/p95316.923ms/103,849,264heap bytes/12,922allocations/maxerror1.2398e-5（cpu-defaults.json）。元7ccfe19 repeat293.934msと同程度。policy伝播で小さなheap増があり0allocationとは扱わない。model load/compileを通常推論から除外。
- JET core+LV No errors detected、warm283.652ms/103,846,704bytes/GC0、Profile/Allocs5%取得（profile-cpu-defaults.log）。全test suite exit0（cpu-defaults-tests.log）。activation126/block18496/native651checksが1/8worker両方exit0（cpu-defaults-{validation,single-worker}.log）。JuliaFormatter整形とdiff check済み。Apple M4/Linuxの新auto-loader実機再測定は未実施、過去のM4結果の汎用保証はしない。

## 2026-10-01: Linux Xeon CPU benchmark / automatic defaults

- ユーザーのCPU計測依頼。source `fddc576013bf1f6971bc7ffdcd901c7d3c44ccab`、測定時tracked worktree clean。実機はIntel Xeon E5-2699 v3 / 18 physical cores・36 logical CPUs / x86_64 Linux / Julia1.13.1。以前のi9/macOS・M4の値とは直接速度比較しない。
- `Pkg.instantiate(; workspace=true)` で依存を準備し、`resolve_checkpoint`でpinned Jeff0.8B revision `0f212b3e72acb4dde3f7da61e925d6ab7f819990`をScratchへ取得。real weights/Float32/parcel reference case1/B1L256active101、readoutとCPU score返却込み、download/import/load/初回compile/tokenization除外。既存 `tools/benchmark_inference.jl`、30samples/evals1、別process逐次実行。
- CPU自動portable policy、Julia8/OpenBLAS1。LoopVectorization0.12.174のportable SiLU/gate/vector blocksは有効、Octavian・explicit SIMDは無効。BenchmarkTools1.8.0、OpenBLAS_jll0.3.30+0。Octaveという依存ではなくOctavianが比較用のoptional依存であり、今回の推論では使わない。
- run1: median448.356783ms / p95641.087973ms / min424.765486ms / max641.477599ms。ばらつきを受け同条件repeat: median641.8124805ms / p95799.071833ms / min438.434235ms / max803.772692ms。速いrunだけを安定した性能値として採用しない。両方heap95,672,592bytes（91.2405MiB）/12,842allocations、saved independent PyTorch reference guard通過、max logit error1.2397766e-5。
- load3.2220/3.1751s、first forward21.3870/21.1470s。retained model3,010,637,944bytes、process peak RSS4,356,997,120/4,429,197,312bytes。peakはstartup/load/compile込み、warm heap allocationとは別指標。
- hostは他workloadから隔離していない。CPU affinity0–35、CPU0 governor schedutil、process/ancestor cgroupsはcpu.max=`max 100000`・nr_throttled=0。別のCPU使用processが観測されたが、run間差の原因は未特定。環境設定を変更せず両結果を記録した。
- 集約JSON `docs/src/assets/benchmarks/cpu-2026-10-01-fddc576-linux-xeon.json`、条件・全再現commandは `docs/src/performance.md`。個別JSON/stdout/setup/download logs/Manifest/lscpu/cgroup snapshotはignored `artifacts/benchmarks/cpu-2026-10-01-fddc576/`。推論source変更はなく、実モデル参照guardと記録の整合性を確認する。

## 2026-10-01: Linux Xeon / requested Octavian chunk path

- ユーザー指定 `octavian_delta=true` / `recurrent_delta=false` をinternal `with_cpu_settings` scopeで測定。`using JeffClient, Octavian`後にscope内で既存 `tools/benchmark_inference.jl` をincludeする。mainの `initialize_cpu!()` はglobal defaultsを再初期化するが、task-local scopeは保持される。workerにもpolicyが伝播する。benchmark前のflags/extensionと終了後のdefaults復元をassertし、両process exit0。
- 同fddc576/real0.8B/pinned revision/Float32/parcel B1L256active101/Julia8/OpenBLAS1、chunk64、他policyは既定のまま。state128×128、chunk full64/tail37でextensionの寸法guardを満たし、worker-local state product2箇所が `Octavian.matmul_serial!` を使う。MLP/その他GEMMはOpenBLASのまま。Octavian0.3.29。
- requested30samples: median753.9773705ms / p95881.313247ms / min684.412186ms / max886.270485ms、114,761,392heap bytes /22,536allocations、max logit error1.1444092e-5。chunk方式を保持してoctavianだけfalseの別process逐次control30samples: median752.4623015ms /p95881.499424ms /min724.976491ms /max882.363954ms、114,706,096bytes /21,384allocations、maxerror1.335144e-5。両run saved independent reference guard通過、recorded cpu_*差はoctavian_deltaのみ。
- このpairのOctavian中央値は約0.201%高く、速度改善は確認できない。以前のdefault recurrent448/642msとの比較はrun間のばらつきとアルゴリズム変更を含むため、Octavian単独の効果と扱わない。default変更・新規最適化は行わない。
- JSON `docs/src/assets/benchmarks/cpu-2026-10-01-fddc576-linux-xeon-octavian.json`、再現commandはperformance.mdのLinux/Octavian節、個別JSON/logsと既存Octavian専用検証logはignored `artifacts/benchmarks/cpu-2026-10-01-fddc576/`。
- 既存 `tools/verify_cpu_octavian.jl` は144/144checks、exit0。state/RHS入力保持、view/Matrix/Transpose、beta0のNaN出力上書きを検証。`@code_warntype`の戻り値はMatrix{Float32}。

## 2026-10-01: CUDA.jl / ONNX CUDA verification on Linux

- Native CUDA implementation goal (in progress, 2026-10-02): user targets real ONNX median178.591ms, then requests extensive tuning and low heap allocation. Clarification: GPU0 is for benchmarks, GPU1 for debugging/validation; do not add model parallelism or dual-GPU throughput features. Added CUDA weakdep/extension, cuBLAS products/triangular fallback, recurrent Delta register-state kernel, fused conv/normalization, batched full attention, packed MLP/Delta projections, model-owned slot scratch with stream completion/locking/weak owners. Generic baseline1013.66ms/heap46.68MB, recurrent89.61ms, scratch86.59ms/heap3.88MB, batched attention74.04ms, owned buffers72.82ms/heap432784bytes/GPU5688bytes399events, inbounds72.21ms, packedMLP71.51ms, packedDelta69.86ms (all F32 B1L256active101, 5warmup/30samples, same real three-case refs, all255scores maxerr<9e-6). Every stage has artifact log; latest native-packed-delta.log. test/cuda.jl135 checks passes on GPU1 including independent21 mask/length probes in cuda_reference.json, GC/revisit/ownership/concurrency/failure/device restoration. GPU profile: initial scratch version projectionSGEMM33.95ms/Delta29.46ms; aftercopies eliminated tune remaining. Register/row sweep didn't show stable large gain, not adopted. tools/benchmark_native_cuda.jl added, not yet final benchmarked. @code_warntype/JET job and AllocCheck setup underway; full CPU suite/docs/final measurements still outstanding. Root workspace Pkg.resolve needed to refresh ignored Manifest CUDA extension metadata before tools+secondary LOAD_PATH CUDA import activates extension. Goal remains active; do not treat current implementation as final audited completion.

- 2026-10-02 real Jeff ONNX CUDA計測: pinned0.8B revision0f212b3e72acb4dde3f7da61e925d6ab7f819990、tools/_onnx_export.pyをPythonCall経由、USE_HUB_KERNELS=NO/inspect.unwrap torch Deltaでexport。onnx/onnxruntime/onnxscriptをignored python-deps targetへ導入し既存Python環境は変更せず。CPU export maxerr6.68e-6。CUDA全255logits/3caseがatol=rtol=2e-3で一致、maxabs0.003813/0.002560/0.001807。verbose配置CUDA19921nodes/CPU24/52Memcpy、GPU-onlyではない。verboseを止めて別process、5warmup/30samples、F32 B1L256active101 parcel、入力転送/readout/CPU返却/同期込みmedian178.591ms/p95181.143ms。load104.8s/tokenization/compile/decide除外。RTX3060 CUDA6.4.1/runtime12.8 ORT.jl1.4.0/ORT1.20.1。JSON docs/src/assets/benchmarks/jeff-onnx-cuda-2026-10-02.json。export_driver.py/benchmark_jeff.jlと全logs/modelはignored artifacts/cuda-validationに保持。下記fixture時点の実モデル未検証はこの測定で更新。

- 2026-10-02再検証: driver580.178.04でnvidia-smi正常、CUDA6.4.1 functional=true、同期CuArray roundtrip成功。runtime13.4ではONNXRunTime1.4.0のversion checkで拒否。ignored artifacts/cuda-validation/envのみでCUDA.set_runtime_version!(v"12.8")、Julia再起動後runtime12.8.0でidentity fixtureとORT付属MatMul fixtureのONNXBackend(:cuda)成功。Float32 2×3×4積がJulia積と一致、decideの2行とも期待choice=d。verbose logで全1ノードCUDAExecutionProvider配置を確認。ログonnx-2026-10-02.log/onnx-provider-2026-10-02.log。実Jeff checkpoint推論・速度は未検証。下記10/01のdriver failureは過去の状態。

- ユーザー依頼のCUDA検証。root/tools依存は変更せず、ignored `artifacts/cuda-validation/env` へlocal JeffClient/CUDA6.4.1/cuDNN6.4.1を導入。ONNXRunTime1.4.0はroot workspaceと同version。runtime自動選択13.4.0。
- 実機RTX3060×2、loaded NVIDIA kernel module580.173.02、system libcuda/libnvidia-ml580.178.04。`nvidia-smi`はNVML driver/library version mismatch。CUDA.functional()=false、functional(true)/CuArray broadcast→synchronize→ArrayのprobeはCUDA error804 COMPAT_NOT_SUPPORTED_ON_DEVICE、device countはerror3 NOT_INITIALIZED。
- `using CUDA` + cuDNN import後、ONNXBackend(:cuda)のfixture session作成はCUDA not functionalで失敗。同fixture CPU sessionは非正方Float32入力2×3のidentityと一致。実モデル0.8BのCUDA forward/exportは行っておらず、GPUモデル対応済みとは扱わない。
- ONNXRunTime1.4.0 sourceのruntime範囲は>=12.0/<13.0で、自動選択13.4も不適合。再検証にはdriver整合性とCUDA12 runtimeが必要。システムdriver変更/再起動は行わない。
- NativeBackend(fixture; device=:cuda)も実際に呼び出し、CUDAimport済みでもUnsupported native device Val{:cuda}で失敗。src/native.jlとProject.toml/extにCUDA native extensionはなく、ONNXCUDA providerとNative Julia CUDA対応は別事項。
- 集約JSON `docs/src/assets/benchmarks/cuda-2026-10-01-fddc576-linux.json`、解説docs/src/inference.md。setup/probeログ・Manifest・original reportはignored artifacts/cuda-validation/に保存。

## Linux Python comparison and full-sequence benchmark (2026-10-01)

- Original Python CPU through PythonCall initially failed because installed FLA selected GPU Triton. Process-only `inspect.unwrap` selection of Transformers' original torch chunk/recurrent functions plus `USE_HUB_KERNELS=NO` succeeds; no Python source/environment edits. Python3.12.3/PyTorch2.14.0+cu130/Transformers5.17.0/FLA0.5.2, Jeff f067882.
- Same real Float32 B1 parcel case: Python full256 median808.495ms/p95812.716; cropped101 median432.196/p95434.206. Fresh default Julia8/BLAS1 (computed101) median652.822/p95813.772. The apparent1.24x padded-Python ratio is unequal work.
- Default Julia1/BLAS8 median945.116/p95975.633; Octavian chunk1/8 median974.025/p951027.077. Both slower than corresponding8/1 trials; no general threading conclusion beyond this host/input.
- Added CPU-only `tools/benchmark_inference.jl --python-reference`: task-local trim=false, final_query=false, new final_token_only=false, recurrent=false, Octavian=false, chunk64. New full-sequence branch computes all final MLP/RMS columns before last-token readout; production default final_token_only=true remains. Full suite passed including52 new independent-reference/GC/workspace/ownership/restoration checks. JuliaFormatter applied.
- Full256 reference profile Julia1/BLAS8 median1797.095ms/p951969.846, heap378678568bytes/20951allocations, error1.2397766e-5. Python808.495ms =>Python2.22x faster. Unpinned affinity0-35, unisolated host, sequential separate processes; kernels/batching/harness still differ. Raw JSON and source hashes saved in docs asset.
- Results consolidated in docs/src/profiling.md; performance.md now describes configuration and links to results. Historical CPU/Metal results and allocation investigations retained in collapsed sections.
- Full-sequence fixture `@code_warntype` returns Matrix{Float32}; JET optimization report: No errors detected. Warm fixture cumulative Julia heap33376bytes measured separately.

## Apple Silicon matched comparison / standard driver (2026-10-01)

- 旧M4 CPU/Metal性能表・JSON・README/PLANの速度比はユーザー指示で破棄。新結果はdocs/src/profiling.mdとassets/benchmarks/m4-2026-10-01-matched.json。Apple M4/10cores/24GiB/macOS27.0.1/Julia1.13.1、F32、real0.8Bの同一weights、parcel B1/L256/active101。全256列のMLP/最終residual/RMS、readout255列、CPUscore返却。load/compile/tokenization除外。separate processes/30samples/evals1/two fresh runs、host非隔離・affinityなし。
- strict CPUはJuliaworker/BLAS/Accelerate1、PyTorchintra/inter1。Python545.492/528.398ms、Julia462.219/461.751ms（1.14〜1.18倍）。maxerror Python0/Julia6.67572e-6、全255列guard通過。8worker/Accelerate automatic10は362〜364ms、PyTorch8は3033msだがスレッド予算が異なるためequal-thread比としない。
- GPUは全実装でCPUpreparedinputのupload/完了/CPU返却込み。PyTorchMPSF32 343.291/344.179ms（maxerror1.00136e-5）、Metal full-sequence adapter197.249/196.869ms（6.67572e-6）。Metalは15cases×2passes、GC/padding/maskholes、scalar indexingoffで通過、max3.361702e-5。大きなMetal/PyTorch不一致は確認されない。
- MLX0.32.3/mlx-lm0.31.3はstock Q/K RMSのepsがmean-square1e-6で、Transformersのsum-square1e-6と異なる。stockguard失敗(max0.0132904)。process-localでeps/headwidthへ揃え、weightsをsanitizeの+1より前にF32cast。元package未変更。adaptedGPU136.827/137.028ms、error1.52588e-5。CPU2321.826ms、error7.15256e-6、frameworkmanagedで1threadとは未確認。MLXはbenchmark1caseのみ検証。
- Apple Silicon macOSの標準再現はtools/mac-M-series.sh。デフォルトCPU1vs1+MPSF32vsMetalを逐次30samples×2process、--include-auto-cpu/--mlxは別枠、--setup-pythonはuv環境を準備。JSON/logs/runtime/hash/summaryをignored artifactsへ保存し、sample/thread/shape/compute lengthをreport側でguard。jeff-metal-performance skillもこのdriverへ更新。benchmark-local source/method adaptersは本体のデフォルトを変更しない。

## Linux CPU Python gap investigation / comparison driver (2026-10-01)

- Before-change Profile/Allocs and phase probes are in ignored artifacts/cpu-python-gap/. Full256 Julia1/BLAS8 attention dominated: phase medians Delta1248.79ms/full190.78ms; MLP activation41.38ms. PyTorch's one-forward profiler measured807.275ms, GEMM553.028ms, batched GEMM59.327ms, SiLU16.25ms. These diagnostic stage sums are not benchmark medians. PyTorch uses MKL2024.2/oneDNN and batched/contiguous kernels; Python interpreter overhead is not the dominant measured cost.
- Portable SIMD guards previously sent entire blocks with zero/exceptional lanes through scalar exp. Added bounded SIMD for eligible lanes, scalar repair for exceptional lanes, zero-block handling and parallel disjoint blocks; signed zero/NaN/Inf/subnormal tests retained. Activated Delta z once before head loops. Packed Q/K/V/gates as (width, sequence, head) for contiguous chunk GEMMs; retained independent-reference, workspace and alias tests. Full-head scores now use guarded in-place portable softmax with scalar subnormal repair and nonfinite fallback.
- Sequential pinned cores0–7, Julia8,0/GC1, full256/chunk64, Octavian/recurrent/final-query/final-token/trim disabled: 30 samples/10 warm-ups OpenBLAS1 median875.3305985ms/p951101.118259; MKL1 median809.808807ms/p951081.069995. These development trials alone do not prove Julia faster than Python. Old clean Julia8/BLAS1 trial1777.0909005ms used2 warm-ups, so label its differing protocol. Artifacts packed-softmax-{openblas,mkl}-full-warm10.json.
- Packed MLP gate/up trial measured881.795373ms/p951082.983544 with OpenBLAS1, versus preceding875.3305985ms. No improvement established; removed packed_mlp policy, combined weights/buffers and view gate dispatch. Keep artifacts/packed-mlp-* for investigation rather than adopting the trial.
- Added tools/linux-cpu.sh and Julia adapters/config/report for sequential alternating fresh processes with equal physical-core affinity, Float32/full sequence, explicit warm-ups and all readout validation. --threads1 offers strict comparison; default8 uses Julia workers8/BLAS1 versus PyTorch intra8/inter1. Optional --mkl-project points to a separately prepared environment; root runtime dependencies remain unchanged. New project skill: .agents/skills/jeff-linux-cpu-benchmark/SKILL.md.
- The real checkpoint's decision_config max_options is254 but its trained readout/reference has255 columns. A PreparedBatch count of254 masks the last score and validates only254. Linux adapter now sets count=model.readout.out_features (255), validates all scores and records validated_options/output_shape. Do not claim a max_options-based guard validates all255. FLA adaptation needs USE_HUB_KERNELS=NO before PythonCall initializes Python; setting Julia ENV afterward is too late for Python's imported environment.
- Linux adapter smoke run: all255 guards passed, PyTorch3 samples/2 warm-ups median810.591ms; Julia3 samples/2 warm-ups median878.418ms. Validated summary under artifacts/cpu-python-gap/linux-driver-smoke/. Not a published steady-state measurement. Fixed Julia1.13 include_string binding world age with a setup-only invokelatest call; BenchmarkTools seconds=Inf throws Int64(Inf), so adapter uses finite3600s and report requires exact requested sample count.
- First complete new-driver MKL comparison (30samples/10warm-ups/2fresh processes/core0–7/alternating order) passed all255 guards: Python803.2209705/p95805.711175 vs Julia801.3039985/p95985.742095; repeat Python812.480/p95813.762 vs Julia832.177778/p951038.755694. A 0.24% faster first Julia median is not a stable win; second median is slower and both tails worse. Saved docs/src/assets/benchmarks/linux-cpu-2026-10-01-matched-mkl.json and updated performance.md transparently; goal remains active. Optional MKL0.9.1/MKL_jll2025.2 verified in isolated Manifest, PyTorchMKL2024.2.
- Current packed-softmax-current-profile.log: full256 @code_warntype Matrix{Float32}, JET No errors detected, warm diagnostic1.162778876s/324399376heap bytes/GC0. Main sampled allocation sites include residual creation, native_rms, full-head outputs/attention intermediates. This diagnostic uses only initial warm-up, not the final benchmark's10 warm-ups; do not report its latency as a steady benchmark. Added per-sample Julia GC timing to the adapter for subsequent trials; prior published runs do not contain that series.
- Kept legacy benchmark CLI default warm-ups at2 to preserve Mac adapter pairing; Linux adapter explicitly passes10. Linux config now records dependency versions and Manifest hashes in future runs. Report checks all255 validation, fullseq/flags, affinity, dtype, warm-ups and complete run counts; deliberate malformed-result probes exercise rejection rather than relying solely on success flags.
- Follow-up after user's allocation hypothesis: gc-series-before-rms-reuse.json records30 MKL1/Julia8 full256 samples, median790.923545/p951006.677879, 324399376heap bytes. All samples had GC (typically2–6ms); the slowest1110.331049/1006.677879ms samples included141.473726/134.822394ms GC. Median after subtracting reported GC787.644594ms is a diagnostic only, not an inference benchmark; GC accounts for part of those tails, not the entire excess.
- New RMS/residual reuse trial in src/native_cpu.jl: allocate normalized/residual matrices per forward, pass typed buffers through cpu_hidden_forward, reuse pre/post normalization after attention's workers complete, compute full final RMS into the buffer for fullseq mode. Input hidden supplied by the caller is preserved at the first layer; subsequent residual writes reuse only forward-owned buffers. CPU RMS helper copies weights if output aliases them. Added numeric/input/weight preservation, in-place and weight-alias/dimension-guard tests. Initial full suite passed the implementation; new direct RMS tests need a fresh verification pass because they were edited during that process's run. Follow-up benchmark running at gc-series-after-rms-reuse.json; do not assume adoption/performance win before reading it and rerunning JET/guards.
- Before RMS trial: current-full-tests.log exit0; current-vector-one-worker.log and current-vector-eight-workers.log exit0 (softmax22, parallel activation20, packed-layout111 included); current-docs-build.log exit0. CLI/report rejection probes passed, including partial255→254 validation, different affinity/dtype/warm-ups, validation=false and a missing run. Future changes require scoped revalidation.
- RMS reuse benchmark gc-series-after-rms-reuse.json passed all255 guard (max1.04904175e-5): heap249627968bytes versus324399376 (23.05% reduction), allocations33848 versus34402. Median790.0729825ms versus790.923545 (~0.1%, no material median improvement); p95971.405937 versus1006.677879 (individual trials, not stable improvement proof). GC median4.4463715ms vs5.412066;4 samples >20ms GC vs2 before, and p95 sample971.4ms had only3.313ms GC. GC explains some spikes but is not the only cause. Goal is still unmet by the fresh paired trials. Limit scratch normalization to portable-vector-math policy, preserving Apple's existing normalization path. Fresh full tests and JET/Profile checks are in progress; do not assert their outcome until terminal logs are read.
- Fresh RMS trial checks completed: rms-reuse-complete-tests.log exit0, including39 new RMS numerical/input/weight alias checks and all existing native/reference tests. rms-reuse-profile.log exit0, @code_warntype Matrix{Float32}, JET No errors detected, measured heap249628016bytes. That profile overlapped fixture tests; use its type/allocation evidence, not its elapsed time or sample proportions as an isolated performance baseline. Larger remaining allocation sites are full attention, head outputs and RoPE; a separate isolated profile is appropriate before the next full-attention workspace experiment.
- Full attention now owns Q/gate, K/V, normalized Q, projection, mask and per-worker score/value buffers for one forward; layers reuse them only after workers finish. RoPE tables are computed once per forward; mask values are overwritten each call. `gc-series-full-workspace.json` passed all255 logits (max1.04904175e-5), with97363280heap bytes versus249627968, median794.927215/p95925.062679ms. The large memory reduction alone did not improve the median materially. Small-fixture tests poison all scratch, revisit masks at the same length, vary RoPE width/length, exercise serial/parallel execution and GC.
- Enabling the existing owned in-place Delta RMS path in the portable policy reduced heap further to58739024bytes. `gc-series-full-workspace-inplace-delta.json` median766.207368/p95804.539332ms, with no >20ms GC samples and median GC0. Adopted `inplace_delta_rms=portable`; Apple's policy remains false. Julia1/BLAS8 diagnostic full-workspace trial1178.9582/p951221.0645ms did not justify changing the8-worker/BLAS1 partition.
- After these changes, `full-workspace-inplace-default-tests.log` exit0 includes144 full-attention workspace and39 RMS checks. Isolated `full-workspace-final-profile.log` exit0: @code_warntype Matrix{Float32}, JET No errors detected, warmed diagnostic58739024heap bytes. Its short-warm elapsed time is not a steady-state benchmark. Final paired comparison is separate.
- Linux timing now forces full GC after cold validation and **before** ten genuine untimed forwards for both implementations, leaving automatic GC enabled during timing and forcing no per-sample collection. Post-warm collection could disturb the warmed CPU state; the final paired comparison uses the corrected common protocol rather than relabeling earlier results.
- Final matched MKL driver completed (linux-final-mkl/,30samples/10warm-ups/3fresh processes/core0–7, alternating order). All255 guards passed withmax1.04904175e-5. Python medians812.8836385/809.8490945/805.7775280ms; Julia767.6575920/771.4192525/795.9745345ms, speedup1.0589/1.0498/1.0123. Julia p95789.467262/795.555048/824.219088 versus Python813.567330/812.308831/807.168165: all medians improved, third tail did not. Reported Julia GC median0/max12.633627/13.076994/16.939991ms, no >20ms sample, slowest samples GC0. This verifies this optional-MKL median target, not a tail guarantee or default-OpenBLAS win. Exact timing/GC series and hashes published in the updated matched-MKL JSON.

- Final default OpenBLAS comparison (linux-final-openblas/,same cores/full256/30samples/10warm-ups/2fresh processes) passed all255 guards: repeat1 Python806.360969/p95808.247431ms vs Julia848.217362/p95867.577712ms; repeat2 Python809.922706/p95814.285923ms vs Julia841.139490/p95874.878904ms. Default OpenBLAS remains3.9–5.2% slower by median on this host/input; keep that limitation alongside the optional-MKL win. Published matched-openblas JSON preserves timing/GC/provenance.

- Final checks: JuliaFormatter completed without changing measured kernel/benchmark source hashes; final-vector-one.log and final-vector-eight.log both exit0, including exceptional SIMD lanes, softmax/subnormals,144 full-attention workspace checks,39 RMS checks and independent-reference/ownership tests. final-docs-build.log exit0 with CI=false (build only). Driver shell syntax/help, both skill validators and git diff --check pass. Earlier full suite and isolated JET checks cover the same final inference source.

## Reusable CPU project skills (2026-10-01)

- `.agents/skills/jeff-linux-cpu-benchmark/SKILL.md` describes the Linux driver, matched thread/input/GC conditions, complete-readout guards and evidence publication. `.agents/skills/jeff-cpu-performance/SKILL.md` describes the repeated Profile/Allocs → owned workspace experiment → numerical/type checks → fresh matched comparison loop. The skills reference each other and reuse repository tools; measured results remain here and in the docs. Both skill packages passed the skill-creator validator.

## Native CUDA final implementation (2026-10-02)

- CUDA optional extension now implements NativeBackend safetensors inference, with recurrent Delta, batched full attention, packed projections and model-owned scratch. GPU 0 benchmarks, GPU 1 debugging; no multi-GPU model splitting. Tested CUDA6.4.1/runtime12.8/cuBLAS12.8.4/Julia1.13.1/RTX3060. cuBLAS direct gemmEx call mirrors this CUDA version and uses retained device scalar coefficients; revalidate other releases.
- Full B1/L256/active101 strict Float32 pipeline: 30 samples/5warmups median70.4345425ms/p9571.1009665 vs ONNX178.590797ms (2.54x). Warm GPU allocations0/0bytes; Juliaheap176064bytes typical, GC0. Layer-category scratch reuse reduces retained workspace971119620→79764484bytes, using hidden pingpong copies to preserve lifetime. GPU pool active4.099GiB/reserved4.125GiB is separate and includes constructor/model allocations.
- Optional JEFF_CUDA_TRIM_PADDING=1:101computed tokens, median34.499957ms/p9535.03712685, heap172432bytes, GPUalloc0, scratch30971724bytes. Default remains fullsequence. Only leading zeros removed; retain interiorholes.
- CUDA.@profile requires explicit show in scripts: final73.66ms trace GPUbusy68.55, primarySGEMM32.52, delta28.44ms. Do not add synchronization host time to GPUtime. Launch/register and multirowwarp trials rejected for lack of improvement. Raw logs ignored artifacts/cuda-validation; published JSON in docs/src/assets/benchmarks.
- Final tinyCUDA184checks, real15cases/bothtrim150checks, CPUfullsuite passed; current JET logits/delta/full No errors. Correctness includes sequence1–512, batch, same-length different masks, lastonly masks, GC, retained results, concurrentcalls/failure recovery/device restoration. CUDA scalar indexing disabled. Native CUDA tests are optional hardware suite.
- AllocCheck registered0.2.6 incompatible with CUDA6 GPUCompiler2.9. Isolated dev commit c1588c51b23cd842cdf4f5cef2f60877ecde3dfa supportsGPUCompiler2 without downgrading. Static scratch inspection reports coldalloc/runtime paths; warmedCUDA.@timed/Profile.Allocs remain authoritative for actual allocation traffic. No runtime dependency added.
