# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog], and this project adheres to
[Semantic Versioning].

## [Unreleased]

## [0.9.0] - 2026-08-08

### Added

- The `Optimizer` trait with `Sgd`, `Adam`, and `AdamW`: a
  training-step strategy is a uniform, object-safe slot the loop can
  hand any implementation — deliberately an open trait, not a closed
  enum, so custom optimizers have the same standing as the built-in
  ones. Hyperparameters are single-value payloads written at the
  call site; Adam carries its moments as `Field`s and its
  bias-correction powers as payloads (exact, no `powf`); AdamW
  applies decoupled decay under a structural default policy (rank
  two and above decays; biases and norm gains are spared) with a
  `step_where` predicate override. Optimizer steps are pure field
  algebra: identical runs are bit-identical, and fields from
  `recorded_gradients` drive the same trajectory as the engine's
  backward, held by tests.
- `Network::update_each`: the identity-aware update — the rule
  receives the parameter's `Value` besides its payloads, so
  per-parameter policy (selective decay, clipping, logging) reads
  the parameter's symbol, shape, or rank at the call site. `update`
  now accepts `FnMut` and delegates to it.

### Changed

- `Field::scale` takes a single-value factor and spreads it to each
  entry's shape through `broadcast_to`-style broadcasting — the
  scalar arithmetic optimizer state needs. Scalar fields scale
  exactly as before; tensor fields gain the case that previously
  panicked. The factor is now passed by reference, and the method
  requires the `Tensorial` payload contract.

- `Network::differentiate(loss, wrt)`: reverse-mode differentiation
  as a tape-to-tape transform. Gradients record as ordinary computed
  nodes — compilable, emittable, readable, and differentiable again
  (higher-order derivatives work by re-application; relu Hessians
  are exact zeros). The transform runs the engine's own derivative
  rules over a recording payload, so derivative knowledge cannot
  fork, and it mirrors the engine scan's seed and accumulation
  order: a compiled plan over `[loss, gradients...]` reproduces
  `Evaluation::backward` bitwise, held by per-variant closure tests.
- `Evaluation::recorded_gradients`: assembles the update direction
  from recorded gradient values — the bridge from `differentiate` to
  `Network::update`, so a training step is one forward run of a
  compiled `[loss, gradients...]` plan with no backward pass. The
  `makemore_mlp_compiled` example is that loop: bit-identical to
  `makemore_mlp` under matched seeds (the closure suite pins the
  routes bitwise), at speed parity and a measurably lower memory
  peak, because forward-only liveness frees what the gradient
  computation no longer needs.
- `Activation::gain` and `init::scaled`: the principled link between
  a layer's nonlinearity and its initialization. Each activation
  states the standard factor by which it shrinks a unit-variance
  signal, and the gain-parameterized fan initializer compensates it —
  `init::scaled(seed, activation.gain())` is the general form behind
  the named classics, which stay frozen (`kaiming` is the relu gain;
  seeded outputs never change).
- `Activation::Sigmoid`, `Activation::LeakyRelu`, and
  `Activation::Elu`, with the public `Activation::express` that
  records each variant's expression: the new three are short
  compositions with stable spellings — sigmoid through the fused
  `tanh`, leaky relu and ELU through `maximum` with correct
  subgradients at zero and no overflow at finite extremes — and
  their gradients are the chain rule, closed under `differentiate`
  like every composition.
- `Value::step`, `Value::fold`, and `Value::scatter`: the three
  adjoints that close the op set under differentiation (the
  `maximum` family's locally constant mask, `unfold`'s adjoint, and
  `gather`'s), each with its StableHLO lowering — `step` as
  `compare` plus `select`, `scatter` and `fold` as contractions —
  and emission conformance coverage, including a differentiated
  module in the E2 shape verified against the reference
  interpreter.

- `Value::logsumexp` as a fused operation: the max-shifted reduction
  is finite for every finite operand, with the softmax as its
  gradient, replacing the composition over `log_softmax` that
  returned `inf` once finite logits differed by more than the
  representable range. It lowers to StableHLO as its shift-form
  decomposition and joins the emission conformance suite.

### Changed

- `cross_entropy` composes the expanded form
  `((targets.sum_along(1) * logsumexp(logits)).sum() -
  (targets * logits).sum()) / targets.sum()`: exact mathematics,
  and no term can evaluate `0 * -inf` into `NaN` for finite extreme
  logits. The targets' domain (finite, nonnegative, positive total
  mass) is now documented. Loss values may differ from 0.8.0 in the
  last bits, as any re-associated float expression may.

### Fixed

- The 0.7.0 deep-audit invariant batch: plans take one snapshot for
  validation and execution and reject a shorter sibling that does
  not contain their graph prefix; scalar payloads reject recorded
  shapes they cannot carry, `backward` checks the recorded target
  shape besides the payload's, and debug runs assert every rule's
  output shape against the recorded column; `counted`, `selection`,
  and the private constructors prove the tensor invariant at
  construction; `scatter` validates the adjoint contract instead of
  silently discarding gradient rows; a single-window `unfold` no
  longer overflows its unused stride in debug builds; the backend
  seams check the length of every `Elementary::map`/`gemm` answer;
  and the CUDA pool loads `cudaFree`, frees above-cap buffers on
  return, returns buffers through an RAII loan on every error path,
  and caps its parked bytes.

## [0.8.0] - 2026-08-05

### Added

- StableHLO emission, the crate's first exit to the XLA world:
  `Plan::emit_stablehlo` serializes a forward plan as a textual
  StableHLO module — parameters then inputs as `@main`'s arguments,
  the readable set as the result list, leaves as dense constants.
  Lowering is near-1:1 over the whole op set; the fused
  `log_softmax` decomposes into its stable shift form, the one-hot
  `gather` becomes a `dot_general` against the selection (which
  crosses the boundary as its dense matrix), and `unfold` lowers to
  a static gather as a documented completeness fallback. Matched
  window-GEMM fusion groups raise to `stablehlo.convolution` — the
  pattern library earning twice, fused executor at home and the
  richer op abroad. A typed builder owns every fragment of MLIR
  syntax; nothing heavier than string building enters the crate.
- Emission conformance, two tiers riding external toolchains the
  crate never links: `POORGRAD_STABLEHLO_VALIDATOR` names a parser
  and `POORGRAD_STABLEHLO_EVALUATOR` an executor (scripts under
  `tools/` serve both from any Python with `jax`), and the suite's
  round-trip and execution tests check every emitted module against
  the plan's own results, passing vacuously without a toolchain.
  Verified beyond the reference interpreter on real backends:
  compiled XLA-CPU runs the emitted batch-8 CNN probe eleven times
  faster than the plan (0.24 against 2.6 ms), and Apple's
  experimental `jax-metal` plugin runs all five conformance modules
  on the GPU within the oracle envelope. Numbers and readings in
  ACCELERATION.md.
- `broadcast_to` and `broadcast_pair`: explicit broadcasting under
  the right-aligned NumPy rule as composites over the named
  expansions — the target shape is always written, never inferred
  by an operator, and the gradient is the chain rule over the
  existing adjoints.
- `concat` and `stack`, the designed route: `concat` sums each
  value zero-padded to the combined extent at its offset (each
  operand's gradient is its own `narrow` window back), `stack`
  lifts through `unsqueeze`. Consumer-shaped tests close the
  transformer rung's other gaps by composition: masked axis-aware
  softmax is a broadcast additive mask before the existing axis
  softmax, and multi-head attention is a loop of rank-2 heads
  joined by `concat` — no batched matmul.
- The `makemore_transformer` example — the attention act: a
  one-block pre-norm transformer over eight characters of context.
  The batch packs its samples into one token row so each head's
  attention is a single rank-2 matmul pair under a block-diagonal
  causal mask (the sequence-packing idiom); heads join through
  `concat`, prediction rows come back through a one-hot `gather`,
  and `RmsNorm` feeds both residual branches. Mean minibatch loss
  2.205 against the MLP act's 2.2450, on a 179-node tape, 5000
  steps in 12 s.
- Elementwise map kernels on the `metal` backend: `exp`, `ln`,
  `sqrt`, and `tanh` as one-thread-per-element GPU kernels in the
  same compiled library, pooled buffers, and poison contract as the
  GEMM path. Measured on the M1 Pro, the GPU passes the scalar path
  near 128k elements and vForce near 512k (2.7 against 1.2 Gelem/s
  at 8M), so the map chain runs Metal first — the reverse of the
  GEMM order — with a size gate that adapts to whether `accelerate`
  is compiled behind it.
- The `broadcast` bench group, measuring elementwise operations
  over broadcast views against their materialized twins.

### Changed

- Broadcast views compute at slice speed: binary elementwise
  operations walk same-shape dense operands by innermost runs (unit
  stride as a slice, zero stride held for the run), and elementwise
  maps over a broadcast view transform only the distinct elements
  and keep the layout — a view in, a view out, and the backend seam
  reads the contiguous window. Bias-style adds over 2M elements
  went from 0.17 to 7.6 Gelem/s; a transcendental over a broadcast
  row computes its 1k distinct elements instead of 2M.
- Reshapes that only insert or remove extent-1 axes keep strided
  views as views, so a multi-axis `broadcast_to` records no
  intermediate copy: squeeze and unsqueeze of a broadcast view are
  layout edits, not materializations.

### Fixed

- A transposed view's elementwise map could reach the backend seam
  through the new window path (its window is exactly as wide as its
  volume), silently replacing the documented bitwise scalar
  fallback for non-contiguous views under `accelerate`. The window
  path now requires a strictly narrower window: only broadcast
  views, which compute fewer elements, earn staying views.

## [0.7.0] - 2026-08-04

### Added

- The `makemore_mlp_batchnorm` example — makemore's third act: the
  character MLP with its hidden preactivation batch-normalized
  before the tanh, the hidden bias retired in favor of the learned
  shift, running statistics maintained in the loop from the batch
  statistics the training plan's keep-set exposes, and the
  single-row sampling twin fed those estimates per draw. Final
  loss matches the plain MLP at this shallow depth, as the lecture
  it follows predicts: the norm buys robustness, not loss.
- Window-GEMM fusion, the plan tier's first pattern: plans
  recognize the canonical im2col chain feeding a `matmul` and
  execute it as one `Tensorial::windowed_product` call, never
  materializing the chain. Matching is structural and
  provenance-blind, keep-set nodes are fusion barriers, and fusion
  follows the plan's memory posture — forward-only plans always
  fuse, compact training plans fuse (backward rebuilds patches
  with one `windowed_patches` fast fill, bit-identically), and the
  default retain-all training plan stays unfused, because per-step
  patch re-allocation in backward measured as a peak-RSS
  regression on the deeper consumer. Profile-driven: the CIFAR-10
  step was ~50% strided-view iteration and materialization, under
  2% elementwise arithmetic — which also retired the planned
  elementwise-chain and `MaxAlong` fusions as worthless. The MNIST
  example's compact training dropped from 114.6 to 106.8 ms/step
  at unchanged memory, with byte-identical output.
- `Tensorial::windowed_product` and `windowed_patches`: the im2col
  product and its patch-matrix half as payload calls, with composed
  defaults that are the bitwise references and a `Tensor` fast path
  that fills patches in contiguous runs instead of the general
  odometer walk. The descriptors are the method arguments, so
  payloads and backends never see graph structure.
- Rematerialization, opt-in via `compile_training_compact`: the
  plan drops its large intermediates (im2col patches, padded
  copies, pooling lanes — the allocator's page-returning size
  class) right after their last forward consumer, and `backward`
  recomputes them on demand, memoized with prompt eviction and
  bit-identical gradients. The trade is explicit because it does
  not always win — measured at 9% less peak RSS for 22% more step
  time on the MNIST example, but negative on the deeper CIFAR-10
  stack, where gradient cotangent buffers dominate; the default
  `compile_training` stays retain-all. `describe` reports the drop
  set and the remat peak either way.
- The retention contract: every operation now declares which
  payload values its derivative rule reads (both operands for
  `mul` and `matmul`, its own output for `tanh` and `log_softmax`,
  the selection for `gather`, nothing for the view family — whose
  backwards read shapes that placeholders answer). Training plans
  use it to compute and report their memory *floor* — on the MNIST
  convnet, 3.3M of 12.3M elements (3.75x) is releasable with
  gradients still bit-identical, which tests prove by forcing the
  releases. Training runs do not execute the releases by default:
  A/B measurement showed per-step mid-run freeing regresses peak
  RSS under the system allocator (fragmentation), so the floor
  awaits rematerialization or arena reuse — while forward-only
  plans keep executing theirs, where the win is measured.
- `Plan`, `Network::compile`, and `Network::compile_training`: the
  first lowering tier. A plan is a compiled execution schedule —
  dead-node elimination against declared targets, a keep-set that
  alone answers reads, and (for forward-only plans) buffer liveness
  that frees every intermediate after its last consumer. Plan runs
  are bit-identical to the interpreter's, survive every `update`
  generation (compile once, train forever), and refuse `backward`
  unless compiled for training, so freed buffers can never leak
  into gradients. `Plan::describe` renders the schedule: per-node
  liveness spans and the static peak-live-volume estimate. The
  MNIST example runs on plans — compile-once training plus a
  forward-only probe whose liveness cuts its live volume 6.8x
  (28M of 191M elements) and the process peak RSS by 31%, with
  byte-identical output.
- The `cifar10` example: a three-stage VGG-style convnet on real
  32x32 color images, the plan tier's pressure consumer. One
  training plan serves all 2000 generations, and the forward-only
  probe plan holds the 500-image accuracy probe's live volume 8.8x
  below retain-all. Reaches 65.2% test accuracy (chance is 10%)
  in about 13 CPU minutes at 392 ms/step; downloads and caches the
  binary archive on first run.

- `Tensorial::unfold` and `Tensorial::fold`: single-axis sliding
  windows (torch semantics, with a dilation parameter) as a strided
  view over the shared buffer, and their adjoint — each source
  position sums its own window contributions in window order, so
  folding is deterministic under any evaluation strategy. The
  substrate for convolution and pooling. Breaking for custom
  payload implementations, which must add both methods.
- `Value::pad` and `Value::unfold`: the corresponding recorded
  operations. `pad` places a value inside zeros along one axis and
  is `narrow`'s adjoint (each is the other's gradient rule);
  `unfold` records the sliding-window view with `fold` as its
  gradient, so overlapping windows accumulate correctly.
- `conv2d` and the `Conv2d` layer: 2-D convolution as a composed
  formula — padding, two unfolds, and an im2col reshape feeding one
  rank-2 `matmul` on the accelerated GEMM path — with stride and
  symmetric zero padding, torch-shaped weights, and the gradient
  from the chain rule alone.
- `max_pool` and `average_pool`: spatial pooling over the same
  window view; the maximum folds with the left-biased binary
  `maximum`, so ties route deterministically to the earliest
  window position.
- The `mnist` example: a LeNet-style convolutional network trained
  on MNIST through the composed convolution and pooling formulas —
  the convolution rung's first consumer. It downloads and caches
  the IDX files on first run and reports test accuracy, per-step
  time, and the loss chart.
- `Network::forward_for`: the target-sliced run — it evaluates only
  the ancestors of the declared targets, leaving every skipped slot
  an O(1) shape-correct placeholder that `of` and `backward` refuse
  to answer with, so skipped reads fail loudly. Sliced gradients
  drive `update` soundly (a parameter outside the closure receives
  its true gradient, zero), and results are bit-identical to full
  runs. With the training and evaluation expressions sharing one
  tape, the MNIST example dropped from 517 to 95 ms per step
  (5.4x) with an unchanged 98.22% test accuracy. Every example
  loop now slices its runs the same way; the makemore family
  reproduced byte-identical output after the switch.

- `BatchNorm`: batch normalization at tensor granularity over
  `[batch, features]` values. `express` records the training mode —
  normalization by the batch's own mean and biased variance — and
  returns a `Normalization` carrying the output and the statistic
  values; `express_with` records the inference mode over statistics
  supplied as values, fed per run, so running estimates live with
  the training loop rather than on the tape.
- `LayerNorm` and `RmsNorm`: the stateless normalization siblings,
  taking per-sample statistics along the feature axis — full
  standardization with a per-feature affine, and root-mean-square
  re-scaling with a per-feature scale, respectively. No running
  estimates and no training/inference split: one recorded
  expression serves both. All three norms share one epsilon
  contract: a single-value constant broadcast in-graph to the
  variance's shape.
- `Value::mean_along`: the mean-reduction composite, `sum_along`
  divided by the reduced axis's extent.
- `Differentiable::counted`: the shape-derived constant constructor
  — a payload of a given shape holding an integer count — that lets
  composed formulas mint axis extents as payloads. Breaking for
  custom payload implementations, which must add the method.
- The `cuda` feature: large dense `f32`/`f64` products through
  cuBLAS on an NVIDIA GPU, Linux only. The libraries
  (`libcudart`/`libcublas`) are bound at run time by `dlopen`, so
  the build never links them and a machine without the toolkit or a
  device declines at run time; typed setup errors make the GPU
  tests skip only in those two environments and fail loudly on any
  other defect. `Backend::Cuda` joins the diagnostics enum between
  `Metal` and `Simd`. Built blind against the documented APIs and
  not yet validated on NVIDIA hardware: treat it as experimental
  until the first measured run, which will also tune its provisional
  flop threshold.

## [0.6.0] - 2026-08-02

### Added

- The `simd` feature: a portable CPU acceleration backend over the
  `matrixmultiply` crate's tuned, single-threaded microkernels with
  runtime instruction-set dispatch (AVX-512F, AVX2+FMA, AVX, NEON).
  It accelerates dense `f32` and `f64` products on every platform —
  the acceleration story for Linux — and sits last in the chain on
  macOS. `Backend::Simd` joins the diagnostics enum, and the ubuntu
  CI job now executes the backend grid it used to only compile.

## [0.5.4] - 2026-08-02

### Fixed

- `backward` no longer runs the derivative rules of operands used
  only as shape or index data (a broadcast's reference, a gather's
  selection): a `None` cotangent no longer marks its operand as an
  ancestor, so a singular expression behind such a reference cannot
  leak `NaN` into unrelated gradients (audit finding PG-001).
- `narrow` rejects zero-length windows at both the recording and
  payload boundaries instead of manufacturing the empty tensors the
  payload forbids by construction (audit finding PG-002).
- `narrow` and `pad` compute their window ends with checked
  arithmetic, so an overflowing `start + len` fails identically in
  debug and release builds instead of wrapping past the range check
  in release (audit finding PG-005).
- The Metal test grid and the backend status test skip only when the
  machine has no Metal device; every other setup failure — a shader
  that does not compile, a missing kernel, a rejected pipeline — is
  now a hard test failure instead of a silent skip (audit finding
  PG-006).
- `cargo test` no longer executes the Criterion benchmarks: the
  bench targets set `test = false` and CI names its test targets
  instead of `--all-targets` (which implies `--benches` and forces
  them regardless), with doctests run explicitly (audit finding
  PG-007).
- Documentation drift: the README's unsafe-code claim now names the
  default build, the `Tensor` storage list includes the one-hot
  selection, operand links are attributed to the tape's operand
  column, and the accelerate module no longer claims to be the only
  unsafe code in every build.

### Added

- Add dense-payload twins of the `tensor-regression` run benches:
  the existing cases build their payloads with `Tensor::filled`,
  which is constant storage and bypasses the dense matmul and slice
  paths, so the new cases are the ones that price the accelerated
  tiers.

### Changed

- Shrink the metal kernel's staging depth (BK 16 to 8) with a
  banded epilogue: six threadgroups stay resident per core instead
  of three, measured worth a few percent (~1.45 TFLOP/s at
  2048-square); wider 64x128 tiles measured no better and were not
  kept.

## [0.5.3] - 2026-08-02

### Changed

- Give the elementwise paths slice fast lanes: `map` and `zip` over
  contiguous dense buffers (and dense-with-constant pairs) run
  straight over slices instead of the per-element iterator
  dispatch, which measured 40x below memory speed. Dense multiplies
  went from 235 Melem/s to 5.7 Gelem/s and the gradient seed's
  constant-plus-dense add to 12.8 Gelem/s on an M1 Pro; a wide
  accelerated training step dropped from 112 ms to 19 ms. Every
  lane hands the combiner the same pairs in the same order, so
  results stay bit-identical across lanes, pinned by a test.

## [0.5.2] - 2026-08-02

### Documentation

- Add ACCELERATION.md: what each build supports, how routing and
  determinism work, the safety layering, the seam for payload
  authors, and every measured number; the README's acceleration
  section moves up beside the design bet and shrinks to the claim,
  the one command, and a pointer.

### Fixed

- Skip the metal GPU tests on machines without a Metal device (the
  virtualized CI runners), reporting the skip instead of failing:
  the backend already declines cleanly there, and the tests now
  honor the same contract.

## [0.5.1] - 2026-08-02

### Added

- Add the elementwise acceleration seam: `Elementary::map` offers a
  whole-buffer transcendental (`MapOperation`: `exp`, `ln`, `sqrt`,
  `tanh`) to the backend chain, and the tensor's elementwise
  operations consult it for contiguous dense buffers before the
  scalar path. The `accelerate` feature answers through vForce's
  vectorized transcendentals; measured on an M1 Pro, a wide
  training step dropped from 145 ms to 112 ms — the scalar `tanh`
  wall — with strided views and small buffers keeping the scalar
  path bit-for-bit.

### Changed

- Specialize the metal backend's pipelines per shape: the tiled
  kernel's dimensions and strides bake as Metal function constants,
  one cached pipeline per recurring shape (record-once training
  replays a handful), with the generic params-driven pipeline as
  the fallback past the cache cap.
- Raise the metal kernel's occupancy threefold: a GPU-counter trace
  showed the dedicated output-staging tile capping compute
  occupancy at one resident threadgroup per core, so the epilogue
  now reuses the operand staging area as a half-tile buffer in two
  coalesced passes, cutting the threadgroup footprint from 26.9 KB
  to 9.5 KB. Together with the per-shape pipelines, measured on an
  M1 Pro at 2048-square: 534 to about 1400 GFLOP/s.

## [0.5.0] - 2026-08-02

### Added

- Add the `metal` feature: large dense `f32` products (Metal has no
  `f64`) run on the GPU through hand-written simdgroup-matrix
  kernels — no MPS, no vendor library — compiled from source at
  first use, with shared-mode buffers from a size-classed pool on
  unified memory. The kernels read operands through the task's
  strides, so transposed, narrowed, and broadcast views pass through
  without copies. Accelerate leads the chain where both features are
  compiled (it measured ahead at every size), so Metal serves the
  stride patterns BLAS declines and everything large in metal-only
  builds — about twenty times the built-in slice path. A failed
  setup or runtime error poisons the backend into declining forever,
  degrading to slow, never to wrong; `Backend::Metal.status()`
  reports readiness, doubling as warmup for the one-time kernel
  compilation.
- Add the `throughput` example: the acceleration ladder measured on
  a wide dense model — the raw 2048-square product and whole
  training steps — with the dimensions shrinking eightfold when no
  backend is compiled in so the run still terminates.
- Add `init::Sample` and make the initializer factories
  element-generic: `uniform`, `normal`, `xavier`, and `kaiming` now
  produce `Tensor<Element>` for any element implementing `Sample`,
  with the element inferred from the network the closure feeds. The
  generator pipeline stays in `f64` and converts once at the end, so
  the `f64` path is bit-identical to every previous release (pinned
  by a golden-bits test) and the `f32` path is the same stream
  rounded once per element. Context-free factory calls bound to
  nothing now need a type annotation.

### Changed

- Move the tensor examples (`mlp_xor` and the makemore family) to
  `Tensor<f32>`: the field's training dtype, and the one every
  acceleration rung favors. The scalar examples and the crate-root
  doctest stay `f64`, and `f64` tensors remain fully supported and
  tested. The facade example still trains bit-identically to its
  hand-rolled twin from matching seeds.

## [0.4.0] - 2026-08-02

### Added

- Add the `accelerate` feature, the backend chain's first resident:
  dense `f32`/`f64` matrix products above a small flop threshold
  route to Apple's Accelerate framework (`cblas_sgemm`/`cblas_dgemm`
  — the AMX/SME matrix units on Apple Silicon, AVX kernels on Intel
  Macs), with transposed and narrowed views mapping to BLAS
  transpose flags and leading dimensions without copies; stride
  patterns BLAS cannot express and small tasks decline to the
  built-in paths. macOS only, zero dependencies, and a safe stub
  elsewhere. The default build is untouched and keeps
  `#![forbid(unsafe_code)]`; with the feature on, `unsafe` is
  confined to the backend module under a crate-wide `deny`.
- Add `Backend` and `BackendUnavailable`: the backend diagnostics
  surface, present in every build so no user code ever needs a
  `cfg` — `Backend::ALL` lists the defined backends in chain order
  and `Backend::status` reports `Ok`, `NotCompiled`,
  `PlatformUnsupported`, or a setup/poison reason.

## [0.3.1] - 2026-08-02

### Added

- Add the acceleration seam: `GemmTask` describes one dense
  matrix-multiplication job (spanning slices plus per-axis strides,
  so transposed and narrowed views pass through unmaterialized), and
  the provided `Elementary::gemm` offers each task to the compiled
  backend chain before the built-in paths compute. The chain is
  empty until the first backend feature lands, so behavior and
  results are unchanged; custom payload implementations keep the
  default.

## [0.3.0] - 2026-08-02

### Added

- Back `Tensor` with a strided layout over an extensible `Storage`
  representation (a shared dense buffer or a non-allocating constant), so
  `transpose` and the broadcasts are O(1) views instead of copies and the
  `backward` gradient seed no longer allocates a zeroed buffer per node.
- Add the view operations `Value::reshape` and `Value::permute` (with the
  `reshape`-based conveniences `squeeze` and `unsqueeze`), each a
  differentiable graph node whose gradient routes back by the inverse view.
  `permute` generalizes `transpose` to any rank.
- Add `Value::narrow` (a slice window along one axis): the forward is an
  O(1) view and the gradient scatters back into the excluded positions as
  zeros.
- Add `Value::gather` and `Tensor::selection`: an embedding-style row
  lookup, `table.gather(selection)`, where `selection` is a one-hot
  `[count, vocab]` input stored as its `usize` indices. The gradient
  scatter-adds into the table only (repeated rows accumulate); the
  selection is data and takes no gradient.
- Add `Value::log_softmax`, a fused, numerically stable log-softmax along
  a named axis (the max-shifted forward cannot be composed from recorded
  operations), and `cross_entropy`, the classification loss composed on
  top of it, normalizing by the targets' total mass — the batch size for
  one-hot targets.
- Add the elementwise operations `Value::sqrt`, `Value::powf`,
  `Value::maximum`, and `Value::relu`, and `Activation::Relu` for layers
  and neurons. The `Elementary` payload contract gains `sqrt`, `maximum`,
  and the 0/1 indicator `step`; `Tensorial` gains the `max_along`
  reduction.
- Add the composite expressions `Value::abs`, `Value::softmax`, and
  `Value::logsumexp` — formulas recorded as several primitive nodes, with
  the softmax pair composed stably on top of the fused log-softmax core —
  collected in a dedicated composition tier beside the single-node opcode
  methods.
- Add the `init` module: deterministic initializer factories (`uniform`,
  `normal`, and the fan-aware `xavier` and `kaiming`, which scale rank-2
  weights from the requested shape and zero rank-1 biases) matching the
  shape-to-payload closures `Layer` and `Mlp` take. Every factory is
  seeded explicitly and owns its generator state, so initialization is
  reproducible without a `rand` dependency.
- Add the `makemore_bigram` example: a character-level bigram language
  model over names — a `[vocab, vocab]` logit table read by `gather`,
  scored by `cross_entropy` on per-run one-hot minibatches, and sampled
  through the composite `softmax`.
- Add the `makemore_mlp` example: the Bengio-style character-level MLP —
  a three-character context embedded by `gather`, flattened by `reshape`,
  and squashed through a hand-rolled tanh hidden layer, with a
  single-row twin expression of the same parameters recorded for
  sampling since input shapes are baked in at recording time.
- Add the `makemore_mlp_facade` example: the same model on the `Mlp`
  facade, training bit-identically to `makemore_mlp` from matching
  seeds. The makemore examples live in `examples/makemore/` (declared
  as explicit example targets) and share their corpus machinery and
  dataset there.
- Add the `makemore_mlp_parallel` example: the same model trained data
  parallel — every step fans shard-shaped forward and backward runs
  across rayon's threads against the shared network, sums the gradient
  fields in a deterministic pairwise tree, and averages, computing the
  full-batch gradient exactly while cutting the wall clock
  several-fold.
- Add the `makemore_embedding_map` example: the MLP with a
  two-dimensional character embedding, rendered in the terminal before
  and after training by a small reusable labeled scatter chart
  (`examples/makemore/chart.rs`) whose marks are the letters
  themselves.
- Add the `gemm` benchmark group: the dense matmul path measured
  across sizes, element types, and transposed operands, reported in
  elements per second — one element per floating-point operation.

### Changed

- Accept `impl Into<Shape>` in `Tensor::new`, `Tensor::filled`, and
  `Value::reshape`: axis literals keep working unchanged, and a `Shape`
  or its reference now passes directly instead of being decomposed into
  an axis iterator. Other iterator sources go through `Shape::new`.
- Use plain verbs consistently for operations: rename `Field::scaled` to
  `Field::scale`, `Network::updated` to `Network::update`,
  `Tensorial::{transposed, permuted, narrowed, padded}` to
  `Tensorial::{transpose, permute, narrow, pad}` and
  `Value::{transposed, permuted, squeezed, unsqueezed}` to
  `Value::{transpose, permute, squeeze, unsqueeze}`; align the internal
  tape, layout, storage, and test helpers with the same rule.
- Make `Gradients` an alias for `Field` rather than a wrapper around it.
  `Evaluation::backward` still returns `Gradients`, but the result is a field
  directly, so `Network::update` and the field algebra take it without a
  conversion.
- Read tensor elements through `Tensor::iter` (logical row-major order),
  `Tensor::as_slice` (a borrowed slice when contiguous), or `Tensor::to_vec`,
  and compare tensors by logical value across storage representations.
- Multiply dense matrices on a slice path: `matmul` now reads dense
  rank-2 operands — including transposed, narrowed, and broadcast
  views — through their layout strides instead of per-element logical
  access, in loops shaped for the compiler's auto-vectorizer. The
  per-element accumulation order is unchanged (seeded from the first
  term), so results are bit-identical to the logical path, which
  non-dense storages keep. Measured on an Apple M1 Pro: 26 GFLOP/s
  `f32` and 13 GFLOP/s `f64` for square products, from 0.41 before.

### Removed

- Remove `Gradients::as_field` and `Gradients::into_field`. Pass the gradients
  themselves instead: `network.update(&gradients, ..)`.
- Remove `Tensor::elements`; use `iter`, `as_slice`, or `to_vec` instead.

## [0.2.0] - 2026-07-27

### Added

- Add declared inputs and per-run payload binding through `Network::input`
  and `Network::forward_with`, allowing one recorded graph to evaluate
  different samples concurrently.
- Add `Value::sum_along` and `Value::broadcast_along` for explicit axis-wise
  tensor reductions and broadcasting.
- Add the tensor-native `Mlp` facade and an end-to-end XOR training example.
- Add Criterion benchmarks for recording, execution, training, scaling, and
  memory behavior.
- Add continuous integration checks and finite-difference gradient tests.

### Changed

- Rebuild `Layer` at tensor granularity. `Layer::new` now accepts weight and
  bias payloads, and `Layer::express` accepts and returns one batched tensor
  value instead of slices and vectors of scalar values.
- Store parameter payloads per network generation so updates take
  O(parameters) work while preserving older generations.
- Track fork ancestry in symbols and fields so divergent network branches are
  rejected reliably.
- Restrict backward passes to scalar targets and to the target's ancestors.
- Reorganize internals into engine, neural, and payload modules while retaining
  the crate-root public exports.

### Fixed

- Restore operand shapes when differentiating broadcasts.
- Reject parameter updates whose payload shape changes.
- Reject empty tensors and detect tensor-volume overflow.

### Documentation

- Expand the README and crate documentation around concurrency, inputs,
  generations, tensor operations, layers, and MLPs.
- Add the project logo and refresh the terminology guide.

## [0.1.0] - 2026-07-26

- Initial release.

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[Unreleased]: https://github.com/shergin/poorgrad/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/shergin/poorgrad/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/shergin/poorgrad/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/shergin/poorgrad/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/shergin/poorgrad/compare/v0.5.4...v0.6.0
[0.5.4]: https://github.com/shergin/poorgrad/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/shergin/poorgrad/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/shergin/poorgrad/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/shergin/poorgrad/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/shergin/poorgrad/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/shergin/poorgrad/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/shergin/poorgrad/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/shergin/poorgrad/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/shergin/poorgrad/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shergin/poorgrad/releases/tag/v0.1.0
