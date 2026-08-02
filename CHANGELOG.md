# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog], and this project adheres to
[Semantic Versioning].

## [Unreleased]

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
[Unreleased]: https://github.com/shergin/poorgrad/compare/v0.5.3...HEAD
[0.5.3]: https://github.com/shergin/poorgrad/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/shergin/poorgrad/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/shergin/poorgrad/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/shergin/poorgrad/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/shergin/poorgrad/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/shergin/poorgrad/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/shergin/poorgrad/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/shergin/poorgrad/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shergin/poorgrad/releases/tag/v0.1.0
