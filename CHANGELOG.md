# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog], and this project adheres to
[Semantic Versioning].

## [Unreleased]

### Added

- Back `Tensor` with a strided layout over an extensible `Storage`
  representation (a shared dense buffer or a non-allocating constant), so
  `transposed` and the broadcasts are O(1) views instead of copies and the
  `backward` gradient seed no longer allocates a zeroed buffer per node.
- Add the view operations `Value::reshape` and `Value::permuted` (with the
  `reshape`-based conveniences `squeezed` and `unsqueezed`), each a
  differentiable graph node whose gradient routes back by the inverse view.
  `permuted` generalizes `transposed` to any rank.
- Add `Value::narrow` (a slice window along one axis): the forward is an
  O(1) view and the gradient scatters back into the excluded positions as
  zeros.
- Add `Value::gather` and `Tensor::selection`: an embedding-style row
  lookup, `table.gather(selection)`, where `selection` is a one-hot
  `[count, vocab]` input stored as its `usize` indices. The gradient
  scatter-adds into the table only (repeated rows accumulate); the
  selection is data and takes no gradient.

### Changed

- Make `Gradients` an alias for `Field` rather than a wrapper around it.
  `Evaluation::backward` still returns `Gradients`, but the result is a field
  directly, so `Network::updated` and the field algebra take it without a
  conversion.
- Read tensor elements through `Tensor::iter` (logical row-major order),
  `Tensor::as_slice` (a borrowed slice when contiguous), or `Tensor::to_vec`,
  and compare tensors by logical value across storage representations.

### Removed

- Remove `Gradients::as_field` and `Gradients::into_field`. Pass the gradients
  themselves instead: `network.updated(&gradients, ..)`.
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
[Unreleased]: https://github.com/shergin/poorgrad/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/shergin/poorgrad/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shergin/poorgrad/releases/tag/v0.1.0
