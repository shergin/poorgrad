# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog], and this project adheres to
[Semantic Versioning].

## [Unreleased]

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
