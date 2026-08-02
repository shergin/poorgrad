# poorgrad 0.5.3 correctness and quality audit

- **Audit date:** 2026-08-02
- **Audited revision:** `c5e02127f5d2fc424d37a8c24f4b69d370e8cd78` (`Prepare poorgrad 0.5.3 release`)
- **Primary focus:** numerical correctness, graph and shape invariants, panic behavior, concurrency, optional backend safety, tests, CI, documentation, and dependency risk

## Overall assessment

The crate has a compact and unusually legible design, good public documentation, strong finite-difference and concurrency tests, a deliberately narrow dependency set, and sensible unsafe-code containment. The ordinary scalar and tensor paths, graph generation/forking model, parameter updates, layout/view logic, and both Apple backends passed the existing test grids on the audited machine.

The audit nevertheless found **two high-severity correctness defects that should block a release**, five medium-severity defects or control gaps, and four low-severity hardening items:

| Severity | Count | Summary |
|---|---:|---|
| High | 2 | A structurally non-differentiable operand can contaminate a valid gradient with `NaN`; `narrow(..., len = 0)` creates the empty tensors that the payload explicitly forbids. |
| Medium | 5 | Scalar reshaping breaks graph/payload shape coherence; public `scatter` silently accepts an inconsistent row count; range addition wraps in release; Metal tests skip every initialization error; CI executes very large Criterion benchmarks. |
| Low | 4 | Metal's aggregate address arithmetic can exceed `u32`; backend seams trust returned vector lengths; cross-entropy's target-mass precondition is incomplete; MLP zero-width validation happens late and indirectly. |

No confirmed memory-safety vulnerability was found in `poorgrad`'s unsafe blocks. The default configuration forbids unsafe code, and the optional FFI blocks are small and generally have adequate local safety arguments. The most important result of this audit is functional rather than memory-safety related: the public API can currently produce incorrect gradients and invalid tensor states.

### Release recommendation

Do not publish the audited revision as-is. Fix **PG-001** and **PG-002**, add the proposed regression tests, and run the complete validation matrix. **PG-003** through **PG-007** should preferably be resolved in the same release because they concern core invariants or the reliability of the release gates. The low-severity items can follow, provided their limitations are documented.

## Severity model

- **High:** wrong results, violated core invariants, or reliable crashes through a reasonable public API path.
- **Medium:** narrower correctness defects, debug/release divergence, or test/CI gaps capable of hiding a material regression.
- **Low:** extreme-scale limitations, misuse hardening, or incomplete validation/documentation with bounded impact.

## Findings

### PG-001 — High — `None` cotangents still mark operands as differentiable ancestors

**Locations:** [evaluation.rs:114-150](src/engine/evaluation.rs#L114-L150), [broadcast.rs:45-50](src/engine/function/broadcast.rs#L45-L50), [broadcast_along.rs:46-47](src/engine/function/broadcast_along.rs#L46-L47), [gather.rs:55-58](src/engine/function/gather.rs#L55-L58)

`Evaluation::backward` marks every operand link as an ancestor before it asks the operation for cotangents:

```rust
for &link in links {
    ancestors[link.index()] = true;
}
let cotangents = function.backward(...);

for (&link, cotangent) in links.iter().zip(cotangents) {
    if let Some(contribution) = cotangent {
        // accumulate
    }
}
```

That is inconsistent with the meaning of `Cotangents<Data> = ... Option<Data>`. `Broadcast`, `BroadcastAlong`, and `Gather` deliberately return `None` for operands used only as shape or index data. Those operands are not in the differentiable dependency cone, but the ancestor mask nevertheless schedules their producers' derivative rules.

This is not merely wasted work. A singular expression behind a shape-only reference receives a zero gradient, executes anyway, and can turn that zero into `NaN`. The `NaN` then reaches an otherwise unrelated leaf.

#### Public-API reproduction

```rust
use poorgrad::Network;

let network = Network::new();
let x = network.leaf(0.0_f64);
let singular_reference = x / x;
let source = network.leaf(2.0);
let output = source.broadcast_like(singular_reference);

let evaluation = network.forward();
let gradient = *evaluation.backward(output).of(x);

assert_eq!(gradient, 0.0); // fails: gradient is NaN
```

The forward result is the finite value `2.0`, and `Broadcast::backward` explicitly returns `None` for `singular_reference`. Therefore `output` has no differentiable dependence on `x`; its gradient must be exactly zero. The external audit harness printed:

```text
none_cotangent_gradient=NaN
```

This also contradicts the public guarantee in `Evaluation::backward` that expressions outside the target's dependency cone—including singular expressions—cannot disturb the result.

#### Impact

- A model can receive `NaN` gradients from a branch that is structurally non-differentiable.
- The failure is data-dependent and can be difficult to localize because the forward output remains valid.
- Any future operation using `None` cotangents inherits the same defect.

#### Recommendation

Mark an operand as an ancestor only when its cotangent is `Some`, in the same loop that accumulates the contribution:

```rust
for (&link, cotangent) in links.iter().zip(cotangents) {
    if let Some(contribution) = cotangent {
        let slot = link.index();
        ancestors[slot] = true;
        gradients[slot] = gradients[slot].clone() + contribution;
    }
}
```

Keep marking structural dependencies even when the numerical contribution happens to be zero; `Some(zero)` is still a differentiable edge, while `None` is not.

Add regression tests for a singular shape-only reference through both `broadcast_like` and `broadcast_along`. A focused operation-level test should also establish that every `None` edge is excluded from reverse reachability.

---

### PG-002 — High — zero-length `narrow` manufactures an invalid empty `Tensor`

**Locations:** [tensor.rs:162-195](src/payload/tensor.rs#L162-L195), [tensor.rs:833-869](src/payload/tensor.rs#L833-L869), [narrow.rs:25-48](src/engine/function/narrow.rs#L25-L48), [layout.rs:103-111](src/payload/layout.rs#L103-L111)

Both public tensor constructors explicitly reject empty tensors. Their documentation explains why: generic reductions seed from an existing element, and `Differentiable` provides only shape-preserving identities. `Tensorial::narrow`, however, accepts `len == 0` in both record-time inference and direct payload execution.

That creates a `Tensor` with a zero-volume shape through a safe, documented public operation. Downstream methods then encounter assumptions that the constructors normally guarantee.

#### Public-API reproduction

```rust
use poorgrad::{Tensor, Tensorial};

let tensor = Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]);
let empty = tensor.narrow(1, 0, 0); // succeeds; shape is [2, 0]
let _ = empty.sum_along(1);         // panics
```

The observed panic is:

```text
src/payload/layout.rs:109: attempt to calculate the remainder with a divisor of zero
```

`sum()` also panics because it cannot obtain a first element. The graph form records successfully and fails later during `forward`, defeating the stated policy that invalid shapes are rejected while the expression is recorded.

#### Impact

- A normal edge case creates a state that the type's constructors declare unsupported.
- Later behavior varies by operation: some iterators appear empty, reductions panic, and matrix operations fail at a different boundary.
- Errors move from record time to evaluation time, weakening one of the crate's central guarantees.

#### Recommendation

The design already explains why generic empty tensors are unsupported, so the smallest coherent fix is to reject `len == 0` in both:

1. `Narrow::infer_shape`, so graph recording fails immediately.
2. `Tensorial for Tensor<Element>::narrow`, so direct payload calls preserve the same invariant.

Document the additional panic condition. Add direct and graph regression tests for zero-length windows on every axis position, including `start == extent`. Supporting empty tensors instead would require a broader design: reduction identities, empty `Storage` behavior, layout indexing, matmul semantics, gather/scatter behavior, and gradient seeding would all need explicit definitions.

---

### PG-003 — Medium — scalar reshape breaks recorded-shape/payload-shape coherence

**Locations:** [tensorial.rs:83-135](src/payload/tensorial.rs#L83-L135), [tensorial.rs:137-189](src/payload/tensorial.rs#L137-L189), [value.rs:253-260](src/engine/value.rs#L253-L260), [evaluation.rs:95-109](src/engine/evaluation.rs#L95-L109)

The `f32` and `f64` implementations of `Tensorial::reshape` ignore the requested shape and return the scalar unchanged. Record-time inference, meanwhile, accepts every shape of volume one. A scalar value can therefore acquire a recorded shape such as `[1]`, while its evaluated payload still reports the scalar shape `[]`.

#### Public-API reproduction

```rust
use poorgrad::{Differentiable, Network};

let network = Network::new();
let scalar = network.leaf(7.0_f64);
let vector = scalar.reshape([1]);
let evaluation = network.forward();

assert_eq!(vector.shape().axes(), &[1]);
assert_eq!(evaluation.of(vector).shape().axes(), &[]);

evaluation.backward(vector); // accepted despite vector's recorded rank being 1
```

Observed output:

```text
recorded_shape=[1] payload_shape=[]
backward_accepted_recorded_rank_one=true
```

`Evaluation::backward` checks the materialized payload's shape, not the target `Value`'s recorded shape, so this mismatch bypasses the documented scalar-target restriction. It also provides a route into named-axis graph operations that are supposed to reject scalar `Value`s.

#### Impact

- The graph's shape column can disagree with the values it describes.
- Record-time validation and runtime validation no longer enforce the same model.
- Singleton-axis operations often happen to be numerically isomorphic, which makes the defect quiet rather than harmless.

#### Recommendation

Choose and enforce one model:

- Preferably, scalar payloads remain rank zero: make scalar `reshape` reject every non-scalar shape, which prevents the only public route that forges axes on `f32`/`f64` values.
- If scalar payloads are intended to model arbitrary volume-one shapes, their runtime representation needs to carry that shape; a bare `f32`/`f64` cannot satisfy `Differentiable::shape` coherently.

As defense in depth, validate `output.shape().rank()` in `Evaluation::backward`. Consider carrying the recorded shape snapshot into `forward` and asserting that every operation's returned payload shape equals the inferred shape. Add scalar tests for `reshape([1])`, `unsqueeze`, named-axis calls, and non-scalar backward targets.

---

### PG-004 — Medium — `scatter` silently drops rows and does not validate its adjoint contract

**Location:** [tensor.rs:946-965](src/payload/tensor.rs#L946-L965)

`Tensorial::scatter` is public and documents `self` as a `[count, ...]` gradient paired with a selection containing `count` indices. Its implementation computes a row size from `self.shape[1..]` and iterates only over the selection indices. It does not check:

- that `self` has rank at least one;
- that `self.shape[0] == selection.indices.len()`;
- that `rows` agrees with the selection vocabulary; or
- that every destination fits the requested output row count.

If the gradient has more rows than the selection, trailing gradients are silently discarded:

```rust
use poorgrad::{Tensor, Tensorial};

let gradient = Tensor::new([2, 1], [10.0_f64, 20.0]);
let selection = Tensor::selection([0], 3, 1.0);
let result = gradient.scatter(&selection, 3);

assert_eq!(result.to_vec(), [10.0, 0.0, 0.0]); // 20.0 disappeared
```

The inverse mismatch or a destination outside `rows` panics by incidental indexing rather than by a clear precondition check. Graph-generated `Gather::backward` currently supplies coherent arguments, which bounds the ordinary graph impact, but direct public calls and future callers are exposed.

#### Recommendation

Before reading data, assert:

```text
gradient.rank() >= 1
gradient.axes()[0] == indices.len()
selection.shape()[1] == rows
```

The last check also proves that every validated selection index is below `rows`. Add tests for too few and too many gradient rows, scalar gradients, mismatched vocabulary/output rows, and repeated valid destinations.

---

### PG-005 — Medium — range checks overflow and change behavior between debug and release

**Locations:** [narrow.rs:34-41](src/engine/function/narrow.rs#L34-L41), [tensor.rs:838-849](src/payload/tensor.rs#L838-L849), [tensor.rs:876-899](src/payload/tensor.rs#L876-L899)

The `narrow` and `pad` bounds checks form the window end as `start + len`. In debug builds an overflowing addition panics before the intended assertion. In optimized builds it wraps, can satisfy the assertion, and is reused in the actual selection condition.

#### Release-mode reproduction

```rust
use poorgrad::{Tensor, Tensorial};

let tensor = Tensor::new([1], [5.0_f64]);
let padded = tensor.pad(0, usize::MAX, 2);
assert_eq!(padded.to_vec(), [0.0, 0.0]); // invalid request accepted; value lost
```

Observed behavior:

```text
debug:   attempt to add with overflow
release: overflow_pad=[0.0, 0.0], overflow_pad_panicked=false
```

`narrow` has the same check pattern at both graph-recording and payload-execution boundaries. Its layout then also computes `start * stride` and adds it to the offset, relying on the failed range proof.

#### Recommendation

Compute the end exactly once with `checked_add`:

```rust
let end = start.checked_add(len).expect("window end overflows `usize`");
assert!(end <= extent, ...);
```

Use `end` in messages and loop conditions. Retain or add checked offset arithmetic in `Layout::narrow` as defense in depth. Add boundary tests that run identically in debug and release; include `usize::MAX` starts and lengths.

---

### PG-006 — Medium — Metal correctness tests skip every initialization failure

**Locations:** [metal_tests.rs:7-18](src/backend/metal/tests/metal_tests.rs#L7-L18), [context.rs:53-72](src/backend/metal/context.rs#L53-L72), [backend_tests.rs:8-17](src/backend/tests/backend_tests.rs#L8-L17)

The Metal test helper says that only the absence of a Metal device should skip tests and every other setup failure should be hard. The implementation catches every error from `context()` and returns `None`:

```rust
match context() {
    Ok(context) => Some(context),
    Err(reason) => {
        eprintln!("skipping: {reason}");
        None
    }
}
```

`Context::new` uses the same `String` error channel for all of these cases:

- no device;
- no command queue;
- shader compilation failure;
- missing kernel function;
- pipeline creation failure; and
- insufficient threadgroup capacity.

The public backend status test likewise accepts any `Initialization(_)` on a macOS Metal build. Consequently, a shader syntax error, renamed kernel, or pipeline incompatibility can turn the entire GPU correctness grid green by skipping it—even on the macOS runner selected specifically to expose Metal.

The audited local machine did initialize Metal and passed the naive, tiled, specialized, transposed, and end-to-end all-feature tests. The finding concerns the regression gate, not a failure observed in the current kernels.

#### Recommendation

Introduce a typed initialization error with at least `NoDevice` and `Failed(reason)` variants. Skip only `NoDevice`; panic or return a test failure for compilation, kernel lookup, pipeline, command queue, and poison errors. If the `macos-26` runner is expected to provide a GPU, add an explicit CI preflight that requires `Backend::Metal.status()` to succeed rather than silently accepting an unavailable backend.

---

### PG-007 — Medium — `cargo test --all-targets` executes the large Criterion benchmarks in CI

**Locations:** [ci.yml:1-28](.github/workflows/ci.yml#L1-L28), [ci.yml:35-44](.github/workflows/ci.yml#L35-L44), [Cargo.toml:80-106](Cargo.toml#L80-L106), [gemm.rs:89-106](benches/gemm.rs#L89-L106)

The CI comment says the test suite “also compiles the benchmarks,” but both jobs run `cargo test --all-targets`. Because every Criterion target has `harness = false`, Cargo executes the benchmark binaries in Criterion's test mode; it does not merely compile them.

This was reproduced locally. After the unit tests, `cargo test --all-targets` began printing Criterion `Testing ... Success` lines and reached `Testing gemm/f32/square-2048`, where it was manually terminated. On the default Linux job no accelerated backend is compiled, so that case enters the built-in debug GEMM. One invocation alone performs `2048^3 = 8,589,934,592` inner multiply/add iterations. Other benchmark targets also construct 100,000-node graphs and large buffers.

#### Impact

- CI duration and success depend on benchmark implementation details.
- A correct change can appear hung or time out in the Ubuntu gate.
- The “test” gate performs uncontrolled performance workloads that are not assertions.

#### Recommendation

Separate execution from compilation. For example:

```sh
cargo test --lib
cargo test --doc
cargo check --examples --benches
```

Alternatively set `test = false` on every benchmark target and retain a separate `cargo check --benches` step. Keep `cargo bench` in an explicitly triggered performance workflow, preferably release-only and with a timeout.

---

### PG-008 — Low — Metal checks individual `u32` values but not aggregate addresses

**Locations:** [metal/mod.rs:84-93](src/backend/metal/mod.rs#L84-L93), [gemm.metal:11-19](src/backend/metal/shaders/gemm.metal#L11-L19), [gemm.metal:96-117](src/backend/metal/shaders/gemm.metal#L96-L117), [gemm.metal:158-165](src/backend/metal/shaders/gemm.metal#L158-L165)

`fits_u32` checks each extent and stride independently. The shaders then calculate addresses with 32-bit `uint` multiplication and addition. Individual fit does not imply that expressions such as `row * row_stride + column * column_stride` or `row * n + column` fit.

For example, a row-major left operand with `m = 65,537`, `k = 65,536`, and stride `[65,536, 1]` passes `fits_u32`, but the address of row 65,536 starts at `2^32` elements and wraps to zero in shader arithmetic.

This requires an input buffer just over 16 GiB, and the pool rounds it to a still larger size class. Actual reachability therefore depends on the device's `maxBufferLength`; allocation failure safely poisons/declines the backend and falls through to the CPU. It remains a latent wrong-result boundary on a device capable of accepting such a buffer, and Apple exposes `MTLDevice.maxBufferLength` precisely because the limit is device-specific ([Apple documentation](https://developer.apple.com/documentation/metal/mtldevice/maxbufferlength)).

#### Recommendation

Either:

- check each matrix's maximum addressed element and `m * n - 1` with checked host arithmetic, declining when any exceeds `u32::MAX`; or
- move address arithmetic and relevant parameters to Metal `ulong`/64-bit values.

Also use checked multiplication for host output byte counts. A unit test can exercise the pure eligibility arithmetic without allocating the matrices.

---

### PG-009 — Low — acceleration seams accept malformed result lengths

**Locations:** [elementary.rs:65-94](src/payload/elementary.rs#L65-L94), [tensor.rs:522-533](src/payload/tensor.rs#L522-L533), [tensor.rs:594-603](src/payload/tensor.rs#L594-L603)

The public `Elementary::map` and `Elementary::gemm` extension points document that returning `Some` asserts a correctly sized result. `Tensor` passes those vectors directly to its private unchecked `dense` constructor. A buggy safe trait implementation—or a regression in a built-in backend—can therefore create storage whose logical shape and buffer length disagree. Later access may silently ignore excess elements or panic on a short vector.

This is primarily a containment issue because the semantic contract is documented and the built-in implementations returned correct lengths in testing.

#### Recommendation

Assert `mapped.len() == input.len()` and `product.len() == rows * columns` at the seam before constructing storage. Use checked multiplication for the expected product length. Add a deliberately malformed test implementation to verify that failure occurs at the backend boundary with a precise message.

---

### PG-010 — Low — `cross_entropy` omits its positive target-mass precondition

**Location:** [loss.rs:14-50](src/neural/loss.rs#L14-L50)

The implementation divides by `targets.sum()`. The documentation describes one-hot, soft, and weighted targets, but does not require nonnegative finite targets with positive total mass. An all-zero target payload produces `0 / 0` and a `NaN` loss and gradient. Negative or non-finite weights have similarly surprising semantics.

Because targets may be fed per run and `Data` is generic, the current composed graph cannot validate this condition at record time. That makes the missing runtime precondition especially important to document.

#### Recommendation

State explicitly that target entries must be finite and nonnegative and their total mass must be positive. If checked behavior is desired, introduce a checked/fused loss path capable of validating runtime data and returning a typed error; otherwise document that invalid distributions propagate IEEE-754 `NaN`/infinity.

---

### PG-011 — Low — MLP accepts zero-width topology entries until initialization fails elsewhere

**Location:** [mlp.rs:23-59](src/neural/mlp.rs#L23-L59)

`Mlp::new` validates only `sizes.len() >= 2`. A topology such as `[2, 0, 1]` invokes the caller's initializer with empty parameter shapes. Built-in tensor initializers then fail indirectly because tensors cannot be empty; a stateful custom initializer may already have advanced or performed side effects before the eventual `Layer::new` failure.

#### Recommendation

Assert that every width is greater than zero before invoking the initializer, and document the condition in `# Panics`. Add tests for zero input, hidden, and output widths.

## Additional quality observations

These are not release blockers, but addressing them would make the crate's quality claims more precise.

1. **Documentation drift.** The README says tensor storage is only dense or constant, but `Storage::Selection` is present ([README.md:187-193](README.md#L187-L193)). It says each `Function` variant owns operand links, while links live in the tape's separate operand column ([README.md:206-212](README.md#L206-L212)). The Accelerate module says it is the crate's only unsafe code, which is false when `metal` is enabled ([accelerate.rs:12-13](src/backend/accelerate.rs#L12-L13)). The README's unconditional “no unsafe in this crate” wording at [README.md:59-61](README.md#L59-L61) should say “in the default build.”
2. **No explicit MSRV.** `Cargo.toml` sets edition 2024 but not `rust-version`, and CI always follows floating `stable`. Edition 2024 implies a floor, but an explicit `rust-version` plus an MSRV CI job would turn compatibility into a tested contract.
3. **Sparse doctest coverage.** Rustdoc builds cleanly, but only the crate-level example is compiled as a doctest. The extensive public item documentation is useful prose; adding executable examples for shape-changing operations, feeds, fields, and neural helpers would protect it against API drift.
4. **CI has no dependency-advisory gate.** Neither `cargo-audit` nor `cargo-deny` is installed or run. A scheduled lockfile audit would be valuable, especially because advisories can appear without source changes.

## Correctness and safety review notes

### Autograd and graph engine

- Arithmetic, transcendental, matrix, reduction, reshape, permutation, gather, log-softmax, and neural-composition derivative formulas were reviewed. Apart from PG-001 and the shape/invariant issues above, the rules match their documented mathematics.
- Existing central-difference tests cover scalar arithmetic/transcendentals and representative tensor/layer expressions. Fan-out accumulation and disconnected singular branches are tested; PG-001 occupies the missing case where a connected operation has a structurally non-differentiable operand.
- Tape allocation order is a valid topological order. Snapshot evaluation and reverse replay preserve generation isolation.
- Network lineage/branch checks, stale field rejection, fork divergence, parameter-store updates, and feed validation are well defended and tested.
- Poisoning a tape after a caught recording panic is an intentional, documented contract rather than an accidental failure mode.

### Tensor representation and numerics

- Constructor volume checks, element-count checks, immutable shared storage, strided views, permutation inversion, gather/scatter adjoint math for valid arguments, and matrix gradients are coherent.
- Dense, constant, and selection representations generally preserve logical row-major iteration.
- The built-in GEMM paths accumulate terms in the documented order; tests include contiguous, transposed, narrowed, broadcast, constant, `f32`, negative-zero, and nested-element cases.
- `log_softmax` uses the standard max-shift stabilization and the correct vector-Jacobian product `g - softmax * sum(g)`.
- IEEE-754 domain behavior for logarithm, square root, division, and exponent-side powers is mostly documented rather than hidden.

### Concurrency

- `Differentiable: Send + Sync`, compile-time assertions, immutable evaluations, per-run feed overlays, and snapshot ownership form a consistent concurrency story.
- Recording is serialized through the tape mutex; forward/backward run from snapshots without holding it.
- The `cow_vec` 1.4.0 implementation used by the tape was inspected at its raw-pointer boundary. Its arena keeps allocations stable, the arena lifetime is retained by `Arc`, allocation is mutex-protected, and public reads expose only shared references. No concrete soundness violation was found in the subset on which `poorgrad` relies.

### Unsafe and FFI

- [lib.rs:41-48](src/lib.rs#L41-L48) forbids unsafe code in the default build and retains crate-wide `deny(unsafe_code)` when a backend feature opens only its module scope.
- Accelerate converts dimensions and leading strides to `i32`, validates matrix spans through `GemmTask`, allocates exact output buffers, and keeps FFI pointers live for each synchronous call.
- Metal sizes buffers before raw copies, waits for command completion before reading output, and retains command resources for the call. The unsafe `Context: Send + Sync` relies on Metal's threading contract; Apple documents `MTLDevice` and `MTLComputePipelineState` as `Sendable`, and explicitly says command queues are thread-safe ([MTLDevice](https://developer.apple.com/documentation/metal/mtldevice), [MTLComputePipelineState](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate), [MTLCommandQueue](https://developer.apple.com/documentation/metal/MTLCommandQueue?language=objc)). Per-call command buffers and encoders are not shared.
- No demonstrated out-of-bounds FFI access, aliasing violation, use-after-free, or data race was found. PG-008 is a shader integer-correctness limit, not a confirmed host memory-safety issue.

### Dependencies

The default runtime dependency graph is small: `cow_vec 1.4.0` (with `typed-arena 2.0.2`), `smallvec 1.15.2`, and `static_assertions 1.1.0`. All features add the `objc2 0.6.4` family on macOS. Criterion and Rayon are development-only.

A targeted RustSec review found that the historical `smallvec` memory-safety advisories are patched by the locked `1.15.2` ([package advisory list](https://rustsec.org/packages/smallvec.html)), and the July 2026 `crossbeam-epoch` advisory is patched in the locked `0.9.20` ([RUSTSEC-2026-0204](https://rustsec.org/advisories/RUSTSEC-2026-0204.html)). This manual check is not equivalent to a complete `cargo audit`; the scanner was unavailable in the audit environment. No known-vulnerable locked version was identified by the targeted review.

## Validation performed

The following checks were run against the audited source. The version-only release commit did not change Rust source from its immediate predecessor.

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | Passed |
| Strict Clippy, all features, library, benches, and every tracked example, with `-D warnings` | Passed |
| `cargo test --lib` | Passed: 209/209 |
| `cargo test --release --all-features --lib` | Passed: 221/221 |
| `cargo +stable test --all-features --lib` on Rust 1.90.0 | Passed: 221/221 |
| `cargo test --all-features --doc` | Passed: 1/1 |
| `RUSTDOCFLAGS=-Dwarnings cargo doc --all-features --no-deps` | Passed |
| `cargo check --lib --features accelerate` | Passed |
| `cargo check --lib --features metal` | Passed |
| `cargo package --offline --allow-dirty` | Passed and verified with this report included: 126 files, 1.3 MiB unpacked, 750.7 KiB compressed |
| External public-API reproduction harness, debug and release | Reproduced PG-001 through PG-005 as described |
| `cargo test --all-targets` | Intentionally stopped after it executed the Criterion suites and entered the debug 2048-square GEMM; see PG-007 |

The all-feature tests executed successfully on an Apple Silicon macOS host with both Accelerate and Metal available; the Metal grid did not skip locally. The repository also defines stable Ubuntu and macOS CI jobs. A Linux target was not installed locally, so Linux runtime behavior was not independently reproduced beyond the repository's portable stubs and CI configuration.

An ignored local `examples/makemore.rs` experiment is not tracked and is excluded from the published package. It was intentionally excluded from strict Clippy scope; every tracked example target passed.

## Prioritized remediation plan

1. **Repair reverse reachability (PG-001).** Move ancestor marking behind `Some(cotangent)` and add singular shape-only reference tests.
2. **Restore the non-empty tensor invariant (PG-002).** Reject zero-length narrow operations at recording and payload boundaries.
3. **Enforce shape coherence (PG-003).** Prevent scalar payloads from recording non-scalar shapes; validate backward against the recorded target shape.
4. **Harden public shape/range APIs (PG-004, PG-005).** Validate scatter's complete contract and replace unchecked window arithmetic with checked operations.
5. **Make release gates trustworthy (PG-006, PG-007).** Distinguish no-device skips from backend failures and stop test jobs from executing benchmarks.
6. **Close backend boundaries (PG-008, PG-009).** Check aggregate Metal indices and returned backend vector lengths.
7. **Clarify neural and release contracts (PG-010, PG-011 and quality observations).** Document target mass, reject zero widths early, correct drift, declare MSRV, and add advisory automation.

After steps 1–5, run the complete debug/release/all-feature matrix on stable Rust, the macOS hardware backend grid, and the Ubuntu job with the corrected CI commands. Add the five external reproductions as in-tree regression tests before considering the defects closed.

## Audit limitations

- This was a source and dynamic test audit, not a formal proof.
- No fuzzing campaign, Miri run, ThreadSanitizer run, or exhaustive scheduler exploration was performed.
- `cargo-audit`, `cargo-deny`, `cargo-semver-checks`, and coverage tooling were unavailable; dependencies were reviewed manually and through the locked tree.
- The extreme Metal address case in PG-008 was established from host/shader arithmetic but not allocated and run because it requires more than 16 GiB for one operand and is device-limit dependent.
- Remote CI history was not used to claim that PG-007 has already timed out; the benchmark execution and local non-completion were reproduced directly from the checked-in workflow command.
