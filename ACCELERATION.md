# Acceleration

How poorgrad uses the hardware the GPU poor already own. The stack
is deliberately explicit: backends are opt-in cargo features,
enabling a feature is the whole activation, and every number below
is measured, not asserted — `cargo bench` and the `throughput`
example rerun all of them on your machine.

## The ladder

Measured on an Apple M1 Pro (16-core GPU, 32 GB unified memory);
matrix numbers at 512- to 2048-square:

| build | matrix products | `exp` `ln` `sqrt` `tanh` | runs on |
|---|---|---|---|
| default | 26 GFLOP/s `f32`, 13 `f64` | scalar | everywhere |
| `--features accelerate` | 1.6 TFLOP/s `f32`, 550 GFLOP/s `f64` | vectorized | macOS; a safe stub elsewhere |
| `--features metal` | 1.4 TFLOP/s `f32` at large sizes; no `f64` | scalar | macOS; a safe stub elsewhere |

The naive definition — every element read through the strided
logical access — measures 0.4 GFLOP/s and remains the correctness
anchor and the fallback for exotic layouts. End to end, the same
training source spans a factor of four thousand with zero source
changes.

## Turning it on

```sh
cargo run --release --example throughput
cargo run --release --features accelerate --example throughput
cargo run --release --features accelerate,metal --example throughput
```

There is nothing to call, configure, or detect: dispatch happens
inside the payload's operations, above per-task thresholds, with
silent fallback to the built-in paths. Two optional touch points
exist, both diagnostics rather than switches:

```rust
// Loud mode: refuse to run slow instead of falling back silently.
poorgrad::Backend::Metal.status().expect("metal backend unavailable");
```

`Backend::ALL` lists every defined backend in chain order, and
`status` answers in every build — `NotCompiled` is an ordinary
answer, not a compile error, so interrogating the chain never needs
a `cfg`. For the `metal` feature, `status` doubles as warmup: it
forces the one-time kernel compilation so the first large product
does not pay it.

## What each build does

**The default build** is pure safe Rust under
`#![forbid(unsafe_code)]`. Dense matrix products run on a slice
path whose loops are shaped for the compiler's auto-vectorizer —
every output element owns an independent accumulator, so
vectorizing reorders no floating-point sum, which keeps the result
bit-identical to the naive definition while compiling to NEON
multiply-add on Apple Silicon.

**The `accelerate` feature** links Apple's Accelerate framework and
costs zero dependencies. Dense `f32`/`f64` products above a small
threshold become one `cblas_sgemm`/`cblas_dgemm` call, executing on
the AMX/SME matrix units (AVX kernels on Intel Macs) with
function-call latency — no device, no queue, no state. Transposed
and narrowed views map to BLAS transpose flags and leading
dimensions without copies; stride patterns BLAS cannot express (a
broadcast operand) decline down the chain. Whole-buffer `exp`,
`ln`, `sqrt`, and `tanh` over contiguous tensors run through
vForce's vectorized transcendentals.

**The `metal` feature** runs large `f32` products (Metal has no
`f64`) on the GPU through the crate's own simdgroup-matrix kernels —
no MPS, no vendor library — compiled from source at first use, with
one pipeline specialized per recurring shape (record-once training
replays a handful) and shared-mode buffers from a pool on unified
memory. Stated as measured: the AMX units currently beat this
kernel at every size, so where both features are compiled
Accelerate leads and Metal serves what BLAS declines; in metal-only
builds it runs everything large at about fifty times the built-in
path. A failed setup or a runtime error poisons the backend into
declining forever, with the reason held for `status` — a numerics
library degrades to slow, never to wrong.

## Routing

Backends form a compile-time chain tried in declaration order —
`Accelerate`, then `Metal` — and each may decline any task: below
its threshold, outside its stride mapping, beyond its integer
range, or with its device gone. Whatever the whole chain declines
lands on the built-in paths, so every task computes correctly in
every build; features change speed, never behavior classes. There
is deliberately no runtime switch: selection is per-build
(features) and per-task (thresholds, dtype — `f64` never reaches
Metal), never per-call-site.

## Determinism

Results are a pure function of the payloads, the compiled features,
and the machine. Within one binary, two identical runs can never
disagree — there is nothing that could change between them, and the
test suite verifies the hardware paths down to the bit (repeated
products compare bitwise). What a backend build forfeits is
bit-identity *with the built-in paths*: hardware sums and rounds in
its own order (AMX partitions sums, vForce rounds differently than
libm), the same way any BLAS differs from a textbook loop. One
cargo caveat: features unify across a dependency graph, so any
dependency enabling `poorgrad/accelerate` enables it for the whole
binary.

## Safety

The default build keeps `#![forbid(unsafe_code)]` verbatim. A
backend build drops `forbid` but keeps a crate-wide
`#![deny(unsafe_code)]`, with exactly one scoped `allow` per
backend module — `unsafe` outside them stays a compile error, and
what is inside is the FFI boundary itself: cblas and vForce calls
with their safety arguments written out, and the Metal encode path.
`accelerate` adds no crates; `metal` adds the `objc2` binding
family on macOS targets only.

## The seam, for payload authors

Acceleration enters through two provided methods on the
[`Elementary`](src/payload/elementary.rs) trait, both defaulting to
"compute on the built-in paths": `Elementary::gemm` receives a
`GemmTask` (spanning slices plus per-axis strides, so views pass
through unmaterialized) and `Elementary::map` receives a
`MapOperation` with a contiguous buffer. `f32` and `f64` forward to
the backend chain; a custom element type may route to its own
kernels by overriding them, and everything else keeps the defaults.
The engine above never learns any of this exists: operations stay
backend-blind, and the seam lives entirely in the payload tier.
The vocabulary — seam, backend, GEMM — is defined in
[TERMINOLOGY.md](TERMINOLOGY.md).
