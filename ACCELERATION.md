# Acceleration

How topos uses the hardware the GPU poor already own. The stack
is deliberately explicit: backends are opt-in cargo features,
enabling a feature is the whole activation, and every number below
is measured, not asserted — `cargo bench` and the `throughput`
example rerun the in-crate numbers on your machine, and the
[tools/](tools/) scripts rerun the emission ones.

## The ladder

Measured on an Apple M1 Pro (16-core GPU, 32 GB unified memory);
matrix numbers at 512- to 2048-square:

| build | matrix products | `exp` `ln` `sqrt` `tanh` (`f32` `tanh`, 2M-8M) | runs on |
|---|---|---|---|
| default | 26 GFLOP/s `f32`, 13 `f64` | scalar, 0.4 Gelem/s | everywhere |
| `--features simd` | 96 GFLOP/s `f32`, 47 `f64` | scalar, 0.4 Gelem/s | everywhere |
| `--features accelerate` | 1.6 TFLOP/s `f32`, 550 GFLOP/s `f64` | vForce, 1.2 Gelem/s | macOS; a safe stub elsewhere |
| `--features metal` | 1.4 TFLOP/s `f32` at large sizes; no `f64` | GPU, 2.2-2.7 Gelem/s above 128k elements | macOS; a safe stub elsewhere |

The `simd` row is that backend's NEON kernels on the same machine;
its AVX-512/AVX2 kernels on x86 Linux are expected to be of the
same order but are not measured here — `cargo bench` reruns the
numbers on yours. Where the slice path pays 8x for a transposed
operand, the `simd` kernels pack first and run every stride form at
full speed.

The naive definition — every element read through the strided
logical access — measures 0.4 GFLOP/s and remains the correctness
anchor and the fallback for exotic layouts. End to end, the same
training source spans a factor of four thousand with zero source
changes.

## Views and broadcasting

Broadcasting is metadata, not memory: a broadcast is a stride-0
view over the source buffer, a squeeze of a broadcast view stays a
view, and the elementwise paths read views at slice speed. Two
mechanisms carry this, both in the default build. Binary operations
walk same-shape operands by innermost runs — a unit stride is a
slice, a zero stride holds one element for the run — which is the
loop shape the auto-vectorizer wants and the odometer walk defeats.
Elementwise maps over a broadcast view transform only the distinct
elements and keep the layout, so the result is still a view and the
backend seam sees a contiguous window.

Measured on 2M-element `f32` operands (the `broadcast` bench):

| case | odometer walk | view paths |
|---|---|---|
| matrix plus broadcast row (bias add) | 0.17 Gelem/s | 7.6 Gelem/s |
| broadcast column times broadcast row | 0.14 Gelem/s | 9.8 Gelem/s |
| `exp` over a broadcast row | 0.19 Gelem/s | the 1k distinct elements only |

The graph tier records broadcasts explicitly — `broadcast_to` and
`broadcast_pair` compose the named expansions under the right-aligned
NumPy rule — and the payload tier makes them free; no operation ever
broadcasts implicitly.

## End to end

The ladder above is pure matrix benchmarks; these are whole
training steps — forward, backward, and update — from the two
convolutional examples on the same machine (`mnist` on its compact
plan, `cifar10` on its default plan, batch 64, `f32`):

| build | `mnist` ms/step | `cifar10` ms/step |
|---|---|---|
| default | 106.8 | 391.5 |
| `--features accelerate` | 82.0 (-23%) | 261.9 (-33%) |
| `--features metal` | 107.7 (0%) | 313.3 (-20%) |

Three readings worth their lines:

- The wins track the products' share of the whole step (about a
  third, by profile), not the ladder's headline ratios: convolution
  GEMMs at these sizes are skinny (im2col columns 27 to 288), and
  everything that is not a product — window fills, folds, pads —
  runs the built-in paths regardless of features.
- Metal's flat `mnist` row is its thresholds working, not failing:
  every training product there sits below the flop bar, the chain
  declines each one, and the run reproduces the default build bit
  for bit. `cifar10`'s larger products clear the bar. The Metal
  runtime does hold device state either way — a few hundred
  megabytes of resident overhead on the small example.
- The accuracies (98.20-98.22%, 65.20-65.47%) differ across builds
  by design: each backend sums in its own order, so each build
  follows its own, equally valid training trajectory — the
  determinism boundary documented below, demonstrated end to end.

## Turning it on

```sh
cargo run --release --example throughput
cargo run --release --features simd --example throughput
cargo run --release --features accelerate --example throughput
cargo run --release --features accelerate,metal --example throughput
```

There is nothing to call, configure, or detect: dispatch happens
inside the payload's operations, above per-task thresholds, with
silent fallback to the built-in paths. One labeled choice exists —
the `Numerics` posture on a compile request, documented under
Routing — and two optional touch points, both diagnostics rather
than switches:

```rust
// Loud mode: refuse to run slow instead of falling back silently.
topos::Backend::Metal.status().expect("metal backend unavailable");
```

`Backend::ALL` lists every implementer, `Backend::coverage` the
full matrix — formula by implementer, each cell carrying its
certified fidelity and forwarding precisions — `Formula::chain` each
offer order, and `serves`, `compiled`, and `status` answer in every
build; `NotCompiled` is an ordinary answer, not a compile error, so
interrogating the stack never needs a `cfg`. For the `metal` feature, `status` doubles as warmup: it
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
vForce's vectorized transcendentals — including over broadcast
views, whose distinct-element windows are contiguous.

**The `simd` feature** is the portable rung: the `matrixmultiply`
crate's hand-tuned, single-threaded CPU microkernels, selecting
AVX-512F, AVX2+FMA, AVX, or NEON at run time. It accelerates dense
`f32` and `f64` products above a small threshold on every platform
— this is the acceleration story for Linux — and its strided API
takes transposed and narrowed views directly, so the forms that
cost the slice path most are exactly where it gains most. Stride-0
broadcast operands decline down the chain. On macOS it sits behind
the Apple backends as mop-up; elementwise transcendentals stay
scalar in this build.

**The `cuda` feature** (Linux only) runs large `f32` and `f64`
products through cuBLAS on an NVIDIA GPU. The libraries
(`libcudart`, `libcublas`) are bound at run time by `dlopen`, never
at link time: the build succeeds on every machine, and a missing
toolkit or device is a `status` answer, not a build failure.
Discrete memory sets the economics — PCIe copies bound every task,
so the threshold is high (~200-square) and the arm is copy-bound
even where it wins; GeForce-class cards also run `f64` at a small
fraction of their `f32` rate. Status: built against the documented
APIs, correctness- and skip-gated in CI, but not yet validated on
NVIDIA hardware — treat it as experimental until measured numbers
replace this sentence.

**The `metal` feature** runs large `f32` products (Metal has no
`f64`) on the GPU through the crate's own simdgroup-matrix kernels —
no MPS, no vendor library — compiled from source at first use, with
one pipeline specialized per recurring shape (record-once training
replays a handful) and shared-mode buffers from a pool on unified
memory. Stated as measured: the AMX units currently beat this
kernel at every size, so where both Apple features are compiled
Accelerate leads and Metal serves what BLAS declines; in metal-only
builds it runs everything large at about fifty times the built-in
path. Whole-buffer `exp`, `ln`, `sqrt`, and `tanh` run on the GPU
through one-thread-per-element kernels in the same compiled
library, and here the order reverses: the GPU passes vForce near
512k elements (2.7 against 1.2 Gelem/s at 8M), so Metal leads the
map chain with a size gate that adapts to whether `accelerate`
stands behind it. A failed setup or a runtime error poisons the
backend into declining forever, with the reason held for `status` —
a numerics library degrades to slow, never to wrong.

## Routing

Backends form a compile-time chain whose per-formula order is
declared data: `Formula::chain` lists each formula's members
hardware-greediest first, by measurement — products try
`Accelerate`, then `Metal`, then `Cuda`, then `Simd`; elementwise
maps try `Metal`, then `Accelerate`, the measured crossover — and
membership agrees with the `Backend::coverage` matrix, whose cells
also carry each kernel's certified fidelity and forwarding
precisions. Coverage declares *may*; the offer decides *will*: each
member may decline any task — below its threshold, outside its
stride mapping, beyond its integer range, or with its device gone —
and whatever the whole chain declines lands on the built-in paths,
so every task computes correctly in every build; features change
speed, never behavior classes. Selection is per-build (features)
and per-task (thresholds, precision — `f64` never reaches Metal),
never per-call-site. The one run-scoped control is a posture, not a
router: `Numerics::Exact` demands bit-identity fidelity, which no
offer-dispatched kernel meets today, so those runs compute on the
reference paths — the same bits as the default build, reachable in
every build — while `Numerics::Fast` (the default) demands only the
envelope fidelity: the chain exactly as described above, its thresholds
serving as cost heuristics inside the posture rather than
correctness boundaries.

```rust
// One process, both answers: the oracle and the fast path.
let exact = network.compile(Request::roots([loss]).numerics(Numerics::Exact));
let fast = network.compile(Request::roots([loss]));
```

## Past the boundary: emitted plans on XLA

Everything above accelerates topos's own execution. There is a
second road, for serving: a compiled forward plan is a closed, pure
tensor function, and `Plan::emit_stablehlo` writes it down as a
textual StableHLO module that any XLA-world runtime compiles and
runs. Nothing links in-crate — the boundary is text, and the
runners are scripts in [tools/](tools/) driven by two environment
variables the test suite also honors (`TOPOS_STABLEHLO_VALIDATOR`
parses every emitted module, `TOPOS_STABLEHLO_EVALUATOR` executes
it and checks the results against the interpreter's own).

Measured steady state on the same machine, modules emitted from
real plans, compile costs (16-121 ms) amortized:

| per run | topos plan (`accelerate`) | XLA-CPU | XLA on the GPU (`jax-metal`) | reference interpreter |
|---|---|---|---|---|
| batch-8 CNN forward | 2.6 ms | 0.24 ms | 5.4 ms | 3.0 s |
| 256-token attention block | 0.46 ms | 0.37 ms | 0.65 ms | 6.3 s |

Three readings worth their lines:

- The CNN row is the emission story's whole argument in one number:
  the same tape serves eleven times faster by writing the plan down
  and letting XLA's threaded convolution kernels run it. Training
  and inspection stay at home, where the oracle and the determinism
  contract live; serving rides an industrial compiler for free.
- The attention row says the interpreter is within a quarter of XLA
  on GEMM-bound work — both ride BLAS, and the view paths above
  keep everything around the products out of the way.
- XLA-on-Metal loses to every CPU path at these sizes (Apple's
  PJRT plugin is experimental and version-frozen), and the
  reference interpreter is a specification tool, three orders of
  magnitude off — each is a conformance target, not a runner.

Every emitted module is checked twice before any of this: an
external parser must accept the text, and the StableHLO reference
interpreter — the specification's executable semantics — must
reproduce the plan's results within an envelope. Both checks run in
the ordinary test suite whenever the toolchain (any Python with
`jax`) is present, and pass vacuously when it is not.

## Determinism

Results are a pure function of the payloads, the compiled features,
the numerics posture, and the machine. Within one binary, two
identical runs can never disagree — there is nothing that could
change between them, and the test suite verifies the hardware paths
down to the bit (repeated products compare bitwise). What a backend
build's `Fast` runs forfeit is bit-identity *with the built-in
paths*: hardware sums and rounds in its own order (AMX partitions
sums, vForce rounds differently than libm, the simd kernels pack
and re-associate — though single-threaded, so nothing varies
between runs), the same way any BLAS differs from a textbook loop.
`Numerics::Exact` restores the built-in bits inside the same binary
— a labeled per-request choice, so the reference result is always
one compile away and comparable to the fast one in one process (the
test suite asserts exactly this over a supra-threshold product).
This also defuses the one cargo caveat: features unify across a
dependency graph, so any dependency enabling `topos/accelerate`
enables it for the whole binary — but an `Exact` request computes
reference bits regardless of what the build unified.

## Safety

The default build keeps `#![forbid(unsafe_code)]` verbatim. A
backend build drops `forbid` but keeps a crate-wide
`#![deny(unsafe_code)]`, with exactly one scoped `allow` per
backend module — `unsafe` outside them stays a compile error, and
what is inside is the FFI boundary itself: cblas and vForce calls
with their safety arguments written out, the Metal encode path, the
two `matrixmultiply` call sites, and the cuda module's dlopen and
call boundary. `accelerate` adds no crates; `metal` adds the
`objc2` binding family on macOS targets only; `simd` adds
`matrixmultiply` and its single helper crate; `cuda` adds
`libloading` on Linux targets only.

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
