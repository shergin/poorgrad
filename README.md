# poorgrad

<p align="center">
  <img src="poorgrad.png" alt="poorgrad logo" width="480">
</p>

**A fully concurrent, thread-safe autograd engine, written the way Rust
wants it written. CPU-only by default; teraflop-class when you flip a
flag. Small enough to audit; deterministic enough to ship.**

`poorgrad` begins from
[Karpathy's `micrograd`](https://github.com/karpathy/micrograd) and then
takes the road the others don't: no `Rc<RefCell<...>>`, no single-threaded
assumption, no graph rebuilt on every pass, and a payload generic over
scalars and tensors alike. Sharing a computation graph across threads is
not a feature bolted on with locks; it is what the types guarantee.

The discipline is the product: three dependencies,
`#![forbid(unsafe_code)]` unless you opt into the two audited FFI
backends, shape errors surfaced when an expression is recorded —
before anything runs — seeded runs bit-identical forever with
golden-bit tests holding the line, and a suite on dual-platform CI
that checks the hardware paths down to the last bit.

## The bet

Most autograd engines are define-by-run: the graph is built dynamically as
code executes, mutated in place, and single-threaded by assumption.
`poorgrad` goes the other way, and everything below follows from that one
choice:

- **Record once, run anywhere.** Expressions record a static tape;
  `forward` replays an O(1) snapshot of it, and `backward` replays the
  evaluation's own copy, so runs never lock the graph and never disturb
  each other. Declared inputs take per-run payloads through
  `forward_with` — feeds are run state, not graph state — so one shared
  network serves any number of threads, each feeding its own data and
  differentiating its own target.
- **Values are `Copy`.** A `Value` is a borrow of its network plus a
  position: operators never consume their operands, handles cross threads
  freely, and a value outliving its graph is a compile error, not a bug
  report. Every type's `Send + Sync` contract is asserted at compile time.
- **Mutation is a state transition.** A gradient step produces the next
  network generation in O(parameters): the parameter store is rebuilt,
  the recorded structure is shared untouched through an append-only
  arena, replaced payloads are reclaimed with their generation, and
  older generations stay fully usable. Snapshot isolation, for networks.
- **Performance falls out of structure.** One `Mutex` in the engine
  itself, taken briefly per operation — the arena inside
  [`cow_vec`](https://crates.io/crates/cow_vec) holds the only other,
  and training never touches it; O(1)
  forks; dense matrix products on a slice path shaped for the
  compiler's auto-vectorizer — tens of GFLOP/s on Apple Silicon,
  bit-identical to the logical definition — with a documented seam
  (`Elementary::gemm` over a `GemmTask`) through which an element
  type can route dense products to its own kernel;
  no `unsafe` in this crate, with `#![forbid(unsafe_code)]` keeping
  it a promise rather than a claim (the arena's `unsafe` core is
  `cow_vec`'s, encapsulated behind its tested interface). CPU-only by
  default, on purpose: the engine is the point — and the claims are
  measured, not asserted: `cargo bench` runs the suite.

## Acceleration

Opt-in cargo features route dense math to the hardware a Mac
already owns: `accelerate` (the AMX/SME matrix units through
`cblas`, vForce for whole-buffer transcendentals) and `metal` (the
crate's own simdgroup GPU kernels). Enabling a feature is the whole
activation — the same training source spans a factor of four
thousand, from the 0.4 GFLOP/s naive definition to AMX's measured
1.6 TFLOP/s `f32`, with zero source changes:

```sh
cargo run --release --features accelerate,metal --example throughput
```

What each build supports, how routing and determinism work, and
every measured number: [ACCELERATION.md](ACCELERATION.md).

## Where it fits

- **Rust services that learn in production.** Train, fine-tune, or
  calibrate dense models inside the process that serves them —
  personalization weights, ranking adjustments, anomaly
  thresholds — without a Python runtime, a framework heavier than
  the service, or a model file crossing a process boundary.
- **Serving and training the same network, concurrently.** Runs
  never lock the graph: one shared network answers inference on
  every thread while a training loop steps generations in the
  background, and moving to the next generation is swapping a
  value — snapshot isolation, for models.
- **Reproducibility as a requirement, not a hope.** No `rand`, no
  clocks, seeds all the way down: a model trained in CI is
  evidence, a rerun experiment is a checksum, and a regression
  bisect converges, because two identical runs cannot differ.
- **Scientific and financial fitting in `f64`.** First-class double
  precision with hardware acceleration behind it — 550 GFLOP/s
  measured through the `accelerate` feature — for calibration,
  curve fitting, and gradient-based optimization where `f32`
  rounding is a liability. Accelerated `f64` is the exception, not
  the rule, in ML stacks.
- **Parallel what-ifs on O(1) forks.** Forking a network copies
  nothing, so training several learning rates or data shards in
  parallel over shared structure is the
  [threaded example](examples/gradient_descent.rs), not an
  architecture project.

## A taste

```rust
use poorgrad::Network;

let network = Network::new();
let w = network.parameter(0.0_f64);
let x = network.input(0.0);
let y = network.input(0.0);

// Operators record the graph; values are `Copy` and never consumed.
let error = w * x - y;
let loss = error * error;

let w_symbol = w.symbol();
let x_symbol = x.symbol();
let y_symbol = y.symbol();
let loss_symbol = loss.symbol();

// The graph is recorded once; every step feeds one sample of the line
// `y = 2 * x` and steps to the next generation, which shares the recorded
// graph while replacing the parameter payloads.
let samples = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
let mut network = network;
for step in 0..100 {
    let (sample_x, sample_y) = samples[step % samples.len()];
    let loss = network.resolve(loss_symbol);
    let evaluation = network.forward_with([(x_symbol, sample_x), (y_symbol, sample_y)]);
    let gradients = evaluation.backward(loss);
    network = network.update(&gradients, |w, g| w - 0.02 * g);
}

let learned = network.resolve(w_symbol).payload().unwrap();
assert!((learned - 2.0).abs() < 1e-6);
```

The [threaded example](examples/gradient_descent.rs) goes further:
per-sample gradients computed on separate rayon threads over one shared
network, then three learning rates trained in parallel on O(1) forks.

## What's inside

The engine builds, evaluates, differentiates, and trains computation
graphs over a generic payload — scalars or tensors alike. The complete
machinery is grouped into private [`payload`](src/payload/),
[`engine`](src/engine/), and [`neural`](src/neural/) modules while the
crate root keeps the public API flat. From tape to training:

- [`Value`](src/engine/value.rs) — a `Copy` proxy to a value allocated in a
  `Network`, and the only graph handle in the public API. It borrows the
  network, so it cannot outlive it. Arithmetic operators build the graph
  (`let x = v1 + v2;` allocates a new computed node on the same network) and
  never consume their operands.
- [`Network`](src/engine/network.rs) — the single owner of the state of
  every value of a graph, backed by the arena-based
  [`cow_vec`](https://crates.io/crates/cow_vec) crate: allocation is
  append-only, cloning forks the network in O(1), and the whole structure is
  `Send + Sync`. A gradient step is a state transition: `update` produces
  the next generation, rebuilding only the parameter store while sharing
  everything else. `input` declares a per-run input with a default
  payload; `forward_with` binds fed payloads to inputs for one run,
  validated against their recorded shapes.
- [`Symbol`](src/engine/symbol.rs) — a detached, `Copy` identifier for a value.
  `Network::resolve` turns it into a proxy in a compatible generation, while
  rejecting unrelated or divergent networks. Training loops keep symbols of
  the loss and parameters across `update` steps.
- [`Evaluation`](src/engine/evaluation.rs) and
  [`Gradients`](src/engine/field.rs) — the per-run results of `forward`
  and `backward`, read back with the same `Value` proxies that built the
  graph. Runs never mutate the network, so any number of them can execute
  concurrently.
- [`Field`](src/engine/field.rs) — a value-aligned buffer tied to a network
  lineage rather than one generation, with elementwise algebra (`+`,
  `scale`, `zip`, `map`). `Gradients` is an alias for it: the field one
  backward run produces, combined across runs and carried across generations
  as optimizer state (momentum, Adam) with no conversion; `update` takes a
  compatible field covering the current graph as its update direction.
- [`Tensor`](src/payload/tensor.rs) — the built-in tensor payload: an
  immutable runtime shape over a shared element buffer read through a
  strided layout, so `transpose` and the broadcasts are O(1) views rather
  than copies (tensors are immutable, so aliasing a buffer is always safe).
  Its storage is an extensible representation — a dense `Arc`-shared buffer
  or a non-allocating constant today, with room for more — and elements are
  read with `iter`, `as_slice`, or `to_vec`. A `Network<Tensor<f64>>` uses
  the same graph, evaluation, differentiation, and update APIs as a scalar
  network. The
  [`Tensorial`](src/payload/tensorial.rs) trait provides `matmul`,
  `transpose`, the reductions `sum` and `sum_along`, and the explicit
  broadcasts `broadcast_like` and `broadcast_along` (scalars implement
  scalar semantics for the same trait bound). Broadcasting
  is explicit by design: a single value spread across a named
  reference's shape, or a payload repeated along one named axis — never
  an implicit alignment rule. Shapes are inferred and checked when
  expressions are recorded — a shape mismatch
  panics at the offending line, before anything runs: the record-once
  answer to type-level shape checking.
- [`Tape`](src/engine/tape/tape.rs) — internal: the append-only record (a
  Wengert list) shared by a network and all of its proxies, and the engine's
  single synchronization point.
- [`Function`](src/engine/function/mod.rs) — internal: a statically sized
  enum of the differentiable operations, each variant owning its operand
  links and parameters and implementing the `Operation` trait (forward math
  and gradient routing per operation, dispatched with a plain `match`).
- [`Neuron`](src/neural/neuron.rs) — a scalar-granularity affine unit with
  weights, a bias, and an `Activation`. Its parameters are allocated on the
  network and retained as symbols across compatible generations.
- [`Layer`](src/neural/layer.rs) — a dense layer at tensor granularity:
  `activation(x.matmul(w) + b)` over a `[batch, inputs]` value, one weight
  matrix and one bias vector, with the bias explicitly broadcast over the
  batch axis.
- [`Mlp`](src/neural/mlp.rs) — dense layers described by a sequence of widths
  such as `[3, 4, 4, 1]`: tanh hidden layers, an affine output, and
  caller-controlled initialization from each parameter's requested shape.
- [`cross_entropy`](src/neural/loss.rs) — the classification loss as a
  composed formula over the fused, numerically stable `Value::log_softmax`:
  the mean negative log-likelihood of one-hot (or soft) targets, fed per
  run like any other input.
- [`init`](src/neural/init.rs) — deterministic, explicitly seeded
  initializer factories (`uniform`, `normal`, and the fan-aware `xavier`
  and `kaiming`) producing the shape-to-payload closures `Layer` and `Mlp`
  take, with no `rand` dependency: seeded runs stay bit-identical forever.

## The name

A poor man's autograd: no GPU required, none wanted — and the poor, it
turns out, were sitting on a matrix coprocessor the whole time. The
name is the only modest thing about the design.

## Terminology

The vocabulary used across code and docs — the scientific meaning of each
term and its mapping to the Rust types — is collected in
[TERMINOLOGY.md](TERMINOLOGY.md).

## Examples

- [`chain`](examples/chain.rs) — build a small expression graph by chaining
  `Value` proxies with arithmetic operators, then evaluate it and compute
  gradients: `cargo run --example chain`.
- [`gradient_descent`](examples/gradient_descent.rs) — fit `w * x + b` to a
  line, threaded with rayon: one shared network differentiated for
  per-sample targets on separate threads, then trained in parallel on O(1)
  forks, one per learning rate: `cargo run --example gradient_descent`.
- [`mlp_xor`](examples/mlp_xor.rs) — train a tanh MLP on XOR at tensor
  granularity: the graph is recorded once, and every training step feeds
  a different minibatch through `forward_with` while the tape never
  grows: `cargo run --example mlp_xor`.
- [`makemore_bigram`](examples/makemore/bigram.rs) — train a
  character-level bigram language model on names and sample new ones: a
  `[vocab, vocab]` logit table read by `gather`, scored by
  `cross_entropy`, fed one-hot minibatches per run, and sampled through
  the composite `softmax`: `cargo run --example makemore_bigram`.
- [`makemore_mlp`](examples/makemore/mlp.rs) — the Bengio-style sequel:
  a three-character context embedded by `gather`, flattened by
  `reshape`, and pushed through a tanh hidden layer hand-rolled from
  raw parameters, with a single-row twin expression of the same
  parameters recorded for sampling. Beats the bigram's loss:
  `cargo run --release --example makemore_mlp`.
- [`makemore_mlp_facade`](examples/makemore/mlp_facade.rs) — the same
  model with the hand-rolled layers replaced by `Mlp`; matching seeds
  make it train bit-identically to `makemore_mlp`, demonstrating that
  the facade is packaging, not different math:
  `cargo run --release --example makemore_mlp_facade`.
- [`makemore_mlp_parallel`](examples/makemore/mlp_parallel.rs) — the
  same model trained data parallel: each step fans its minibatch out as
  eight shard-sized runs on the shared network and averages the
  gradient fields, cutting the wall clock several-fold while computing
  the same batch gradient:
  `cargo run --release --example makemore_mlp_parallel`.
- [`makemore_embedding_map`](examples/makemore/embedding_map.rs) — the
  MLP with a two-dimensional embedding, drawn in the terminal before
  and after training by a hand-rolled labeled scatter chart: watch the
  vowels drift into their own cluster:
  `cargo run --release --example makemore_embedding_map`.
- [`throughput`](examples/throughput.rs) — the acceleration ladder
  measured on a wide dense model: raw 2048-square products and whole
  training steps, with the dimensions shrinking eightfold when no
  backend is compiled in:
  `cargo run --release --features accelerate,metal --example throughput`.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
