# poorgrad

<p align="center">
  <img src="https://raw.githubusercontent.com/shergin/poorgrad/master/poorgrad.png" alt="poorgrad logo" width="480">
</p>

**An autodiff compiler stack in miniature, written the way Rust wants
it written: record a graph once, inspect every node, lower it to a
plan you can read, verify every optimization against the interpreter
that ships inside — and emit StableHLO when you want the XLA world's
muscle. Small enough to learn from; rigorous enough to catch a vendor
compiler computing GPT-2 wrong.**

`poorgrad` begins from
[Karpathy's `micrograd`](https://github.com/karpathy/micrograd) and then
takes the road the others don't: no `Rc<RefCell<...>>`, no single-threaded
assumption, no graph rebuilt on every pass, and a payload generic over
scalars and tensors alike. Sharing a computation graph across threads is
not a feature bolted on with locks; it is what the types guarantee.

Where micrograd teaches autodiff by being small, `poorgrad` also
teaches what an ML compiler does. Eager frameworks hide the graph —
the schedule is the program; lazy frameworks hide the schedule — the
graph is whatever the scheduler did. `poorgrad` shows both and lets
you diff them: the tape is the specification, a compiled `Plan` is
the schedule, `Plan::describe()` prints every decision — dead-node
elimination, buffer liveness, fusion, rematerialization — and one
assert checks any of them against the interpreter, bit for bit.

The discipline is the product: no dependency doing poorgrad's own
work,
`#![forbid(unsafe_code)]` unless you opt into the FFI backends —
whose `unsafe` is scoped by a crate-wide `deny`, argued block by
block, and small enough to audit in one sitting — shape errors
surfaced when an expression is recorded, before anything runs,
seeded runs bit-identical forever with golden-bit tests holding the
line, and a suite on dual-platform CI that checks the hardware
paths down to the last bit.

## The bet

Most autograd engines are define-by-run: the graph is built dynamically as
code executes, mutated in place, and single-threaded by assumption.
`poorgrad` goes the other way, and everything below follows from that one
choice:

- **Record once, run anywhere.** Expressions record a static tape;
  `forward` replays an O(1) snapshot of it, and `backward` replays the
  run's own copy, so runs never lock the graph and never disturb
  each other. Declared inputs take per-run payloads through
  `forward_with` — feeds are run state, not graph state — so one shared
  network serves any number of threads, each feeding its own data and
  differentiating its own target.
- **The tape is the spec; a `Plan` is the schedule.** `compile` lowers
  the recorded graph into an execution plan — dead-node elimination
  against declared targets, buffer liveness, pattern fusion (the
  canonical im2col chain of a convolution executes as one window-GEMM
  call, never materialized), and opt-in rematerialization that trades
  backward time for memory — whose runs are bit-identical to the
  interpreter's and survive every generation of a training run. Every
  optimization's default was set by measurement, and where a trade
  did not always win it is a labeled option, never silent behavior.
  `Plan::describe()` prints the decisions, and the naive interpreter
  ships forever as the executable oracle every plan is tested against.
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
  no `unsafe` in the default build, with `#![forbid(unsafe_code)]`
  keeping it a promise rather than a claim (the optional backends
  open only their own scoped modules; the arena's `unsafe` core is
  `cow_vec`'s, encapsulated behind its tested interface). CPU-only by
  default, on purpose: the engine is the point — and the claims are
  measured, not asserted: `cargo bench` runs the suite.

## Acceleration

Opt-in cargo features route dense math to the hardware you already
own: `accelerate` (the AMX/SME matrix units through `cblas`, vForce
for whole-buffer transcendentals) and `metal` (the crate's own
simdgroup GPU kernels) on a Mac, and `simd` (tuned CPU microkernels
with runtime AVX-512/AVX2/NEON dispatch, measured 96 GFLOP/s `f32`)
on Linux and everything else. Enabling a feature is the whole
activation — the same training source spans a factor of four
thousand, from the 0.4 GFLOP/s naive definition to AMX's measured
1.6 TFLOP/s `f32`, with zero source changes:

```sh
cargo run --release --features simd --example throughput
cargo run --release --features accelerate,metal --example throughput
```

What each build supports, how routing and determinism work, and
every measured number: [ACCELERATION.md](ACCELERATION.md).

## Emission

A compiled forward plan is a closed, pure tensor function, and
`Plan::emit_stablehlo` writes it down as a textual StableHLO module
— the exchange dialect of the XLA world. The whole op set lowers,
matched convolution chains raise to `stablehlo.convolution`, and
every emitted module is checked twice against toolchains the crate
never links: an external parser must accept the text, and the
StableHLO reference interpreter must reproduce the interpreter's
own results (`tools/`, driven by two environment variables the test
suite honors).

On the emitted path, performance is XLA's and correctness is still
ours to check. Measured both ways: the same tape serves a
convolutional forward eleven times faster by handing the emitted
plan to XLA-CPU, and the `gpt2` example generates at 132 ms/token
through XLA against the tape's 195, reproducing its text exactly —
while Apple's experimental Metal plugin ran the same module at 26
ms/token and *wrong*, a verdict provable because the tape, XLA-CPU,
and the reference interpreter agree with each other. Training,
inspection, and the determinism contract stay at home. The numbers
and their readings: [ACCELERATION.md](ACCELERATION.md).

## Notebooks

The crate runs in [Evcxr](https://github.com/evcxr/evcxr), the Rust
Jupyter kernel, with no wrapper API: `Network::leaked()` hands back a
`&'static Network` so recorded proxies survive a cell boundary, and
the rest is the ordinary crate.

```rust
:dep poorgrad = { version = "0.10", features = ["evcxr"] }
use poorgrad::*;

let mut network: &'static Network<f64> = Network::leaked();
let w: Value<'static, f64> = network.parameter(0.0);
```

The `evcxr` feature also draws poorgrad's own types as cell output —
tensors as tables or heatmaps, gradients as a norm profile along the
tape, and a `Plan` as its whole schedule with the live volume plotted
beside it. Nothing about the core API changes: the feature adds
inherent methods to existing types and no new vocabulary, and the
charts come from the same `malevich` renderer the examples use.

The idiom, what leaking costs, and the rough edges:
[NOTEBOOKS.md](NOTEBOOKS.md).

## Where it fits

- **Learning what an ML compiler does.** The examples are a
  curriculum: chained scalar expressions, the makemore acts, LeNet
  on MNIST, a VGG-style convnet on CIFAR-10, a one-block
  transformer, and GPT-2 with the released weights — every stage of
  the stack (recording, differentiation, plans, fusion,
  rematerialization, emission) landed with a consumer that uses it
  and a measured number that grades it.
  [TERMINOLOGY.md](TERMINOLOGY.md) keeps the vocabulary honest, and
  `Plan::describe()` makes optimization something you read, not
  something you trust.
- **Systems research at legible scale.** The IR is ~30 documented
  operations; a new pass, transform, or numeric idea can be tried
  in a day and graded against a built-in bitwise oracle on real
  consumers. The stack is small enough to hold in your head and
  rigorous enough that a disagreement means something.
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
- **Rust services that learn in production.** Train, fine-tune, or
  calibrate dense models inside the process that serves them,
  without a Python runtime or a model file crossing a process
  boundary. Runs never lock the graph: one shared network answers
  inference on every thread while a training loop steps generations
  in the background, and parallel what-ifs ride O(1) forks — the
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
    let run = network.forward_with([(x_symbol, sample_x), (y_symbol, sample_y)]);
    let gradients = run.backward(loss);
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
  everything else — replaced payloads drop with that generation, so a
  linear train loop does not retain weights per step. Structure lives in
  the shared arena: train-only forks stay clean, but nodes recorded on a
  sibling after `clone` can pin arena memory until every sharer drops;
  [`Network::compacted`](src/engine/network.rs) rebuilds private arenas
  from the live nodes when that trade must be unwound. `input` declares a
  per-run input with a default payload; `forward_with` binds fed payloads
  to inputs for one run, validated against their recorded shapes;
  `forward_for` additionally slices the run to the ancestors of declared
  targets, so a tape carrying several expressions (a training batch and an
  evaluation twin) evaluates only the one the run is for — reads of
  skipped values fail loudly rather than answer with a placeholder.
- [`Symbol`](src/engine/symbol.rs) — a detached, `Copy` identifier for a value.
  `Network::resolve` turns it into a proxy in a compatible generation, while
  rejecting unrelated or divergent networks. Training loops keep symbols of
  the loss and parameters across `update` steps.
- [`Run`](src/engine/run.rs) and
  [`Gradients`](src/engine/field.rs) — the per-run results of `forward`
  and `backward`, read back with the same `Value` proxies that built the
  graph. Runs never mutate the network, so any number of them can execute
  concurrently.
- [`Plan`](src/engine/plan.rs) — a compiled execution schedule derived
  from the tape by `Network::compile` (forward-only: aggressive buffer
  liveness, refuses `backward`), `compile_training` (retain-all, exact
  gradients), or `compile_training_compact` (drops large intermediates
  and rematerializes them during `backward`, bit-exactly). Plans fuse
  recognized patterns — matching is structural, so hand-written
  compositions fuse identically to facade-recorded ones, and keep-set
  values are fusion barriers — and `describe()` renders the whole
  schedule: liveness spans, drop sets, fusion groups, and the static
  live-volume story.
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
  Its storage is an extensible representation — a dense `Arc`-shared buffer,
  a non-allocating constant, or a compact one-hot selection today, with room
  for more — and elements are read with `iter`, `as_slice`, or `to_vec`. A `Network<Tensor<f64>>` uses
  the same graph, evaluation, differentiation, and update APIs as a scalar
  network. The
  [`Tensorial`](src/payload/tensorial.rs) trait provides `matmul`,
  `transpose`, the reductions `sum` and `sum_along`, the explicit
  broadcasts `broadcast_like` and `broadcast_along`, the window pair
  `unfold`/`fold` behind convolution and pooling, and the fused
  `windowed_product` the plan tier's window-GEMM pattern executes
  (scalars implement
  scalar semantics for the same trait bound). Broadcasting
  is explicit by design: a single value spread across a named
  reference's shape, or a payload repeated along one named axis — never
  an implicit alignment rule. Shapes are inferred and checked when
  expressions are recorded — a shape mismatch
  panics at the offending line, before anything runs: the record-once
  answer to type-level shape checking.
- [`Tape`](src/engine/network/tape/tape.rs) — internal: the append-only record (a
  Wengert list) shared by a network and all of its proxies, and the engine's
  single synchronization point.
- [`Function`](src/engine/function/mod.rs) — internal: a statically sized
  enum of the differentiable operations, each variant owning its parameters
  and implementing the `Operation` trait (forward math and gradient routing
  per operation, dispatched with a plain `match`); operand links live in
  the tape's parallel operand column.
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
- [`BatchNorm`](src/neural/batch_norm.rs) — batch normalization over
  `[batch, features]` values: `express` normalizes by the batch's own
  statistics and returns them for running-estimate upkeep, while
  `express_with` normalizes by statistics fed per run — the training and
  inference modes of one layer, with the running estimates living in the
  training loop rather than on the tape.
- [`LayerNorm`](src/neural/layer_norm.rs) and
  [`RmsNorm`](src/neural/rms_norm.rs) — the stateless normalization
  siblings: per-sample statistics along the feature axis (standardize
  plus affine, or root-mean-square re-scaling alone), so there are no
  running estimates and one recorded expression serves training and
  inference alike.
- [`Conv2d`](src/neural/convolution.rs) — 2-D convolution over
  `[batch, channels, height, width]` values as a composed formula, not
  a primitive: padding, two sliding-window `unfold`s, and an im2col
  reshape route the whole computation into one rank-2 `matmul` on the
  accelerated GEMM path, and the gradient falls out of the chain rule.
  Torch-shaped `[filters, channels, kh, kw]` weights; stride and
  symmetric zero padding.
- [`max_pool` and `average_pool`](src/neural/pooling.rs) — spatial
  pooling over the same window view: the average via `mean_along`, the
  maximum via a left-biased `maximum` fold whose ties route
  deterministically to the earliest window position.
- [`cross_entropy`](src/neural/loss.rs) — the classification loss as a
  composed formula over the fused, numerically stable `Value::log_softmax`:
  the mean negative log-likelihood of one-hot (or soft) targets, fed per
  run like any other input.
- [`Optimizer`](src/neural/optimizer.rs) — the training-step
  strategy as an open, object-safe trait: `Sgd` (stateless), `Adam`
  (moments as `Field`s, bias correction exact via carried powers),
  and `AdamW` (decoupled decay, sparing rank-one parameters by
  default, any other policy via a predicate). Steps are pure field
  algebra, so identical runs are bit-identical, and gradients from
  a compiled plan drive the same trajectory as the engine's
  backward.
- [`init`](src/neural/init.rs) — deterministic, explicitly seeded
  initializer factories (`uniform`, `normal`, the fan-aware `xavier`
  and `kaiming`, and the gain-parameterized `scaled` behind them —
  pair it with any `Activation::gain()`) producing the
  shape-to-payload closures `Layer` and `Mlp` take, with no `rand`
  dependency: seeded runs stay bit-identical forever.

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
- [`makemore_mlp_compiled`](examples/makemore/mlp_compiled.rs) — the
  same model with its backward pass *recorded*:
  `Network::differentiate` appends the chain rule to the tape, one
  forward-only plan compiles loss and gradients together, and every
  training step is a single plan run with no backward pass. Matching
  seeds make it train bit-identically to `makemore_mlp`, at a lower
  memory peak — the plan's liveness frees what the gradient no
  longer needs: `cargo run --release --example makemore_mlp_compiled`.
- [`mnist`](examples/mnist/main.rs) — a LeNet-style convolutional
  network on MNIST: two conv/relu/max-pool stages and a dense head,
  every convolution one im2col + GEMM under the hood. Downloads and
  caches the four IDX files on first run, then reports test accuracy,
  per-step time, and the loss chart:
  `cargo run --release --example mnist`.
- [`cifar10`](examples/cifar10/main.rs) — a three-stage VGG-style
  convnet on real 32x32 color images, the plan tier's pressure
  consumer: a training plan compiled once serves every generation,
  and the forward-only probe plan's liveness keeps the 500-image
  accuracy probe's footprint flat. Downloads and caches the binary
  archive on first run: `cargo run --release --example cifar10`.
- [`makemore_mlp_batchnorm`](examples/makemore/mlp_batchnorm.rs) —
  makemore's third act: the same MLP with the hidden preactivation
  batch-normalized before the tanh (and its bias retired in favor of
  the learned shift). The training plan keeps the batch statistics
  readable so the loop can fold them into running estimates — plain
  payloads, fed to the single-row sampling twin per draw. At this
  shallow depth the final loss matches the plain MLP, as it should:
  the norm buys robustness to initialization, not loss:
  `cargo run --release --example makemore_mlp_batchnorm`.
- [`makemore_transformer`](examples/makemore/transformer.rs) — the
  attention act: a one-block pre-norm transformer over eight
  characters of context. The batch packs its samples into one token
  row so each head's attention is a single rank-2 matmul pair under
  a block-diagonal causal mask; heads join through `concat`,
  prediction rows come back through a one-hot `gather`, and
  `RmsNorm` feeds both residual branches. Beats the MLP acts'
  loss on the same last-position metric:
  `cargo run --release --example makemore_transformer`.
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
- [`gpt2`](examples/gpt2/main.rs) — text generation with OpenAI's
  released GPT-2 (124M) weights, the whole model recorded from the
  existing op surface: twelve pre-norm blocks of per-head attention
  under a causal mask, GELU held as scalar leaves, the tied head as
  the transposed embedding table, and one fixed-context sampling
  plan fed the token window per step. The checkpoint and tokenizer
  (byte-level BPE, hand-rolled like every algorithm an example exists
  to show) download and cache on first run. A third argument
  picks the engine: `tape` runs the plan at home, `xla` emits it as
  StableHLO and serves generation through a resident XLA process —
  measured faster than the tape and reproducing its text:
  `cargo run --release --features accelerate --example gpt2 -- "Once upon a time" 40 xla`.
  The full guide — engines, setup, the Metal cautionary tale — is
  [examples/gpt2/README.md](examples/gpt2/README.md).
- [`throughput`](examples/throughput.rs) — the acceleration ladder
  measured on a wide dense model: raw 2048-square products and whole
  training steps, with the dimensions shrinking eightfold when no
  backend is compiled in:
  `cargo run --release --features accelerate,metal --example throughput`
  (or `--features simd` off macOS).

## Contributing

Issues and pull requests are welcome — bug reports, feature ideas,
questions, confusing-documentation reports (those are bugs too),
and benchmark numbers from your machine all help. For larger
changes, opening an issue first is appreciated: designs here are
decided by measurement and written down before code, and a short
conversation up front saves reworking a PR. CI expects `cargo fmt`,
`cargo clippy --all-targets -- -D warnings`, and `cargo test` to
pass; matching that locally is the whole checklist.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
