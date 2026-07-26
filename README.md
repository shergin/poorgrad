# poorgrad

**An experiment in what a fully concurrent, thread-safe autograd engine could
look like in idiomatic Rust.**

`poorgrad` is a small, self-educational scalar automatic differentiation engine,
loosely inspired by [Karpathy's `micrograd`](https://github.com/karpathy/micrograd).
It is deliberately experimental: it exists to explore a design question, not to
be a production framework or to compete with anything.

## The question

What would a *completely concurrent and thread-safe* autograd library look like
if it were written the way Rust wants it to be written?

Most autograd engines are define-by-run: the graph is built dynamically as code
executes, mutated in place, and single-threaded by assumption. `poorgrad` bets
the other way:

- **Mostly static.** The computation graph is described up front and then
  executed, rather than rebuilt on every pass. A graph that is known ahead of
  time is far easier to schedule, parallelize, and optimize.
- **Completely concurrent and thread-safe.** The graph is safe to share and
  evaluate across threads by construction, not as an afterthought bolted on with
  locks.
- **Hyper-optimized, but idiomatic.** The goal is to be fast *without* leaning on
  pervasive `unsafe` or un-Rust-like tricks. Arena-backed nodes, copy-on-mutation
  state, and ownership that models the data flow — performance that falls out of
  good structure rather than fighting it.
- **CPU-only.** No GPU, no accelerators. The interesting part is the engine
  itself.

## The name

`poorgrad` is a poor man's autograd — the joke being "no GPU required." That is
still true, but the real reason it exists is the question above.

## Status

The engine builds, evaluates, differentiates, and trains scalar graphs.
The core types:

- [`Value`](src/value.rs) — a `Copy` proxy to a value allocated in a
  `Network`, and the only graph handle in the public API. It borrows the
  network, so it cannot outlive it. Arithmetic operators build the graph
  (`let x = v1 + v2;` allocates a new computed node on the same network) and
  never consume their operands.
- [`Network`](src/network.rs) — the single owner of the state of every value
  of a graph, backed by the arena-based
  [`cow_vec`](https://crates.io/crates/cow_vec) crate: allocation is
  append-only, cloning forks the network in O(1), and the whole structure is
  `Send + Sync`. A gradient step is a state transition: `updated` produces
  the next generation, replacing only the parameter leaves while sharing
  everything else.
- [`Symbol`](src/symbol.rs) — a detached, `Copy` name of a value: the
  identity that persists across network generations, while a proxy is that
  identity's view in one generation. `Network::resolve` looks a symbol up
  in a generation. Training loops keep symbols of the loss and the
  parameters across `updated` steps.
- [`Evaluation`](src/evaluation.rs) and [`Gradients`](src/gradients.rs) —
  the per-run results of `forward` and `backward`, read back with the same
  `Value` proxies that built the graph. Runs never mutate the network, so
  any number of them can execute concurrently.
- [`Field`](src/field.rs) — a value-aligned buffer tied to a network
  lineage rather than one generation, with elementwise algebra (`+`,
  `scaled`, `zip`, `map`). Gradients convert into fields to be combined
  across runs and carried across generations as optimizer state (momentum,
  Adam); `updated` takes any field as its update direction.
- [`Tape`](src/tape.rs) — internal: the append-only record (a Wengert list)
  shared by a network and all of its proxies, and the engine's single
  synchronization point.
- [`Function`](src/function/mod.rs) — internal: a statically sized enum of
  the differentiable operations, each variant owning its operand links and
  parameters and implementing the `Operation` trait (forward math and
  gradient routing per operation, dispatched with a plain `match`).
- [`Neuron`](src/neuron.rs) — the smallest learnable building block:
  weights and a bias (allocated as parameters, held as symbols so the
  neuron survives generations) plus an `Activation`; `express` records
  `activation(weights . inputs + bias)` against a given generation.
- [`Layer`](src/layer.rs) — a dense row of neurons sharing the same
  inputs, one output value per neuron; layers chain by feeding one
  layer's outputs to the next.

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

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
