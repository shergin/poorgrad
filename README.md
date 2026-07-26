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

Early scaffolding. The core types:

- [`Value`](src/value.rs) — a cheap proxy to a value allocated in a `Network`,
  and the only graph handle in the public API. Arithmetic operators on
  proxies build the graph: `let x = v1 + v2;` allocates a new computed node
  on the same network, without cloning it.
- [`Network`](src/network.rs) — a memory management bag owning the values of
  a graph, backed by the arena-based
  [`cow_vec`](https://crates.io/crates/cow_vec) crate: allocation is
  append-only, cloning forks the network in O(1), and the whole structure is
  `Send + Sync`.
- [`ValueInner`](src/value_inner.rs) and [`Function`](src/function.rs) —
  internal: the stored node, and the operation that produced it referencing
  its inputs by index.
- [`Neuron`](src/neuron.rs) — the smallest learnable building block
  (placeholder).

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
