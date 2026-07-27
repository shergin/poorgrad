# poorgrad

**A fully concurrent, thread-safe autograd engine, written the way Rust
wants it written.**

`poorgrad` begins from
[Karpathy's `micrograd`](https://github.com/karpathy/micrograd) and then
takes the road the others don't: no `Rc<RefCell<...>>`, no single-threaded
assumption, no graph rebuilt on every pass, and a payload generic over
scalars and tensors alike. Sharing a computation graph across threads is
not a feature bolted on with locks; it is what the types guarantee.

## The bet

Most autograd engines are define-by-run: the graph is built dynamically as
code executes, mutated in place, and single-threaded by assumption.
`poorgrad` goes the other way, and everything below follows from that one
choice:

- **Record once, run anywhere.** Expressions record a static tape;
  `forward` replays an O(1) snapshot of it, and `backward` replays the
  evaluation's own copy, so runs never lock the graph and never disturb
  each other. One shared network serves any number of threads, each
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
  forks; no `unsafe` in this crate, with `#![forbid(unsafe_code)]` keeping
  it a promise rather than a claim (the arena's `unsafe` core is
  `cow_vec`'s, encapsulated behind its tested interface). CPU-only, on
  purpose: the engine is the point — and the claims are measured, not
  asserted: `cargo bench` runs the suite.

## A taste

```rust
use poorgrad::Network;

let network = Network::new();
let w = network.parameter(0.0_f64);
let x = network.leaf(3.0);
let y = network.leaf(15.0);

// Operators record the graph; values are `Copy` and never consumed.
let error = w * x - y;
let loss = error * error;

let w_symbol = w.symbol();
let loss_symbol = loss.symbol();

// A training step is a state transition: each generation shares
// everything but the parameters with the one before it.
let mut network = network;
for _ in 0..100 {
    let loss = network.resolve(loss_symbol);
    let evaluation = network.forward();
    let gradients = evaluation.backward(loss);
    network = network.updated(gradients.as_field(), |w, g| w - 0.01 * g);
}

let learned = network.resolve(w_symbol).data().unwrap();
assert!((learned - 5.0).abs() < 1e-6);
```

The [threaded example](examples/gradient_descent.rs) goes further:
per-sample gradients computed on separate rayon threads over one shared
network, then three learning rates trained in parallel on O(1) forks.

## What's inside

The engine builds, evaluates, differentiates, and trains computation
graphs over a generic payload — scalars or tensors alike. The complete
machinery, from tape to training:

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
  the next generation, rebuilding only the parameter store while sharing
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
- [`Tensor`](src/tensor.rs) — the first non-scalar payload: dense and
  fixed-shape, with O(1) clones behind `Arc`s. A `Network<Tensor<f64>>`
  runs the whole engine — training loop, fields, momentum — unchanged,
  and the [`Tensorial`](src/tensorial.rs) tier adds `matmul`,
  `transposed`, `sum`, and the explicit `broadcast_like` (scalars
  implement it degenerately, so one bound covers both worlds).
  Broadcasting is explicit by design: a single value spread across a
  named reference's shape, never an implicit alignment rule. Shapes are
  inferred and checked when expressions are recorded — a shape mismatch
  panics at the offending line, before anything runs: the record-once
  answer to type-level shape checking.
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

## The name

A poor man's autograd: no GPU required, none wanted. The name is the only
modest thing about the design.

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
