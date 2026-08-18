# Notebooks

topos runs in [Evcxr](https://github.com/evcxr/evcxr), the Rust
Jupyter kernel and REPL, with no wrapper API and no separate build. The
`evcxr` feature adds rich cell output; everything else in a notebook is
the ordinary crate.

```sh
cargo install --locked evcxr_jupyter
evcxr_jupyter --install
```

Then, in the first cell:

```rust
:dep topos = { version = "0.11", features = ["evcxr"] }
use topos::*;
```

A `~/.config/evcxr/init.evcxr` saves typing it every session, and the
compilation cache is worth turning on — cells recompile a real crate,
so the difference is felt immediately:

```
:cache 500
:dep topos = { version = "0.11", features = ["evcxr"] }
```

## The two rules

Evcxr compiles every cell as its own crate and carries the variables
between them. Two consequences shape everything else.

**A persisted variable cannot borrow another one.** A `Value` proxy
borrows the tape that recorded it, so it lives and dies inside one
cell; the detached `Symbol` is the cross-cell currency. End a
recording cell with `.symbol()` bindings and reenter through
`Tape::resolve`:

```rust
let tape: Tape<f64> = Tape::new();
let w: Symbol = tape.parameter(0.0).symbol();
let x: Symbol = tape.input(0.0).symbol();
let y: Symbol = tape.input(0.0).symbol();
```

```rust
let loss: Symbol = {
    let (w, x, y) = (tape.resolve(w), tape.resolve(x), tape.resolve(y));
    ((w * x - y) * (w * x - y)).symbol()
};
```

**A persisted variable needs an explicit type.** Evcxr works out a
cell's variable types by compiling that cell alone, and a later cell
cannot inform an earlier one. This is a property of Evcxr rather than
of this crate — it cannot infer `let v = vec![1.0_f64];` either — so
annotate every binding that has to survive the cell.

The annotations look heavier than they are. They appear once per
binding in small teaching graphs, where spelling the type out is worth
doing anyway; a real model goes through `Module` and reads its
parameters back through `named_parameters`, which is one annotated
binding for the whole network.

Give each one its own `let`. Evcxr persists a destructured tuple of
primitives, but it works out the types of a tuple's parts by inference
rather than from the annotation, and it cannot name a type that came
from a dependency — so `let (w, x): (Symbol, Symbol) = …;` binds
nothing that survives the cell, however it is annotated.

## Sealing and training

`Tape::into_network` consumes the persisted tape — Evcxr tracks the
move, so the `tape` variable simply ends where the `network` one
begins. The network is the immutable spec; the parameters are yours,
and training is a pure data loop:

```rust
let network: Network<f64> = tape.into_network();
let mut parameters: Parameters<f64> = network.parameters();
```

```rust
let samples: Vec<(f64, f64)> = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
for step in 0..300 {
    let (sx, sy) = samples[step % 3];
    let gradients = network.forward(&parameters, [(x, sx), (y, sy)]).backward(loss);
    parameters = parameters.step(&gradients, |p, g| p - 0.02 * g);
}
```

Afterwards the trained payload reads by name, and a fresh
materialization still answers the record-site initial:

```rust
parameters.of(w)             // 2.0  — the trained state
network.parameters().of(w)   // 0.0  — the initials, materialized fresh
```

Access is phase-scoped and names are forever: a `Value` dies at the
seal, a `Symbol` resolves through every phase. A notebook is the first
place that contract becomes something you can see rather than
something you read about.

## Recording more later

`Network::into_tape` consumes the network and reopens recording —
another tracked move. Symbols keep resolving (linear extension never
moves a node), and `Parameters::carried` moves the trained state
across, seeding any new slots from their record-site initials:

```rust
let tape: Tape<f64> = network.into_tape();
let cube: Symbol = { let w = tape.resolve(w); (w * w * w).symbol() };
let network: Network<f64> = tape.into_network();
let parameters: Parameters<f64> = parameters.carried(&network);
```

## Cell output

With the `evcxr` feature on, ending a cell with a topos value draws
it instead of dumping `Debug`:

| Ends with | Shows |
| --- | --- |
| a `Value` | its shape, extremes, and payload — a table when small, a chart when large |
| a `Tensor` payload | the same, straight from the payload |
| a `Tape` | how much graph is recorded so far |
| a `Network` | the sealed spec's node count |
| `Parameters` | how many slots the state carries |
| a `Plan` | the whole schedule, and the live volume plotted along it |
| a `Field` of gradients | one Euclidean norm per node, plotted along the tape |
| a `Run` | the same profile for a completed forward pass |
| a `Symbol` | what it is, and how to read through it |

Every card is a pure `to_html(Theme)` string with an `evcxr_display`
that emits it, so cell output is covered by ordinary `cargo test` and
the terminal REPL — which cannot draw HTML — gets a readable
`text/plain` alternative of the same value. Charts are drawn by
[malevich](https://crates.io/crates/malevich), the same renderer the
examples use.

Supplying `evcxr_display` also makes cells compile once instead of
twice: Evcxr tries `(expr).evcxr_display();` first and only falls back
to a second compile with `Debug` formatting when that fails.

## Known rough edges

- **A shape mistake panics the cell.** The panic-on-misuse contract is
  unchanged in a notebook. Evcxr catches the panic and the session
  survives with its variables; only that cell's work is lost.
- **Cells are a real compile.** Expect roughly a quarter of a second for
  a small cell and one to two seconds when generics instantiate. The
  compilation cache above helps; nothing makes it Python-snappy, and
  that is the honest trade for the rest of the stack.
- **`Run` is owned.** Like a `Field`, it carries its own structure
  freeze rather than borrowing the network, so a run can outlive the
  cell that produced it.
