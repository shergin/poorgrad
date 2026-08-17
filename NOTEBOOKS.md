# Notebooks

topos runs in [Evcxr](https://github.com/evcxr/evcxr), the Rust
Jupyter kernel and REPL, with no wrapper API and no separate build. The
`evcxr` feature adds rich cell output and two leaking constructors;
everything else in a notebook is the ordinary crate.

```sh
cargo install --locked evcxr_jupyter
evcxr_jupyter --install
```

Then, in the first cell:

```rust
:dep topos = { version = "0.10", features = ["evcxr"] }
use topos::*;
```

A `~/.config/evcxr/init.evcxr` saves typing it every session, and the
compilation cache is worth turning on — cells recompile a real crate,
so the difference is felt immediately:

```
:cache 500
:dep topos = { version = "0.10", features = ["evcxr"] }
```

## The two rules

Evcxr compiles every cell as its own crate and carries the variables
between them. Two consequences shape everything else.

**A persisted variable cannot borrow another one.** A `Value` proxy
borrows the network that recorded it, so this is rejected the moment
the cell ends:

```rust
let network = Network::new();
let w = network.parameter(0.0);   // error: `w` borrows `network`
```

Leak the network instead. A `&'static Network` borrows from nothing, so
its proxies are `Value<'static, _>` and persist like any other value:

```rust
let mut network: &'static Network<f64> = Network::leaked();
let w: Value<'static, f64> = network.parameter(0.0);
let x: Value<'static, f64> = network.input(0.0);
let y: Value<'static, f64> = network.input(0.0);
let loss: Value<'static, f64> = (w * x - y) * (w * x - y);
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

## Training across cells

`Network::update` returns the next generation, and a proxy stays bound
to the generation that recorded it. Keep one owned generation for the
loop and leak once at the end, so re-running the cell costs one
parameter store rather than one per step:

```rust
let samples: Vec<(f64, f64)> = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];

{
    let first = network.forward_with([(x.symbol(), 1.0), (y.symbol(), 2.0)]).backward(loss);
    let mut current = network.update(&first, |p, g| p - 0.02 * g);
    for step in 1..300 {
        let (sx, sy) = samples[step % 3];
        let target = current.resolve(loss.symbol());
        let gradients = current.forward_with([(x.symbol(), sx), (y.symbol(), sy)]).backward(target);
        current = current.update(&gradients, |p, g| p - 0.02 * g);
    }
    network = current.leak();
}
```

Afterwards both generations are still in scope, and the difference is
the point:

```rust
w.payload()                              // 0.0  — the generation that recorded it
network.resolve(w.symbol()).payload()    // 2.0  — the generation that trained
```

A proxy is generation-bound and a `Symbol` is not. That is not a
notebook quirk to work around; it is the contract the generation
machinery was built for, and a notebook is the first place it becomes
something you can see rather than something you read about.

## What leaking costs

`Network::leaked` and `Network::leak` never free. The recorded graph is
shared between generations rather than copied, so one leaked generation
costs its parameter store and nothing else. A session that leaks once
per cell run stays negligible, and the process reclaims all of it when
the notebook shuts down.

The one mistake worth naming: leaking *inside* a training loop rather
than after it leaks a parameter store per step.

## Cell output

With the `evcxr` feature on, ending a cell with a topos value draws
it instead of dumping `Debug`:

| Ends with | Shows |
| --- | --- |
| a `Value` | its shape, extremes, and payload — a table when small, a chart when large |
| a `Tensor` payload | the same, straight from the payload |
| a `Network` | how much graph is recorded |
| a `Plan` | the whole schedule, and the live volume plotted along it |
| a `Field` of gradients | one Euclidean norm per node, plotted along the tape |
| a `Run` | the same profile for a completed forward pass |
| a `Symbol` | what it is, and that resolving is how you read through it |

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
- **`Run` is owned.** Like a `Field`, it carries structure and a
  witness rather than borrowing the network, so a run can outlive the
  cell that produced it (or the generation that ran).
