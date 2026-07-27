# Terminology

The vocabulary used across poorgrad's code and docs. Each entry gives the
meaning of the term in the automatic-differentiation literature and how it
maps onto this crate's types. This file is part of the codebase contract:
when a concept is added, renamed, or changes meaning, update it in the same
change.

## Mathematics

**Automatic differentiation (autodiff, AD).** Computing exact derivatives of
a program by decomposing it into primitive operations with known local
derivatives and composing them via the chain rule. Distinct from numeric
differentiation (finite differences; approximate) and symbolic
differentiation (expression rewriting; can blow up). Poorgrad implements
reverse-mode AD over scalar programs.

**Reverse-mode AD (backpropagation).** The AD flavor that computes the
derivative of *one* output with respect to *all* inputs in a single backward
sweep costing about one forward evaluation. Its mirror image, forward mode,
computes one input against all outputs. Reverse mode wins for machine
learning (one loss, many parameters). In poorgrad:
[`Evaluation::backward`](src/evaluation.rs).

**Chain rule.** The composition law of derivatives: each operation knows the
derivative of its output with respect to each operand and multiplies the
incoming gradient through. Implemented locally by every `Function` variant
in [`Operation::backward`](src/function/operation.rs).

**Gradient.** The vector of partial derivatives of one chosen scalar (the
*target*) with respect to every other value. A gradient is always "of a
target"; there is no target-free gradient of a network. In poorgrad:
[`Gradients`](src/gradients.rs), produced by one backward sweep and tied to
one evaluation and one target.

**Gradient accumulation.** When a value feeds several consumers, its
gradient is the *sum* of the contributions along every path (the
multivariate chain rule). This is why `Operation::backward` implementations
add into the gradient buffer instead of assigning.

**Seed (cotangent).** The gradient planted at the target before the backward
sweep; `one` for a plain gradient. Seeding several nodes with arbitrary
weights computes a vector-Jacobian product, the general form of reverse
mode. In poorgrad [`Evaluation::backward`](src/evaluation.rs) seeds
`one_like` at the target.

**Gradient descent.** Iteratively moving parameters against the gradient of
a loss: `w <- w - learning_rate * dLoss/dw`. One step is
[`Network::updated`](src/network.rs) with an update closure; see
[`examples/gradient_descent.rs`](examples/gradient_descent.rs).

## Graph model

**Computation graph.** The directed acyclic graph whose nodes are values and
whose edges link operations to their operands. In poorgrad the graph is
implicit in the tape: each recorded `Function` names its operands by
position, and allocation order is a topological order.

**Tape (Wengert list, "gradient tape").** The append-only record of every
operation in execution order — the recipe, not the result: it holds no
gradient values. Replayed forward it evaluates the program; replayed
backward with the chain rule it yields gradients for any target. In
poorgrad: [`Tape`](src/tape.rs), crate-internal, shared by a network and all
of its proxies, and the engine's single synchronization point.

**Node.** One recorded entry of the graph: the operation that produced a
value, its operand links, and its parameters. In poorgrad a node is a
[`Function<Data>`](src/function/function.rs) stored on the tape beside its
inferred `Shape`; neither changes once recorded.

**Shape.** The extent of a payload along every axis; a scalar is rank 0.
Shapes are inferred for every node when its expression is recorded — the
shape-level mirror of `forward`, an abstract interpretation of the tape —
so shape mismatches panic at the offending expression, before anything
runs. In the record-once model this recovers most of the benefit of
type-level shapes at no type-system cost. Shapes are lineage-invariant
and stored as a separate cold column beside the hot function column
(data-oriented layout: runs replay functions, never shapes). In poorgrad:
[`Shape`](src/shape.rs), reachable via `Value::shape` and
`Differentiable::shape`.

**Operation.** A differentiable primitive: how to compute a payload from
operand values (`forward`) and how to route the incoming gradient back to
the operands (`backward`). In poorgrad: the
[`Operation`](src/function/operation.rs) trait, implemented by each
`Function` variant (`Leaf`, `Parameter`, `Add`, `Sub`, `Mul`, `Div`,
`Neg`, `Tanh`, `Exp`, `Ln`, `MatMul`, `Transpose`, `Sum`, `Broadcast` under
[`src/function/`](src/function/)) and dispatched with a plain `match`.
Arithmetic variants need only `Differentiable`; the transcendental and
tensor-native ones raise the bound of running (not building) a graph to
`Elementary` and `Tensorial` respectively.

**Leaf.** A node with no operands: an input or constant supplied at
recording time. Gradients stop there and get read out; its `backward` is a
no-op. In poorgrad: `Function::Leaf`, allocated with
[`Network::leaf`](src/network.rs); payload literals in expressions
(`x * 2.0`) record leaves implicitly, one per appearance.

**Parameter.** A trainable leaf: identical to `Leaf` during runs, but
designated as updatable so a training step knows which leaves to replace.
In poorgrad: `Function::Parameter`, allocated with
[`Network::parameter`](src/network.rs) and replaced by `Network::updated`.

**Topological (allocation) order.** Any ordering in which every operand
precedes its consumers. Poorgrad's recording enforces it by construction —
a proxy must exist before it can be an operand — so `forward` is one
left-to-right scan and `backward` one right-to-left scan, with no explicit
sorting.

## Engine mechanics

**Network.** The single owner of the state of one computation graph: it owns
the tape, hands out proxies, and is the boundary of type homogeneity (one
`Data` type per network). Mutation happens only through state transitions
that produce new generations. In poorgrad: [`Network`](src/network.rs).

**Value (proxy).** A `Copy` handle pairing a borrow of the network's tape
with a node position. Proxies cannot outlive their network, are never
consumed by operators (`let x = v1 + v2;` records a node and keeps `v1`,
`v2` usable), and cross threads freely. In poorgrad:
[`Value`](src/value.rs).

**Symbol.** A detached, `Copy` name of a value: the identity that
persists across time, while `Value` is that identity's state in one
generation. Each generation acts as an environment;
[`Network::resolve`](src/network.rs) looks a symbol up in it and returns
that generation's proxy; a failed resolution panics as a programmer
error, while `try_resolve` probes and returns `None`. The symbol carries
its lineage, so resolving into an unrelated network panics rather than
misbinding; within a lineage, resolution is positional. In poorgrad:
[`Symbol`](src/symbol.rs), obtained with `Value::symbol`.

**Generation.** A network state produced by a state transition: a fork
(`Network::clone`) or a gradient step (`Network::updated`). Generations
share all unchanged nodes through the arena, keep positions stable
(symbols keep resolving), and leave older generations fully usable —
snapshot isolation.

**Run.** One forward or backward execution over a network. Runs never
mutate the network, so any number can execute concurrently; their results
are per-run buffers read back with the same proxies that built the graph:
[`Evaluation`](src/evaluation.rs) (a payload per node, generation-pinned,
carrying its own tape snapshot so `backward` differentiates it without
touching the network) and [`Gradients`](src/gradients.rs) (a gradient per node, for one target;
convertible into a `Field` for combination and optimizer state). Every
position-indexed buffer — evaluations, gradients, fields — answers the
same read-back accessor, `of(value)`.

**Field.** A value-aligned buffer: one payload per node, tied to a network
*lineage* rather than to a single generation, so it can be combined across
runs (averaging data-parallel gradients) and carried across generations
(momentum velocity, Adam moments). Supports elementwise algebra — `+`,
`scaled`, `zip`, `map` — with kinship (same lineage, same length) checked
on every combination; `Network::updated` takes any field as its update
direction. In physics terms, a `Gradients` is a discrete gradient field
over the graph. In poorgrad: [`Field`](src/field.rs).

**Lineage.** The family of networks descending from a common origin
through forks and updates. Positions are stable within a lineage, which is
what makes symbols resolve and fields combine across generations. Tracked
by a `Copy` identity minted from a process-global counter at network
creation and carried through every transition; kinship is equality. In
poorgrad: the crate-internal `Lineage` in [`src/tape.rs`](src/tape.rs),
embedded in every `Symbol` and `Field`.

**Payload (`Data`).** The numeric value a node carries: a scalar
(`f32`/`f64`) or an elementwise [`Tensor`](src/tensor.rs). Its contract is
the [`Differentiable`](src/differentiable.rs) trait — arithmetic
operators, `zero_like`/`one_like`, and `Send + Sync`;
[`Elementary`](src/elementary.rs) adds the transcendentals activations
need.

**Tensor.** A dense, fixed-shape payload: proof that the payload
contract holds beyond scalars, since a `Network<Tensor<f64>>` runs the
engine unchanged. Shape and elements live behind `Arc`s so cloning is
O(1); elementwise operations require identical shapes, and the
tensor-native tier adds `matmul`, `transposed`, `sum`, and the explicit
`broadcast_like`. In poorgrad: [`Tensor`](src/tensor.rs).

**Tensorial.** The payload tier of tensor-native operations — matrix
multiplication, transposition, reduction, and explicit broadcast — with
scalars implementing it degenerately (a scalar is a rank-0 tensor).
Summation and broadcasting are adjoint: each is the other's gradient
rule. Broadcasting is explicit by design: `broadcast_like` spreads a
single value across a named reference's shape, and no operation aligns
shapes implicitly. In poorgrad: the [`Tensorial`](src/tensorial.rs)
trait, recorded into graphs via `Value::matmul`, `transposed`, `sum`,
and `broadcast_like`.

**Arena.** Append-only storage in which every recorded node lives exactly
once, shared by all generations of a network; allocations never move or
drop while the arena lives, which is what makes forks cheap and references
stable. Provided by the [`cow_vec`](https://crates.io/crates/cow_vec) crate
inside `Tape`.

**Fork.** An O(1) copy of a network sharing the underlying arena but owning
an independent node list; later recordings on either side never affect the
other. In poorgrad: `Network::clone`, built on [`Tape::fork`](src/tape.rs).

## Neural building blocks

**Neuron.** The smallest learnable unit: a weighted sum of inputs plus a
bias, passed through an activation —
`activation(weights . inputs + bias)`. Its parameters are allocated on a
network at construction but held as symbols, so the neuron itself is
detached: it survives generations, and `express` records its expression
against whichever generation it is given. In poorgrad:
[`Neuron`](src/neuron.rs).

**Layer.** A row of neurons sharing the same inputs: a dense (fully
connected) layer computing one output per neuron. Layers chain by feeding
one layer's outputs to the next as inputs. Detached like its neurons —
parameters live on the network, symbols in the layer. In poorgrad:
[`Layer`](src/layer.rs).

**Activation.** The nonlinearity applied to a neuron's weighted sum, which
is what gives stacked neurons expressive power beyond affine maps. It is a
graph operation like any other, so it participates in differentiation
(`Function::Tanh`, recorded by `Value::tanh`; the derivative
`1 - tanh(x)^2` reuses the node's own output). In poorgrad: the
[`Activation`](src/neuron.rs) enum selecting `Identity` or `Tanh`.

## Further reading

- R. E. Wengert, "A simple automatic derivative evaluation program" (1964)
  — the original tape.
- A. Griewank and A. Walther, *Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation* (2008).
- A. G. Baydin et al., "Automatic Differentiation in Machine Learning: a
  Survey", JMLR (2018).
- A. Karpathy, [micrograd](https://github.com/karpathy/micrograd) — the
  educational engine poorgrad is loosely inspired by.
