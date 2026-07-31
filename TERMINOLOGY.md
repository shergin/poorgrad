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
[`Evaluation::backward`](src/engine/evaluation.rs). The sweep executes derivative
rules only for the target's ancestors; every other value's gradient is
exactly zero, so expressions the target does not depend on — including
singular ones — cannot disturb the result.

**Chain rule.** The composition law of derivatives: each operation knows the
derivative of its output with respect to each operand and multiplies the
incoming gradient through. Implemented locally by every `Function` variant
in [`Operation::backward`](src/engine/function/operation.rs).

**Gradient.** The vector of partial derivatives of one chosen scalar (the
*target*) with respect to every other value. A gradient is always "of a
target"; there is no target-free gradient of a network. In poorgrad:
[`Gradients`](src/engine/field.rs), produced by one backward sweep and tied to
one evaluation and one target; it is a named role of `Field`, not a separate
type.

**Gradient accumulation.** When a value feeds several consumers, its
gradient is the *sum* of the contributions along every path (the
multivariate chain rule). In poorgrad the rule is stated once, in the
engine: `Operation::backward` returns one cotangent per operand, and
[`Evaluation::backward`](src/engine/evaluation.rs) adds each into the
gradient buffer — no operation can assign where it should accumulate,
because no operation ever touches the buffer.

**Seed (cotangent).** The gradient planted at the target before the backward
sweep; `one` for a plain gradient. Seeding several nodes with arbitrary
weights computes a vector-Jacobian product, the general form of reverse
mode. In poorgrad [`Evaluation::backward`](src/engine/evaluation.rs) seeds
`one_like` at the target, which must be rank 0: a non-scalar value is
reduced explicitly with `sum` before differentiation, never summed
implicitly.

**Gradient descent.** Iteratively moving parameters against the gradient of
a loss: `w <- w - learning_rate * dLoss/dw`. One step is
[`Network::update`](src/engine/network.rs) with an update closure; see
[`examples/gradient_descent.rs`](examples/gradient_descent.rs).

## Graph model

**Computation graph.** The directed acyclic graph whose nodes are values and
whose edges link operations to their operands. In poorgrad the graph is
implicit in the tape: each recorded node lists its operand links, in the
operation's positional order, in the tape's operands column, and
allocation order is a topological order.

**Tape (Wengert list, "gradient tape").** The append-only record of every
operation in execution order — the recipe, not the result: it holds no
gradient values. Replayed forward it evaluates the program; replayed
backward with the chain rule it yields gradients for any target. In
poorgrad: [`Tape`](src/engine/tape/tape.rs), crate-internal, shared by a
network and all of its proxies, and the engine's single synchronization
point. Beside its immutable columns the tape carries the generation's
parameter store: the one piece of state that changes across generations.

**Node.** One recorded entry of the graph: the operation that produced a
value, its operand links, and its parameters. In poorgrad a node is a
[`Function<Data>`](src/engine/function/function.rs) (the operation and its
parameters) stored on the tape beside its operand links (the
[`Operands`](src/engine/tape/operands.rs) column) and its inferred `Shape`;
none of them change once recorded.

**Shape.** The extent of a payload along every axis; a scalar is rank 0.
Shapes are inferred for every node when its expression is recorded — the
shape-level mirror of `forward`, an abstract interpretation of the tape —
so shape mismatches panic at the offending expression, before anything
runs. In the record-once model this recovers most of the benefit of
type-level shapes at no type-system cost. Shapes are lineage-invariant —
`update` validates every replacement payload against the recorded
shape — and stored as a separate cold column beside the hot function and
operands columns (data-oriented layout: runs replay functions and operand
links, never shapes). In poorgrad:
[`Shape`](src/payload/shape.rs), reachable via `Value::shape` and
`Differentiable::shape`.

**Operation.** A differentiable primitive: how to compute a payload from
operand payloads (`forward`) and the cotangent to hand back to each
operand (`backward`). Operation APIs use plain verbs; when a name denotes
the result, it uses a result noun (`sum`, `maximum`, `step`). Suffix
families (`_along`, `_like`) preserve that form, and operation names do
not use participles. The rules are pure and positional: a variant owns
only its parameters (an axis, a target shape) and declares its arity,
operands arrive as a slice — payload references for the value rules,
shapes for shape inference — gathered by the engine from the tape's
operands column, and `backward` returns one cotangent per operand
(`None` for an operand that is data, like a gather's selection) for the
engine to accumulate. No rule ever sees the tape, a `ValueId`, or a run
buffer, so every rule is plain math, testable without a network. In
poorgrad: the [`Operation`](src/engine/function/operation.rs) trait,
implemented by each computed `Function` variant (`Add`, `Sub`, `Mul`,
`Div`, `Neg`, `Tanh`, `Exp`, `Ln`, `Sqrt`, `Powf`, `Maximum`, `Relu`,
`MatMul`, `Transpose`, `Sum`, `SumAlong`, `Broadcast`, `BroadcastAlong`,
`Reshape`, `Permute`, `Narrow`, `Gather`, `LogSoftmax` under
[`src/engine/function/`](src/engine/function/)) and dispatched with a
plain `match`.
`Leaf`, `Parameter`, and `Input` are supplied rather than computed, so
the enum's dispatch handles them directly instead of through the trait.
Arithmetic variants need only `Differentiable`; the transcendental and
tensor-native ones raise the bound of running (not building) a graph to
`Elementary` and `Tensorial` respectively.

**Leaf.** A node with no operands: a constant supplied at recording
time. Gradients stop there and get read out; its `backward` is a no-op.
Parameters and inputs are the other leaf kinds: trainable and fed
per-run respectively. In poorgrad: `Function::Leaf`, allocated with
[`Network::leaf`](src/engine/network.rs); payload literals in expressions
(`x * 2.0`) record leaves implicitly, one per appearance.

**Parameter.** A trainable leaf: identical to `Leaf` during runs, but
designated as updatable so a training step knows which leaves to replace.
In poorgrad: `Function::Parameter`, allocated with
[`Network::parameter`](src/engine/network.rs) and replaced by
`Network::update`. The node holds only its slot; the payload lives in the
generation's parameter store.

**Input.** A declared per-run leaf: `Network::input` records it with a
default payload, and `forward_with` binds a fed payload to it for one
run, validated against the recorded shape at the feed site. Unfed
inputs fall back to their defaults, so plain `forward` stays total.
Feeds are run state, not graph state — feeding never touches the tape,
which is what lets concurrent runs forward one shared network on
different batches. In poorgrad: `Function::Input`, fed via
[`Network::forward_with`](src/engine/network.rs).

**Topological (allocation) order.** Any ordering in which every operand
precedes its consumers. Poorgrad's recording enforces it by construction —
a proxy must exist before it can be an operand — so `forward` is one
left-to-right scan and `backward` one right-to-left scan, with no explicit
sorting.

## Engine mechanics

**Network.** The single owner of the state of one computation graph: it owns
the tape, hands out proxies, and is the boundary of type homogeneity (one
`Data` type per network). Mutation happens only through state transitions
that produce new generations. In poorgrad: [`Network`](src/engine/network.rs).

**Value (proxy).** A `Copy` handle pairing a borrow of the network's tape
with a node position. Proxies cannot outlive their network, are never
consumed by operators (`let x = v1 + v2;` records a node and keeps `v1`,
`v2` usable), and cross threads freely. In poorgrad:
[`Value`](src/engine/value.rs).

**Composite (operation).** A method that expands to several primitive
nodes: a formula over opcodes whose gradient the chain rule pays with no
dedicated backward rule. The operation surface has three tiers, marked by
files rather than by types: [`value.rs`](src/engine/value.rs) holds the
opcode mnemonics, each recording exactly one computed node (payload
literals additionally record a leaf — data injection, not computation);
[`composite.rs`](src/engine/composite.rs) holds the composites (`abs` as
`maximum(-self)`, `softmax` as `exp(log_softmax)` — stable by inheritance,
since log-probabilities cannot make `exp` overflow — and `logsumexp`,
recovered from the fused normalizer with the softmax as its composed
gradient); and named formulas whose operands play distinct roles (a
loss's logits and targets have no natural `self`) are free functions in
domain modules. Composites compile against the public operation surface
alone — they need no privileged engine access — and once recorded they
are indistinguishable from hand-written primitives, keeping the tape a
uniform IR. A formula moves down a tier and earns a `Function` variant
only when floating point breaks the composed form, as it did for
`log_softmax`.

**Symbol.** A detached, `Copy` name of a value: the identity that
persists across time, while `Value` is that identity's state in one
generation. Each generation acts as an environment;
[`Network::resolve`](src/engine/network.rs) looks a symbol up in it and
returns that generation's proxy; a failed resolution panics as a programmer
error, while `try_resolve` probes and returns `None`. The symbol carries its
lineage and its branch, so resolving into an unrelated network — or into a
fork that diverged before the symbol was minted — panics rather than
misbinding; within a branch, resolution is positional. In poorgrad:
[`Symbol`](src/engine/symbol.rs), obtained with `Value::symbol`.

**Generation.** A network state produced by a state transition: a fork
(`Network::clone`) or a gradient step (`Network::update`). Generations
share the recorded structure through the arena and differ only in their
parameter store; positions stay stable (symbols keep resolving), and
older generations remain fully usable — snapshot isolation.

**Parameter store.** The per-generation home of parameter payloads,
slot-indexed and separate from the immutable tape columns: structure is
recorded once, state turns over per generation. Forks share the store in
O(1); `update` rebuilds it in O(parameters), so replaced payloads are
reclaimed when their generation drops instead of accumulating in the
arena. In poorgrad: the crate-internal
[`ParameterStore`](src/engine/tape/parameter_store.rs).

**Run.** One forward or backward execution over a network. Runs never
mutate the network, so any number can execute concurrently; their results
are per-run buffers read back with the same proxies that built the graph:
[`Evaluation`](src/engine/evaluation.rs) (a payload per node,
generation-pinned, carrying its own tape snapshot so `backward`
differentiates it without touching the network) and
[`Gradients`](src/engine/field.rs) (a gradient per node, for one target;
a `Field`, so it combines and carries optimizer state directly). Every
position-indexed buffer — evaluations, gradients, fields — answers the same
read-back accessor, `of(value)`.

**Field.** A value-aligned buffer: one payload per node, tied to a network
*lineage* rather than to a single generation, so it can be combined across
runs (averaging data-parallel gradients) and carried across generations
(momentum velocity, Adam moments). Supports elementwise algebra — `+`,
`scale`, `zip`, `map` — with kinship (same lineage, same length,
agreeing branch chains) checked on every combination; `Network::update`
takes any field as its update direction. In physics terms, a `Gradients` is
a discrete gradient field over the graph, which is why `Gradients` is an
alias for `Field` rather than a wrapper around it: the buffer's invariant is
alignment to a graph, not differentiation, and Adam's second moment or an
evaluation's forward payloads are fields that are not gradients at all. In
poorgrad: [`Field`](src/engine/field.rs).

**Lineage.** The family of networks descending from a common origin
through forks and updates. Within a lineage, positions are attributed to
branches, and they are stable within a branch — which is what makes
symbols resolve and fields combine across generations. Tracked by a
`Copy` identity minted from a process-global counter at network creation
and carried through every transition; kinship is equality. In poorgrad: the
crate-internal [`Lineage`](src/engine/tape/identity.rs), embedded in every
`Symbol` and `Field`.

**Branch.** A contiguous run of recordings within a lineage. A fork or an
update hands both sides a shared one-shot claim on the current branch:
the first side to record continues it, and every other sibling starts a
fresh branch at its own length, so divergent forks stop sharing identity
exactly where their recordings part ways. Symbols carry the branch that
owned their position when they were minted, and resolution checks branch
membership before the positional lookup, so a divergent sibling's symbol
panics instead of misbinding. Linear histories never mint branches:
chains stay as short as the program's real divergence. In poorgrad: the
crate-internal [`Branch`](src/engine/tape/identity.rs) and its segment chain.

**Payload (`Data`).** The numeric value a node carries: a scalar
(`f32`/`f64`) or an elementwise [`Tensor`](src/payload/tensor.rs). Its
contract is the [`Differentiable`](src/payload/differentiable.rs) trait —
arithmetic operators, `zero_like`/`one_like`, and `Send + Sync`;
[`Elementary`](src/payload/elementary.rs) adds the transcendentals, the
correctly rounded `sqrt` (which `powf(0.5)` is not), and the order pair
`maximum`/`step` that activations and stable normalization need — order
enters the contract as payload-returning operations, never as
`PartialOrd`, whose `bool` answer cannot express an elementwise
comparison. `step` is the Heaviside 0/1 indicator of `self >= threshold`
that carries the `maximum` family's derivative; ties answer one, so
`maximum` hands a tied gradient to its left operand and the relu
subgradient at zero is one.

**Tensor.** A fixed-shape payload backed by a shared element buffer read
through a strided layout: proof that the payload contract holds beyond
scalars, since a `Network<Tensor<f64>>` runs the engine unchanged.
Cloning shares the buffer and copies only metadata, so it is O(1).
Elementwise operations require identical shapes; the tensor-native tier
adds `matmul`, `transpose`, the reductions `sum` and `sum_along`, and
the explicit broadcasts `broadcast_like` and `broadcast_along`. Because
tensors are immutable and buffer-shared, `transpose` and the broadcasts
are O(1) views (or constants) rather than copies: no operation ever
writes through an alias. Elements are read in logical row-major order
through `iter`, as a contiguous slice through `as_slice` when the
representation allows, or copied out with `to_vec`. In poorgrad:
[`Tensor`](src/payload/tensor.rs).

**Storage.** The buffer representation behind a `Tensor`, and the
extension seam for how elements are held: today an `Arc`-shared row-major
`Dense` buffer addressed by a `Layout`, and a non-allocating `Constant`
that fills its shape with a single value. Each variant carries exactly
its own metadata — the strides live inside `Dense`, not at the tensor
level — so a future representation (a sparse or a SIMD-aligned buffer) is
a new variant that a shared logical element access reaches without
disturbing the operations. `Constant` is the first non-`Dense` variant:
it makes `filled`, `zero_like`, `one_like`, and whole-shape broadcasts
O(1) and closed under algebra, which most visibly keeps `backward`'s
per-node gradient seed from allocating a zeroed buffer for every node.
`Selection` is the second: a one-hot `[count, vocab]` matrix stored as its
row indices, which keeps an embedding lookup's token indices as `usize`
inside a homogeneous payload and lets a `Gather` read them directly.
In poorgrad: the crate-internal [`Storage`](src/payload/storage.rs).

**Layout.** How a dense buffer's logical indices map onto its flat
storage: the shape, the per-axis strides, and the offset of the first
element. The element at multi-index `(i0, ..., in)` lives at
`offset + sum(i_k * strides_k)`. A contiguous row-major layout has
`strides_k = product(shape[k + 1 ..])` and offset zero; view operations
produce a new layout over the same buffer without moving any element. A
stride of `0` marks a broadcast axis, whose steps do not advance within
the buffer, which is how `broadcast_along` repeats without copying. In
poorgrad: the crate-internal [`Layout`](src/payload/layout.rs).

**Contiguity.** Whether a dense layout addresses a row-major slice of its
buffer starting at its offset (extent-1 axes impose no constraint;
stride-0 broadcast axes are never contiguous). A contiguous tensor
exposes its elements as a borrowed slice and takes a flat iteration fast
path, while a strided view walks its layout with an odometer. Contiguity
is a property of the strides, computed on demand, not a stored flag.

**Tensorial.** The payload tier of tensor-native operations — matrix
multiplication, transposition, reductions, and explicit broadcasts —
with scalars implementing it degenerately (a scalar is a rank-0 tensor;
the degenerate impls satisfy the bound of running a graph, while
recording tensor-native expressions demands proper ranks). `matmul` stops
at rank 2, as does `transpose`, with `permute` its rank-general
generalization; the axis-wise pair is rank-general; `reshape` reinterprets
the elements in logical order; there is no batched matmul yet. Summation
and broadcasting are adjoint in two matched pairs: `sum` with
`broadcast_like` (the whole shape) and `sum_along` with `broadcast_along`
(one named axis), each the other's gradient rule. The view operations route their gradient the
same adjoint way: `reshape` and `permute` invert their view, and
`narrow` selects a window whose gradient `pad`s back into the excluded
positions as zeros (`narrow` with `pad` as the third adjoint pair),
and `gather` selects table rows by a one-hot `Selection` whose gradient
`scatter`s back, accumulating rows selected more than once (`gather` with
`scatter` as the fourth pair, and the embedding lookup). The selection is
data, so `gather`'s backward has no gradient term for it at all: the
non-differentiability of the indices is a structural property of the
operation, not a runtime flag. `max_along` is `sum_along`'s
order-theoretic sibling — the same axis reduction, folding with the
elementwise `maximum` — and serves stable normalization rather than
recording: `log_softmax`, the one fused operation, shifts by the axis
maximum before exponentiating (which no composition of recorded
operations could do) and routes its gradient as
`g - exp(output) * sum_along(g)`, recovering the probabilities from the
node's own output. Broadcasting is explicit by design: a
single value spread across a named reference's shape, or a payload
repeated along one named axis of a reference — the axis is always
written, and no operation aligns shapes implicitly. In poorgrad: the
[`Tensorial`](src/payload/tensorial.rs) trait, recorded into graphs via
`Value::matmul`, `transpose`, `sum`, `sum_along`, `broadcast_like`,
`broadcast_along`, `reshape`, `permute`, `narrow`, `gather`,
`log_softmax`, and the `reshape`-based `squeeze` and `unsqueeze`.

**Arena.** Append-only storage in which every recorded node lives exactly
once, shared by all generations of a network; allocations never move or
drop while the arena lives, which is what makes forks cheap and references
stable. Provided by the [`cow_vec`](https://crates.io/crates/cow_vec) crate
inside `Tape`. Training never touches it: parameter payloads live in the
parameter store, so the arena holds structure only.

**Fork.** An O(1) copy of a network sharing the underlying arena but owning
an independent node list; later recordings on either side never affect the
other. The two sides contend for the current branch on their first
recording, and the loser diverges onto a fresh one, so their later
symbols never misbind (see Branch). In poorgrad: `Network::clone`, built
on [`Tape::fork`](src/engine/tape/tape.rs).

## Neural building blocks

**Neuron.** The smallest learnable unit: a weighted sum of inputs plus a
bias, passed through an activation —
`activation(weights . inputs + bias)`. Its parameters are allocated on a
network at construction but held as symbols, so the neuron itself is
detached: it survives generations, and `express` records its expression
against whichever generation it is given. It is the scalar-granularity
teaching block; `Layer` records at tensor granularity and does not
build on it. In poorgrad: [`Neuron`](src/neural/neuron.rs).

**Layer.** A dense (fully connected) layer at tensor granularity:
`activation(x . w + b)` over a `[batch, inputs]` value, with one
`[inputs, outputs]` weight parameter and one `[outputs]` bias met
through the explicit axis broadcast — a handful of tensor nodes instead
of one node per scalar weight. Layers chain by feeding one layer's
output batch to the next. Detached like `Neuron`: parameters live on
the network, symbols in the layer. In poorgrad:
[`Layer`](src/neural/layer.rs).

**Mlp.** A multilayer perceptron: dense layers chained by a topology of
value widths (`[3, 4, 4, 1]`), hidden layers squashing with `Tanh` and
an affine output layer. A facade over `Layer`, detached the same way,
with initialization owned by the caller through a shape-to-payload
initializer. In poorgrad: [`Mlp`](src/neural/mlp.rs).

**Activation.** The nonlinearity applied to a neuron's weighted sum, which
is what gives stacked neurons expressive power beyond affine maps. It is a
graph operation like any other, so it participates in differentiation
(`Function::Tanh`, recorded by `Value::tanh`, whose derivative
`1 - tanh(x)^2` reuses the node's own output; `Function::Relu`, recorded
by `Value::relu`, whose gradient is masked by the 0/1 `step` indicator —
a dedicated unary variant because recording cannot construct a zero
payload for a generic `Data`, while the rule reaches one at run time
through `zero_like`). In poorgrad: the
[`Activation`](src/neural/activation.rs) enum selecting `Identity`,
`Tanh`, or `Relu`.

**Loss.** A scalar training objective written as a composed formula over
recorded operations, not as a primitive: its gradient falls out of the
chain rule with no dedicated backward rule. A formula earns a fused
`Function` variant only where composition cannot express it — the
cross-entropy loss
`-(targets * log_softmax(logits)).sum() / targets.sum()` keeps only
`log_softmax` fused (for the stabilizing max shift) and stays composition
everywhere else. The normalizer is the targets' total mass — the batch
size for one-hot targets, so the reduction is the standard mean, while
soft or weighted targets normalize by their own weight. The same one-hot
`Selection` that feeds an embedding gather serves as the targets, fed per
run. Losses are the third tier of the operation surface (see Composite):
free functions rather than `Value` methods, because their operands play
distinct roles and a method would arbitrarily privilege one of them. In
poorgrad: [`cross_entropy`](src/neural/loss.rs) in the loss module.

**Initializer.** The shape-to-payload closure a caller hands to a
building block at construction: initialization is caller-owned, and
`Layer` and `Mlp` record whatever they are given. The `init` module
manufactures deterministic initializers — `uniform` and `normal` fill
any shape, while the fan-aware `xavier` and `kaiming` read the fan-in
off the requested rank-2 shape and zero rank-1 shapes, a bias
identifying itself structurally by its rank. Every factory takes an
explicit seed and each closure owns its splitmix64 generator state: no
global generator, no clock, bit-identical runs forever — which is why
the crate carries its own few-line generator instead of a `rand`
dependency, whose standard generator is unstable across versions. In
poorgrad: [`init`](src/neural/init.rs), the crate's one public module,
qualified because `uniform` and `normal` are meaningless names without
it.

## Further reading

- R. E. Wengert, "A simple automatic derivative evaluation program" (1964)
  — the original tape.
- A. Griewank and A. Walther, *Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation* (2008).
- A. G. Baydin et al., "Automatic Differentiation in Machine Learning: a
  Survey", JMLR (2018).
- A. Karpathy, [micrograd](https://github.com/karpathy/micrograd) — the
  educational engine poorgrad is loosely inspired by.
