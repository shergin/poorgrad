# Terminology

The vocabulary used across topos's code and docs. Each entry gives the
meaning of the term in the automatic-differentiation literature and how it
maps onto this crate's types. This file is part of the codebase contract:
when a concept is added, renamed, or changes meaning, update it in the same
change.

## Mathematics

**Automatic differentiation (autodiff, AD).** Computing exact derivatives of
a program by decomposing it into primitive operations with known local
derivatives and composing them via the chain rule. Distinct from numeric
differentiation (finite differences; approximate) and symbolic
differentiation (expression rewriting; can blow up). Topos implements
reverse-mode AD over scalar programs.

**Reverse-mode AD (backpropagation).** The AD flavor that computes the
derivative of *one* output with respect to *all* inputs in a single backward
sweep costing about one forward evaluation. Its mirror image, forward mode,
computes one input against all outputs. Reverse mode wins for machine
learning (one loss, many parameters). In topos:
[`Run::backward`](src/engine/run.rs). The sweep executes derivative
rules only for the target's ancestors; every other value's gradient is
exactly zero, so expressions the target does not depend on — including
singular ones — cannot disturb the result.

**Chain rule.** The composition law of derivatives: each operation knows the
derivative of its output with respect to each operand and multiplies the
incoming gradient through. Implemented locally by every `Function` variant
in [`Operation::backward`](src/function/operation.rs).

**Gradient.** The vector of partial derivatives of one chosen scalar (the
*target*) with respect to every other value. A gradient is always "of a
target"; there is no target-free gradient of a network. In topos:
[`Gradients`](src/graph/field.rs), produced by one backward sweep and tied to
one run and one target; it is a named role of `Field`, not a separate
type.

**Gradient accumulation.** When a value feeds several consumers, its
gradient is the *sum* of the contributions along every path (the
multivariate chain rule). In topos the rule is stated once, in the
engine: `Operation::backward` returns one cotangent per operand, and
[`Run::backward`](src/engine/run.rs) adds each into the
gradient buffer — no operation can assign where it should accumulate,
because no operation ever touches the buffer.

**Seed (cotangent).** The gradient planted at the target before the backward
sweep; `one` for a plain gradient. Seeding several nodes with arbitrary
weights computes a vector-Jacobian product, the general form of reverse
mode. In topos [`Run::backward`](src/engine/run.rs) seeds
`one_like` at the target, which must be rank 0: a non-scalar value is
reduced explicitly with `sum` before differentiation, never summed
implicitly.

**Gradient descent.** Iteratively moving parameters against the gradient of
a loss: `w <- w - learning_rate * dLoss/dw`. One step is
[`Parameters::step`](src/graph/parameters.rs) with an update
closure — a pure data transformation of the caller-owned state; see
[`examples/gradient_descent.rs`](examples/gradient_descent.rs).

## Graph model

**Computation graph.** The directed acyclic graph whose nodes are values and
whose edges link operations to their operands. In topos the graph is
implicit in the tape: each recorded node lists its operand links, in the
operation's positional order, in the tape's operands column, and
allocation order is a topological order.

**Tape (Wengert list, "gradient tape").** The append-only record of every
operation in execution order — the recipe, not the result: it holds no
gradient values. Replayed forward it evaluates the program; replayed
backward with the chain rule it yields gradients for any target. In
topos: the public [`Tape`](src/graph/tape.rs), the construction
phase of a network — expressions record onto it through [`Value`]
proxies, shape-checked at the recording expression, behind the engine's
single synchronization point (a mutex quarantined to this phase).
`Tape::into_network` consumes the tape and seals the recording into an
immutable [`Network`]; `Network::into_tape` consumes the network to
reopen it. Both conversions are consuming, so one recording's history is
linear by ownership — two divergent futures of a prefix cannot be
constructed, which is what lets every detached carrier identify nodes by
origin and position alone.

**Node.** One recorded entry of the graph: the operation that produced a
value, its operand links, and its parameters. In topos a node is a
[`Function<Data>`](src/function/function.rs) (the operation and its
parameters) stored on the tape beside its operand links (the
[`Operands`](src/graph/operands.rs) column) and its inferred `Shape`;
none of them change once recorded.

**Shape.** The extent of a payload along every axis; a scalar is rank 0.
Shapes are inferred for every node when its expression is recorded — the
shape-level mirror of `forward`, an abstract interpretation of the tape —
so shape mismatches panic at the offending expression, before anything
runs. In the record-once model this recovers most of the benefit of
type-level shapes at no type-system cost. Shapes never change once
recorded — `Parameters::step` and the checkpoint installs validate every
replacement payload against the current shape — and are stored as a
separate cold column beside the hot function and operands columns inside
[`Structure`](src/graph/structure.rs) (data-oriented layout:
runs replay functions and operand links, never shapes). The three
columns always move together: seals, freezes, and reopens share one
`Structure`. Construction boundaries (`Tensor::new`,
`Tensor::filled`, `Value::reshape`) accept `impl Into<Shape>` — axis
literals, vectors, slices, and shapes or their references all convert —
so the nominal type is never decomposed at the rim; `Shape::new` remains
the iterator-based base constructor. In topos:
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
topos: the [`Operation`](src/function/operation.rs) trait,
implemented by each computed `Function` variant (`Add`, `Sub`, `Mul`,
`Div`, `Neg`, `Tanh`, `Exp`, `Ln`, `Sqrt`, `Powf`, `Maximum`, `Relu`,
`MatMul`, `Transpose`, `Sum`, `SumAlong`, `Broadcast`, `BroadcastAlong`,
`Reshape`, `Permute`, `Narrow`, `Gather`, `LogSoftmax` under
[`src/function/`](src/function/)) and dispatched with a
plain `match`.
`Leaf`, `Parameter`, and `Input` are supplied rather than computed, so
the enum's dispatch handles them directly instead of through the trait.
Arithmetic variants need only `Differentiable`; the transcendental and
tensor-native ones raise the bound of running (not building) a graph to
`Elementary` and `Tensorial` respectively.

**Leaf.** A node with no operands: a constant supplied at recording
time. Gradients stop there and get read out; its `backward` is a no-op.
Parameters and inputs are the other leaf kinds: trainable and fed
per-run respectively. In topos: `Function::Leaf`, allocated with
[`Tape::leaf`](src/graph/tape.rs); payload literals in expressions
(`x * 2.0`) record leaves implicitly, one per appearance.

**Parameter.** A trainable leaf: identical to `Leaf` during runs, but
designated as updatable so a training step knows which leaves to replace.
In topos: `Function::Parameter`, allocated with
[`Tape::parameter`](src/graph/tape.rs) from its record-site
initial. The node holds only its slot; live payloads are the caller's
[`Parameters`](src/graph/parameters.rs) state, stepped by
`Parameters::step` — training never touches the recorded node.

**Input.** A declared per-run leaf: `Tape::input` records it with a
default payload — part of the spec, so a network with its defaults is
runnable standalone — and a `forward` feed binds a fed payload to it for
one run, validated against the recorded shape at the feed site. Unfed
inputs fall back to their defaults. Feeds are run state, not graph
state — feeding never touches the spec, which is what lets concurrent
runs forward one shared network on different batches. In topos:
`Function::Input`, fed via the feed pairs of
[`Network::forward`](src/graph/network.rs) and `Plan::forward`.

**Topological (allocation) order.** Any ordering in which every operand
precedes its consumers. Topos's recording enforces it by construction —
a proxy must exist before it can be an operand — so `forward` is one
left-to-right scan and `backward` one right-to-left scan, with no explicit
sorting.

## Engine mechanics

**Network.** The sealed phase of a recording: the immutable spec of one
computation graph — structure, shapes, parameter initials, and input
defaults, with no live state and no lock. Born only from
`Tape::into_network`, shared for concurrent runs by `&Network` or
`Arc<Network>`, reopened for further recording by the consuming
`Network::into_tape`, and deliberately not `Clone`: a second sealed copy
could be reopened into a divergent future, which the ownership rule
exists to make unrepresentable. Runs and plans read parameter payloads
from a caller-supplied [`Parameters`] per call. It is the boundary of
type homogeneity (one `Data` type per network). In topos:
[`Network`](src/graph/network.rs).

**Parameters.** The live parameter payloads of one network, as a
caller-owned value: the state half of the spec/state split. Born from
the record-site initials (`Network::parameters`) or a checkpoint,
passed by reference into every run and plan, stepped as pure data
(`step`/`step_each`), read by symbol (`of`), carried across an
`into_tape` round trip (`carried` — existing slots keep payloads, new
slots take their initials), and installed into by name
(`with_payloads`, the checkpoint route). `Clone` is honest and
O(parameters), which is the whole cost of a what-if: one spec, any
number of states. Optimizer state is `Field` algebra held beside a
`Parameters` value; nothing hides in the graph. In topos:
[`Parameters`](src/graph/parameters.rs).

**Value (proxy).** The operand of recording: a `Copy` handle pairing a
borrow of the tape with a node position, alive only inside the
construction phase — the same meaning `llvm::Value` has as the
payload-free node handle composed through `IRBuilder`. Proxies are
never consumed by operators (`let x = v1 + v2;` records a node and
keeps `v1`, `v2` usable) and cross threads freely, but they cannot
cross `Tape::into_network`: the seal consumes the tape, so a proxy
outliving the phase is a borrow error at the exact line, and
`Value::symbol` is the documented bridge out. Payloads live in
[`Parameters`] and [`Run`]s, read by [`Symbol`]. In topos:
[`Value`](src/graph/value.rs).

**Composite (operation).** A method that expands to several primitive
nodes: a formula over opcodes whose gradient the chain rule pays with no
dedicated backward rule. The operation surface has three tiers, marked by
files rather than by types: [`value.rs`](src/graph/value.rs) holds the
opcode mnemonics, each recording exactly one computed node (payload
literals additionally record a leaf — data injection, not computation);
[`composite.rs`](src/graph/composite.rs) holds the composites (`abs` as
`maximum(-self)`, `softmax` as `exp(log_softmax)` — stable by inheritance,
since log-probabilities cannot make `exp` overflow — and `mean_along`,
`sum_along` divided by the reduced axis's
extent minted as a `counted` literal); and named formulas whose operands play distinct roles (a
loss's logits and targets have no natural `self`) are free functions in
domain modules. Composites compile against the public operation surface
alone — they need no privileged engine access — and once recorded they
are indistinguishable from hand-written primitives, keeping the tape a
uniform IR. A formula moves down a tier and earns a `Function` variant
only when floating point breaks the composed form, as it did for
`log_softmax` — and later for `logsumexp`, whose composition over
`log_softmax` (the normalizer read back from one lane) returned `inf`
once finite logits differed by more than the representable range.

**Differentiate (gradient recording).** Reverse-mode differentiation
as a tape-to-tape transform: `Tape::differentiate(loss, wrt)`
appends the gradient computation as ordinary computed nodes and
returns one symbol per `wrt` entry, so gradients are first-class
values — compilable, emittable, readable, and differentiable again
for higher-order derivatives. The transform runs the very same
derivative rules the engine's `backward` runs, over a recording
`Trace` payload instead of buffers (the rules are generic over the
payload traits, so interpretation and transformation are two
payloads of one rule — derivative knowledge cannot fork). The
recorded scan mirrors the engine's seed, ancestor masking, and
accumulation order, so a compiled plan over `[loss, gradients...]`
reproduces `Run::backward` bitwise; per-variant closure tests
hold that contract. Differentiation appends nodes, so it is a
construction-phase operation and lives on the tape. In topos:
`Tape::differentiate`, the `Trace` payload, and the closure suite in
`graph/tests/differentiate_tests.rs`.

**Symbol.** A detached, `Copy` name of a value: an origin plus a node
position, and the sole currency of every phase after recording. Access
is phase-scoped, names are forever: a `Value` dies at the seal, while a
symbol is `'static`, crosses threads, cells, and checkpoints, and stays
valid across every `into_tape` round trip, because linear extension
never moves a recorded node. Reads go through `Parameters::of`,
`Run::of`, and `Field::of`; run-time naming (feeds, `backward` targets,
compile roots and observes) speaks symbols; `Tape::resolve` turns one
back into a proxy when a network reopens. Resolving a foreign symbol
panics rather than misbinding:
origin equality plus coverage are the two integer compares that remain
of graph identity. In topos:
[`Symbol`](src/graph/symbol.rs), obtained with `Value::symbol`.

**Seal / reopen.** The two phase transitions of one recording:
`Tape::into_network` seals the construction phase into the immutable
spec, and `Network::into_tape` reopens it for further recording. Both
consume their operand and move the columns and stores without copying,
so the history stays linear by ownership and neither is callable
through `&` or `Arc` — sharing and extending exclude each other by the
ownership rules alone. State survives the round trip through
`Parameters::carried`. There are no generations: training steps the
caller's `Parameters` and mints no new network, so the questions the
old generation machinery answered (which generation does a symbol,
field, or plan bind to; who may record next) can no longer be asked.

**Slot store.** A dense table of payloads keyed by [`SlotId`](src/graph/slot.rs),
each row also holding the tape [`ValueId`](src/graph/value.rs) of the
node that names that slot. Parameter initials and input defaults share
this layout ([`SlotStore`](src/graph/slot_store.rs)): structure
is recorded once; the caller's `Parameters` carries the live parameter
table (`step` rebuilds payloads in O(parameters)), and the spec's input
table holds defaults (runs may overlay feeds without touching the
graph). Slots are installed in recording order and never move, which is
what keeps symbols naming their parameters across steps and reopens.

**Run.** One forward or backward execution over a network, and the type
that reifies the forward one: [`Run`](src/engine/run.rs) is a payload
per node, owning its structure freeze so `backward` needs no network
borrow; kinship is the origin-and-coverage check every detached carrier
makes. Each run carries its *posture*, the
producer-specific state as one explicit sum: complete or target-sliced
from the interpreter, observed from a forward-only plan (only the
keep-set answers reads, `backward` refused), training from an
engine-backward plan (everything `backward` reads retained) — so an
impossible combination cannot be represented. Runs never mutate the network, so any number can
execute concurrently; a backward sweep reads a `Run` and returns
[`Gradients`](src/graph/field.rs) (a gradient per node, for one target;
a `Field`, so it combines and carries optimizer state directly). Both
are read back with the same proxies that built the graph: every
position-indexed buffer — runs, gradients, fields — answers the same
read-back accessor, `of(value)`.

**Target-sliced run.** A forward run restricted to the ancestor
closure of declared targets:
[`Network::forward_for`](src/graph/network.rs) marks
reachability over the operand links in one descending sweep and skips
every node outside the closure, leaving an O(1), shape-correct zero
placeholder (`counted(shape, 0)`) in each skipped slot. The
placeholders keep `Parameters::step` sound — a parameter outside the
closure receives its true gradient, exactly zero — while reads stay
loud: `Run::of` and
`Run::backward` panic on a skipped value rather than answer
with a placeholder. Observability is declared, never inferred — the
same contract the plan-lowering path generalizes into the keep-set.
With several expressions recorded on one tape (the training and
evaluation twins of the examples), slicing to one expression's
targets skips the other entirely.

**Plan (lowering).** A compiled execution schedule derived from the
tape: the ancestor closure of declared roots (dead-node
elimination), the readable set (roots plus observes), per-node free
lists (buffer liveness), and captured shapes. Produced by
[`Network::compile`](src/engine/plan.rs) from one explicit
[`Request`](src/engine/request.rs): roots (what a run must
compute; recorded gradient symbols enter as ordinary roots),
observes (extra readable interiors), and the optional `backward`
posture, which holds every closure value the engine's reverse scan
reads — a request without it compiles forward liveness, whose runs
refuse `backward`;
run by `Plan::forward`, whose results
are bit-identical to the interpreter's — a plan changes what is
*stored*, never what is *computed*. Plans are graph-structural and
self-contained: a plan freezes its own copy of the spec at compile
time and takes the caller's `Parameters` per call, so a plan never
held state, compile-once amortizes over a whole training run, and
reopening the network simply records past the plan's prefix.
`Plan::describe` renders the decisions —
per-node liveness spans and the static live-volume story. The tape
remains the specification and the plain interpreter the executable
oracle every plan is differentially tested against.

**Keep-set.** The declared observable values of a plan: its roots
plus explicitly observed interiors. Only the keep-set answers
`Run::of` on a plan run — an interior value stays unreadable
even when liveness happens to retain it, so the read contract never
depends on the optimizer's choices. Observability is declared, never
inferred; the target-sliced run's computed set is this idea's
first, implicit form.

**Liveness (buffer).** Which run buffers a plan still needs at each
step: a slot whose last consumer has run, and which nothing later
can read, is released immediately behind a non-allocating
placeholder, so a run's peak memory follows the widest genuine
dependency window instead of the whole tape. Forward-only plans keep
only the keep-set; engine-backward plans additionally keep what the
read contract names.

**Reads (contract).** Which payload *values* each operation's
derivative rule reads when it runs: per-operand flags and an output
flag, declared beside every `backward` (`Mul` reads both operands,
`Tanh` its own output, `Gather` its selection's indices, the view
family nothing at all — shape-only reads need no entry, because
freed slots hold shape-correct placeholders). It gives an
engine-backward plan its memory *floor*: the view chains, padded
copies, and pure-arithmetic intermediates are releasable with
gradients still exact to the bit. Engine runs report the floor
rather than executing it — per-step mid-run freeing measured as a
peak-RSS regression under the system allocator, and the graded route
for training memory is recorded gradients under forward liveness —
while forward-only plans execute their releases, where the win is
measured. Keeping a rule and its read set in step is part of
changing either.

**Pattern (catalog).** A closed, documented set of patterns — same
spirit as `Function` — over the plan's frozen columns
(`src/engine/pattern/`). A pattern is a compile-time match, not a
tape rewrite. Compilation *discovers* once: every closed candidate
(unnamed interiors unreadable, every wanted consumer inside the
match) pools in priority order, posture-blind and
consumer-independent. Each consumer then *elects* its own catalog
from the pool under its repertoire — the patterns it can act on: a
forward run fuses its elected groups (its kernel table lives beside
the plan; the home repertoire is empty on engine-backward plans), and
StableHLO
emission raises its elected groups with a total repertoire, on every
memory posture. Electing is claiming, first-wins; unsupported
candidates never claim, so their regions stay free, and an unelected
region simply runs or lowers its recorded primitives — a pattern is
an offer, never an obligation. A repertoire is a kernel library with
an admission bar: a home kernel must reproduce the recorded chain
bit for bit under the run's `Numerics` posture, while a raise needs
only emission's conformance envelope. Matchers share one `View` (wanted,
keep-set, consumer counts); dispatch everywhere is a plain enum
`match`; the tape and the backends never see a pattern.

**Fusion (window-GEMM).** The catalog's first pattern: the
canonical im2col chain — pads, two unfolds, the permute, the patch
reshape — feeding a `matmul`. The match lives in the catalog, and
fusion and raising are its two actions: `Plan::forward` executes the
group as one `Tensorial::windowed_product` call with the chain never
materialized, and `Plan::emit_stablehlo` raises the group to
`stablehlo.convolution`. Matching is structural and provenance-blind
(a hand-written chain identical to `conv2d`'s fuses identically), a
keep-set node inside the chain is a fusion barrier, and home fusion
follows the plan's memory posture: forward-only plans always fuse (a
pure win — the chain simply never exists, and recorded-gradient
training plans are forward-only, so they fuse too), while
engine-backward plans stay unfused so their memory contract stays
exact for the reverse scan. The raise is not posture-gated: an
engine-backward plan emits `stablehlo.convolution` exactly like its
forward twin. `describe` prints the groups; recognition proposes,
payloads and backends dispose — neither ever sees graph structure.

**Pool window (reduce_window).** The catalog's first raise-only
pattern: the canonical `max_pool` spelling — two square
unfolds, the lane permute and reshape, the left-associated `maximum`
fold in lane order, the trailing squeeze — rooted at the squeeze
reshape. It has no home action: it sits outside the home repertoire,
so forward runs execute the recorded fold unchanged on every memory
posture. Emission raises the group
to `stablehlo.reduce_window` over the rank-4 source, so the unfolded
lanes never cross the boundary. A balanced fold tree, a permuted lane
order, or an omitted squeeze is a documented false negative;
`describe` never mentions raise-only matches.

**Batch-norm raise (batch_norm_training / batch_norm_inference).**
Two raise-only catalog patterns over the recorded `BatchNorm` formulas,
rooted at the trailing shift `Add`. The training variant matches the
batch's own statistics — each `mean_along` verified as
`Div(SumAlong, counted leaf)` through `Differentiable::is_counted`,
so an unverified divisor (an unbiased variance) never raises — and
emits `stablehlo.batch_norm_training`, whose three results name the
output, mean, and variance; the statistics are *named results*, the
keep-set refinement: they may be observed (training loops read them
for running estimates) and are emit-skipped, receiving their SSA
names from the raise at the root. The inference variant matches the
same tail over supplied statistics and emits
`stablehlo.batch_norm_inference` with the statistics as ordinary
arguments. The single-value epsilon leaf rides as the operation's
`f32` attribute. Unnamed interiors (the centering, the deviation)
still bar when readable, and a statistic feeding an expression
outside the formula bars the match.

**Field.** A value-aligned buffer: one payload per node, carrying its
network family's origin rather than borrowing anything, so it can be
combined across runs (averaging data-parallel gradients) and carried
across training steps (momentum velocity, Adam moments). Supports
elementwise algebra — `+`, `scale`, `zip`, `map` — with kinship (same
origin, same covered length) checked on every combination;
`Parameters::step` takes any field as its update direction. In physics terms, a `Gradients` is
a discrete gradient field over the graph, which is why `Gradients` is an
alias for `Field` rather than a wrapper around it: the buffer's invariant is
alignment to a graph, not differentiation, and Adam's second moment or a
run's forward payloads are fields that are not gradients at all. In
topos: [`Field`](src/graph/field.rs).

**Origin.** The identity of one tape-network family: a `Copy` token
minted from a process-global counter at `Tape::new` and carried through
every `into_network`/`into_tape` round trip; same-origin is equality.
Because both conversions consume their operand, at most one live tape
or network carries an origin at a time, and positions within it are
stable forever — which is why origin plus node position is the whole
identity a detached carrier (`Symbol`, `Field`, `Parameters`, `Plan`,
`Run`) needs, and the whole runtime price of detachment: two integer
compares. In topos: the crate-internal
[`Origin`](src/graph/origin.rs), embedded in every `Symbol`.

**Payload (`Data`).** The numeric value a node carries: a scalar
(`f32`/`f64`) or an elementwise [`Tensor`](src/payload/tensor.rs). Its
contract is the [`Differentiable`](src/payload/differentiable.rs) trait —
arithmetic operators, `zero_like`/`one_like`, the shape-derived constant
constructor `counted` (a payload of a given shape holding an integer
count, which lets composed formulas mint axis extents — a mean's
divisor — without borrowing a payload to copy the shape from), the
`Accumulator` associated type naming what accumulating operations
(matmul, the sum reductions, `fold`, `scatter`) compute in before one
final rounding — `Self` for the IEEE singles, `f32` for `Bf16` — and
`Send + Sync`;
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
representation allows, or copied out with `to_vec`. In topos:
[`Tensor`](src/payload/tensor.rs).

**Bf16 (brain float 16).** The truncated-single float format — one sign
bit, the eight exponent bits of `f32`, seven stored mantissa bits — and
topos's first payload beyond the IEEE singles: a `u16` newtype whose
every operation converts to `f32`, computes there, and rounds the result
to nearest-even. Same range as `f32`, precision of about two decimal
digits; integers are exact up to 256. Half the memory of `f32` at rest,
deterministic on every platform, and an ordinary `Differentiable` +
`Elementary` implementation (plus the scalar-identity `Tensorial`), so
`Tensor<Bf16>` and `Network<Bf16>` run the engine unchanged. The
accumulating operations — matmul, the sum reductions, `fold`, and
`scatter` — are the documented exception to the per-op semantic: its
`Accumulator` is `f32` (the bf16 hardware convention), every term
promotes exactly, and only the final total rounds; the emitted
StableHLO states the same semantic through `f32`-typed contractions
and reduces with explicit `convert`s. In topos:
[`Bf16`](src/payload/bf16.rs).

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
In topos: the crate-internal [`Storage`](src/payload/storage.rs).

**Layout.** How a dense buffer's logical indices map onto its flat
storage: the shape, the per-axis strides, and the offset of the first
element. The element at multi-index `(i0, ..., in)` lives at
`offset + sum(i_k * strides_k)`. A contiguous row-major layout has
`strides_k = product(shape[k + 1 ..])` and offset zero; view operations
produce a new layout over the same buffer without moving any element. A
stride of `0` marks a broadcast axis, whose steps do not advance within
the buffer, which is how `broadcast_along` repeats without copying. In
topos: the crate-internal [`Layout`](src/payload/layout.rs).

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
`unfold` slides windows along an axis whose gradient `fold`s back with
per-position accumulation (the fourth adjoint pair; see the sliding
windows entry),
and `gather` selects table rows by a one-hot `Selection` whose gradient
`scatter`s back, accumulating rows selected more than once (`gather` with
`scatter` as the fourth pair, and the embedding lookup). The selection is
data, so `gather`'s backward has no gradient term for it at all: the
non-differentiability of the indices is a structural property of the
operation, not a runtime flag. `max_along` is `sum_along`'s
order-theoretic sibling — the same axis reduction, folding with the
elementwise `maximum` — and serves stable normalization rather than
recording: `log_softmax` and `logsumexp`, the two fused operations,
shift by the axis maximum before exponentiating (which no composition
of recorded operations could do); the former routes its gradient as
`g - exp(output) * sum_along(g)` and the latter as the softmax
`exp(operand - output)`, each recovering the probabilities from the
node's own output. Broadcasting is explicit by design: a
single value spread across a named reference's shape, or a payload
repeated along one named axis of a reference — the axis is always
written, and no operation aligns shapes implicitly. In topos: the
[`Tensorial`](src/payload/tensorial.rs) trait, recorded into graphs via
`Value::matmul`, `transpose`, `sum`, `sum_along`, `broadcast_like`,
`broadcast_along`, `reshape`, `permute`, `narrow`, `gather`,
`scatter`, `fold`, `step`, `log_softmax`, `logsumexp`, and the
`reshape`-based `squeeze` and `unsqueeze`. The last three adjoints
(`scatter` to `gather`, `fold` to `unfold`, `step` as the `maximum`
family's locally constant mask) close the op set under
differentiation: every derivative rule's expansion is made of
operations that themselves have rules.

**Unfold (sliding windows) / fold.** The windowing pair behind
convolution and pooling. `unfold(axis, size, step, dilation)` replaces
an axis with a `(count, size)` pair — window `w` starts at `w * step`
and takes every `dilation`-th element — as a strided *view* over the
shared buffer: overlapping windows alias elements read-only, which
immutability makes safe. `fold` is its adjoint and gradient rule: the
window pair folds back onto an axis of a given extent, each source
position summing, output-centrically and in window order, the window
elements read from it — deterministic under any evaluation strategy.
Two `unfold`s produce 2-D windows (torch semantics). In topos:
[`Tensorial::unfold`/`fold`](src/payload/tensorial.rs) over
[`Layout::unfold`](src/payload/layout.rs), recorded by
[`Value::unfold`](src/graph/value.rs) (with `fold` as its gradient
rule, a payload method rather than an opcode until transposed
convolution needs the forward direction); `Value::pad` records
`narrow`'s adjoint the same way.

**Arena.** Append-only storage in which every recorded node lives exactly
once; allocations never move or drop while the arena lives, which is
what makes structure freezes cheap and references stable. Provided by
the [`cow_vec`](https://crates.io/crates/cow_vec) crate inside the
tape's columns: runs, plans, and `differentiate` clone the columns in
O(1) and share the arena, and the seal moves them without copying.
Training never touches it: parameter payloads live in the caller's
`Parameters`, so the arena holds structure only. What-ifs need no
arena story at all — one spec, any number of `Parameters` clones, and
state was the only thing ever worth copying.

## Acceleration

**GEMM.** General matrix-matrix multiplication, the dense core of
`matmul` and the unit of acceleration: one job multiplying an
`m x k` operand by a `k x n` operand into a contiguous row-major
product. In topos: [`GemmTask`](src/payload/gemm.rs), which
describes the job as two spanning slices read through per-axis
strides — a transposed or narrowed view is a stride pattern, not a
copy — plus the three extents, validated at construction and
read-only thereafter.

**Seam.** The point where payload math may hand a job to hardware
without the engine knowing: a provided `Elementary` method answers
`None` (compute on the built-in paths) unless the element type
forwards to the backend chain, as `f32` and `f64` do. The seam has
two hooks — `Elementary::gemm` for one dense product, consulted
first by `Tensor`'s `matmul`, and `Elementary::map` for one
whole-buffer transcendental (a `MapOperation`: `exp`, `ln`, `sqrt`,
`tanh`), consulted by the tensor's elementwise operations for
contiguous dense buffers. The hooks crossed with the forwarding
precisions are the leaf entries of the acceleration vocabulary,
[`Formula`](src/backend/formula.rs). Both live in the payload tier, so
`Operation` rules stay backend-blind (the columns-as-IR rule) and
custom payload implementations keep the defaults. In topos:
[`Elementary`](src/payload/elementary.rs).

**Backend.** An implementer of named formulas, in LLVM's sense of
the word: the hardware kernel providers, the crate's own fused
kernels (`Backend::Fused`), and the StableHLO translation library
(`Backend::StableHlo`) are one axis, in
[`backend`](src/backend/mod.rs). What an implementer can do is the
coverage matrix (`Backend::coverage`): one cell per
[`Formula`](src/backend/formula.rs), declaring the certification
bar (`Bar::BitIdentical` or `Bar::Envelope`) and the forwarding
precisions its kernels accept — the single declared truth that
offer chains agree with by test, that the plan's election reads
(the `Fused` column, under the bar the request's numerics demands),
and that emission reads (the `StableHlo` column, total). Each
implementer answers for itself through the crate-internal
`Implementer` contract: a descriptor in the backend's own module,
always compiled — its coverage row, dispatch attribute, build
facts, and status — with its kernels behind the feature `cfg` in a
`kernels` submodule, so the enum stays the public axis and every
answer is a plain-match delegation, no trait object anywhere. How
kernels are reached is the `Dispatch` attribute: offered buffer
jobs down `Formula::chain` at run time (the four hardware
implementers), elected onto plans at compile time (`Fused`), or
translated into a foreign module (`StableHlo`). Every offered
member may decline any job (wrong size, wrong platform, unavailable
device), and the built-in paths answer when the whole chain
declines: coverage declares *may*, the offer decides *will*, the
reference paths define *is*. Each job type carries its formula and
precision (the crate-internal `Job` contract), so a job can only
walk its own chain. The chain is compile-time: enabling a feature
is the activation, no per-call-site routing exists, and within one
binary two identical runs can never disagree; election keys on
`Backend::compiled` (build facts), never on device presence, so a
plan's shape depends only on the binary. The one run-scoped control
is the `Numerics` posture below — admission by bar, never
re-routing. The offer chains have four hardware residents, tried in
order.
`Backend::Accelerate` (the `accelerate` feature) leads: it takes
dense `f32` and `f64` products above a small flop threshold through
`cblas_sgemm`/`cblas_dgemm` (the AMX/SME matrix units on Apple
Silicon), declining stride patterns BLAS cannot express, and maps
whole-buffer transcendentals through vForce
(`vvtanhf`/`vvexpf`/`vvlogf`/`vvsqrtf` and their `f64` twins) — the
vectorized form of the loops that scalar libm calls keep serial.
`Backend::Metal` (the `metal` feature) runs large `f32` tasks on
the GPU through hand-written simdgroup-matrix kernels compiled from
source at first use — Metal has no `f64` — serving what BLAS
declines and everything large in metal-only builds; a failed setup
or runtime error poisons it into declining forever, degrading to
slow rather than wrong. `Backend::Cuda` (the `cuda` feature, Linux
only) runs large `f32` and `f64` products through cuBLAS on an
NVIDIA GPU, binding `libcudart`/`libcublas` at run time by `dlopen`
so a machine without them declines at run time instead of failing
to build; PCIe copies bound every task, so its threshold is high,
and it shares the BLAS stride classification with `accelerate`
under a column-major swap. `Backend::Simd` (the `simd` feature)
closes the chain: the `matrixmultiply` crate's tuned, single-threaded CPU
microkernels with runtime instruction-set dispatch (AVX-512F,
AVX2+FMA, AVX, NEON) for both `f32` and `f64` — the portable rung,
real on every platform where the Apple backends are macOS-only, and
mop-up behind them on macOS. Whatever the whole chain declines
lands on the built-in paths. The two in-process implementers stand
outside the offer chains: `Backend::Fused` holds the crate's fused
kernels for composed formulas (`windowed_product` today, the one
cell at the bit-identity bar, since the oracle's bits live in this
process), and `Backend::StableHlo` holds the total translation
column emission elects by. `Backend::coverage`,
`Backend::serves`, `Backend::compiled`, and [`Backend::status`]
answer for every implementer in every build — coverage as declared,
and `NotCompiled` as an ordinary result, not a compile error — and
the default build still compiles
no backend and keeps `#![forbid(unsafe_code)]` verbatim; a backend
build confines `unsafe` to the backend's `kernels` submodule under
a crate-wide `deny` with one scoped allow — the always-compiled
descriptor half sits outside the allow.

**Numerics (Exact / Fast).** The two-valued numerics posture of a
plan's runs, chosen on the compile request and carried by the plan
and its runs (`Request::numerics`, `Plan::numerics`). `Fast` — the
default, and the fixed posture of interpreter runs and host-side
payload calls — is the backend chain as compiled, its per-task flop
thresholds serving as cost heuristics inside the posture. `Exact`
demands the bit-identity bar (`Numerics::bar`): only kernels
certified bit-identical to the reference may serve — today the
fused window kernel and nothing offer-dispatched — so chain work
computes on the built-in reference paths: the same bits as the
default build, in every build — the oracle, always one compile
away, which makes an exact and a fast result comparable in one
process. Reordering float math is
always this labeled choice, never a silent effect of a feature flag;
a run's `backward` re-enters its forward posture, so gradients
follow the same paths. In topos:
[`Numerics`](src/backend/numerics.rs).

## Neural building blocks

**Module.** A named, parameterized recording function — the unit of
model composition: `express(&tape, input)` records the module's
formula through the public op surface and answers the output value,
and `visit` walks its parameter tree for the serialization boundary.
Modules hold parameter `Symbol`s, never payloads (the caller's
`Parameters` owns state), so they are detached: `'static`, storable,
and recording against the family's tape whenever it is open.
Expression is record-time only — the cost never reaches a run, a
plan, or a kernel — which is why composing through `dyn Module` (in
`Sequential`) sits inside the sanctioned dynamic-dispatch exceptions.
Programmatic access (tying, freezing) goes through typed accessors
(`weights()`, struct fields), never names; names exist only as the
structured `Path`/`Segment` checkpoint identity. Distinct from a Rust
module (a namespace): this is the ML term of art. In topos:
[`Module`](src/neural/module.rs).

**Sequential.** The ordered module chain: each stage's output feeds
the next, stages heterogeneous behind `dyn Module`, appended with the
boxing `then`. Skip connections stay hand-written where they are
used — the pre-norm blocks of the transformer examples add around a
normalized inner expression, a shape no generic wrapper fits. In
topos: [`Sequential`](src/neural/sequential.rs).

**Checkpoint.** A module tree's parameter payloads, captured and
restored in two identities: positional (`snapshot`/`restore`, visit
order — sufficient for resuming the same code) and named
(`named_snapshot`/`named_restore`, matched by structured path — what
survives code evolution and what foreign name-to-tensor checkpoints
require). A checkpoint is pure state, so both directions are plain
`Parameters` transforms — no graph is touched, and nothing mutates. In
topos: [`checkpoint`](src/neural/checkpoint.rs).

**Linear.** The dense (fully connected) affine transform at tensor
granularity: `x . w + b` over a `[batch, inputs]` value, with one
`[inputs, outputs]` weight parameter and one `[outputs]` bias met
through the explicit axis broadcast — a handful of tensor nodes
instead of one node per scalar weight, and deliberately *unfused*: an
activation is its own composition stage, which unlocks the orderings
a bundled activation forbids. Weight tying goes through the symbols a
module already exposes, the way GPT-2's example records its head over
the embedding table's transpose. In topos:
[`Linear`](src/neural/linear.rs).

**Mlp.** A multilayer perceptron: affine stages chained by a topology
of value widths (`[3, 4, 4, 1]`), hidden stages squashing with `Tanh`
and an affine output stage. The convenience constructor over
`Linear`, with initialization owned by the caller through a
shape-to-payload initializer. In topos:
[`Mlp`](src/neural/mlp.rs).

**Batch normalization (BatchNorm).** Standardizing every feature of a
`[batch, features]` value by minibatch statistics and applying a learned
per-feature affine `scale * normalized + shift` (Ioffe & Szegedy, 2015).
Training mode normalizes by the batch's own mean and biased variance,
with gradients flowing through the statistics; inference mode normalizes
by running estimates accumulated during training. In topos:
[`BatchNorm`](src/neural/batch_norm.rs), whose `express` records the
training-mode expression and returns a `Normalization` — the output plus
the batch-statistic values — and whose `express_with` records the
inference-mode expression over statistics supplied as values. The layer
stores no running statistics: they are fed as per-run inputs on the
inference expression, and their exponential moving average lives in
payload land with the training loop, so the tape stays a pure record of
the computation.

**Layer normalization (LayerNorm).** Batch normalization's stateless
sibling: every sample is standardized by its own feature statistics —
the mean and biased variance taken along the feature axis instead of
the batch axis — and passed through the learned per-feature affine
(Ba, Kiros & Hinton, 2016). Samples normalize independently, so there
is no batch coupling, no running estimates, and no training/inference
split: one recorded expression serves both. The transformer stack's
norm. In topos: [`LayerNorm`](src/neural/layer_norm.rs).

**Convolution (Conv2d).** Sliding a stack of learned kernels across a
`[batch, channels, height, width]` value: each output position is the
kernel-weighted sum of a window of the (zero-padded) input, plus a
per-filter bias. In topos it is a composed formula, not a
primitive: [`conv2d`](src/neural/convolution.rs) records `pad`, two
single-axis `unfold`s, an axis permutation, the im2col reshape (the
formula's one deliberate copy, which turns the whole computation into
a single rank-2 `matmul` on the GEMM seam), and the bias broadcast —
so the backward is pure chain rule through those operations' adjoints.
The [`Conv2d`](src/neural/convolution.rs) facade holds torch-shaped
`[filters, channels, kernel_height, kernel_width]` weights and records
the weight-side `permute` + `reshape` to the GEMM operand per run.

**Im2col.** Rewriting convolution as a matrix product by laying every
kernel-sized window out as a matrix row (`[windows, channels *
kernel]`), so one GEMM computes all positions and filters at once. The
classic eager-framework trade: a `kernel`-fold memory copy buys the
fastest kernel the machine has. In topos the copy is not special
code — it is `reshape`'s ordinary view-else-copy fallback firing on
the overlapping window view.

**Pooling.** Downsampling a spatial value by reducing each window to
one number. In topos [`max_pool`](src/neural/pooling.rs) composes
over the same `unfold` windows as convolution and folds the window
lanes with the binary `maximum`, left-biased, so a tied maximum
routes its gradient to the earliest window position — deterministic,
like every tie rule in the crate. No dedicated reduce opcode exists
for pooling (a fused `MaxAlong` stays a deferred option), and no
other pooling flavor ships without a consumer: an average pool is one
`mean_along` over the same windows, caller territory until then.

**RMS normalization (RMSNorm).** Layer normalization without the
centering and the shift: every sample is divided by the root mean
square of its features, `sqrt(mean(x^2) + epsilon)`, and scaled per
feature (Zhang & Sennrich, 2019) — re-scaling alone, on the
observation that the re-centering half contributes little. Stateless
like `LayerNorm`, and the cheaper modern default of transformer
stacks. In topos: [`RmsNorm`](src/neural/rms_norm.rs).

**Dropout (mask-fed).** Randomly silencing features during training
so co-adapted detectors cannot lean on each other (Srivastava et
al., 2014). In topos the randomness stays outside the graph: a
[`Dropout`](src/neural/dropout.rs) module multiplies its input by a
declared mask *input* whose default payload is all ones, so an unfed
run is the identity — inference is the absence of a feed, not a
mode — and the seeded
[`init::dropout`](src/neural/init.rs) factory draws inverted-dropout
masks (each element `0` or `1 / keep`) host-side, fed per training
step like any other run state. An in-graph RNG opcode is rejected
permanently: it would break bit-exact seeded replay, make the
interpreter-versus-backend differential test meaningless, and hide
generator state inside the spec. The keep probability is caller
territory, chosen where the mask is drawn.

**Running statistics.** The exponential moving averages of the batch
means and variances that batch normalization accumulates during training
and normalizes by at inference. Deliberately not engine state: the
training loop reads each batch's statistics from a `Run`,
averages them as plain payloads, and feeds the estimates to the
inference expression per run — the same division of labor as minibatch
assembly.

**Optimizer.** A training-step strategy: how gradients and the current
`Parameters` state become the next state. The loop-land analogue of
what `Activation` is to a layer — a uniform slot — kept an open,
object-safe trait so custom optimizers are ordinary implementations,
with `Field` algebra as the designed state carrier (moments are
fields, carried across steps beside the parameters). `Sgd` is the
stateless base case, `Adam` adds bias-corrected moments (powers
carried as payloads, so steps are exact and deterministic), and
`AdamW` adds decoupled weight decay under a structural policy: rank
two and above decays, rank one is spared, decided through the
identity-aware `Parameters::step_each`. The learning rate is a
per-step argument — schedules stay caller-owned loop arithmetic. In
topos: the [`Optimizer`](src/neural/optimizer.rs) trait and
[`Adam`/`AdamW`](src/neural/adam.rs).

**Activation.** The nonlinearity applied to a stage's affine output,
which is what gives stacked affine maps expressive power. It is a
graph operation like any other, so it participates in differentiation
(`Function::Tanh`, recorded by `Value::tanh`, whose derivative
`1 - tanh(x)^2` reuses the node's own output; `Function::Relu`, recorded
by `Value::relu`, whose gradient is masked by the 0/1 `step` indicator —
a dedicated unary variant so the mask costs one node and no zero leaf
per occurrence, while the rule reaches its zero at run time through
`zero_like`). The enum carries exactly that dedicated pair; every
other activation is a short caller-side composition over the public
surface whose gradient is the chain rule, the way GPT-2's example
composes its GELU — the once-shipped `Sigmoid`, `LeakyRelu`, and
`Elu` variants were retired when no consumer materialized. Each
variant also states its initialization `gain`, the factor
[`init::scaled`](src/neural/init.rs) compensates at initialization.
In topos: the [`Activation`](src/neural/activation.rs) enum and its
`Activation::express`.

**Loss.** A scalar training objective written as a composed formula over
recorded operations, not as a primitive: its gradient falls out of the
chain rule with no dedicated backward rule. A formula earns a fused
`Function` variant only where composition cannot express it — the
cross-entropy loss composes the expanded form
`((targets.sum_along(1) * logsumexp(logits)).sum() - (targets *
logits).sum()) / targets.sum()`, keeping only `logsumexp` fused (for
the stabilizing max shift) and staying composition everywhere else;
the expansion is exact mathematics, and it is the stable spelling
because no term multiplies a zero target by an infinite
log-probability. Target weights must be finite and nonnegative with
positive total mass. The normalizer is the targets' total mass — the batch
size for one-hot targets, so the reduction is the standard mean, while
soft or weighted targets normalize by their own weight. The same one-hot
`Selection` that feeds an embedding gather serves as the targets, fed per
run. Losses are the third tier of the operation surface (see Composite):
free functions rather than `Value` methods, because their operands play
distinct roles and a method would arbitrarily privilege one of them. In
topos: [`cross_entropy`](src/neural/loss.rs) in the loss module.

**Initializer.** The shape-to-payload closure a caller hands to a
building block at construction: initialization is caller-owned, and
`Linear` and `Mlp` record whatever they are given. The `init` module
manufactures deterministic initializers — `uniform` and `normal` fill
any shape, while the fan-aware `xavier` and `kaiming` read the fan-in
off the requested rank-2 shape and zero rank-1 shapes, a bias
identifying itself structurally by its rank. Every factory takes an
explicit seed and each closure owns its splitmix64 generator state: no
global generator, no clock, bit-identical runs forever — which is why
the crate carries its own few-line generator instead of a `rand`
dependency, whose standard generator is unstable across versions. The
factories are element-generic through `init::Sample`: the generator
pipeline runs in `f64` and converts once at the end, so the `f64`
path is the identity (seeded outputs stay bit-identical forever,
pinned by a golden-bits test) and the `f32` path is the same stream
rounded once per element, with the element inferred from the network
the closure feeds. In topos: [`init`](src/neural/init.rs), the
crate's one public module, qualified because `uniform` and `normal`
are meaningless names without it.

## Further reading

- R. E. Wengert, "A simple automatic derivative evaluation program" (1964)
  — the original tape.
- A. Griewank and A. Walther, *Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation* (2008).
- A. G. Baydin et al., "Automatic Differentiation in Machine Learning: a
  Survey", JMLR (2018).
- A. Karpathy, [micrograd](https://github.com/karpathy/micrograd) — the
  educational engine topos is loosely inspired by.
