# Roadmap

What is planned and what has been decided, so neither gets lost. This
file is part of the codebase contract: when an item ships or a decision
changes, update it in the same change.

## Next: tensors at module granularity (phase 3)

- Per-run input feeding: a `forward_with(feeds)` that binds payloads to
  leaf symbols for one run. Inputs become run arguments instead of graph
  state, fed payloads validate against recorded shapes, and different
  threads can forward the same network on different batches
  concurrently — true data parallelism.
- `Layer` at tensor granularity: `x.matmul(w) + b` records three nodes
  instead of `inputs * outputs` scalar nodes; decide whether `Neuron`
  remains a scalar-mode teaching type or retires.
- `Mlp`: chained layers with micrograd-style topology (`&[3, 4, 4, 1]`),
  plus a capstone example (XOR or two-moons) exercising feeds, tensor
  layers, and training end to end.

## Training API extensions

- Direct parameter assignment (checkpoint loading, re-initialization): a
  batched generational setter (`with_parameters`) and, under `&mut self`,
  an in-place exclusive form. The invariant stays: a shared network
  never observably changes under readers.
- Multi-seed backward (`grad_outputs`-style cotangents): gradients of
  weighted target combinations without synthetic sum nodes, and the path
  to separate per-task gradients for gradient surgery.
- `Field` reductions (`dot`, `norm`) for gradient surgery and
  global-norm clipping; per-parameter clipping already works through the
  update closure.
- Update-closure identity: pass the parameter's `Value` (and accept
  `FnMut`) for ad-hoc per-parameter logic and logging. Optimizer state
  itself is already covered by field algebra.
- Parameter interpolation between generations (weight averaging,
  exponential moving averages); builds on direct assignment.

## Op set gaps

- `powf`: the last `Elementary` operation without a `Function` variant.
  The exponent-side gradient needs `ln(base)` and a positive-domain
  decision, so it earns its own design pass rather than a stamping.
- ReLU and friends: need comparison or max on the payload contract — a
  deliberate trait extension.
- Axis-wise `Sum` (today only the full reduction), batched matmul, and
  rank > 2 transpose.

## Performance passes (phase 4)

- Backward pruning: start the reverse scan at the target's index, then
  an ancestor mask to skip non-contributing nodes entirely.
- Shape-aware preallocation of run buffers (shapes are already stored).
- A GEMM backend (`matrixmultiply` or BLAS) behind a feature flag.
- In-place field algebra to cut allocation churn in training loops.
- Literal deduplication (hash-consing repeated constants).
- Micro: `ValueId` as `u32` (node 24 to 16 bytes), shape interning.
- In `cow_vec` (optional, design-consistent): a documented
  element-stability guarantee, which would legitimize lock-free reads
  downstream.

## Publishing

- Version bump and a crates.io release once phase 3 lands and every
  README promise is demonstrated by an example.

## Settled — do not relitigate without new evidence

- Broadcasting is explicit only: `broadcast_like` spreads a single value
  across a named reference's shape; there are no implicit alignment
  rules.
- Runtime shapes with record-time inference beat type-level shapes here:
  the record-once model surfaces shape errors before anything runs, at
  no type-system cost.
- No const-generic neuron arity and no `SmallVec` weights (fan-in is
  data-sized and unbounded); `Shape` itself *is* a `SmallVec` (rank is
  structural and tiny).
- Values are borrows: `Copy` proxies that cannot outlive their network.
  Detached identity lives in `Symbol`; cross-generation state lives in
  `Field` with lineage kinship.
- `CowVec` stays: `updated`'s copy-on-write `set` is load-bearing. A
  bespoke append-only store becomes interesting only if parameters ever
  move out of the graph.
- Positional misuse panics everywhere (`resolve`, `of`, operators,
  recording, `updated`); `try_resolve` is the one probing form.
- Consumer-shaped concerns stay out of general-purpose crates: what
  poorgrad needs beyond `cow_vec`'s identity lives here, under
  conceptual names.
