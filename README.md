# topos

An autodiff compiler stack in miniature, written in Rust: record a
graph once, inspect every node, differentiate it, compile it into a
plan you can read, and emit StableHLO when you want an industrial
backend — with the plain interpreter shipping forever as the
executable spec that everything faster must match, bit for bit.

## The goal

Be for ML what LLVM became for compilers — in the way LLVM actually
happened: a research project whose first-principles design quality
made it the easiest place to build the next thing, so adoption
followed on its own. For topos that means three commitments:

- **The whole modern ML-compiler stack, small enough to read.**
  Record a graph, inspect it, differentiate it, compile it, fuse it,
  hand it to an industrial backend — every stage visible and
  printable, none of it magic.
- **Every result provable.** The plain interpreter is the executable
  spec; anything faster must match it, bit for bit by default. A
  claim is one assert away from proof.
- **Built for learning and research.** New ideas plug in at named
  seams — payloads, backends, optimizers, modules, emission targets —
  with the oracle as ground truth. The core stays closed and simple
  on purpose.

Wider adoption may happen; it is never chased. No benchmark races, no
coverage races, no kernel zoo, no plugin bazaar.

## The rules

Nine invariants govern every change:

1. **The tape is the spec.** Recording never changes meaning;
   optimization is always a derived artifact.
2. **Say what you want to read.** Observability is declared (roots
   and observes), never inferred; reads outside it fail loudly.
3. **Bit-exact by default.** Seeded runs replay exactly; anything
   that reorders float math is a labeled option, never silent.
4. **Composition first.** A fused or native form must earn its place
   with a measured reason — float behavior or real cost.
5. **Facades are the stable surface.** Users go through facades;
   internals may change their spelling freely underneath.
6. **The interpreter is the oracle.** It ships forever, and every
   plan, backend, and emitted module is differentially tested
   against it.
7. **Unsafe stays caged.** Feature-gated, scoped, documented; the
   default build forbids it.
8. **Static shapes; tapes are cheap.** One tape per shape bucket,
   never symbolic shapes; plans survive every generation.
9. **Consumers before machinery.** Nothing lands without a real
   in-repo user and a number that grades it. Numbers decide — and
   the rule cuts both ways: machinery whose consumer disappears is
   retired.

## Where to look

- [`examples/`](examples/) — the curriculum, from chained scalar
  expressions through the makemore acts to convnets and GPT-2 with
  the released weights; every stage of the stack landed with a
  consumer that uses it and a measured number that grades it.
- [TERMINOLOGY.md](TERMINOLOGY.md) — the vocabulary, from the
  scientific meaning of each term to the Rust type it names.
- [ACCELERATION.md](ACCELERATION.md) — the opt-in hardware backends
  and StableHLO emission, with every claim measured.
- [NOTEBOOKS.md](NOTEBOOKS.md) — the crate in a Jupyter notebook,
  with no wrapper API.

## The name

A topos — the Greek *τόπος* — is a place, and this crate is one: a
small, self-contained place where the whole compiler stack stays in
view. It began life as `poorgrad`, a poor man's autograd, one road
out of [Karpathy's `micrograd`](https://github.com/karpathy/micrograd).

## Contributing

Issues and pull requests are welcome. Designs here are decided by
measurement and written down before code, so for larger changes,
opening an issue first is appreciated. CI expects `cargo fmt`,
`cargo clippy --all-targets -- -D warnings`, and `cargo test` to
pass; matching that locally is the whole checklist.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.
