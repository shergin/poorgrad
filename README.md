# topos

An autodiff compiler stack in miniature, written in Rust: record a
graph once, inspect every node, differentiate it, compile it into a
plan you can read, and emit StableHLO when you want an industrial
backend. The plain interpreter ships forever as the executable spec;
everything faster must match it, bit for bit.

What it is trying to be:

- **The whole modern ML-compiler stack, small enough to read.**
  Every stage visible and printable, none of it magic.
- **Every result provable.** A claim is one assert away from proof.
- **A place for learning and research.** New ideas plug in at named
  seams, with the interpreter as ground truth. The core stays closed
  and simple on purpose.

Wider adoption may happen; it is never chased. No benchmark races,
no coverage races, no kernel zoo, no plugin bazaar.

## The name

A topos — the Greek *τόπος* — is a place: a small, self-contained
place where the whole compiler stack stays in view.

## License

Licensed under either of MIT or Apache-2.0, at your option.
