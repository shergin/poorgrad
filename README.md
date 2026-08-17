# topos

A small autodiff compiler stack in Rust: record a graph, inspect
it, differentiate it, compile it, emit it. The interpreter ships
as the spec; everything faster must match it, bit for bit.

The stack stays small enough to read. Every result is provable.
New ideas plug in at named seams; the core stays closed.

See [`examples/`](examples/).

## The name

A topos is a place — here, one where the whole compiler stack
stays in view.

## License

Licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
