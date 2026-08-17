//! StableHLO emission: writing a compiled plan down as interchange text
//! for the XLA world.
//!
//! A [`Plan`](crate::Plan) is already a closed, pure, statically shaped
//! tensor function — exactly the input industrial compilers such as XLA
//! and IREE schedule best — so topos does not grow a code generator:
//! [`Plan::emit_stablehlo`](crate::Plan::emit_stablehlo) serializes the
//! plan into the StableHLO dialect's textual form and any toolchain
//! outside the crate takes it from there. The tape stays the observable
//! spec and the interpreter stays the semantic oracle the emitted
//! module is checked against; emission is a plan consumer, a sibling of
//! `describe`.

mod builder;
mod lower;

pub use builder::Emittable;
pub use lower::EmitError;
