use static_assertions::assert_impl_all;

use super::{Origin, ValueId};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Symbol: Send, Sync, Copy);

/// A detached, `Copy` name for a recorded value: the currency of every
/// phase after recording.
///
/// Unlike [`Value`](crate::Value), a symbol carries no tape borrow and
/// no payload; it is an origin plus a node position, valid across
/// threads, notebook cells, checkpoints, and
/// [`Network::into_tape`](crate::Network::into_tape) round trips —
/// linear extension never moves a recorded node. Take one with
/// [`Value::symbol`](crate::Value::symbol) before sealing the tape;
/// read through it with [`Parameters::of`](crate::Parameters::of),
/// [`Run::of`](crate::Run::of), or [`Field::of`](crate::Field::of),
/// and turn it back into a proxy with
/// [`Tape::resolve`](crate::Tape::resolve) when a network reopens.
///
/// The origin participates in equality and hashing, so equally
/// positioned nodes from unrelated graphs do not compare equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub(crate) origin: Origin,
    pub(crate) id: ValueId,
}
