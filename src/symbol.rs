use static_assertions::assert_impl_all;

use super::{Lineage, ValueId};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Symbol: Send, Sync, Copy);

/// A detached, `Copy` name of a value, resolved against network
/// generations.
///
/// It is the identity of a value across time: while `Value` is a
/// generation-bound view, a `Symbol` carries no borrow of any network, so
/// it can be stored anywhere and outlive any particular generation — a
/// training loop keeps the symbols of its loss and parameters while the
/// network variable is reassigned to updated generations. Each generation
/// acts as an environment: `Network::resolve` looks the symbol up in it
/// and returns that generation's `Value`. The symbol carries its lineage,
/// so kinship is checked at resolution: positions are stable across forks
/// and updates, a symbol resolves to the same node in every related
/// generation, and resolving it in an unrelated network panics instead of
/// silently misbinding. Lineage also takes part in equality and hashing,
/// so symbols from unrelated networks never collide as map keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub(crate) lineage: Lineage,
    pub(crate) id: ValueId,
}
