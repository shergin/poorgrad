use static_assertions::assert_impl_all;

use super::{Branch, Lineage, ValueId};

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
/// and returns that generation's `Value`. The symbol carries its lineage
/// and the branch that owned its position when it was minted, so kinship
/// is checked at resolution: positions are stable across updates and
/// non-divergent forks, a symbol resolves to the same node in every
/// related generation, and resolving it in an unrelated network — or in
/// a fork that diverged before the symbol was minted — panics instead of
/// silently misbinding. Lineage and branch take part in equality and
/// hashing, so symbols from unrelated networks never collide as map
/// keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub(crate) lineage: Lineage,
    pub(crate) branch: Branch,
    pub(crate) id: ValueId,
}
