use static_assertions::assert_impl_all;

use super::{Branch, Origin, ValueId};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Symbol: Send, Sync, Copy);

/// A detached, `Copy` identifier for a value in compatible network generations.
///
/// Unlike [`Value`](crate::Value), a symbol carries no network borrow and can
/// outlive the generation that produced it.
/// [`Network::resolve`](crate::Network::resolve) turns it back into a
/// generation-bound value, which is useful when a training loop repeatedly
/// replaces a network with [`Network::update`](crate::Network::update).
///
/// A symbol records its graph origin, branch, and node position. Resolution
/// succeeds only when that node exists on a compatible branch; unrelated
/// networks, divergent forks, and generations that do not contain the node are
/// rejected. This provenance also participates in equality and hashing, so
/// equally positioned nodes from unrelated graphs do not compare equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub(crate) origin: Origin,
    pub(crate) branch: Branch,
    pub(crate) id: ValueId,
}
