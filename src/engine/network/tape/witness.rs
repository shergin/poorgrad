use std::sync::Arc;

use crate::engine::{Symbol, ValueId};

use super::{Misbinding, Origin, Segment, chain_probe, chains_agree};

/// A read-only witness of graph identity: the origin naming the tape
/// family and the branch chain mapping positions to branches.
///
/// Detached carriers — [`Field`](crate::Field), [`Plan`](crate::Plan),
/// tape snapshots — hold one instead of borrowing a tape, and answer
/// two questions through it: does another carrier agree on the
/// structural map over a shared prefix, and does a symbol name a node
/// on that map? A witness never panics and carries no length: coverage
/// discipline — how many nodes a carrier holds, and whether equality or
/// containment is required — belongs to each call site, along with its
/// panic messages. It is deliberately not `PartialEq`: whole-chain
/// equality is not prefix agreement, so the only offered comparison is
/// [`Witness::agrees_with`].
///
/// The live counterpart under the tape lock is [`Identity`](super::Identity):
/// origin, chain, and tip together. A witness is what `Identity` exports.
#[derive(Debug, Clone)]
pub(crate) struct Witness {
    origin: Origin,
    chain: Arc<Vec<Segment>>,
}

impl Witness {
    pub(crate) fn new(origin: Origin, chain: Arc<Vec<Segment>>) -> Self {
        Self { origin, chain }
    }

    /// Returns the token of the origin this witness belongs to.
    ///
    /// Module-internal: outside the tape module the witness is opaque,
    /// and callers go through the check methods.
    pub(super) fn origin(&self) -> Origin {
        self.origin
    }

    /// Returns whether both witnesses name the same tape family.
    pub(crate) fn same_origin(&self, other: &Witness) -> bool {
        self.origin == other.origin
    }

    /// Returns whether `other` belongs to the same family and
    /// attributes `[0, length)` to the same branches.
    pub(crate) fn agrees_with(&self, other: &Witness, length: usize) -> bool {
        self.same_origin(other) && chains_agree(&self.chain, &other.chain, length)
    }

    /// Probes for the node `symbol` names within the first `length`
    /// covered positions, answering with the reason when it fails.
    pub(crate) fn probe(&self, symbol: Symbol, length: usize) -> Result<ValueId, Misbinding> {
        chain_probe(self.origin, &self.chain, length, symbol)
    }
}

#[cfg(test)]
#[path = "tests/witness_tests.rs"]
mod tests;
