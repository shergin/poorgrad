use std::sync::Arc;

use crate::engine::{Symbol, ValueId};

use super::{Lineage, Misbinding, Segment, chain_probe, chains_agree};

/// A read-only witness of graph identity: the lineage naming the tape
/// family and the branch chain mapping positions to branches.
///
/// Detached carriers — [`Field`](crate::Field), [`Plan`](crate::Plan),
/// tape snapshots — hold one instead of borrowing a tape, and answer
/// two questions through it: does another carrier agree on the
/// structural map over a shared prefix, and does a symbol name a node
/// on that map? A kinship never panics and carries no length: coverage
/// discipline — how many nodes a carrier holds, and whether equality or
/// containment is required — belongs to each call site, along with its
/// panic messages. It is deliberately not `PartialEq`: whole-chain
/// equality is not prefix agreement, so the only offered comparison is
/// [`Kinship::agrees_with`].
#[derive(Debug, Clone)]
pub(crate) struct Kinship {
    lineage: Lineage,
    chain: Arc<Vec<Segment>>,
}

impl Kinship {
    pub(crate) fn new(lineage: Lineage, chain: Arc<Vec<Segment>>) -> Self {
        Self { lineage, chain }
    }

    /// Returns the token of the lineage this witness belongs to.
    ///
    /// Module-internal: outside the tape module the witness is opaque,
    /// and callers go through the check methods.
    pub(super) fn lineage(&self) -> Lineage {
        self.lineage
    }

    /// Returns the branch chain this witness carries.
    ///
    /// Module-internal, like [`Kinship::lineage`]: the chain never
    /// leaves the tape module.
    pub(super) fn chain(&self) -> &Arc<Vec<Segment>> {
        &self.chain
    }

    /// Returns whether both witnesses name the same tape family.
    pub(crate) fn is_family(&self, other: &Kinship) -> bool {
        self.lineage == other.lineage
    }

    /// Returns whether `other` belongs to the same family and
    /// attributes `[0, length)` to the same branches.
    pub(crate) fn agrees_with(&self, other: &Kinship, length: usize) -> bool {
        self.is_family(other) && chains_agree(&self.chain, &other.chain, length)
    }

    /// Probes for the node `symbol` names within the first `length`
    /// covered positions, answering with the reason when it fails.
    pub(crate) fn probe(&self, symbol: Symbol, length: usize) -> Result<ValueId, Misbinding> {
        chain_probe(self.lineage, &self.chain, length, symbol)
    }
}

#[cfg(test)]
#[path = "tests/kinship_tests.rs"]
mod tests;
