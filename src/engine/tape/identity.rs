use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use static_assertions::assert_impl_all;

use crate::engine::{Symbol, ValueId};

// The token is what lets symbols and fields cross threads detached from
// any network, so its thread-safety and `Copy` are load-bearing.
assert_impl_all!(Lineage: Send, Sync, Copy);

/// Mints a fresh process-globally unique identity.
///
/// `Relaxed` suffices: only uniqueness matters, and the identity
/// reaches other threads through the structure it identifies.
fn next_identity() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// An opaque token identifying a family of related tapes.
///
/// Every tape mints its identity from a process-global counter at
/// creation, and forks and updates carry it forward, so two tapes share a
/// lineage exactly when they descend from a common origin; kinship is
/// plain equality. Being a `Copy` integer rather than a reference-counted
/// token, it rides inside every `Symbol` without costing `Copy`, and
/// creating fields and evaluations never touches an atomic counter.
/// Within a lineage, positions are attributed to branches: divergent
/// forks stop sharing identity exactly where their recordings part ways.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Lineage(u64);

impl Lineage {
    /// Mints a fresh lineage identity.
    pub(super) fn new() -> Self {
        Self(next_identity())
    }
}

/// A globally unique identity for one contiguous run of recordings.
///
/// A branch names an index range of a tape: symbols carry the branch
/// that owned their position when they were minted, so a divergent
/// fork which fills the same positions with different nodes under a
/// different branch rejects them instead of misbinding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Branch(u64);

impl Branch {
    /// Mints a fresh branch identity.
    pub(super) fn new() -> Self {
        Self(next_identity())
    }
}

/// One contiguous index range of a tape attributed to a branch.
///
/// The range starts at `start` and ends where the next segment starts,
/// or at the tape's current length for the tip segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Segment {
    pub(super) branch: Branch,
    pub(super) start: usize,
}

/// Returns whether two chains attribute the index range `[0, length)`
/// to the same branches.
///
/// Segments starting at or beyond `length` are ignored: they describe
/// nodes outside the compared range, so a longer tape stays kin with a
/// field taken before it grew.
pub(crate) fn chains_agree(
    left: &Arc<Vec<Segment>>,
    right: &Arc<Vec<Segment>>,
    length: usize,
) -> bool {
    if Arc::ptr_eq(left, right) {
        return true;
    }
    let trimmed = |chain: &[Segment]| {
        chain
            .iter()
            .take_while(|segment| segment.start < length)
            .count()
    };
    left[..trimmed(left)] == right[..trimmed(right)]
}

/// Why a symbol fails to bind against a lineage and chain: the probing
/// counterpart of the resolution panics. Callers map each reason to
/// their own site-specific message, so diagnostics stay as precise as
/// the open-coded checks this replaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Misbinding {
    /// The symbol belongs to an unrelated tape family.
    ForeignLineage,
    /// The symbol's branch parted ways with this chain: it is absent
    /// entirely, or its position is owned by another branch.
    DivergentBranch,
    /// The symbol names a position beyond the covered prefix.
    OutOfCoverage,
}

/// Locates the node `symbol` names on a chain covering `[0, length)`:
/// the single spelling of symbol resolution behind tape-side lookups
/// and detached field reads, answerable with or without a tape borrow.
///
/// The branch is checked against the whole chain before coverage, so a
/// symbol from a fork this chain never carried reports divergence even
/// when its position is also out of coverage — divergence is the
/// sharper diagnosis.
pub(crate) fn chain_probe(
    lineage: Lineage,
    chain: &[Segment],
    length: usize,
    symbol: Symbol,
) -> Result<ValueId, Misbinding> {
    if symbol.lineage != lineage {
        return Err(Misbinding::ForeignLineage);
    }
    if !chain.iter().any(|segment| segment.branch == symbol.branch) {
        return Err(Misbinding::DivergentBranch);
    }
    let index = symbol.id.index();
    if index >= length {
        return Err(Misbinding::OutOfCoverage);
    }
    let owner = chain
        .iter()
        .take_while(|segment| segment.start <= index)
        .last()
        .expect("the root segment starts at zero");
    if owner.branch != symbol.branch {
        return Err(Misbinding::DivergentBranch);
    }
    Ok(symbol.id)
}

#[cfg(test)]
#[path = "tests/identity_tests.rs"]
mod tests;
