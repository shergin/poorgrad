use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use static_assertions::assert_impl_all;

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

/// Returns whether `chain` attributes position `index` to `branch`
/// within the first `length` covered positions: the detached twin of
/// a tape-side branch check, answerable by a [`Field`](crate::Field)
/// that borrows no tape.
pub(crate) fn chain_attributes(
    chain: &[Segment],
    branch: Branch,
    index: usize,
    length: usize,
) -> bool {
    if index >= length {
        return false;
    }
    let owner = chain
        .iter()
        .take_while(|segment| segment.start <= index)
        .last();
    matches!(owner, Some(segment) if segment.branch == branch)
}

#[cfg(test)]
#[path = "tests/identity_tests.rs"]
mod tests;
