use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use static_assertions::assert_impl_all;

use crate::engine::{Symbol, ValueId};

use super::Witness;

// The token is what lets symbols and fields cross threads detached from
// any network, so its thread-safety and `Copy` are load-bearing.
assert_impl_all!(Origin: Send, Sync, Copy);

/// Mints a fresh process-globally unique id for [`Origin`] and [`Branch`].
///
/// `Relaxed` suffices: only uniqueness matters, and the identity
/// reaches other threads through the structure it identifies.
fn next_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// An opaque token identifying a family of related tapes.
///
/// Every tape mints its origin from a process-global counter at
/// creation, and forks and updates carry it forward, so two tapes share
/// an origin exactly when they descend from a common construction;
/// same-origin checks are plain equality. Being a `Copy` integer rather
/// than a reference-counted token, it rides inside every `Symbol`
/// without costing `Copy`, and creating fields and evaluations never
/// touches an atomic counter. Within an origin, positions are
/// attributed to branches: divergent forks stop sharing identity
/// exactly where their recordings part ways.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Origin(u64);

impl Origin {
    /// Mints a fresh origin identity.
    pub(super) fn new() -> Self {
        Self(next_id())
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
        Self(next_id())
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

/// The tape's relationship to its chain's tip branch: the writer
/// protocol deciding which tape may extend the tip branch.
///
/// The shared token is the only synchronization between sibling tapes;
/// the `Tip` value itself is part of the tape's locked state, and every
/// method assumes the tape lock is held.
#[derive(Debug)]
enum Tip {
    /// This tape alone may extend the tip branch.
    Owned,
    /// The tip is shared with sibling tapes after a fork or an update:
    /// the first sibling to record claims the token and continues the
    /// branch, every other sibling mints its own branch on its first
    /// recording. This keeps linear histories from growing the chain.
    Contended(Arc<AtomicBool>),
}

impl Tip {
    /// Secures the right to record at the current tip before a push.
    ///
    /// An owned tip records freely. A contended tip races its siblings
    /// on the shared token: the winner continues the tip branch, a
    /// loser mints a fresh branch starting at `length`, its own tape's
    /// length. Either way the tip is owned afterwards. The winner never
    /// touches `chain`: the copy-on-write clone of a shared chain
    /// happens only on the losing path. `AcqRel` documents the token as
    /// a synchronization point between sibling tapes; the data it
    /// guards is only the branch continuation decision.
    fn claim(&mut self, chain: &mut Arc<Vec<Segment>>, length: usize) {
        let Tip::Contended(token) = &self else {
            return;
        };
        let won = token
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok();
        if !won {
            Arc::make_mut(chain).push(Segment {
                branch: Branch::new(),
                start: length,
            });
        }
        *self = Tip::Owned;
    }

    /// Prepares the tip for duplication and returns the copy's tip.
    ///
    /// Both sides must re-win the right to extend the tip branch, so an
    /// owned tip becomes contended on a fresh token shared with the
    /// copy. An already contended tip hands the copy the same token:
    /// every tape sharing an unextended tip contends on one token, so
    /// exactly one of them ever continues the branch.
    fn share(&mut self) -> Tip {
        match &self {
            Tip::Contended(token) => Tip::Contended(Arc::clone(token)),
            Tip::Owned => {
                let token = Arc::new(AtomicBool::new(false));
                *self = Tip::Contended(Arc::clone(&token));
                Tip::Contended(token)
            }
        }
    }
}

/// Why a symbol fails to bind against an origin and chain: the probing
/// counterpart of the resolution panics. Callers map each reason to
/// their own site-specific message, so diagnostics stay as precise as
/// the open-coded checks this replaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Misbinding {
    /// The symbol belongs to an unrelated tape family.
    ForeignOrigin,
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
    origin: Origin,
    chain: &[Segment],
    length: usize,
    symbol: Symbol,
) -> Result<ValueId, Misbinding> {
    if symbol.origin != origin {
        return Err(Misbinding::ForeignOrigin);
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

/// Live graph identity under the tape lock: origin, branch chain, and tip.
///
/// It is the writer-side package for structural identity. Detached
/// carriers never hold one; they take a [`Witness`] via
/// [`Identity::witness`] — a frozen origin plus chain, without the tip
/// protocol. Tip claim and share assume the tape mutex is held.
#[derive(Debug)]
pub(crate) struct Identity {
    origin: Origin,
    chain: Arc<Vec<Segment>>,
    tip: Tip,
}

impl Identity {
    /// Creates a fresh identity: new origin, root branch at index zero,
    /// owned tip.
    pub(crate) fn new() -> Self {
        Self {
            origin: Origin::new(),
            chain: Arc::new(vec![Segment {
                branch: Branch::new(),
                start: 0,
            }]),
            tip: Tip::Owned,
        }
    }

    /// Returns the origin token of this identity.
    pub(crate) fn origin(&self) -> Origin {
        self.origin
    }

    /// Secures the right to record at the current tip before a push.
    ///
    /// Wires this identity's chain and `length` (the tape's live node
    /// count) into the tip claim protocol.
    pub(crate) fn claim(&mut self, length: usize) {
        self.tip.claim(&mut self.chain, length);
    }

    /// Prepares this identity for a fork or generation step and returns
    /// the sibling's identity: shared origin and chain, contended tip.
    pub(crate) fn share(&mut self) -> Identity {
        Identity {
            origin: self.origin,
            chain: Arc::clone(&self.chain),
            tip: self.tip.share(),
        }
    }

    /// Returns a read-only witness of this identity's origin and chain.
    pub(crate) fn witness(&self) -> Witness {
        Witness::new(self.origin, Arc::clone(&self.chain))
    }

    /// Returns the branch that owns `index` on the live chain.
    ///
    /// # Panics
    /// Panics if `index` is out of range for a non-empty interpretation
    /// of the chain's covered positions; callers assert coverage first.
    pub(crate) fn branch_of(&self, index: usize) -> Branch {
        self.chain
            .iter()
            .rev()
            .find(|segment| segment.start <= index)
            .expect("the root segment starts at zero")
            .branch
    }

    /// Probes for the node `symbol` names within the first `length`
    /// covered positions against the live chain.
    pub(crate) fn probe(&self, symbol: Symbol, length: usize) -> Result<ValueId, Misbinding> {
        chain_probe(self.origin, &self.chain, length, symbol)
    }

    /// Returns whether `witness` names this identity's origin and
    /// attributes `[0, length)` to the same branches as the live chain.
    pub(crate) fn agrees_with(&self, witness: &Witness, length: usize) -> bool {
        self.witness().agrees_with(witness, length)
    }

    /// Returns whether `witness` names this identity's origin.
    pub(crate) fn same_origin(&self, witness: &Witness) -> bool {
        self.origin == witness.origin()
    }
}

#[cfg(test)]
#[path = "tests/tip_tests.rs"]
mod tip_tests;
