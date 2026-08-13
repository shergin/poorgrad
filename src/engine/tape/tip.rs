use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{Branch, Segment};

/// The tape's relationship to its chain's tip branch: the writer
/// protocol deciding which tape may extend the tip branch.
///
/// The shared token is the only synchronization between sibling tapes;
/// the `Tip` value itself is part of the tape's locked state, and every
/// method assumes the tape lock is held.
#[derive(Debug)]
pub(super) enum Tip {
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
    pub(super) fn claim(&mut self, chain: &mut Arc<Vec<Segment>>, length: usize) {
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
    pub(super) fn share(&mut self) -> Tip {
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

#[cfg(test)]
#[path = "tests/tip_tests.rs"]
mod tests;
