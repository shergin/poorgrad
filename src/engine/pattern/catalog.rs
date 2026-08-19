use smallvec::SmallVec;

use crate::Differentiable;

use super::batch_norm;
use super::pattern::Pattern;
use super::reduce_window;
use super::view::View;
use super::window;

/// One matcher result, not yet claimed.
///
/// Whether the entry also fuses at home is variant policy
/// ([`Pattern::fused`]), not a field here: a per-candidate flag would
/// make "matched, home false" look representable when [`Catalog`]
/// cannot hold it.
pub(crate) struct Candidate {
    pub(crate) pattern: Pattern,
    /// Unnamed interiors: skipped at emit by every match, and at home
    /// too when the match fuses.
    pub(crate) interiors: SmallVec<[usize; 8]>,
    /// Extra results of the raise that may be readable. Claimed so
    /// later matchers cannot take them, and marked `emit_interior` so
    /// their primitive lowers are skipped; never marked
    /// `home_interior`, so runs still execute them.
    pub(crate) named: SmallVec<[usize; 4]>,
}

/// The compiled pattern column of one plan: the pattern rooted at each
/// node, if any, and the two interior skip-masks its consumers read.
///
/// The masks diverge by design: every match skips its interiors (and
/// named results) at emit, while only a home-fusing match skips its
/// unnamed interiors at home — the home mask is a subset of the emit
/// mask by construction.
#[derive(Debug, Clone)]
pub(crate) struct Catalog {
    at: Vec<Option<Pattern>>,
    home_interior: Vec<bool>,
    emit_interior: Vec<bool>,
}

impl Catalog {
    /// Runs every matcher over `view` and claims nodes first-wins.
    /// `fuse` is the memory-posture gate: a homing pattern is stored
    /// only when it is true, which is exactly a forward-only request —
    /// fusing engine-backward would leave the reverse scan nothing to
    /// read. Raise-only matchers run un-gated, storing on every plan
    /// the matcher accepts.
    ///
    /// Matcher order in this body is the first priority axis; within
    /// one matcher, `collect_one` scans in recording order. Adding a
    /// pattern is one call here, in its documented overlap position.
    pub(crate) fn collect<Data: Differentiable>(view: &View<Data>, fuse: bool) -> Self {
        let length = view.len();
        let mut catalog = Self {
            at: vec![None; length],
            home_interior: vec![false; length],
            emit_interior: vec![false; length],
        };
        let mut claimed = vec![false; length];

        if fuse {
            collect_one(view, &mut catalog, &mut claimed, window::match_at);
        }
        collect_one(view, &mut catalog, &mut claimed, reduce_window::match_at);
        // Training before inference: the richer, more specific ending
        // claims first, so a training recording never raises as
        // inference-over-computed-statistics.
        collect_one(view, &mut catalog, &mut claimed, batch_norm::match_training);
        collect_one(
            view,
            &mut catalog,
            &mut claimed,
            batch_norm::match_inference,
        );

        catalog
    }

    /// Returns the pattern rooted at `index`, if one matched. This is
    /// the only read consumers are allowed: home runs and emission
    /// consume stored entries and never rematch.
    pub(crate) fn at(&self, index: usize) -> Option<&Pattern> {
        self.at[index].as_ref()
    }

    pub(crate) fn home_interior(&self, index: usize) -> bool {
        self.home_interior[index]
    }

    pub(crate) fn emit_interior(&self, index: usize) -> bool {
        self.emit_interior[index]
    }

    /// Returns how many home-fusing groups the plan matched.
    pub(crate) fn home_groups(&self) -> usize {
        self.at
            .iter()
            .flatten()
            .filter(|pattern| pattern.fused().is_some())
            .count()
    }

    /// Pins `last_consumer` so the reads of a home-fusing match
    /// outlive the skipped chain. Raise-only patterns pin nothing: their
    /// chains actually run, so ordinary last-consumer is correct.
    pub(crate) fn pin_liveness(&self, last_consumer: &mut [Option<usize>]) {
        for (index, pattern) in self.at.iter().enumerate() {
            let Some(group) = pattern.as_ref().and_then(Pattern::fused) else {
                continue;
            };
            for slot in group.reads() {
                let latest = last_consumer[slot].unwrap_or(0).max(index);
                last_consumer[slot] = Some(latest);
            }
        }
    }
}

/// Runs `matcher` over every wanted, unclaimed node in recording
/// order and stores the accepted candidates into `catalog`.
///
/// A candidate is rejected wholesale if it is not closed or any of
/// its root, interiors, or named results is already claimed. Extra
/// reads are not claimed: two patterns may share an input.
fn collect_one<Data: Differentiable>(
    view: &View<Data>,
    catalog: &mut Catalog,
    claimed: &mut [bool],
    matcher: fn(usize, &View<Data>) -> Option<Candidate>,
) {
    for index in 0..view.len() {
        if !view.wanted(index) || claimed[index] {
            continue;
        }
        let Some(candidate) = matcher(index, view) else {
            continue;
        };
        if !view.closed(index, &candidate.interiors, &candidate.named) {
            continue;
        }
        if candidate
            .interiors
            .iter()
            .chain(candidate.named.iter())
            .any(|&node| claimed[node])
        {
            continue;
        }
        let homes = candidate.pattern.fused().is_some();
        claimed[index] = true;
        for &node in candidate.interiors.iter().chain(candidate.named.iter()) {
            claimed[node] = true;
        }
        for &node in &candidate.interiors {
            catalog.emit_interior[node] = true;
            if homes {
                catalog.home_interior[node] = true;
            }
        }
        // Named results skip their primitive emit but still execute at
        // home.
        for &node in &candidate.named {
            catalog.emit_interior[node] = true;
        }
        catalog.at[index] = Some(candidate.pattern);
    }
}

#[cfg(test)]
#[path = "tests/catalog_tests.rs"]
mod tests;
