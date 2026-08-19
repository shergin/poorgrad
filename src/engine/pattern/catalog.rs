use smallvec::SmallVec;

use crate::{Differentiable, Tensorial};

use super::batch_norm;
use super::pattern::Pattern;
use super::reduce_window;
use super::view::View;
use super::window;

/// One matcher result, not yet claimed.
///
/// Home and emit actions are variant policy ([`Pattern::homes`] /
/// [`Pattern::raises`]), not fields here: storing flags on a
/// candidate and then dropping them would make "matched, home false"
/// look representable when [`Catalog`] cannot hold it.
pub(crate) struct Candidate {
    pub(crate) pattern: Pattern,
    /// Unnamed interiors. Home-fusing matches skip them at home;
    /// raising matches skip them at emit.
    pub(crate) interiors: SmallVec<[usize; 8]>,
    /// Extra results of a raise that may be readable. Claimed so
    /// later matchers cannot take them, and marked `emit_interior` so
    /// their primitive lowers are skipped; never marked
    /// `home_interior`, so a raise-only run still executes them.
    pub(crate) named: SmallVec<[usize; 4]>,
}

/// Whether this plan may apply home-fusing actions.
///
/// Raise-only matchers ignore it. A homing motif is stored only when
/// `fuse` is true, which is exactly a forward-only request: fusing
/// engine-backward would leave the reverse scan nothing to read.
#[derive(Clone, Copy)]
pub(crate) struct PostureGate {
    pub(crate) fuse: bool,
}

impl PostureGate {
    pub(crate) fn from_backward(backward: bool) -> Self {
        Self { fuse: !backward }
    }
}

/// The compiled motif column of one plan: the pattern rooted at each
/// node, if any, and the two interior skip-masks its consumers read.
///
/// The home and emit masks are allowed to diverge: a raise-only motif
/// skips its interiors (and named results) at emit while executing
/// all of them at home.
#[derive(Debug, Clone)]
pub(crate) struct Catalog {
    at: Vec<Option<Pattern>>,
    home_interior: Vec<bool>,
    emit_interior: Vec<bool>,
}

impl Catalog {
    /// Runs every matcher over `view` and claims nodes first-wins.
    ///
    /// Matcher order in this body is the first priority axis; within
    /// one matcher, `collect_one` scans in recording order. Adding a
    /// motif is one call here, in its documented overlap position.
    pub(crate) fn collect<Data: Differentiable>(view: &View<Data>, gate: PostureGate) -> Self {
        let length = view.len();
        let mut catalog = Self {
            at: vec![None; length],
            home_interior: vec![false; length],
            emit_interior: vec![false; length],
        };
        let mut claimed = vec![false; length];

        // Homing matchers live inside the posture gate; raise-only
        // matchers run un-gated below it, storing on every plan the
        // matcher accepts — including engine-backward, whose home run
        // still executes every node.
        if gate.fuse {
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

    pub(crate) fn at(&self, index: usize) -> Option<&Pattern> {
        self.at[index].as_ref()
    }

    pub(crate) fn home_interior(&self, index: usize) -> bool {
        self.home_interior[index]
    }

    pub(crate) fn home_interiors(&self) -> &[bool] {
        &self.home_interior
    }

    pub(crate) fn emit_interiors(&self) -> &[bool] {
        &self.emit_interior
    }

    /// Returns how many home-fusing groups the plan matched.
    pub(crate) fn home_groups(&self) -> usize {
        self.at
            .iter()
            .flatten()
            .filter(|pattern| pattern.homes())
            .count()
    }

    /// Pins `last_consumer` so the extra reads of a home-fusing match
    /// outlive the skipped chain. Raise-only motifs pin nothing: their
    /// chains actually run, so ordinary last-consumer is correct.
    pub(crate) fn pin_liveness(&self, last_consumer: &mut [Option<usize>]) {
        for (index, pattern) in self.at.iter().enumerate() {
            let Some(pattern) = pattern else {
                continue;
            };
            if !pattern.homes() {
                continue;
            }
            for slot in pattern.extra_reads() {
                let latest = last_consumer[slot].unwrap_or(0).max(index);
                last_consumer[slot] = Some(latest);
            }
        }
    }

    /// Returns the payload of a home action at `index`, if this node
    /// is a home-fusing root. The arms are the truth of
    /// [`Pattern::homes`]: a raise-only variant answers `None`, so the
    /// node runs its recorded rule.
    pub(crate) fn home<Data: Tensorial>(&self, index: usize, values: &[Data]) -> Option<Data> {
        match self.at[index].as_ref()? {
            Pattern::WindowProduct(group) => Some(values[group.source].windowed_product(
                &values[group.kernel],
                group.kernel_height,
                group.kernel_width,
                group.stride,
                group.padding,
            )),
            Pattern::ReduceWindow(_)
            | Pattern::BatchNormTraining(_)
            | Pattern::BatchNormInference(_) => None,
        }
    }
}

/// Runs `matcher` over every wanted, unclaimed node in recording
/// order and stores the accepted candidates into `catalog`.
///
/// A candidate is rejected wholesale if it is not closed or any of
/// its root, interiors, or named results is already claimed. Extra
/// reads are not claimed: two motifs may share an input.
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
        let homes = candidate.pattern.homes();
        let raises = candidate.pattern.raises();
        claimed[index] = true;
        for &node in candidate.interiors.iter().chain(candidate.named.iter()) {
            claimed[node] = true;
        }
        for &node in &candidate.interiors {
            if homes {
                catalog.home_interior[node] = true;
            }
            if raises {
                catalog.emit_interior[node] = true;
            }
        }
        // Named results of a raise skip their primitive emit but still
        // execute at home.
        if raises {
            for &node in &candidate.named {
                catalog.emit_interior[node] = true;
            }
        }
        catalog.at[index] = Some(candidate.pattern);
    }
}

#[cfg(test)]
#[path = "tests/catalog_tests.rs"]
mod tests;
