use smallvec::SmallVec;

use crate::Differentiable;

use super::batch_norm;
use super::pattern::Pattern;
use super::reduce_window;
use super::view::View;
use super::window;

/// One discovered match: a pattern, the node it roots at, and the
/// claim set an electing consumer takes over.
///
/// A candidate records structural truth only. Whether anyone acts on
/// it is election policy: every consumer elects its own [`Catalog`]
/// from the same pool under its own repertoire, so a per-candidate
/// action flag would make unrepresentable states look representable.
///
/// [`Catalog`]: super::catalog::Catalog
#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub(crate) pattern: Pattern,
    /// The node the pattern roots at. An electing consumer's action
    /// produces this value (and the named results) in the recorded
    /// chain's place.
    pub(crate) root: usize,
    /// Unnamed interiors: skipped by any consumer that elects the
    /// entry, never readable.
    pub(crate) interiors: SmallVec<[usize; 8]>,
    /// Extra results the action also produces; unlike interiors they
    /// may be readable. Skipped alongside the interiors by the
    /// electing consumer, which names them at the root.
    pub(crate) named: SmallVec<[usize; 4]>,
}

/// The plan's discovered pattern pool: every closed candidate over the
/// frozen columns, in priority order — matcher order first, recording
/// order within one matcher.
///
/// Discovery is consumer-independent and posture-blind: it depends
/// only on structure, `wanted`, and `readable`, so it runs once at
/// compile time and every consumer elects its own catalog from the
/// same pool. Candidates may overlap; election resolves the claims.
#[derive(Debug, Clone)]
pub(crate) struct Candidates {
    /// The plan prefix length the pool was discovered over: the column
    /// length of every elected catalog.
    length: usize,
    all: Vec<Candidate>,
}

impl Candidates {
    /// Runs every matcher over `view` and pools all closed candidates.
    ///
    /// Matcher order in this body is the first priority axis; within
    /// one matcher, nodes are scanned in recording order. Adding a
    /// pattern is one call here, in its documented overlap position.
    pub(crate) fn discover<Data: Differentiable>(view: &View<Data>) -> Self {
        let mut all = Vec::new();
        discover_one(view, &mut all, window::match_at);
        discover_one(view, &mut all, reduce_window::match_at);
        // Training before inference: the richer, more specific ending
        // takes priority, so a training recording never raises as
        // inference-over-computed-statistics.
        discover_one(view, &mut all, batch_norm::match_training);
        discover_one(view, &mut all, batch_norm::match_inference);
        Self {
            length: view.len(),
            all,
        }
    }

    pub(crate) fn length(&self) -> usize {
        self.length
    }

    /// Returns the pool in priority order, the order every election
    /// claims in.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Candidate> {
        self.all.iter()
    }
}

/// Runs `matcher` over every wanted node in recording order and pools
/// the candidates that pass the closure check. No claiming happens
/// here: two candidates may overlap, and each consumer's election
/// resolves the conflict under its own repertoire.
fn discover_one<Data: Differentiable>(
    view: &View<Data>,
    all: &mut Vec<Candidate>,
    matcher: fn(usize, &View<Data>) -> Option<Candidate>,
) {
    for index in 0..view.len() {
        if !view.wanted(index) {
            continue;
        }
        let Some(candidate) = matcher(index, view) else {
            continue;
        };
        if !view.closed(index, &candidate.interiors, &candidate.named) {
            continue;
        }
        all.push(candidate);
    }
}
