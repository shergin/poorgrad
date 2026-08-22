use super::backend::BackendUnavailable;
use super::coverage::{Coverage, Dispatch};
use super::formula::Formula;

/// What it means to be a backend: the questions every backend
/// answers for itself, in its own module, in every build.
///
/// Each [`Backend`](super::Backend) variant has a manifest — a unit
/// struct implementing this trait — that is always compiled and
/// declares what the backend carries, while the kernels themselves
/// sit behind the feature `cfg` in a `kernels` submodule. The enum
/// stays the public axis and dispatches to the manifests by plain
/// match, so the contract is monomorphized away: no trait object
/// exists anywhere on the path. The methods are associated
/// functions on purpose — a manifest has no state, only answers.
pub(crate) trait Manifest {
    /// How this backend's kernels are reached.
    const DISPATCH: Dispatch;

    /// This backend's coverage of one formula: the certified fidelity
    /// and the forwarding precisions its kernel accepts, or
    /// `Absent`.
    ///
    /// The match must stay exhaustive — that is the compile-time
    /// gate: a new formula cannot compile until every backend has
    /// declared its coverage.
    fn coverage(formula: Formula) -> Coverage;

    /// Whether this backend is in this build at all: build facts
    /// only, no lazy setup, no device probe. Elections key on this
    /// answer, never on `status`.
    fn compiled() -> bool;

    /// Whether this backend would accept work in this build on
    /// this machine, forcing its lazy setup if it has one.
    fn status() -> Result<(), BackendUnavailable>;
}
