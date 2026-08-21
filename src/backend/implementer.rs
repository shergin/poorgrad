use super::backend::BackendUnavailable;
use super::coverage::{Cell, Dispatch};
use super::formula::Formula;

/// What it means to be an implementer: the questions every backend
/// answers for itself, in its own module, in every build.
///
/// Each [`Backend`](super::Backend) variant has a descriptor — a
/// unit struct implementing this trait — that is always compiled,
/// while its kernels sit behind the feature `cfg` in a `kernels`
/// submodule. The enum stays the public axis and dispatches to the
/// descriptors by plain match, so the contract is monomorphized
/// away: no trait object exists anywhere on the path. The methods
/// are associated functions on purpose — a descriptor has no state,
/// only answers.
pub(crate) trait Implementer {
    /// How this implementer's kernels are reached.
    const DISPATCH: Dispatch;

    /// This implementer's row of the coverage matrix: one cell per
    /// formula, declaring the certification bar and the forwarding
    /// precisions its kernels accept.
    ///
    /// The match must stay exhaustive — that is the compile-time
    /// gate: a new formula cannot compile until every implementer
    /// has decided its cell.
    fn coverage(formula: Formula) -> Cell;

    /// Whether this implementer is in this build at all: build
    /// facts only, no lazy setup, no device probe. Elections key on
    /// this answer, never on `status`.
    fn compiled() -> bool;

    /// Whether this implementer would accept work in this build on
    /// this machine, forcing its lazy setup if it has one.
    fn status() -> Result<(), BackendUnavailable>;
}
