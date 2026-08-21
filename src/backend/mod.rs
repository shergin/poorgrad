//! The acceleration stack's one relation: an implementer may serve
//! a named formula, under a bar, against the oracle.
//!
//! Coverage declares *may* — the [`Backend::coverage`] matrix over
//! [`Formula`] and [`Precision`], with a certification [`Bar`] per
//! cell. The offer decides *will* — [`offered`] walks a job's
//! declared chain, admitting each member by bar-meets-posture, and
//! every member may still decline (thresholds, stride mappings,
//! device presence). The oracle defines *is* — the reference paths
//! are the substrate every decline falls to, in every build; on a
//! build without backend features every offer answers `None`, the
//! seam's fixed point, not dead code. Each job type carries its
//! formula and precision through the crate-internal `Job` contract,
//! so a job can only walk its own chain. Everything here exists in
//! every build: interrogating the stack never needs a `cfg`.

#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[allow(unsafe_code)]
mod accelerate;
#[allow(clippy::module_inception)]
mod backend;
mod coverage;
#[cfg(all(feature = "cuda", target_os = "linux"))]
#[allow(unsafe_code)]
mod cuda;
mod formula;
mod job;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(unsafe_code)]
mod metal;
mod numerics;
// Safe stride classification shared by the BLAS-shaped backends;
// compiled exactly where one of them is.
#[cfg(any(
    all(feature = "accelerate", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
mod operand;
// The one arm with no `target_os`: the simd backend is real on
// every platform.
#[cfg(feature = "simd")]
#[allow(unsafe_code)]
mod simd;

pub use backend::{Backend, BackendUnavailable};
pub use coverage::{Bar, Cell, Dispatch, Precisions};
pub use formula::{Formula, Precision};
pub use numerics::Numerics;

pub(crate) use job::MapTask;
pub(crate) use numerics::NumericsScope;

use job::Job;

/// It offers a job down its formula's chain, answering `None` when
/// every member declines: the chain's one entry point, monomorphized
/// per job type.
///
/// Admission is the bar rule, not a posture special case: a chain
/// member serves only if its coverage cell's bar meets the bar the
/// current posture demands, so `Exact` excludes every envelope
/// kernel and would admit a bit-certified one.
pub(crate) fn offered<T: Job>(task: &T) -> Option<Vec<T::Product>> {
    let required = numerics::current().bar();
    T::FORMULA
        .chain(T::PRECISION)
        .iter()
        .filter(|backend| backend.coverage(T::FORMULA).serves_at(required))
        .find_map(|&backend| task.offer(backend))
}

#[cfg(test)]
#[path = "tests/chain_tests.rs"]
mod tests;
