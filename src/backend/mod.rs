//! The compile-time acceleration backend chain.
//!
//! The chain's structure is declared data: [`TaskKind`] names the
//! task vocabulary, each kind's [`TaskKind::chain`] is the offer
//! order with membership as the coverage claim, and
//! [`Backend::serves`] derives from it. Each task type carries its
//! kind and its per-backend entry through the crate-internal
//! `Chained` contract, so [`offered`] — the chain's one entry
//! point — can only walk a task down its own chain. A backend
//! missing from the build answers `None`, and the payload's
//! built-in paths compute whenever the whole chain declines; on a
//! build without backend features every offer is `None` — the
//! seam's fixed point, not dead code. [`Backend::status`] answers
//! for every defined backend in every build, so interrogating the
//! chain never needs a `cfg`.

#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[allow(unsafe_code)]
mod accelerate;
#[allow(clippy::module_inception)]
mod backend;
mod chained;
#[cfg(all(feature = "cuda", target_os = "linux"))]
#[allow(unsafe_code)]
mod cuda;
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
mod task;

pub use backend::{Backend, BackendUnavailable};
pub use numerics::Numerics;
pub use task::TaskKind;

pub(crate) use chained::MapTask;
pub(crate) use numerics::NumericsScope;

use chained::Chained;

/// It offers a task down its kind's chain, answering `None` when
/// every member declines: the chain's one entry point, monomorphized
/// per task type.
pub(crate) fn offered<T: Chained>(task: &T) -> Option<Vec<T::Product>> {
    // The numerics posture outranks the whole chain: `Exact` declines
    // everything, so the built-in reference paths compute.
    if numerics::current() == Numerics::Exact {
        return None;
    }
    T::KIND
        .chain()
        .iter()
        .find_map(|&backend| task.offer(backend))
}

#[cfg(test)]
#[path = "tests/chain_tests.rs"]
mod tests;
