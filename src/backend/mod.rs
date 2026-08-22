//! The acceleration stack's one relation: an implementer may serve
//! a named formula, at a fidelity, against the oracle.
//!
//! Coverage declares *may* — the [`Backend::coverage`] matrix over
//! [`Formula`] and [`Precision`], with a certified [`Fidelity`] per
//! cell. The offer decides *will* — [`offered`] walks a task's
//! declared chain, admitting each member by fidelity-meets-posture, and
//! every member may still decline (thresholds, stride mappings,
//! device presence). The oracle defines *is* — the reference paths
//! are the substrate every decline falls to, in every build; on a
//! build without backend features every offer answers `None`, the
//! seam's fixed point, not dead code. Each task type carries its
//! formula and precision through the crate-internal `Task` contract,
//! so a task can only walk its own chain. Everything here exists in
//! every build: interrogating the stack never needs a `cfg`.

// Every backend module is compiled in every build: it holds the
// always-answering manifest, and keeps its kernels (with their
// scoped `unsafe` allow) behind the feature `cfg` internally.
mod accelerate;
#[allow(clippy::module_inception)]
mod backend;
mod coverage;
mod cuda;
mod formula;
mod fused;
mod manifest;
mod metal;
mod numerics;
mod task;
// Safe stride classification shared by the BLAS-shaped backends;
// compiled exactly where one of them is.
#[cfg(any(
    all(feature = "accelerate", target_os = "macos"),
    all(feature = "cuda", target_os = "linux")
))]
mod operand;
mod simd;
mod stablehlo;

pub use backend::{Backend, BackendUnavailable};
pub use coverage::{Coverage, Dispatch, Fidelity};
pub use formula::{Formula, Precision};
pub use numerics::Numerics;

pub(crate) use numerics::NumericsScope;
pub(crate) use task::MapTask;

use task::Task;

/// It offers a task down its formula's chain, answering `None` when
/// every member declines: the chain's one entry point, monomorphized
/// per task type.
///
/// Admission is the fidelity rule, not a posture special case: a chain
/// member serves only if its declared fidelity meets the fidelity the
/// current posture demands, so `Exact` excludes every envelope
/// kernel and would admit a bit-certified one.
pub(crate) fn offered<T: Task>(task: &T) -> Option<T::Product> {
    let required = numerics::current().fidelity();
    T::FORMULA
        .chain(T::PRECISION)
        .iter()
        .filter(|backend| backend.coverage(T::FORMULA).meets(required))
        .find_map(|&backend| task.offer(backend))
}

#[cfg(test)]
#[path = "tests/chain_tests.rs"]
mod tests;
