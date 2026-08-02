//! The compile-time acceleration backend chain.
//!
//! Feature-gated backend modules join this chain as arms tried in
//! declaration order, so that order is the documented priority. Each
//! backend may decline a task (wrong shape, unprofitable size,
//! unavailable device) by answering `None`, and the payload's
//! built-in paths compute whenever the whole chain declines. Today
//! no backend feature exists, the chain is empty, and every entry
//! answers `None` — the functions are the seam's fixed point, not
//! dead code.

use crate::GemmTask;

/// It offers an `f32` task to every compiled backend, hardware-greediest
/// first, answering `None` when none accepts.
pub(crate) fn gemm_f32(_task: &GemmTask<'_, f32>) -> Option<Vec<f32>> {
    None
}

/// It offers an `f64` task to every compiled backend, answering `None`
/// when none accepts.
pub(crate) fn gemm_f64(_task: &GemmTask<'_, f64>) -> Option<Vec<f64>> {
    None
}
