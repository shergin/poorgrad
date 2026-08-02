//! The compile-time acceleration backend chain.
//!
//! Feature-gated backend modules join this chain as arms tried in
//! declaration order, so that order is the documented priority. Each
//! backend may decline a task (wrong shape, unprofitable size,
//! unavailable device) by answering `None`, and the payload's
//! built-in paths compute whenever the whole chain declines. The
//! first resident is the `accelerate` feature's arm; on a build
//! without backend features the chain is empty and every entry
//! answers `None` — the seam's fixed point, not dead code.
//! [`Backend::status`] answers for every defined backend in every
//! build, so interrogating the chain never needs a `cfg`.

#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[allow(unsafe_code)]
mod accelerate;
#[allow(clippy::module_inception)]
mod backend;

pub use backend::{Backend, BackendUnavailable};

use crate::GemmTask;

/// It offers an `f32` task to every compiled backend, hardware-greediest
/// first, answering `None` when none accepts.
pub(crate) fn gemm_f32(task: &GemmTask<'_, f32>) -> Option<Vec<f32>> {
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    if let Some(product) = accelerate::gemm_f32(task) {
        return Some(product);
    }
    let _ = task;
    None
}

/// It offers an `f64` task to every compiled backend, answering `None`
/// when none accepts.
pub(crate) fn gemm_f64(task: &GemmTask<'_, f64>) -> Option<Vec<f64>> {
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    if let Some(product) = accelerate::gemm_f64(task) {
        return Some(product);
    }
    let _ = task;
    None
}
