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
#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(unsafe_code)]
mod metal;
// The one arm with no `target_os`: the simd backend is real on
// every platform.
#[cfg(feature = "simd")]
#[allow(unsafe_code)]
mod simd;

pub use backend::{Backend, BackendUnavailable};

use crate::{GemmTask, MapOperation};

/// It offers an `f32` task to every compiled backend, hardware-greediest
/// first, answering `None` when none accepts.
pub(crate) fn gemm_f32(task: &GemmTask<'_, f32>) -> Option<Vec<f32>> {
    // Accelerate leads: the measured crossover has AMX ahead of the
    // current Metal kernel at every size, so Metal serves what BLAS
    // declines (stride patterns like broadcasts) and metal-only
    // builds. The order flips back if the kernel ever earns it.
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    if let Some(product) = accelerate::gemm_f32(task) {
        return Some(product);
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if let Some(product) = metal::gemm_f32(task) {
        return Some(product);
    }
    #[cfg(feature = "simd")]
    if let Some(product) = simd::gemm_f32(task) {
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
    #[cfg(feature = "simd")]
    if let Some(product) = simd::gemm_f64(task) {
        return Some(product);
    }
    let _ = task;
    None
}

/// It offers an `f32` elementwise map to every compiled backend,
/// answering `None` when none accepts.
pub(crate) fn map_f32(operation: MapOperation, elements: &[f32]) -> Option<Vec<f32>> {
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    if let Some(mapped) = accelerate::map_f32(operation, elements) {
        return Some(mapped);
    }
    let _ = (operation, elements);
    None
}

/// It offers an `f64` elementwise map to every compiled backend,
/// answering `None` when none accepts.
pub(crate) fn map_f64(operation: MapOperation, elements: &[f64]) -> Option<Vec<f64>> {
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    if let Some(mapped) = accelerate::map_f64(operation, elements) {
        return Some(mapped);
    }
    let _ = (operation, elements);
    None
}
