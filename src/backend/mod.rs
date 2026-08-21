//! The compile-time acceleration backend chain.
//!
//! The chain's structure is declared data: [`TaskKind`] names the
//! task vocabulary, each kind's [`TaskKind::chain`] is the offer
//! order with membership as the coverage claim, and
//! [`Backend::serves`] derives from it. Dispatch iterates the
//! declared chain through one `cfg`'d match per entry point, so a
//! backend missing from the build answers `None` and the payload's
//! built-in paths compute whenever the whole chain declines. On a
//! build without backend features every match is empty and every
//! entry answers `None` — the seam's fixed point, not dead code.
//! [`Backend::status`] answers for every defined backend in every
//! build, so interrogating the chain never needs a `cfg`.

#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[allow(unsafe_code)]
mod accelerate;
#[allow(clippy::module_inception)]
mod backend;
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

pub(crate) use numerics::NumericsScope;

use crate::{GemmTask, MapOperation};

/// It offers an `f32` product down [`TaskKind::GemmF32`]'s chain,
/// answering `None` when every member declines.
pub(crate) fn gemm_f32(task: &GemmTask<'_, f32>) -> Option<Vec<f32>> {
    // The numerics posture outranks the whole chain: `Exact` declines
    // everything, so the built-in reference paths compute.
    if numerics::current() == Numerics::Exact {
        return None;
    }
    TaskKind::GemmF32
        .chain()
        .iter()
        .find_map(|&backend| offer_gemm_f32(backend, task))
}

/// It offers an `f64` product down [`TaskKind::GemmF64`]'s chain,
/// answering `None` when every member declines.
pub(crate) fn gemm_f64(task: &GemmTask<'_, f64>) -> Option<Vec<f64>> {
    if numerics::current() == Numerics::Exact {
        return None;
    }
    TaskKind::GemmF64
        .chain()
        .iter()
        .find_map(|&backend| offer_gemm_f64(backend, task))
}

/// It offers an `f32` elementwise map down [`TaskKind::MapF32`]'s
/// chain, answering `None` when every member declines.
pub(crate) fn map_f32(operation: MapOperation, elements: &[f32]) -> Option<Vec<f32>> {
    if numerics::current() == Numerics::Exact {
        return None;
    }
    TaskKind::MapF32
        .chain()
        .iter()
        .find_map(|&backend| offer_map_f32(backend, operation, elements))
}

/// It offers an `f64` elementwise map down [`TaskKind::MapF64`]'s
/// chain, answering `None` when every member declines.
pub(crate) fn map_f64(operation: MapOperation, elements: &[f64]) -> Option<Vec<f64>> {
    if numerics::current() == Numerics::Exact {
        return None;
    }
    TaskKind::MapF64
        .chain()
        .iter()
        .find_map(|&backend| offer_map_f64(backend, operation, elements))
}

/// It offers the task to one backend; a member missing from this
/// build answers `None`, the chain's fixed point.
fn offer_gemm_f32(backend: Backend, task: &GemmTask<'_, f32>) -> Option<Vec<f32>> {
    match backend {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        Backend::Accelerate => accelerate::gemm_f32(task),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Backend::Metal => metal::gemm_f32(task),
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        Backend::Cuda => cuda::gemm_f32(task),
        #[cfg(feature = "simd")]
        Backend::Simd => simd::gemm_f32(task),
        _ => {
            let _ = task;
            None
        }
    }
}

/// The `f64` twin of [`offer_gemm_f32`].
fn offer_gemm_f64(backend: Backend, task: &GemmTask<'_, f64>) -> Option<Vec<f64>> {
    match backend {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        Backend::Accelerate => accelerate::gemm_f64(task),
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        Backend::Cuda => cuda::gemm_f64(task),
        #[cfg(feature = "simd")]
        Backend::Simd => simd::gemm_f64(task),
        _ => {
            let _ = task;
            None
        }
    }
}

/// It offers the map to one backend; a member missing from this
/// build answers `None`.
fn offer_map_f32(backend: Backend, operation: MapOperation, elements: &[f32]) -> Option<Vec<f32>> {
    match backend {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        Backend::Accelerate => accelerate::map_f32(operation, elements),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Backend::Metal => metal::map_f32(operation, elements),
        _ => {
            let _ = (operation, elements);
            None
        }
    }
}

/// The `f64` twin of [`offer_map_f32`].
fn offer_map_f64(backend: Backend, operation: MapOperation, elements: &[f64]) -> Option<Vec<f64>> {
    match backend {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        Backend::Accelerate => accelerate::map_f64(operation, elements),
        _ => {
            let _ = (operation, elements);
            None
        }
    }
}

#[cfg(test)]
#[path = "tests/chain_tests.rs"]
mod tests;
