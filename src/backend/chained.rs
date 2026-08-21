use crate::{GemmTask, MapOperation};

#[cfg(all(feature = "accelerate", target_os = "macos"))]
use super::accelerate;
use super::backend::Backend;
#[cfg(all(feature = "cuda", target_os = "linux"))]
use super::cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
use super::metal;
#[cfg(feature = "simd")]
use super::simd;
use super::task::TaskKind;

/// A task the chain can be offered: the compile-time twin of
/// [`TaskKind`], carried by the task type itself.
///
/// `KIND` names the chain the task dispatches under and `offer` is
/// the task's entry into one backend, so a task can only ever walk
/// its own chain — the kind-to-dispatch link holds by construction,
/// not by convention. The implementations are the closed set of
/// task types the chain understands; a future kind arrives as a new
/// implementation beside its [`TaskKind`] variant.
pub(crate) trait Chained: Sized {
    /// The element type of the computed result.
    type Product;

    /// The kind whose chain this task dispatches under.
    const KIND: TaskKind;

    /// It offers this task to one backend; a backend missing from
    /// the build answers `None`, the chain's fixed point.
    fn offer(&self, backend: Backend) -> Option<Vec<Self::Product>>;
}

/// One whole-buffer elementwise transcendental as an offerable
/// task: a [`MapOperation`] paired with its elements, the map
/// chains' twin of [`GemmTask`].
pub(crate) struct MapTask<'buffers, Element> {
    operation: MapOperation,
    elements: &'buffers [Element],
}

impl<'buffers, Element> MapTask<'buffers, Element> {
    /// Creates the task over a whole buffer.
    pub(crate) fn new(operation: MapOperation, elements: &'buffers [Element]) -> Self {
        Self {
            operation,
            elements,
        }
    }
}

impl Chained for GemmTask<'_, f32> {
    type Product = f32;

    const KIND: TaskKind = TaskKind::GemmF32;

    fn offer(&self, backend: Backend) -> Option<Vec<f32>> {
        match backend {
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            Backend::Accelerate => accelerate::gemm_f32(self),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => metal::gemm_f32(self),
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            Backend::Cuda => cuda::gemm_f32(self),
            #[cfg(feature = "simd")]
            Backend::Simd => simd::gemm_f32(self),
            _ => None,
        }
    }
}

impl Chained for GemmTask<'_, f64> {
    type Product = f64;

    const KIND: TaskKind = TaskKind::GemmF64;

    fn offer(&self, backend: Backend) -> Option<Vec<f64>> {
        match backend {
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            Backend::Accelerate => accelerate::gemm_f64(self),
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            Backend::Cuda => cuda::gemm_f64(self),
            #[cfg(feature = "simd")]
            Backend::Simd => simd::gemm_f64(self),
            _ => None,
        }
    }
}

impl Chained for MapTask<'_, f32> {
    type Product = f32;

    const KIND: TaskKind = TaskKind::MapF32;

    fn offer(&self, backend: Backend) -> Option<Vec<f32>> {
        match backend {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => metal::map_f32(self.operation, self.elements),
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            Backend::Accelerate => accelerate::map_f32(self.operation, self.elements),
            _ => {
                let _ = (self.operation, self.elements);
                None
            }
        }
    }
}

impl Chained for MapTask<'_, f64> {
    type Product = f64;

    const KIND: TaskKind = TaskKind::MapF64;

    fn offer(&self, backend: Backend) -> Option<Vec<f64>> {
        match backend {
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            Backend::Accelerate => accelerate::map_f64(self.operation, self.elements),
            _ => {
                let _ = (self.operation, self.elements);
                None
            }
        }
    }
}
