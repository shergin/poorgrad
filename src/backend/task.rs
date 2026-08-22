use crate::{BatchNormTask, GemmTask, MapOperation, Normalized};

#[cfg(all(feature = "accelerate", target_os = "macos"))]
use super::accelerate;
use super::backend::Backend;
#[cfg(all(feature = "cuda", target_os = "linux"))]
use super::cuda;
use super::formula::{Formula, Precision};
#[cfg(all(feature = "metal", target_os = "macos"))]
use super::metal;
#[cfg(feature = "simd")]
use super::simd;

/// A buffer task the chain can be offered: a formula's run-time
/// face, carried by the task type itself.
///
/// `FORMULA` and `PRECISION` name the chain the task dispatches
/// under and `offer` is the task's entry into one backend, so a task
/// can only ever walk its own chain — the vocabulary-to-dispatch
/// link holds by construction, not by convention. The
/// implementations are the closed set of task types the chain
/// understands; a composed formula earning an offerable kernel
/// arrives as a new implementation beside its [`Formula`] entry.
pub(crate) trait Task: Sized {
    /// The task's whole result type.
    type Product;

    /// The vocabulary entry this task instantiates.
    const FORMULA: Formula;

    /// The forwarding precision of this task's buffers.
    const PRECISION: Precision;

    /// It offers this task to one backend; a backend missing from
    /// the build answers `None`, the chain's fixed point.
    fn offer(&self, backend: Backend) -> Option<Self::Product>;
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

impl Task for GemmTask<'_, f32> {
    type Product = Vec<f32>;

    const FORMULA: Formula = Formula::Gemm;
    const PRECISION: Precision = Precision::F32;

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

impl Task for GemmTask<'_, f64> {
    type Product = Vec<f64>;

    const FORMULA: Formula = Formula::Gemm;
    const PRECISION: Precision = Precision::F64;

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

impl Task for MapTask<'_, f32> {
    type Product = Vec<f32>;

    const FORMULA: Formula = Formula::Map;
    const PRECISION: Precision = Precision::F32;

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

impl Task for MapTask<'_, f64> {
    type Product = Vec<f64>;

    const FORMULA: Formula = Formula::Map;
    const PRECISION: Precision = Precision::F64;

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

impl Task for BatchNormTask<'_, f32> {
    type Product = Normalized<f32>;

    const FORMULA: Formula = Formula::BatchNormTraining;
    const PRECISION: Precision = Precision::F32;

    fn offer(&self, backend: Backend) -> Option<Normalized<f32>> {
        match backend {
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            Backend::Accelerate => accelerate::batch_norm_f32(self),
            _ => None,
        }
    }
}

impl Task for BatchNormTask<'_, f64> {
    type Product = Normalized<f64>;

    const FORMULA: Formula = Formula::BatchNormTraining;
    const PRECISION: Precision = Precision::F64;

    fn offer(&self, backend: Backend) -> Option<Normalized<f64>> {
        match backend {
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            Backend::Accelerate => accelerate::batch_norm_f64(self),
            _ => None,
        }
    }
}
