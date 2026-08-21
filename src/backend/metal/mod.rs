//! The metal implementer: the always-compiled descriptor, and the
//! simdgroup-matrix GPU kernels behind the `metal` feature.

use super::backend::BackendUnavailable;
use super::coverage::{Bar, Cell, Dispatch, Precisions};
use super::formula::{Formula, Precision};
use super::implementer::Implementer;

#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(unsafe_code)]
mod kernels;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(super) use kernels::{gemm_f32, map_f32};

/// The GPU rung for very large `f32` work, described in every build.
pub(super) struct Metal;

impl Implementer for Metal {
    const DISPATCH: Dispatch = Dispatch::Offered;

    fn coverage(formula: Formula) -> Cell {
        match formula {
            // Hand-written simdgroup-matrix kernels for products and
            // elementwise maps; Metal has no `f64` at all, and the
            // GPU sums in tile order, so the bar is the envelope.
            Formula::Gemm | Formula::Map => Cell::Serves {
                bar: Bar::Envelope,
                precisions: Precisions::Only(&[Precision::F32]),
            },
            Formula::WindowProduct
            | Formula::ReduceWindow
            | Formula::BatchNormTraining
            | Formula::BatchNormInference => Cell::Absent,
        }
    }

    fn compiled() -> bool {
        cfg!(all(feature = "metal", target_os = "macos"))
    }

    fn status() -> Result<(), BackendUnavailable> {
        if !cfg!(feature = "metal") {
            return Err(BackendUnavailable::NotCompiled);
        }
        if !cfg!(target_os = "macos") {
            return Err(BackendUnavailable::PlatformUnsupported);
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            kernels::status()
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            unreachable!("the cfg! guards above cover this build")
        }
    }
}
