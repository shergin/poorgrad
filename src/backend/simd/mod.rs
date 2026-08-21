//! The simd implementer: the always-compiled descriptor, and the
//! `matrixmultiply` kernels behind the `simd` feature.

use super::backend::BackendUnavailable;
use super::coverage::{Bar, Cell, Dispatch, Precisions};
use super::formula::{Formula, Precision};
use super::implementer::Implementer;

#[cfg(feature = "simd")]
#[allow(unsafe_code)]
mod kernels;

#[cfg(feature = "simd")]
pub(super) use kernels::{gemm_f32, gemm_f64};

/// The portable CPU rung, described in every build.
pub(super) struct Simd;

impl Implementer for Simd {
    const DISPATCH: Dispatch = Dispatch::Offered;

    fn coverage(formula: Formula) -> Cell {
        match formula {
            // Tuned single-threaded microkernels for both
            // precisions; packing reorders sums, so the bar is the
            // envelope.
            Formula::Gemm => Cell::Serves {
                bar: Bar::Envelope,
                precisions: Precisions::Only(Precision::ALL),
            },
            // `matrixmultiply` is GEMM-only.
            Formula::Map
            | Formula::WindowProduct
            | Formula::ReduceWindow
            | Formula::BatchNormTraining
            | Formula::BatchNormInference => Cell::Absent,
        }
    }

    fn compiled() -> bool {
        cfg!(feature = "simd")
    }

    fn status() -> Result<(), BackendUnavailable> {
        if !cfg!(feature = "simd") {
            return Err(BackendUnavailable::NotCompiled);
        }
        // Pure CPU code with runtime instruction-set dispatch: no
        // platform arm, no device, nothing to initialize and nothing
        // to lose at run time.
        Ok(())
    }
}
