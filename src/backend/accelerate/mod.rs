//! The Accelerate implementer: the always-compiled manifest, and
//! the BLAS/vForce kernels behind the `accelerate` feature.

use super::backend::BackendUnavailable;
use super::coverage::{Coverage, Dispatch, Fidelity};
use super::formula::{Formula, Precision};
use super::manifest::Manifest;

#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[allow(unsafe_code)]
mod kernels;

#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub(super) use kernels::{batch_norm_f32, batch_norm_f64, gemm_f32, gemm_f64, map_f32, map_f64};

/// Apple's Accelerate framework, described in every build.
pub(super) struct Accelerate;

impl Manifest for Accelerate {
    const DISPATCH: Dispatch = Dispatch::Offered;

    fn coverage(formula: Formula) -> Coverage {
        match formula {
            // One cblas call per product on the AMX/SME matrix
            // units, and vForce for whole-buffer transcendentals;
            // both take either precision and reorder sums, so the
            // fidelity is the envelope.
            Formula::Gemm | Formula::Map => Coverage::Serves {
                fidelity: Fidelity::Envelope,
                precisions: Precision::ALL,
            },
            // One vDSP pass per feature — mean, centered variance,
            // and the affine as a fused multiply-add — reordering
            // the recorded reductions, so the fidelity is the
            // envelope.
            Formula::BatchNormTraining => Coverage::Serves {
                fidelity: Fidelity::Envelope,
                precisions: Precision::ALL,
            },
            Formula::WindowProduct | Formula::ReduceWindow | Formula::BatchNormInference => {
                Coverage::Absent
            }
        }
    }

    fn compiled() -> bool {
        cfg!(all(feature = "accelerate", target_os = "macos"))
    }

    fn status() -> Result<(), BackendUnavailable> {
        if !cfg!(feature = "accelerate") {
            return Err(BackendUnavailable::NotCompiled);
        }
        if !cfg!(target_os = "macos") {
            return Err(BackendUnavailable::PlatformUnsupported);
        }
        // Accelerate is a link-time dependency with nothing to
        // initialize and nothing to lose at run time.
        Ok(())
    }
}
