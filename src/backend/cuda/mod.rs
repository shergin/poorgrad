//! The cuda implementer: the always-compiled manifest, and the
//! cuBLAS kernels behind the `cuda` feature.

use super::backend::BackendUnavailable;
use super::coverage::{Coverage, Dispatch, Fidelity, Precisions};
use super::formula::{Formula, Precision};
use super::manifest::Manifest;

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[allow(unsafe_code)]
mod kernels;

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(super) use kernels::{gemm_f32, gemm_f64};

/// The NVIDIA rung for large products, described in every build.
pub(super) struct Cuda;

impl Manifest for Cuda {
    const DISPATCH: Dispatch = Dispatch::Offered;

    fn coverage(formula: Formula) -> Coverage {
        match formula {
            // One cuBLAS call per product under the column-major
            // swap, both precisions; device sums reorder, so the fidelity
            // is the envelope.
            Formula::Gemm => Coverage::Serves {
                fidelity: Fidelity::Envelope,
                precisions: Precisions::Only(Precision::ALL),
            },
            // A cuda map would be PCIe-bound: copies alone sink an
            // elementwise pass.
            Formula::Map
            | Formula::WindowProduct
            | Formula::ReduceWindow
            | Formula::BatchNormTraining
            | Formula::BatchNormInference => Coverage::Absent,
        }
    }

    fn compiled() -> bool {
        cfg!(all(feature = "cuda", target_os = "linux"))
    }

    fn status() -> Result<(), BackendUnavailable> {
        if !cfg!(feature = "cuda") {
            return Err(BackendUnavailable::NotCompiled);
        }
        if !cfg!(target_os = "linux") {
            return Err(BackendUnavailable::PlatformUnsupported);
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            kernels::status()
        }
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            unreachable!("the cfg! guards above cover this build")
        }
    }
}
