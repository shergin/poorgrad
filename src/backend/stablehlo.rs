use super::backend::BackendUnavailable;
use super::coverage::{Coverage, Dispatch, Fidelity, Precisions};
use super::formula::Formula;
use super::manifest::Manifest;

/// The StableHLO translation library: elected groups and leaf
/// operations lower into a module a foreign runtime executes.
///
/// The raises live with their consumer (beside the emitter, per the
/// symmetry decision); this manifest holds only the declared
/// coverage emission elects by, and its column stays total by test.
pub(super) struct StableHlo;

impl Manifest for StableHlo {
    const DISPATCH: Dispatch = Dispatch::Translated;

    fn coverage(formula: Formula) -> Coverage {
        // The translation column is total: every formula lowers,
        // leaf entries as single operations and composed entries as
        // raised library calls, under the envelope fidelity — nobody
        // controls the foreign runtime's kernels.
        match formula {
            Formula::Gemm
            | Formula::Map
            | Formula::WindowProduct
            | Formula::ReduceWindow
            | Formula::BatchNormTraining
            | Formula::BatchNormInference => Coverage::Serves {
                fidelity: Fidelity::Envelope,
                precisions: Precisions::Any,
            },
        }
    }

    fn compiled() -> bool {
        true
    }

    fn status() -> Result<(), BackendUnavailable> {
        // Emitting is in-process string building in every build;
        // running the module is the foreign runtime's business.
        Ok(())
    }
}
