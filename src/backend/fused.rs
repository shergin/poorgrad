use super::backend::BackendUnavailable;
use super::coverage::{Bar, Cell, Dispatch, Precisions};
use super::formula::Formula;
use super::implementer::Implementer;

/// The crate's own fused kernels for composed formulas, elected
/// onto plans at compile time and executing in-process through the
/// payload seam.
///
/// The actions live with their consumer (the kernel table beside
/// the plan, per the symmetry decision); this descriptor holds only
/// the declared coverage the election reads.
pub(super) struct Fused;

impl Implementer for Fused {
    const DISPATCH: Dispatch = Dispatch::Elected;

    fn coverage(formula: Formula) -> Cell {
        match formula {
            // `windowed_product` computes through the gemm seam in
            // the recorded accumulation order: bit-identical under
            // both postures, proven by the plan snapshots — the one
            // cell at the bit-identity bar, since the oracle's bits
            // live in this process.
            Formula::WindowProduct => Cell::Serves {
                bar: Bar::BitIdentical,
                precisions: Precisions::Any,
            },
            // The pool kernel waits on a profile (`max` is
            // associative, so it could clear even the bit-identity
            // bar); the batch-norm kernels would reassociate
            // reductions and arrive envelope-only.
            Formula::Gemm
            | Formula::Map
            | Formula::ReduceWindow
            | Formula::BatchNormTraining
            | Formula::BatchNormInference => Cell::Absent,
        }
    }

    fn compiled() -> bool {
        true
    }

    fn status() -> Result<(), BackendUnavailable> {
        // In-process code compiled into every build: nothing to
        // initialize, nothing to lose.
        Ok(())
    }
}
