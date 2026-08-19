use smallvec::{SmallVec, smallvec};

use super::batch_norm::{BatchNormInference, BatchNormTraining};
use super::reduce_window::ReduceWindow;
use super::window::WindowProduct;

/// A recognized motif rooted at one plan node.
///
/// It is a compile-time match over frozen structure, not a tape
/// rewrite and not itself a raise or a fuse. Home (`Plan::forward`)
/// and abroad (`Plan::emit_stablehlo`) are optional actions on the
/// same entry, decided by [`Pattern::homes`] / [`Pattern::raises`] on
/// the variant — not by per-entry flags. Dispatch is a plain `match`,
/// the same shape as `Function`.
#[derive(Debug, Clone)]
pub(crate) enum Pattern {
    /// Canonical im2col chain feeding a rank-2 `matmul`.
    WindowProduct(WindowProduct),
    /// Canonical max-pool window fold ending in the facade squeeze.
    ReduceWindow(ReduceWindow),
    /// Batch normalization by the batch's own statistics, with the
    /// mean and variance as named results.
    BatchNormTraining(BatchNormTraining),
    /// Batch normalization by supplied statistics.
    BatchNormInference(BatchNormInference),
}

impl Pattern {
    /// Returns the slots a home action reads past the root's operand
    /// links; liveness must keep them alive until the fused call.
    /// Raise-only motifs read nothing extra: their chains run.
    pub(crate) fn extra_reads(&self) -> SmallVec<[usize; 4]> {
        match self {
            Pattern::WindowProduct(group) => {
                smallvec![group.source, group.kernel]
            }
            Pattern::ReduceWindow(_)
            | Pattern::BatchNormTraining(_)
            | Pattern::BatchNormInference(_) => SmallVec::new(),
        }
    }

    /// Returns whether a forward run replaces the root with a payload
    /// call. A homing motif is stored only on forward-only plans, so
    /// engine-backward memory contracts stay exact.
    pub(crate) fn homes(&self) -> bool {
        match self {
            Pattern::WindowProduct(_) => true,
            Pattern::ReduceWindow(_)
            | Pattern::BatchNormTraining(_)
            | Pattern::BatchNormInference(_) => false,
        }
    }

    /// Returns whether emission replaces the root with a named
    /// operation.
    pub(crate) fn raises(&self) -> bool {
        match self {
            Pattern::WindowProduct(_)
            | Pattern::ReduceWindow(_)
            | Pattern::BatchNormTraining(_)
            | Pattern::BatchNormInference(_) => true,
        }
    }
}
