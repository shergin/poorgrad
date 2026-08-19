use super::batch_norm::BatchNormalization;
use super::reduce_window::ReduceWindow;
use super::window::WindowProduct;

/// A recognized pattern rooted at one plan node.
///
/// It is a compile-time match over frozen structure, not a tape
/// rewrite. Every pattern raises: `Plan::emit_stablehlo` replaces the
/// matched group with the named operation at its root. Fusing at home
/// is the extra power a variant may carry, answered by
/// [`Pattern::fused`]; a homing pattern is stored only on
/// forward-only plans, so engine-backward memory contracts stay
/// exact. Dispatch is a plain `match`, the same shape as `Function`.
#[derive(Debug, Clone)]
pub(crate) enum Pattern {
    /// Canonical im2col chain feeding a rank-2 `matmul`.
    WindowProduct(WindowProduct),
    /// Canonical max-pool window fold ending in the facade squeeze.
    ReduceWindow(ReduceWindow),
    /// Batch normalization by the batch's own statistics, with the
    /// mean and variance as named results.
    BatchNormTraining(BatchNormalization),
    /// Batch normalization by supplied statistics.
    BatchNormInference(BatchNormalization),
}

impl Pattern {
    /// Returns the window-GEMM group this pattern fuses at home, if
    /// any: the one policy point deciding which variants replace their
    /// root with a payload call in `Plan::forward`. The return type
    /// widens to an enum when a second homing pattern lands.
    pub(crate) fn fused(&self) -> Option<&WindowProduct> {
        match self {
            Pattern::WindowProduct(group) => Some(group),
            Pattern::ReduceWindow(_)
            | Pattern::BatchNormTraining(_)
            | Pattern::BatchNormInference(_) => None,
        }
    }
}
