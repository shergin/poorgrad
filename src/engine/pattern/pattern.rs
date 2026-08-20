use super::batch_norm::BatchNormalization;
use super::reduce_window::ReduceWindow;
use super::window::WindowProduct;

/// A recognized pattern rooted at one plan node.
///
/// It is a compile-time match over frozen structure, not a tape
/// rewrite — and it carries no policy. What to do with a match belongs
/// to the consumers: each owns a repertoire (the patterns it can act
/// on) and an action per variant in its own module — the forward
/// run's kernel table lives beside the plan, the raises beside the
/// emitter. The shape mirrors `Function`, the role does not: a
/// `Function` carries its rules because they are the single spec,
/// while a pattern has as many interpretations as consumers.
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
