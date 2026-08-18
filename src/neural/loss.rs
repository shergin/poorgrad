//! Loss functions: named formulas that compose recorded operations into a
//! scalar training objective.
//!
//! A loss is a composition, not a primitive: it records existing graph
//! operations, so its gradient falls out of the chain rule with no
//! dedicated backward rule. A formula earns a fused `Function` variant
//! only where composition cannot express it — [`Value::log_softmax`] is
//! that fused core here, because its stability demands a max shift no
//! recorded composition can perform — while everything around the core
//! stays plain, readable composition.

use crate::{Tensorial, Value};

/// Records the cross-entropy loss of `logits` against `targets` on their
/// network and returns the rank-0 loss value.
///
/// It composes the mean negative log-likelihood in the expanded form
/// `((targets.sum_along(1) * logsumexp(logits)).sum() -
/// (targets * logits).sum()) / targets.sum()`. The expansion is exact
/// mathematics — each row's `-t . (x - lse)` distributed — and it is the
/// stable spelling: the fused [`Value::logsumexp`] is finite for every
/// finite logit, and no term ever multiplies a zero target by an
/// infinite log-probability (the `0 * -inf = NaN` a
/// `targets * log_softmax` product produces once finite logits differ
/// by more than the representable range). The normalizer is the
/// targets' total mass: the batch size for one-hot targets — the
/// standard mean reduction — while soft or weighted targets normalize
/// by their own weight.
///
/// # Parameters
/// - `logits`: The unnormalized class scores, rank 2 `[batch, classes]`.
/// - `targets`: The target distribution per sample, shaped like `logits`.
///   Feed a one-hot [`Tensor::selection`](crate::Tensor::selection) as a
///   per-run input so one recorded graph serves any batch of labels.
///   Target weights must be finite and nonnegative with a strictly
///   positive total mass; outside that domain the "mean negative
///   log-likelihood" has no interpretation and the result follows IEEE
///   arithmetic (an all-zero target tensor, for instance, divides zero
///   by zero into `NaN`).
///
/// # Panics
/// Panics if the values belong to different networks, `logits` is not
/// rank 2, or the shapes differ.
pub fn cross_entropy<'tape, Data: Tensorial>(
    logits: Value<'tape, Data>,
    targets: Value<'tape, Data>,
) -> Value<'tape, Data> {
    let logits_shape = logits.shape();
    assert_eq!(
        logits_shape.rank(),
        2,
        "cross-entropy logits must be rank 2 [batch, classes], got {logits_shape}"
    );
    let targets_shape = targets.shape();
    assert_eq!(
        targets_shape, logits_shape,
        "cross-entropy targets {targets_shape} must be shaped like the logits {logits_shape}"
    );

    let normalizers = (targets.sum_along(1) * logits.logsumexp(1)).sum();
    let alignment = (targets * logits).sum();
    (normalizers - alignment) / targets.sum()
}

#[cfg(test)]
#[path = "tests/loss_tests.rs"]
mod tests;
