use crate::Differentiable;
use crate::engine::ValueId;

/// A differentiable operation: how a node computes its payload from its
/// operands, and how it routes gradients back to them.
///
/// It is implemented by each computed `Function` variant and dispatched
/// through the enum with a plain `match`, so implementations stay
/// statically sized and the trait never needs to be object safe. Leaves
/// and parameters do not implement it: they are supplied, not computed,
/// and the enum's dispatch handles them directly. Every method receives
/// the node's operand links as a positional slice, exactly as they were
/// recorded into the tape's operands column. `forward` and `backward`
/// read and write per-run buffers indexed by allocation order, relying
/// on the tape guarantee that operands are always recorded before the
/// nodes that use them.
pub(crate) trait Operation<Data: Differentiable> {
    /// Computes this node's payload from the values of earlier nodes.
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data;

    /// Accumulates operand gradients, given this node's computed
    /// `output` payload and its own `gradient`.
    fn backward(
        &self,
        operands: &[ValueId],
        values: &[Data],
        output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    );
}

/// Splits the positional operand list of a unary operation.
///
/// It is generic so the same helper serves operand links and operand
/// shapes.
///
/// # Panics
/// Panics if `operands` does not hold exactly one entry; recording
/// supplies every node's operands, so a mismatch is an engine bug.
pub(crate) fn unary<T>(operands: &[T]) -> &T {
    let [operand] = operands else {
        panic!("unary operation expects exactly one operand");
    };
    operand
}

/// Splits the positional operand list of a binary operation.
///
/// It is generic so the same helper serves operand links and operand
/// shapes.
///
/// # Panics
/// Panics if `operands` does not hold exactly two entries; recording
/// supplies every node's operands, so a mismatch is an engine bug.
pub(crate) fn binary<T>(operands: &[T]) -> (&T, &T) {
    let [left, right] = operands else {
        panic!("binary operation expects exactly two operands");
    };
    (left, right)
}
