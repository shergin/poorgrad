use smallvec::SmallVec;

use crate::Differentiable;

/// One cotangent per operand, in the operation's positional order.
///
/// `None` marks an operand that is data rather than a differentiable
/// value (a broadcast's reference, a gather's selection), so
/// non-differentiability is structural in the rule's signature rather
/// than implicit by omission.
pub(crate) type Cotangents<Data> = SmallVec<[Option<Data>; 2]>;

/// A differentiable operation: how a node computes its payload from its
/// operands' payloads, and the cotangent it hands back to each operand.
///
/// It is implemented by each computed `Function` variant and dispatched
/// through the enum with a plain `match`, so implementations stay
/// statically sized and the trait never needs to be object safe. Leaves
/// and parameters do not implement it: they are supplied, not computed,
/// and the enum's dispatch handles them directly. The rules are pure:
/// operands arrive as a positional slice of payload references gathered
/// by the engine, results are returned rather than written, and no rule
/// ever sees the tape, a `ValueId`, or a run buffer. Gradient
/// accumulation — the multivariate chain rule — is the engine's job,
/// stated once in `Evaluation::backward`.
pub(crate) trait Operation<Data: Differentiable> {
    /// Computes this node's payload from its operands' payloads.
    fn forward(&self, operands: &[&Data]) -> Data;

    /// Computes one cotangent per operand, given this node's computed
    /// `output` payload and its own `gradient`.
    fn backward(&self, operands: &[&Data], output: &Data, gradient: &Data) -> Cotangents<Data>;
}

/// Splits the positional operand list of a unary operation.
///
/// It is generic so the same helper serves operand payloads and operand
/// shapes.
///
/// # Panics
/// Panics if `operands` does not hold exactly one entry; recording
/// checks every node against its `arity`, so a mismatch is an engine
/// bug.
pub(crate) fn unary<T>(operands: &[T]) -> &T {
    let [operand] = operands else {
        panic!("unary operation expects exactly one operand");
    };
    operand
}

/// Splits the positional operand list of a binary operation.
///
/// It is generic so the same helper serves operand payloads and operand
/// shapes.
///
/// # Panics
/// Panics if `operands` does not hold exactly two entries; recording
/// checks every node against its `arity`, so a mismatch is an engine
/// bug.
pub(crate) fn binary<T>(operands: &[T]) -> (&T, &T) {
    let [left, right] = operands else {
        panic!("binary operation expects exactly two operands");
    };
    (left, right)
}
