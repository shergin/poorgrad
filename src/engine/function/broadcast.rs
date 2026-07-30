use crate::engine::ValueId;
use crate::{Shape, Tensorial};

use super::{Operation, binary};

/// The explicit broadcast of a single-value payload across another
/// value's shape, with operands `[operand, like]`.
///
/// It is the only shape-changing expansion in the engine, and it is
/// deliberately explicit: the target shape comes from a named reference
/// value, never from an alignment rule. Broadcasting and summation are
/// adjoint, so the operand's gradient is the sum of the incoming
/// gradient, restored to the operand's own single-value shape; the
/// reference contributes only its shape and receives no gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Broadcast;

impl Broadcast {
    /// Infers the shape of the result: the reference's shape, reachable
    /// only from a single-value operand.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (operand, like) = binary(operands);
        assert_eq!(
            operand.volume(),
            1,
            "broadcast requires a single-element operand, got {operand}"
        );
        like.clone()
    }
}

impl<Data: Tensorial> Operation<Data> for Broadcast {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        let (&operand, &like) = binary(operands);
        values[operand.index()].broadcast_like(&values[like.index()])
    }

    fn backward(
        &self,
        operands: &[ValueId],
        values: &[Data],
        _output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        let (&operand, _) = binary(operands);
        let operand = operand.index();
        // The reduced gradient is rank 0, but the operand may be any
        // volume-1 shape (such as `[1]`); broadcasting the sum back to
        // the operand's own shape keeps the accumulation well-formed.
        let contribution = gradient.sum().broadcast_like(&values[operand]);
        gradients[operand] = gradients[operand].clone() + contribution;
    }
}
