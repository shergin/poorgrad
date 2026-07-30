use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, binary};

/// The explicit broadcast of a single-value payload across another
/// value's shape, with operands `[operand, like]`.
///
/// It is the only shape-changing expansion in the engine, and it is
/// deliberately explicit: the target shape comes from a named reference
/// value, never from an alignment rule. Broadcasting and summation are
/// adjoint, so the operand's gradient is the sum of the incoming
/// gradient, restored to the operand's own single-value shape; the
/// reference contributes only its shape, which is what its `None`
/// cotangent states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Broadcast;

impl Broadcast {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

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
    fn forward(&self, operands: &[&Data]) -> Data {
        let (&operand, &like) = binary(operands);
        operand.broadcast_like(like)
    }

    fn backward(&self, operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        let (&operand, _) = binary(operands);
        // The reduced gradient is rank 0, but the operand may be any
        // volume-1 shape (such as `[1]`); broadcasting the sum back to
        // the operand's own shape keeps the accumulation well-formed.
        smallvec![Some(gradient.sum().broadcast_like(operand)), None]
    }
}
