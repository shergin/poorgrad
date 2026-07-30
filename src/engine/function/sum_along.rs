use crate::engine::ValueId;
use crate::{Shape, Tensorial};

use super::{Operation, unary};

/// The sum of a payload along one named axis.
///
/// It is the axis-wise form of `Sum`, and `BroadcastAlong` is its
/// adjoint: the operand's gradient is the incoming gradient repeated
/// back along the reduced axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SumAlong {
    pub(crate) axis: usize,
}

impl SumAlong {
    /// Infers the shape of the result: the operand's shape with the
    /// axis removed.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        unary(operands).without_axis(self.axis)
    }
}

impl<Data: Tensorial> Operation<Data> for SumAlong {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        values[unary(operands).index()].sum_along(self.axis)
    }

    fn backward(
        &self,
        operands: &[ValueId],
        values: &[Data],
        _output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        let operand = unary(operands).index();
        let contribution = gradient.broadcast_along(self.axis, &values[operand]);
        gradients[operand] = gradients[operand].clone() + contribution;
    }
}
