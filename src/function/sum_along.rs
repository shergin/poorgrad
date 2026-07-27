use crate::{Shape, Tensorial, ValueId};

use super::Operation;

/// The sum of a payload along one named axis.
///
/// It is the axis-wise form of `Sum`, and `BroadcastAlong` is its
/// adjoint: the operand's gradient is the incoming gradient repeated
/// back along the reduced axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SumAlong {
    pub(crate) operand: ValueId,
    pub(crate) axis: usize,
}

impl SumAlong {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.operand);
    }

    /// Infers the shape of the result: the operand's shape with the
    /// axis removed.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape {
        shape_of(self.operand).without_axis(self.axis)
    }
}

impl<Data: Tensorial> Operation<Data> for SumAlong {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.operand.index()].sum_along(self.axis)
    }

    fn backward(&self, values: &[Data], _output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let operand = self.operand.index();
        let contribution = gradient.broadcast_along(self.axis, &values[operand]);
        gradients[operand] = gradients[operand].clone() + contribution;
    }
}
