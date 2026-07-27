use crate::engine::ValueId;
use crate::{Shape, Tensorial};

use super::Operation;

/// The explicit repetition of a payload along one named axis of a
/// reference value's shape.
///
/// It is the axis-wise form of `Broadcast`, and `SumAlong` is its
/// adjoint: the operand's gradient is the incoming gradient summed
/// along the repeated axis. The axis is always named, so no shape
/// alignment is ever inferred; the reference contributes only its
/// shape and receives no gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BroadcastAlong {
    pub(crate) operand: ValueId,
    pub(crate) like: ValueId,
    pub(crate) axis: usize,
}

impl BroadcastAlong {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.operand);
        visitor(self.like);
    }

    /// Infers the shape of the result: the reference's shape, reachable
    /// only from an operand shaped like the reference without the axis.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape {
        let operand = shape_of(self.operand);
        let like = shape_of(self.like);
        assert_eq!(
            operand,
            like.without_axis(self.axis),
            "broadcast along axis {} of {like} requires the remaining shape",
            self.axis
        );
        like
    }
}

impl<Data: Tensorial> Operation<Data> for BroadcastAlong {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.operand.index()].broadcast_along(self.axis, &values[self.like.index()])
    }

    fn backward(&self, _values: &[Data], _output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let operand = self.operand.index();
        let contribution = gradient.sum_along(self.axis);
        gradients[operand] = gradients[operand].clone() + contribution;
    }
}
