use crate::engine::ValueId;
use crate::{Shape, Tensorial};

use super::{Operation, binary};

/// The explicit repetition of a payload along one named axis of a
/// reference value's shape, with operands `[operand, like]`.
///
/// It is the axis-wise form of `Broadcast`, and `SumAlong` is its
/// adjoint: the operand's gradient is the incoming gradient summed
/// along the repeated axis. The axis is always named, so no shape
/// alignment is ever inferred; the reference contributes only its
/// shape and receives no gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BroadcastAlong {
    pub(crate) axis: usize,
}

impl BroadcastAlong {
    /// Infers the shape of the result: the reference's shape, reachable
    /// only from an operand shaped like the reference without the axis.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (operand, like) = binary(operands);
        assert_eq!(
            operand,
            &like.without_axis(self.axis),
            "broadcast along axis {} of {like} requires the remaining shape",
            self.axis
        );
        like.clone()
    }
}

impl<Data: Tensorial> Operation<Data> for BroadcastAlong {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        let (&operand, &like) = binary(operands);
        values[operand.index()].broadcast_along(self.axis, &values[like.index()])
    }

    fn backward(
        &self,
        operands: &[ValueId],
        _values: &[Data],
        _output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        let (&operand, _) = binary(operands);
        let operand = operand.index();
        let contribution = gradient.sum_along(self.axis);
        gradients[operand] = gradients[operand].clone() + contribution;
    }
}
