use crate::engine::ValueId;
use crate::{Shape, Tensorial};

use super::Operation;

/// The matrix product of two values.
///
/// The gradient routes through the transposed operands:
/// `d(A . B)/dA = gradient . B^T` and `d(A . B)/dB = A^T . gradient`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MatMul {
    pub(crate) left: ValueId,
    pub(crate) right: ValueId,
}

impl MatMul {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.left);
        visitor(self.right);
    }

    /// Infers the shape `[m, n]` of a `[m, k] . [k, n]` product.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape {
        let left = shape_of(self.left);
        let right = shape_of(self.right);
        assert_eq!(
            left.rank(),
            2,
            "matmul requires rank-2 operands, got {left}"
        );
        assert_eq!(
            right.rank(),
            2,
            "matmul requires rank-2 operands, got {right}"
        );
        assert_eq!(
            left.axes()[1],
            right.axes()[0],
            "matmul cannot multiply {left} by {right}"
        );
        Shape::new([left.axes()[0], right.axes()[1]])
    }
}

impl<Data: Tensorial> Operation<Data> for MatMul {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.left.index()].matmul(&values[self.right.index()])
    }

    fn backward(&self, values: &[Data], _output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let left = self.left.index();
        let right = self.right.index();
        gradients[left] = gradients[left].clone() + gradient.matmul(&values[right].transposed());
        gradients[right] = gradients[right].clone() + values[left].transposed().matmul(gradient);
    }
}
