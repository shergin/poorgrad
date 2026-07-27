use crate::engine::ValueId;
use crate::{Differentiable, Shape};

use super::Operation;

/// The product of two values.
///
/// The derivative with respect to each operand is the other operand, so
/// `backward` scales the incoming gradient by the opposite side's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Mul {
    pub(crate) left: ValueId,
    pub(crate) right: ValueId,
}

impl Mul {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.left);
        visitor(self.right);
    }

    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape {
        let left = shape_of(self.left);
        let right = shape_of(self.right);
        assert_eq!(
            left, right,
            "multiplication requires operands of equal shapes"
        );
        left
    }
}

impl<Data: Differentiable> Operation<Data> for Mul {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.left.index()].clone() * values[self.right.index()].clone()
    }

    fn backward(&self, values: &[Data], _output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let left = self.left.index();
        let right = self.right.index();
        gradients[left] = gradients[left].clone() + gradient.clone() * values[right].clone();
        gradients[right] = gradients[right].clone() + gradient.clone() * values[left].clone();
    }
}
