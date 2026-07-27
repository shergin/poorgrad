use crate::engine::ValueId;
use crate::{Differentiable, Shape};

use super::Operation;

/// The quotient of two values.
///
/// The derivative with respect to the left operand is `1 / right`; with
/// respect to the right operand it is `-left / right^2`, which equals
/// `-output / right`, so `backward` reuses the node's own output the way
/// `Tanh` does. Gradients inherit the payload's division semantics near
/// zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Div {
    pub(crate) left: ValueId,
    pub(crate) right: ValueId,
}

impl Div {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.left);
        visitor(self.right);
    }

    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape {
        let left = shape_of(self.left);
        let right = shape_of(self.right);
        assert_eq!(left, right, "division requires operands of equal shapes");
        left
    }
}

impl<Data: Differentiable> Operation<Data> for Div {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.left.index()].clone() / values[self.right.index()].clone()
    }

    fn backward(&self, values: &[Data], output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let left = self.left.index();
        let right = self.right.index();
        gradients[left] = gradients[left].clone() + gradient.clone() / values[right].clone();
        gradients[right] =
            gradients[right].clone() + -(gradient.clone() * output.clone() / values[right].clone());
    }
}
