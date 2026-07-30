use crate::engine::ValueId;
use crate::{Differentiable, Shape};

use super::{Operation, binary};

/// The quotient of two values, with operands `[left, right]`.
///
/// The derivative with respect to the left operand is `1 / right`; with
/// respect to the right operand it is `-left / right^2`, which equals
/// `-output / right`, so `backward` reuses the node's own output the way
/// `Tanh` does. Gradients inherit the payload's division semantics near
/// zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Div;

impl Div {
    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (left, right) = binary(operands);
        assert_eq!(left, right, "division requires operands of equal shapes");
        left.clone()
    }
}

impl<Data: Differentiable> Operation<Data> for Div {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        let (&left, &right) = binary(operands);
        values[left.index()].clone() / values[right.index()].clone()
    }

    fn backward(
        &self,
        operands: &[ValueId],
        values: &[Data],
        output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        let (&left, &right) = binary(operands);
        let left = left.index();
        let right = right.index();
        gradients[left] = gradients[left].clone() + gradient.clone() / values[right].clone();
        gradients[right] =
            gradients[right].clone() + -(gradient.clone() * output.clone() / values[right].clone());
    }
}
