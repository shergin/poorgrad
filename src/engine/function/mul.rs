use crate::engine::ValueId;
use crate::{Differentiable, Shape};

use super::{Operation, binary};

/// The product of two values, with operands `[left, right]`.
///
/// The derivative with respect to each operand is the other operand, so
/// `backward` scales the incoming gradient by the opposite side's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Mul;

impl Mul {
    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (left, right) = binary(operands);
        assert_eq!(
            left, right,
            "multiplication requires operands of equal shapes"
        );
        left.clone()
    }
}

impl<Data: Differentiable> Operation<Data> for Mul {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        let (&left, &right) = binary(operands);
        values[left.index()].clone() * values[right.index()].clone()
    }

    fn backward(
        &self,
        operands: &[ValueId],
        values: &[Data],
        _output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        let (&left, &right) = binary(operands);
        let left = left.index();
        let right = right.index();
        gradients[left] = gradients[left].clone() + gradient.clone() * values[right].clone();
        gradients[right] = gradients[right].clone() + gradient.clone() * values[left].clone();
    }
}
