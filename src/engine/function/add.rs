use crate::engine::ValueId;
use crate::{Differentiable, Shape};

use super::{Operation, binary};

/// The sum of two values, with operands `[left, right]`.
///
/// The derivative with respect to each operand is one, so `backward`
/// routes the incoming gradient to both operands unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Add;

impl Add {
    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (left, right) = binary(operands);
        assert_eq!(left, right, "addition requires operands of equal shapes");
        left.clone()
    }
}

impl<Data: Differentiable> Operation<Data> for Add {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        let (&left, &right) = binary(operands);
        values[left.index()].clone() + values[right.index()].clone()
    }

    fn backward(
        &self,
        operands: &[ValueId],
        _values: &[Data],
        _output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        let (&left, &right) = binary(operands);
        let left = left.index();
        let right = right.index();
        gradients[left] = gradients[left].clone() + gradient.clone();
        gradients[right] = gradients[right].clone() + gradient.clone();
    }
}
