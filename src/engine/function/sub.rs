use crate::engine::ValueId;
use crate::{Differentiable, Shape};

use super::{Operation, binary};

/// The difference of two values, with operands `[left, right]`.
///
/// The derivative with respect to the left operand is one and with
/// respect to the right operand minus one, so `backward` routes the
/// incoming gradient onward and negated respectively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sub;

impl Sub {
    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (left, right) = binary(operands);
        assert_eq!(left, right, "subtraction requires operands of equal shapes");
        left.clone()
    }
}

impl<Data: Differentiable> Operation<Data> for Sub {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        let (&left, &right) = binary(operands);
        values[left.index()].clone() - values[right.index()].clone()
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
        gradients[right] = gradients[right].clone() + -gradient.clone();
    }
}
