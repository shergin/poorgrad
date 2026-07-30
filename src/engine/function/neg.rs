use crate::engine::ValueId;
use crate::{Differentiable, Shape};

use super::{Operation, unary};

/// The negation of a value.
///
/// The derivative with respect to the operand is minus one, so `backward`
/// routes the negated incoming gradient to the operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Neg;

impl Neg {
    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        unary(operands).clone()
    }
}

impl<Data: Differentiable> Operation<Data> for Neg {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        -values[unary(operands).index()].clone()
    }

    fn backward(
        &self,
        operands: &[ValueId],
        _values: &[Data],
        _output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        let operand = unary(operands).index();
        gradients[operand] = gradients[operand].clone() + -gradient.clone();
    }
}
