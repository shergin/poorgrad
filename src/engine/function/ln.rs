use crate::engine::ValueId;
use crate::{Elementary, Shape};

use super::{Operation, unary};

/// The natural logarithm of a value.
///
/// The derivative of `ln(x)` is `1 / x`, so `backward` divides the
/// incoming gradient by the operand's value. Gradients inherit the
/// payload's logarithm and division semantics outside the positive
/// domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Ln;

impl Ln {
    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        unary(operands).clone()
    }
}

impl<Data: Elementary> Operation<Data> for Ln {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        values[unary(operands).index()].ln()
    }

    fn backward(
        &self,
        operands: &[ValueId],
        values: &[Data],
        _output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        let operand = unary(operands).index();
        gradients[operand] =
            gradients[operand].clone() + gradient.clone() / values[operand].clone();
    }
}
