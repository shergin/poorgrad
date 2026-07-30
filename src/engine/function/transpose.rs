use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, unary};

/// The transposition of a value.
///
/// Transposition is linear and self-adjoint in shape: the gradient of the
/// operand is the transposed incoming gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Transpose;

impl Transpose {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Infers the shape of the result: the operand's axes reversed.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert!(
            operand.rank() <= 2,
            "transpose supports rank 2 at most, got {operand}"
        );
        Shape::new(operand.axes().iter().rev().copied())
    }
}

impl<Data: Tensorial> Operation<Data> for Transpose {
    fn forward(&self, operands: &[&Data]) -> Data {
        unary(operands).transposed()
    }

    fn backward(&self, _operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        smallvec![Some(gradient.transposed())]
    }
}
