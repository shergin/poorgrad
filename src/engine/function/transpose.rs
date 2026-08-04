use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, Retention, unary};

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

    /// Returns the retention of the derivative rule below.
    /// It reads no payloads: the cotangent transposes back.
    pub(crate) fn retains(&self) -> Retention {
        Retention::NOTHING
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
        unary(operands).transpose()
    }

    fn backward(&self, _operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        smallvec![Some(gradient.transpose())]
    }
}
