use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, Reads, unary};

/// A reshape of a value to a new shape of the same volume.
///
/// Reshaping preserves logical row-major order, so it is a bijection on
/// elements: the gradient of the operand is the incoming gradient reshaped
/// back to the operand's own shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reshape {
    pub(crate) shape: Shape,
}

impl Reshape {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads its operand for shape only, which a placeholder answers.
    pub(crate) fn reads(&self) -> Reads {
        Reads::NOTHING
    }

    /// Infers the result shape: the requested shape, which must match the
    /// operand's volume.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert_eq!(
            operand.volume(),
            self.shape.volume(),
            "reshape from {operand} to {} changes the number of elements",
            self.shape
        );
        self.shape.clone()
    }
}

impl<Data: Tensorial> Operation<Data> for Reshape {
    fn forward(&self, operands: &[&Data]) -> Data {
        unary(operands).reshape(self.shape.clone())
    }

    fn backward(&self, operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        let &operand = unary(operands);
        smallvec![Some(gradient.reshape(operand.shape()))]
    }
}
