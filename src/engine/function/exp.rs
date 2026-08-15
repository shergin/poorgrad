use smallvec::smallvec;

use crate::{Elementary, Shape};

use super::{Cotangents, Operation, Reads, unary};

/// The exponential of a value.
///
/// The derivative of `e^x` is `e^x` itself — the canonical case of
/// reusing the node's own output in `backward`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Exp;

impl Exp {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads its own output, which is its derivative.
    pub(crate) fn reads(&self) -> Reads {
        Reads {
            operands: [false, false],
            output: true,
        }
    }

    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        unary(operands).clone()
    }
}

impl<Data: Elementary> Operation<Data> for Exp {
    fn forward(&self, operands: &[&Data]) -> Data {
        unary(operands).exp()
    }

    fn backward(&self, _operands: &[&Data], output: &Data, gradient: &Data) -> Cotangents<Data> {
        smallvec![Some(gradient.clone() * output.clone())]
    }
}
