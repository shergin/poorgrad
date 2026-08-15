use smallvec::smallvec;

use crate::{Elementary, Shape};

use super::{Cotangents, Operation, Reads, unary};

/// The hyperbolic tangent of a value.
///
/// The derivative is `1 - tanh(x)^2`: one minus the square of the node's
/// own output, so `backward` reuses the computed output instead of
/// recomputing the transcendental.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Tanh;

impl Tanh {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads its own output: the derivative is `1 - output^2`.
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

impl<Data: Elementary> Operation<Data> for Tanh {
    fn forward(&self, operands: &[&Data]) -> Data {
        unary(operands).tanh()
    }

    fn backward(&self, _operands: &[&Data], output: &Data, gradient: &Data) -> Cotangents<Data> {
        let derivative = output.one_like() - output.clone() * output.clone();
        smallvec![Some(gradient.clone() * derivative)]
    }
}
