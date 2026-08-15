use smallvec::smallvec;

use crate::{Differentiable, Shape};

use super::{Cotangents, Operation, Reads, binary};

/// The difference of two values, with operands `[left, right]`.
///
/// The derivative with respect to the left operand is one and with
/// respect to the right operand minus one, so `backward` hands the
/// incoming gradient onward and negated respectively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sub;

impl Sub {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Returns the read set of the derivative rule below.
    /// It reads no payloads: the cotangents are the gradient and its negation.
    pub(crate) fn reads(&self) -> Reads {
        Reads::NOTHING
    }

    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (left, right) = binary(operands);
        assert_eq!(left, right, "subtraction requires operands of equal shapes");
        left.clone()
    }
}

impl<Data: Differentiable> Operation<Data> for Sub {
    fn forward(&self, operands: &[&Data]) -> Data {
        let (&left, &right) = binary(operands);
        left.clone() - right.clone()
    }

    fn backward(&self, _operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        smallvec![Some(gradient.clone()), Some(-gradient.clone())]
    }
}
