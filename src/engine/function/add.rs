use smallvec::smallvec;

use crate::{Differentiable, Shape};

use super::{Cotangents, Operation, binary};

/// The sum of two values, with operands `[left, right]`.
///
/// The derivative with respect to each operand is one, so `backward`
/// hands the incoming gradient to both operands unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Add;

impl Add {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (left, right) = binary(operands);
        assert_eq!(left, right, "addition requires operands of equal shapes");
        left.clone()
    }
}

impl<Data: Differentiable> Operation<Data> for Add {
    fn forward(&self, operands: &[&Data]) -> Data {
        let (&left, &right) = binary(operands);
        left.clone() + right.clone()
    }

    fn backward(&self, _operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        smallvec![Some(gradient.clone()), Some(gradient.clone())]
    }
}

#[cfg(test)]
#[path = "tests/add_tests.rs"]
mod tests;
