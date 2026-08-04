use smallvec::smallvec;

use crate::{Differentiable, Shape};

use super::{Cotangents, Operation, Retention, binary};

/// The product of two values, with operands `[left, right]`.
///
/// The derivative with respect to each operand is the other operand, so
/// `backward` scales the incoming gradient by the opposite side's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Mul;

impl Mul {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Returns the retention of the derivative rule below.
    /// It reads both operand payloads: each side's cotangent scales by the other.
    pub(crate) fn retains(&self) -> Retention {
        Retention {
            operands: [true, true],
            output: false,
        }
    }

    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (left, right) = binary(operands);
        assert_eq!(
            left, right,
            "multiplication requires operands of equal shapes"
        );
        left.clone()
    }
}

impl<Data: Differentiable> Operation<Data> for Mul {
    fn forward(&self, operands: &[&Data]) -> Data {
        let (&left, &right) = binary(operands);
        left.clone() * right.clone()
    }

    fn backward(&self, operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        let (&left, &right) = binary(operands);
        smallvec![
            Some(gradient.clone() * right.clone()),
            Some(gradient.clone() * left.clone()),
        ]
    }
}
