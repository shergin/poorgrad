use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, binary};

/// The matrix product of two values, with operands `[left, right]`.
///
/// The gradient routes through the transposed operands:
/// `d(A . B)/dA = gradient . B^T` and `d(A . B)/dB = A^T . gradient`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MatMul;

impl MatMul {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Infers the shape `[m, n]` of a `[m, k] . [k, n]` product.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (left, right) = binary(operands);
        assert_eq!(
            left.rank(),
            2,
            "matmul requires rank-2 operands, got {left}"
        );
        assert_eq!(
            right.rank(),
            2,
            "matmul requires rank-2 operands, got {right}"
        );
        assert_eq!(
            left.axes()[1],
            right.axes()[0],
            "matmul cannot multiply {left} by {right}"
        );
        Shape::new([left.axes()[0], right.axes()[1]])
    }
}

impl<Data: Tensorial> Operation<Data> for MatMul {
    fn forward(&self, operands: &[&Data]) -> Data {
        let (&left, &right) = binary(operands);
        left.matmul(right)
    }

    fn backward(&self, operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        let (&left, &right) = binary(operands);
        smallvec![
            Some(gradient.matmul(&right.transpose())),
            Some(left.transpose().matmul(gradient)),
        ]
    }
}
