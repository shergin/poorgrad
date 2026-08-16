use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, Reads, binary};

/// The matrix product of two values, with operands `[left, right]`.
///
/// Operands of rank above two multiply batched: the trailing two axes
/// contract as the plain product and every leading axis is a batch
/// axis, required identical on both operands — no broadcast batching,
/// per the design's minimality (`notes/batched-matmul.md`).
///
/// The gradient routes through the transposed operands:
/// `d(A . B)/dA = gradient . B^T` and `d(A . B)/dB = A^T . gradient`,
/// where the batched transpose swaps the trailing two axes through
/// `permute`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MatMul;

impl MatMul {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Returns the read set of the derivative rule below.
    /// It reads both operand payloads for the transposed products.
    pub(crate) fn reads(&self) -> Reads {
        Reads {
            operands: [true, true],
            output: false,
        }
    }

    /// Infers the shape of a `[b..., m, k] . [b..., k, n]` product:
    /// `[b..., m, n]`, with the batch prefix `b...` (empty for the
    /// plain rank-2 product) required identical on both operands.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (left, right) = binary(operands);
        assert!(
            left.rank() >= 2,
            "matmul requires rank-2 or higher operands, got {left}"
        );
        assert_eq!(
            left.rank(),
            right.rank(),
            "matmul operands must agree in rank, got {left} and {right}"
        );
        let split = left.rank() - 2;
        assert_eq!(
            &left.axes()[..split],
            &right.axes()[..split],
            "matmul batch axes must agree, got {left} and {right}"
        );
        assert_eq!(
            left.axes()[split + 1],
            right.axes()[split],
            "matmul cannot multiply {left} by {right}"
        );
        Shape::new(
            left.axes()[..=split]
                .iter()
                .copied()
                .chain([right.axes()[split + 1]]),
        )
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
            Some(gradient.matmul(&swapped(right))),
            Some(swapped(left).matmul(gradient)),
        ]
    }
}

/// Returns `value` with its trailing two axes swapped: the dedicated
/// transpose for rank two and below (where scalar payloads answer
/// identity), `permute` for the batched ranks — so the adjoint closes
/// inside the existing op set.
fn swapped<Data: Tensorial>(value: &Data) -> Data {
    let rank = value.shape().rank();
    if rank <= 2 {
        return value.transpose();
    }
    let mut order: Vec<usize> = (0..rank).collect();
    order.swap(rank - 2, rank - 1);
    value.permute(&order)
}
