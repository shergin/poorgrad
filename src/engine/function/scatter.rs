use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, Retention, binary};

/// The rows of a gradient scatter-added into `rows` rows by a one-hot
/// selection, with operands `[gradient, selection]`:
/// [`Gather`](super::Gather)'s adjoint, [`Tensorial::scatter`] as a
/// node.
///
/// It exists as an opcode because `gather`'s derivative rule speaks
/// `scatter`, so recorded gradients of embedding lookups need it on
/// the tape. The compact form is deliberate: composing the same map
/// as `matmul(transpose(selection), gradient)` is exact mathematics
/// but densifies the one-hot at embedding scale. The gradient of the
/// scattered operand is the incoming gradient gathered back by the
/// same selection — the pair is adjoint in both directions — and the
/// selection is data, not a differentiable dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Scatter {
    pub(crate) rows: usize,
}

impl Scatter {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Returns the retention of the derivative rule below.
    /// It reads the selection payload (the gather needs its indices);
    /// the gradient contributes nothing.
    pub(crate) fn retains(&self) -> Retention {
        Retention {
            operands: [false, true],
            output: false,
        }
    }

    /// Infers the result shape `[rows, ...gradient.shape[1..]]`,
    /// requiring the selection to be rank 2 with one row per gradient
    /// row and a vocabulary of exactly `rows`.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (gradient, selection) = binary(operands);
        assert!(
            gradient.rank() >= 1,
            "scatter needs a gradient with a leading selection axis, got {gradient}"
        );
        assert_eq!(
            selection.rank(),
            2,
            "scatter selection must be rank 2 [count, vocab], got {selection}"
        );
        assert_eq!(
            selection.axes()[0],
            gradient.axes()[0],
            "scatter gradient rows {} disagree with the selection count {}",
            gradient.axes()[0],
            selection.axes()[0]
        );
        assert_eq!(
            selection.axes()[1],
            self.rows,
            "scatter rows {} disagree with the selection vocabulary {}",
            self.rows,
            selection.axes()[1]
        );
        Shape::new(std::iter::once(self.rows).chain(gradient.axes()[1..].iter().copied()))
    }
}

impl<Data: Tensorial> Operation<Data> for Scatter {
    fn forward(&self, operands: &[&Data]) -> Data {
        let (&gradient, &selection) = binary(operands);
        gradient.scatter(selection, self.rows)
    }

    fn backward(&self, operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        let (_, &selection) = binary(operands);
        smallvec![Some(gradient.gather(selection)), None]
    }
}

#[cfg(test)]
#[path = "tests/scatter_tests.rs"]
mod tests;
