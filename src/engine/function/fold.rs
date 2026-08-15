use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, Reads, unary};

/// The `(count, size)` window pair at `axis`, `axis + 1` folded back
/// onto an axis of `extent`: [`Unfold`](super::Unfold)'s adjoint,
/// with the output-centric deterministic semantics of
/// [`Tensorial::fold`].
///
/// It exists as an opcode because `unfold`'s derivative rule speaks
/// `fold`, so recorded gradients of windowed values need it on the
/// tape — the adjoint pairing the convolution design specified as
/// opcode-ready. The gradient of the operand is the incoming gradient
/// unfolded by the same parameters: the pair is self-adjoint in both
/// directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Fold {
    pub(crate) axis: usize,
    pub(crate) size: usize,
    pub(crate) step: usize,
    pub(crate) dilation: usize,
    pub(crate) extent: usize,
}

impl Fold {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads no payloads: the cotangent unfolds by the parameters.
    pub(crate) fn reads(&self) -> Reads {
        Reads::NOTHING
    }

    /// Infers the result shape: the `(count, size)` pair at `axis`
    /// replaced by `extent`, requiring the operand pair to be exactly
    /// what unfolding an `extent` axis by these parameters produces.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert!(
            self.axis + 1 < operand.rank(),
            "fold needs a (count, size) pair at axis {}, but {operand} has no pair there",
            self.axis
        );
        assert!(self.size > 0, "fold windows must hold at least one element");
        assert!(self.step > 0, "fold step must be positive");
        assert!(self.dilation > 0, "fold dilation must be positive");
        let span = self
            .dilation
            .checked_mul(self.size - 1)
            .and_then(|reach| reach.checked_add(1))
            .expect("fold window span overflows `usize`");
        assert!(
            span <= self.extent,
            "fold window span {span} exceeds the target extent {}",
            self.extent
        );
        let count = (self.extent - span) / self.step + 1;
        assert_eq!(
            operand.axes()[self.axis],
            count,
            "fold expects {count} windows at axis {} for extent {}, got {}",
            self.axis,
            self.extent,
            operand.axes()[self.axis]
        );
        assert_eq!(
            operand.axes()[self.axis + 1],
            self.size,
            "fold expects window size {} at axis {}, got {}",
            self.size,
            self.axis + 1,
            operand.axes()[self.axis + 1]
        );
        let mut folded: Vec<usize> = operand.axes().to_vec();
        folded[self.axis] = self.extent;
        folded.remove(self.axis + 1);
        Shape::new(folded)
    }
}

impl<Data: Tensorial> Operation<Data> for Fold {
    fn forward(&self, operands: &[&Data]) -> Data {
        unary(operands).fold(self.axis, self.size, self.step, self.dilation, self.extent)
    }

    fn backward(&self, _operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        smallvec![Some(gradient.unfold(
            self.axis,
            self.size,
            self.step,
            self.dilation
        ))]
    }
}

#[cfg(test)]
#[path = "tests/fold_tests.rs"]
mod tests;
