use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, Reads, unary};

/// The log-softmax of a payload along one named axis:
/// `x - ln(sum(exp(x)))`, the logarithm of the softmax probabilities.
///
/// It is a fused primitive rather than a composition because the stable
/// forward must shift by the axis maximum before exponentiating, and no
/// composition of recorded operations can express that shift without a
/// differentiable `max`. The gradient is `g - softmax * sum(g)` along the
/// axis, recovering the probabilities from the node's own output as
/// `exp(output)` — the shift cancels analytically and never appears in
/// the backward rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LogSoftmax {
    pub(crate) axis: usize,
}

impl LogSoftmax {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads its own output to recover the probabilities.
    pub(crate) fn reads(&self) -> Reads {
        Reads {
            operands: [false, false],
            output: true,
        }
    }

    /// Infers the shape of the result: the operand's shape, with the
    /// axis checked against its rank.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert!(
            self.axis < operand.rank(),
            "axis {} is out of rank for {operand}",
            self.axis
        );
        operand.clone()
    }
}

impl<Data: Tensorial> Operation<Data> for LogSoftmax {
    fn forward(&self, operands: &[&Data]) -> Data {
        let &operand = unary(operands);
        // Shifting by the axis maximum keeps every exponent at or below
        // zero, so the sum cannot overflow; the shift cancels in the
        // final subtraction, leaving the result stable (not exact: the
        // shifted rounding differs from the unshifted ideal, and a
        // difference beyond the representable range still underflows to
        // `-inf` — the mathematically faithful log-probability).
        let peak = operand
            .max_along(self.axis)
            .broadcast_along(self.axis, operand);
        let shifted = operand.clone() - peak;
        let normalizer = shifted.exp().sum_along(self.axis).ln();
        shifted.clone() - normalizer.broadcast_along(self.axis, &shifted)
    }

    fn backward(&self, _operands: &[&Data], output: &Data, gradient: &Data) -> Cotangents<Data> {
        let total = gradient
            .sum_along(self.axis)
            .broadcast_along(self.axis, gradient);
        smallvec![Some(gradient.clone() - output.exp() * total)]
    }
}

#[cfg(test)]
#[path = "tests/log_softmax_tests.rs"]
mod tests;
