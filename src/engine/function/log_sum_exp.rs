use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, Reads, unary};

/// The log-sum-exp of a payload along one named axis:
/// `ln(sum(exp(x)))`, the softmax family's normalizer and a smooth
/// maximum; like `SumAlong`, the reduced axis is removed.
///
/// It is a fused primitive for the same reason as `LogSoftmax`: the
/// stable forward shifts by the axis maximum, which no composition of
/// recorded operations can express. The former composition
/// (`x - log_softmax(x)`, read from one arbitrary lane) returned `inf`
/// whenever that lane's log-probability underflowed to `-inf` for
/// finite extreme logits; the fused form is finite for every finite
/// operand. The gradient is the softmax, recovered from the operand
/// and the node's own output as `exp(operand - output)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LogSumExp {
    pub(crate) axis: usize,
}

impl LogSumExp {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads its operand and its own output to recover the softmax.
    pub(crate) fn reads(&self) -> Reads {
        Reads {
            operands: [true, false],
            output: true,
        }
    }

    /// Infers the shape of the result: the operand's shape with the
    /// axis removed, checked against its rank.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert!(
            self.axis < operand.rank(),
            "axis {} is out of rank for {operand}",
            self.axis
        );
        operand.without_axis(self.axis)
    }
}

impl<Data: Tensorial> Operation<Data> for LogSumExp {
    fn forward(&self, operands: &[&Data]) -> Data {
        let &operand = unary(operands);
        // Shifting by the axis maximum keeps every exponent at or below
        // zero: the sum lands between one and the axis extent, its
        // logarithm between zero and `ln(extent)`, so `peak + ln(sum)`
        // is finite for every finite operand — even where the shifted
        // difference itself underflows to `-inf`.
        let peak = operand.max_along(self.axis);
        let shifted = operand.clone() - peak.broadcast_along(self.axis, operand);
        peak + shifted.exp().sum_along(self.axis).ln()
    }

    fn backward(&self, operands: &[&Data], output: &Data, gradient: &Data) -> Cotangents<Data> {
        let &operand = unary(operands);
        // The derivative of log-sum-exp is the softmax; the shift
        // cancels analytically, and `operand - output` reconstructs the
        // stable log-probabilities directly.
        let probabilities = (operand.clone() - output.broadcast_along(self.axis, operand)).exp();
        smallvec![Some(
            gradient.broadcast_along(self.axis, operand) * probabilities
        )]
    }
}

#[cfg(test)]
#[path = "tests/log_sum_exp_tests.rs"]
mod tests;
