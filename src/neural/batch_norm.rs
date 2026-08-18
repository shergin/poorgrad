use std::marker::PhantomData;

use static_assertions::assert_impl_all;

use crate::{Differentiable, Symbol, Tape, Tensorial, Value};

use super::{Module, Visitor};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(BatchNorm<f64>: Send, Sync);

/// A batch-normalization layer over `[batch, features]` values (Ioffe &
/// Szegedy, 2015): every feature is standardized and passed through the
/// learned per-feature affine `scale * normalized + shift`.
///
/// For sample `i` and feature `j`, with the statistics taken along the
/// batch axis:
///
/// ```text
/// m_j = mean_i(input[i, j])
/// v_j = mean_i((input[i, j] - m_j)^2)
/// output[i, j] = (input[i, j] - m_j) / sqrt(v_j + epsilon)
///                * scale[j] + shift[j]
/// ```
///
/// The layer records two expressions, mirroring the unfed/fed run
/// split. [`BatchNorm::express`] normalizes by
/// the batch's own statistics — the training mode — and returns them
/// alongside the output, because the statistics are part of the layer's
/// contract: gradients flow through them into the input, and the caller
/// reads their evaluated payloads to maintain running estimates.
/// [`BatchNorm::express_with`] normalizes by supplied statistics — the
/// inference mode. The layer stores no running statistics itself: record
/// them as per-run inputs on the inference expression and keep their
/// exponential moving average in payload land with the training loop, so
/// the tape stays a pure record of the computation.
///
/// Parameters are stored as [`Symbol`]s and resolved when an expression
/// is recorded on the family's [`Tape`], like
/// [`Layer`](super::Layer).
#[derive(Debug, Clone)]
pub struct BatchNorm<Data> {
    scale: Symbol,
    shift: Symbol,
    epsilon: Symbol,
    _marker: PhantomData<Data>,
}

impl<Data: Differentiable> BatchNorm<Data> {
    /// Allocates the layer's parameters on `tape` from their initial
    /// payloads and returns the layer.
    ///
    /// `scale` and `shift` are rank-1 `[features]` parameters (the
    /// standard initialization is ones and zeros), and `epsilon` is a
    /// single-value constant broadcast across the variances before the
    /// square root so a feature with no spread stays finite. Callers own
    /// initialization; the layer records whatever it is given.
    ///
    /// # Panics
    /// Panics if `scale` is not rank 1, `shift` is not shaped like
    /// `scale`, or `epsilon` holds more than one value.
    pub fn new(tape: &Tape<Data>, scale: Data, shift: Data, epsilon: Data) -> Self {
        let scale_shape = scale.shape();
        let shift_shape = shift.shape();
        let epsilon_shape = epsilon.shape();
        assert_eq!(
            scale_shape.rank(),
            1,
            "batch-norm scale must be rank 1, got {scale_shape}"
        );
        assert_eq!(
            shift_shape, scale_shape,
            "batch-norm shift {shift_shape} must be shaped like the scale {scale_shape}"
        );
        assert_eq!(
            epsilon_shape.volume(),
            1,
            "batch-norm epsilon must hold a single value, got {epsilon_shape}"
        );
        Self {
            scale: tape.parameter(scale).symbol(),
            shift: tape.parameter(shift).symbol(),
            epsilon: tape.leaf(epsilon).symbol(),
            _marker: PhantomData,
        }
    }

    /// Returns the symbols of the layer's parameters: the scale, then
    /// the shift.
    pub fn parameters(&self) -> impl Iterator<Item = Symbol> + '_ {
        [self.scale, self.shift].into_iter()
    }
}

impl<Data: Tensorial> BatchNorm<Data> {
    /// Records the training-mode expression over the `[batch, features]`
    /// value `input` on `tape` — normalization by the batch's own
    /// mean and biased variance — and returns the output together with
    /// the statistic values it normalized by.
    ///
    /// # Panics
    /// Panics if the layer's parameters or `input` are not allocated on
    /// `tape`, or if `input` is not a rank-2 `[batch, features]`
    /// value agreeing with the parameters on the feature count.
    pub fn express<'tape>(
        &self,
        tape: &'tape Tape<Data>,
        input: Value<'tape, Data>,
    ) -> Normalization<'tape, Data> {
        let mean = input.mean_along(0);
        let centered = input - mean.broadcast_along(0, input);
        // The biased (population) variance, which normalization uses at
        // training time; an unbiased running estimate is the caller's
        // averaging policy, not the graph's.
        let variance = (centered * centered).mean_along(0);
        let output = self.normalize(tape, centered, variance);
        Normalization {
            output,
            mean,
            variance,
        }
    }

    /// Records the inference-mode expression over the `[batch, features]`
    /// value `input` on `tape`: normalization by the supplied
    /// `[features]` statistics instead of the batch's own.
    ///
    /// Record `mean` and `variance` as per-run inputs and feed the
    /// running estimates maintained during training, so one recorded
    /// expression serves every generation of the estimates.
    ///
    /// # Panics
    /// Panics if the values are not allocated on `tape`, `input` is
    /// not a rank-2 `[batch, features]` value agreeing with the
    /// parameters on the feature count, or the statistics are not
    /// `[features]` values.
    pub fn express_with<'tape>(
        &self,
        tape: &'tape Tape<Data>,
        input: Value<'tape, Data>,
        mean: Value<'tape, Data>,
        variance: Value<'tape, Data>,
    ) -> Value<'tape, Data> {
        let centered = input - mean.broadcast_along(0, input);
        self.normalize(tape, centered, variance)
    }

    /// Records the shared tail of both expressions: division by the
    /// epsilon-stabilized deviation and the learned affine.
    fn normalize<'tape>(
        &self,
        tape: &'tape Tape<Data>,
        centered: Value<'tape, Data>,
        variance: Value<'tape, Data>,
    ) -> Value<'tape, Data> {
        let scale = tape.resolve(self.scale);
        let shift = tape.resolve(self.shift);
        let epsilon = tape.resolve(self.epsilon);
        let centered_shape = centered.shape();
        let scale_shape = scale.shape();
        assert_eq!(
            centered_shape.rank(),
            2,
            "batch-norm input must be rank 2 [batch, features], got {centered_shape}"
        );
        assert_eq!(
            centered_shape.axes()[1],
            scale_shape.axes()[0],
            "batch-norm input {centered_shape} and scale {scale_shape} disagree on features"
        );
        // `(input - m_j) / sqrt(v_j + epsilon)`; the epsilon expands
        // in-graph to the variance's `[features]` shape, the family's
        // shared single-value epsilon contract.
        let deviation = (variance + epsilon.broadcast_like(variance)).sqrt();
        let normalized = centered / deviation.broadcast_along(0, centered);
        normalized * scale.broadcast_along(0, centered) + shift.broadcast_along(0, centered)
    }
}

/// A recorded batch-normalization expression: the output together with
/// the batch statistics it normalized by.
///
/// The statistics are ordinary computed values: read them from each
/// [`Run`](crate::Run) to maintain the running
/// estimates that [`BatchNorm::express_with`] consumes at inference.
#[derive(Debug)]
pub struct Normalization<'tape, Data> {
    /// The normalized, affine-transformed `[batch, features]` output.
    pub output: Value<'tape, Data>,
    /// The batch's per-feature `[features]` mean.
    pub mean: Value<'tape, Data>,
    /// The batch's per-feature `[features]` biased variance.
    pub variance: Value<'tape, Data>,
}

// Manual implementations avoid the `Data: Copy` bound a derive would
// add: the struct copies three proxies, never `Data`.
impl<Data> Clone for Normalization<'_, Data> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Data> Copy for Normalization<'_, Data> {}

#[cfg(test)]
#[path = "tests/batch_norm_tests.rs"]
mod tests;

/// The inference-mode module adapter of a [`BatchNorm`]:
/// normalization by fixed statistic values named at construction —
/// typically per-run inputs fed with the running estimates maintained
/// during training — so the mode is visible where the chain is
/// composed, never a hidden flag. Training mode is deliberately not a
/// module: it returns the batch statistics besides the output, and a
/// module must not hide values its caller needs.
pub struct BatchNormInference<Data> {
    norm: BatchNorm<Data>,
    mean: Symbol,
    variance: Symbol,
}

impl<Data: Differentiable> BatchNorm<Data> {
    /// Returns the inference-mode module normalizing by the
    /// `[features]` values named `mean` and `variance`.
    pub fn inference(&self, mean: Symbol, variance: Symbol) -> BatchNormInference<Data> {
        BatchNormInference {
            norm: self.clone(),
            mean,
            variance,
        }
    }
}

impl<Data: Tensorial> Module<Data> for BatchNormInference<Data> {
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Data>,
        input: Value<'tape, Data>,
    ) -> Value<'tape, Data> {
        let mean = tape.resolve(self.mean);
        let variance = tape.resolve(self.variance);
        self.norm.express_with(tape, input, mean, variance)
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.parameter("scale", self.norm.scale);
        visitor.parameter("shift", self.norm.shift);
    }
}
