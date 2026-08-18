use std::marker::PhantomData;

use static_assertions::assert_impl_all;

use crate::{Differentiable, Symbol, Tape, Tensorial, Value};

use super::{Module, Visitor};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(RmsNorm<f64>: Send, Sync);

/// A root-mean-square normalization layer over `[batch, features]`
/// values (Zhang & Sennrich, 2019): every sample is divided by the root
/// mean square of its own features and scaled per feature.
///
/// For sample `i` and feature `j`:
///
/// ```text
/// r_i = sqrt(mean_j(input[i, j]^2) + epsilon)
/// output[i, j] = input[i, j] / r_i * scale[j]
/// ```
///
/// It is [`LayerNorm`](super::LayerNorm) without the centering and the
/// shift — re-scaling alone, on the observation that the re-centering
/// half contributes little — which drops one statistic and two graph
/// passes. Like `LayerNorm` it is stateless: samples normalize
/// independently, so there are no running estimates and no
/// training/inference split — one recorded expression serves both.
///
/// Parameters are stored as [`Symbol`]s and resolved when the expression
/// is recorded on the family's [`Tape`], like
/// [`Layer`](super::Layer).
#[derive(Debug, Clone)]
pub struct RmsNorm<Data> {
    scale: Symbol,
    epsilon: Symbol,
    _marker: PhantomData<Data>,
}

impl<Data: Differentiable> RmsNorm<Data> {
    /// Allocates the layer's parameter on `tape` from its initial
    /// payload and returns the layer.
    ///
    /// `scale` is a rank-1 `[features]` parameter (the standard
    /// initialization is ones), and `epsilon` is a single-value
    /// constant added under the square root so an all-zero sample stays
    /// finite. Callers own initialization; the layer records whatever
    /// it is given.
    ///
    /// # Panics
    /// Panics if `scale` is not rank 1 or `epsilon` holds more than one
    /// value.
    pub fn new(tape: &Tape<Data>, scale: Data, epsilon: Data) -> Self {
        let scale_shape = scale.shape();
        let epsilon_shape = epsilon.shape();
        assert_eq!(
            scale_shape.rank(),
            1,
            "rms-norm scale must be rank 1, got {scale_shape}"
        );
        assert_eq!(
            epsilon_shape.volume(),
            1,
            "rms-norm epsilon must hold a single value, got {epsilon_shape}"
        );
        Self {
            scale: tape.parameter(scale).symbol(),
            epsilon: tape.leaf(epsilon).symbol(),
            _marker: PhantomData,
        }
    }

    /// Returns the symbols of the layer's parameters: the scale alone.
    pub fn parameters(&self) -> impl Iterator<Item = Symbol> + '_ {
        [self.scale].into_iter()
    }
}

impl<Data: Tensorial> RmsNorm<Data> {
    /// Records the layer's expression over the `[batch, features]`
    /// value `input` on `tape` and returns the `[batch, features]`
    /// output value.
    ///
    /// # Panics
    /// Panics if the layer's parameter or `input` are not allocated on
    /// `tape`, or if `input` is not a rank-2 `[batch, features]`
    /// value agreeing with the scale on the feature count.
    pub fn express<'tape>(
        &self,
        tape: &'tape Tape<Data>,
        input: Value<'tape, Data>,
    ) -> Value<'tape, Data> {
        let scale = tape.resolve(self.scale);
        let epsilon = tape.resolve(self.epsilon);
        let input_shape = input.shape();
        let scale_shape = scale.shape();
        assert_eq!(
            input_shape.rank(),
            2,
            "rms-norm input must be rank 2 [batch, features], got {input_shape}"
        );
        assert_eq!(
            input_shape.axes()[1],
            scale_shape.axes()[0],
            "rms-norm input {input_shape} and scale {scale_shape} disagree on features"
        );
        // The per-sample mean power `mean_j(input^2)`, `[batch]`-shaped,
        // stabilized under the root: `r_i = sqrt(power_i + epsilon)`.
        let power = (input * input).mean_along(1);
        let root = (power + epsilon.broadcast_like(power)).sqrt();
        // `input / r_i * scale[j]`.
        input / root.broadcast_along(1, input) * scale.broadcast_along(0, input)
    }
}

#[cfg(test)]
#[path = "tests/rms_norm_tests.rs"]
mod tests;

impl<Data: Tensorial> Module<Data> for RmsNorm<Data> {
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Data>,
        input: Value<'tape, Data>,
    ) -> Value<'tape, Data> {
        RmsNorm::express(self, tape, input)
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.parameter("scale", self.scale);
    }
}
