use std::marker::PhantomData;

use static_assertions::assert_impl_all;

use crate::{Differentiable, Network, Symbol, Tensorial, Value};

use super::{Module, Visitor};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(LayerNorm<f64>: Send, Sync);

/// A layer-normalization layer over `[batch, features]` values (Ba,
/// Kiros & Hinton, 2016): every sample is standardized by its own
/// feature statistics and passed through the learned per-feature affine
/// `scale * normalized + shift`.
///
/// For sample `i` and feature `j`:
///
/// ```text
/// m_i = mean_j(input[i, j])
/// v_i = mean_j((input[i, j] - m_i)^2)
/// output[i, j] = (input[i, j] - m_i) / sqrt(v_i + epsilon)
///                * scale[j] + shift[j]
/// ```
///
/// It is [`BatchNorm`](super::BatchNorm) with the statistics taken along
/// the feature axis instead of the batch axis, which removes all of the
/// batch coupling: samples normalize independently, so there are no
/// running estimates and no training/inference split — one recorded
/// expression serves both.
///
/// Parameters are stored as [`Symbol`]s and resolved when the expression
/// is recorded in a compatible [`Network`] generation, like
/// [`Layer`](super::Layer).
#[derive(Debug, Clone)]
pub struct LayerNorm<Data> {
    scale: Symbol,
    shift: Symbol,
    epsilon: Symbol,
    _marker: PhantomData<Data>,
}

impl<Data: Differentiable> LayerNorm<Data> {
    /// Allocates the layer's parameters on `network` from their initial
    /// payloads and returns the layer.
    ///
    /// `scale` and `shift` are rank-1 `[features]` parameters (the
    /// standard initialization is ones and zeros), and `epsilon` is a
    /// single-value constant broadcast across the per-sample variances
    /// before the square root so a sample with no spread stays finite.
    /// Callers own initialization; the layer records whatever it is
    /// given.
    ///
    /// # Panics
    /// Panics if `scale` is not rank 1, `shift` is not shaped like
    /// `scale`, or `epsilon` holds more than one value.
    pub fn new(network: &Network<Data>, scale: Data, shift: Data, epsilon: Data) -> Self {
        let scale_shape = scale.shape();
        let shift_shape = shift.shape();
        let epsilon_shape = epsilon.shape();
        assert_eq!(
            scale_shape.rank(),
            1,
            "layer-norm scale must be rank 1, got {scale_shape}"
        );
        assert_eq!(
            shift_shape, scale_shape,
            "layer-norm shift {shift_shape} must be shaped like the scale {scale_shape}"
        );
        assert_eq!(
            epsilon_shape.volume(),
            1,
            "layer-norm epsilon must hold a single value, got {epsilon_shape}"
        );
        Self {
            scale: network.parameter(scale).symbol(),
            shift: network.parameter(shift).symbol(),
            epsilon: network.leaf(epsilon).symbol(),
            _marker: PhantomData,
        }
    }

    /// Returns the symbols of the layer's parameters: the scale, then
    /// the shift.
    pub fn parameters(&self) -> impl Iterator<Item = Symbol> + '_ {
        [self.scale, self.shift].into_iter()
    }
}

impl<Data: Tensorial> LayerNorm<Data> {
    /// Records the layer's expression over the `[batch, features]`
    /// value `input` on `network` and returns the `[batch, features]`
    /// output value.
    ///
    /// # Panics
    /// Panics if the layer's parameters or `input` are not allocated on
    /// `network`, or if `input` is not a rank-2 `[batch, features]`
    /// value agreeing with the parameters on the feature count.
    pub fn express<'network>(
        &self,
        network: &'network Network<Data>,
        input: Value<'network, Data>,
    ) -> Value<'network, Data> {
        let scale = network.resolve(self.scale);
        let shift = network.resolve(self.shift);
        let epsilon = network.resolve(self.epsilon);
        let input_shape = input.shape();
        let scale_shape = scale.shape();
        assert_eq!(
            input_shape.rank(),
            2,
            "layer-norm input must be rank 2 [batch, features], got {input_shape}"
        );
        assert_eq!(
            input_shape.axes()[1],
            scale_shape.axes()[0],
            "layer-norm input {input_shape} and scale {scale_shape} disagree on features"
        );
        // The per-sample statistics `m_i` and biased `v_i`, both
        // `[batch]`-shaped, repeated back across the feature axis.
        let mean = input.mean_along(1);
        let centered = input - mean.broadcast_along(1, input);
        let variance = (centered * centered).mean_along(1);
        // `(input - m_i) / sqrt(v_i + epsilon)`; the epsilon expands
        // in-graph because the variance's `[batch]` shape is a
        // per-expression fact the single-value leaf cannot know.
        let deviation = (variance + epsilon.broadcast_like(variance)).sqrt();
        let normalized = centered / deviation.broadcast_along(1, input);
        // The learned per-feature affine `scale[j] * n + shift[j]`.
        normalized * scale.broadcast_along(0, input) + shift.broadcast_along(0, input)
    }
}

#[cfg(test)]
#[path = "tests/layer_norm_tests.rs"]
mod tests;

impl<Data: Tensorial> Module<Data> for LayerNorm<Data> {
    fn express<'network>(
        &self,
        network: &'network Network<Data>,
        input: Value<'network, Data>,
    ) -> Value<'network, Data> {
        LayerNorm::express(self, network, input)
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.parameter("scale", self.scale);
        visitor.parameter("shift", self.shift);
    }
}
