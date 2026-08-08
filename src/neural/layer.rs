use std::marker::PhantomData;

use static_assertions::assert_impl_all;

use crate::{Differentiable, Network, Symbol, Tensorial, Value};

use super::Activation;

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Layer<f64>: Send, Sync);

/// A dense tensor layer computing `activation(input.matmul(weights) + bias)`.
///
/// The weights are one `[inputs, outputs]` parameter and the bias is one
/// `[outputs]` parameter. The bias is broadcast explicitly across the batch
/// axis, so expressing the layer records a small, fixed number of graph nodes
/// regardless of parameter count. Parameters are stored as [`Symbol`]s and
/// resolved when [`Layer::express`] records the layer in a compatible
/// [`Network`] generation.
#[derive(Debug, Clone)]
pub struct Layer<Data> {
    weights: Symbol,
    bias: Symbol,
    activation: Activation,
    _marker: PhantomData<Data>,
}

impl<Data: Differentiable> Layer<Data> {
    /// Allocates the layer's parameters on `network` from their initial
    /// payloads and returns the layer.
    ///
    /// The shapes are taken from the payloads: `weights` must be a
    /// rank-2 `[inputs, outputs]` payload and `bias` a rank-1
    /// `[outputs]` payload agreeing on `outputs`. Callers own
    /// initialization (fan-in scaling, randomness); the layer records
    /// whatever it is given.
    ///
    /// # Panics
    /// Panics if `weights` is not rank 2, `bias` is not rank 1, or the
    /// two disagree on the number of outputs.
    pub fn new(network: &Network<Data>, weights: Data, bias: Data, activation: Activation) -> Self {
        let weights_shape = weights.shape();
        let bias_shape = bias.shape();
        assert_eq!(
            weights_shape.rank(),
            2,
            "layer weights must be rank 2, got {weights_shape}"
        );
        assert_eq!(
            bias_shape.rank(),
            1,
            "layer bias must be rank 1, got {bias_shape}"
        );
        assert_eq!(
            weights_shape.axes()[1],
            bias_shape.axes()[0],
            "layer weights {weights_shape} and bias {bias_shape} disagree on outputs"
        );
        Self {
            weights: network.parameter(weights).symbol(),
            bias: network.parameter(bias).symbol(),
            activation,
            _marker: PhantomData,
        }
    }

    /// Returns the symbols of the layer's parameters: the weights, then
    /// the bias.
    pub fn parameters(&self) -> impl Iterator<Item = Symbol> + '_ {
        [self.weights, self.bias].into_iter()
    }
}

impl<Data: Tensorial> Layer<Data> {
    /// Records the layer's expression over the `[batch, inputs]` value
    /// `input` on `network` and returns the `[batch, outputs]` output
    /// value.
    ///
    /// # Panics
    /// Panics if the layer's parameters or `input` are not allocated on
    /// `network`, or if `input` and the weights are not compatible rank-2
    /// matrices.
    pub fn express<'network>(
        &self,
        network: &'network Network<Data>,
        input: Value<'network, Data>,
    ) -> Value<'network, Data> {
        let weights = network.resolve(self.weights);
        let bias = network.resolve(self.bias);
        let product = input.matmul(weights);
        // The bias is repeated across the batch axis; its gradient sums
        // back along the same axis, one contribution per sample.
        let shifted = product + bias.broadcast_along(0, product);
        self.activation.express(shifted)
    }
}

#[cfg(test)]
#[path = "tests/layer_tests.rs"]
mod tests;
