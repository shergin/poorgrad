use std::marker::PhantomData;

use static_assertions::assert_impl_all;

use super::{Activation, Differentiable, Network, Symbol, Tensorial, Value};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Layer<f64>: Send, Sync);

/// A dense layer at tensor granularity: `activation(x . w + b)`.
///
/// Its weights are one `[inputs, outputs]` parameter and its bias one
/// `[outputs]` parameter, so expressing the layer records a handful of
/// tensor nodes instead of one node per scalar weight; the bias meets
/// the batch matrix through the explicit axis broadcast. Like `Neuron`
/// it is detached: parameters are allocated on a `Network` at
/// construction but held as `Symbol`s, so the layer survives
/// generations and records its expression against whichever generation
/// it is given. Layers chain by feeding one layer's output batch to the
/// next.
#[derive(Debug, Clone)]
pub struct Layer<Data> {
    weights: Symbol,
    bias: Symbol,
    activation: Activation,
    payload: PhantomData<Data>,
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
            payload: PhantomData,
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
    /// Panics if the layer's parameters are not allocated on `network`,
    /// or if `input`'s shape does not multiply with the weights.
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
        match self.activation {
            Activation::Identity => shifted,
            Activation::Tanh => shifted.tanh(),
        }
    }
}

#[cfg(test)]
#[path = "tests/layer_tests.rs"]
mod tests;
