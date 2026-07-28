use static_assertions::assert_impl_all;

use crate::{Differentiable, Network, Shape, Symbol, Tensorial, Value};

use super::{Activation, Layer};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Mlp<f64>: Send, Sync);

/// A multilayer perceptron: dense layers chained by topology.
///
/// A topology such as `[3, 4, 4, 1]` defines three layers that map a
/// `[batch, 3]` input to a `[batch, 1]` output. Hidden layers use
/// [`Activation::Tanh`], and the output layer uses [`Activation::Identity`].
/// The contained layers retain parameter [`Symbol`]s, allowing
/// [`Mlp::express`] to record the network in each compatible generation.
#[derive(Debug, Clone)]
pub struct Mlp<Data> {
    layers: Vec<Layer<Data>>,
}

impl<Data: Differentiable> Mlp<Data> {
    /// Allocates the perceptron's layers on `network` and returns it.
    ///
    /// `sizes` lists the value widths from the input width to the
    /// output width. `initializer` produces the initial payload for
    /// each parameter from its shape — `[inputs, outputs]` weights and
    /// `[outputs]` biases, layer by layer. The initializer is responsible for
    /// returning payloads with the requested shapes, and callers control
    /// details such as fan-in scaling, randomness, and symmetry breaking.
    ///
    /// # Panics
    /// Panics if `sizes` has fewer than two entries. It also propagates
    /// [`Layer::new`] validation failures if initialized weights and biases do
    /// not form valid layer parameter shapes.
    pub fn new(
        network: &Network<Data>,
        sizes: &[usize],
        mut initializer: impl FnMut(&Shape) -> Data,
    ) -> Self {
        assert!(
            sizes.len() >= 2,
            "an MLP topology needs an input and an output width"
        );
        let layers = sizes
            .windows(2)
            .enumerate()
            .map(|(index, pair)| {
                let activation = if index == sizes.len() - 2 {
                    Activation::Identity
                } else {
                    Activation::Tanh
                };
                let weights = initializer(&Shape::new([pair[0], pair[1]]));
                let bias = initializer(&Shape::new([pair[1]]));
                Layer::new(network, weights, bias, activation)
            })
            .collect();
        Self { layers }
    }

    /// Returns the symbols of all parameters, layer by layer: each
    /// layer's weights, then its bias.
    pub fn parameters(&self) -> impl Iterator<Item = Symbol> + '_ {
        self.layers.iter().flat_map(Layer::parameters)
    }
}

impl<Data: Tensorial> Mlp<Data> {
    /// Records the perceptron's expression over the `[batch, inputs]`
    /// value `input` on `network` and returns the `[batch, outputs]`
    /// output value.
    ///
    /// # Panics
    /// Panics if the parameters or `input` are not allocated on `network`, or
    /// if `input` and the initialized layer shapes are incompatible.
    pub fn express<'network>(
        &self,
        network: &'network Network<Data>,
        input: Value<'network, Data>,
    ) -> Value<'network, Data> {
        self.layers
            .iter()
            .fold(input, |value, layer| layer.express(network, value))
    }
}

#[cfg(test)]
#[path = "tests/mlp_tests.rs"]
mod tests;
