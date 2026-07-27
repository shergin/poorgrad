use static_assertions::assert_impl_all;

use crate::{Differentiable, Network, Shape, Symbol, Tensorial, Value};

use super::{Activation, Layer};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Mlp<f64>: Send, Sync);

/// A multilayer perceptron: dense layers chained by topology.
///
/// The topology lists the value widths, micrograd-style: `[3, 4, 4, 1]`
/// is three layers taking a `[batch, 3]` value to a `[batch, 1]` value.
/// Hidden layers squash with `Tanh`; the output layer stays affine
/// (`Identity`), as befits regression-style losses. Like its layers the
/// perceptron is detached: parameters live on the network, symbols in
/// the facade, and `express` records against whichever generation it is
/// given.
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
    /// `[outputs]` biases, layer by layer — so callers own
    /// initialization (fan-in scaling, randomness, symmetry breaking).
    ///
    /// # Panics
    /// Panics if `sizes` has fewer than two entries, or if
    /// `initializer` returns a payload whose shape differs from the one
    /// it was asked for.
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
    /// Panics if the parameters are not allocated on `network`, or if
    /// `input`'s shape does not match the topology.
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
