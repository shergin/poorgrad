use static_assertions::assert_impl_all;

use super::{Activation, Differentiable, Elementary, Network, Neuron, Symbol, Value};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Layer<f64>: Send, Sync);

/// A dense layer: a row of neurons sharing the same inputs.
///
/// It computes one output per neuron, with every neuron seeing every
/// input (a fully connected layer); layers chain by feeding one layer's
/// outputs to the next as inputs. Like `Neuron` it is detached: its
/// parameters are allocated on a `Network` at construction but held as
/// `Symbol`s, so the layer survives generations and records its
/// expression against whichever generation it is given.
#[derive(Debug, Clone)]
pub struct Layer<Data> {
    neurons: Vec<Neuron<Data>>,
}

impl<Data: Differentiable> Layer<Data> {
    /// Allocates a layer of `outputs` neurons with `inputs` weights each
    /// on `network` and returns the layer.
    ///
    /// All neurons share the same `activation`. `initializer` produces the
    /// initial payload of every parameter, neuron by neuron, the weights
    /// first and the bias last.
    pub fn new(
        network: &Network<Data>,
        inputs: usize,
        outputs: usize,
        activation: Activation,
        mut initializer: impl FnMut() -> Data,
    ) -> Self {
        let neurons = (0..outputs)
            .map(|_| Neuron::new(network, inputs, activation, &mut initializer))
            .collect();
        Self { neurons }
    }

    /// Returns the symbols of all parameters, neuron by neuron.
    pub fn parameters(&self) -> impl Iterator<Item = Symbol> + '_ {
        self.neurons.iter().flat_map(Neuron::parameters)
    }
}

impl<Data: Elementary> Layer<Data> {
    /// Records the layer's expression over `inputs` on `network` and
    /// returns one output value per neuron.
    ///
    /// # Panics
    /// Panics if the number of inputs differs from the neurons' number of
    /// weights, or if the layer's parameters are not allocated on
    /// `network`.
    pub fn express<'network>(
        &self,
        network: &'network Network<Data>,
        inputs: &[Value<'network, Data>],
    ) -> Vec<Value<'network, Data>> {
        self.neurons
            .iter()
            .map(|neuron| neuron.express(network, inputs))
            .collect()
    }
}

#[cfg(test)]
#[path = "tests/layer_tests.rs"]
mod tests;
