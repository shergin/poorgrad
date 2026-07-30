use std::iter;
use std::marker::PhantomData;

use static_assertions::assert_impl_all;

use crate::{Differentiable, Elementary, Network, Symbol, Value};

use super::Activation;

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Neuron<f64>: Send, Sync);

/// A learnable affine unit followed by an [`Activation`].
///
/// A neuron computes `activation(bias + sum(weight_i * input_i))`. Its
/// parameters are allocated on a [`Network`] at construction and retained as
/// [`Symbol`]s, so [`Neuron::express`] can resolve them and record the same
/// expression in each compatible network generation.
#[derive(Debug, Clone)]
pub struct Neuron<Data> {
    weights: Vec<Symbol>,
    bias: Symbol,
    activation: Activation,
    _marker: PhantomData<Data>,
}

impl<Data: Differentiable> Neuron<Data> {
    /// Allocates a neuron's parameters on `network` and returns the
    /// neuron.
    ///
    /// `initializer` produces the initial payload of each parameter: the
    /// weights first, one per input, then the bias.
    pub fn new(
        network: &Network<Data>,
        inputs: usize,
        activation: Activation,
        mut initializer: impl FnMut() -> Data,
    ) -> Self {
        let weights = (0..inputs)
            .map(|_| network.parameter(initializer()).symbol())
            .collect();
        let bias = network.parameter(initializer()).symbol();
        Self {
            weights,
            bias,
            activation,
            _marker: PhantomData,
        }
    }

    /// Returns the symbols of the neuron's parameters: the weights first,
    /// then the bias.
    pub fn parameters(&self) -> impl Iterator<Item = Symbol> + '_ {
        self.weights.iter().copied().chain(iter::once(self.bias))
    }
}

impl<Data: Elementary> Neuron<Data> {
    /// Records the neuron's expression over `inputs` on `network` and
    /// returns its output value.
    ///
    /// # Panics
    /// Panics if the number of inputs differs from the number of weights,
    /// if an input or parameter is not allocated on `network`, or if their
    /// payload shapes are incompatible for elementwise arithmetic.
    pub fn express<'network>(
        &self,
        network: &'network Network<Data>,
        inputs: &[Value<'network, Data>],
    ) -> Value<'network, Data> {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "neuron expects a different number of inputs"
        );
        let mut sum = network.resolve(self.bias);
        for (weight, input) in self.weights.iter().zip(inputs) {
            let weight = network.resolve(*weight);
            sum = sum + weight * *input;
        }
        match self.activation {
            Activation::Identity => sum,
            Activation::Tanh => sum.tanh(),
            Activation::Relu => sum.relu(),
        }
    }
}

#[cfg(test)]
#[path = "tests/neuron_tests.rs"]
mod tests;
