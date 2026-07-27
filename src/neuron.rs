use std::iter;
use std::marker::PhantomData;

use static_assertions::assert_impl_all;

use super::{Differentiable, Elementary, Network, Symbol, Value};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Neuron<f64>: Send, Sync);

/// The nonlinearity applied to a neuron's weighted sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// No nonlinearity: the neuron stays affine, as befits output layers
    /// of regressions.
    Identity,
    /// The hyperbolic tangent, squashing the output into `(-1, 1)`.
    Tanh,
}

/// A single neuron: weights, a bias, and an activation.
///
/// It is the smallest learnable building block, computing
/// `activation(weights . inputs + bias)`. Its parameters are allocated on
/// a `Network` at construction but held as `Symbol`s, so the neuron itself
/// is detached like any name: it survives generations and training steps,
/// and `express` records its expression against whichever generation it is
/// given.
#[derive(Debug, Clone)]
pub struct Neuron<Data> {
    weights: Vec<Symbol>,
    bias: Symbol,
    activation: Activation,
    payload: PhantomData<Data>,
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
            payload: PhantomData,
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
    /// or if the neuron's parameters are not allocated on `network`.
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
        }
    }
}

#[cfg(test)]
#[path = "tests/neuron_tests.rs"]
mod tests;
