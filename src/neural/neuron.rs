use std::iter;
use std::marker::PhantomData;

use static_assertions::assert_impl_all;

use crate::{Differentiable, Elementary, Symbol, Tape, Value};

use super::Activation;

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Neuron<f64>: Send, Sync);

/// A learnable affine unit followed by an [`Activation`].
///
/// A neuron computes `activation(bias + sum(weight_i * input_i))`. Its
/// parameters are allocated on a [`Tape`] at construction and retained as
/// [`Symbol`]s, so [`Neuron::express`] can resolve them and record the same
/// expression whenever the tape reopens.
#[derive(Debug, Clone)]
pub struct Neuron<Data> {
    weights: Vec<Symbol>,
    bias: Symbol,
    activation: Activation,
    _marker: PhantomData<Data>,
}

impl<Data: Differentiable> Neuron<Data> {
    /// Allocates a neuron's parameters on `tape` and returns the
    /// neuron.
    ///
    /// `initializer` produces the initial payload of each parameter: the
    /// weights first, one per input, then the bias.
    pub fn new(
        tape: &Tape<Data>,
        inputs: usize,
        activation: Activation,
        mut initializer: impl FnMut() -> Data,
    ) -> Self {
        let weights = (0..inputs)
            .map(|_| tape.parameter(initializer()).symbol())
            .collect();
        let bias = tape.parameter(initializer()).symbol();
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
    /// Records the neuron's expression over `inputs` on `tape` and
    /// returns its output value.
    ///
    /// # Panics
    /// Panics if the number of inputs differs from the number of weights,
    /// if an input or parameter is not allocated on `tape`, or if their
    /// payload shapes are incompatible for elementwise arithmetic.
    pub fn express<'tape>(
        &self,
        tape: &'tape Tape<Data>,
        inputs: &[Value<'tape, Data>],
    ) -> Value<'tape, Data> {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "neuron expects a different number of inputs"
        );
        let mut sum = tape.resolve(self.bias);
        for (weight, input) in self.weights.iter().zip(inputs) {
            let weight = tape.resolve(*weight);
            sum = sum + weight * *input;
        }
        self.activation.express(sum)
    }
}

#[cfg(test)]
#[path = "tests/neuron_tests.rs"]
mod tests;
