use std::marker::PhantomData;

use static_assertions::assert_impl_all;

use crate::{Differentiable, Symbol, Tape, Tensorial, Value};

use super::{Module, Visitor};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Linear<f64>: Send, Sync);

/// The affine transform `input.matmul(weights) + bias`, unfused: an
/// activation is its own composition stage, which unlocks the
/// orderings a bundled activation forbids (pre-norm blocks,
/// activation-before-projection).
///
/// The weights are one `[inputs, outputs]` parameter and the bias is
/// one `[outputs]` parameter. The bias is broadcast explicitly across
/// the batch axis, so expressing the transform records a small, fixed
/// number of graph nodes regardless of parameter count. Parameters
/// are stored as [`Symbol`]s and resolved when the module records on
/// the family's [`Tape`].
#[derive(Debug, Clone)]
pub struct Linear<Data> {
    weights: Symbol,
    bias: Symbol,
    _marker: PhantomData<Data>,
}

impl<Data: Differentiable> Linear<Data> {
    /// Allocates the transform's parameters on `tape` from their
    /// initial payloads and returns the module.
    ///
    /// The shapes are taken from the payloads: `weights` must be a
    /// rank-2 `[inputs, outputs]` payload and `bias` a rank-1
    /// `[outputs]` payload agreeing on `outputs`. Callers own
    /// initialization (fan-in scaling, randomness); the module records
    /// whatever it is given.
    ///
    /// # Panics
    /// Panics if `weights` is not rank 2, `bias` is not rank 1, or the
    /// two disagree on the number of outputs.
    pub fn new(tape: &Tape<Data>, weights: Data, bias: Data) -> Self {
        let weights_shape = weights.shape();
        let bias_shape = bias.shape();
        assert_eq!(
            weights_shape.rank(),
            2,
            "linear weights must be rank 2, got {weights_shape}"
        );
        assert_eq!(
            bias_shape.rank(),
            1,
            "linear bias must be rank 1, got {bias_shape}"
        );
        assert_eq!(
            weights_shape.axes()[1],
            bias_shape.axes()[0],
            "linear weights {weights_shape} and bias {bias_shape} disagree on outputs"
        );
        Self {
            weights: tape.parameter(weights).symbol(),
            bias: tape.parameter(bias).symbol(),
            _marker: PhantomData,
        }
    }

    /// Builds the module over existing parameter symbols: the weight
    /// tying constructor. The symbols may come from another module's
    /// typed accessors (an embedding's table tied into an output
    /// head); shape validation happens where the module expresses.
    pub fn from_symbols(weights: Symbol, bias: Symbol) -> Self {
        Self {
            weights,
            bias,
            _marker: PhantomData,
        }
    }

    /// Returns the symbol of the `[inputs, outputs]` weight matrix.
    pub fn weights(&self) -> Symbol {
        self.weights
    }

    /// Returns the symbol of the `[outputs]` bias vector.
    pub fn bias(&self) -> Symbol {
        self.bias
    }

    /// Returns the symbols of the transform's parameters: the weights,
    /// then the bias.
    pub fn parameters(&self) -> impl Iterator<Item = Symbol> + '_ {
        [self.weights, self.bias].into_iter()
    }
}

impl<Data: Tensorial> Module<Data> for Linear<Data> {
    /// Records the transform over the `[batch, inputs]` value `input`
    /// and returns the `[batch, outputs]` output value.
    ///
    /// # Panics
    /// Panics if the parameters or `input` are not allocated on
    /// `tape`, or if `input` and the weights are not compatible
    /// rank-2 matrices.
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Data>,
        input: Value<'tape, Data>,
    ) -> Value<'tape, Data> {
        let weights = tape.resolve(self.weights);
        let bias = tape.resolve(self.bias);
        let product = input.matmul(weights);
        // The bias is repeated across the batch axis; its gradient sums
        // back along the same axis, one contribution per sample.
        product + bias.broadcast_along(0, product)
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.parameter("weights", self.weights);
        visitor.parameter("bias", self.bias);
    }
}

#[cfg(test)]
#[path = "tests/linear_tests.rs"]
mod tests;
