use crate::{Elementary, Tape, Tensorial, Value};

use super::Module;

/// The nonlinearity applied to a neural building block's affine output.
///
/// Both variants are dedicated graph operations recorded as one node.
/// Anything else — a sigmoid, a leaky slope, an ELU scale, a GELU —
/// stays caller territory, composed from the same public surface the
/// way GPT-2's example composes its GELU; the composed variants this
/// enum once carried were retired when no consumer materialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Applies the hyperbolic tangent elementwise.
    Tanh,
    /// Applies the rectified linear unit elementwise.
    Relu,
}

impl Activation {
    /// Returns this activation's initialization gain: the factor by
    /// which the nonlinearity shrinks the variance of a unit-variance
    /// signal, compensated at initialization as
    /// `deviation = gain / sqrt(fan_in)` — the general form behind
    /// the named classics, served by
    /// [`init::scaled`](super::init::scaled).
    ///
    /// The values are the standard ones: `Tanh` uses the
    /// conventional `5/3`, and `Relu` halves the signal's variance,
    /// compensated by `sqrt(2)` (He et al., 2015).
    pub fn gain(self) -> f64 {
        match self {
            Activation::Tanh => 5.0 / 3.0,
            Activation::Relu => 2.0_f64.sqrt(),
        }
    }

    /// Records this activation's expression over `value` and returns
    /// the result: one node per variant.
    pub fn express<'tape, Data: Elementary>(self, value: Value<'tape, Data>) -> Value<'tape, Data> {
        match self {
            Activation::Tanh => value.tanh(),
            Activation::Relu => value.relu(),
        }
    }
}

#[cfg(test)]
#[path = "tests/activation_tests.rs"]
mod tests;

impl<Data: Tensorial> Module<Data> for Activation {
    /// A stateless stage: the network is unused, and the default
    /// no-op `visit` stands.
    fn express<'tape>(
        &self,
        _tape: &'tape Tape<Data>,
        input: Value<'tape, Data>,
    ) -> Value<'tape, Data> {
        Activation::express(*self, input)
    }
}
