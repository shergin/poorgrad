use crate::{Elementary, Network, Tensorial, Value};

use super::Module;

/// The nonlinearity applied to a neural building block's affine output.
///
/// Only `Tanh` and `Relu` are dedicated graph operations; every other
/// variant records a short composition over the existing op set, so its
/// gradient is the chain rule with no dedicated backward rule, and each
/// spelling is chosen to stay finite for every finite input. Constants
/// are minted as integer [`counted`](crate::Differentiable::counted)
/// ratios per the settled literal decision — activations that need an
/// arbitrary float constant (a custom leaky slope, an ELU scale, a
/// GELU) stay caller territory, composed from the same public surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Leaves the affine output unchanged.
    Identity,
    /// Applies the hyperbolic tangent elementwise.
    Tanh,
    /// Applies the rectified linear unit elementwise.
    Relu,
    /// Applies the logistic sigmoid elementwise, composed as
    /// `(tanh(x / 2) + 1) / 2`: the naive `1 / (1 + exp(-x))`
    /// overflows for finite extreme inputs, while the fused `tanh`
    /// saturates — stability by inheritance, like `softmax` over
    /// `log_softmax`.
    Sigmoid,
    /// Applies the leaky rectified linear unit elementwise with the
    /// conventional slope of `1/100`, composed as
    /// `maximum(x, x / 100)`; the tie at zero routes the gradient to
    /// the left operand, so the subgradient there is one, matching
    /// `Relu`.
    LeakyRelu,
    /// Applies the exponential linear unit elementwise with the
    /// conventional scale of one, composed as
    /// `maximum(x, exp(-relu(-x)) - 1)`: the exponent is clamped to
    /// at most zero, so it cannot overflow, the negative branch wins
    /// exactly where `exp(x) - 1 > x` (everywhere below zero), and
    /// the tie at zero keeps the subgradient at one.
    Elu,
}

impl Activation {
    /// Returns this activation's initialization gain: the factor by
    /// which the nonlinearity shrinks the variance of a unit-variance
    /// signal, compensated at initialization as
    /// `deviation = gain / sqrt(fan_in)` — the general form behind
    /// the named classics, served by
    /// [`init::scaled`](super::init::scaled).
    ///
    /// The values are the standard ones: `Identity` and `Sigmoid`
    /// pass variance through near the origin (gain one); `Tanh` uses
    /// the conventional `5/3`; `Relu` halves the signal's variance,
    /// compensated by `sqrt(2)` (He et al., 2015); `LeakyRelu`
    /// generalizes it to `sqrt(2 / (1 + slope^2))` with this
    /// variant's `1/100` slope; and `Elu` shares `Relu`'s gain,
    /// which is how its paper trains (Clevert et al., 2015, uses He
    /// initialization).
    pub fn gain(self) -> f64 {
        match self {
            Activation::Identity | Activation::Sigmoid => 1.0,
            Activation::Tanh => 5.0 / 3.0,
            Activation::Relu | Activation::Elu => 2.0_f64.sqrt(),
            Activation::LeakyRelu => (2.0 / (1.0 + 0.01_f64 * 0.01)).sqrt(),
        }
    }

    /// Records this activation's expression over `value` and returns
    /// the result: one node for the dedicated operations, a short
    /// composition for the rest.
    pub fn express<'network, Data: Elementary>(
        self,
        value: Value<'network, Data>,
    ) -> Value<'network, Data> {
        match self {
            Activation::Identity => value,
            Activation::Tanh => value.tanh(),
            Activation::Relu => value.relu(),
            Activation::Sigmoid => {
                let halved = value / Data::counted(value.shape(), 2);
                (halved.tanh() + Data::counted(value.shape(), 1)) / Data::counted(value.shape(), 2)
            }
            Activation::LeakyRelu => value.maximum(value / Data::counted(value.shape(), 100)),
            Activation::Elu => {
                value.maximum((-(-value).relu()).exp() - Data::counted(value.shape(), 1))
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/activation_tests.rs"]
mod tests;

impl<Data: Tensorial> Module<Data> for Activation {
    /// A stateless stage: the network is unused, and the default
    /// no-op `visit` stands.
    fn express<'network>(
        &self,
        _network: &'network Network<Data>,
        input: Value<'network, Data>,
    ) -> Value<'network, Data> {
        Activation::express(*self, input)
    }
}
