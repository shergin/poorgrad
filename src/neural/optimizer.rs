use crate::{Differentiable, Gradients, Parameters, Tensorial};

/// A training-step strategy: how gradients and the current parameter
/// state become the next state.
///
/// It is the loop-land analogue of what [`Activation`](super::Activation)
/// is to a layer — a uniform slot the training loop can hand any
/// strategy — kept an *open* trait rather than a closed enum on
/// purpose: custom optimizers are ordinary implementations with the
/// same standing as the built-in ones, holding whatever state they
/// need as plain fields ([`Field`](crate::Field) algebra was designed
/// as the state carrier). The trait is object-safe, so a comparison
/// loop can iterate `&mut dyn Optimizer` implementations side by side.
///
/// The learning rate is a per-step argument, not optimizer state:
/// schedules stay caller-owned loop arithmetic, visible on the page
/// like every other training decision.
pub trait Optimizer<Data: Tensorial> {
    /// Returns `parameters` stepped by `gradients` at `learning_rate`,
    /// updating this optimizer's own state.
    ///
    /// The gradients may come from [`Run::backward`](crate::Run::backward)
    /// or from [`Run::recorded_gradients`](crate::Run::recorded_gradients)
    /// over a compiled gradient plan — a field is a field.
    fn step(
        &mut self,
        parameters: &Parameters<Data>,
        gradients: &Gradients<Data>,
        learning_rate: &Data,
    ) -> Parameters<Data>;
}

/// Plain stochastic gradient descent: the strategy every example's
/// hand-written loop applies, as the trait's simplest implementation —
/// stateless, so the struct is a unit, and every richer optimizer is
/// this plus state.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sgd;

impl<Data: Tensorial> Optimizer<Data> for Sgd {
    fn step(
        &mut self,
        parameters: &Parameters<Data>,
        gradients: &Gradients<Data>,
        learning_rate: &Data,
    ) -> Parameters<Data> {
        parameters.step(gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        })
    }
}

/// Asserts that an optimizer hyperparameter holds exactly one value,
/// the contract every scalar factor spreads from.
pub(super) fn assert_single_value<Data: Differentiable>(payload: &Data, name: &str) {
    assert_eq!(
        payload.shape().volume(),
        1,
        "optimizer {name} must hold a single value"
    );
}

#[cfg(test)]
#[path = "tests/optimizer_tests.rs"]
mod tests;
