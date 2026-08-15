use crate::{Network, Tensor};

use super::Activation;

/// Evaluates `activation` over `inputs` on a fresh network and returns
/// the outputs.
fn evaluated(activation: Activation, inputs: &[f64]) -> Vec<f64> {
    let network = Network::new();
    let value = network.leaf(Tensor::new([inputs.len()], inputs.to_vec()));
    let expressed = activation.express(value);
    let run = network.forward();
    run.of(expressed).to_vec()
}

/// Evaluates `activation`'s gradient over `inputs`.
fn gradient(activation: Activation, inputs: &[f64]) -> Vec<f64> {
    let network = Network::new();
    let value = network.parameter(Tensor::new([inputs.len()], inputs.to_vec()));
    let loss = activation.express(value).sum();
    let run = network.forward();
    run.backward(loss).of(value).to_vec()
}

#[test]
fn sigmoid_matches_the_logistic_function() {
    let outputs = evaluated(Activation::Sigmoid, &[0.0, 2.0, -2.0]);
    assert!((outputs[0] - 0.5).abs() < 1e-12);
    for (output, input) in outputs.iter().zip([0.0_f64, 2.0, -2.0]) {
        assert!((output - 1.0 / (1.0 + (-input).exp())).abs() < 1e-12);
    }
}

#[test]
fn sigmoid_saturates_finitely_at_the_extremes() {
    // The naive `1 / (1 + exp(-x))` overflows `exp` here; the tanh
    // composition saturates instead.
    let outputs = evaluated(Activation::Sigmoid, &[1.0e308, -1.0e308]);
    assert_eq!(outputs[0], 1.0);
    assert_eq!(outputs[1], 0.0);
}

#[test]
fn sigmoid_gradient_is_the_classic_product() {
    let inputs = [0.0_f64, 1.5, -0.75];
    let gradients = gradient(Activation::Sigmoid, &inputs);
    for (gradient, input) in gradients.iter().zip(inputs) {
        let sigma = 1.0 / (1.0 + (-input).exp());
        assert!((gradient - sigma * (1.0 - sigma)).abs() < 1e-12);
    }
}

#[test]
fn leaky_relu_keeps_a_hundredth_of_the_negative_side() {
    let outputs = evaluated(Activation::LeakyRelu, &[3.0, -2.0, 0.0]);
    assert_eq!(outputs, &[3.0, -0.02, 0.0]);
}

#[test]
fn leaky_relu_subgradient_at_zero_is_one() {
    let gradients = gradient(Activation::LeakyRelu, &[0.0, 2.0, -2.0]);
    assert_eq!(gradients, &[1.0, 1.0, 0.01]);
}

#[test]
fn elu_is_identity_above_and_saturating_below() {
    let outputs = evaluated(Activation::Elu, &[2.0, 0.0, -1.0, -1.0e308]);
    assert_eq!(outputs[0], 2.0);
    assert_eq!(outputs[1], 0.0);
    assert!((outputs[2] - ((-1.0_f64).exp() - 1.0)).abs() < 1e-12);
    // The clamped exponent keeps the extreme finite: the curve
    // saturates at minus one instead of overflowing.
    assert_eq!(outputs[3], -1.0);
}

#[test]
fn elu_subgradient_at_zero_is_one() {
    // ELU with unit scale is differentiable at zero; the maximum
    // spelling's left-biased tie keeps the composition's subgradient
    // at exactly one rather than double-counting the branches.
    let gradients = gradient(Activation::Elu, &[0.0, 1.0, -1.0]);
    assert_eq!(gradients[0], 1.0);
    assert_eq!(gradients[1], 1.0);
    assert!((gradients[2] - (-1.0_f64).exp()).abs() < 1e-12);
}

#[test]
fn compositions_differentiate_as_tape_bitwise() {
    // Facade compositions are made of closed operations, so recorded
    // gradients agree with the engine's for every new variant.
    for activation in [Activation::Sigmoid, Activation::LeakyRelu, Activation::Elu] {
        let network = Network::new();
        let value = network.parameter(Tensor::new([3], [0.8_f64, -1.3, 0.0]));
        let loss = activation.express(value).sum();
        let recorded = network.differentiate(loss.symbol(), [value.symbol()]);
        let run = network.forward();
        let engine = run.backward(network.resolve(loss.symbol()));
        for (recorded, computed) in run
            .of(network.resolve(recorded[0]))
            .to_vec()
            .iter()
            .zip(engine.of(value).to_vec())
        {
            assert_eq!(recorded.to_bits(), computed.to_bits());
        }
    }
}
