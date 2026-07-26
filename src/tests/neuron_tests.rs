use crate::Network;

use super::{Activation, Neuron};

#[test]
fn new_allocates_weights_and_bias_as_parameters() {
    let network = Network::<f64>::new();
    let neuron = Neuron::new(&network, 3, Activation::Identity, || 0.0);
    assert_eq!(network.len(), 4);
    assert_eq!(neuron.parameters().count(), 4);
}

#[test]
fn express_records_the_affine_expression() {
    let network = Network::new();
    let mut counter = 0.0_f64;
    let neuron = Neuron::new(&network, 2, Activation::Identity, || {
        counter += 1.0;
        counter
    });

    // The weights are 1 and 2, the bias is 3.
    let first = network.leaf(10.0);
    let second = network.leaf(100.0);
    let output = neuron.express(&network, &[first, second]);

    let evaluation = network.forward();
    assert_eq!(*evaluation.value(output), 10.0 + 200.0 + 3.0);
}

#[test]
fn express_applies_the_activation() {
    let network = Network::new();
    let neuron = Neuron::new(&network, 1, Activation::Tanh, || 1.0_f64);

    let input = network.leaf(0.25);
    let output = neuron.express(&network, &[input]);

    let evaluation = network.forward();
    assert!((evaluation.value(output) - 1.25_f64.tanh()).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "number of inputs")]
fn express_rejects_wrong_arity() {
    let network = Network::new();
    let neuron = Neuron::new(&network, 2, Activation::Identity, || 0.0_f64);
    let input = network.leaf(1.0);
    neuron.express(&network, &[input]);
}

#[test]
fn neuron_trains_toward_a_target() {
    let network = Network::new();
    let neuron = Neuron::new(&network, 1, Activation::Tanh, || 0.0_f64);
    let input = network.leaf(1.0);
    let target = network.leaf(0.5);

    let output = neuron.express(&network, &[input]);
    let error = output + -target;
    let loss = error * error;

    let output_symbol = output.symbol();
    let loss_symbol = loss.symbol();

    let mut network = network;
    for _ in 0..200 {
        let loss = network.resolve(loss_symbol).unwrap();
        let evaluation = network.forward();
        let gradients = network.backward(&evaluation, loss);
        network = network.updated(gradients.as_field(), |parameter, gradient| {
            parameter - 0.5 * gradient
        });
    }

    let output = network.resolve(output_symbol).unwrap();
    let evaluation = network.forward();
    assert!((evaluation.value(output) - 0.5).abs() < 1e-3);
}
