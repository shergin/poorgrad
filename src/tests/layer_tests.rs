use crate::Network;

use super::{Activation, Layer};

#[test]
fn new_allocates_parameters_for_every_neuron() {
    let network = Network::<f64>::new();
    let layer = Layer::new(&network, 3, 2, Activation::Identity, || 0.0);
    // Two neurons, each with three weights and a bias.
    assert_eq!(network.len(), 8);
    assert_eq!(layer.parameters().count(), 8);
}

#[test]
fn express_returns_one_output_per_neuron() {
    let network = Network::new();
    let mut counter = 0.0_f64;
    let layer = Layer::new(&network, 2, 2, Activation::Identity, || {
        counter += 1.0;
        counter
    });

    // The first neuron's weights are 1 and 2 with bias 3; the second's
    // are 4 and 5 with bias 6.
    let first = network.leaf(10.0);
    let second = network.leaf(100.0);
    let outputs = layer.express(&network, &[first, second]);

    let evaluation = network.forward();
    assert_eq!(outputs.len(), 2);
    assert_eq!(*evaluation.value(outputs[0]), 10.0 + 200.0 + 3.0);
    assert_eq!(*evaluation.value(outputs[1]), 40.0 + 500.0 + 6.0);
}

#[test]
fn layer_trains_toward_targets() {
    let network = Network::new();
    let layer = Layer::new(&network, 1, 2, Activation::Identity, || 0.0_f64);
    let input = network.leaf(1.0);
    let targets = [network.leaf(1.0), network.leaf(-1.0)];

    let outputs = layer.express(&network, &[input]);
    let mut loss = None;
    for (output, target) in outputs.iter().zip(targets) {
        let error = *output + -target;
        let squared = error * error;
        loss = Some(match loss {
            Some(total) => total + squared,
            None => squared,
        });
    }
    let loss = loss.expect("layer has outputs");

    let loss_symbol = loss.symbol();
    let output_symbols = [outputs[0].symbol(), outputs[1].symbol()];

    let mut network = network;
    for _ in 0..100 {
        let loss = network.resolve(loss_symbol).unwrap();
        let evaluation = network.forward();
        let gradients = network.backward(&evaluation, loss);
        network = network.updated(gradients.as_field(), |parameter, gradient| {
            parameter - 0.2 * gradient
        });
    }

    let evaluation = network.forward();
    let first = network.resolve(output_symbols[0]).unwrap();
    let second = network.resolve(output_symbols[1]).unwrap();
    assert!((evaluation.value(first) - 1.0).abs() < 1e-3);
    assert!((evaluation.value(second) + 1.0).abs() < 1e-3);
}
