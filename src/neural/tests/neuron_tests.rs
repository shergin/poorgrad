use crate::Tape;

use super::{Activation, Neuron};

#[test]
fn new_allocates_weights_and_bias_as_parameters() {
    let tape = Tape::<f64>::new();
    let neuron = Neuron::new(&tape, 3, Activation::Identity, || 0.0);
    assert_eq!(tape.len(), 4);
    assert_eq!(neuron.parameters().count(), 4);
}

#[test]
fn express_records_the_affine_expression() {
    let tape = Tape::new();
    let mut counter = 0.0_f64;
    let neuron = Neuron::new(&tape, 2, Activation::Identity, || {
        counter += 1.0;
        counter
    });

    // The weights are 1 and 2, the bias is 3.
    let first = tape.leaf(10.0);
    let second = tape.leaf(100.0);
    let output = neuron.express(&tape, &[first, second]).symbol();

    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(*run.of(output), 10.0 + 200.0 + 3.0);
}

#[test]
fn express_applies_the_activation() {
    let tape = Tape::new();
    let neuron = Neuron::new(&tape, 1, Activation::Tanh, || 1.0_f64);

    let input = tape.leaf(0.25);
    let output = neuron.express(&tape, &[input]).symbol();

    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert!((run.of(output) - 1.25_f64.tanh()).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "number of inputs")]
fn express_rejects_wrong_arity() {
    let tape = Tape::new();
    let neuron = Neuron::new(&tape, 2, Activation::Identity, || 0.0_f64);
    let input = tape.leaf(1.0);
    neuron.express(&tape, &[input]);
}

#[test]
fn neuron_trains_toward_a_target() {
    let tape = Tape::new();
    let neuron = Neuron::new(&tape, 1, Activation::Tanh, || 0.0_f64);
    let input = tape.leaf(1.0);
    let target = tape.leaf(0.5);

    let output = neuron.express(&tape, &[input]);
    let error = output - target;
    let loss = error * error;

    let (output, loss) = (output.symbol(), loss.symbol());
    let network = tape.into_network();

    let mut parameters = network.parameters();
    for _ in 0..200 {
        let run = network.forward(&parameters, []);
        let gradients = run.backward(loss);
        parameters = parameters.step(&gradients, |parameter, gradient| parameter - 0.5 * gradient);
    }

    let run = network.forward(&parameters, []);
    assert!((run.of(output) - 0.5).abs() < 1e-3);
}
