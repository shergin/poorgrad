use std::thread;

use crate::Field;

use super::Network;

#[test]
fn new_network_is_empty() {
    let network = Network::<f64>::new();
    assert!(network.is_empty());
    assert_eq!(network.len(), 0);
}

#[test]
fn parameter_carries_payload_like_a_leaf() {
    let network = Network::new();
    let parameter = network.parameter(1.5_f64);
    let input = network.leaf(2.0);
    assert_eq!(parameter.data(), Some(1.5));

    let output = parameter * input;
    assert_eq!(*network.forward().of(output), 3.0);
}

#[test]
fn leaf_allocates_on_the_network() {
    let network = Network::new();
    let first = network.leaf(2.0_f64);
    let second = network.leaf(3.0);
    assert_eq!(network.len(), 2);
    assert_ne!(first.id(), second.id());
    assert_eq!(first.data(), Some(2.0));
    assert_eq!(second.data(), Some(3.0));
}

#[test]
fn forked_parameter_stores_diverge_independently() {
    let network = Network::new();
    let shared = network.parameter(1.0_f64);
    let fork = network.clone();

    // Both branches assign the same slot to their post-fork parameter,
    // but each branch owns its store from the first divergent allocation.
    let original_extra = network.parameter(2.0);
    let forked_extra = fork.parameter(3.0);

    assert_eq!(original_extra.data(), Some(2.0));
    assert_eq!(forked_extra.data(), Some(3.0));
    assert_eq!(network.resolve(shared.symbol()).data(), Some(1.0));
    assert_eq!(fork.resolve(shared.symbol()).data(), Some(1.0));
}

#[test]
fn input_defaults_flow_through_forward() {
    let network = Network::new();
    let input = network.input(3.0_f64);
    let doubled = input * 2.0;

    assert_eq!(*network.forward().of(doubled), 6.0);
    assert_eq!(input.data(), Some(3.0));
}

#[test]
fn forward_with_overrides_inputs_per_run() {
    let network = Network::new();
    let input = network.input(1.0_f64);
    let doubled = input * 2.0;
    let input_symbol = input.symbol();

    let fed = network.forward_with([(input_symbol, 10.0)]);
    assert_eq!(*fed.of(doubled), 20.0);

    // Feeds are run-local: a plain forward returns to the default and
    // the recorded default payload remains unchanged.
    assert_eq!(*network.forward().of(doubled), 2.0);
    assert_eq!(input.data(), Some(1.0));
}

#[test]
#[should_panic(expected = "only inputs can be fed")]
fn forward_with_rejects_non_inputs() {
    let network = Network::new();
    let constant = network.leaf(1.0_f64);
    network.forward_with([(constant.symbol(), 2.0)]);
}

#[test]
#[should_panic(expected = "different network lineage")]
fn forward_with_rejects_foreign_symbols() {
    let network = Network::<f64>::new();
    let foreign = Network::new().input(1.0).symbol();
    network.forward_with([(foreign, 2.0)]);
}

#[test]
fn concurrent_forwards_feed_independent_batches() {
    let network = Network::new();
    let input = network.input(0.0_f64);
    let squared = input * input;
    let input_symbol = input.symbol();

    thread::scope(|scope| {
        for fed in [2.0, 3.0, 4.0] {
            let network = &network;
            scope.spawn(move || {
                let evaluation = network.forward_with([(input_symbol, fed)]);
                assert_eq!(*evaluation.of(squared), fed * fed);
            });
        }
    });
}

#[test]
fn training_feeds_batches_without_regrowing_the_tape() {
    let network = Network::new();
    let weight = network.parameter(0.0_f64);
    let bias = network.parameter(0.0);
    let input = network.input(0.0);
    let target = network.input(0.0);
    let error = weight * input + bias - target;
    let loss = error * error;

    let weight_symbol = weight.symbol();
    let bias_symbol = bias.symbol();
    let input_symbol = input.symbol();
    let target_symbol = target.symbol();
    let loss_symbol = loss.symbol();
    let recorded_nodes = network.len();

    let samples = [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)];
    let mut network = network;
    for step in 0..600 {
        let (sample_input, sample_target) = samples[step % samples.len()];
        let loss = network.resolve(loss_symbol);
        let evaluation =
            network.forward_with([(input_symbol, sample_input), (target_symbol, sample_target)]);
        let gradients = evaluation.backward(loss);
        network = network.updated(gradients.as_field(), |parameter, gradient| {
            parameter - 0.05 * gradient
        });
    }

    assert_eq!(network.len(), recorded_nodes);
    let learned_weight = network.resolve(weight_symbol).data().unwrap();
    let learned_bias = network.resolve(bias_symbol).data().unwrap();
    assert!((learned_weight - 2.0).abs() < 1e-3);
    assert!((learned_bias - 1.0).abs() < 1e-3);
}

#[test]
fn updated_replaces_parameters_and_keeps_everything_else() {
    let network = Network::new();
    let parameter = network.parameter(1.0_f64);
    let input = network.leaf(2.0);
    let output = parameter * input;

    let evaluation = network.forward();
    let gradients = evaluation.backward(output);
    let updated = network.updated(gradients.as_field(), |parameter, gradient| {
        parameter - gradient
    });

    assert_eq!(updated.len(), network.len());
    assert_eq!(updated.resolve(parameter.symbol()).data(), Some(-1.0));
    assert_eq!(updated.resolve(input.symbol()).data(), Some(2.0));
    assert_eq!(parameter.data(), Some(1.0));
}

#[test]
fn gradient_descent_converges() {
    let network = Network::new();
    let parameter = network.parameter(0.0_f64);
    let target = network.leaf(3.0);
    let error = parameter - target;
    let loss = error * error;

    let parameter_symbol = parameter.symbol();
    let loss_symbol = loss.symbol();

    let mut network = network;
    for _ in 0..30 {
        let loss = network.resolve(loss_symbol);
        let gradients = network.forward().backward(loss);
        network = network.updated(gradients.as_field(), |parameter, gradient| {
            parameter - 0.3 * gradient
        });
    }

    let learned = network.resolve(parameter_symbol).data().unwrap();
    assert!((learned - 3.0).abs() < 1e-6);
}

#[test]
fn momentum_descent_converges() {
    let network = Network::new();
    let parameter = network.parameter(0.0_f64);
    let target = network.leaf(3.0);
    let error = parameter - target;
    let loss = error * error;

    let parameter_symbol = parameter.symbol();
    let loss_symbol = loss.symbol();

    let mut network = network;
    let mut velocity: Option<Field<f64>> = None;
    for _ in 0..40 {
        let loss = network.resolve(loss_symbol);
        let gradients = network.forward().backward(loss);
        let step = match velocity {
            Some(previous) => previous.scaled(0.5) + gradients.into_field(),
            None => gradients.into_field(),
        };
        network = network.updated(&step, |parameter, direction| parameter - 0.1 * direction);
        velocity = Some(step);
    }

    let learned = network.resolve(parameter_symbol).data().unwrap();
    assert!((learned - 3.0).abs() < 1e-3);
}

#[test]
#[should_panic(expected = "stale")]
fn updated_rejects_stale_gradients() {
    let network = Network::new();
    let parameter = network.parameter(1.0_f64);
    let gradients = network.forward().backward(parameter);
    network.leaf(2.0);
    network.updated(gradients.as_field(), |parameter, _gradient| *parameter);
}

#[test]
#[should_panic(expected = "different network")]
fn updated_rejects_foreign_gradients() {
    let first = Network::new();
    let parameter = first.parameter(1.0_f64);
    let gradients = first.forward().backward(parameter);
    let second = Network::<f64>::new();
    second.updated(gradients.as_field(), |parameter, _gradient| *parameter);
}
