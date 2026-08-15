use std::thread;

use crate::{Field, Tensor};

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
    assert_eq!(parameter.payload(), Some(1.5));

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
    assert_eq!(first.payload(), Some(2.0));
    assert_eq!(second.payload(), Some(3.0));
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

    assert_eq!(original_extra.payload(), Some(2.0));
    assert_eq!(forked_extra.payload(), Some(3.0));
    assert_eq!(network.resolve(shared.symbol()).payload(), Some(1.0));
    assert_eq!(fork.resolve(shared.symbol()).payload(), Some(1.0));
}

#[test]
fn input_defaults_flow_through_forward() {
    let network = Network::new();
    let input = network.input(3.0_f64);
    let doubled = input * 2.0;

    assert_eq!(*network.forward().of(doubled), 6.0);
    assert_eq!(input.payload(), Some(3.0));
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
    assert_eq!(input.payload(), Some(1.0));
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
                let run = network.forward_with([(input_symbol, fed)]);
                assert_eq!(*run.of(squared), fed * fed);
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
        let run =
            network.forward_with([(input_symbol, sample_input), (target_symbol, sample_target)]);
        let gradients = run.backward(loss);
        network = network.update(&gradients, |parameter, gradient| {
            parameter - 0.05 * gradient
        });
    }

    assert_eq!(network.len(), recorded_nodes);
    let learned_weight = network.resolve(weight_symbol).payload().unwrap();
    let learned_bias = network.resolve(bias_symbol).payload().unwrap();
    assert!((learned_weight - 2.0).abs() < 1e-3);
    assert!((learned_bias - 1.0).abs() < 1e-3);
}

#[test]
fn update_replaces_parameters_and_keeps_everything_else() {
    let network = Network::new();
    let parameter = network.parameter(1.0_f64);
    let input = network.leaf(2.0);
    let output = parameter * input;

    let run = network.forward();
    let gradients = run.backward(output);
    let next = network.update(&gradients, |parameter, gradient| parameter - gradient);

    assert_eq!(next.len(), network.len());
    assert_eq!(next.resolve(parameter.symbol()).payload(), Some(-1.0));
    assert_eq!(next.resolve(input.symbol()).payload(), Some(2.0));
    assert_eq!(parameter.payload(), Some(1.0));
}

#[test]
fn compacted_preserves_values_and_lineage() {
    let network = Network::new();
    let weight = network.parameter(2.0_f64);
    let input = network.input(3.0);
    let product = weight * input;
    let weight_symbol = weight.symbol();
    let input_symbol = input.symbol();
    let product_symbol = product.symbol();

    let compacted = network.compacted();
    assert_eq!(compacted.len(), network.len());
    assert_eq!(compacted.resolve(weight_symbol).payload(), Some(2.0));
    assert_eq!(compacted.resolve(input_symbol).payload(), Some(3.0));
    assert_eq!(
        *compacted.forward().of(product_symbol),
        *network.forward().of(product_symbol)
    );

    // Same lineage: a field from the original still steps the compacted
    // generation, and the reverse.
    let gradients = network.forward().backward(product_symbol);
    let next = compacted.update(&gradients, |parameter, gradient| parameter - gradient);
    assert_eq!(next.resolve(weight_symbol).payload(), Some(-1.0));
}

#[test]
fn compacted_detaches_structure_from_sibling_recordings() {
    let parent = Network::new();
    let weight = parent.parameter(1.0_f64);
    let loss = weight * weight;
    let loss_symbol = loss.symbol();
    let parent_len = parent.len();

    // Sibling forks record structure the parent never sees; a plain
    // clone would keep sharing the arena those nodes landed in.
    for index in 0..8 {
        let fork = parent.clone();
        for offset in 0..16 {
            let leaf = fork.leaf((index * 16 + offset) as f64);
            let _ = leaf * leaf;
        }
        assert!(fork.len() > parent_len);
        drop(fork);
    }
    assert_eq!(parent.len(), parent_len);

    let compacted = parent.compacted();
    assert_eq!(compacted.len(), parent_len);
    assert_eq!(
        *compacted.forward().of(loss_symbol),
        *parent.forward().of(loss_symbol)
    );

    // The compacted side records into a private arena: growing it does
    // not change the parent's live length, and symbols from before the
    // compact still resolve.
    let extra = compacted.leaf(9.0);
    let _ = extra * compacted.resolve(weight.symbol());
    assert_eq!(parent.len(), parent_len);
    assert!(compacted.len() > parent_len);
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
        network = network.update(&gradients, |parameter, gradient| parameter - 0.3 * gradient);
    }

    let learned = network.resolve(parameter_symbol).payload().unwrap();
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
            Some(previous) => previous.scale(&0.5) + gradients,
            None => gradients,
        };
        network = network.update(&step, |parameter, direction| parameter - 0.1 * direction);
        velocity = Some(step);
    }

    let learned = network.resolve(parameter_symbol).payload().unwrap();
    assert!((learned - 3.0).abs() < 1e-3);
}

#[test]
#[should_panic(expected = "stale")]
fn update_rejects_stale_gradients() {
    let network = Network::new();
    let parameter = network.parameter(1.0_f64);
    let gradients = network.forward().backward(parameter);
    network.leaf(2.0);
    network.update(&gradients, |parameter, _gradient| *parameter);
}

#[test]
#[should_panic(expected = "different network")]
fn update_rejects_foreign_gradients() {
    let first = Network::new();
    let parameter = first.parameter(1.0_f64);
    let gradients = first.forward().backward(parameter);
    let second = Network::<f64>::new();
    second.update(&gradients, |parameter, _gradient| *parameter);
}

#[test]
fn forward_for_evaluates_only_the_ancestor_closure() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2], [2.0_f64, 3.0]));
    let wanted = (x * x).sum();
    let unwanted = (x + x).sum();

    let run = network.forward_for([wanted.symbol()], std::iter::empty());
    assert_eq!(run.of(wanted).to_vec(), &[13.0]);

    // The skipped expression is differentiable from a full run, and the
    // sliced gradients match the full ones exactly.
    let sliced = run.backward(wanted);
    let full = network.forward().backward(wanted);
    assert_eq!(sliced.of(x).to_vec(), full.of(x).to_vec());
    let _ = unwanted;
}

#[test]
#[should_panic(expected = "not computed by this target-sliced run")]
fn sliced_reads_outside_the_closure_are_rejected() {
    let network = Network::new();
    let x = network.leaf(2.0_f64);
    let wanted = x * x;
    let unwanted = x + x;

    let run = network.forward_for([wanted.symbol()], std::iter::empty());
    run.of(unwanted);
}

#[test]
#[should_panic(expected = "not computed by this target-sliced run")]
fn sliced_backward_outside_the_closure_is_rejected() {
    let network = Network::new();
    let x = network.leaf(2.0_f64);
    let wanted = x * x;
    let unwanted = x + x;

    let run = network.forward_for([wanted.symbol()], std::iter::empty());
    run.backward(unwanted);
}

#[test]
fn forward_for_binds_feeds_like_forward_with() {
    let network = Network::new();
    let x = network.input(Tensor::new([2], [0.0_f64, 0.0]));
    let doubled = x * Tensor::new([2], [2.0, 2.0]);

    let run = network.forward_for(
        [doubled.symbol()],
        [(x.symbol(), Tensor::new([2], [4.0, 5.0]))],
    );
    assert_eq!(run.of(doubled).to_vec(), &[8.0, 10.0]);
}

#[test]
fn sliced_gradients_step_parameters_like_full_gradients() {
    // Two expressions share one tape; slicing to the first must step
    // its parameter exactly as a full run does, while the second
    // expression's parameter receives its true gradient — zero — and
    // stays put.
    let network = Network::new();
    let first = network.parameter(Tensor::new([2], [1.0_f64, 2.0]));
    let second = network.parameter(Tensor::new([2], [5.0, 6.0]));
    let first_loss = (first * first).sum();
    let _second_loss = (second * second).sum();

    let first_symbol = first.symbol();
    let second_symbol = second.symbol();

    let run = network.forward_for([first_loss.symbol()], std::iter::empty());
    let gradients = run.backward(first_loss);
    let stepped = network.update(
        &gradients,
        |parameter: &Tensor<f64>, gradient: &Tensor<f64>| parameter.clone() - gradient.clone(),
    );

    // `d(sum(w^2))/dw = 2w`, so the first parameter steps by `-2w`.
    assert_eq!(
        stepped.resolve(first_symbol).payload().unwrap().to_vec(),
        &[-1.0, -2.0]
    );
    assert_eq!(
        stepped.resolve(second_symbol).payload().unwrap().to_vec(),
        &[5.0, 6.0]
    );
}
