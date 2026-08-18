use std::thread;

use crate::{Field, Tape, Tensor};

#[test]
fn sealing_preserves_the_recorded_length() {
    let tape = Tape::new();
    tape.parameter(1.5_f64);
    tape.leaf(2.0);
    let recorded = tape.len();
    let network = tape.into_network();
    assert!(!network.is_empty());
    assert_eq!(network.len(), recorded);
}

#[test]
fn forward_reads_parameters_from_the_callers_state() {
    let tape = Tape::new();
    let parameter = tape.parameter(1.5_f64);
    let input = tape.leaf(2.0);
    let output = (parameter * input).symbol();
    let network = tape.into_network();

    let parameters = network.parameters();
    assert_eq!(*network.forward(&parameters, []).of(output), 3.0);
}

#[test]
fn input_defaults_flow_through_forward() {
    let tape = Tape::new();
    let input = tape.input(3.0_f64);
    let doubled = (input * 2.0).symbol();
    let network = tape.into_network();
    let parameters = network.parameters();
    assert_eq!(*network.forward(&parameters, []).of(doubled), 6.0);
}

#[test]
fn feeds_override_inputs_per_run() {
    let tape = Tape::new();
    let input = tape.input(1.0_f64);
    let doubled = (input * 2.0).symbol();
    let input = input.symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let fed = network.forward(&parameters, [(input, 10.0)]);
    assert_eq!(*fed.of(doubled), 20.0);

    // Feeds are run-local: an unfed forward returns to the default,
    // which lives in the spec untouched.
    assert_eq!(*network.forward(&parameters, []).of(doubled), 2.0);
}

#[test]
#[should_panic(expected = "only inputs can be fed")]
fn feeds_reject_non_inputs() {
    let tape = Tape::new();
    let constant = tape.leaf(1.0_f64).symbol();
    let network = tape.into_network();
    network.forward(&network.parameters(), [(constant, 2.0)]);
}

#[test]
#[should_panic(expected = "symbol belongs to a different network")]
fn feeds_reject_foreign_symbols() {
    let tape = Tape::<f64>::new();
    let network = tape.into_network();
    let foreign = Tape::new().input(1.0_f64).symbol();
    network.forward(&network.parameters(), [(foreign, 2.0)]);
}

#[test]
#[should_panic(expected = "parameters belong to a different network")]
fn forward_rejects_foreign_parameters() {
    let tape = Tape::new();
    tape.parameter(1.0_f64);
    let network = tape.into_network();

    let other = Tape::new();
    other.parameter(1.0_f64);
    let foreign = other.into_network().parameters();
    network.forward(&foreign, []);
}

#[test]
fn concurrent_forwards_share_one_network_and_state() {
    let tape = Tape::new();
    let input = tape.input(0.0_f64);
    let squared = (input * input).symbol();
    let input = input.symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    thread::scope(|scope| {
        for fed in [2.0, 3.0, 4.0] {
            let network = &network;
            let parameters = &parameters;
            scope.spawn(move || {
                let run = network.forward(parameters, [(input, fed)]);
                assert_eq!(*run.of(squared), fed * fed);
            });
        }
    });
}

#[test]
fn training_steps_state_without_touching_the_network() {
    let tape = Tape::new();
    let weight = tape.parameter(0.0_f64);
    let bias = tape.parameter(0.0);
    let input = tape.input(0.0);
    let target = tape.input(0.0);
    let error = weight * input + bias - target;
    let loss = (error * error).symbol();
    let (weight, bias) = (weight.symbol(), bias.symbol());
    let (input, target) = (input.symbol(), target.symbol());
    let network = tape.into_network();
    let recorded = network.len();

    let samples = [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)];
    let mut parameters = network.parameters();
    for step in 0..600 {
        let (sample_input, sample_target) = samples[step % samples.len()];
        let run = network.forward(
            &parameters,
            [(input, sample_input), (target, sample_target)],
        );
        let gradients = run.backward(loss);
        parameters = parameters.step(&gradients, |parameter, gradient| {
            parameter - 0.05 * gradient
        });
    }

    assert_eq!(network.len(), recorded);
    assert!((*parameters.of(weight) - 2.0).abs() < 1e-3);
    assert!((*parameters.of(bias) - 1.0).abs() < 1e-3);
}

#[test]
fn gradient_descent_converges() {
    let tape = Tape::new();
    let parameter = tape.parameter(0.0_f64);
    let target = tape.leaf(3.0);
    let error = parameter - target;
    let loss = (error * error).symbol();
    let parameter = parameter.symbol();
    let network = tape.into_network();

    let mut parameters = network.parameters();
    for _ in 0..30 {
        let gradients = network.forward(&parameters, []).backward(loss);
        parameters = parameters.step(&gradients, |parameter, gradient| parameter - 0.3 * gradient);
    }
    assert!((*parameters.of(parameter) - 3.0).abs() < 1e-6);
}

#[test]
fn momentum_descent_converges() {
    let tape = Tape::new();
    let parameter = tape.parameter(0.0_f64);
    let target = tape.leaf(3.0);
    let error = parameter - target;
    let loss = (error * error).symbol();
    let parameter = parameter.symbol();
    let network = tape.into_network();

    let mut parameters = network.parameters();
    let mut velocity: Option<Field<f64>> = None;
    for _ in 0..40 {
        let gradients = network.forward(&parameters, []).backward(loss);
        let step = match velocity {
            Some(previous) => previous.scale(&0.5) + gradients,
            None => gradients,
        };
        parameters = parameters.step(&step, |parameter, direction| parameter - 0.1 * direction);
        velocity = Some(step);
    }
    assert!((*parameters.of(parameter) - 3.0).abs() < 1e-3);
}

#[test]
fn forward_for_evaluates_only_the_ancestor_closure() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([2], [2.0_f64, 3.0]));
    let wanted = (x * x).sum().symbol();
    let _unwanted = (x + x).sum();
    let x = x.symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let run = network.forward_for(&parameters, [wanted], []);
    assert_eq!(run.of(wanted).to_vec(), &[13.0]);

    // The sliced gradients match the full ones exactly.
    let sliced = run.backward(wanted);
    let full = network.forward(&parameters, []).backward(wanted);
    assert_eq!(sliced.of(x).to_vec(), full.of(x).to_vec());
}

#[test]
#[should_panic(expected = "not computed by this target-sliced run")]
fn sliced_reads_outside_the_closure_are_rejected() {
    let tape = Tape::new();
    let x = tape.leaf(2.0_f64);
    let wanted = (x * x).symbol();
    let unwanted = (x + x).symbol();
    let network = tape.into_network();

    let run = network.forward_for(&network.parameters(), [wanted], []);
    run.of(unwanted);
}

#[test]
#[should_panic(expected = "not computed by this target-sliced run")]
fn sliced_backward_outside_the_closure_is_rejected() {
    let tape = Tape::new();
    let x = tape.leaf(2.0_f64);
    let wanted = (x * x).symbol();
    let unwanted = (x + x).symbol();
    let network = tape.into_network();

    let run = network.forward_for(&network.parameters(), [wanted], []);
    run.backward(unwanted);
}

#[test]
fn forward_for_binds_feeds_like_forward() {
    let tape = Tape::new();
    let x = tape.input(Tensor::new([2], [0.0_f64, 0.0]));
    let doubled = (x * Tensor::new([2], [2.0, 2.0])).symbol();
    let x = x.symbol();
    let network = tape.into_network();

    let run = network.forward_for(
        &network.parameters(),
        [doubled],
        [(x, Tensor::new([2], [4.0, 5.0]))],
    );
    assert_eq!(run.of(doubled).to_vec(), &[8.0, 10.0]);
}

#[test]
fn sliced_gradients_step_parameters_like_full_gradients() {
    // Two expressions share one recording; slicing to the first must
    // step its parameter exactly as a full run does, while the second
    // expression's parameter receives its true gradient — zero — and
    // stays put.
    let tape = Tape::new();
    let first = tape.parameter(Tensor::new([2], [1.0_f64, 2.0]));
    let second = tape.parameter(Tensor::new([2], [5.0, 6.0]));
    let first_loss = (first * first).sum().symbol();
    let _second_loss = (second * second).sum();
    let (first, second) = (first.symbol(), second.symbol());
    let network = tape.into_network();

    let parameters = network.parameters();
    let run = network.forward_for(&parameters, [first_loss], []);
    let gradients = run.backward(first_loss);
    let stepped = parameters.step(&gradients, |parameter: &Tensor<f64>, gradient| {
        parameter.clone() - gradient.clone()
    });

    // `d(sum(w^2))/dw = 2w`, so the first parameter steps by `-2w`.
    assert_eq!(stepped.of(first).to_vec(), &[-1.0, -2.0]);
    assert_eq!(stepped.of(second).to_vec(), &[5.0, 6.0]);
}

#[test]
fn reopened_networks_serve_old_runs_and_new_expressions() {
    let tape = Tape::new();
    let weight = tape.parameter(2.0_f64);
    let squared = (weight * weight).symbol();
    let weight = weight.symbol();
    let network = tape.into_network();
    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    assert_eq!(*run.of(squared), 4.0);

    // Linear extension: reopen, record, reseal. The old run keeps
    // answering, and the extended spec serves the new expression.
    let tape = network.into_tape();
    let cubed = (tape.resolve(squared) * tape.resolve(weight)).symbol();
    let network = tape.into_network();
    let parameters = parameters.carried(&network);

    assert_eq!(*run.of(squared), 4.0);
    assert_eq!(*network.forward(&parameters, []).of(cubed), 8.0);
}
