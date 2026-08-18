use crate::{Tape, Tensor};

#[test]
fn parameters_materialize_the_record_site_initials() {
    let tape = Tape::new();
    let weight = tape.parameter(1.5_f64).symbol();
    let network = tape.into_network();

    let parameters = network.parameters();
    assert_eq!(parameters.len(), 1);
    assert_eq!(*parameters.of(weight), 1.5);

    // Every call answers an independent state: stepping one leaves a
    // fresh materialization at the initials.
    let run = network.forward(&parameters, []);
    let gradients = run.backward(weight);
    let stepped = parameters.step(&gradients, |parameter, gradient| parameter - gradient);
    assert_eq!(*stepped.of(weight), 0.5);
    assert_eq!(*network.parameters().of(weight), 1.5);
}

#[test]
fn step_each_passes_the_parameter_symbol() {
    let tape = Tape::new();
    let first = tape.parameter(1.0_f64).symbol();
    let second = tape.parameter(2.0).symbol();
    let loss = (tape.resolve(first) * tape.resolve(second)).symbol();
    let network = tape.into_network();

    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    let gradients = run.backward(loss);
    let stepped = parameters.step_each(&gradients, |symbol, current, direction| {
        if symbol == first {
            current - direction
        } else {
            *current
        }
    });
    assert_eq!(*stepped.of(first), -1.0);
    assert_eq!(*stepped.of(second), 2.0);
}

#[test]
fn cloned_states_diverge_independently() {
    let tape = Tape::new();
    let weight = tape.parameter(0.0_f64).symbol();
    let loss = (tape.resolve(weight) * tape.resolve(weight)).symbol();
    let network = tape.into_network();

    let parameters = network.parameters();
    let fast = parameters.clone();
    let run = network.forward(&fast, []);
    let gradients = run.backward(loss);
    let fast = fast.step(&gradients, |parameter, _| parameter + 1.0);

    assert_eq!(*fast.of(weight), 1.0);
    assert_eq!(*parameters.of(weight), 0.0);
}

#[test]
#[should_panic(expected = "symbol belongs to a different network")]
fn of_rejects_foreign_symbols() {
    let tape = Tape::new();
    tape.parameter(1.0_f64);
    let network = tape.into_network();
    let parameters = network.parameters();

    let foreign = Tape::new().parameter(1.0_f64).symbol();
    parameters.of(foreign);
}

#[test]
#[should_panic(expected = "does not name a parameter")]
fn of_rejects_non_parameter_symbols() {
    let tape = Tape::new();
    tape.parameter(1.0_f64);
    let constant = tape.leaf(2.0).symbol();
    let network = tape.into_network();
    network.parameters().of(constant);
}

#[test]
#[should_panic(expected = "field belongs to a different network")]
fn step_rejects_foreign_fields() {
    let first = Tape::new();
    let weight = first.parameter(1.0_f64).symbol();
    let first = first.into_network();
    let gradients = first.forward(&first.parameters(), []).backward(weight);

    let second = Tape::new();
    second.parameter(1.0_f64);
    let second = second.into_network();
    second
        .parameters()
        .step(&gradients, |parameter, _| *parameter);
}

#[test]
#[should_panic(expected = "field is stale")]
fn step_rejects_stale_fields_after_extension() {
    let tape = Tape::new();
    let weight = tape.parameter(1.0_f64).symbol();
    let network = tape.into_network();
    let parameters = network.parameters();
    let gradients = network.forward(&parameters, []).backward(weight);

    // Reopen and record one more parameter: the old field no longer
    // covers the new slot, and stepping the carried state with it must
    // be loud.
    let tape = network.into_tape();
    tape.parameter(2.0);
    let network = tape.into_network();
    let carried = parameters.carried(&network);
    carried.step(&gradients, |parameter, _| *parameter);
}

#[test]
fn carried_keeps_payloads_and_seeds_new_slots() {
    let tape = Tape::new();
    let old = tape.parameter(1.0_f64).symbol();
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let gradients = run.backward(old);
    let parameters = network
        .parameters()
        .step(&gradients, |parameter, gradient| parameter + gradient);
    assert_eq!(*parameters.of(old), 2.0);

    let tape = network.into_tape();
    let new = tape.parameter(7.0).symbol();
    let network = tape.into_network();

    let carried = parameters.carried(&network);
    assert_eq!(carried.len(), 2);
    assert_eq!(*carried.of(old), 2.0);
    assert_eq!(*carried.of(new), 7.0);
}

#[test]
#[should_panic(expected = "parameters do not cover this network")]
fn forward_rejects_uncarried_parameters() {
    let tape = Tape::new();
    tape.parameter(1.0_f64);
    let network = tape.into_network();
    let parameters = network.parameters();

    let tape = network.into_tape();
    tape.parameter(2.0);
    let network = tape.into_network();
    network.forward(&parameters, []);
}

#[test]
fn with_payloads_replaces_named_parameters() {
    let tape = Tape::new();
    let kept = tape.parameter(Tensor::new([2], [1.0_f64, 2.0])).symbol();
    let replaced = tape.parameter(Tensor::new([2], [3.0, 4.0])).symbol();
    let network = tape.into_network();

    let parameters = network
        .parameters()
        .with_payloads([(replaced, Tensor::new([2], [9.0, 9.0]))]);
    assert_eq!(parameters.of(kept).to_vec(), &[1.0, 2.0]);
    assert_eq!(parameters.of(replaced).to_vec(), &[9.0, 9.0]);
}

#[test]
#[should_panic(expected = "must preserve the parameter's shape")]
fn with_payloads_rejects_shape_changes() {
    let tape = Tape::new();
    let weight = tape.parameter(Tensor::new([2], [1.0_f64, 2.0])).symbol();
    let network = tape.into_network();
    network
        .parameters()
        .with_payloads([(weight, Tensor::new([3], [1.0, 2.0, 3.0]))]);
}
