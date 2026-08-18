use crate::{Activation, Linear, Module, Sequential, Tape, Tensor};

use super::{named_restore, named_snapshot, restore, snapshot};

/// Builds the same two-stage topology on `tape` with `fill`-valued
/// parameters: the "same code" whose recording order both positional
/// identities rely on.
fn model(tape: &Tape<Tensor<f64>>, fill: f64) -> Sequential<Tensor<f64>> {
    Sequential::new()
        .then(Linear::new(
            tape,
            Tensor::filled([2, 3], fill),
            Tensor::filled([3], fill),
        ))
        .then(Activation::Tanh)
        .then(Linear::new(
            tape,
            Tensor::filled([3, 1], fill),
            Tensor::filled([1], fill),
        ))
}

#[test]
fn positional_checkpoints_round_trip() {
    let input_shape = [1, 2];
    let trained_tape = Tape::new();
    let trained = model(&trained_tape, 0.5);
    let trained_input = trained_tape.leaf(Tensor::filled(input_shape, 1.0_f64));
    let trained_output = trained.express(&trained_tape, trained_input).symbol();
    let trained_network = trained_tape.into_network();
    let trained_parameters = trained_network.parameters();
    let payloads = snapshot(&trained_parameters, &trained);
    assert_eq!(payloads.len(), 4);

    // A fresh process: same code, different initialization.
    let fresh_tape = Tape::new();
    let fresh = model(&fresh_tape, 0.0);
    let fresh_input = fresh_tape.leaf(Tensor::filled(input_shape, 1.0_f64));
    let fresh_output = fresh.express(&fresh_tape, fresh_input).symbol();
    let fresh_network = fresh_tape.into_network();
    let restored = restore(&fresh_network.parameters(), &fresh, payloads);

    assert_eq!(
        trained_network
            .forward(&trained_parameters, [])
            .of(trained_output)
            .to_vec(),
        fresh_network
            .forward(&restored, [])
            .of(fresh_output)
            .to_vec(),
    );
}

#[test]
#[should_panic(expected = "payloads but the module has")]
fn positional_restore_rejects_a_count_mismatch() {
    let tape = Tape::new();
    let module = model(&tape, 0.0);
    let parameters = tape.into_network().parameters();
    let _ = restore(&parameters, &module, vec![Tensor::filled([2, 3], 1.0_f64)]);
}

#[test]
fn named_checkpoints_round_trip() {
    let trained_tape = Tape::new();
    let trained = model(&trained_tape, 0.25);
    let trained_parameters = trained_tape.into_network().parameters();
    let entries = named_snapshot(&trained_parameters, &trained);
    let rendered: Vec<String> = entries.iter().map(|(path, _)| path.to_string()).collect();
    assert_eq!(rendered, ["0.weights", "0.bias", "2.weights", "2.bias"]);

    let fresh_tape = Tape::new();
    let fresh = model(&fresh_tape, 0.0);
    let fresh_parameters = fresh_tape.into_network().parameters();
    let restored = named_restore(&fresh_parameters, &fresh, entries);
    let payloads = snapshot(&restored, &fresh);
    assert_eq!(payloads[0].to_vec(), vec![0.25; 6]);
}

#[test]
#[should_panic(expected = "missing entries for: 2.weights")]
fn named_restore_rejects_missing_entries() {
    let tape = Tape::new();
    let module = model(&tape, 0.5);
    let parameters = tape.into_network().parameters();
    let mut entries = named_snapshot(&parameters, &module);
    entries.remove(2);
    let _ = named_restore(&parameters, &module, entries);
}

#[test]
#[should_panic(expected = "no parameter matches")]
fn named_restore_rejects_unexpected_entries() {
    let tape = Tape::new();
    let first = model(&tape, 0.5);
    let second = Sequential::new().then(Linear::new(
        &tape,
        Tensor::filled([2, 3], 0.5_f64),
        Tensor::filled([3], 0.5),
    ));
    let parameters = tape.into_network().parameters();
    // Entries snapshotted from the larger model cannot all match the
    // smaller one.
    let entries = named_snapshot(&parameters, &first);
    let _ = named_restore(&parameters, &second, entries);
}

#[test]
fn tied_parameters_restore_once() {
    let tape = Tape::new();
    let head = Linear::new(
        &tape,
        Tensor::filled([2, 2], 0.5_f64),
        Tensor::filled([2], 0.0),
    );
    let tied = Linear::from_symbols(head.weights(), head.bias());
    let model = Sequential::new().then(head).then(tied);
    let parameters = tape.into_network().parameters();

    let entries = named_snapshot(&parameters, &model);
    // One symbol under two paths: both entries are present, and the
    // restore takes the later one in visit order.
    assert_eq!(entries.len(), 4);
    let restored = named_restore(&parameters, &model, entries);
    let payloads = snapshot(&restored, &model);
    assert_eq!(payloads[0].to_vec(), vec![0.5; 4]);
}
