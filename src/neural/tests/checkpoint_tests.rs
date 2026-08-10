use crate::{Activation, Linear, Module, Network, Sequential, Tensor};

use super::{named_restore, named_snapshot, restore, snapshot};

/// Builds the same two-stage topology on `network` with `fill`-valued
/// parameters: the "same code" whose recording order both positional
/// identities rely on.
fn model(network: &Network<Tensor<f64>>, fill: f64) -> Sequential<Tensor<f64>> {
    Sequential::new()
        .then(Linear::new(
            network,
            Tensor::filled([2, 3], fill),
            Tensor::filled([3], fill),
        ))
        .then(Activation::Tanh)
        .then(Linear::new(
            network,
            Tensor::filled([3, 1], fill),
            Tensor::filled([1], fill),
        ))
}

#[test]
fn positional_checkpoints_round_trip() {
    let trained_network = Network::new();
    let trained = model(&trained_network, 0.5);
    let payloads = snapshot(&trained_network, &trained);
    assert_eq!(payloads.len(), 4);

    // A fresh process: same code, different initialization.
    let fresh_network = Network::new();
    let fresh = model(&fresh_network, 0.0);
    let restored_network = restore(&fresh_network, &fresh, payloads);

    let input_shape = [1, 2];
    let trained_input = trained_network.leaf(Tensor::filled(input_shape, 1.0_f64));
    let trained_output = trained.express(&trained_network, trained_input);
    let restored_input = restored_network.leaf(Tensor::filled(input_shape, 1.0_f64));
    let restored_output = fresh.express(&restored_network, restored_input);

    assert_eq!(
        trained_network.forward().of(trained_output).to_vec(),
        restored_network.forward().of(restored_output).to_vec(),
    );
}

#[test]
#[should_panic(expected = "payloads but the module has")]
fn positional_restore_rejects_a_count_mismatch() {
    let network = Network::new();
    let module = model(&network, 0.0);
    let _ = restore(&network, &module, vec![Tensor::filled([2, 3], 1.0_f64)]);
}

#[test]
fn named_checkpoints_round_trip() {
    let trained_network = Network::new();
    let trained = model(&trained_network, 0.25);
    let entries = named_snapshot(&trained_network, &trained);
    let rendered: Vec<String> = entries.iter().map(|(path, _)| path.to_string()).collect();
    assert_eq!(rendered, ["0.weights", "0.bias", "2.weights", "2.bias"]);

    let fresh_network = Network::new();
    let fresh = model(&fresh_network, 0.0);
    let restored_network = named_restore(&fresh_network, &fresh, entries);
    let payloads = snapshot(&restored_network, &fresh);
    assert_eq!(payloads[0].to_vec(), vec![0.25; 6]);
}

#[test]
#[should_panic(expected = "missing entries for: 2.weights")]
fn named_restore_rejects_missing_entries() {
    let network = Network::new();
    let module = model(&network, 0.5);
    let mut entries = named_snapshot(&network, &module);
    entries.remove(2);
    let _ = named_restore(&network, &module, entries);
}

#[test]
#[should_panic(expected = "no parameter matches")]
fn named_restore_rejects_unexpected_entries() {
    let network = Network::new();
    let first = model(&network, 0.5);
    let second = Sequential::new().then(Linear::new(
        &network,
        Tensor::filled([2, 3], 0.5_f64),
        Tensor::filled([3], 0.5),
    ));
    // Entries snapshotted from the larger model cannot all match the
    // smaller one.
    let entries = named_snapshot(&network, &first);
    let _ = named_restore(&network, &second, entries);
}

#[test]
fn tied_parameters_restore_once() {
    let network = Network::new();
    let head = Linear::new(
        &network,
        Tensor::filled([2, 2], 0.5_f64),
        Tensor::filled([2], 0.0),
    );
    let tied = Linear::from_symbols(head.weights(), head.bias());
    let model = Sequential::new().then(head).then(tied);

    let entries = named_snapshot(&network, &model);
    // One symbol under two paths: both entries are present, and the
    // restore takes the later one in visit order.
    assert_eq!(entries.len(), 4);
    let restored = named_restore(&network, &model, entries);
    let payloads = snapshot(&restored, &model);
    assert_eq!(payloads[0].to_vec(), vec![0.5; 4]);
}
