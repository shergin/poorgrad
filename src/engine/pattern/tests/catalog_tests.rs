use crate::graph::Network;
use crate::{Tape, Tensor, conv2d};

use super::super::view::View;
use super::Catalog;

/// Records `count` convolutions and returns the network; `shared`
/// routes every convolution through one input value, so their chains
/// share a source.
fn conv_network(count: usize, shared: bool) -> Network<Tensor<f64>> {
    let tape = Tape::new();
    let shared_input = tape.leaf(Tensor::new(
        [1, 1, 4, 4],
        (0..16).map(|v| v as f64 * 0.3 - 2.0).collect::<Vec<_>>(),
    ));
    for group in 0..count {
        let input = if shared {
            shared_input
        } else {
            tape.leaf(Tensor::new(
                [1, 1, 4, 4],
                (0..16)
                    .map(|v| v as f64 * 0.2 + group as f64)
                    .collect::<Vec<_>>(),
            ))
        };
        let weights = tape.leaf(Tensor::new(
            [2, 1, 2, 2],
            (0..8)
                .map(|v| v as f64 * 0.25 - group as f64)
                .collect::<Vec<_>>(),
        ));
        let bias = tape.leaf(Tensor::new([2], [0.1, -0.1]));
        let _output = conv2d(input, weights, bias, 1, 0);
    }
    tape.into_network()
}

/// Collects the catalog over the whole network with every node wanted
/// and none readable.
fn collect(network: &Network<Tensor<f64>>, fuse: bool) -> Catalog {
    let length = network.structure().len();
    let wanted = vec![true; length];
    let readable = vec![false; length];
    let view = View::new(network.structure(), &wanted, &readable);
    Catalog::collect(&view, fuse)
}

#[test]
fn the_posture_gate_stores_no_homing_motif() {
    let network = conv_network(1, false);
    let length = network.structure().len();
    let catalog = collect(&network, false);
    assert_eq!(catalog.home_groups(), 0);
    assert!((0..length).all(|index| !catalog.home_interior(index)));
}

#[test]
fn disjoint_groups_all_claim() {
    let network = conv_network(2, false);
    let catalog = collect(&network, true);
    assert_eq!(catalog.home_groups(), 2);
}

#[test]
fn a_shared_source_feeds_two_groups() {
    // Extra reads are not claimed: two convolutions over one input
    // both match, the source being an argument of each fused call
    // rather than anyone's private interior.
    let network = conv_network(2, true);
    let catalog = collect(&network, true);
    assert_eq!(catalog.home_groups(), 2);
}
