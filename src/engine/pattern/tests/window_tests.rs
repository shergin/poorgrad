use crate::function::Function;
use crate::graph::Network;
use crate::{Tape, Tensor, conv2d};

use super::super::pattern::Pattern;
use super::super::view::View;
use super::match_at;

/// Records a padded strided convolution and returns its network.
fn conv_network() -> Network<Tensor<f64>> {
    let tape = Tape::new();
    let input = tape.leaf(Tensor::new(
        [1, 2, 4, 4],
        (0..32).map(|v| v as f64 / 8.0 - 2.0).collect::<Vec<_>>(),
    ));
    let weights = tape.leaf(Tensor::new(
        [2, 2, 2, 2],
        (0..16).map(|v| v as f64 / 4.0 - 2.0).collect::<Vec<_>>(),
    ));
    let bias = tape.leaf(Tensor::new([2], [0.25, -0.5]));
    let _output = conv2d(input, weights, bias, 2, 1);
    tape.into_network()
}

/// Returns the index of the network's single `matmul` node.
fn matmul_index(network: &Network<Tensor<f64>>) -> usize {
    let functions = &network.structure().functions;
    (0..functions.len())
        .find(|&index| matches!(functions.get(index), Some(Function::MatMul(_))))
        .expect("the conv records one matmul")
}

#[test]
fn the_conv_chain_matches_with_its_geometry() {
    let network = conv_network();
    let structure = network.structure();
    let length = structure.functions.len();
    let wanted = vec![true; length];
    let readable = vec![false; length];
    let view = View::new(
        &structure.functions,
        &structure.operands,
        &structure.shapes,
        &wanted,
        &readable,
    );
    let matmul = matmul_index(&network);

    let candidate = match_at(matmul, &view).expect("the canonical chain matches");
    let group = match candidate.pattern {
        Pattern::WindowProduct(group) => group,
    };
    assert_eq!(structure.shapes[group.source].axes(), [1, 2, 4, 4]);
    assert_eq!(structure.shapes[group.kernel].axes(), [8, 2]);
    assert_eq!(group.kernel_height, 2);
    assert_eq!(group.kernel_width, 2);
    assert_eq!(group.stride, 2);
    assert_eq!(group.padding, 1);
    // The reshape, permute, two unfolds, and two symmetric pads.
    assert_eq!(candidate.interiors.len(), 6);
    assert!(candidate.named.is_empty());
}

#[test]
fn a_kept_interior_bars_the_match() {
    let network = conv_network();
    let structure = network.structure();
    let length = structure.functions.len();
    let matmul = matmul_index(&network);
    let patches = structure
        .operands
        .get(matmul)
        .expect("plan columns are fixed")
        .as_slice()[0]
        .index();

    let wanted = vec![true; length];
    let mut readable = vec![false; length];
    readable[patches] = true;
    let view = View::new(
        &structure.functions,
        &structure.operands,
        &structure.shapes,
        &wanted,
        &readable,
    );
    assert!(match_at(matmul, &view).is_none());
}
