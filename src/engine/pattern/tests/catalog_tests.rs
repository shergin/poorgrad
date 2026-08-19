use crate::function::Function;
use crate::graph::Network;
use crate::{Tape, Tensor, conv2d, max_pool};

use super::super::candidates::Candidates;
use super::super::pattern::Pattern;
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

/// Discovers the pool over the whole network with every node wanted
/// and none readable.
fn discover(network: &Network<Tensor<f64>>) -> Candidates {
    let length = network.structure().len();
    let wanted = vec![true; length];
    let readable = vec![false; length];
    let view = View::new(network.structure(), &wanted, &readable);
    Candidates::discover(&view)
}

#[test]
fn an_empty_repertoire_elects_nothing() {
    // Discovery pools the conv candidate regardless of who will act;
    // a consumer that supports nothing claims nothing and skips
    // nothing.
    let network = conv_network(1, false);
    let length = network.structure().len();
    let catalog = Catalog::elect(&discover(&network), |_| false);
    assert_eq!(catalog.groups(), 0);
    assert!((0..length).all(|index| !catalog.interior(index)));
}

#[test]
fn disjoint_groups_all_claim() {
    let network = conv_network(2, false);
    let catalog = Catalog::elect(&discover(&network), |_| true);
    assert_eq!(catalog.groups(), 2);
}

#[test]
fn a_shared_source_feeds_two_groups() {
    // Extra reads are not claimed: two convolutions over one input
    // both match, the source being an argument of each fused call
    // rather than anyone's private interior.
    let network = conv_network(2, true);
    let catalog = Catalog::elect(&discover(&network), |_| true);
    assert_eq!(catalog.groups(), 2);
}

#[test]
fn a_repertoire_selects_among_candidates() {
    // One recording holding a conv and a pool: a consumer that
    // supports only window reductions elects the pool and leaves the
    // conv region to its primitives, while a total repertoire elects
    // both from the same pool.
    let tape = Tape::new();
    let input = tape.leaf(Tensor::new(
        [1, 1, 4, 4],
        (0..16).map(|v| v as f64 * 0.4 - 1.5).collect::<Vec<_>>(),
    ));
    let weights = tape.leaf(Tensor::new(
        [2, 1, 2, 2],
        (0..8).map(|v| v as f64 * 0.5 - 1.0).collect::<Vec<_>>(),
    ));
    let bias = tape.leaf(Tensor::new([2], [0.1, -0.1]));
    let pooled = max_pool(conv2d(input, weights, bias, 1, 0), 2, 1);
    let pool_root = pooled.symbol().id.index();
    let network = tape.into_network();
    let structure = network.structure();
    let matmul = (0..structure.len())
        .find(|&index| matches!(structure.functions.get(index), Some(Function::MatMul(_))))
        .expect("the conv records one matmul");
    let patches = structure
        .operands
        .get(matmul)
        .expect("plan columns are fixed")
        .as_slice()[0]
        .index();

    let candidates = discover(&network);
    let total = Catalog::elect(&candidates, |_| true);
    assert_eq!(total.groups(), 2);
    assert!(total.at(matmul).is_some());
    assert!(total.at(pool_root).is_some());
    assert!(total.interior(patches));

    let pools_only = Catalog::elect(&candidates, |pattern| {
        matches!(pattern, Pattern::ReduceWindow(_))
    });
    assert_eq!(pools_only.groups(), 1);
    assert!(pools_only.at(matmul).is_none());
    assert!(pools_only.at(pool_root).is_some());
    // The unsupported conv candidate did not claim: its chain is not
    // in the selective consumer's skip mask.
    assert!(!pools_only.interior(patches));
}
