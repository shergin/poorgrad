use crate::{Shape, Tensor, Tensorial};

use super::{Fold, Operation};

/// The window pair of an eight-element axis unfolded by size 3, step 2.
fn windows() -> Tensor<f64> {
    Tensor::new([8], (1..=8).map(|v| v as f64).collect::<Vec<_>>()).unfold(0, 3, 2, 1)
}

#[test]
fn forward_matches_the_payload_fold() {
    let rule = Fold {
        axis: 0,
        size: 3,
        step: 2,
        dilation: 1,
        extent: 8,
    };
    assert_eq!(rule.arity(), 1);

    let windows = windows();
    let result = rule.forward(&[&windows]);
    assert_eq!(result.to_vec(), windows.fold(0, 3, 2, 1, 8).to_vec());
}

#[test]
fn backward_unfolds_by_the_same_parameters() {
    let rule = Fold {
        axis: 0,
        size: 3,
        step: 2,
        dilation: 1,
        extent: 8,
    };
    let windows = windows();
    let output = rule.forward(&[&windows]);

    let seed = Tensor::new([8], (0..8).map(|v| v as f64 * 0.5).collect::<Vec<_>>());
    let cotangents = rule.backward(&[&windows], &output, &seed);
    assert_eq!(cotangents.len(), 1);
    let cotangent = cotangents[0].as_ref().unwrap();
    assert_eq!(cotangent.to_vec(), seed.unfold(0, 3, 2, 1).to_vec());
}

#[test]
fn infer_shape_replaces_the_pair_with_the_extent() {
    let rule = Fold {
        axis: 1,
        size: 3,
        step: 1,
        dilation: 1,
        extent: 5,
    };
    assert_eq!(
        rule.infer_shape(&[Shape::new([2, 3, 3, 7])]),
        Shape::new([2, 5, 7])
    );
}

#[test]
#[should_panic(expected = "expects 3 windows")]
fn infer_shape_rejects_an_inconsistent_count() {
    Fold {
        axis: 0,
        size: 3,
        step: 1,
        dilation: 1,
        extent: 5,
    }
    .infer_shape(&[Shape::new([4, 3])]);
}

#[test]
#[should_panic(expected = "no pair there")]
fn infer_shape_rejects_a_missing_pair() {
    Fold {
        axis: 0,
        size: 3,
        step: 1,
        dilation: 1,
        extent: 5,
    }
    .infer_shape(&[Shape::new([3])]);
}
