use crate::{Shape, Tensor};

use super::{Operation, Scatter};

#[test]
fn forward_matches_the_payload_scatter() {
    let rule = Scatter { rows: 3 };
    assert_eq!(rule.arity(), 2);

    let gradient = Tensor::new([3, 2], (1..=6).map(|v| v as f64).collect::<Vec<_>>());
    let selection = Tensor::selection(vec![0_usize, 2, 0], 3, 1.0);
    let result = rule.forward(&[&gradient, &selection]);
    // Token 0 is selected twice: rows one and three accumulate.
    assert_eq!(result.to_vec(), &[6.0, 8.0, 0.0, 0.0, 3.0, 4.0]);
}

#[test]
fn backward_gathers_by_the_same_selection() {
    let rule = Scatter { rows: 3 };
    let gradient = Tensor::filled([2, 2], 1.0_f64);
    let selection = Tensor::selection(vec![0_usize, 2], 3, 1.0);
    let output = rule.forward(&[&gradient, &selection]);

    let seed = Tensor::new([3, 2], (1..=6).map(|v| v as f64).collect::<Vec<_>>());
    let cotangents = rule.backward(&[&gradient, &selection], &output, &seed);
    assert_eq!(cotangents.len(), 2);
    let cotangent = cotangents[0].as_ref().unwrap();
    // Each gradient row reads its own scattered target row back.
    assert_eq!(cotangent.to_vec(), &[1.0, 2.0, 5.0, 6.0]);
    assert!(cotangents[1].is_none());
}

#[test]
fn infer_shape_replaces_the_count_with_the_rows() {
    assert_eq!(
        Scatter { rows: 5 }.infer_shape(&[Shape::new([3, 4]), Shape::new([3, 5])]),
        Shape::new([5, 4])
    );
}

#[test]
#[should_panic(expected = "disagree with the selection vocabulary")]
fn infer_shape_rejects_a_foreign_vocabulary() {
    Scatter { rows: 5 }.infer_shape(&[Shape::new([3, 4]), Shape::new([3, 6])]);
}

#[test]
#[should_panic(expected = "disagree with the selection count")]
fn infer_shape_rejects_a_mismatched_count() {
    Scatter { rows: 5 }.infer_shape(&[Shape::new([2, 4]), Shape::new([3, 5])]);
}
