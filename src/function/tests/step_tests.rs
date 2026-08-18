use crate::{Shape, Tensor};

use super::{Operation, Step};

#[test]
fn forward_is_the_left_biased_indicator() {
    let rule = Step;
    assert_eq!(rule.arity(), 2);

    let operand = Tensor::new([4], [-1.0_f64, 0.0, 0.5, 2.0]);
    let threshold = Tensor::new([4], [0.0_f64, 0.0, 1.0, 1.0]);
    let result = rule.forward(&[&operand, &threshold]);
    // Ties answer one, per the elementwise `step` contract.
    assert_eq!(result.to_vec(), &[0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn backward_declares_both_operands_data() {
    let rule = Step;
    let operand = Tensor::new([2], [1.0_f64, -1.0]);
    let threshold = Tensor::filled([2], 0.0_f64);
    let output = rule.forward(&[&operand, &threshold]);

    let seed = Tensor::filled([2], 1.0_f64);
    let cotangents = rule.backward(&[&operand, &threshold], &output, &seed);
    assert_eq!(cotangents.len(), 2);
    assert!(cotangents[0].is_none());
    assert!(cotangents[1].is_none());
}

#[test]
fn infer_shape_preserves_the_operand_shape() {
    assert_eq!(
        Step.infer_shape(&[Shape::new([2, 3]), Shape::new([2, 3])]),
        Shape::new([2, 3])
    );
}

#[test]
#[should_panic(expected = "equal shapes")]
fn infer_shape_rejects_mismatched_operands() {
    Step.infer_shape(&[Shape::new([2, 3]), Shape::new([3, 2])]);
}
