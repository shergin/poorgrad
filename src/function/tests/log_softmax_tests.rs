use crate::{Elementary, Shape, Tensor};

use super::{LogSoftmax, Operation};

#[test]
fn forward_matches_the_naive_composition() {
    let rule = LogSoftmax { axis: 1 };
    assert_eq!(rule.arity(), 1);

    let logits = Tensor::new([1, 3], [1.0_f64, 2.0, 3.0]);
    let result = rule.forward(&[&logits]);

    let normalizer = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
    for (computed, raw) in result.to_vec().into_iter().zip([1.0, 2.0, 3.0]) {
        assert!((computed - (raw - normalizer)).abs() < 1e-12);
    }
}

#[test]
fn forward_survives_extreme_logits() {
    let rule = LogSoftmax { axis: 1 };

    // Naively, `exp(1000)` overflows to infinity; the max shift keeps the
    // computation exact.
    let logits = Tensor::new([1, 2], [1000.0_f64, 1000.0]);
    let result = rule.forward(&[&logits]);
    for computed in result.to_vec() {
        assert!((computed - 0.5_f64.ln()).abs() < 1e-12);
    }
}

#[test]
fn forward_normalizes_each_lane_of_the_axis() {
    let rule = LogSoftmax { axis: 0 };

    let logits = Tensor::new([2, 2], [0.0_f64, 5.0, 0.0, -5.0]);
    let probabilities = rule.forward(&[&logits]).exp();
    let totals = probabilities.to_vec();
    assert!((totals[0] + totals[2] - 1.0).abs() < 1e-12);
    assert!((totals[1] + totals[3] - 1.0).abs() < 1e-12);
}

#[test]
fn backward_is_the_seed_minus_scaled_probabilities() {
    let rule = LogSoftmax { axis: 1 };
    let logits = Tensor::new([1, 3], [1.0_f64, 2.0, 3.0]);
    let output = rule.forward(&[&logits]);

    // Seeding one class turns the cotangent into the classic
    // `onehot - softmax`.
    let seed = Tensor::new([1, 3], [1.0_f64, 0.0, 0.0]);
    let cotangents = rule.backward(&[&logits], &output, &seed);
    assert_eq!(cotangents.len(), 1);

    let cotangent = cotangents[0].as_ref().unwrap();
    let softmax = output.exp();
    let picked = [1.0, 0.0, 0.0];
    for ((computed, probability), picked) in cotangent
        .to_vec()
        .into_iter()
        .zip(softmax.to_vec())
        .zip(picked)
    {
        assert!((computed - (picked - probability)).abs() < 1e-12);
    }
}

#[test]
fn infer_shape_preserves_the_operand_shape() {
    let rule = LogSoftmax { axis: 1 };
    assert_eq!(
        rule.infer_shape(&[Shape::new([4, 27])]),
        Shape::new([4, 27])
    );
}

#[test]
#[should_panic(expected = "out of rank")]
fn infer_shape_rejects_excessive_axes() {
    LogSoftmax { axis: 2 }.infer_shape(&[Shape::new([4, 27])]);
}
