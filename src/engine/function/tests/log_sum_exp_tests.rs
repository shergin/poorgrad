use crate::{Differentiable, Elementary, Shape, Tensor};

use super::{LogSumExp, Operation};

#[test]
fn forward_matches_the_naive_composition() {
    let rule = LogSumExp { axis: 1 };
    assert_eq!(rule.arity(), 1);

    let logits = Tensor::new([1, 3], [1.0_f64, 2.0, 3.0]);
    let result = rule.forward(&[&logits]);

    let normalizer = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
    assert_eq!(result.shape(), Shape::new([1]));
    assert!((result.to_vec()[0] - normalizer).abs() < 1e-12);
}

#[test]
fn forward_stays_finite_where_the_difference_overflows() {
    let rule = LogSumExp { axis: 0 };

    // Two finite logits whose difference overflows the representable
    // range: the mathematical answer is approximately the maximum, and
    // the shifted form must reach it without an infinite intermediate
    // poisoning the result — in either lane order.
    for logits in [[-1.0e308_f64, 1.0e308], [1.0e308, -1.0e308]] {
        let result = rule.forward(&[&Tensor::new([2], logits)]);
        let value = result.to_vec()[0];
        assert!(value.is_finite());
        assert!((value - 1.0e308).abs() < 1.0e293);
    }
}

#[test]
fn forward_stays_finite_where_the_difference_overflows_f32() {
    let rule = LogSumExp { axis: 0 };

    for logits in [[-3.0e38_f32, 3.0e38], [3.0e38, -3.0e38]] {
        let result = rule.forward(&[&Tensor::new([2], logits)]);
        let value = result.to_vec()[0];
        assert!(value.is_finite());
        assert!((value - 3.0e38).abs() < 3.0e31);
    }
}

#[test]
fn backward_is_the_scaled_softmax() {
    let rule = LogSumExp { axis: 1 };
    let logits = Tensor::new([1, 3], [1.0_f64, 2.0, 3.0]);
    let output = rule.forward(&[&logits]);

    let seed = Tensor::new([1], [2.0_f64]);
    let cotangents = rule.backward(&[&logits], &output, &seed);
    assert_eq!(cotangents.len(), 1);

    let cotangent = cotangents[0].as_ref().unwrap();
    let normalizer = 1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp();
    for (computed, raw) in cotangent.to_vec().into_iter().zip([1.0, 2.0, 3.0]) {
        let probability = raw.exp() / normalizer;
        assert!((computed - 2.0 * probability).abs() < 1e-12);
    }
}

#[test]
fn backward_stays_finite_at_the_extremes() {
    let rule = LogSumExp { axis: 0 };
    let logits = Tensor::new([2], [-1.0e308_f64, 1.0e308]);
    let output = rule.forward(&[&logits]);

    // The overwhelmed lane's probability underflows to an exact zero;
    // the winning lane's is one — no NaN anywhere in the gradient.
    let seed = Tensor::new([], [1.0_f64]);
    let cotangents = rule.backward(&[&logits], &output, &seed);
    let gradient = cotangents[0].as_ref().unwrap().to_vec();
    assert_eq!(gradient[0], 0.0);
    assert!((gradient[1] - 1.0).abs() < 1e-12);
}

#[test]
fn infer_shape_removes_the_axis() {
    let rule = LogSumExp { axis: 1 };
    assert_eq!(rule.infer_shape(&[Shape::new([4, 27])]), Shape::new([4]));
}

#[test]
#[should_panic(expected = "out of rank")]
fn infer_shape_rejects_excessive_axes() {
    LogSumExp { axis: 2 }.infer_shape(&[Shape::new([4, 27])]);
}
