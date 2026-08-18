use smallvec::smallvec;

use super::{Cotangents, Maximum, Operation};

#[test]
fn rules_are_plain_math_without_a_network() {
    assert_eq!(Maximum.arity(), 2);
    let larger: f64 = Maximum.forward(&[&2.0, &3.0]);
    assert_eq!(larger, 3.0);
}

#[test]
fn backward_hands_the_gradient_to_the_winner() {
    let cotangents = Maximum.backward(&[&2.0_f64, &3.0], &3.0, &1.5);
    let expected: Cotangents<f64> = smallvec![Some(0.0), Some(1.5)];
    assert_eq!(cotangents, expected);
}

#[test]
fn backward_hands_ties_to_the_left_operand() {
    let cotangents = Maximum.backward(&[&2.0_f64, &2.0], &2.0, &1.5);
    let expected: Cotangents<f64> = smallvec![Some(1.5), Some(0.0)];
    assert_eq!(cotangents, expected);
}
