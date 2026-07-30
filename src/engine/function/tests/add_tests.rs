use smallvec::smallvec;

use super::{Add, Cotangents, Operation};

#[test]
fn rules_are_plain_math_without_a_network() {
    assert_eq!(Add.arity(), 2);
    let sum: f64 = Add.forward(&[&2.0, &3.0]);
    assert_eq!(sum, 5.0);
}

#[test]
fn backward_hands_one_cotangent_per_operand() {
    let cotangents = Add.backward(&[&2.0_f64, &3.0], &5.0, &1.5);
    let expected: Cotangents<f64> = smallvec![Some(1.5), Some(1.5)];
    assert_eq!(cotangents, expected);
}
