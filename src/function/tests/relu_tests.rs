use smallvec::smallvec;

use super::{Cotangents, Operation, Relu};

#[test]
fn rules_are_plain_math_without_a_network() {
    assert_eq!(Relu.arity(), 1);
    let rectified: f64 = Relu.forward(&[&-2.0]);
    assert_eq!(rectified, 0.0);
    let passed: f64 = Relu.forward(&[&2.0]);
    assert_eq!(passed, 2.0);
}

#[test]
fn backward_stops_where_the_operand_is_negative() {
    let stopped = Relu.backward(&[&-2.0_f64], &0.0, &1.5);
    let expected: Cotangents<f64> = smallvec![Some(0.0)];
    assert_eq!(stopped, expected);

    let passed = Relu.backward(&[&2.0_f64], &2.0, &1.5);
    let expected: Cotangents<f64> = smallvec![Some(1.5)];
    assert_eq!(passed, expected);
}
