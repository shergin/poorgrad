use smallvec::smallvec;

use super::{Cotangents, Operation, Powf};

#[test]
fn rules_are_plain_math_without_a_network() {
    assert_eq!(Powf.arity(), 2);
    let power: f64 = Powf.forward(&[&2.0, &3.0]);
    assert_eq!(power, 8.0);
}

#[test]
fn backward_routes_the_power_and_exponential_rules() {
    // `d(x^y)/dx = y * x^(y-1) = 12`; `d(x^y)/dy = x^y * ln(x) = 8 ln 2`.
    let cotangents = Powf.backward(&[&2.0_f64, &3.0], &8.0, &1.0);
    let expected: Cotangents<f64> = smallvec![Some(12.0), Some(8.0 * 2.0_f64.ln())];
    assert_eq!(cotangents, expected);
}

#[test]
fn exponent_gradient_is_undefined_for_negative_bases() {
    let cotangents = Powf.backward(&[&-2.0_f64, &2.0], &4.0, &1.0);
    assert_eq!(cotangents[0], Some(-4.0));
    assert!(cotangents[1].unwrap().is_nan());
}
