use smallvec::smallvec;

use super::{Cotangents, Operation, Sqrt};

#[test]
fn rules_are_plain_math_without_a_network() {
    assert_eq!(Sqrt.arity(), 1);
    let root: f64 = Sqrt.forward(&[&9.0]);
    assert_eq!(root, 3.0);
}

#[test]
fn backward_divides_by_twice_the_output() {
    // `d sqrt(x) / dx` at 9 is `1 / (2 * 3)`.
    let cotangents = Sqrt.backward(&[&9.0_f64], &3.0, &6.0);
    let expected: Cotangents<f64> = smallvec![Some(1.0)];
    assert_eq!(cotangents, expected);
}
