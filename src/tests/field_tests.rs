use crate::Network;

#[test]
fn algebra_combines_elementwise() {
    let network = Network::new();
    let a = network.leaf(2.0_f64);
    let b = network.leaf(3.0);
    let product = a * b;
    let sum = a + b;

    let evaluation = network.forward();
    let d_product = network.backward(&evaluation, product).into_field();
    let d_sum = network.backward(&evaluation, sum).into_field();

    let combined = &d_product + &d_sum;
    assert_eq!(*combined.of(a), 4.0);
    assert_eq!(*combined.of(b), 3.0);

    let scaled = combined.scaled(2.0);
    assert_eq!(*scaled.of(a), 8.0);

    let squared = d_product.zip(&d_product, |left, right| left * right);
    assert_eq!(*squared.of(a), 9.0);

    let shifted = d_sum.map(|value| value + 1.0);
    assert_eq!(*shifted.of(b), 2.0);
}

#[test]
#[should_panic(expected = "lineage")]
fn combination_rejects_foreign_lineages() {
    let first = Network::new();
    let second = Network::new();
    let a = first.leaf(1.0_f64);
    let b = second.leaf(1.0);

    let evaluation_first = first.forward();
    let evaluation_second = second.forward();
    let field_first = first.backward(&evaluation_first, a).into_field();
    let field_second = second.backward(&evaluation_second, b).into_field();

    let _ = &field_first + &field_second;
}

#[test]
fn fields_survive_generations_within_a_lineage() {
    let network = Network::new();
    let w = network.parameter(1.0_f64);
    let w_symbol = w.symbol();

    let evaluation = network.forward();
    let gradients = network.backward(&evaluation, w).into_field();

    // The next generation is kin to the previous one, so the field still
    // resolves against its values and can drive its update.
    let updated = network.updated(&gradients, |parameter, direction| parameter - direction);
    let w = updated.resolve(w_symbol).unwrap();
    assert_eq!(*gradients.of(w), 1.0);
    assert_eq!(w.data(), Some(0.0));
}
