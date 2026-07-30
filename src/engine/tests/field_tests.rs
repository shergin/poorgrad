use crate::Network;

#[test]
fn algebra_combines_elementwise() {
    let network = Network::new();
    let a = network.leaf(2.0_f64);
    let b = network.leaf(3.0);
    let product = a * b;
    let sum = a + b;

    let evaluation = network.forward();
    let d_product = evaluation.backward(product);
    let d_sum = evaluation.backward(sum);

    let combined = &d_product + &d_sum;
    assert_eq!(*combined.of(a), 4.0);
    assert_eq!(*combined.of(b), 3.0);

    let result = combined.scale(2.0);
    assert_eq!(*result.of(a), 8.0);

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
    let field_first = evaluation_first.backward(a);
    let field_second = evaluation_second.backward(b);

    let _ = &field_first + &field_second;
}

#[test]
#[should_panic(expected = "divergent forks")]
fn combination_rejects_divergent_forks() {
    let network = Network::new();
    let _anchor = network.leaf(1.0_f64);
    let fork = network.clone();

    // Equal lengths, divergent branches: the fields describe different
    // nodes at the same positions and must not combine.
    let mine = network.leaf(2.0);
    let theirs = fork.leaf(3.0);

    let field_mine = network.forward().backward(mine);
    let field_theirs = fork.forward().backward(theirs);

    let _ = &field_mine + &field_theirs;
}

#[test]
fn fields_survive_generations_within_a_lineage() {
    let network = Network::new();
    let w = network.parameter(1.0_f64);
    let w_symbol = w.symbol();

    let evaluation = network.forward();
    let gradients = evaluation.backward(w);

    // The next generation is kin to the previous one, so the field still
    // resolves against its values and can drive its update.
    let next = network.update(&gradients, |parameter, direction| parameter - direction);
    let w = next.resolve(w_symbol);
    assert_eq!(*gradients.of(w), 1.0);
    assert_eq!(w.payload(), Some(0.0));
}
