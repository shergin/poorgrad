use crate::Tape;

#[test]
fn algebra_combines_elementwise() {
    let tape = Tape::new();
    let a = tape.leaf(2.0_f64);
    let b = tape.leaf(3.0);
    let product = a * b;
    let sum = a + b;
    let (a, b, product, sum) = (a.symbol(), b.symbol(), product.symbol(), sum.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    let d_product = run.backward(product);
    let d_sum = run.backward(sum);

    let combined = &d_product + &d_sum;
    assert_eq!(*combined.of(a), 4.0);
    assert_eq!(*combined.of(b), 3.0);

    let result = combined.scale(&2.0);
    assert_eq!(*result.of(a), 8.0);

    let squared = d_product.zip(&d_product, |left, right| left * right);
    assert_eq!(*squared.of(a), 9.0);

    let shifted = d_sum.map(|value| value + 1.0);
    assert_eq!(*shifted.of(b), 2.0);
}

#[test]
#[should_panic(expected = "fields belong to different networks")]
fn combination_rejects_foreign_networks() {
    let first = Tape::new();
    let second = Tape::new();
    let a = first.leaf(1.0_f64).symbol();
    let b = second.leaf(1.0).symbol();

    let first = first.into_network();
    let second = second.into_network();
    let field_first = first.forward(&first.parameters(), []).backward(a);
    let field_second = second.forward(&second.parameters(), []).backward(b);

    let _ = &field_first + &field_second;
}

#[test]
fn fields_step_later_parameter_states() {
    let tape = Tape::new();
    let w = tape.parameter(1.0_f64).symbol();
    let network = tape.into_network();
    let parameters = network.parameters();

    let run = network.forward(&parameters, []);
    let gradients = run.backward(w);
    assert_eq!(*gradients.of(w), 1.0);

    // A field is detached state: it still steps parameter states minted
    // after the run it came from.
    let stepped = parameters.step(&gradients, |parameter, direction| parameter - direction);
    assert_eq!(*stepped.of(w), 0.0);
    let again = stepped.step(&gradients, |parameter, direction| parameter - direction);
    assert_eq!(*again.of(w), -1.0);
}
