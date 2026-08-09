use crate::{Network, Tensor};

#[test]
fn reads_answer_symbols_and_values_alike() {
    let network = Network::new();
    let x = network.parameter(3.0_f64);
    let loss = x * x;

    let evaluation = network.forward();
    assert_eq!(evaluation.of(loss), evaluation.of(loss.symbol()));

    let gradients = evaluation.backward(loss.symbol());
    assert_eq!(*gradients.of(x.symbol()), 6.0);
    assert_eq!(gradients.of(x), gradients.of(x.symbol()));
}

#[test]
fn field_reads_survive_a_generation_update() {
    let network = Network::new();
    let weight = network.parameter(1.0_f64);
    let loss = weight * weight;
    let symbol = weight.symbol();

    let gradients = network.forward().backward(loss);
    let updated = network.update(&gradients, |current, direction| current - direction * 0.25);

    // The field predates the new generation, and the detached symbol
    // reads it without any network at hand — the detachment fields
    // were built for.
    assert_eq!(*gradients.of(symbol), 2.0);
    // The same symbol resolves in the new generation's evaluation.
    assert_eq!(*updated.forward().of(symbol), 0.5);
}

#[test]
fn plans_and_derivatives_accept_bound_values() {
    let network = Network::new();
    let x = network.parameter(Tensor::new([2], vec![1.0_f64, 2.0]));
    let loss = (x * x).sum();

    let gradient_symbols = network.differentiate(loss, [x]);
    // `From<Value> for Symbol` serves the one position `ValueRef`
    // cannot: a list that must be homogeneous in `Symbol`.
    let plan = network.compile(
        std::iter::once(loss.into()).chain(gradient_symbols.iter().copied()),
        [],
    );
    let evaluation = plan.forward(&network, []);
    assert_eq!(evaluation.of(loss).to_vec(), vec![5.0]);
    assert_eq!(evaluation.of(gradient_symbols[0]).to_vec(), vec![2.0, 4.0]);
}

#[test]
fn sliced_runs_take_targets_in_either_form() {
    let network = Network::new();
    let a = network.parameter(2.0_f64);
    let b = network.parameter(3.0_f64);
    let product = a * b;
    let _unrelated = a + b;

    let evaluation = network.forward_for([product], []);
    assert_eq!(*evaluation.of(product.symbol()), 6.0);
}

#[test]
#[should_panic(expected = "symbol belongs to a different network lineage")]
fn foreign_symbols_are_rejected_by_evaluation_reads() {
    let network = Network::new();
    let x = network.parameter(1.0_f64);
    let _loss = x * x;
    let foreign = Network::new();
    let stranger = foreign.parameter(1.0_f64);

    let evaluation = network.forward();
    let _ = evaluation.of(stranger.symbol());
}

#[test]
#[should_panic(expected = "was not evaluated by this target-sliced run")]
fn sliced_runs_stay_loud_for_skipped_symbols() {
    let network = Network::new();
    let a = network.parameter(2.0_f64);
    let b = network.parameter(3.0_f64);
    let product = a * b;
    let unrelated = a + b;

    let evaluation = network.forward_for([product], []);
    let _ = evaluation.of(unrelated.symbol());
}

#[test]
#[should_panic(expected = "symbol was allocated after this field was produced")]
fn late_symbols_are_rejected_by_field_reads() {
    let network = Network::new();
    let x = network.parameter(2.0_f64);
    let loss = x * x;

    let gradients = network.forward().backward(loss);
    let late = x + x;
    let _ = gradients.of(late.symbol());
}

#[test]
#[should_panic(expected = "symbol belongs to a divergent fork of the network")]
fn diverged_symbols_are_rejected_by_field_reads() {
    let network = Network::new();
    let x = network.parameter(2.0_f64);
    let fork = network.clone();

    // The original extends first and keeps the tip branch; the fork's
    // recording then mints its own branch over the same positions.
    let left_loss = x * x;
    let forked_x = fork.resolve(x.symbol());
    let right = forked_x + forked_x;

    let gradients = network.forward().backward(left_loss);
    let _ = gradients.of(right.symbol());
}
