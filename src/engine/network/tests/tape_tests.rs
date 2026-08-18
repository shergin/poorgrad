use super::Tape;

#[test]
fn new_tape_is_empty() {
    let tape = Tape::<f64>::new();
    assert!(tape.is_empty());
    assert_eq!(tape.len(), 0);
}

#[test]
fn leaf_allocates_on_the_tape() {
    let tape = Tape::new();
    let first = tape.leaf(2.0_f64);
    let second = tape.leaf(3.0);
    assert_eq!(tape.len(), 2);
    assert_ne!(first.id(), second.id());
    assert_eq!(first.payload(), Some(2.0));
    assert_eq!(second.payload(), Some(3.0));
}

#[test]
fn parameter_and_input_carry_their_initials() {
    let tape = Tape::new();
    let parameter = tape.parameter(1.5_f64);
    let input = tape.input(3.0);
    assert_eq!(parameter.payload(), Some(1.5));
    assert_eq!(input.payload(), Some(3.0));
}

#[test]
fn resolve_answers_the_recorded_node() {
    let tape = Tape::new();
    let weight = tape.parameter(2.0_f64);
    let symbol = weight.symbol();
    assert_eq!(tape.resolve(symbol).payload(), Some(2.0));
    assert_eq!(tape.resolve(symbol).id(), weight.id());
}

#[test]
#[should_panic(expected = "symbol belongs to a different network")]
fn resolve_rejects_foreign_symbols() {
    let tape = Tape::<f64>::new();
    let foreign = Tape::new().leaf(1.0_f64).symbol();
    tape.resolve(foreign);
}

#[test]
fn into_network_and_back_preserves_symbols_and_length() {
    let tape = Tape::new();
    let weight = tape.parameter(2.0_f64);
    let doubled = weight * 2.0;
    let (weight, doubled) = (weight.symbol(), doubled.symbol());
    let recorded = tape.len();

    let network = tape.into_network();
    assert_eq!(network.len(), recorded);

    // Reopening keeps the origin and every recorded node, so old
    // symbols resolve and extension appends after them.
    let tape = network.into_tape();
    assert_eq!(tape.len(), recorded);
    assert_eq!(tape.resolve(weight).payload(), Some(2.0));
    let tripled = tape.resolve(weight) * 3.0;
    assert!(tripled.id().index() >= recorded);
    let _ = tape.resolve(doubled);
}
