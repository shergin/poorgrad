use super::*;
use crate::Tape;

#[test]
fn a_parameters_card_shows_its_payload() {
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(2.5);
    let html = w.to_html(Theme::DARK);
    assert!(html.contains("value"));
    assert!(html.contains("2.5"));
}

#[test]
fn an_uncomputed_value_says_so_instead_of_inventing_a_number() {
    let tape: Tape<f64> = Tape::new();
    let a = tape.parameter(1.0);
    let b = tape.parameter(2.0);
    let sum = a + b;
    let html = sum.to_html(Theme::DARK);
    assert!(html.contains("not yet computed"));
    assert!(html.contains("forward"));
}

#[test]
fn a_symbol_card_explains_that_it_carries_no_payload() {
    let tape: Tape<f64> = Tape::new();
    let symbol = tape.parameter(1.0).symbol();
    let html = symbol.to_html(Theme::DARK);
    assert!(html.contains("resolve"));
    assert!(html.contains("symbol"));
}

#[test]
fn a_run_card_profiles_the_whole_pass() {
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(2.0);
    let _squared = w * w;
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let html = run.to_html(Theme::DARK);
    assert!(html.contains("run"));
    assert!(html.contains("nodes"));
}

#[test]
fn value_rendering_is_deterministic_and_theme_aware() {
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(1.0);
    assert_eq!(w.to_html(Theme::DARK), w.to_html(Theme::DARK));
    assert!(w.to_html(Theme::DARK).contains("#0d1117"));
    assert!(w.to_html(Theme::LIGHT).contains("#ffffff"));
}
