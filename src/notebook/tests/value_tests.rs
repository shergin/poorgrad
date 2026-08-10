use super::*;
use crate::Network;

#[test]
fn a_parameters_card_shows_its_payload() {
    let network: Network<f64> = Network::new();
    let w = network.parameter(2.5);
    let html = w.to_html(Theme::DARK);
    assert!(html.contains("value"));
    assert!(html.contains("2.5"));
}

#[test]
fn an_uncomputed_value_says_so_instead_of_inventing_a_number() {
    let network: Network<f64> = Network::new();
    let a = network.parameter(1.0);
    let b = network.parameter(2.0);
    let sum = a + b;
    let html = sum.to_html(Theme::DARK);
    assert!(html.contains("not yet computed"));
    assert!(html.contains("forward"));
}

#[test]
fn a_computed_value_shows_the_payload_of_its_own_generation() {
    let network: Network<f64> = Network::new();
    let w = network.parameter(3.0);
    let doubled = w + w;
    let evaluation = network.forward();
    assert_eq!(*evaluation.of(doubled), 6.0);

    // The proxy still reports its generation's parameter, which is the
    // contract the notebook documentation warns about.
    assert!(w.to_html(Theme::DARK).contains("3"));
}

#[test]
fn a_symbol_card_explains_that_it_carries_no_payload() {
    let network: Network<f64> = Network::new();
    let symbol = network.parameter(1.0).symbol();
    let html = symbol.to_html(Theme::DARK);
    assert!(html.contains("resolve"));
    assert!(html.contains("symbol"));
}

#[test]
fn an_evaluation_card_profiles_the_whole_pass() {
    let network: Network<f64> = Network::new();
    let w = network.parameter(2.0);
    let squared = w * w;
    let _ = squared;
    let evaluation = network.forward();
    let html = evaluation.to_html(Theme::DARK);
    assert!(html.contains("evaluation"));
    assert!(html.contains("nodes"));
}

#[test]
fn value_rendering_is_deterministic_and_theme_aware() {
    let network: Network<f64> = Network::new();
    let w = network.parameter(1.0);
    assert_eq!(w.to_html(Theme::DARK), w.to_html(Theme::DARK));
    assert!(w.to_html(Theme::DARK).contains("#0d1117"));
    assert!(w.to_html(Theme::LIGHT).contains("#ffffff"));
}
