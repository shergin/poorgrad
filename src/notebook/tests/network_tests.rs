use super::*;
use crate::Value;

#[test]
fn a_leaked_network_hands_out_proxies_that_outlive_their_scope() {
    // The whole notebook idiom rests on this: a `'static` network makes
    // `Value<'static, _>`, which is what Evcxr can persist between cells.
    let proxy: Value<'static, f64> = {
        let network: &'static Network<f64> = Network::leaked();
        network.parameter(1.5)
    };
    assert_eq!(proxy.payload(), Some(1.5));
}

#[test]
fn leaking_a_generation_keeps_symbols_resolving() {
    let network: &'static Network<f64> = Network::leaked();
    let w: Value<'static, f64> = network.parameter(1.0);
    let gradients = network.forward().backward(w);
    let trained: &'static Network<f64> = network.update(&gradients, |p, g| p - g).leak();

    // The old generation is untouched and the new one holds the step.
    assert_eq!(w.payload(), Some(1.0));
    assert_eq!(trained.resolve(w.symbol()).payload(), Some(0.0));
}

#[test]
fn the_card_reports_the_recorded_node_count() {
    let network: Network<f64> = Network::new();
    assert!(network.to_html(Theme::DARK).contains("0 recorded nodes"));

    let a = network.parameter(1.0);
    let b = network.parameter(2.0);
    let _sum = a + b;
    assert!(network.to_html(Theme::DARK).contains("3 recorded nodes"));
}

#[test]
fn one_node_reads_in_the_singular() {
    let network: Network<f64> = Network::new();
    let _only = network.parameter(1.0);
    assert!(network.to_html(Theme::DARK).contains("1 recorded node<"));
}

#[test]
fn rendering_is_deterministic_and_theme_aware() {
    let network: Network<f64> = Network::new();
    let _leaf = network.parameter(1.0);
    assert_eq!(network.to_html(Theme::DARK), network.to_html(Theme::DARK));
    assert!(network.to_html(Theme::DARK).contains("#0d1117"));
    assert!(network.to_html(Theme::LIGHT).contains("#ffffff"));
}
