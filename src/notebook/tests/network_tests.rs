use super::*;
use crate::Tape;

#[test]
fn the_card_reports_the_recorded_node_count() {
    let empty: Network<f64> = Tape::new().into_network();
    assert!(empty.to_html(Theme::DARK).contains("0 recorded nodes"));

    let tape: Tape<f64> = Tape::new();
    let a = tape.parameter(1.0);
    let b = tape.parameter(2.0);
    let _sum = a + b;
    let network = tape.into_network();
    assert!(network.to_html(Theme::DARK).contains("3 recorded nodes"));
}

#[test]
fn one_node_reads_in_the_singular() {
    let tape: Tape<f64> = Tape::new();
    let _only = tape.parameter(1.0);
    let network = tape.into_network();
    assert!(network.to_html(Theme::DARK).contains("1 recorded node<"));
}

#[test]
fn rendering_is_deterministic_and_theme_aware() {
    let tape: Tape<f64> = Tape::new();
    let _leaf = tape.parameter(1.0);
    let network = tape.into_network();
    assert_eq!(network.to_html(Theme::DARK), network.to_html(Theme::DARK));
    assert!(network.to_html(Theme::DARK).contains("#0d1117"));
    assert!(network.to_html(Theme::LIGHT).contains("#ffffff"));
}
