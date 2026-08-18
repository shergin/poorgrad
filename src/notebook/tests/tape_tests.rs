use super::*;

#[test]
fn the_card_reports_the_recorded_node_count() {
    let tape: Tape<f64> = Tape::new();
    assert!(tape.to_html(Theme::DARK).contains("0 recorded nodes"));

    let a = tape.parameter(1.0);
    let b = tape.parameter(2.0);
    let _sum = a + b;
    assert!(tape.to_html(Theme::DARK).contains("3 recorded nodes"));
}

#[test]
fn one_node_reads_in_the_singular() {
    let tape: Tape<f64> = Tape::new();
    let _only = tape.parameter(1.0);
    assert!(tape.to_html(Theme::DARK).contains("1 recorded node<"));
}

#[test]
fn rendering_is_deterministic_and_theme_aware() {
    let tape: Tape<f64> = Tape::new();
    let _leaf = tape.parameter(1.0);
    assert_eq!(tape.to_html(Theme::DARK), tape.to_html(Theme::DARK));
    assert!(tape.to_html(Theme::DARK).contains("#0d1117"));
    assert!(tape.to_html(Theme::LIGHT).contains("#ffffff"));
}
