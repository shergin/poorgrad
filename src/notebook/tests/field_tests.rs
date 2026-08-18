use super::*;
use crate::Tape;

#[test]
fn a_gradient_card_reports_the_node_count_and_norms() {
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(3.0);
    let squared = (w * w).symbol();
    let network = tape.into_network();
    let gradients = network.forward(&network.parameters(), []).backward(squared);

    let html = gradients.to_html(Theme::DARK);
    assert!(html.contains("gradients"));
    assert!(html.contains("nodes"));
    // d(w^2)/dw is 2w, so the largest gradient norm on this tape is 6.
    assert!(html.contains("max norm 6"));
}

#[test]
fn a_gradient_card_plots_once_there_is_more_than_one_node() {
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(2.0);
    let x = tape.parameter(3.0);
    let product = (w * x).symbol();
    let network = tape.into_network();
    let gradients = network.forward(&network.parameters(), []).backward(product);
    assert!(gradients.to_html(Theme::DARK).contains("<pre"));
}

#[test]
fn non_finite_gradients_are_counted_rather_than_hidden() {
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(0.0);
    // `ln(0)` is negative infinity, and its derivative is infinite too:
    // exactly the case a notebook must not silently average away.
    let logged = w.ln().symbol();
    let network = tape.into_network();
    let gradients = network.forward(&network.parameters(), []).backward(logged);
    let html = gradients.to_html(Theme::DARK);
    assert!(html.contains("non-finite"));
}

#[test]
fn the_plain_text_form_carries_the_same_header_as_the_card() {
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(3.0);
    let squared = (w * w).symbol();
    let network = tape.into_network();
    let gradients = network.forward(&network.parameters(), []).backward(squared);
    let text = profile_text("gradients", &gradients);
    assert!(text.contains("gradients"));
    assert!(text.contains("max norm 6"));
}

#[test]
fn gradient_rendering_is_deterministic() {
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(1.5);
    let squared = (w * w).symbol();
    let network = tape.into_network();
    let gradients = network.forward(&network.parameters(), []).backward(squared);
    assert_eq!(
        gradients.to_html(Theme::DARK),
        gradients.to_html(Theme::DARK)
    );
}
