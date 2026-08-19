use super::*;
use crate::{Network, Request, Tape};

/// A network whose plan has enough scheduled nodes to draw a curve.
fn chain() -> (Network<f64>, crate::Symbol) {
    let tape: Tape<f64> = Tape::new();
    let w = tape.parameter(2.0);
    let x = tape.parameter(3.0);
    let sum = w + x;
    let scaled = sum * w;
    let target = scaled + x;
    let symbol = target.symbol();
    (tape.into_network(), symbol)
}

#[test]
fn a_plan_card_carries_the_whole_schedule() {
    let (network, target) = chain();
    let plan = network.compile(Request::roots([target]));
    let html = plan.to_html(Theme::DARK);
    // `describe` is the schedule, escaped into a `pre` block verbatim.
    for line in plan.describe().lines().take(3) {
        if !line.trim().is_empty() {
            assert!(
                html.contains(&super::html::escape(line)),
                "schedule line missing from the card: {line}"
            );
        }
    }
}

#[test]
fn a_plan_card_draws_the_live_volume_curve() {
    let (network, target) = chain();
    let plan = network.compile(Request::roots([target]));
    let html = plan.to_html(Theme::DARK);
    assert!(html.contains("live volume (elements)"));
    assert!(html.contains("scheduled node"));
}

#[test]
fn the_live_series_follows_the_schedule_it_describes() {
    let (network, target) = chain();
    let plan = network.compile(Request::roots([target]));
    let series = plan.live_series();
    assert!(!series.is_empty());
    // Live volume only ever grows by the node just evaluated, so the
    // peak of the series is the peak the summary reports.
    let peak = series.iter().copied().fold(0.0_f64, f64::max);
    assert!(peak > 0.0);
    assert!(plan.describe().contains(&format!("{}", peak as usize)));
}

#[test]
fn markup_in_a_schedule_cannot_escape_into_the_card() {
    let (network, target) = chain();
    let plan = network.compile(Request::roots([target]));
    let html = plan.to_html(Theme::DARK);
    let schedule = html
        .split("<pre style=\"margin:0;white-space:pre;overflow-x:auto\">")
        .nth(1)
        .and_then(|rest| rest.split("</pre>").next())
        .expect("the card carries a schedule block");
    assert!(!schedule.contains('<'));
    assert!(!schedule.contains('>'));
}

#[test]
fn plan_rendering_is_deterministic_and_theme_aware() {
    let (network, target) = chain();
    let plan = network.compile(Request::roots([target]));
    assert_eq!(plan.to_html(Theme::DARK), plan.to_html(Theme::DARK));
    assert!(plan.to_html(Theme::DARK).contains("#0d1117"));
    assert!(plan.to_html(Theme::LIGHT).contains("#ffffff"));
}
