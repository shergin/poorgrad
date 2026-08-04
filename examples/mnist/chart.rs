//! The MNIST training chart, drawn with `malevich`: the raw minibatch
//! loss, a rolling mean, and the uniform-model cost as the line every
//! run starts from and should fall far below.

use malevich::stat::Window;
use malevich::{Frame, Line, Plot, Rule};

/// What a uniform model costs on ten classes: `ln 10`.
const UNIFORM_COST: f64 = 2.302585092994046;

/// Renders the training curve for `losses`, one entry per step: the
/// per-step minibatch loss, its rolling mean over a twentieth of the
/// run, and the uniform start as a rule, sized to the terminal.
pub fn loss_chart(title: &str, losses: &[f32]) -> String {
    let losses: Vec<f64> = losses.iter().copied().map(f64::from).collect();
    let window_len = (losses.len() / 20).max(2);
    let smoothed = Window::new(window_len).mean(&losses);
    Plot::new()
        .layer(Line::y(&losses[..]).label("minibatch"))
        .layer(Line::y(&smoothed[..]).label("rolling mean"))
        .layer(Rule::h(UNIFORM_COST).label("uniform start"))
        .title(title)
        .x_label("step")
        .y_label("loss")
        .render_best(&Frame::detect())
}
