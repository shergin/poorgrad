//! The shared makemore training chart, drawn with `malevich`: every
//! training example prints the same picture of its run — the raw
//! minibatch loss, a rolling mean, and the corpus's bigram limit as
//! the line to beat.

use malevich::stat::Window;
use malevich::{Frame, Line, Plot, Rule};

/// The corpus's bigram limit: the mean loss a fully converged bigram
/// model reaches on the names corpus. The bigram example approaches it
/// from above; the MLP examples aim below it.
const BIGRAM_LIMIT: f64 = 2.45;

/// Renders the training curve for `losses`, one entry per step: the
/// per-step minibatch loss, its rolling mean over a twentieth of the
/// run, and the bigram limit as a rule, sized to the terminal.
pub fn loss_chart(title: &str, losses: &[f32]) -> String {
    let losses: Vec<f64> = losses.iter().copied().map(f64::from).collect();
    let window_len = (losses.len() / 20).max(2);
    let smoothed = Window::new(window_len).mean(&losses);
    Plot::new()
        .layer(Line::y(&losses[..]).label("minibatch"))
        .layer(Line::y(&smoothed[..]).label("rolling mean"))
        .layer(Rule::h(BIGRAM_LIMIT).label("bigram limit"))
        .title(title)
        .x_label("step")
        .y_label("loss")
        .render_best(&Frame::detect())
}
