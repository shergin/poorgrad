//! The gradient card: one magnitude per recorded node.
//!
//! A field holds a payload for every node, which is far too much to
//! print and exactly the right amount to plot. Reducing each payload
//! to its Euclidean norm turns a backward pass into one curve along
//! the tape, where a vanishing or exploding region is visible at a
//! glance and a `NaN` cannot hide.

use malevich::{Frame, Line, Plot, Theme};

use super::render::Renderable;
use super::{html, render};
use crate::Field;

/// The Euclidean norm of every node's payload, in tape order.
fn norms<Data: Renderable>(field: &Field<Data>) -> Vec<f64> {
    field
        .payloads()
        .iter()
        .map(|payload| {
            payload
                .cells()
                .iter()
                .map(|cell| cell * cell)
                .sum::<f64>()
                .sqrt()
        })
        .collect()
}

/// The header both representations share.
fn summary(label: &str, norms: &[f64]) -> String {
    let finite: Vec<f64> = norms
        .iter()
        .copied()
        .filter(|norm| norm.is_finite())
        .collect();
    let largest = finite.iter().copied().fold(0.0_f64, f64::max);
    let mean = if finite.is_empty() {
        0.0
    } else {
        finite.iter().sum::<f64>() / finite.len() as f64
    };
    let unusual = norms.len() - finite.len();
    let mut parts = vec![
        label.to_string(),
        format!("{} nodes", norms.len()),
        format!("max norm {}", render::number(largest)),
        format!("mean norm {}", render::number(mean)),
    ];
    if unusual > 0 {
        parts.push(format!("{unusual} non-finite"));
    }
    parts.join("  \u{b7}  ")
}

/// Renders a per-node magnitude profile as a self-contained HTML card.
pub(crate) fn profile_card<Data: Renderable>(
    theme: Theme,
    label: &str,
    field: &Field<Data>,
) -> String {
    let norms = norms(field);
    let header = html::escape(&summary(label, &norms));
    if norms.len() < 2 {
        let only = norms.first().copied().unwrap_or(0.0);
        let body = format!(
            "<div style=\"font-size:20px\">{}</div>",
            html::escape(&render::number(only))
        );
        return html::card(theme, &header, &body);
    }
    let mut frame = Frame::plain(72, 18);
    frame.theme = theme;
    let plot = Plot::new()
        .layer(Line::y(&norms[..]).label("norm"))
        .x_label("node");
    html::card(theme, &header, &plot.to_html(&frame))
}

/// Renders a per-node magnitude profile as plain text.
pub(crate) fn profile_text<Data: Renderable>(label: &str, field: &Field<Data>) -> String {
    let norms = norms(field);
    let header = summary(label, &norms);
    if norms.len() < 2 {
        let only = norms.first().copied().unwrap_or(0.0);
        return format!("{header}\n{}", render::number(only));
    }
    let plot = Plot::new()
        .layer(Line::y(&norms[..]).label("norm"))
        .x_label("node");
    format!("{header}\n{}", plot.render(&Frame::plain(72, 18)))
}

// `Renderable` is deliberately crate private: it names the closed set of
// payload types a card can draw, and it is a rendering detail rather
// than something a caller should bound its own code on. The lint fires
// because these inherent methods are public; nothing outside the crate
// can name the trait, so there is no leak to close. Silencing it also
// keeps `cargo check` warning-free, which Evcxr requires.
#[allow(private_bounds)]
impl<Data: Renderable> Field<Data> {
    /// Renders the field as a self-contained HTML card: one Euclidean
    /// norm per recorded node, plotted along the tape.
    ///
    /// Rendering is pure and deterministic for a given field and theme.
    pub fn to_html(&self, theme: Theme) -> String {
        profile_card(theme, "gradients", self)
    }

    /// Displays the field when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        html::show(
            &self.to_html(Theme::detect()),
            &profile_text("gradients", self),
        );
    }
}

#[cfg(test)]
#[path = "tests/field_tests.rs"]
mod tests;
