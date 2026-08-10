//! Payload rendering: the one place that turns a `Differentiable`
//! into a table, a chart, or a number.
//!
//! Every display in this module ultimately shows payloads, so the
//! choice of form lives here rather than in each type's display. Small
//! payloads become tables because exact values matter at that size;
//! large ones become `malevich` charts because shape matters more than
//! any individual element.

use malevich::{Frame, Theme};

use super::html;
use crate::{Bf16, Differentiable, Shape, Tensor};

/// The largest payload rendered as an exact table rather than a chart.
const TABLE_LIMIT: usize = 144;

/// The longest row a one-dimensional table prints before the payload
/// becomes a chart instead.
const ROW_LIMIT: usize = 24;

/// The frame every chart in a notebook card is drawn into.
fn chart_frame(theme: Theme) -> Frame {
    let mut frame = Frame::plain(72, 20);
    frame.theme = theme;
    frame
}

/// An element that a chart or table can show as a number.
///
/// It exists so the displays can be generic over the payload types the
/// crate actually has without widening any public trait: rendering
/// needs a real number, and `Differentiable` deliberately does not
/// promise one.
pub(crate) trait Scalar: Copy {
    /// Returns this element as an `f64`, the type every renderer works in.
    fn to_f64(self) -> f64;

    /// Returns the element type's name, for the card header.
    fn type_name() -> &'static str;
}

impl Scalar for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn type_name() -> &'static str {
        "f32"
    }
}

impl Scalar for f64 {
    fn to_f64(self) -> f64 {
        self
    }

    fn type_name() -> &'static str {
        "f64"
    }
}

impl Scalar for Bf16 {
    fn to_f64(self) -> f64 {
        f64::from(f32::from(self))
    }

    fn type_name() -> &'static str {
        "bf16"
    }
}

/// A payload a notebook can draw: its shape, and its elements in
/// row-major order.
pub(crate) trait Renderable: Differentiable {
    /// Returns every element in row-major order.
    fn cells(&self) -> Vec<f64>;

    /// Returns the element type's name, for the card header.
    fn element_name() -> &'static str;
}

impl<Element: Scalar> Renderable for Tensor<Element>
where
    Tensor<Element>: Differentiable,
{
    fn cells(&self) -> Vec<f64> {
        self.iter().map(Scalar::to_f64).collect()
    }

    fn element_name() -> &'static str {
        Element::type_name()
    }
}

impl Renderable for f32 {
    fn cells(&self) -> Vec<f64> {
        vec![f64::from(*self)]
    }

    fn element_name() -> &'static str {
        "f32"
    }
}

impl Renderable for f64 {
    fn cells(&self) -> Vec<f64> {
        vec![*self]
    }

    fn element_name() -> &'static str {
        "f64"
    }
}

impl Renderable for Bf16 {
    fn cells(&self) -> Vec<f64> {
        vec![Scalar::to_f64(*self)]
    }

    fn element_name() -> &'static str {
        "bf16"
    }
}

/// Formats one number at a width a table can align, without trailing
/// zeros and without scientific notation for ordinary magnitudes.
pub(crate) fn number(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let magnitude = value.abs();
    if !(1e-4..1e6).contains(&magnitude) {
        return format!("{value:.3e}");
    }
    let text = format!("{value:.4}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_string()
}

/// Renders `[2, 3]`-style axis text for a shape.
pub(crate) fn shape_text(shape: &Shape) -> String {
    let axes: Vec<String> = shape.axes().iter().map(usize::to_string).collect();
    format!("[{}]", axes.join(", "))
}

/// The minimum, maximum, and mean of a payload's elements.
///
/// Non-finite elements are excluded from all three, so a single `NaN`
/// does not erase the rest of the summary; the header reports their
/// presence separately.
fn extremes(cells: &[f64]) -> Option<(f64, f64, f64)> {
    let finite: Vec<f64> = cells
        .iter()
        .copied()
        .filter(|cell| cell.is_finite())
        .collect();
    if finite.is_empty() {
        return None;
    }
    let minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    Some((minimum, maximum, mean))
}

/// The card header for a payload: shape, element type, and the
/// summary statistics that say whether the numbers are sane.
pub(crate) fn header<Data: Renderable>(shape: &Shape, cells: &[f64]) -> String {
    let mut parts = vec![shape_text(shape), Data::element_name().to_string()];
    if let Some((minimum, maximum, mean)) = extremes(cells) {
        parts.push(format!(
            "min {} max {} mean {}",
            number(minimum),
            number(maximum),
            number(mean)
        ));
    }
    let unusual = cells.iter().filter(|cell| !cell.is_finite()).count();
    if unusual > 0 {
        parts.push(format!("{unusual} non-finite"));
    }
    html::escape(&parts.join("  \u{b7}  "))
}

/// Renders elements as an HTML table of `columns` per row, shading each
/// cell by where its value falls between the payload's extremes.
fn table(theme: Theme, cells: &[f64], columns: usize) -> String {
    use std::fmt::Write as _;

    let (low, high) = match extremes(cells) {
        Some((minimum, maximum, _)) => (minimum, maximum),
        None => (0.0, 0.0),
    };
    let span = high - low;
    let muted = html::muted_color(theme);
    let mut markup = String::from("<table style=\"border-collapse:collapse\">");
    for (index, cell) in cells.iter().enumerate() {
        if index % columns == 0 {
            let _ = write!(markup, "{}<tr>", if index == 0 { "" } else { "</tr>" });
        }
        // A flat payload gets no shading: every cell would tint identically
        // and the color would imply a variation that is not there.
        let weight = if span > 0.0 && cell.is_finite() {
            (cell - low) / span
        } else {
            0.0
        };
        let tint = format!("rgba(88,166,255,{:.3})", 0.10 + 0.55 * weight);
        let _ = write!(
            markup,
            "<td style=\"padding:2px 8px;text-align:right;background-color:{};\
             border:1px solid {muted}33\">{}</td>",
            if span > 0.0 {
                tint
            } else {
                "transparent".to_string()
            },
            html::escape(&number(*cell))
        );
    }
    markup.push_str("</tr></table>");
    markup
}

/// Renders elements as plain text laid out in rows of `columns`.
fn table_text(cells: &[f64], columns: usize) -> String {
    let rendered: Vec<String> = cells.iter().copied().map(number).collect();
    let width = rendered.iter().map(String::len).max().unwrap_or(1);
    let mut text = String::new();
    for (index, cell) in rendered.iter().enumerate() {
        if index > 0 && index % columns == 0 {
            text.push('\n');
        } else if index > 0 {
            text.push(' ');
        }
        let _ = std::fmt::Write::write_fmt(&mut text, format_args!("{cell:>width$}"));
    }
    text
}

/// The number of columns a payload's table prints, which is its last
/// axis for a shaped payload and its whole length for a flat one.
fn columns_of(shape: &Shape) -> usize {
    shape.axes().last().copied().unwrap_or(1).max(1)
}

/// Renders a payload's body: the exact values when they are few, and a
/// chart of them when they are many.
pub(crate) fn body<Data: Renderable>(theme: Theme, data: &Data) -> (String, String) {
    let shape = data.shape();
    let cells = data.cells();
    let columns = columns_of(&shape);

    if cells.len() == 1 {
        let value = html::escape(&number(cells[0]));
        return (
            format!("<div style=\"font-size:20px\">{value}</div>"),
            number(cells[0]),
        );
    }

    let small_row = shape.rank() <= 1 && cells.len() <= ROW_LIMIT;
    let small_grid = shape.rank() >= 2 && cells.len() <= TABLE_LIMIT;
    if small_row || small_grid {
        return (table(theme, &cells, columns), table_text(&cells, columns));
    }

    let frame = chart_frame(theme);
    let plot = if shape.rank() >= 2 {
        malevich::heatmap(columns, &cells[..])
    } else {
        malevich::Plot::new().layer(malevich::Line::y(&cells[..]))
    };
    (plot.to_html(&frame), plot.render(&frame))
}

/// Renders a complete payload card: header, then body.
pub(crate) fn payload_card<Data: Renderable>(theme: Theme, label: &str, data: &Data) -> String {
    let shape = data.shape();
    let cells = data.cells();
    let (body_html, _) = body(theme, data);
    let head = format!(
        "{}  \u{b7}  {}",
        html::escape(label),
        header::<Data>(&shape, &cells)
    );
    html::card(theme, &head, &body_html)
}

/// Renders a complete payload as plain text: header, then body.
pub(crate) fn payload_text<Data: Renderable>(label: &str, data: &Data) -> String {
    let shape = data.shape();
    let cells = data.cells();
    let (_, body_text) = body(Theme::DARK, data);
    let mut parts = vec![shape_text(&shape), Data::element_name().to_string()];
    if let Some((minimum, maximum, mean)) = extremes(&cells) {
        parts.push(format!(
            "min {} max {} mean {}",
            number(minimum),
            number(maximum),
            number(mean)
        ));
    }
    format!("{label}  {}\n{body_text}", parts.join("  "))
}

#[cfg(test)]
#[path = "tests/render_tests.rs"]
mod tests;
