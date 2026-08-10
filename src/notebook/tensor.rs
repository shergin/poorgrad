//! The payload card, for a tensor held on its own.
//!
//! A notebook reaches a bare `Tensor` constantly — a batch, a weight
//! read back out of a generation, a running estimate — so the payload
//! renderer is reachable from the type directly and not only through
//! the values that carry it.

use malevich::Theme;

use super::render::{Renderable, Scalar};
use super::{html, render};
use crate::{Differentiable, Tensor};

// `Scalar` is crate private on purpose: it names the closed set of
// element types a card can draw and is a rendering detail rather than
// something a caller should bound its own code on. Nothing outside the
// crate can name it, so the lint has no leak to close, and silencing it
// keeps `cargo check` warning-free, which Evcxr requires.
#[allow(private_bounds)]
impl<Element: Scalar> Tensor<Element>
where
    Tensor<Element>: Differentiable + Renderable,
{
    /// Renders the tensor as a self-contained HTML card: shape, element
    /// type, and extremes, then the values — an exact table while they
    /// are few, a chart once they are many.
    ///
    /// Rendering is pure and deterministic for a given tensor and theme.
    pub fn to_html(&self, theme: Theme) -> String {
        render::payload_card(theme, "tensor", self)
    }

    /// Displays the tensor when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        html::show(
            &self.to_html(Theme::detect()),
            &render::payload_text("tensor", self),
        );
    }
}

#[cfg(test)]
#[path = "tests/tensor_tests.rs"]
mod tests;
