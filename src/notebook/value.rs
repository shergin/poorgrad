//! Cards for the two ways to designate a value, and for a finished run.

use malevich::Theme;

use super::render::Renderable;
use super::{html, render};
use crate::{Run, Symbol, Value};

// `Renderable` is deliberately crate private: it names the closed set of
// payload types a card can draw, and it is a rendering detail rather
// than something a caller should bound its own code on. The lint fires
// because these inherent methods are public; nothing outside the crate
// can name the trait, so there is no leak to close. Silencing it also
// keeps `cargo check` warning-free, which Evcxr requires.
#[allow(private_bounds)]
impl<Data: Renderable> Value<'_, Data> {
    /// Renders the proxy's current payload as a self-contained HTML
    /// card.
    ///
    /// The payload shown is the one this proxy's own generation holds.
    /// A proxy recorded before a training run keeps reporting that
    /// generation's value, which is the contract rather than a
    /// staleness bug; resolve the [`Symbol`] against a newer network to
    /// read the newer payload.
    pub fn to_html(&self, theme: Theme) -> String {
        let Some(payload) = self.payload() else {
            let header = format!(
                "value  \u{b7}  {}  \u{b7}  not yet computed",
                render::shape_text(&self.shape())
            );
            return html::card(
                theme,
                &html::escape(&header),
                "<div>run <code>forward</code> to give this value a payload</div>",
            );
        };
        render::payload_card(theme, "value", &payload)
    }

    /// Displays the value when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        let plain = match self.payload() {
            Some(payload) => render::payload_text("value", &payload),
            None => format!(
                "value  {}  not yet computed",
                render::shape_text(&self.shape())
            ),
        };
        html::show(&self.to_html(Theme::detect()), &plain);
    }
}

impl Symbol {
    /// Renders the symbol as a self-contained HTML card.
    ///
    /// A symbol is a detached name and carries no payload of its own,
    /// so the card says what it is and how to read through it rather
    /// than inventing a value.
    pub fn to_html(&self, theme: Theme) -> String {
        html::card(
            theme,
            "symbol",
            "<div>a detached name; <code>network.resolve(symbol)</code> \
             reads it in any compatible generation</div>",
        )
    }

    /// Displays the symbol when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        html::show(
            &self.to_html(Theme::detect()),
            "symbol  \u{b7}  resolve it against a network to read a payload",
        );
    }
}

// `Renderable` is deliberately crate private: it names the closed set of
// payload types a card can draw, and it is a rendering detail rather
// than something a caller should bound its own code on. The lint fires
// because these inherent methods are public; nothing outside the crate
// can name the trait, so there is no leak to close. Silencing it also
// keeps `cargo check` warning-free, which Evcxr requires.
#[allow(private_bounds)]
impl<Data: Renderable> Run<Data> {
    /// Renders the run as a self-contained HTML card.
    ///
    /// A run holds a value per node, so the card shows the profile of
    /// their magnitudes: the shape of the forward pass, where it grew,
    /// and whether anything went non-finite.
    pub fn to_html(&self, theme: Theme) -> String {
        super::field::profile_card(theme, "run", self.field())
    }

    /// Displays the run when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        html::show(
            &self.to_html(Theme::detect()),
            &super::field::profile_text("run", self.field()),
        );
    }
}

#[cfg(test)]
#[path = "tests/value_tests.rs"]
mod tests;
