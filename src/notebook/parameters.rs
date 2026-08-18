//! The parameters card: the caller-owned state's one-line summary.

use malevich::Theme;

use super::html;
use crate::{Differentiable, Parameters};

impl<Data: Differentiable> Parameters<Data> {
    /// Renders the state as a self-contained HTML card: how many
    /// parameter slots it carries, and how to read one.
    ///
    /// Rendering is pure and deterministic for a given state and
    /// theme.
    pub fn to_html(&self, theme: Theme) -> String {
        let header = html::escape(&self.summary());
        html::card(
            theme,
            &header,
            "<div>caller-owned state; <code>of(symbol)</code> reads a \
             payload, <code>step</code> trains</div>",
        )
    }

    /// Displays the state when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        html::show(&self.to_html(Theme::detect()), &self.summary());
    }

    /// The one-line description both representations share.
    fn summary(&self) -> String {
        let slots = self.len();
        let plural = if slots == 1 { "" } else { "s" };
        format!("parameters  \u{b7}  {slots} slot{plural}")
    }
}
