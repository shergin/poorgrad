//! The tape card: the recording phase's one-line summary.

use malevich::Theme;

use super::html;
use crate::{Differentiable, Tape};

impl<Data: Differentiable> Tape<Data> {
    /// Renders the tape as a self-contained HTML card: how much graph
    /// is recorded so far, and the reminder that sealing is the way
    /// out of the phase.
    ///
    /// Rendering is pure and deterministic for a given tape and theme.
    pub fn to_html(&self, theme: Theme) -> String {
        let header = html::escape(&self.summary());
        html::card(
            theme,
            &header,
            "<div>the recording phase; <code>into_network()</code> seals \
             it into a runnable spec</div>",
        )
    }

    /// Displays the tape when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        html::show(&self.to_html(Theme::detect()), &self.summary());
    }

    /// The one-line description both representations share.
    fn summary(&self) -> String {
        let nodes = self.len();
        let plural = if nodes == 1 { "" } else { "s" };
        format!("tape  \u{b7}  {nodes} recorded node{plural}")
    }
}

#[cfg(test)]
#[path = "tests/tape_tests.rs"]
mod tests;
