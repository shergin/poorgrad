//! The network card: the sealed spec's one-line summary.

use malevich::Theme;

use super::html;
use crate::{Differentiable, Network};

impl<Data: Differentiable> Network<Data> {
    /// Renders the network as a self-contained HTML card: how much
    /// graph is recorded, and the reminder that payloads live in the
    /// caller's state.
    ///
    /// Rendering is pure and deterministic for a given network and
    /// theme, which is what makes it testable.
    pub fn to_html(&self, theme: Theme) -> String {
        let header = html::escape(&self.summary());
        html::card(
            theme,
            &header,
            "<div>the immutable spec; read payloads through \
             <code>parameters.of(symbol)</code> or a run</div>",
        )
    }

    /// Displays the network when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        html::show(&self.to_html(Theme::detect()), &self.summary());
    }

    /// The one-line description both representations share.
    fn summary(&self) -> String {
        let nodes = self.len();
        let plural = if nodes == 1 { "" } else { "s" };
        format!("network  \u{b7}  {nodes} recorded node{plural}")
    }
}

#[cfg(test)]
#[path = "tests/network_tests.rs"]
mod tests;
