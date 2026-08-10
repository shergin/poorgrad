//! The plan card: the schedule as text, and its memory as a curve.
//!
//! This is the display the crate exists for. Eager frameworks hide the
//! graph and lazy ones hide the schedule; a plan card puts the whole
//! schedule in the cell output and draws the live volume beside it, so
//! what a pass costs is a picture rather than a claim.

use malevich::{Frame, Line, Plot, Theme};

use super::html;
use crate::{Differentiable, Plan};

impl<Data: Differentiable> Plan<Data> {
    /// Renders the plan as a self-contained HTML card: every scheduled
    /// node with its operation, shape, and liveness, then the live
    /// volume the analysis licenses, plotted along the schedule.
    ///
    /// Rendering is pure and deterministic for a given plan and theme.
    pub fn to_html(&self, theme: Theme) -> String {
        let schedule = format!(
            "<pre style=\"margin:0;white-space:pre;overflow-x:auto\">{}</pre>",
            html::escape(&self.describe())
        );
        let series = self.live_series();
        if series.len() < 2 {
            return html::card(theme, "plan", &schedule);
        }
        let mut frame = Frame::plain(72, 16);
        frame.theme = theme;
        let plot = Plot::new()
            .layer(Line::y(&series[..]).label("live"))
            .title("live volume (elements)")
            .x_label("scheduled node");
        let body = format!(
            "{schedule}<div style=\"margin-top:10px\">{}</div>",
            plot.to_html(&frame)
        );
        html::card(theme, "plan", &body)
    }

    /// Displays the plan when it is the last expression in an Evcxr
    /// cell.
    pub fn evcxr_display(&self) {
        html::show(&self.to_html(Theme::detect()), &self.describe());
    }
}

#[cfg(test)]
#[path = "tests/plan_tests.rs"]
mod tests;
