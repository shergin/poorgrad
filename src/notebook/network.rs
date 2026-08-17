//! Leaking constructors and the network card.

use malevich::Theme;

use super::html;
use crate::{Differentiable, Network};

impl<Data: Differentiable + 'static> Network<Data> {
    /// Returns a new empty network that lives for the rest of the
    /// process, so the proxies recorded on it are `Value<'static, _>`
    /// and survive an Evcxr cell boundary.
    ///
    /// It is `Box::leak(Box::new(Network::new()))` under a name that
    /// says why. See the [module documentation](super) for the idiom
    /// and for what the leak actually costs.
    ///
    /// # Examples
    /// ```
    /// use topos::{Network, Value};
    ///
    /// let network: &'static Network<f64> = Network::leaked();
    /// let w: Value<'static, f64> = network.parameter(2.0);
    /// assert_eq!(w.payload(), Some(2.0));
    /// ```
    pub fn leaked() -> &'static Network<Data> {
        Network::new().leak()
    }

    /// Returns this network at `'static`, so a generation produced by
    /// [`update`](Network::update) can replace the one a notebook
    /// persists and keep handing out proxies that outlive the cell.
    ///
    /// Leak once per cell run rather than once per training step: the
    /// recorded graph is shared, but every leaked generation keeps its
    /// own parameter store forever.
    ///
    /// # Examples
    /// ```
    /// use topos::{Network, Value};
    ///
    /// let network: &'static Network<f64> = Network::leaked();
    /// let w: Value<'static, f64> = network.parameter(1.0);
    /// let gradients = network.forward().backward(w);
    /// let trained: &'static Network<f64> = network.update(&gradients, |p, g| p - g).leak();
    /// assert_eq!(trained.resolve(w.symbol()).payload(), Some(0.0));
    /// ```
    pub fn leak(self) -> &'static Network<Data> {
        Box::leak(Box::new(self))
    }
}

impl<Data: Differentiable> Network<Data> {
    /// Renders the network as a self-contained HTML card: how much
    /// graph is recorded, and the reminder that proxies are bound to
    /// one generation.
    ///
    /// Rendering is pure and deterministic for a given network and
    /// theme, which is what makes it testable.
    pub fn to_html(&self, theme: Theme) -> String {
        let header = html::escape(&self.summary());
        html::card(
            theme,
            &header,
            "<div>resolve a <code>Symbol</code> against this generation to read a payload</div>",
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
