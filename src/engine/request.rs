use crate::graph::Symbol;

/// A compile request: the explicit product of what a plan computes.
///
/// The request names its roots (what must be computed — a loss, a
/// logits head, recorded gradient symbols; no root is special), the
/// extra interior values to observe (readable after a run, alongside
/// the roots), and whether run buffers support the engine reverse
/// scan ([`Request::backward`]). The request never touches the
/// graph: recorded gradients enter as ordinary roots, produced by a
/// visible [`Tape::differentiate`](crate::Tape::differentiate)
/// beforehand, so a request is cheap and re-runnable.
///
/// Roots and observes are detached [`Symbol`]s; a [`Value`](crate::Value)
/// still in scope converts through `Into<Symbol>`, and validation
/// happens when [`Network::compile`](crate::Network::compile) resolves
/// them.
///
/// # Examples
/// ```
/// # use topos::{Request, Tape};
/// # let tape = Tape::new();
/// # let weight = tape.parameter(1.0_f64);
/// # let loss = (weight * weight).sum().symbol();
/// # let network = tape.into_network();
/// // Pure inference: a forward-only plan over one root.
/// let inference = network.compile(Request::roots([loss]));
///
/// // Engine training: run buffers retain what `backward` reads.
/// let training = network.compile(Request::roots([loss]).backward());
/// assert!(!inference.can_backward());
/// assert!(training.can_backward());
/// ```
#[derive(Debug, Clone)]
pub struct Request {
    pub(crate) roots: Vec<Symbol>,
    pub(crate) observe: Vec<Symbol>,
    pub(crate) backward: bool,
}

impl Request {
    /// Opens a request over `roots`, the closure sources a run must
    /// compute; every root is readable after a run.
    pub fn roots(roots: impl IntoIterator<Item = impl Into<Symbol>>) -> Self {
        Self {
            roots: roots.into_iter().map(Into::into).collect(),
            observe: Vec::new(),
            backward: false,
        }
    }

    /// Adds interior values the caller also wants readable after a
    /// run; like roots, they seed the plan's reachability closure.
    /// Repeated calls accumulate.
    pub fn observe(mut self, extra: impl IntoIterator<Item = impl Into<Symbol>>) -> Self {
        self.observe.extend(extra.into_iter().map(Into::into));
        self
    }

    /// Requests runs that answer [`Run::backward`](crate::Run::backward):
    /// buffers retain what the engine's reverse scan reads — the
    /// retain-all posture, which the graded consumers preferred on
    /// both axes over freeing or rematerializing mid-run. A request
    /// that never calls this compiles a forward-only plan, whose runs
    /// refuse `backward`; [`Plan::can_backward`](crate::Plan::can_backward)
    /// answers which kind a plan is.
    pub fn backward(mut self) -> Self {
        self.backward = true;
        self
    }
}
