use std::marker::PhantomData;

use crate::Differentiable;

use super::{Designation, Symbol, Value, ValueRef};

/// A compile request: the explicit product of what a plan computes.
///
/// The request names its roots (what must be computed — a loss, a
/// logits head, recorded gradient symbols; no root is special), the
/// extra interior values to observe (readable after a run, alongside
/// the roots), and whether run buffers support the engine reverse
/// scan ([`Compile::engine_backward`]). The builder never touches the
/// tape: recorded gradients enter as ordinary roots, produced by a
/// visible [`Network::differentiate`](crate::Network::differentiate)
/// beforehand, so a request is cheap and re-runnable.
///
/// # Examples
/// ```
/// # use poorgrad::{Compile, Network};
/// # let network = Network::new();
/// # let weight = network.parameter(1.0_f64);
/// # let loss = (weight * weight).sum();
/// // Pure inference: a forward-only plan over one root.
/// let inference = network.compile(Compile::roots([loss]));
///
/// // Engine training: run buffers retain what `backward` reads.
/// let training = network.compile(Compile::roots([loss]).engine_backward());
/// assert!(!inference.can_backward());
/// assert!(training.can_backward());
/// ```
#[derive(Debug, Clone)]
pub struct Compile<Data> {
    pub(crate) roots: Vec<Symbol>,
    pub(crate) observe: Vec<Symbol>,
    pub(crate) engine_backward: bool,
    /// The payload type the request compiles for: roots are detached
    /// `Symbol`s, so nothing else pins `Data` until the network does.
    payload: PhantomData<fn() -> Data>,
}

impl<Data: Differentiable> Compile<Data> {
    /// Opens a request over `roots`, the closure sources a run must
    /// compute; every root is readable after a run. References
    /// detach to [`Symbol`]s immediately and are validated against
    /// the network when [`Network::compile`](crate::Network::compile)
    /// resolves them.
    pub fn roots(roots: impl IntoIterator<Item = impl ValueRef<Data>>) -> Self {
        Self {
            roots: roots.into_iter().map(detach).collect(),
            observe: Vec::new(),
            engine_backward: false,
            payload: PhantomData,
        }
    }

    /// Adds interior values the caller also wants readable after a
    /// run; like roots, they seed the plan's reachability closure.
    /// Repeated calls accumulate.
    pub fn observe(mut self, extra: impl IntoIterator<Item = impl ValueRef<Data>>) -> Self {
        self.observe.extend(extra.into_iter().map(detach));
        self
    }

    /// Requests engine reverse mode: run buffers retain what
    /// [`Run::backward`](crate::Run::backward) reads — the retain-all
    /// posture, which the graded consumers preferred on both axes
    /// over freeing or rematerializing mid-run. A request that never
    /// calls this compiles a forward-only plan, whose runs refuse
    /// `backward`.
    pub fn engine_backward(mut self) -> Self {
        self.engine_backward = true;
        self
    }
}

/// Returns the detached name of `reference`: a symbol passes through,
/// a bound proxy detaches through its own tape. Lineage and branch
/// validation happens where the symbol is resolved, at compile time.
fn detach<Data: Differentiable>(reference: impl ValueRef<Data>) -> Symbol {
    match reference.designation() {
        Designation::Bound { tape, id } => Value::bind(tape, id).symbol(),
        Designation::Named(symbol) => symbol,
    }
}
