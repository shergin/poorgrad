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
/// # use poorgrad::{Compile, Memory, Network};
/// # let network = Network::new();
/// # let weight = network.parameter(1.0_f64);
/// # let loss = (weight * weight).sum();
/// // Pure inference: a forward-only plan over one root.
/// let inference = network.compile(Compile::roots([loss]));
///
/// // Engine training: run buffers retain what `backward` reads.
/// let training = network.compile(
///     Compile::roots([loss]).engine_backward(Memory::Retain),
/// );
/// assert!(!inference.can_backward());
/// assert!(training.can_backward());
/// ```
#[derive(Debug, Clone)]
pub struct Compile<Data> {
    pub(crate) roots: Vec<Symbol>,
    pub(crate) observe: Vec<Symbol>,
    pub(crate) engine_backward: Option<Memory>,
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
            engine_backward: None,
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

    /// Requests engine reverse mode: run buffers retain or
    /// rematerialize what [`Run::backward`](crate::Run::backward)
    /// reads, per `memory`. A request that never calls this compiles
    /// a forward-only plan, whose runs refuse `backward`.
    pub fn engine_backward(mut self, memory: Memory) -> Self {
        self.engine_backward = Some(memory);
        self
    }
}

/// The forward-value memory policy of an engine-backward plan: what a
/// run holds for `backward`, chosen explicitly at the compile call
/// site.
///
/// The policy is a closed set of alternatives, so it is a plain
/// `Copy` enum parameter, with each variant's measured trade
/// documented where it is chosen. It changes what a run *stores*,
/// never what it computes: every posture is bit-exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Memory {
    /// Hold every closure value: the fastest and, on the measured
    /// consumers, usually the smallest-RSS choice, since the
    /// allocator recycles the uniform per-step cycle perfectly;
    /// `describe` reports the release floor the analysis licenses.
    Retain,
    /// Trade backward time for memory: large intermediates (the
    /// im2col patches, padded copies, and pooling lanes at or above
    /// the allocator's page-returning size class) are dropped right
    /// after their last forward consumer and rematerialized on demand
    /// during `backward`, bit-exactly. The trade does not always win:
    /// on the MNIST example it cut peak RSS 9% below retain-all for
    /// 22% more step time, while on the deeper CIFAR-10 example it
    /// cost time *and* memory (gradient cotangent buffers, not
    /// forward values, dominate there — their eviction is future work
    /// that may flip the default). Reach for it when activations, not
    /// gradients, are what does not fit.
    Remat,
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
