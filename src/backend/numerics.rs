use std::cell::Cell;

use super::coverage::Bar;

/// The numerics posture of an execution scope: which certification
/// bar a kernel must clear to serve.
///
/// `Exact` demands the bit-identity bar — the reference *bits*, in
/// every build. Today no offer-dispatched kernel clears it, so chain
/// work computes on the built-in reference paths, and the one
/// bit-certified kernel (the fused window product) serves under both
/// postures. `Fast` demands only the envelope bar: the chain as
/// compiled, backends engaging above their per-task thresholds,
/// which are cost heuristics inside this posture, never correctness
/// boundaries.
///
/// The posture is a value, not a build flag: it rides a
/// [`Request`](crate::Request) onto the plan and its runs, so an
/// exact oracle result and a fast result are comparable in one
/// process. The default — for interpreter runs and host-side payload
/// calls outside any run — is `Fast`: enabling a backend feature
/// keeps meaning "use it", and features change speed, never behavior
/// classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Numerics {
    /// Only bit-certified kernels serve: the reference bits, the
    /// same in every build.
    Exact,
    /// The compiled backend chain above its cost thresholds.
    #[default]
    Fast,
}

impl Numerics {
    /// The certification bar this posture demands of every kernel.
    pub fn bar(self) -> Bar {
        match self {
            Numerics::Exact => Bar::BitIdentical,
            Numerics::Fast => Bar::Envelope,
        }
    }
}

thread_local! {
    /// The posture the chain entries consult; written only through
    /// [`NumericsScope`], so it always restores.
    static CURRENT: Cell<Numerics> = const { Cell::new(Numerics::Fast) };
}

/// Returns the posture of the current scope.
pub(crate) fn current() -> Numerics {
    CURRENT.with(Cell::get)
}

/// Installs a posture for the enclosing scope; dropping restores the
/// previous one, so run-scoped postures nest and never leak.
pub(crate) struct NumericsScope {
    previous: Numerics,
}

impl NumericsScope {
    pub(crate) fn enter(numerics: Numerics) -> Self {
        let previous = CURRENT.with(|cell| cell.replace(numerics));
        Self { previous }
    }
}

impl Drop for NumericsScope {
    fn drop(&mut self) {
        CURRENT.with(|cell| cell.set(self.previous));
    }
}
