use std::cell::Cell;

/// The numerics posture of an execution scope: whether the compiled
/// backend chain may take tasks.
///
/// `Exact` makes every chain entry decline, so all work computes on
/// the built-in reference paths — bit-identical to the default build,
/// in every build. `Fast` is the chain as compiled: backends engage
/// above their per-task thresholds, which are cost heuristics inside
/// this posture, never correctness boundaries.
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
    /// The backend chain declines every task: reference kernels only,
    /// the same bits as the default build.
    Exact,
    /// The compiled backend chain above its cost thresholds.
    #[default]
    Fast,
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
