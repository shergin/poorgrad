use super::{Symbol, Tape, Value, ValueId};
use crate::Differentiable;

/// Either handle to a recorded value, accepted by the read and naming
/// surfaces: a generation-bound [`Value`] proxy, or a detached
/// [`Symbol`] name.
///
/// A proxy proves its provenance by carrying its tape, so readers
/// check identity by pointer and skip resolution; a name is validated
/// against the reader's own graph state with the same lineage,
/// branch, and allocation checks
/// [`Network::resolve`](crate::Network::resolve) performs, and the
/// same panic messages. Both forms keep every guarantee — a reference
/// never weakens a check, it only chooses which proof to present.
///
/// The trait is sealed: proxies and names are the closed set of ways
/// to designate a value, and dispatch is a monomorphized match on the
/// form, never a trait object.
pub trait ValueRef<Data>: sealed::Sealed<Data> {}

impl<Data: Differentiable> ValueRef<Data> for Value<'_, Data> {}
impl<Data> ValueRef<Data> for Symbol {}

/// The two designation forms behind [`ValueRef`].
pub(crate) enum Designation<'reference, Data> {
    /// A proxy bound to its tape: identity is proven by pointer.
    Bound {
        tape: &'reference Tape<Data>,
        id: ValueId,
    },
    /// A detached name, validated against the reader's graph state.
    Named(Symbol),
}

pub(crate) mod sealed {
    use super::super::{Symbol, Value};
    use super::Designation;
    use crate::Differentiable;

    pub trait Sealed<Data> {
        /// Returns which designation form this reference presents.
        fn designation(&self) -> Designation<'_, Data>;
    }

    impl<Data: Differentiable> Sealed<Data> for Value<'_, Data> {
        fn designation(&self) -> Designation<'_, Data> {
            Designation::Bound {
                tape: self.tape(),
                id: self.id(),
            }
        }
    }

    impl<Data> Sealed<Data> for Symbol {
        fn designation(&self) -> Designation<'_, Data> {
            Designation::Named(*self)
        }
    }
}

#[cfg(test)]
#[path = "tests/reference_tests.rs"]
mod tests;
