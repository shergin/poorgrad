use static_assertions::assert_impl_all;

use super::{Differentiable, Field, Value};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Gradients<f64>: Send, Sync);

/// The gradients of one backward run over a `Network`.
///
/// It holds the derivative of the run's target with respect to every node:
/// the gradient role of a `Field`. Read it per value with `of`, or enter
/// the field algebra with `as_field`/`into_field` to combine gradients
/// across runs and build update directions such as a momentum velocity.
/// Like every field it is tied to a network lineage rather than to a
/// single generation, so it carries no borrow of the network that
/// produced it.
#[derive(Debug, Clone)]
pub struct Gradients<Data> {
    field: Field<Data>,
}

impl<Data: Differentiable> Gradients<Data> {
    pub(crate) fn new(field: Field<Data>) -> Self {
        Self { field }
    }

    /// Returns the gradient of the run's target with respect to `value`.
    ///
    /// # Panics
    /// Panics if `value` belongs to a different lineage or was allocated
    /// after this backward run.
    pub fn of(&self, value: Value<'_, Data>) -> &Data {
        self.field.of(value)
    }

    /// Returns the underlying value-aligned field.
    pub fn as_field(&self) -> &Field<Data> {
        &self.field
    }

    /// Converts into the underlying value-aligned field.
    pub fn into_field(self) -> Field<Data> {
        self.field
    }
}
