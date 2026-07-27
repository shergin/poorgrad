use std::ops::Add;

use static_assertions::assert_impl_all;

use super::{Differentiable, Lineage, Value};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Field<f64>: Send, Sync);

/// A value-aligned buffer: one payload for every node of a network.
///
/// It is the general form of per-node data, of which a gradient buffer is
/// one role. A field is tied to a network *lineage* rather than to a
/// single generation: positions are stable across forks and updates, so a
/// field can be carried across training steps (a momentum velocity, Adam
/// moments) and combined across runs (averaging data-parallel gradients).
/// The elementwise algebra — `+`, `scaled`, `zip`, `map` — checks kinship
/// (same lineage, same length) on every combination. In physics terms, a
/// `Gradients` is a discrete gradient field over the graph; other fields
/// assign velocities, moments, or learning rates to the same nodes.
#[derive(Debug, Clone)]
pub struct Field<Data> {
    lineage: Lineage,
    values: Vec<Data>,
}

impl<Data: Differentiable> Field<Data> {
    pub(crate) fn new(lineage: Lineage, values: Vec<Data>) -> Self {
        Self { lineage, values }
    }

    /// Returns the value assigned to `value`'s node.
    ///
    /// # Panics
    /// Panics if `value` belongs to a different lineage or was allocated
    /// after this field was produced.
    pub fn of(&self, value: Value<'_, Data>) -> &Data {
        assert!(
            self.lineage == value.tape().lineage(),
            "value belongs to a different network lineage"
        );
        self.values
            .get(value.id().index())
            .expect("value was allocated after this field was produced")
    }

    /// Returns a field with every entry multiplied by `factor`.
    pub fn scaled(&self, factor: Data) -> Self {
        self.map(|value| value.clone() * factor.clone())
    }

    /// Returns a field with every entry passed through `transform`.
    pub fn map(&self, transform: impl Fn(&Data) -> Data) -> Self {
        Self {
            lineage: self.lineage,
            values: self.values.iter().map(transform).collect(),
        }
    }

    /// Combines two fields entry by entry with `combine`.
    ///
    /// # Panics
    /// Panics if the fields belong to different lineages or cover
    /// different numbers of nodes.
    pub fn zip(&self, other: &Self, combine: impl Fn(&Data, &Data) -> Data) -> Self {
        self.assert_kinship(other);
        Self {
            lineage: self.lineage,
            values: self
                .values
                .iter()
                .zip(&other.values)
                .map(|(left, right)| combine(left, right))
                .collect(),
        }
    }

    pub(crate) fn as_slice(&self) -> &[Data] {
        &self.values
    }

    pub(crate) fn lineage(&self) -> Lineage {
        self.lineage
    }

    /// Panics if `other` cannot combine with `self`.
    fn assert_kinship(&self, other: &Self) {
        assert!(
            self.lineage == other.lineage,
            "fields belong to different network lineages"
        );
        assert_eq!(
            self.values.len(),
            other.values.len(),
            "fields cover different generations of the network"
        );
    }
}

impl<Data: Differentiable> Add for &Field<Data> {
    type Output = Field<Data>;

    fn add(self, rhs: Self) -> Field<Data> {
        self.zip(rhs, |left, right| left.clone() + right.clone())
    }
}

impl<Data: Differentiable> Add for Field<Data> {
    type Output = Field<Data>;

    fn add(self, rhs: Self) -> Field<Data> {
        &self + &rhs
    }
}

#[cfg(test)]
#[path = "tests/field_tests.rs"]
mod tests;
