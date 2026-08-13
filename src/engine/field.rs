use std::ops::Add;

use static_assertions::assert_impl_all;

use crate::{Differentiable, Tensorial};

use super::{Designation, Kinship, Misbinding, ValueRef};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Field<f64>: Send, Sync);

/// A value-aligned buffer over the nodes captured by a graph snapshot.
///
/// The [`Gradients`] of a backward run are one kind of field. Other fields can
/// hold optimizer state such as momentum or moments, or combine gradients from
/// several runs; an [`Evaluation`](super::Evaluation) holds its forward payloads
/// in one too. Fields carry graph lineage and branch information rather than
/// borrowing one network generation, allowing a compatible field to be reused
/// across parameter updates.
///
/// Field operations require both operands to cover the same number of nodes in
/// compatible branches of the same graph lineage. A field produced before the
/// graph grows still covers its original prefix; accessing a newer node or
/// using that field to update the larger graph is rejected.
#[derive(Debug, Clone)]
pub struct Field<Data> {
    kinship: Kinship,
    values: Vec<Data>,
}

/// The gradients of one backward run: the derivative of the run's target with
/// respect to every node.
///
/// It is an alias rather than a distinct type because gradients *are* a field,
/// the one that differentiation produces, so every field operation applies to
/// them unchanged. Read a single gradient with [`Field::of`], and combine runs
/// or carry optimizer state with the rest of the field algebra. The alias names
/// the role at the API boundary, most visibly on
/// [`Evaluation::backward`](super::Evaluation::backward), while the type keeps
/// the one invariant it actually enforces: alignment to a graph, not
/// differentiation.
pub type Gradients<Data> = Field<Data>;

impl<Data: Differentiable> Field<Data> {
    pub(crate) fn new(kinship: Kinship, values: Vec<Data>) -> Self {
        Self { kinship, values }
    }

    /// Returns the value assigned to the node named by `value` — a
    /// bound [`Value`](super::Value) or a detached
    /// [`Symbol`](super::Symbol).
    ///
    /// A field borrows no tape, so the two forms present different
    /// proofs: a bound value's tape must agree with the field's
    /// branch chain, while a symbol's own lineage, branch, and
    /// position are checked against the chain directly — the
    /// detachment fields were built for.
    ///
    /// # Panics
    /// Panics if `value` belongs to a different lineage or a divergent
    /// fork, or was allocated after this field was produced.
    /// Returns every node's payload in tape order, for the displays
    /// that plot a whole field rather than read one value out of it.
    #[cfg(feature = "evcxr")]
    pub(crate) fn payloads(&self) -> &[Data] {
        &self.values
    }

    pub fn of(&self, value: impl ValueRef<Data>) -> &Data {
        let index = match value.designation() {
            Designation::Bound { tape, id } => {
                assert!(
                    self.kinship.lineage() == tape.lineage(),
                    "value belongs to a different network lineage"
                );
                assert!(
                    tape.agrees_with_chain(self.kinship.chain(), self.values.len()),
                    "value belongs to a divergent fork of the network"
                );
                id.index()
            }
            Designation::Named(symbol) => match self.kinship.probe(symbol, self.values.len()) {
                Ok(id) => id.index(),
                Err(Misbinding::ForeignLineage) => {
                    panic!("symbol belongs to a different network lineage")
                }
                Err(Misbinding::DivergentBranch) => {
                    panic!("symbol belongs to a divergent fork of the network")
                }
                Err(Misbinding::OutOfCoverage) => {
                    panic!("symbol was allocated after this field was produced")
                }
            },
        };
        self.values
            .get(index)
            .expect("value was allocated after this field was produced")
    }

    /// Returns a field with every entry passed through `transform`.
    pub fn map(&self, transform: impl Fn(&Data) -> Data) -> Self {
        Self {
            kinship: self.kinship.clone(),
            values: self.values.iter().map(transform).collect(),
        }
    }

    /// Combines two fields entry by entry with `combine`.
    ///
    /// # Panics
    /// Panics if the fields belong to different lineages or divergent
    /// forks, or cover different numbers of nodes.
    pub fn zip(&self, other: &Self, combine: impl Fn(&Data, &Data) -> Data) -> Self {
        self.assert_kinship(other);
        Self {
            kinship: self.kinship.clone(),
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

    pub(crate) fn kinship(&self) -> &Kinship {
        &self.kinship
    }

    /// Panics if `other` cannot combine with `self`.
    fn assert_kinship(&self, other: &Self) {
        assert!(
            self.kinship.is_family(&other.kinship),
            "fields belong to different network lineages"
        );
        assert_eq!(
            self.values.len(),
            other.values.len(),
            "fields cover different generations of the network"
        );
        assert!(
            self.kinship.agrees_with(&other.kinship, self.values.len()),
            "fields belong to divergent forks of the network"
        );
    }
}

impl<Data: Tensorial> Field<Data> {
    /// Returns a field with every entry multiplied by the single-value
    /// `factor`, spread to each entry's shape.
    ///
    /// It is the scalar arithmetic of optimizer state: bias-correction
    /// and decay factors multiply every parameter's entry regardless of
    /// its shape. For scalar payloads the spread is the identity, so
    /// scalar fields scale exactly as they always did.
    ///
    /// # Panics
    /// For tensor payloads, panics if `factor` holds more than one
    /// value.
    pub fn scale(&self, factor: &Data) -> Self {
        self.map(|value| value.clone() * factor.broadcast_like(value))
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
