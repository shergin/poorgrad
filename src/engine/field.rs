use std::ops::Add;
use std::sync::Arc;

use static_assertions::assert_impl_all;

use crate::{Differentiable, Tensorial};

use super::{Designation, Lineage, Segment, ValueRef, chain_attributes, chains_agree};

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
    lineage: Lineage,
    chain: Arc<Vec<Segment>>,
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
    pub(crate) fn new(lineage: Lineage, chain: Arc<Vec<Segment>>, values: Vec<Data>) -> Self {
        Self {
            lineage,
            chain,
            values,
        }
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
    pub fn of(&self, value: impl ValueRef<Data>) -> &Data {
        let index = match value.designation() {
            Designation::Bound { tape, id } => {
                assert!(
                    self.lineage == tape.lineage(),
                    "value belongs to a different network lineage"
                );
                assert!(
                    tape.agrees_with_chain(&self.chain, self.values.len()),
                    "value belongs to a divergent fork of the network"
                );
                id.index()
            }
            Designation::Named(symbol) => {
                assert!(
                    self.lineage == symbol.lineage,
                    "symbol belongs to a different network lineage"
                );
                let index = symbol.id.index();
                assert!(
                    index < self.values.len(),
                    "symbol was allocated after this field was produced"
                );
                assert!(
                    chain_attributes(&self.chain, symbol.branch, index, self.values.len()),
                    "symbol belongs to a divergent fork of the network"
                );
                index
            }
        };
        self.values
            .get(index)
            .expect("value was allocated after this field was produced")
    }

    /// Returns a field with every entry passed through `transform`.
    pub fn map(&self, transform: impl Fn(&Data) -> Data) -> Self {
        Self {
            lineage: self.lineage,
            chain: Arc::clone(&self.chain),
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
            lineage: self.lineage,
            chain: Arc::clone(&self.chain),
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

    pub(crate) fn chain(&self) -> &Arc<Vec<Segment>> {
        &self.chain
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
        assert!(
            chains_agree(&self.chain, &other.chain, self.values.len()),
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
