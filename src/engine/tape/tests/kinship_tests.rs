use std::sync::Arc;

use crate::engine::{Symbol, ValueId};

use super::super::{Branch, Lineage, Misbinding, Segment};
use super::Kinship;

/// Builds a witness over `segments` with a fresh lineage.
fn kinship(segments: Vec<Segment>) -> Kinship {
    Kinship::new(Lineage::new(), Arc::new(segments))
}

fn segment(branch: Branch, start: usize) -> Segment {
    Segment { branch, start }
}

#[test]
fn agreement_is_limited_to_the_compared_prefix() {
    let root = Branch::new();
    let minted = Branch::new();
    let short = kinship(vec![segment(root, 0)]);
    let long = Kinship::new(
        short.lineage(),
        Arc::new(vec![segment(root, 0), segment(minted, 2)]),
    );

    // The chains disagree only past position 2, so a witness taken
    // before the divergence still agrees over its own prefix.
    assert!(short.agrees_with(&long, 2));
    assert!(long.agrees_with(&short, 2));
    assert!(!short.agrees_with(&long, 3));
}

#[test]
fn foreign_lineages_never_agree() {
    let root = Branch::new();
    let left = kinship(vec![segment(root, 0)]);
    let right = kinship(vec![segment(root, 0)]);

    assert!(!left.is_family(&right));
    assert!(!left.agrees_with(&right, 1));
}

#[test]
fn probe_classifies_every_misbinding() {
    let root = Branch::new();
    let minted = Branch::new();
    let witness = kinship(vec![segment(root, 0), segment(minted, 2)]);
    let named = |lineage, branch, index| Symbol {
        lineage,
        branch,
        id: ValueId(index),
    };

    let owned = named(witness.lineage(), root, 1);
    assert_eq!(witness.probe(owned, 4), Ok(ValueId(1)));

    let foreign = named(Lineage::new(), root, 1);
    assert_eq!(witness.probe(foreign, 4), Err(Misbinding::ForeignLineage));

    let unrelated = named(witness.lineage(), Branch::new(), 1);
    assert_eq!(
        witness.probe(unrelated, 4),
        Err(Misbinding::DivergentBranch)
    );

    let occupied = named(witness.lineage(), root, 2);
    assert_eq!(witness.probe(occupied, 4), Err(Misbinding::DivergentBranch));

    let late = named(witness.lineage(), minted, 4);
    assert_eq!(witness.probe(late, 4), Err(Misbinding::OutOfCoverage));
}
