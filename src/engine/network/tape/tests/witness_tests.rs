use std::sync::Arc;

use crate::engine::{Symbol, ValueId};

use super::super::{Branch, Misbinding, Origin, Segment};
use super::Witness;

/// Builds a witness over `segments` with a fresh origin.
fn witness(segments: Vec<Segment>) -> Witness {
    Witness::new(Origin::new(), Arc::new(segments))
}

fn segment(branch: Branch, start: usize) -> Segment {
    Segment { branch, start }
}

#[test]
fn agreement_is_limited_to_the_compared_prefix() {
    let root = Branch::new();
    let minted = Branch::new();
    let short = witness(vec![segment(root, 0)]);
    let long = Witness::new(
        short.origin(),
        Arc::new(vec![segment(root, 0), segment(minted, 2)]),
    );

    // The chains disagree only past position 2, so a witness taken
    // before the divergence still agrees over its own prefix.
    assert!(short.agrees_with(&long, 2));
    assert!(long.agrees_with(&short, 2));
    assert!(!short.agrees_with(&long, 3));
}

#[test]
fn foreign_origins_never_agree() {
    let root = Branch::new();
    let left = witness(vec![segment(root, 0)]);
    let right = witness(vec![segment(root, 0)]);

    assert!(!left.same_origin(&right));
    assert!(!left.agrees_with(&right, 1));
}

#[test]
fn probe_classifies_every_misbinding() {
    let root = Branch::new();
    let minted = Branch::new();
    let proof = witness(vec![segment(root, 0), segment(minted, 2)]);
    let named = |origin, branch, index| Symbol {
        origin,
        branch,
        id: ValueId(index),
    };

    let owned = named(proof.origin(), root, 1);
    assert_eq!(proof.probe(owned, 4), Ok(ValueId(1)));

    let foreign = named(Origin::new(), root, 1);
    assert_eq!(proof.probe(foreign, 4), Err(Misbinding::ForeignOrigin));

    let unrelated = named(proof.origin(), Branch::new(), 1);
    assert_eq!(proof.probe(unrelated, 4), Err(Misbinding::DivergentBranch));

    let occupied = named(proof.origin(), root, 2);
    assert_eq!(proof.probe(occupied, 4), Err(Misbinding::DivergentBranch));

    let late = named(proof.origin(), minted, 4);
    assert_eq!(proof.probe(late, 4), Err(Misbinding::OutOfCoverage));
}
