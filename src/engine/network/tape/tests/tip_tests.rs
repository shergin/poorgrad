use std::sync::Arc;

use super::super::{Branch, Segment};
use super::Tip;

/// Builds a one-segment chain rooted at zero, as `Tape::new` does.
fn root_chain() -> Arc<Vec<Segment>> {
    Arc::new(vec![Segment {
        branch: Branch::new(),
        start: 0,
    }])
}

#[test]
fn claim_on_owned_tip_is_a_no_op() {
    let mut tip = Tip::Owned;
    let mut chain = root_chain();
    let before = Arc::clone(&chain);

    tip.claim(&mut chain, 5);

    assert!(matches!(tip, Tip::Owned));
    assert!(Arc::ptr_eq(&chain, &before));
}

#[test]
fn share_from_owned_contends_both_sides_on_one_token() {
    let mut tip = Tip::Owned;
    let sibling = tip.share();

    let (Tip::Contended(mine), Tip::Contended(theirs)) = (&tip, &sibling) else {
        panic!("both sides must contend after a share");
    };
    assert!(Arc::ptr_eq(mine, theirs));
}

#[test]
fn share_from_contended_joins_the_existing_token() {
    let mut tip = Tip::Owned;
    let first = tip.share();
    let second = tip.share();

    let (Tip::Contended(left), Tip::Contended(right)) = (&first, &second) else {
        panic!("shared tips must contend");
    };
    assert!(Arc::ptr_eq(left, right));
}

#[test]
fn winning_claim_continues_the_branch_without_touching_the_chain() {
    let mut tip = Tip::Owned;
    let _sibling = tip.share();
    let mut chain = root_chain();
    let before = Arc::clone(&chain);

    tip.claim(&mut chain, 1);

    assert!(matches!(tip, Tip::Owned));
    assert!(Arc::ptr_eq(&chain, &before));
    assert_eq!(chain.len(), 1);
}

#[test]
fn losing_claim_mints_a_fresh_branch_at_its_own_length() {
    let mut winner = Tip::Owned;
    let mut loser = winner.share();
    let shared = root_chain();
    let mut winner_chain = Arc::clone(&shared);
    let mut loser_chain = Arc::clone(&shared);

    winner.claim(&mut winner_chain, 1);
    loser.claim(&mut loser_chain, 1);

    assert!(matches!(loser, Tip::Owned));
    // The loser copies the shared chain on write and extends only its
    // own copy; the winner's chain stays the shared one, untouched.
    assert!(!Arc::ptr_eq(&loser_chain, &shared));
    assert!(Arc::ptr_eq(&winner_chain, &shared));
    assert_eq!(winner_chain.len(), 1);
    assert_eq!(loser_chain.len(), 2);
    let minted = loser_chain.last().expect("the loser minted a segment");
    assert_eq!(minted.start, 1);
    assert_ne!(minted.branch, loser_chain[0].branch);
}

#[test]
fn exactly_one_of_many_contenders_wins() {
    let mut tips = vec![Tip::Owned];
    for _ in 0..3 {
        let sibling = tips[0].share();
        tips.push(sibling);
    }
    let shared = root_chain();

    let mut winners = 0;
    for tip in &mut tips {
        let mut chain = Arc::clone(&shared);
        tip.claim(&mut chain, 1);
        if chain.len() == 1 {
            winners += 1;
        }
    }
    assert_eq!(winners, 1);
}
