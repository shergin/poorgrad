use std::thread;

use crate::Network;

#[test]
fn try_resolve_probes_foreign_and_unrecorded_symbols() {
    let network = Network::<f64>::new();
    let foreign = Network::new().leaf(1.0).symbol();
    assert!(network.try_resolve(foreign).is_none());

    let fork = network.clone();
    let late = network.leaf(2.0).symbol();
    assert!(fork.try_resolve(late).is_none());
}

#[test]
#[should_panic(expected = "different network lineage")]
fn resolve_rejects_foreign_symbols() {
    let network = Network::<f64>::new();
    let other = Network::new();
    let foreign = other.leaf(1.0);
    network.resolve(foreign.symbol());
}

#[test]
#[should_panic(expected = "not allocated")]
fn resolve_rejects_unrecorded_symbols() {
    let network = Network::<f64>::new();
    let fork = network.clone();
    let late = network.leaf(1.0);
    fork.resolve(late.symbol());
}

#[test]
fn clone_forks_the_network() {
    let network = Network::new();
    let v1 = network.leaf(1.0_f64);

    let fork = network.clone();
    network.leaf(2.0);

    assert_eq!(network.len(), 2);
    assert_eq!(fork.len(), 1);
    let rebound = fork.resolve(v1.symbol());
    assert_eq!(rebound.data(), Some(1.0));
}

#[test]
#[should_panic(expected = "divergent fork")]
fn resolve_rejects_symbols_from_divergent_forks() {
    let network = Network::<f64>::new();
    let _anchor = network.leaf(0.0);
    let fork = network.clone();

    // The original network records first and keeps the shared branch;
    // the fork's recording lands on a fresh branch at the same index.
    let _mine = network.leaf(1.0);
    let theirs = fork.leaf(2.0);
    network.resolve(theirs.symbol());
}

#[test]
#[should_panic(expected = "not allocated")]
fn resolve_rejects_continued_branch_symbols_on_the_losing_fork() {
    let network = Network::<f64>::new();
    let _anchor = network.leaf(0.0);
    let fork = network.clone();

    // The winner's symbol stays on the shared branch, but past the
    // point where the loser's copy of that branch ends.
    let mine = network.leaf(1.0);
    let _theirs = fork.leaf(2.0);
    fork.resolve(mine.symbol());
}

#[test]
fn divergent_forks_probe_and_share_correctly() {
    let network = Network::<f64>::new();
    let anchor = network.leaf(0.0);
    let fork = network.clone();

    let mine = network.leaf(1.0);
    let theirs = fork.leaf(2.0);

    // Post-divergence symbols probe to `None` across branches, in both
    // directions; the pre-fork symbol keeps resolving on both sides.
    assert!(network.try_resolve(theirs.symbol()).is_none());
    assert!(fork.try_resolve(mine.symbol()).is_none());
    assert_eq!(network.resolve(anchor.symbol()).data(), Some(0.0));
    assert_eq!(fork.resolve(anchor.symbol()).data(), Some(0.0));
}

#[test]
#[should_panic(expected = "divergent fork")]
fn updated_rejects_fields_from_divergent_forks() {
    let network = Network::new();
    let w = network.parameter(1.0_f64);
    let fork = network.clone();

    // Both sides grow to the same length with different nodes.
    let _mine = network.leaf(2.0);
    let _theirs = fork.leaf(3.0);

    let evaluation = network.forward();
    let gradients = evaluation.backward(w);
    fork.updated(gradients.as_field(), |parameter, gradient| {
        parameter - gradient
    });
}

#[test]
fn concurrent_divergence_keeps_branches_exclusive() {
    let network = Network::<f64>::new();
    let anchor = network.leaf(0.0);
    let forks: Vec<_> = (0..4).map(|_| network.clone()).collect();

    // Every fork records concurrently; the tip claim guarantees at most
    // one continues the shared branch.
    let symbols: Vec<_> = thread::scope(|scope| {
        let handles: Vec<_> = forks
            .iter()
            .map(|fork| scope.spawn(|| fork.leaf(1.0).symbol()))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("recording thread panicked"))
            .collect()
    });

    for (owner, symbol) in symbols.iter().enumerate() {
        for (other, fork) in forks.iter().enumerate() {
            if owner == other {
                assert_eq!(fork.resolve(*symbol).data(), Some(1.0));
            } else {
                assert!(fork.try_resolve(*symbol).is_none());
            }
        }
        assert!(network.try_resolve(*symbol).is_none());
    }
    for fork in &forks {
        assert_eq!(fork.resolve(anchor.symbol()).data(), Some(0.0));
    }
}
