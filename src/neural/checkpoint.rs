//! Module checkpoints: capturing a module tree's parameter payloads
//! and restoring them into a [`Parameters`] state.
//!
//! Two identities, two tiers. The positional pair
//! ([`snapshot`]/[`restore`]) uses the module tree's stable visit
//! order — sufficient for resuming the same code, with no names
//! anywhere. The named pair ([`named_snapshot`]/[`named_restore`])
//! matches by structured [`Path`], which is what survives code
//! evolution and what foreign checkpoints (name-to-tensor maps)
//! require; missing and unexpected paths are loud errors.
//!
//! A checkpoint is pure state, so both directions are plain
//! [`Parameters`] transforms: no graph is touched, and shape
//! mismatches panic through
//! [`Parameters::with_payloads`]'s validation. The library stops at
//! the name-to-payload map; file formats stay at the edge.

use std::collections::HashMap;

use crate::{Parameters, Symbol, Tensorial};

use super::module::{Module, Path, named_parameters, parameters};

/// Returns the payloads of every parameter in `module`'s tree, in
/// visit order: the positional checkpoint.
///
/// # Panics
/// Panics if a visited symbol does not name a parameter `state`
/// carries.
pub fn snapshot<Data: Tensorial, M: Module<Data> + ?Sized>(
    state: &Parameters<Data>,
    module: &M,
) -> Vec<Data> {
    parameters(module)
        .into_iter()
        .map(|symbol| state.of(symbol).clone())
        .collect()
}

/// Returns a new state with `module`'s parameters replaced by
/// `payloads`, matched in visit order: the positional restore.
/// Parameters outside the module keep their payloads.
///
/// # Panics
/// Panics if the payload count differs from the module's parameter
/// count, or if a payload's shape differs from its parameter's.
pub fn restore<Data: Tensorial, M: Module<Data> + ?Sized>(
    state: &Parameters<Data>,
    module: &M,
    payloads: Vec<Data>,
) -> Parameters<Data> {
    let symbols = parameters(module);
    assert_eq!(
        payloads.len(),
        symbols.len(),
        "the checkpoint holds {} payloads but the module has {} parameters",
        payloads.len(),
        symbols.len(),
    );
    state.with_payloads(symbols.into_iter().zip(payloads))
}

/// Returns every parameter payload in `module`'s tree with its
/// structured path, in visit order: the named checkpoint, the form
/// that survives code evolution and maps to foreign layouts.
///
/// # Panics
/// Panics as [`snapshot`] panics.
pub fn named_snapshot<Data: Tensorial, M: Module<Data> + ?Sized>(
    state: &Parameters<Data>,
    module: &M,
) -> Vec<(Path, Data)> {
    named_parameters(module)
        .into_iter()
        .map(|(path, symbol)| (path, state.of(symbol).clone()))
        .collect()
}

/// Returns a new state with `module`'s parameters replaced by
/// `entries`, matched by path: the named restore.
///
/// Tied parameters (one symbol under several paths) take the last
/// matching entry in visit order. Parameters outside the module keep
/// their payloads.
///
/// # Panics
/// Panics if a module parameter has no entry, an entry matches no
/// parameter, or a payload's shape differs from its parameter's.
pub fn named_restore<Data: Tensorial, M: Module<Data> + ?Sized>(
    state: &Parameters<Data>,
    module: &M,
    entries: impl IntoIterator<Item = (Path, Data)>,
) -> Parameters<Data> {
    let mut entries: HashMap<Path, Data> = entries.into_iter().collect();
    let mut replacements: Vec<(Symbol, Data)> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for (path, symbol) in named_parameters(module) {
        match entries.remove(&path) {
            Some(payload) => replacements.push((symbol, payload)),
            None => missing.push(path.to_string()),
        }
    }
    assert!(
        missing.is_empty(),
        "the checkpoint is missing entries for: {}",
        missing.join(", "),
    );
    assert!(
        entries.is_empty(),
        "the checkpoint holds entries no parameter matches: {}",
        entries
            .keys()
            .map(Path::to_string)
            .collect::<Vec<_>>()
            .join(", "),
    );
    state.with_payloads(replacements)
}

#[cfg(test)]
#[path = "tests/checkpoint_tests.rs"]
mod tests;
