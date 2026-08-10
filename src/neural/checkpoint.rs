//! Module checkpoints: capturing a module tree's parameter payloads
//! and restoring them into a network generation.
//!
//! Two identities, two tiers. The positional pair
//! ([`snapshot`]/[`restore`]) uses the module tree's stable visit
//! order — sufficient for resuming the same code, with no names
//! anywhere. The named pair ([`named_snapshot`]/[`named_restore`])
//! matches by structured [`Path`], which is what survives code
//! evolution and what foreign checkpoints (name-to-tensor maps)
//! require; missing and unexpected paths are loud errors.
//!
//! Restoring never mutates: it builds a new network generation
//! through [`Network::update_each`], so shape mismatches panic
//! through the update's existing validation, and the old generation
//! stays fully usable. The library stops at the name-to-payload map;
//! file formats stay at the edge.

use std::collections::HashMap;

use crate::{Network, Symbol, Tensorial, Value};

use super::module::{Module, Path, named_parameters, parameters};

/// Returns the payloads of every parameter in `module`'s tree, in
/// visit order: the positional checkpoint.
///
/// # Panics
/// Panics if a visited symbol does not resolve in this generation or
/// does not name a parameter, leaf, or input.
pub fn snapshot<Data: Tensorial, M: Module<Data> + ?Sized>(
    network: &Network<Data>,
    module: &M,
) -> Vec<Data> {
    parameters(module)
        .into_iter()
        .map(|symbol| {
            network
                .resolve(symbol)
                .payload()
                .expect("a module parameter stores a payload")
        })
        .collect()
}

/// Returns a new network generation with `module`'s parameters
/// replaced by `payloads`, matched in visit order: the positional
/// restore. Parameters outside the module keep their payloads.
///
/// # Panics
/// Panics if the payload count differs from the module's parameter
/// count, or if a payload's shape differs from its parameter's
/// recorded shape.
pub fn restore<Data: Tensorial, M: Module<Data> + ?Sized>(
    network: &Network<Data>,
    module: &M,
    payloads: Vec<Data>,
) -> Network<Data> {
    let symbols = parameters(module);
    assert_eq!(
        payloads.len(),
        symbols.len(),
        "the checkpoint holds {} payloads but the module has {} parameters",
        payloads.len(),
        symbols.len(),
    );
    let replacements: HashMap<Symbol, Data> = symbols.into_iter().zip(payloads).collect();
    next_generation(network, replacements)
}

/// Returns every parameter payload in `module`'s tree with its
/// structured path, in visit order: the named checkpoint, the form
/// that survives code evolution and maps to foreign layouts.
///
/// # Panics
/// Panics as [`snapshot`] panics.
pub fn named_snapshot<Data: Tensorial, M: Module<Data> + ?Sized>(
    network: &Network<Data>,
    module: &M,
) -> Vec<(Path, Data)> {
    named_parameters(module)
        .into_iter()
        .map(|(path, symbol)| {
            let payload = network
                .resolve(symbol)
                .payload()
                .expect("a module parameter stores a payload");
            (path, payload)
        })
        .collect()
}

/// Returns a new network generation with `module`'s parameters
/// replaced by `entries`, matched by path: the named restore.
///
/// Tied parameters (one symbol under several paths) take the last
/// matching entry in visit order. Parameters outside the module keep
/// their payloads.
///
/// # Panics
/// Panics if a module parameter has no entry, an entry matches no
/// parameter, or a payload's shape differs from its parameter's
/// recorded shape.
pub fn named_restore<Data: Tensorial, M: Module<Data> + ?Sized>(
    network: &Network<Data>,
    module: &M,
    entries: impl IntoIterator<Item = (Path, Data)>,
) -> Network<Data> {
    let mut entries: HashMap<Path, Data> = entries.into_iter().collect();
    let mut replacements: HashMap<Symbol, Data> = HashMap::new();
    let mut missing: Vec<String> = Vec::new();
    for (path, symbol) in named_parameters(module) {
        match entries.remove(&path) {
            Some(payload) => {
                replacements.insert(symbol, payload);
            }
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
    next_generation(network, replacements)
}

/// Builds the generation carrying `replacements`, leaving every other
/// parameter's payload unchanged.
///
/// The update needs a direction field; an empty `recorded_gradients`
/// over a parameter-sliced run supplies the all-zeros one without any
/// engine addition, and the rule ignores it.
fn next_generation<Data: Tensorial>(
    network: &Network<Data>,
    mut replacements: HashMap<Symbol, Data>,
) -> Network<Data> {
    let targets: Vec<Symbol> = replacements.keys().copied().collect();
    let evaluation = network.forward_for(targets, []);
    let none: [(Value<'_, Data>, Value<'_, Data>); 0] = [];
    let zeros = evaluation.recorded_gradients(none);
    network.update_each(&zeros, |value, current, _direction| {
        replacements
            .remove(&value.symbol())
            .unwrap_or_else(|| current.clone())
    })
}

#[cfg(test)]
#[path = "tests/checkpoint_tests.rs"]
mod tests;
