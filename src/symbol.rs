use static_assertions::assert_impl_all;

use super::ValueId;

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Symbol: Send, Sync, Copy);

/// A detached, `Copy` name of a value, resolved against network
/// generations.
///
/// It is the identity of a value across time: while `Value` is a
/// generation-bound view, a `Symbol` carries no borrow of any network, so
/// it can be stored anywhere and outlive any particular generation — a
/// training loop keeps the symbols of its loss and parameters while the
/// network variable is reassigned to updated generations. Each generation
/// acts as an environment: `Network::resolve` looks the symbol up in it
/// and returns that generation's `Value`. Resolution is positional and
/// positions are stable across forks and updates, so a symbol taken from
/// one generation resolves to the same node in every related generation;
/// resolving it in an unrelated network is not detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(pub(crate) ValueId);
