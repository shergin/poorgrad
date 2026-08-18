use std::sync::atomic::{AtomicU64, Ordering};

use static_assertions::assert_impl_all;

// The token is what lets symbols, fields, and parameters cross threads
// detached from any tape, so its thread-safety and `Copy` are
// load-bearing.
assert_impl_all!(Origin: Send, Sync, Copy);

/// An opaque token identifying one tape-network family.
///
/// A [`Tape`](super::Tape) mints its origin at creation and the
/// `into_network`/`into_tape` conversions carry it forward, so every
/// phase of one linear history shares one origin; same-origin checks
/// are plain equality. Being a `Copy` integer rather than a
/// reference-counted token, it rides inside every `Symbol` without
/// costing `Copy`. Because both conversions consume their operand, at
/// most one live tape or network carries an origin at a time, which is
/// why origin plus node position is the whole identity a detached
/// carrier needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Origin(u64);

impl Origin {
    /// Mints a fresh process-globally unique origin.
    ///
    /// `Relaxed` suffices: only uniqueness matters, and the origin
    /// reaches other threads through the structure it identifies.
    pub(crate) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}
