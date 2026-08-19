//! The pattern catalog: a closed set of patterns matched once at
//! compile time and stored as one column on the plan.
//!
//! A pattern is a compile-time match over frozen structure — not a
//! tape rewrite, and not itself a fuse or a raise. The same catalog
//! entry serves two consumers: at home, `Plan::forward` replaces a
//! fusing variant's group with one payload call; abroad,
//! `Plan::emit_stablehlo` raises every group to the named operation
//! the target holds a library kernel for. Matching is structural and
//! provenance-blind, so a hand-rolled equivalent of a facade formula
//! matches identically, and the tape stays the spec throughout.

mod batch_norm;
mod catalog;
mod pattern;
mod reduce_window;
mod view;
mod window;

pub(crate) use batch_norm::BatchNormalization;
pub(crate) use catalog::Catalog;
pub(crate) use pattern::Pattern;
pub(crate) use reduce_window::ReduceWindow;
pub(crate) use view::View;
pub(crate) use window::WindowProduct;
