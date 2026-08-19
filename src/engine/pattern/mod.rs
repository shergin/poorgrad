//! The pattern layer: a closed set of patterns discovered once at
//! compile time, from which each consumer elects the entries it can
//! act on.
//!
//! A pattern is a compile-time match over frozen structure — not a
//! tape rewrite. Discovery ([`Candidates::discover`]) is
//! consumer-independent and posture-blind: it pools every closed
//! candidate over the plan's columns, in priority order. Each consumer
//! then elects its own [`Catalog`] under its repertoire — the patterns
//! it supports: a forward run fuses its elected groups into payload
//! calls, and emission raises its elected groups to the named
//! operations the target holds library kernels for. An unelected
//! region simply runs or lowers its recorded primitives, so a pattern
//! is an offer, never an obligation. Matching is structural and
//! provenance-blind — a hand-rolled equivalent of a facade formula
//! matches identically — and the tape stays the spec throughout.

mod batch_norm;
mod candidates;
mod catalog;
mod pattern;
mod reduce_window;
mod view;
mod window;

pub(crate) use batch_norm::BatchNormalization;
pub(crate) use candidates::Candidates;
pub(crate) use catalog::Catalog;
pub(crate) use pattern::Pattern;
pub(crate) use reduce_window::ReduceWindow;
pub(crate) use view::View;
pub(crate) use window::WindowProduct;
