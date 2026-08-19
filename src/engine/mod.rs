//! The executor tier: the interpreting [`Run`], the compiled [`Plan`]
//! with its compile [`Request`], and the forward entry points on the
//! sealed spec. Everything here reads the graph tier; nothing below
//! it depends on it.

mod pattern;
mod plan;
mod request;
mod run;

pub use plan::Plan;
pub use request::Request;
pub use run::Run;

pub(crate) use pattern::{Pattern, ReduceWindow, WindowProduct};
pub(crate) use run::Posture;
