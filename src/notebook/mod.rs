//! Rich display for Evcxr notebooks and the Evcxr REPL.
//!
//! The module adds no types and no vocabulary. It implements
//! `evcxr_display` — the method name Evcxr's code generator calls on a
//! cell's final expression — on the types the crate already has, and
//! it adds [`Network::leaked`] and [`Network::leak`], two constructors
//! that hand back a `&'static Network` so recorded proxies survive a
//! cell boundary. Everything a notebook does here is the ordinary API.
//!
//! # Persisting across cells
//!
//! Evcxr compiles every cell as its own crate and keeps the variables
//! between them, which imposes two rules that shape the whole idiom.
//!
//! **A persisted variable cannot borrow another one.** A
//! [`Value`](crate::Value) proxy borrows its network, so a plain
//! `let w = network.parameter(0.0);` is rejected the moment the cell
//! ends. Leaking the network resolves it: a `&'static Network` is
//! borrowed from nothing, so its proxies are `Value<'static, _>` and
//! persist like any other value.
//!
//! ```no_run
//! use poorgrad::{Network, Value};
//!
//! let mut network: &'static Network<f64> = Network::leaked();
//! let w: Value<'static, f64> = network.parameter(0.0);
//! let x: Value<'static, f64> = network.input(0.0);
//! let y: Value<'static, f64> = network.input(0.0);
//! let loss: Value<'static, f64> = (w * x - y) * (w * x - y);
//! ```
//!
//! **A persisted variable needs an explicit type.** Evcxr infers a
//! cell's variable types by compiling that cell alone, and a later
//! cell cannot inform an earlier one. This is a property of Evcxr, not
//! of this crate — it cannot infer `let v = vec![1.0_f64];` either —
//! so annotate every binding that has to survive.
//!
//! # Crossing generations
//!
//! [`Network::update`](crate::Network::update) returns the next
//! generation, and a proxy stays bound to the generation that recorded
//! it. In a notebook that distinction becomes visible, because both
//! generations are still in scope:
//!
//! ```no_run
//! # use poorgrad::{Network, Value};
//! # let mut network: &'static Network<f64> = Network::leaked();
//! # let w: Value<'static, f64> = network.parameter(0.0);
//! # let loss: Value<'static, f64> = w * w;
//! // A training cell keeps one owned generation and leaks once at the
//! // end, so re-running it costs one parameter store, not one per step.
//! let first = network.forward().backward(loss);
//! let mut current = network.update(&first, |p, g| p - 0.02 * g);
//! for _ in 1..300 {
//!     let target = current.resolve(loss.symbol());
//!     let gradients = current.forward().backward(target);
//!     current = current.update(&gradients, |p, g| p - 0.02 * g);
//! }
//! network = current.leak();
//!
//! w.payload();                                 // the generation that recorded it
//! network.resolve(w.symbol()).payload();       // the generation that trained
//! ```
//!
//! Read a trained parameter through its [`Symbol`](crate::Symbol):
//! symbols are detached names that resolve in any compatible
//! generation, which is exactly what a notebook needs and exactly what
//! the generation machinery was built for.
//!
//! # Leaking, honestly
//!
//! [`Network::leaked`] and [`Network::leak`] never free. One leaked
//! generation costs its parameter store; the recorded graph is shared,
//! not copied. A session that leaks once per cell run stays negligible,
//! and a process that ends when the notebook does reclaims all of it.
//! Leaking inside a training loop instead of after it is the one
//! mistake worth naming: that leaks a store per step.
//!
//! # Cell output
//!
//! Every display is a pure `to_html` string plus a three-line
//! `evcxr_display` that emits it. The HTML path serves Jupyter and the
//! `text/plain` path serves the terminal REPL, which cannot draw HTML;
//! Evcxr picks the richest one its frontend supports. Because the
//! strings are pure, they are snapshot-tested like any other output.
//!
//! Supplying `evcxr_display` also makes cells compile once instead of
//! twice: Evcxr tries `(expr).evcxr_display();` first and falls back to
//! a second compile with `Debug` formatting only when that fails.

mod field;
mod html;
mod network;
mod plan;
mod render;
mod tensor;
mod value;
