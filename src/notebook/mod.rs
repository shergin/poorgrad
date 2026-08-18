//! Rich display for Evcxr notebooks and the Evcxr REPL.
//!
//! The module adds no types and no vocabulary. It implements
//! `evcxr_display` — the method name Evcxr's code generator calls on a
//! cell's final expression — on the types the crate already has.
//! Everything a notebook does here is the ordinary API.
//!
//! # Persisting across cells
//!
//! Evcxr compiles every cell as its own crate and keeps the variables
//! between them, which imposes two rules that shape the idiom.
//!
//! **A persisted variable cannot borrow another one.** A
//! [`Value`](crate::Value) proxy borrows its tape, so it lives and
//! dies inside one cell; the detached [`Symbol`](crate::Symbol) is
//! the cross-cell currency. End a recording cell with `.symbol()`
//! bindings and reenter through [`Tape::resolve`](crate::Tape::resolve):
//!
//! ```no_run
//! use topos::{Symbol, Tape};
//!
//! let tape: Tape<f64> = Tape::new();
//! let w: Symbol = tape.parameter(0.0).symbol();
//! let x: Symbol = tape.input(0.0).symbol();
//! let y: Symbol = tape.input(0.0).symbol();
//! let loss: Symbol = {
//!     let (w, x, y) = (tape.resolve(w), tape.resolve(x), tape.resolve(y));
//!     ((w * x - y) * (w * x - y)).symbol()
//! };
//! ```
//!
//! **A persisted variable needs an explicit type.** Evcxr infers a
//! cell's variable types by compiling that cell alone, and a later
//! cell cannot inform an earlier one. This is a property of Evcxr, not
//! of this crate — it cannot infer `let v = vec![1.0_f64];` either —
//! so annotate every binding that has to survive.
//!
//! # Sealing and training
//!
//! [`Tape::into_network`](crate::Tape::into_network) consumes the
//! persisted tape — Evcxr tracks the move, so the `tape` variable
//! simply ends where the `network` one begins. Training is pure data
//! over the caller-owned state:
//!
//! ```no_run
//! # use topos::{Network, Parameters, Symbol, Tape};
//! # let tape: Tape<f64> = Tape::new();
//! # let w: Symbol = tape.parameter(0.0).symbol();
//! # let loss: Symbol = { let w = tape.resolve(w); (w * w).symbol() };
//! let network: Network<f64> = tape.into_network();
//! let mut parameters: Parameters<f64> = network.parameters();
//! for _ in 0..300 {
//!     let gradients = network.forward(&parameters, []).backward(loss);
//!     parameters = parameters.step(&gradients, |p, g| p - 0.02 * g);
//! }
//! parameters.of(w);            // the trained payload, by name
//! ```
//!
//! To record more later, reopen with
//! [`Network::into_tape`](crate::Network::into_tape) — another
//! tracked move — and carry the state across with
//! [`Parameters::carried`](crate::Parameters::carried). Symbols keep
//! resolving through every round trip.
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
mod parameters;
mod plan;
mod render;
mod tape;
mod tensor;
mod value;
