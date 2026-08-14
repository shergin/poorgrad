//! Live graph ownership: the public [`Network`], node handles
//! ([`Value`], [`Symbol`], [`ValueRef`]), and the locked tape with
//! structural identity.

mod network;
mod reference;
mod symbol;
mod tape;
mod value;

pub use network::Network;
pub use reference::ValueRef;
pub use symbol::Symbol;
pub use value::Value;

pub(crate) use reference::Designation;
pub(crate) use tape::{Branch, Misbinding, Operands, Origin, SlotId, Tape, Witness};
pub(crate) use value::ValueId;
