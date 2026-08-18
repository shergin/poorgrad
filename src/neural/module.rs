use std::fmt::{self, Display};

use crate::{Symbol, Tape, Tensorial, Value};

/// A named, parameterized recording function: the unit of model
/// composition.
///
/// A module holds its parameters as detached [`Symbol`]s and records
/// its formula through the public operation surface — it never owns
/// payloads (the caller's [`Parameters`](crate::Parameters) do) and
/// never touches the engine.
/// Expression happens at record time, once per topology; the cost
/// never reaches a run, a plan, or a kernel, which is why composing
/// through `dyn Module` (see [`Sequential`](super::Sequential)) sits
/// squarely inside the sanctioned dynamic-dispatch exceptions.
///
/// Programmatic access to a module's parameters — tying, freezing,
/// inspection — goes through its typed accessors (`weights()`,
/// struct fields), never through names. [`Module::visit`] exists for
/// the serialization boundary alone, where checkpoints need stable
/// structured paths.
pub trait Module<Data: Tensorial>: Send + Sync {
    /// Records this module's formula over `input` on `tape` and
    /// returns the output value.
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Data>,
        input: Value<'tape, Data>,
    ) -> Value<'tape, Data>;

    /// Walks this module's parameters and children in a stable order,
    /// announcing each parameter under its local name and each child
    /// under a path segment. Stateless modules do nothing, which is
    /// the default.
    fn visit(&self, visitor: &mut dyn Visitor) {
        let _ = visitor;
    }
}

/// One step of a parameter path: a child's position or a field's
/// static name.
///
/// Paths stay structured everywhere inside the library; dotted text
/// exists only where a serialization format needs text. The `Index`
/// variant exists because positional children (a `Sequential`'s
/// stages) have runtime indices, not compile-time names, so leaf
/// names can stay `&'static str` literals and traversal stays
/// allocation-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Segment {
    /// A positional child, as in a `Sequential`'s stages.
    Index(usize),
    /// A named child or parameter: a static literal the module author
    /// writes once, next to the field it names.
    Name(&'static str),
}

impl Display for Segment {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Segment::Index(index) => write!(formatter, "{index}"),
            Segment::Name(name) => write!(formatter, "{name}"),
        }
    }
}

/// The full structured path of one parameter in a module tree, ending
/// in the parameter's own name. [`Display`] renders the conventional
/// dotted form (`blocks.0.attention.query.weights`) for humans and
/// format adapters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Path(Vec<Segment>);

impl Path {
    /// Returns the path's segments, outermost first.
    pub fn segments(&self) -> &[Segment] {
        &self.0
    }
}

impl Display for Path {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, segment) in self.0.iter().enumerate() {
            if position > 0 {
                write!(formatter, ".")?;
            }
            write!(formatter, "{segment}")?;
        }
        Ok(())
    }
}

/// The traversal callback of [`Module::visit`]: a path-segment stack
/// plus a parameter sink. Concrete walkers — parameter collection,
/// checkpoint save and restore — are implementations.
pub trait Visitor {
    /// Announces one parameter under its local static name.
    fn parameter(&mut self, name: &'static str, symbol: Symbol);

    /// Pushes a path segment for a child's parameters.
    fn enter(&mut self, segment: Segment);

    /// Pops the segment pushed by the matching [`Visitor::enter`].
    fn leave(&mut self);
}

/// Returns the symbols of every parameter in `module`'s tree, in
/// visit order.
pub fn parameters<Data: Tensorial, M: Module<Data> + ?Sized>(module: &M) -> Vec<Symbol> {
    struct Collector(Vec<Symbol>);
    impl Visitor for Collector {
        fn parameter(&mut self, _name: &'static str, symbol: Symbol) {
            self.0.push(symbol);
        }
        fn enter(&mut self, _segment: Segment) {}
        fn leave(&mut self) {}
    }
    let mut collector = Collector(Vec::new());
    module.visit(&mut collector);
    collector.0
}

/// Returns every parameter in `module`'s tree with its structured
/// path, in visit order: the name map of the serialization boundary.
pub fn named_parameters<Data: Tensorial, M: Module<Data> + ?Sized>(
    module: &M,
) -> Vec<(Path, Symbol)> {
    struct Collector {
        stack: Vec<Segment>,
        named: Vec<(Path, Symbol)>,
    }
    impl Visitor for Collector {
        fn parameter(&mut self, name: &'static str, symbol: Symbol) {
            let mut segments = self.stack.clone();
            segments.push(Segment::Name(name));
            self.named.push((Path(segments), symbol));
        }
        fn enter(&mut self, segment: Segment) {
            self.stack.push(segment);
        }
        fn leave(&mut self) {
            self.stack.pop();
        }
    }
    let mut collector = Collector {
        stack: Vec::new(),
        named: Vec::new(),
    };
    module.visit(&mut collector);
    collector.named
}

#[cfg(test)]
#[path = "tests/module_tests.rs"]
mod tests;
