use crate::{Tape, Tensorial, Value};

use super::{Module, Segment, Visitor};

/// An ordered chain of modules: each stage's output feeds the next.
///
/// Stages are heterogeneous behind `dyn Module`, which is the
/// sanctioned record-time exception to the static-dispatch rule:
/// expression happens once per topology and its cost never reaches a
/// run, while the static alternative (tuple arities behind macros)
/// cannot hold a depth chosen at runtime. [`Sequential::then`] boxes
/// internally, so call sites never spell `Box`.
pub struct Sequential<Data> {
    stages: Vec<Box<dyn Module<Data>>>,
}

impl<Data: Tensorial> Sequential<Data> {
    /// Creates an empty chain: the identity until stages arrive.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Appends `stage` to the chain and returns it, builder style.
    pub fn then(mut self, stage: impl Module<Data> + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Returns the number of stages.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Returns `true` if the chain holds no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

impl<Data: Tensorial> Default for Sequential<Data> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Data: Tensorial> Module<Data> for Sequential<Data> {
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Data>,
        input: Value<'tape, Data>,
    ) -> Value<'tape, Data> {
        self.stages
            .iter()
            .fold(input, |value, stage| stage.express(tape, value))
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        for (index, stage) in self.stages.iter().enumerate() {
            visitor.enter(Segment::Index(index));
            stage.visit(visitor);
            visitor.leave();
        }
    }
}

#[cfg(test)]
#[path = "tests/sequential_tests.rs"]
mod tests;
