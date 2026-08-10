use crate::{Network, Tensorial, Value};

use super::{Module, Visitor};

/// A skip connection: `input + inner(input)`, the block shape residual
/// networks and transformers are built from.
///
/// The wrapper is generic and static — it holds exactly one module —
/// and boxes into a [`Sequential`](super::Sequential) like any other
/// stage. It is path-transparent: the inner module's parameters keep
/// their own names, without an extra segment, so checkpoints do not
/// grow a level per skip connection.
pub struct Residual<M>(pub M);

impl<Data: Tensorial, M: Module<Data>> Module<Data> for Residual<M> {
    fn express<'network>(
        &self,
        network: &'network Network<Data>,
        input: Value<'network, Data>,
    ) -> Value<'network, Data> {
        input + self.0.express(network, input)
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        self.0.visit(visitor);
    }
}
