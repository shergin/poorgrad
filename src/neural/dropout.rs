use std::marker::PhantomData;

use static_assertions::assert_impl_all;

use crate::{Differentiable, Network, Shape, Symbol, Tensorial, Value};

use super::Module;

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Dropout<f64>: Send, Sync);

/// A mask-fed dropout: the expression multiplies its input by a
/// declared mask input whose default payload is all ones, so an unfed
/// run is the identity — inference is the absence of a feed, not a
/// mode.
///
/// Randomness stays outside the graph: masks are generated host-side
/// by the seeded [`init::dropout`](super::init::dropout) factory
/// (inverted dropout, each element `0` or `1 / keep`) and fed per
/// training step like any other input. That keeps seeded runs
/// bit-identical, gradients exact (`d input = gradient * mask`, and
/// the mask edge carries no gradient of its own), and the emitted
/// form of a training step just one more dynamic argument.
///
/// The module holds only the mask input's [`Symbol`]; the keep
/// probability is caller territory, chosen where the mask is drawn.
/// It carries no parameters, so [`Module::visit`] keeps its stateless
/// default.
#[derive(Debug, Clone)]
pub struct Dropout<Data> {
    mask: Symbol,
    _marker: PhantomData<Data>,
}

impl<Data: Differentiable> Dropout<Data> {
    /// Declares the mask input on `network`, shaped like the values
    /// the module will express over, with the all-ones identity
    /// default.
    pub fn new(network: &Network<Data>, shape: impl Into<Shape>) -> Self {
        let mask = network.input(Data::counted(shape.into(), 1));
        Self {
            mask: mask.symbol(),
            _marker: PhantomData,
        }
    }

    /// Returns the mask input's symbol, for the training loop's feed
    /// pairs.
    pub fn mask(&self) -> Symbol {
        self.mask
    }

    /// Records the masked value: `input * mask`.
    ///
    /// # Panics
    /// Panics if `input`'s shape differs from the declared mask
    /// shape, or if the module's mask does not resolve in `network`'s
    /// generation.
    pub fn express<'network>(
        &self,
        network: &'network Network<Data>,
        input: Value<'network, Data>,
    ) -> Value<'network, Data> {
        input * network.resolve(self.mask)
    }
}

impl<Data: Tensorial> Module<Data> for Dropout<Data> {
    fn express<'network>(
        &self,
        network: &'network Network<Data>,
        input: Value<'network, Data>,
    ) -> Value<'network, Data> {
        Dropout::express(self, network, input)
    }
}

#[cfg(test)]
#[path = "tests/dropout_tests.rs"]
mod tests;
