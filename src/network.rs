use std::ptr;

use super::{Differentiable, Evaluation, Function, Gradients, Operation, Tape, Value};

/// A memory management bag owning the state of every value of one
/// computation graph.
///
/// It is the single place where value state lives: it owns the `Tape` the
/// nodes are recorded on, and every `Value` it hands out is a `Copy` proxy
/// borrowing that tape, unable to outlive the network. An expression such
/// as `let x = v1 + v2;` grows this same network without cloning it and
/// without disturbing anything allocated before; independent state only
/// comes from cloning the network itself, which forks it in O(1) into a
/// fork sharing the underlying arena but keeping an independent tape. All
/// synchronization lives inside the tape's single `Mutex`, taken briefly
/// per operation. The network is `Send + Sync` whenever `Data` is, so
/// scoped threads can build and evaluate the same graph concurrently.
#[derive(Debug)]
pub struct Network<Data> {
    tape: Tape<Data>,
}

impl<Data: Differentiable> Network<Data> {
    /// Creates an empty `Network`.
    pub fn new() -> Self {
        Self { tape: Tape::new() }
    }

    /// Allocates a leaf (a network input or a learnable parameter) and
    /// returns a proxy to it.
    pub fn leaf(&self, data: Data) -> Value<'_, Data> {
        let id = self.tape.record(Function::leaf(data));
        Value::bind(&self.tape, id)
    }

    /// Returns this network's own proxy for the node behind `value`, or
    /// `None` if no node with that position is allocated here.
    ///
    /// Proxies borrow the network that created them, so a proxy taken
    /// before a fork resolves against the original network; `rebind`
    /// produces the equivalent proxy for this network. It checks only the
    /// node's position, so `value` is expected to come from this network or
    /// from a network sharing its history.
    pub fn rebind<'network>(
        &'network self,
        value: Value<'_, Data>,
    ) -> Option<Value<'network, Data>> {
        let id = value.id();
        if id.index() >= self.len() {
            return None;
        }
        Some(Value::bind(&self.tape, id))
    }

    /// Returns the number of allocated values.
    pub fn len(&self) -> usize {
        self.tape.len()
    }

    /// Returns `true` if it holds no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evaluates every node in allocation order, materializing the payload
    /// of each value into a fresh `Evaluation`.
    ///
    /// It replays an O(1) snapshot of the tape, so the network is never
    /// locked during the run and concurrent recordings do not disturb it.
    /// Allocation order is dependency order by construction, which is what
    /// makes the single forward scan sufficient.
    pub fn forward(&self) -> Evaluation<'_, Data> {
        let nodes = self.tape.snapshot();
        let mut values = Vec::with_capacity(nodes.len());
        for function in nodes.iter() {
            let value = function.forward(&values);
            values.push(value);
        }
        Evaluation::new(&self.tape, values)
    }

    /// Propagates gradients backward from `output`, returning the gradient
    /// of `output` with respect to every value.
    ///
    /// It seeds the output gradient with `one_like` and accumulates into a
    /// fresh buffer initialized with `zero_like`, scanning the tape in
    /// reverse allocation order and leaving both the network and the
    /// evaluation untouched. That separation of per-run state from the
    /// shared structure is what lets many threads differentiate the same
    /// network at once.
    ///
    /// # Panics
    /// Panics if `evaluation` or `output` belongs to a different network,
    /// or if the network has grown since `evaluation` ran.
    pub fn backward<'network>(
        &'network self,
        evaluation: &Evaluation<'_, Data>,
        output: Value<'_, Data>,
    ) -> Gradients<'network, Data> {
        assert!(
            ptr::eq(evaluation.tape(), &self.tape),
            "evaluation belongs to a different network"
        );
        assert!(
            ptr::eq(output.tape(), &self.tape),
            "output belongs to a different network"
        );
        let nodes = self.tape.snapshot();
        let values = evaluation.values();
        assert_eq!(
            values.len(),
            nodes.len(),
            "evaluation is stale: the network has grown since it ran"
        );

        let mut gradients: Vec<Data> = values.iter().map(|value| value.zero_like()).collect();
        let output_index = output.id().index();
        gradients[output_index] = values[output_index].one_like();
        for index in (0..nodes.len()).rev() {
            let function = nodes.get(index).expect("snapshot cannot shrink");
            let gradient = gradients[index].clone();
            function.backward(values, &gradient, &mut gradients);
        }
        Gradients::new(&self.tape, gradients)
    }
}

impl<Data: Differentiable> Clone for Network<Data> {
    /// Forks the network in O(1).
    ///
    /// The fork shares the underlying arena but keeps an independent tape:
    /// later allocations on either network never affect the other, while
    /// every node allocated before the fork stays reachable in both.
    fn clone(&self) -> Self {
        Self {
            tape: self.tape.fork(),
        }
    }
}

impl<Data: Differentiable> Default for Network<Data> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests/network_tests.rs"]
mod tests;
