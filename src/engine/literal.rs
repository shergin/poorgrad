use std::ops::{Add, Div, Mul, Sub};

use crate::{Differentiable, Tensor};

use super::Value;

// Coherence forbids the generic reverse (`impl Mul<Value<Data>> for Data`
// leaves the `Data` parameter uncovered), so the foreign scalar payloads
// get concrete implementations instead.
macro_rules! literal_operand_for {
    ($($payload:ty),*) => {$(
        impl<'network> Add<Value<'network, $payload>> for $payload {
            type Output = Value<'network, $payload>;

            fn add(self, rhs: Value<'network, $payload>) -> Self::Output {
                rhs.literal(self) + rhs
            }
        }

        impl<'network> Sub<Value<'network, $payload>> for $payload {
            type Output = Value<'network, $payload>;

            fn sub(self, rhs: Value<'network, $payload>) -> Self::Output {
                rhs.literal(self) - rhs
            }
        }

        impl<'network> Mul<Value<'network, $payload>> for $payload {
            type Output = Value<'network, $payload>;

            fn mul(self, rhs: Value<'network, $payload>) -> Self::Output {
                rhs.literal(self) * rhs
            }
        }

        impl<'network> Div<Value<'network, $payload>> for $payload {
            type Output = Value<'network, $payload>;

            fn div(self, rhs: Value<'network, $payload>) -> Self::Output {
                rhs.literal(self) / rhs
            }
        }
    )*};
}

literal_operand_for!(f32, f64);

// `Tensor` is local, so its reversed literal operators can stay generic.
impl<'network, Element: Differentiable> Add<Value<'network, Tensor<Element>>> for Tensor<Element> {
    type Output = Value<'network, Tensor<Element>>;

    fn add(self, rhs: Value<'network, Tensor<Element>>) -> Self::Output {
        rhs.literal(self) + rhs
    }
}

impl<'network, Element: Differentiable> Sub<Value<'network, Tensor<Element>>> for Tensor<Element> {
    type Output = Value<'network, Tensor<Element>>;

    fn sub(self, rhs: Value<'network, Tensor<Element>>) -> Self::Output {
        rhs.literal(self) - rhs
    }
}

impl<'network, Element: Differentiable> Mul<Value<'network, Tensor<Element>>> for Tensor<Element> {
    type Output = Value<'network, Tensor<Element>>;

    fn mul(self, rhs: Value<'network, Tensor<Element>>) -> Self::Output {
        rhs.literal(self) * rhs
    }
}

impl<'network, Element: Differentiable> Div<Value<'network, Tensor<Element>>> for Tensor<Element> {
    type Output = Value<'network, Tensor<Element>>;

    fn div(self, rhs: Value<'network, Tensor<Element>>) -> Self::Output {
        rhs.literal(self) / rhs
    }
}
