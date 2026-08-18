use std::ops::{Add, Div, Mul, Sub};

use crate::{Differentiable, Tensor};

use super::Value;

// Coherence forbids the generic reverse (`impl Mul<Value<Data>> for Data`
// leaves the `Data` parameter uncovered), so the foreign scalar payloads
// get concrete implementations instead.
macro_rules! literal_operand_for {
    ($($payload:ty),*) => {$(
        impl<'tape> Add<Value<'tape, $payload>> for $payload {
            type Output = Value<'tape, $payload>;

            fn add(self, rhs: Value<'tape, $payload>) -> Self::Output {
                rhs.literal(self) + rhs
            }
        }

        impl<'tape> Sub<Value<'tape, $payload>> for $payload {
            type Output = Value<'tape, $payload>;

            fn sub(self, rhs: Value<'tape, $payload>) -> Self::Output {
                rhs.literal(self) - rhs
            }
        }

        impl<'tape> Mul<Value<'tape, $payload>> for $payload {
            type Output = Value<'tape, $payload>;

            fn mul(self, rhs: Value<'tape, $payload>) -> Self::Output {
                rhs.literal(self) * rhs
            }
        }

        impl<'tape> Div<Value<'tape, $payload>> for $payload {
            type Output = Value<'tape, $payload>;

            fn div(self, rhs: Value<'tape, $payload>) -> Self::Output {
                rhs.literal(self) / rhs
            }
        }
    )*};
}

literal_operand_for!(f32, f64);

// `Tensor` is local, so its reversed literal operators can stay generic.
impl<'tape, Element: Differentiable> Add<Value<'tape, Tensor<Element>>> for Tensor<Element> {
    type Output = Value<'tape, Tensor<Element>>;

    fn add(self, rhs: Value<'tape, Tensor<Element>>) -> Self::Output {
        rhs.literal(self) + rhs
    }
}

impl<'tape, Element: Differentiable> Sub<Value<'tape, Tensor<Element>>> for Tensor<Element> {
    type Output = Value<'tape, Tensor<Element>>;

    fn sub(self, rhs: Value<'tape, Tensor<Element>>) -> Self::Output {
        rhs.literal(self) - rhs
    }
}

impl<'tape, Element: Differentiable> Mul<Value<'tape, Tensor<Element>>> for Tensor<Element> {
    type Output = Value<'tape, Tensor<Element>>;

    fn mul(self, rhs: Value<'tape, Tensor<Element>>) -> Self::Output {
        rhs.literal(self) * rhs
    }
}

impl<'tape, Element: Differentiable> Div<Value<'tape, Tensor<Element>>> for Tensor<Element> {
    type Output = Value<'tape, Tensor<Element>>;

    fn div(self, rhs: Value<'tape, Tensor<Element>>) -> Self::Output {
        rhs.literal(self) / rhs
    }
}
