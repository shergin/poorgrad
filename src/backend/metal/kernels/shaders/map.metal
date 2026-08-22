// The elementwise map kernels: one thread per element, one kernel per
// transcendental, compiled with fast math off for parity with the CPU
// paths. The count rides as a small constant so a grid rounded up to
// the threadgroup width never writes past the buffer.

#include <metal_stdlib>

using namespace metal;

kernel void map_exp_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        destination[index] = exp(source[index]);
    }
}

kernel void map_ln_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        destination[index] = log(source[index]);
    }
}

kernel void map_sqrt_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        destination[index] = sqrt(source[index]);
    }
}

kernel void map_tanh_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        destination[index] = tanh(source[index]);
    }
}
