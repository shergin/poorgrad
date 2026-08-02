// Dense GEMM kernels for the metal backend: a naive kernel and a
// threadgroup-staged simdgroup-matrix kernel, both exact for any
// shape and any stride pattern. Operands are read through the task's
// strides, so transposed and narrowed views need no host-side
// handling, and both kernels write a contiguous row-major product.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct GemmParams {
    uint m;
    uint n;
    uint k;
    uint a_row_stride;
    uint a_column_stride;
    uint b_row_stride;
    uint b_column_stride;
};

// One thread per output element; any shape, any strides.
kernel void gemm_naive_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* product [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 position [[thread_position_in_grid]])
{
    if (position.x >= params.n || position.y >= params.m) {
        return;
    }
    float total = 0.0f;
    for (uint step = 0; step < params.k; step++) {
        total += a[position.y * params.a_row_stride + step * params.a_column_stride]
            * b[step * params.b_row_stride + position.x * params.b_column_stride];
    }
    product[position.y * params.n + position.x] = total;
}

// The tiled kernel: a 128-thread threadgroup (4 simdgroups in
// 2 x 2) computes a 64 x 64 output tile, staging operand tiles
// through threadgroup memory with guarded, zero-filled loads, so any
// shape and any stride pattern is exact. Each simdgroup owns a
// 32 x 32 quadrant as a 4 x 4 grid of 8x8 accumulators. This is the
// best-measured of the shapes tried (~0.5 TFLOP/s at 2048-square);
// the tuning ledger lives in notes/gemm-acceleration.md.
constant constexpr uint BM = 64;
constant constexpr uint BN = 64;
constant constexpr uint BK = 16;
constant constexpr uint A_PAD = BK + 4;
constant constexpr uint B_PAD = BN + 4;
constant constexpr uint THREADS = 128;

kernel void gemm_tiled_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* product [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float a_tile[BM * A_PAD];
    threadgroup float b_tile[BK * B_PAD];
    threadgroup float out_tile[BM * B_PAD];

    const uint tile_row = group.y * BM;
    const uint tile_column = group.x * BN;
    const uint quadrant_row = (simdgroup_id / 2) * 32;
    const uint quadrant_column = (simdgroup_id % 2) * 32;

    simdgroup_float8x8 accumulator[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            accumulator[i][j] = simdgroup_float8x8(0.0f);
        }
    }

    for (uint k0 = 0; k0 < params.k; k0 += BK) {
        // Stage A (BM x BK) and B (BK x BN) cooperatively, reading
        // global memory through the strides and zero-filling outside
        // the matrix, so edge tiles and views need no special case.
        for (uint index = lane; index < BM * BK; index += THREADS) {
            const uint row = index / BK;
            const uint column = index % BK;
            const uint global_row = tile_row + row;
            const uint global_column = k0 + column;
            a_tile[row * A_PAD + column] =
                (global_row < params.m && global_column < params.k)
                ? a[global_row * params.a_row_stride + global_column * params.a_column_stride]
                : 0.0f;
        }
        for (uint index = lane; index < BK * BN; index += THREADS) {
            const uint row = index / BN;
            const uint column = index % BN;
            const uint global_row = k0 + row;
            const uint global_column = tile_column + column;
            b_tile[row * B_PAD + column] =
                (global_row < params.k && global_column < params.n)
                ? b[global_row * params.b_row_stride + global_column * params.b_column_stride]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; kk += 8) {
            for (uint i = 0; i < 4; i++) {
                simdgroup_float8x8 a_fragment;
                simdgroup_load(
                    a_fragment, a_tile + (quadrant_row + i * 8) * A_PAD + kk, A_PAD);
                for (uint j = 0; j < 4; j++) {
                    simdgroup_float8x8 b_fragment;
                    simdgroup_load(
                        b_fragment, b_tile + kk * B_PAD + quadrant_column + j * 8, B_PAD);
                    simdgroup_multiply_accumulate(
                        accumulator[i][j], a_fragment, b_fragment, accumulator[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Epilogue: stage the tile through threadgroup memory with the
    // sanctioned store intrinsic, then stream it out coalesced and
    // guarded, so edges are exact and no thread_elements layout
    // assumption is made.
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            simdgroup_store(
                accumulator[i][j],
                out_tile + (quadrant_row + i * 8) * B_PAD + quadrant_column + j * 8,
                B_PAD);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = lane; index < BM * BN; index += THREADS) {
        const uint row = index / BN;
        const uint column = index % BN;
        const uint global_row = tile_row + row;
        const uint global_column = tile_column + column;
        if (global_row < params.m && global_column < params.n) {
            product[global_row * params.n + global_column] = out_tile[row * B_PAD + column];
        }
    }
}
