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
// 32 x 32 quadrant as a 4 x 4 grid of 8x8 accumulators. The body is
// shared by two entry points: the generic kernel reads its
// dimensions from the params buffer, and the specialized kernel
// bakes them as function constants per recurring shape — record-once
// training replays a handful of shapes, and baked bounds let the
// compiler unroll and pipeline the K loop (tinygrad's per-shape
// lesson without its codegen stack). The tuning ledger lives in
// notes/gemm-acceleration.md.
constant constexpr uint BM = 64;
constant constexpr uint BN = 64;
constant constexpr uint BK = 16;
constant constexpr uint A_PAD = BK + 4;
constant constexpr uint B_PAD = BN + 4;
constant constexpr uint THREADS = 128;

static inline void gemm_tiled_body(
    device const float* a,
    device const float* b,
    device float* product,
    const uint m,
    const uint n,
    const uint k,
    const uint a_row_stride,
    const uint a_column_stride,
    const uint b_row_stride,
    const uint b_column_stride,
    threadgroup float* a_tile,
    threadgroup float* b_tile,
    threadgroup float* out_tile,
    uint2 group,
    uint lane,
    uint simdgroup_id)
{
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

    for (uint k0 = 0; k0 < k; k0 += BK) {
        // Stage A (BM x BK) and B (BK x BN) cooperatively, reading
        // global memory through the strides and zero-filling outside
        // the matrix, so edge tiles and views need no special case.
        for (uint index = lane; index < BM * BK; index += THREADS) {
            const uint row = index / BK;
            const uint column = index % BK;
            const uint global_row = tile_row + row;
            const uint global_column = k0 + column;
            a_tile[row * A_PAD + column] =
                (global_row < m && global_column < k)
                ? a[global_row * a_row_stride + global_column * a_column_stride]
                : 0.0f;
        }
        for (uint index = lane; index < BK * BN; index += THREADS) {
            const uint row = index / BN;
            const uint column = index % BN;
            const uint global_row = k0 + row;
            const uint global_column = tile_column + column;
            b_tile[row * B_PAD + column] =
                (global_row < k && global_column < n)
                ? b[global_row * b_row_stride + global_column * b_column_stride]
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
        if (global_row < m && global_column < n) {
            product[global_row * n + global_column] = out_tile[row * B_PAD + column];
        }
    }
}

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
    gemm_tiled_body(
        a, b, product,
        params.m, params.n, params.k,
        params.a_row_stride, params.a_column_stride,
        params.b_row_stride, params.b_column_stride,
        a_tile, b_tile, out_tile,
        group, lane, simdgroup_id);
}

// The per-shape constants; a pipeline created without them can only
// be the generic or naive kernel, which never reference them.
constant uint SPEC_M [[function_constant(0)]];
constant uint SPEC_N [[function_constant(1)]];
constant uint SPEC_K [[function_constant(2)]];
constant uint SPEC_A_ROW_STRIDE [[function_constant(3)]];
constant uint SPEC_A_COLUMN_STRIDE [[function_constant(4)]];
constant uint SPEC_B_ROW_STRIDE [[function_constant(5)]];
constant uint SPEC_B_COLUMN_STRIDE [[function_constant(6)]];

kernel void gemm_specialized_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* product [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float a_tile[BM * A_PAD];
    threadgroup float b_tile[BK * B_PAD];
    threadgroup float out_tile[BM * B_PAD];
    gemm_tiled_body(
        a, b, product,
        SPEC_M, SPEC_N, SPEC_K,
        SPEC_A_ROW_STRIDE, SPEC_A_COLUMN_STRIDE,
        SPEC_B_ROW_STRIDE, SPEC_B_COLUMN_STRIDE,
        a_tile, b_tile, out_tile,
        group, lane, simdgroup_id);
}
