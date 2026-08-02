use std::ffi::c_void;
use std::ptr::NonNull;

use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLSize,
};

use crate::GemmTask;

use super::context::Context;

/// The kernels' parameter block; layout mirrors `GemmParams` in
/// `shaders/gemm.metal`.
#[repr(C)]
struct GemmParams {
    m: u32,
    n: u32,
    k: u32,
    a_row_stride: u32,
    a_column_stride: u32,
    b_row_stride: u32,
    b_column_stride: u32,
}

/// Which kernel to dispatch; production always tiles, the naive
/// kernel is the correctness anchor the tests compare against.
#[derive(Clone, Copy)]
pub(super) enum Kernel {
    /// The tests' correctness anchor; production always tiles.
    #[cfg_attr(not(test), allow(dead_code))]
    Naive,
    Tiled,
}

/// It runs one task on the GPU: pooled shared buffers in, one
/// synchronous dispatch, product out.
pub(super) fn executed(
    context: &Context,
    task: &GemmTask<'_, f32>,
    kernel: Kernel,
) -> Result<Vec<f32>, String> {
    let m = task.m();
    let n = task.n();
    let params = GemmParams {
        m: m as u32,
        n: n as u32,
        k: task.k() as u32,
        a_row_stride: task.a_strides()[0] as u32,
        a_column_stride: task.a_strides()[1] as u32,
        b_row_stride: task.b_strides()[0] as u32,
        b_column_stride: task.b_strides()[1] as u32,
    };

    let a_buffer = context.pool.take(&context.device, size_of_val(task.a()))?;
    let b_buffer = context.pool.take(&context.device, size_of_val(task.b()))?;
    let product_buffer = context
        .pool
        .take(&context.device, m * n * size_of::<f32>())?;

    // SAFETY: the buffers are shared-mode with `contents()` valid for
    // their whole length, the pool sized each to at least the span
    // being copied, and nothing else aliases them until `give`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            task.a().as_ptr(),
            a_buffer.contents().as_ptr().cast::<f32>(),
            task.a().len(),
        );
        std::ptr::copy_nonoverlapping(
            task.b().as_ptr(),
            b_buffer.contents().as_ptr().cast::<f32>(),
            task.b().len(),
        );
    }

    let command_buffer = context
        .queue
        .commandBuffer()
        .ok_or_else(|| "no command buffer".to_string())?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| "no compute encoder".to_string())?;
    encoder.setComputePipelineState(match kernel {
        Kernel::Naive => &context.naive,
        Kernel::Tiled => &context.tiled,
    });
    // SAFETY: buffer indices match the kernel signature, the buffers
    // outlive the encoder, and the params block is a plain `repr(C)`
    // value alive across the call, copied by `setBytes`.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&a_buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&b_buffer), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&product_buffer), 0, 2);
        encoder.setBytes_length_atIndex(
            NonNull::new(&params as *const GemmParams as *mut c_void)
                .expect("a stack reference is never null"),
            size_of::<GemmParams>(),
            3,
        );
    }
    let (groups, threads) = match kernel {
        Kernel::Naive => (
            MTLSize {
                width: n.div_ceil(16),
                height: m.div_ceil(16),
                depth: 1,
            },
            MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            },
        ),
        Kernel::Tiled => (
            MTLSize {
                width: n.div_ceil(64),
                height: m.div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        ),
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    if command_buffer.status() != MTLCommandBufferStatus::Completed {
        let reason = command_buffer
            .error()
            .map(|error| error.localizedDescription().to_string())
            .unwrap_or_else(|| "command buffer failed without an error".to_string());
        return Err(reason);
    }

    let mut product = vec![0.0_f32; m * n];
    // SAFETY: the dispatch completed, the product buffer is
    // shared-mode and sized at least `m * n` floats, and `product`
    // is exclusively borrowed at exactly that length.
    unsafe {
        std::ptr::copy_nonoverlapping(
            product_buffer.contents().as_ptr().cast::<f32>(),
            product.as_mut_ptr(),
            m * n,
        );
    }

    context.pool.give(a_buffer);
    context.pool.give(b_buffer);
    context.pool.give(product_buffer);
    Ok(product)
}
