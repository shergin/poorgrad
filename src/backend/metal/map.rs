use std::ffi::c_void;
use std::ptr::NonNull;

use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLSize,
};

use crate::MapOperation;

use super::context::Context;

/// How many threads one map threadgroup holds; every kernel is a
/// plain one-thread-per-element grid.
const THREADGROUP_WIDTH: usize = 256;

/// Returns the pipeline slot of `operation` in
/// [`Context::maps`](super::context::Context), matching the build
/// order in `Context::new`.
pub(super) fn map_pipeline_index(operation: MapOperation) -> usize {
    match operation {
        MapOperation::Exp => 0,
        MapOperation::Ln => 1,
        MapOperation::Sqrt => 2,
        MapOperation::Tanh => 3,
    }
}

/// It runs one elementwise map on the GPU: pooled shared buffers in,
/// one synchronous dispatch, mapped elements out.
pub(super) fn executed(
    context: &Context,
    operation: MapOperation,
    elements: &[f32],
) -> Result<Vec<f32>, String> {
    let count = elements.len();
    let source_buffer = context.pool.take(&context.device, size_of_val(elements))?;
    let destination_buffer = context.pool.take(&context.device, size_of_val(elements))?;

    // SAFETY: the buffers are shared-mode with `contents()` valid for
    // their whole length, the pool sized each to at least the span
    // being copied, and nothing else aliases them until `give`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            elements.as_ptr(),
            source_buffer.contents().as_ptr().cast::<f32>(),
            count,
        );
    }

    let command_buffer = context
        .queue
        .commandBuffer()
        .ok_or_else(|| "no command buffer".to_string())?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| "no compute encoder".to_string())?;
    encoder.setComputePipelineState(&context.maps[map_pipeline_index(operation)]);
    let bound = count as u32;
    // SAFETY: buffer indices match the kernel signature, the buffers
    // outlive the encoder, and the count is a plain `u32` alive
    // across the call, copied by `setBytes`.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&source_buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&destination_buffer), 0, 1);
        encoder.setBytes_length_atIndex(
            NonNull::new(&bound as *const u32 as *mut c_void)
                .expect("a stack reference is never null"),
            size_of::<u32>(),
            2,
        );
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: count.div_ceil(THREADGROUP_WIDTH),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    );
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

    let mut mapped = vec![0.0_f32; count];
    // SAFETY: the dispatch completed, the destination buffer is
    // shared-mode and sized at least `count` floats, and `mapped` is
    // exclusively borrowed at exactly that length.
    unsafe {
        std::ptr::copy_nonoverlapping(
            destination_buffer.contents().as_ptr().cast::<f32>(),
            mapped.as_mut_ptr(),
            count,
        );
    }

    context.pool.give(source_buffer);
    context.pool.give(destination_buffer);
    Ok(mapped)
}
