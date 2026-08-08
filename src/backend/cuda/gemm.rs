use std::ffi::c_void;

use crate::GemmTask;
use crate::backend::operand::Operand;

use super::context::{Context, DEVICE_TO_HOST, HOST_TO_DEVICE, OP_N, OP_T, cublas_status_name};

/// Returns the `cublasOperation_t` for a classified operand.
fn operation(operand: &Operand) -> i32 {
    if operand.transposed { OP_T } else { OP_N }
}

/// A device buffer on loan from the pool, returned on drop: the
/// success path and every early error return alike, so no path
/// leaks a device allocation. Returning after an error is sound
/// because the caller poisons the module on any error and never
/// issues another task, so an error-parked buffer is never handed
/// out again.
struct Loan<'context> {
    context: &'context Context,
    bytes: usize,
    buffer: *mut c_void,
}

impl<'context> Loan<'context> {
    /// Takes a buffer of at least `bytes` from the context's pool.
    fn take(context: &'context Context, bytes: usize) -> Result<Self, String> {
        let buffer = context.pool.take(&context.api, bytes)?;
        Ok(Self {
            context,
            bytes,
            buffer,
        })
    }
}

impl Drop for Loan<'_> {
    fn drop(&mut self) {
        self.context
            .pool
            .give(&self.context.api, self.bytes, self.buffer);
    }
}

/// It runs one `f32` task on the GPU: pooled device buffers in, two
/// host-to-device copies, one gemm call under the column-major
/// swap, one device-to-host copy out.
///
/// cuBLAS is column-major, so the row-major product `C = A * B` is
/// computed as its transpose: operands swapped, `m` and `n`
/// swapped, each operand keeping exactly the transpose flag and
/// leading dimension its row-major classification produced, and the
/// result laid out row-major with `ldc = n`.
pub(super) fn executed_f32(
    context: &Context,
    task: &GemmTask<'_, f32>,
    a: &Operand,
    b: &Operand,
    m: i32,
    n: i32,
    k: i32,
) -> Result<Vec<f32>, String> {
    let api = &context.api;
    let a_bytes = size_of_val(task.a());
    let b_bytes = size_of_val(task.b());
    let volume = task.m() * task.n();
    let product_bytes = volume * size_of::<f32>();

    let a_device = Loan::take(context, a_bytes)?;
    let b_device = Loan::take(context, b_bytes)?;
    let product_device = Loan::take(context, product_bytes)?;

    // SAFETY: each copy names a live host span of exactly the byte
    // count given and a device buffer the pool sized to at least
    // that count; `cudaMemcpy` on the default stream returns only
    // when the copy is complete.
    let status = unsafe {
        let status = (api.memcpy)(
            a_device.buffer,
            task.a().as_ptr().cast::<c_void>(),
            a_bytes,
            HOST_TO_DEVICE,
        );
        if status != 0 {
            status
        } else {
            (api.memcpy)(
                b_device.buffer,
                task.b().as_ptr().cast::<c_void>(),
                b_bytes,
                HOST_TO_DEVICE,
            )
        }
    };
    if status != 0 {
        return Err(format!(
            "host-to-device copy failed: {}",
            api.error_string(status)
        ));
    }

    let one = 1.0_f32;
    let zero = 0.0_f32;
    // SAFETY: the handle and device pointers are live; the scalar
    // pointers address stack values for the duration of the call
    // (the handle is in host pointer mode, its creation default);
    // the classified leading dimensions describe the operand
    // buffers, whose spans `GemmTask` validated.
    let status = unsafe {
        (api.sgemm)(
            context.handle,
            operation(b),
            operation(a),
            n,
            m,
            k,
            &one,
            b_device.buffer.cast_const().cast::<f32>(),
            b.leading,
            a_device.buffer.cast_const().cast::<f32>(),
            a.leading,
            &zero,
            product_device.buffer.cast::<f32>(),
            n,
        )
    };
    if status != 0 {
        return Err(format!(
            "cublasSgemm failed: {}",
            cublas_status_name(status)
        ));
    }

    let mut product = vec![0.0_f32; volume];
    // SAFETY: the device buffer holds `volume` floats written by the
    // gemm; the host vector is exclusively borrowed at that length;
    // a device-to-host `cudaMemcpy` on the default stream serializes
    // after the gemm and blocks until the bytes are on the host.
    let status = unsafe {
        (api.memcpy)(
            product.as_mut_ptr().cast::<c_void>(),
            product_device.buffer.cast_const(),
            product_bytes,
            DEVICE_TO_HOST,
        )
    };
    if status != 0 {
        return Err(format!(
            "device-to-host copy failed: {}",
            api.error_string(status)
        ));
    }
    // SAFETY: no arguments; surfaces any error the stream deferred.
    let status = unsafe { (api.device_synchronize)() };
    if status != 0 {
        return Err(format!(
            "device synchronization failed: {}",
            api.error_string(status)
        ));
    }

    Ok(product)
}

/// The `f64` twin of [`executed_f32`]; see its safety arguments.
pub(super) fn executed_f64(
    context: &Context,
    task: &GemmTask<'_, f64>,
    a: &Operand,
    b: &Operand,
    m: i32,
    n: i32,
    k: i32,
) -> Result<Vec<f64>, String> {
    let api = &context.api;
    let a_bytes = size_of_val(task.a());
    let b_bytes = size_of_val(task.b());
    let volume = task.m() * task.n();
    let product_bytes = volume * size_of::<f64>();

    let a_device = Loan::take(context, a_bytes)?;
    let b_device = Loan::take(context, b_bytes)?;
    let product_device = Loan::take(context, product_bytes)?;

    // SAFETY: identical to the `f32` twin's copy argument.
    let status = unsafe {
        let status = (api.memcpy)(
            a_device.buffer,
            task.a().as_ptr().cast::<c_void>(),
            a_bytes,
            HOST_TO_DEVICE,
        );
        if status != 0 {
            status
        } else {
            (api.memcpy)(
                b_device.buffer,
                task.b().as_ptr().cast::<c_void>(),
                b_bytes,
                HOST_TO_DEVICE,
            )
        }
    };
    if status != 0 {
        return Err(format!(
            "host-to-device copy failed: {}",
            api.error_string(status)
        ));
    }

    let one = 1.0_f64;
    let zero = 0.0_f64;
    // SAFETY: identical to the `f32` twin's gemm argument.
    let status = unsafe {
        (api.dgemm)(
            context.handle,
            operation(b),
            operation(a),
            n,
            m,
            k,
            &one,
            b_device.buffer.cast_const().cast::<f64>(),
            b.leading,
            a_device.buffer.cast_const().cast::<f64>(),
            a.leading,
            &zero,
            product_device.buffer.cast::<f64>(),
            n,
        )
    };
    if status != 0 {
        return Err(format!(
            "cublasDgemm failed: {}",
            cublas_status_name(status)
        ));
    }

    let mut product = vec![0.0_f64; volume];
    // SAFETY: identical to the `f32` twin's read-back argument.
    let status = unsafe {
        (api.memcpy)(
            product.as_mut_ptr().cast::<c_void>(),
            product_device.buffer.cast_const(),
            product_bytes,
            DEVICE_TO_HOST,
        )
    };
    if status != 0 {
        return Err(format!(
            "device-to-host copy failed: {}",
            api.error_string(status)
        ));
    }
    // SAFETY: no arguments; surfaces any error the stream deferred.
    let status = unsafe { (api.device_synchronize)() };
    if status != 0 {
        return Err(format!(
            "device synchronization failed: {}",
            api.error_string(status)
        ));
    }

    Ok(product)
}
