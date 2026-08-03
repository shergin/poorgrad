use std::ffi::{CStr, c_char, c_void};
use std::fmt;

use libloading::Library;

use super::pool::Pool;

/// `cudaMemcpyKind` constants for the one copy entry point.
pub(super) const HOST_TO_DEVICE: i32 = 1;
pub(super) const DEVICE_TO_HOST: i32 = 2;

/// `cublasOperation_t` constants.
pub(super) const OP_N: i32 = 0;
pub(super) const OP_T: i32 = 1;

/// The `cudaError_t` codes the setup distinguishes: everything else
/// is reported by its message.
const NO_DEVICE: i32 = 100;
const INSUFFICIENT_DRIVER: i32 = 35;

/// Why the one-time setup failed.
///
/// The variants matter to different audiences: a machine without
/// the NVIDIA libraries or without a device is an expected
/// environment where the backend declines and the GPU tests skip,
/// while every other failure is a broken backend that the tests
/// must report loudly.
#[derive(Debug)]
pub(super) enum SetupError {
    NoLibrary(&'static str),
    NoDevice,
    Failed(String),
}

impl fmt::Display for SetupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLibrary(name) => write!(formatter, "`{name}` is not available"),
            Self::NoDevice => write!(formatter, "no CUDA device"),
            Self::Failed(reason) => write!(formatter, "{reason}"),
        }
    }
}

/// The dlopen-resolved API surface: plain C function pointers
/// copied out of their libraries, which stay open alongside them.
///
/// Only what the arm calls at run time is stored; setup-only
/// symbols (device count, device selection, handle creation) are
/// resolved and dropped inside [`Context::new`].
pub(super) struct Api {
    /// Held open so the copied function pointers stay valid; the
    /// context lives in a process-wide static and is never dropped,
    /// so the libraries are never unloaded.
    _cudart: Library,
    _cublas: Library,
    pub(super) malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> i32,
    pub(super) memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32,
    pub(super) device_synchronize: unsafe extern "C" fn() -> i32,
    get_error_string: unsafe extern "C" fn(i32) -> *const c_char,
    pub(super) sgemm: unsafe extern "C" fn(
        *mut c_void,
        i32,
        i32,
        i32,
        i32,
        i32,
        *const f32,
        *const f32,
        i32,
        *const f32,
        i32,
        *const f32,
        *mut f32,
        i32,
    ) -> i32,
    pub(super) dgemm: unsafe extern "C" fn(
        *mut c_void,
        i32,
        i32,
        i32,
        i32,
        i32,
        *const f64,
        *const f64,
        i32,
        *const f64,
        i32,
        *const f64,
        *mut f64,
        i32,
    ) -> i32,
}

impl Api {
    /// Renders a `cudaError_t` through the runtime's own message
    /// table.
    pub(super) fn error_string(&self, status: i32) -> String {
        // SAFETY: `cudaGetErrorString` returns a pointer to a static
        // NUL-terminated message for every status value.
        let message = unsafe { CStr::from_ptr((self.get_error_string)(status)) };
        message.to_string_lossy().into_owned()
    }
}

/// Renders a `cublasStatus_t` by name; cuBLAS added its own string
/// function too late in the 11.x line to rely on.
pub(super) fn cublas_status_name(status: i32) -> String {
    let name = match status {
        1 => "CUBLAS_STATUS_NOT_INITIALIZED",
        3 => "CUBLAS_STATUS_ALLOC_FAILED",
        7 => "CUBLAS_STATUS_INVALID_VALUE",
        8 => "CUBLAS_STATUS_ARCH_MISMATCH",
        11 => "CUBLAS_STATUS_MAPPING_ERROR",
        13 => "CUBLAS_STATUS_EXECUTION_FAILED",
        14 => "CUBLAS_STATUS_INTERNAL_ERROR",
        15 => "CUBLAS_STATUS_NOT_SUPPORTED",
        16 => "CUBLAS_STATUS_LICENSE_ERROR",
        other => return format!("cublas status {other}"),
    };
    name.to_string()
}

/// The backend's one-time state: the loaded libraries, the resolved
/// entry points, the cuBLAS handle, and the device buffer pool.
///
/// Built lazily on the first eligible task or `status` call; any
/// failure is the `SetupError` the diagnostics report.
pub(super) struct Context {
    pub(super) api: Api,
    pub(super) handle: *mut c_void,
    pub(super) pool: Pool,
}

// SAFETY: NVIDIA documents the CUDA runtime API as thread-safe and
// cuBLAS as callable from multiple host threads sharing one handle,
// provided its configuration is not mutated concurrently — this
// handle is configured once at creation and only ever passed to
// gemm calls. The raw pointers are opaque tokens owned by the
// libraries; the pool guards its own state with a `Mutex`.
#[allow(unsafe_code)]
unsafe impl Send for Context {}
#[allow(unsafe_code)]
unsafe impl Sync for Context {}

impl Context {
    /// Opens the runtime and cuBLAS libraries, probes for a device,
    /// creates the handle, and builds the pool.
    pub(super) fn new() -> Result<Self, SetupError> {
        let cudart = open(&["libcudart.so.13", "libcudart.so.12", "libcudart.so"])
            .ok_or(SetupError::NoLibrary("libcudart"))?;
        let cublas = open(&["libcublas.so.13", "libcublas.so.12", "libcublas.so"])
            .ok_or(SetupError::NoLibrary("libcublas"))?;

        let get_device_count: unsafe extern "C" fn(*mut i32) -> i32 =
            symbol(&cudart, b"cudaGetDeviceCount\0", "libcudart")?;
        let set_device: unsafe extern "C" fn(i32) -> i32 =
            symbol(&cudart, b"cudaSetDevice\0", "libcudart")?;
        let get_error_string: unsafe extern "C" fn(i32) -> *const c_char =
            symbol(&cudart, b"cudaGetErrorString\0", "libcudart")?;
        let create: unsafe extern "C" fn(*mut *mut c_void) -> i32 =
            symbol(&cublas, b"cublasCreate_v2\0", "libcublas")?;

        let mut count = 0_i32;
        // SAFETY: the pointer addresses a live `i32` for the call.
        let status = unsafe { get_device_count(&mut count) };
        if status == NO_DEVICE || status == INSUFFICIENT_DRIVER || (status == 0 && count == 0) {
            return Err(SetupError::NoDevice);
        }
        if status != 0 {
            // SAFETY: see `Api::error_string`; the table is static.
            let message = unsafe { CStr::from_ptr(get_error_string(status)) };
            return Err(SetupError::Failed(format!(
                "cudaGetDeviceCount failed: {}",
                message.to_string_lossy()
            )));
        }

        // SAFETY: device 0 exists per the count above.
        let status = unsafe { set_device(0) };
        if status != 0 {
            // SAFETY: see `Api::error_string`; the table is static.
            let message = unsafe { CStr::from_ptr(get_error_string(status)) };
            return Err(SetupError::Failed(format!(
                "cudaSetDevice failed: {}",
                message.to_string_lossy()
            )));
        }

        let mut handle = std::ptr::null_mut();
        // SAFETY: the pointer addresses a live slot for the handle.
        let status = unsafe { create(&mut handle) };
        if status != 0 {
            return Err(SetupError::Failed(format!(
                "cublasCreate failed: {}",
                cublas_status_name(status)
            )));
        }

        let api = Api {
            malloc: symbol(&cudart, b"cudaMalloc\0", "libcudart")?,
            memcpy: symbol(&cudart, b"cudaMemcpy\0", "libcudart")?,
            device_synchronize: symbol(&cudart, b"cudaDeviceSynchronize\0", "libcudart")?,
            get_error_string,
            sgemm: symbol(&cublas, b"cublasSgemm_v2\0", "libcublas")?,
            dgemm: symbol(&cublas, b"cublasDgemm_v2\0", "libcublas")?,
            _cudart: cudart,
            _cublas: cublas,
        };
        Ok(Self {
            api,
            handle,
            pool: Pool::new(),
        })
    }
}

/// Opens the first candidate soname that resolves.
fn open(candidates: &[&str]) -> Option<Library> {
    for &name in candidates {
        // SAFETY: loading a shared library runs its initializers;
        // these are NVIDIA's own system libraries named by their
        // fixed sonames, the ordinary way every CUDA process links
        // them.
        if let Ok(library) = unsafe { Library::new(name) } {
            return Some(library);
        }
    }
    None
}

/// Copies one typed function pointer out of an open library.
///
/// The copy stays valid because the owning [`Library`] is stored in
/// the same [`Api`] and the context is never dropped.
fn symbol<Pointer: Copy>(
    library: &Library,
    name: &'static [u8],
    library_name: &'static str,
) -> Result<Pointer, SetupError> {
    // SAFETY: the declared pointer type matches the documented C
    // signature of the named symbol, checked case by case at the
    // call sites above.
    unsafe {
        library
            .get::<Pointer>(name)
            .map(|resolved| *resolved)
            .map_err(|_| {
                SetupError::Failed(format!(
                    "symbol `{}` missing from {library_name}",
                    String::from_utf8_lossy(&name[..name.len() - 1])
                ))
            })
    }
}
