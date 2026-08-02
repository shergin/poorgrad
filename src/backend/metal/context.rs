use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLCompileOptions, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
    MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
};

use super::pool::Pool;

/// How many shape-specialized pipelines the cache holds; new shapes
/// past the cap run the generic tiled pipeline instead, so a
/// pathological stream of distinct shapes degrades to generic speed
/// rather than compiling forever.
const SPECIALIZED_CAPACITY: usize = 32;

/// The seven baked values of one specialized pipeline, in the
/// shader's function-constant order: `m`, `n`, `k`, then the two
/// stride pairs.
pub(super) type ShapeKey = [u32; 7];

/// The backend's one-time state: the device and queue, the two
/// compiled pipelines, and the buffer pool.
///
/// Built lazily on the first eligible task or `status` call; any
/// failure is the reason string the diagnostics report.
pub(super) struct Context {
    pub(super) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(super) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(super) naive: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) tiled: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    specialized: Mutex<HashMap<ShapeKey, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
    pub(super) pool: Pool,
}

// SAFETY: Apple documents devices, command queues, pipeline states,
// and buffers as thread-safe; the one thread-bound Metal object — a
// command encoder — is a per-call local in `gemm` and never stored.
// The pool guards its own state with a `Mutex`. objc2 declines to
// mark the protocols `Send`/`Sync` out of caution, so the aggregate
// carries the contract explicitly.
#[allow(unsafe_code)]
unsafe impl Send for Context {}
#[allow(unsafe_code)]
unsafe impl Sync for Context {}

impl Context {
    /// Creates the device, compiles the kernels from source (fast
    /// math off — parity with the CPU paths matters more than the
    /// last percent), and builds both pipeline states.
    pub(super) fn new() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| "no Metal device".to_string())?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| "no command queue".to_string())?;
        let source = NSString::from_str(include_str!("shaders/gemm.metal"));
        let options = MTLCompileOptions::new();
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|error| error.localizedDescription().to_string())?;
        let naive = pipeline(&device, &library, "gemm_naive_f32")?;
        let tiled = pipeline(&device, &library, "gemm_tiled_f32")?;
        if tiled.maxTotalThreadsPerThreadgroup() < 128 {
            return Err("the tiled kernel needs 128 threads per threadgroup".to_string());
        }
        Ok(Self {
            device,
            queue,
            naive,
            tiled,
            library,
            specialized: Mutex::new(HashMap::new()),
            pool: Pool::new(),
        })
    }

    /// Returns the pipeline specialized to `key`, building and
    /// caching it on the first sight of the shape; `None` — run the
    /// generic pipeline — past the cache cap or on a build failure.
    pub(super) fn specialized(
        &self,
        key: ShapeKey,
    ) -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        {
            let cache = self
                .specialized
                .lock()
                .expect("the pipeline cache is poisoned");
            if let Some(pipeline) = cache.get(&key) {
                return Some(pipeline.clone());
            }
            if cache.len() >= SPECIALIZED_CAPACITY {
                return None;
            }
        }
        // Built outside the lock: specialization costs milliseconds,
        // and a racing duplicate build is benign — last one parks.
        let pipeline = self.build_specialized(key).ok()?;
        let mut cache = self
            .specialized
            .lock()
            .expect("the pipeline cache is poisoned");
        Some(cache.entry(key).or_insert(pipeline).clone())
    }

    /// Builds one pipeline with the shape baked as function constants.
    fn build_specialized(
        &self,
        key: ShapeKey,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
        let constants = MTLFunctionConstantValues::new();
        for (index, value) in key.iter().enumerate() {
            // SAFETY: the pointer addresses a live `u32` for the
            // duration of the call, matching the declared `UInt`
            // type; the value is copied out by the call.
            unsafe {
                constants.setConstantValue_type_atIndex(
                    NonNull::from(value).cast::<c_void>(),
                    MTLDataType::UInt,
                    index,
                );
            }
        }
        let function = self
            .library
            .newFunctionWithName_constantValues_error(
                &NSString::from_str("gemm_specialized_f32"),
                &constants,
            )
            .map_err(|error| error.localizedDescription().to_string())?;
        self.device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| error.localizedDescription().to_string())
    }
}

/// Builds one compute pipeline state from a named kernel.
fn pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
    let function = library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| format!("kernel `{name}` is missing from the library"))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| error.localizedDescription().to_string())
}
