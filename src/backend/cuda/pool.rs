use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use super::context::Api;

/// Device allocations at or below this many bytes round up to a
/// power-of-two size class and are kept for reuse; larger ones are
/// exact-sized and freed on return, so a one-off giant task cannot
/// hoard VRAM.
const CLASS_CAP: usize = 256 * 1024 * 1024;

/// The smallest class, so tiny buffers share one slot size.
const CLASS_FLOOR: usize = 4096;

/// A size-classed pool of device buffers, the port of the metal
/// pool onto `cudaMalloc`: taking rounds the request up to its
/// class and reuses a parked buffer when one exists; giving parks
/// the buffer again. Pooled buffers are never freed — the owning
/// context lives in a process-wide static — which bounds VRAM held
/// at the high-water mark of concurrent tasks.
pub(super) struct Pool {
    parked: Mutex<HashMap<usize, Vec<*mut c_void>>>,
}

impl Pool {
    pub(super) fn new() -> Self {
        Self {
            parked: Mutex::new(HashMap::new()),
        }
    }

    /// Returns a device buffer of at least `bytes`, reusing a
    /// parked one when the class has any.
    pub(super) fn take(&self, api: &Api, bytes: usize) -> Result<*mut c_void, String> {
        let class = class_of(bytes);
        if let Some(buffer) = self
            .parked
            .lock()
            .expect("the pool mutex is poisoned")
            .get_mut(&class)
            .and_then(Vec::pop)
        {
            return Ok(buffer);
        }
        let mut buffer = std::ptr::null_mut();
        // SAFETY: the pointer addresses a live slot for the device
        // pointer; a nonzero status leaves it unused.
        let status = unsafe { (api.malloc)(&mut buffer, class) };
        if status != 0 {
            return Err(format!("cudaMalloc failed: {}", api.error_string(status)));
        }
        Ok(buffer)
    }

    /// Parks a buffer taken for `bytes` back into its class.
    ///
    /// Beyond-cap buffers are exact-sized one-offs; parking them
    /// under their exact size reuses them only for an identical
    /// request, which is precisely the recurring-shape case worth
    /// serving.
    pub(super) fn give(&self, bytes: usize, buffer: *mut c_void) {
        self.parked
            .lock()
            .expect("the pool mutex is poisoned")
            .entry(class_of(bytes))
            .or_default()
            .push(buffer);
    }
}

/// Returns the size class for a request.
fn class_of(bytes: usize) -> usize {
    if bytes > CLASS_CAP {
        return bytes;
    }
    bytes.next_power_of_two().max(CLASS_FLOOR)
}
