use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use super::context::Api;

/// Device allocations at or below this many bytes round up to a
/// power-of-two size class and are kept for reuse; larger ones are
/// exact-sized and freed on return, so a one-off giant task cannot
/// hoard VRAM.
pub(super) const CLASS_CAP: usize = 256 * 1024 * 1024;

/// The smallest class, so tiny buffers share one slot size.
const CLASS_FLOOR: usize = 4096;

/// Total bytes the pool may hold parked across every class; a give
/// beyond it frees instead of parking. Four class caps hold a full
/// three-buffer working set at the largest class with headroom; the
/// constant is provisional until the first hardware measurement,
/// like the flop threshold.
pub(super) const PARKED_CAP: usize = 4 * CLASS_CAP;

/// A size-classed pool of device buffers, the port of the metal
/// pool onto `cudaMalloc`: taking rounds the request up to its
/// class and reuses a parked buffer when one exists; giving parks
/// the buffer again, or frees it when the buffer is an above-cap
/// one-off or the pool already holds [`PARKED_CAP`] parked bytes.
/// VRAM held parked is therefore bounded by the cap, not by the
/// variety of shapes the process ever ran.
pub(super) struct Pool {
    parked: Mutex<Parked>,
}

/// The parked buffers by size class, with the running byte total
/// the cap is enforced against.
struct Parked {
    classes: HashMap<usize, Vec<*mut c_void>>,
    bytes: usize,
}

impl Pool {
    pub(super) fn new() -> Self {
        Self {
            parked: Mutex::new(Parked {
                classes: HashMap::new(),
                bytes: 0,
            }),
        }
    }

    /// Returns a device buffer of at least `bytes`, reusing a
    /// parked one when the class has any.
    pub(super) fn take(&self, api: &Api, bytes: usize) -> Result<*mut c_void, String> {
        let class = class_of(bytes);
        {
            let mut parked = self.parked.lock().expect("the pool mutex is poisoned");
            if let Some(buffer) = parked.classes.get_mut(&class).and_then(Vec::pop) {
                parked.bytes -= class;
                return Ok(buffer);
            }
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

    /// Returns a buffer taken for `bytes`: parked back into its
    /// class for reuse, or freed when it is an above-cap exact-sized
    /// one-off or parking it would push the pool past
    /// [`PARKED_CAP`].
    pub(super) fn give(&self, api: &Api, bytes: usize, buffer: *mut c_void) {
        let class = class_of(bytes);
        if class <= CLASS_CAP {
            let mut parked = self.parked.lock().expect("the pool mutex is poisoned");
            if parked.bytes + class <= PARKED_CAP {
                parked.classes.entry(class).or_default().push(buffer);
                parked.bytes += class;
                return;
            }
        }
        // SAFETY: the buffer came from `cudaMalloc` through `take`
        // and no task references it any longer. The status goes
        // unchecked because nothing here can recover a failed free:
        // the buffer is lost to the process either way, and a broken
        // runtime surfaces through the next call's own status.
        let _ = unsafe { (api.free)(buffer) };
    }
}

/// Returns the size class for a request.
fn class_of(bytes: usize) -> usize {
    if bytes > CLASS_CAP {
        return bytes;
    }
    bytes.next_power_of_two().max(CLASS_FLOOR)
}
