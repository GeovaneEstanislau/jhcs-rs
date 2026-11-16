//! # JHCS Platform Host — Windows & Linux Platform Adapter
//!
//! Implements timer, memory, and device contracts using standard OS APIs.

use jhcs_core::capabilities::{BackendDescriptor, BackendKind};
use jhcs_core::error::ComputeError;
use jhcs_platform::{
    PlatformAdapter, PlatformDeviceEnumerator, PlatformMemory, PlatformTimer,
};
use std::alloc::{alloc, dealloc, Layout};
use std::time::Instant;

pub struct HostTimer {
    init_instant: Instant,
}

impl HostTimer {
    pub fn new() -> Self {
        Self {
            init_instant: Instant::now(),
        }
    }
}

impl Default for HostTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl PlatformTimer for HostTimer {
    fn now_ns(&self) -> u64 {
        self.init_instant.elapsed().as_nanos() as u64
    }
}

pub struct HostMemory;

impl PlatformMemory for HostMemory {
    fn allocate(&self, size: usize, align: usize) -> Result<*mut u8, ComputeError> {
        let layout = Layout::from_size_align(size, align)
            .map_err(|_| ComputeError::InvalidBufferDescriptor)?;
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            Err(ComputeError::OutOfMemory {
                requested: size,
                limit: usize::MAX,
            })
        } else {
            Ok(ptr)
        }
    }

    unsafe fn deallocate(&self, ptr: *mut u8, size: usize, align: usize) {
        if let Ok(layout) = Layout::from_size_align(size, align) {
            dealloc(ptr, layout);
        }
    }
}

pub struct HostDeviceEnumerator;

impl PlatformDeviceEnumerator for HostDeviceEnumerator {
    fn enumerate_devices(&self) -> Vec<BackendDescriptor> {
        vec![
            BackendDescriptor {
                id: "host-cpu-scalar",
                kind: BackendKind::CpuScalar,
                name: "Host CPU (Standard)".to_string(),
                is_available: true,
                memory_capacity_bytes: usize::MAX,
            },
            BackendDescriptor {
                id: "host-cpu-simd",
                kind: BackendKind::CpuSimd,
                name: "Host CPU SIMD (Vectorized)".to_string(),
                is_available: true,
                memory_capacity_bytes: usize::MAX,
            },
        ]
    }

    fn platform_name(&self) -> &'static str {
        #[cfg(target_os = "windows")]
        return "Windows Host";
        #[cfg(target_os = "linux")]
        return "Linux Host";
        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        return "Generic POSIX Host";
    }
}

pub struct HostPlatformAdapter {
    timer: HostTimer,
    memory: HostMemory,
    devices: HostDeviceEnumerator,
}

impl HostPlatformAdapter {
    pub fn new() -> Self {
        Self {
            timer: HostTimer::new(),
            memory: HostMemory,
            devices: HostDeviceEnumerator,
        }
    }
}

impl Default for HostPlatformAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl PlatformAdapter for HostPlatformAdapter {
    fn timer(&self) -> &dyn PlatformTimer {
        &self.timer
    }

    fn memory(&self) -> &dyn PlatformMemory {
        &self.memory
    }

    fn devices(&self) -> &dyn PlatformDeviceEnumerator {
        &self.devices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_platform_timer_and_memory() {
        let adapter = HostPlatformAdapter::new();
        let t1 = adapter.timer().now_ns();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let t2 = adapter.timer().now_ns();
        assert!(t2 > t1);

        let ptr = adapter.memory().allocate(1024, 16).unwrap();
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % 16, 0);
        unsafe {
            adapter.memory().deallocate(ptr, 1024, 16);
        }
    }
}
