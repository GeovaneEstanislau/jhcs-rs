#![cfg_attr(not(feature = "std"), no_std)]

//! # JHCS Platform — Platform Abstraction Traits
//!
//! Defines pluggable platform contracts for high-precision timers,
//! memory buffer mapping, and hardware accelerator enumeration.

extern crate alloc;

use alloc::vec::Vec;
use jhcs_core::capabilities::BackendDescriptor;
use jhcs_core::error::ComputeError;

/// High-resolution monotonic timer interface.
pub trait PlatformTimer: Send + Sync {
    /// Returns current monotonic timestamp in nanoseconds.
    fn now_ns(&self) -> u64;

    /// Returns elapsed nanoseconds since `start_ns`.
    fn elapsed_ns(&self, start_ns: u64) -> u64 {
        self.now_ns().saturating_sub(start_ns)
    }
}

/// Memory allocator and buffer mapper for the underlying OS.
pub trait PlatformMemory: Send + Sync {
    /// Allocates an aligned buffer in host RAM or unified memory.
    fn allocate(&self, size: usize, align: usize) -> Result<*mut u8, ComputeError>;

    /// Deallocates an aligned buffer previously allocated by this platform.
    ///
    /// # Safety
    /// The caller must guarantee `ptr` was allocated with identical size and alignment.
    unsafe fn deallocate(&self, ptr: *mut u8, size: usize, align: usize);
}

/// Hardware accelerator and peripheral enumerator.
pub trait PlatformDeviceEnumerator: Send + Sync {
    /// Enumerates all compute-capable accelerators discovered on this platform.
    fn enumerate_devices(&self) -> Vec<BackendDescriptor>;

    /// Platform description or human-readable environment name.
    fn platform_name(&self) -> &'static str;
}

/// Unified platform adapter combining timer, memory, and device enumeration.
pub trait PlatformAdapter: Send + Sync {
    fn timer(&self) -> &dyn PlatformTimer;
    fn memory(&self) -> &dyn PlatformMemory;
    fn devices(&self) -> &dyn PlatformDeviceEnumerator;
}
