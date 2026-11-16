#![no_std]

//! # JHCS Platform Jinn — Native Jinn OS Ring 3 Adapter Contracts
//!
//! Provides platform binding stubs and contracts for running JHCS natively
//! in Jinn OS user space (Ring 3), interfacing directly with:
//! - Kernel DMA Manager (KDM) via syscalls 60..68
//! - Jinn Dataflow Architecture (JDA) via syscalls 12..14, 26..29
//! - Aladdin Desktop IPC and shared window surfaces

extern crate alloc;

use alloc::string::ToString;
use alloc::vec::Vec;
use jhcs_core::capabilities::{BackendDescriptor, BackendKind};
use jhcs_core::error::ComputeError;
use jhcs_platform::{
    PlatformAdapter, PlatformDeviceEnumerator, PlatformMemory, PlatformTimer,
};

/// High-resolution timer based on the processor's Time Stamp Counter (RDTSC).
pub struct JinnTimer;

impl JinnTimer {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for JinnTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl PlatformTimer for JinnTimer {
    fn now_ns(&self) -> u64 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // Rough TSC cycle conversion; in full integration, calibrated against RTC/PIT
            core::arch::x86_64::_rdtsc() / 2 // Approximate for 2 GHz clock
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            0
        }
    }
}

/// Jinn OS unified memory allocator binding to KDM / JDA Shmem syscalls.
pub struct JinnMemory;

impl PlatformMemory for JinnMemory {
    fn allocate(&self, size: usize, align: usize) -> Result<*mut u8, ComputeError> {
        // Contract: Invokes sys_jda_shm_alloc or sys_alloc_page in Ring 3
        let layout = match core::alloc::Layout::from_size_align(size, align) {
            Ok(l) => l,
            Err(_) => return Err(ComputeError::InvalidBufferDescriptor),
        };

        // When executing under Jinn OS with alloc active:
        let ptr = unsafe { alloc::alloc::alloc(layout) };
        if ptr.is_null() {
            Err(ComputeError::OutOfMemory {
                requested: size,
                limit: 14 * 1024 * 1024, // Legacy profile user cap
            })
        } else {
            Ok(ptr)
        }
    }

    unsafe fn deallocate(&self, ptr: *mut u8, size: usize, align: usize) {
        if let Ok(layout) = core::alloc::Layout::from_size_align(size, align) {
            alloc::alloc::dealloc(ptr, layout);
        }
    }
}

/// Discovers accelerators via Jinn's PCI enumerator and DevFS nodes.
pub struct JinnDeviceEnumerator;

impl PlatformDeviceEnumerator for JinnDeviceEnumerator {
    fn enumerate_devices(&self) -> Vec<BackendDescriptor> {
        let mut list = Vec::new();

        // 1. Core CPU scalar backend (always online)
        list.push(BackendDescriptor {
            id: "jinn-cpu-scalar",
            kind: BackendKind::CpuScalar,
            name: "Jinn OS Ring 3 Scalar Core".to_string(),
            is_available: true,
            memory_capacity_bytes: 14 * 1024 * 1024,
        });

        // 2. Vectorized SIMD unit (available if Etapa B is active)
        list.push(BackendDescriptor {
            id: "jinn-cpu-simd",
            kind: BackendKind::CpuSimd,
            name: "Jinn OS FpuState SIMD".to_string(),
            is_available: true,
            memory_capacity_bytes: 14 * 1024 * 1024,
        });

        // 3. GPU device (identified through PCI Class 0x03 or VirtIO-GPU)
        list.push(BackendDescriptor {
            id: "jinn-gpu-virtio",
            kind: BackendKind::GpuCompute,
            name: "Jinn VirtIO-GPU Accelerator (Planned)".to_string(),
            is_available: false, // Activated when Etapa C is completed
            memory_capacity_bytes: 0,
        });

        list
    }

    fn platform_name(&self) -> &'static str {
        "Jinn OS (Ring 3 Microkernel)"
    }
}

/// Full platform adapter implementation for native Jinn OS.
pub struct JinnPlatformAdapter {
    timer: JinnTimer,
    memory: JinnMemory,
    devices: JinnDeviceEnumerator,
}

impl JinnPlatformAdapter {
    pub const fn new() -> Self {
        Self {
            timer: JinnTimer,
            memory: JinnMemory,
            devices: JinnDeviceEnumerator,
        }
    }
}

impl Default for JinnPlatformAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl PlatformAdapter for JinnPlatformAdapter {
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
