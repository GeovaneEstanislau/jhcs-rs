#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;
use alloc::string::ToString;
use jhcs_core::capabilities::{
    BackendDescriptor, BackendKind, ComputeCapabilities, ParallelismClass,
};
use jhcs_core::contracts::ComputeBackend as ComputeBackendContract;
use jhcs_core::error::ComputeError;
use jhcs_core::task::{ComputeResult, TaskId};

#[derive(Debug, Clone, Copy, Default)]
pub struct TransferMetrics {
    pub upload_ns: u64,
    pub dispatch_ns: u64,
    pub download_ns: u64,
    pub total_transfer_ns: u64,
}

#[cfg(feature = "std")]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatrixUniforms {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

pub struct WgpuBackend {
    capabilities: ComputeCapabilities,
    descriptor: BackendDescriptor,
    #[cfg(feature = "std")]
    device: wgpu::Device,
    #[cfg(feature = "std")]
    queue: wgpu::Queue,
    #[cfg(feature = "std")]
    pipeline: wgpu::ComputePipeline,
    #[cfg(feature = "std")]
    bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuBackend {
    #[cfg(feature = "std")]
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let adapter_info = adapter.get_info();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("JHCS WGPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        let shader_src = include_str!("shader_matmul.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("JHCS MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("JHCS MatMul BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("JHCS MatMul Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("JHCS MatMul Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("matmul_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            capabilities: ComputeCapabilities {
                kind: BackendKind::GpuCompute,
                max_buffer_size_bytes: 256 * 1024 * 1024,
                max_workgroup_size: 256,
                supports_fp64: false,
                supports_fp32: true,
                supports_int32: true,
                max_parallelism: ParallelismClass::MassiveParallelGpu,
                supported_operations: alloc::vec!["matmul".to_string()],
            },
            descriptor: BackendDescriptor {
                id: "gpu-wgpu-primary",
                kind: BackendKind::GpuCompute,
                name: alloc::format!("JHCS WGPU [{}]", adapter_info.name),
                is_available: true,
                memory_capacity_bytes: 1024 * 1024 * 1024,
            },
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    #[cfg(not(feature = "std"))]
    pub fn new() -> Option<Self> {
        None
    }

    pub fn capabilities(&self) -> &ComputeCapabilities {
        &self.capabilities
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    #[cfg(feature = "std")]
    pub fn execute_matmul_instrumented(
        &self,
        _task_id: TaskId,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<TransferMetrics, ComputeError> {
        use std::time::Instant;
        use wgpu::util::DeviceExt;

        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(ComputeError::DimensionMismatch {
                expected: (m, n),
                actual: (c.len() / n.max(1), n),
            });
        }

        let uniforms = MatrixUniforms {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _pad: 0,
        };

        // 1. T_upload: Staging / GPU buffer creation & memory transfer
        let t0 = Instant::now();
        let uniform_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let buf_a = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix A Buffer"),
            contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let buf_b = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix B Buffer"),
            contents: bytemuck::cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let c_byte_len = (c.len() * core::mem::size_of::<f32>()) as u64;
        let buf_c = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C Buffer"),
            size: c_byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer C"),
            size: c_byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_c.as_entire_binding(),
                },
            ],
        });
        let t_upload_ns = t0.elapsed().as_nanos() as u64;

        // 2. T_dispatch: Command encoder dispatch & queue submission
        let t1 = Instant::now();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MatMul Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_x = ((n as u32) + 15) / 16;
            let workgroup_y = ((m as u32) + 15) / 16;
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        encoder.copy_buffer_to_buffer(&buf_c, 0, &staging_buf, 0, c_byte_len);
        let submission_index = self.queue.submit(Some(encoder.finish()));
        let t_dispatch_ns = t1.elapsed().as_nanos() as u64;

        // 3. T_download: Staging buffer map_async sync & readback
        let t2 = Instant::now();
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));
        rx.recv().unwrap().map_err(|_| ComputeError::ExecutionFailed("Buffer mapping failed".to_string()))?;

        {
            let view = slice.get_mapped_range();
            let result_slice: &[f32] = bytemuck::cast_slice(&view);
            c.copy_from_slice(result_slice);
        }
        staging_buf.unmap();
        let t_download_ns = t2.elapsed().as_nanos() as u64;

        Ok(TransferMetrics {
            upload_ns: t_upload_ns,
            dispatch_ns: t_dispatch_ns,
            download_ns: t_download_ns,
            total_transfer_ns: t_upload_ns + t_download_ns,
        })
    }

    pub fn execute_matmul(
        &self,
        task_id: TaskId,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> ComputeResult {
        #[cfg(feature = "std")]
        match self.execute_matmul_instrumented(task_id, a, b, c, m, k, n) {
            Ok(metrics) => ComputeResult::success(task_id, BackendKind::GpuCompute, metrics.dispatch_ns),
            Err(e) => ComputeResult::failure(task_id, BackendKind::GpuCompute, e),
        }

        #[cfg(not(feature = "std"))]
        {
            let _ = (task_id, a, b, c, m, k, n);
            ComputeResult::failure(task_id, BackendKind::GpuCompute, ComputeError::BackendUnavailable)
        }
    }
}

impl ComputeBackendContract for WgpuBackend {
    fn id(&self) -> &'static str {
        "gpu-wgpu-primary"
    }

    fn kind(&self) -> BackendKind {
        BackendKind::GpuCompute
    }

    fn is_available(&self) -> bool {
        self.descriptor.is_available
    }

    fn capabilities(&self) -> &ComputeCapabilities {
        &self.capabilities
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use jhcs_backend_cpu::CpuBackend;

    #[test]
    fn test_wgpu_vs_cpu_numerical_equivalence() {
        let backend = match WgpuBackend::new() {
            Some(b) => b,
            None => {
                println!("No GPU adapter available, skipping test");
                return;
            }
        };

        let m = 32;
        let k = 32;
        let n = 32;
        let mut a = alloc::vec![0.0f32; m * k];
        let mut b = alloc::vec![0.0f32; k * n];

        for i in 0..a.len() {
            a[i] = ((i * 17 + 3) % 100) as f32 * 0.05;
        }
        for i in 0..b.len() {
            b[i] = ((i * 31 + 7) % 100) as f32 * 0.05;
        }

        let mut c_cpu = alloc::vec![0.0f32; m * n];
        let mut c_gpu = alloc::vec![0.0f32; m * n];

        let cpu = CpuBackend::new();
        cpu.execute_matmul(TaskId::new(1), &a, &b, &mut c_cpu, m, k, n);

        let metrics = backend.execute_matmul_instrumented(TaskId::new(2), &a, &b, &mut c_gpu, m, k, n).unwrap();
        println!("GPU MatMul 32x32 metrics: {:?}", metrics);

        for idx in 0..(m * n) {
            let diff = (c_cpu[idx] - c_gpu[idx]).abs();
            assert!(
                diff <= 1e-4,
                "GPU divergence at {}: cpu={}, gpu={}, diff={}",
                idx, c_cpu[idx], c_gpu[idx], diff
            );
        }
    }
}
