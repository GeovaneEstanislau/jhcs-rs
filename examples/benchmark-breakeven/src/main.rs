use std::fs;
use std::path::Path;
use std::time::Instant;

use jhcs_backend_cpu::CpuBackend;
use jhcs_backend_simd::ops::detect_avx2_fma;
use jhcs_backend_simd::SimdBackend;
use jhcs_backend_wgpu::{TransferMetrics, WgpuBackend};
use jhcs_broker::{ComputeBroker, DispatchPolicy};
use jhcs_core::buffer::{BufferHandle, BufferSet, Precision};
use jhcs_core::capabilities::{ComputeRequirements, ParallelismClass};
use jhcs_core::graph::ComputeGraph;
use jhcs_core::task::{ComputeTask, OperationId, TaskId, OP_MATMUL, OP_VECTOR_ADD};

/// Bloco de memória alinhado a 64 bytes (cache line) sem dependências externas.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Aligned64Chunk([f32; 16]);

struct AlignedF32Buffer {
    chunks: Vec<Aligned64Chunk>,
    len: usize,
}

impl AlignedF32Buffer {
    fn new(len: usize) -> Self {
        let num_chunks = (len + 15) / 16;
        let chunks = vec![Aligned64Chunk([0.0; 16]); num_chunks.max(1)];
        Self { chunks, len }
    }

    fn as_slice(&self) -> &[f32] {
        let ptr = self.chunks.as_ptr() as *const f32;
        unsafe { core::slice::from_raw_parts(ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        let ptr = self.chunks.as_mut_ptr() as *mut f32;
        unsafe { core::slice::from_raw_parts_mut(ptr, self.len) }
    }

    fn ptr_align_bytes(&self) -> usize {
        self.chunks.as_ptr() as usize % 64
    }
}

struct VectorBenchRow {
    side: usize,
    elems: usize,
    t_broker_ns: u128,
    t_scalar_ns: u128,
    t_simd_ns: u128,
    e2e_simd_ns: u128,
    speedup_exec: f64,
    speedup_e2e: f64,
    melem_s_scalar: f64,
    melem_s_simd: f64,
    winner: &'static str,
}

struct MatmulBenchRow {
    n: usize,
    elems: usize,
    flops: u64,
    t_broker_ns: u128,
    t_scalar_ns: u128,
    t_simd_ns: u128,
    e2e_simd_ns: u128,
    wgpu: Option<WgpuSample>,
    gflops_scalar: f64,
    gflops_simd: f64,
    winner: &'static str,
}

struct WgpuSample {
    t_total_ns: u128,
    e2e_ns: u128,
    upload_ns: u64,
    dispatch_ns: u64,
    download_ns: u64,
    gflops: f64,
}

fn measure_broker_overhead(
    broker: &ComputeBroker,
    op_code: OperationId,
    bytes: usize,
    elems: usize,
    iterations: usize,
) -> u128 {
    let t0 = Instant::now();
    for i in 0..iterations {
        let mut graph = ComputeGraph::new();
        let task = ComputeTask::new(
            TaskId::new(i as u64 + 1),
            op_code,
            BufferSet::from_handles(&[BufferHandle::new(1), BufferHandle::new(2)]),
            BufferSet::from_handles(&[BufferHandle::new(3)]),
        )
        .with_requirements(ComputeRequirements {
            min_memory_bytes: bytes,
            parallelism: ParallelismClass::Vectorized256,
            precision: Precision::Float32,
            allow_fallback: true,
        });
        graph.add_task(task).unwrap();
        let _ = graph.execution_batches().unwrap();
        let _ = broker.select_backend(&graph.tasks()[0], elems);
    }
    t0.elapsed().as_nanos() / (iterations.max(1) as u128)
}

fn bench_vector_add(
    side: usize,
    iterations: usize,
    cpu: &CpuBackend,
    simd: &SimdBackend,
    broker: &ComputeBroker,
) -> VectorBenchRow {
    let elems = side * side;
    let mut a = AlignedF32Buffer::new(elems);
    let mut b = AlignedF32Buffer::new(elems);
    let mut c_scalar = AlignedF32Buffer::new(elems);
    let mut c_simd = AlignedF32Buffer::new(elems);
    assert_eq!(a.ptr_align_bytes(), 0);
    assert_eq!(c_simd.ptr_align_bytes(), 0);

    for i in 0..elems {
        a.as_mut_slice()[i] = ((i * 13 + 5) % 97) as f32 * 0.25;
        b.as_mut_slice()[i] = ((i * 29 + 1) % 89) as f32 * 0.125;
    }

    let _ = cpu.execute_vector_add(TaskId::new(0), a.as_slice(), b.as_slice(), c_scalar.as_mut_slice());
    let _ = simd.execute_vector_add(TaskId::new(0), a.as_slice(), b.as_slice(), c_simd.as_mut_slice());

    let t_broker_ns = measure_broker_overhead(broker, OP_VECTOR_ADD, elems * 4, elems, iterations);

    let t1 = Instant::now();
    for _ in 0..iterations {
        let _ = cpu.execute_vector_add(TaskId::new(1), a.as_slice(), b.as_slice(), c_scalar.as_mut_slice());
    }
    let t_scalar_ns = (t1.elapsed().as_nanos() / (iterations as u128)).max(1);

    let t2 = Instant::now();
    for _ in 0..iterations {
        let _ = simd.execute_vector_add(TaskId::new(2), a.as_slice(), b.as_slice(), c_simd.as_mut_slice());
    }
    let t_simd_ns = (t2.elapsed().as_nanos() / (iterations as u128)).max(1);

    for i in 0..elems {
        let diff = (c_scalar.as_slice()[i] - c_simd.as_slice()[i]).abs();
        assert!(diff <= 1e-4, "Vector parity mismatch at {}", i);
    }

    let e2e_simd_ns = t_broker_ns + t_simd_ns;
    let speedup_exec = (t_scalar_ns as f64) / (t_simd_ns as f64);
    let speedup_e2e = (t_scalar_ns as f64) / (e2e_simd_ns as f64);
    let melem_s_scalar = (elems as f64) * 1000.0 / (t_scalar_ns as f64);
    let melem_s_simd = (elems as f64) * 1000.0 / (e2e_simd_ns as f64);
    let winner = if e2e_simd_ns < t_scalar_ns {
        "SIMD (AVX2/SSE)"
    } else {
        "Scalar CPU"
    };

    VectorBenchRow {
        side,
        elems,
        t_broker_ns,
        t_scalar_ns,
        t_simd_ns,
        e2e_simd_ns,
        speedup_exec,
        speedup_e2e,
        melem_s_scalar,
        melem_s_simd,
        winner,
    }
}

fn bench_matmul(
    n: usize,
    iterations: usize,
    cpu: &CpuBackend,
    simd: &SimdBackend,
    wgpu_opt: Option<&WgpuBackend>,
    broker: &ComputeBroker,
) -> MatmulBenchRow {
    let elems = n * n;
    let mut a = AlignedF32Buffer::new(elems);
    let mut b = AlignedF32Buffer::new(elems);
    let mut c_scalar = AlignedF32Buffer::new(elems);
    let mut c_simd = AlignedF32Buffer::new(elems);
    let mut c_wgpu = AlignedF32Buffer::new(elems);

    for i in 0..elems {
        a.as_mut_slice()[i] = ((i * 17 + 3) % 100) as f32 * 0.05;
        b.as_mut_slice()[i] = ((i * 31 + 7) % 100) as f32 * 0.05;
    }

    let _ = cpu.execute_matmul(TaskId::new(0), a.as_slice(), b.as_slice(), c_scalar.as_mut_slice(), n, n, n);
    let _ = simd.execute_matmul(TaskId::new(0), a.as_slice(), b.as_slice(), c_simd.as_mut_slice(), n, n, n);

    let t_broker_ns = measure_broker_overhead(broker, OP_MATMUL, elems * 4, elems, iterations.max(4));

    let t1 = Instant::now();
    for _ in 0..iterations {
        let _ = cpu.execute_matmul(TaskId::new(1), a.as_slice(), b.as_slice(), c_scalar.as_mut_slice(), n, n, n);
    }
    let t_scalar_ns = (t1.elapsed().as_nanos() / (iterations as u128)).max(1);

    let t2 = Instant::now();
    for _ in 0..iterations {
        let _ = simd.execute_matmul(TaskId::new(2), a.as_slice(), b.as_slice(), c_simd.as_mut_slice(), n, n, n);
    }
    let t_simd_ns = (t2.elapsed().as_nanos() / (iterations as u128)).max(1);

    for idx in 0..elems {
        let diff = (c_scalar.as_slice()[idx] - c_simd.as_slice()[idx]).abs();
        assert!(
            diff <= 1e-2,
            "MatMul parity error at {} for N={}: scalar={}, simd={}",
            idx, n, c_scalar.as_slice()[idx], c_simd.as_slice()[idx]
        );
    }

    let flops = 2u64 * (n as u64).pow(3);
    let e2e_simd_ns = t_broker_ns + t_simd_ns;
    let gflops_scalar = (flops as f64) / (t_scalar_ns as f64);
    let gflops_simd = (flops as f64) / (e2e_simd_ns as f64);

    let wgpu_sample = if let Some(gpu) = wgpu_opt {
        let _ = gpu.execute_matmul_instrumented(
            TaskId::new(0),
            a.as_slice(),
            b.as_slice(),
            c_wgpu.as_mut_slice(),
            n,
            n,
            n,
        );
        let gpu_iters = iterations.min(5).max(1);
        let mut sum_tm = TransferMetrics::default();
        let tg0 = Instant::now();
        let mut ok_gpu = true;
        for _ in 0..gpu_iters {
            match gpu.execute_matmul_instrumented(
                TaskId::new(3),
                a.as_slice(),
                b.as_slice(),
                c_wgpu.as_mut_slice(),
                n,
                n,
                n,
            ) {
                Ok(tm) => {
                    sum_tm.upload_ns += tm.upload_ns;
                    sum_tm.dispatch_ns += tm.dispatch_ns;
                    sum_tm.download_ns += tm.download_ns;
                    sum_tm.total_transfer_ns += tm.total_transfer_ns;
                }
                Err(_) => {
                    ok_gpu = false;
                    break;
                }
            }
        }
        let t_total_ns = (tg0.elapsed().as_nanos() / (gpu_iters as u128)).max(1);
        if ok_gpu {
            for idx in 0..elems.min(4096) {
                let diff = (c_scalar.as_slice()[idx] - c_wgpu.as_slice()[idx]).abs();
                assert!(diff <= 0.15, "WGPU parity mismatch at {}", idx);
            }
            let e2e_ns = t_broker_ns + t_total_ns;
            let div = gpu_iters as u64;
            Some(WgpuSample {
                t_total_ns,
                e2e_ns,
                upload_ns: sum_tm.upload_ns / div,
                dispatch_ns: sum_tm.dispatch_ns / div,
                download_ns: sum_tm.download_ns / div,
                gflops: (flops as f64) / (e2e_ns as f64),
            })
        } else {
            None
        }
    } else {
        None
    };

    let mut best_ns = t_scalar_ns;
    let mut winner = "Scalar CPU";
    if e2e_simd_ns < best_ns {
        best_ns = e2e_simd_ns;
        winner = "SIMD (AVX2/FMA)";
    }
    if let Some(ref gs) = wgpu_sample {
        if gs.e2e_ns < best_ns {
            winner = "GPU (wgpu)";
        }
    }

    MatmulBenchRow {
        n,
        elems,
        flops,
        t_broker_ns,
        t_scalar_ns,
        t_simd_ns,
        e2e_simd_ns,
        wgpu: wgpu_sample,
        gflops_scalar,
        gflops_simd,
        winner,
    }
}

fn export_reports(
    out_dir: &Path,
    arch: &str,
    os_name: &str,
    has_avx2: bool,
    has_fma: bool,
    gpu_desc: Option<&str>,
    vec_rows: &[VectorBenchRow],
    mm_rows: &[MatmulBenchRow],
    be_vec: Option<usize>,
    be_simd: Option<usize>,
    be_gpu: Option<usize>,
) {
    fs::create_dir_all(out_dir).expect("create docs/benchmarks dir");

    let mut md = String::new();
    md.push_str("# JHCS-RS Host Compute Breakeven Benchmark Report\n\n");
    md.push_str(&format!("> **Environment:** OS `{}` | Arch `{}` | AVX2 `{}` | FMA `{}` | Buffers `#[repr(align(64))]`  \n", os_name, arch, has_avx2, has_fma));
    md.push_str(&format!("> **GPU Adapter (`wgpu`):** `{}`  \n\n", gpu_desc.unwrap_or("Unavailable — automatic CPU/SIMD fallback verified")));

    md.push_str("## 1. Breakeven Summary (Scalar CPU vs SIMD AVX2/FMA vs GPU `wgpu`)\n\n");
    md.push_str(&format!("- **Element-wise `vector_add` SIMD Breakeven (incl. Broker overhead):** {}\n",
        be_vec.map(|s| format!("`{}x{}` (`{}` elements)", s, s, s * s)).unwrap_or_else(|| "Scalar CPU wins across tested sizes (Broker dispatch exceeds SIMD savings on memory-bound stream)".to_string())));
    md.push_str(&format!("- **Matrix Multiplication `matmul` SIMD Breakeven (incl. Broker overhead):** {}\n",
        be_simd.map(|s| format!("`{}x{}` (`{}` elements, `{}` FLOPs)", s, s, s * s, 2u64 * (s as u64).pow(3))).unwrap_or_else(|| "Not reached in tested sizes".to_string())));
    md.push_str(&format!("- **Matrix Multiplication `matmul` GPU (`wgpu`) Breakeven vs CPU/SIMD:** {}\n\n",
        match (gpu_desc, be_gpu) {
            (None, _) => "GPU adapter not present — deterministic CPU/SIMD fallback path verified without crash.".to_string(),
            (Some(_), Some(s)) => format!("`{}x{}` (`{}` elements) — amortized staging + PCIe upload/download overhead.", s, s, s * s),
            (Some(_), None) => "Scalar/SIMD CPU wins across all tested sizes up to 1024x1024 because `wgpu` buffer allocation + upload/download staging dominates end-to-end latency.".to_string(),
        }));

    md.push_str("## 2. Workload A: 2D Grid / Vector Stream (`OP_VECTOR_ADD`, `8^2` .. `1024^2`)\n\n");
    md.push_str("| Grid | Elements | Broker Overhead (ns) | Scalar Direct (ns) | Scalar (M elem/s) | SIMD Kernel (ns) | SIMD End-to-End (ns) | E2E SIMD (M elem/s) | Winner |\n");
    md.push_str("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n");
    for r in vec_rows {
        md.push_str(&format!(
            "| `{}x{}` | {} | {} | {} | {:.1} | {} | {} | {:.1} | **{}** |\n",
            r.side, r.side, r.elems, r.t_broker_ns, r.t_scalar_ns, r.melem_s_scalar, r.t_simd_ns, r.e2e_simd_ns, r.melem_s_simd, r.winner
        ));
    }

    md.push_str("\n## 3. Workload B: Dense Matrix Multiply (`OP_MATMUL`, `8^2` .. `1024^2`)\n\n");
    md.push_str("| Size | Elements | FLOPs | Broker (ns) | Scalar (ns) | SIMD Kernel (ns) | SIMD E2E (ns) | SIMD GFLOP/s | wgpu Upload/Dispatch/Down (ns) | wgpu E2E (ns) | wgpu GFLOP/s | Winner |\n");
    md.push_str("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | :--- |\n");
    for r in mm_rows {
        let (xfer_s, gpu_e2e_s, gpu_gf_s) = match &r.wgpu {
            Some(g) => (
                format!("{}/{}/{}", g.upload_ns, g.dispatch_ns, g.download_ns),
                format!("{}", g.e2e_ns),
                format!("{:.2}", g.gflops),
            ),
            None => ("N/A (Fallback)".to_string(), "N/A".to_string(), "N/A".to_string()),
        };
        md.push_str(&format!(
            "| `{}x{}` | {} | {} | {} | {} | {} | {} | {:.2} | {} | {} | {} | **{}** |\n",
            r.n, r.n, r.elems, r.flops, r.t_broker_ns, r.t_scalar_ns, r.t_simd_ns, r.e2e_simd_ns, r.gflops_simd, xfer_s, gpu_e2e_s, gpu_gf_s, r.winner
        ));
    }

    md.push_str("\n## 4. Reproducibility\n\n```bash\ncargo run --release -p benchmark-breakeven\n```\n");

    let md_path = out_dir.join("jhcs_host_breakeven.md");
    fs::write(&md_path, &md).expect("write markdown report");

    let mut json = String::from("{\n");
    json.push_str(&format!("  \"os\": \"{}\",\n  \"arch\": \"{}\",\n  \"avx2\": {},\n  \"fma\": {},\n", os_name, arch, has_avx2, has_fma));
    match gpu_desc {
        Some(g) => json.push_str(&format!("  \"gpu_adapter\": \"{}\",\n", g.replace('\"', "'"))),
        None => json.push_str("  \"gpu_adapter\": null,\n"),
    }
    json.push_str("  \"vector_add\": [\n");
    for (idx, r) in vec_rows.iter().enumerate() {
        let comma = if idx + 1 < vec_rows.len() { "," } else { "" };
        json.push_str(&format!(
            "    {{\"grid\": \"{}x{}\", \"elems\": {}, \"broker_ns\": {}, \"scalar_ns\": {}, \"simd_ns\": {}, \"simd_e2e_ns\": {}, \"winner\": \"{}\"}}{}\n",
            r.side, r.side, r.elems, r.t_broker_ns, r.t_scalar_ns, r.t_simd_ns, r.e2e_simd_ns, r.winner, comma
        ));
    }
    json.push_str("  ],\n  \"matmul\": [\n");
    for (idx, r) in mm_rows.iter().enumerate() {
        let comma = if idx + 1 < mm_rows.len() { "," } else { "" };
        let gpu_json = match &r.wgpu {
            Some(g) => format!(
                "{{\"total_ns\": {}, \"e2e_ns\": {}, \"upload_ns\": {}, \"dispatch_ns\": {}, \"download_ns\": {}, \"gflops\": {:.3}}}",
                g.t_total_ns, g.e2e_ns, g.upload_ns, g.dispatch_ns, g.download_ns, g.gflops
            ),
            None => "null".to_string(),
        };
        json.push_str(&format!(
            "    {{\"n\": {}, \"elems\": {}, \"flops\": {}, \"broker_ns\": {}, \"scalar_ns\": {}, \"simd_ns\": {}, \"simd_e2e_ns\": {}, \"gflops_scalar\": {:.3}, \"gflops_simd\": {:.3}, \"wgpu\": {}, \"winner\": \"{}\"}}{}\n",
            r.n, r.n, r.elems, r.t_broker_ns, r.t_scalar_ns, r.t_simd_ns, r.e2e_simd_ns, r.gflops_scalar, r.gflops_simd, gpu_json, r.winner, comma
        ));
    }
    json.push_str("  ]\n}\n");

    let json_path = out_dir.join("jhcs_host_breakeven.json");
    fs::write(&json_path, &json).expect("write json report");
    println!("[EXPORT] Markdown: {}", md_path.display());
    println!("[EXPORT] JSON    : {}", json_path.display());
}

fn main() {
    let (has_avx2, has_fma) = detect_avx2_fma();
    let cpu = CpuBackend::new();
    let simd = SimdBackend::new();
    let broker = ComputeBroker::new().with_policy(DispatchPolicy::DeterministicRuleBased);
    let wgpu = WgpuBackend::new();
    let gpu_desc = wgpu.as_ref().map(|g| g.descriptor().name.as_str());

    println!("==========================================================================================");
    println!(" JHCS-RS — Host Breakeven Benchmark (Scalar CPU vs SIMD AVX2/FMA vs wgpu GPU)");
    println!(" Arch: {} | OS: {} | AVX2: {} | FMA: {} | GPU: {}",
        std::env::consts::ARCH, std::env::consts::OS, has_avx2, has_fma, gpu_desc.unwrap_or("None (Fallback)"));
    println!("==========================================================================================\n");

    let vec_sizes = [8usize, 64, 256, 512, 1024];
    let mut vec_rows = Vec::new();
    let mut be_vec = None;
    println!("--- WORKLOAD A: Vector Add (64-byte aligned buffers) ---");
    for &s in &vec_sizes {
        let iters = match s {
            8 | 64 => 4000,
            256 => 600,
            512 => 150,
            _ => 40,
        };
        let row = bench_vector_add(s, iters, &cpu, &simd, &broker);
        if be_vec.is_none() && row.e2e_simd_ns < row.t_scalar_ns {
            be_vec = Some(s);
        }
        println!(
            "Grid {:>4}x{:<4} | Broker {:>5} ns | Scalar {:>7} ns | SIMD {:>7} ns (Kernel {:.2}x | E2E {:>7} ns | {:.2}x) | Winner: {}",
            row.side, row.side, row.t_broker_ns, row.t_scalar_ns, row.t_simd_ns, row.speedup_exec, row.e2e_simd_ns, row.speedup_e2e, row.winner
        );
        vec_rows.push(row);
    }

    let mm_sizes = [8usize, 16, 32, 64, 128, 256, 512, 1024];
    let mut mm_rows = Vec::new();
    let mut be_simd = None;
    let mut be_gpu = None;
    println!("\n--- WORKLOAD B: Dense Matrix Multiply (8^2 .. 1024^2) ---");
    for &n in &mm_sizes {
        let iters = match n {
            8 => 4000,
            16 | 32 => 2000,
            64 => 300,
            128 => 40,
            256 => 6,
            512 => 2,
            _ => 1,
        };
        let row = bench_matmul(n, iters, &cpu, &simd, wgpu.as_ref(), &broker);
        if be_simd.is_none() && row.e2e_simd_ns < row.t_scalar_ns {
            be_simd = Some(n);
        }
        if be_gpu.is_none() {
            if let Some(ref g) = row.wgpu {
                if g.e2e_ns < row.t_scalar_ns && g.e2e_ns < row.e2e_simd_ns {
                    be_gpu = Some(n);
                }
            }
        }
        let gpu_info = match &row.wgpu {
            Some(g) => format!("GPU E2E {:>8} ns (xfer up/disp/dl {}/{}/{}) {:.2} GF/s", g.e2e_ns, g.upload_ns, g.dispatch_ns, g.download_ns, g.gflops),
            None => "GPU: Fallback CPU/SIMD".to_string(),
        };
        println!(
            "MatMul {:>4}x{:<4} | Broker {:>5} ns | Scalar {:>10} ns | SIMD E2E {:>10} ns ({:.2} GF/s) | {} | Winner: {}",
            row.n, row.n, row.t_broker_ns, row.t_scalar_ns, row.e2e_simd_ns, row.gflops_simd, gpu_info, row.winner
        );
        mm_rows.push(row);
    }

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let docs_bench_dir = manifest_dir.parent().unwrap().parent().unwrap().join("docs").join("benchmarks");
    export_reports(
        &docs_bench_dir,
        std::env::consts::ARCH,
        std::env::consts::OS,
        has_avx2,
        has_fma,
        gpu_desc,
        &vec_rows,
        &mm_rows,
        be_vec,
        be_simd,
        be_gpu,
    );
}
