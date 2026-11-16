use std::time::Instant;
use jhcs_core::buffer::{BufferHandle, BufferSet};
use jhcs_core::capabilities::{ComputeRequirements, ParallelismClass};
use jhcs_core::graph::ComputeGraph;
use jhcs_core::task::{ComputeTask, TaskId, OP_MATMUL};
use jhcs_broker::{ComputeBroker, DispatchPolicy};
use jhcs_backend_cpu::CpuBackend;
use jhcs_backend_simd::SimdBackend;

fn run_benchmark_for_size(n: usize, iterations: usize) -> (u128, u128, u128, u128) {
    let size = n * n;
    let mut a = vec![0.0f32; size];
    let mut b = vec![0.0f32; size];
    for i in 0..size {
        a[i] = ((i * 17 + 3) % 100) as f32 * 0.05;
        b[i] = ((i * 31 + 7) % 100) as f32 * 0.05;
    }
    let mut c_scalar = vec![0.0f32; size];
    let mut c_simd = vec![0.0f32; size];

    let cpu = CpuBackend::new();
    let simd = SimdBackend::new();
    let broker = ComputeBroker::new().with_policy(DispatchPolicy::DeterministicRuleBased);

    // Warm-up
    cpu.execute_matmul(TaskId::new(0), &a, &b, &mut c_scalar, n, n, n);
    simd.execute_matmul(TaskId::new(0), &a, &b, &mut c_simd, n, n, n);

    // 1. Measure T_prep + T_sync (Broker Overhead)
    let t0 = Instant::now();
    for i in 0..iterations {
        let mut graph = ComputeGraph::new();
        let task = ComputeTask::new(
            TaskId::new(i as u64 + 1),
            OP_MATMUL,
            BufferSet::from_handles(&[BufferHandle::new(1), BufferHandle::new(2)]),
            BufferSet::from_handles(&[BufferHandle::new(3)]),
        )
        .with_requirements(ComputeRequirements {
            min_memory_bytes: size * 4,
            parallelism: ParallelismClass::Vectorized128,
            precision: jhcs_core::buffer::Precision::Float32,
            allow_fallback: true,
        });
        graph.add_task(task).unwrap();
        let _ = graph.execution_batches().unwrap();
        let _ = broker.select_backend(&graph.tasks()[0], size);
    }
    let t_overhead_ns = t0.elapsed().as_nanos() / (iterations as u128);

    // 2. Measure Scalar Execution Time
    let t1 = Instant::now();
    for _ in 0..iterations {
        cpu.execute_matmul(TaskId::new(1), &a, &b, &mut c_scalar, n, n, n);
    }
    let t_scalar_ns = t1.elapsed().as_nanos() / (iterations as u128);

    // 3. Measure SIMD Execution Time
    let t2 = Instant::now();
    for _ in 0..iterations {
        simd.execute_matmul(TaskId::new(2), &a, &b, &mut c_simd, n, n, n);
    }
    let t_simd_ns = t2.elapsed().as_nanos() / (iterations as u128);

    // 4. Parity verification check
    for idx in 0..size {
        let diff = (c_scalar[idx] - c_simd[idx]).abs();
        assert!(
            diff <= 1e-4,
            "Parity error at {} for N={}: scalar={}, simd={}, diff={}",
            idx, n, c_scalar[idx], c_simd[idx], diff
        );
    }

    (t_overhead_ns, t_scalar_ns, t_simd_ns, size as u128)
}

fn main() {
    println!("==========================================================================================");
    println!(" JHCS — ComputeBroker Empirical Overhead & Break-Even Benchmark");
    println!(" Evaluating: T_prep + T_sync (Overhead) vs T_exec(Scalar) vs T_exec(SIMD)");
    println!("==========================================================================================\n");

    let sizes = [4, 8, 16, 32, 64, 128, 256, 512];
    let mut empirical_break_even: Option<usize> = None;

    println!("{:<6} | {:<10} | {:<14} | {:<14} | {:<14} | {:<9} | {:<10}",
        "N", "FLOPs", "T_overhead (ns)", "T_scalar (ns)", "T_simd (ns)", "Speedup", "Preferred");
    println!("{:-<6}-+-{:-<10}-+-{:-<14}-+-{:-<14}-+-{:-<14}-+-{:-<9}-+-{:-<10}",
        "", "", "", "", "", "", "");

    for &n in &sizes {
        let iters = match n {
            4..=16 => 5000,
            32..=64 => 1000,
            128 => 200,
            256 => 30,
            512 => 4,
            _ => 1,
        };

        let (t_overhead, t_scalar, t_simd, _) = run_benchmark_for_size(n, iters);
        let flops = 2 * (n as u64).pow(3);
        let speedup = (t_scalar as f64) / (t_simd as f64).max(1.0);
        let net_gain = (t_scalar as i128) - ((t_simd + t_overhead) as i128);

        let preferred = if net_gain > 0 && speedup > 1.05 {
            if empirical_break_even.is_none() {
                empirical_break_even = Some(n);
            }
            "SIMD (AVX/SSE)"
        } else {
            "SCALAR"
        };

        println!("{:<6} | {:<10} | {:<14} | {:<14} | {:<14} | {:<8.2}x | {:<10}",
            format!("{}x{}", n, n), flops, t_overhead, t_scalar, t_simd, speedup, preferred);
    }

    println!("------------------------------------------------------------------------------------------");
    if let Some(be) = empirical_break_even {
        println!("\n>>> EMPIRICAL BREAK-EVEN THRESHOLD: N = {} ({} elements / {} FLOPs)",
            be, be * be, 2 * (be as u64).pow(3));
        println!(">>> Workloads with N >= {} amortize broker overhead and yield positive acceleration.", be);
        println!(">>> Workloads with N < {} are faster executed directly via Scalar fallback.\n", be);
    } else {
        println!("\n>>> All tested sizes favor SIMD or require larger matrices for break-even.\n");
    }
}
