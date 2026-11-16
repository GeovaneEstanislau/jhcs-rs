use jhcs_core::buffer::{BufferHandle, BufferSet};
use jhcs_core::capabilities::{ComputeRequirements, ParallelismClass};
use jhcs_core::graph::ComputeGraph;
use jhcs_core::task::{ComputeTask, TaskId, OP_MATMUL, OP_RELU};
use jhcs_broker::{ComputeBroker, DispatchPolicy};

fn main() {
    println!("========================================================");
    println!(" JHCS — Jinn Heterogeneous Compute Service (PoC 01)");
    println!(" Heterogeneous Computing by Design — Multi-Platform DAG");
    println!("========================================================\n");

    let mut broker = ComputeBroker::new().with_policy(DispatchPolicy::DeterministicRuleBased);

    // 1. Construir DAG de tarefas
    let mut graph = ComputeGraph::new();

    // Tarefa 1: Multiplicação de matrizes 4x4
    let task_matmul = ComputeTask::new(
        TaskId::new(1),
        OP_MATMUL,
        BufferSet::from_handles(&[BufferHandle::new(1), BufferHandle::new(2)]),
        BufferSet::from_handles(&[BufferHandle::new(3)]),
    )
    .with_requirements(ComputeRequirements {
        min_memory_bytes: 64 * 4,
        parallelism: ParallelismClass::Vectorized128,
        precision: jhcs_core::buffer::Precision::Float32,
        allow_fallback: true,
    });

    // Tarefa 2: Aplicação de filtro ReLU sobre a matriz calculada
    let task_relu = ComputeTask::new(
        TaskId::new(2),
        OP_RELU,
        BufferSet::from_handles(&[BufferHandle::new(3)]),
        BufferSet::from_handles(&[BufferHandle::new(4)]),
    )
    .with_dependency(TaskId::new(1));

    graph.add_task(task_matmul).unwrap();
    graph.add_task(task_relu).unwrap();

    let batches = graph.execution_batches().unwrap();
    println!("[DAG] Validação concluída. Estágios de execução independentes:");
    for (i, batch) in batches.iter().enumerate() {
        println!("  • Estágio {}: {:?}", i, batch);
    }

    // 2. Dados das matrizes 4x4
    let a = [
        1.0f32, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
    ];

    let b = [
        1.0f32, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]; // Matriz identidade 4x4

    let mut c = [0.0f32; 16];

    println!("\n[Execução] Despachando MatMul 4x4 através do ComputeBroker...");
    let result = broker.execute_matmul(&graph.tasks()[0], &a, &b, &mut c, 4, 4, 4);

    println!("  • Status da Tarefa 1 : {:?}", result.status);
    println!("  • Backend Selecionado: {:?}", result.executed_by);
    println!("  • Matriz C calculada :");
    for row in 0..4 {
        print!("    [ ");
        for col in 0..4 {
            print!("{:>4.1} ", c[row * 4 + col]);
        }
        println!("]");
    }

    // Como B é a identidade, C deve ser idêntica a A
    assert_eq!(a, c);
    println!("\n[Verificação Numérica] Matriz C coincide 100% com a matriz esperada!");
    println!("Demonstração do JHCS concluída com sucesso.\n");
}
