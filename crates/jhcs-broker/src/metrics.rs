//! # Metrics Tracking and Historical Execution Statistics
//!
//! Tracks historical performance metrics and latency telemetry across backends.

extern crate alloc;
use alloc::collections::BTreeMap;
use jhcs_core::capabilities::BackendKind;
use jhcs_core::contracts::ComputeTelemetry;
use jhcs_core::task::ComputeResult;

/// Historical performance metrics for an operation on a specific backend.
#[derive(Debug, Clone, Default)]
pub struct OperationMetrics {
    pub execution_count: u64,
    pub total_prep_ns: u64,
    pub total_transfer_ns: u64,
    pub total_exec_ns: u64,
    pub total_sync_ns: u64,
    pub min_exec_ns: u64,
    pub max_exec_ns: u64,
}

impl OperationMetrics {
    pub fn record(
        &mut self,
        prep_ns: u64,
        transfer_ns: u64,
        exec_ns: u64,
        sync_ns: u64,
    ) {
        self.execution_count += 1;
        self.total_prep_ns += prep_ns;
        self.total_transfer_ns += transfer_ns;
        self.total_exec_ns += exec_ns;
        self.total_sync_ns += sync_ns;

        if self.min_exec_ns == 0 || exec_ns < self.min_exec_ns {
            self.min_exec_ns = exec_ns;
        }
        if exec_ns > self.max_exec_ns {
            self.max_exec_ns = exec_ns;
        }
    }

    pub fn avg_exec_ns(&self) -> u64 {
        if self.execution_count == 0 {
            0
        } else {
            self.total_exec_ns / self.execution_count
        }
    }

    pub fn total_overhead_ns(&self) -> u64 {
        if self.execution_count == 0 {
            0
        } else {
            (self.total_prep_ns + self.total_transfer_ns + self.total_sync_ns)
                / self.execution_count
        }
    }
}

/// Registry of historical telemetry per backend and operation type.
#[derive(Debug, Clone, Default)]
pub struct MetricsRegistry {
    // Key: (BackendKind as u8, &'static str [operation_name])
    stats: BTreeMap<(u8, &'static str), OperationMetrics>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self {
            stats: BTreeMap::new(),
        }
    }

    pub fn record(
        &mut self,
        backend: BackendKind,
        op_name: &'static str,
        prep_ns: u64,
        transfer_ns: u64,
        exec_ns: u64,
        sync_ns: u64,
    ) {
        let key = (backend as u8, op_name);
        self.stats
            .entry(key)
            .or_default()
            .record(prep_ns, transfer_ns, exec_ns, sync_ns);
    }

    pub fn get(&self, backend: BackendKind, op_name: &'static str) -> Option<&OperationMetrics> {
        let key = (backend as u8, op_name);
        self.stats.get(&key)
    }
}

impl ComputeTelemetry for MetricsRegistry {
    fn record_result(&mut self, result: &ComputeResult) {
        let op_name = "general_task";
        self.record(
            result.executed_by,
            op_name,
            result.prep_duration_ns,
            result.transfer_duration_ns,
            result.exec_duration_ns,
            result.sync_duration_ns,
        );
    }

    fn total_executions(&self) -> u64 {
        self.stats.values().map(|m| m.execution_count).sum()
    }
}
