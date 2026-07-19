//! Tensor-oriented transport API.
//!
//! Unlike the removed byte-store transport, operations refer to registered
//! memory and completion events. Implementations can map this contract to CUDA
//! IPC, NCCL, NIXL, UCX, or GPUDirect RDMA without host serialization.

use async_trait::async_trait;
use quillcache_types::{DeviceKind, TensorHandle};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TransportError {
    #[error("tensor registration is invalid: {0}")]
    InvalidRegistration(String),
    #[error("tensor transport is unavailable: {0}")]
    Unavailable(String),
    #[error("tensor transfer exceeds a registered region")]
    OutOfBounds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportKind {
    InProcess,
    CudaIpc,
    CudaP2p,
    Nccl,
    Nixl,
    Rdma,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportCapabilities {
    pub kind: TransportKind,
    pub source_devices: Vec<DeviceKind>,
    pub destination_devices: Vec<DeviceKind>,
    pub supports_device_direct: bool,
    pub supports_cuda_stream: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorTransfer {
    pub source: TensorHandle,
    pub source_offset: u64,
    pub destination: TensorHandle,
    pub destination_offset: u64,
    pub bytes: u64,
    pub cuda_stream: Option<u64>,
    pub deadline_unix_us: u64,
}

impl TensorTransfer {
    pub fn validate(&self) -> Result<(), TransportError> {
        let source_end = self
            .source_offset
            .checked_add(self.bytes)
            .ok_or(TransportError::OutOfBounds)?;
        let destination_end = self
            .destination_offset
            .checked_add(self.bytes)
            .ok_or(TransportError::OutOfBounds)?;
        if source_end > self.source.bytes || destination_end > self.destination.bytes {
            return Err(TransportError::OutOfBounds);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferCompletion {
    pub completion_id: String,
    pub transferred_bytes: u64,
    pub kind: TransportKind,
    pub cuda_event: Option<u64>,
}

#[async_trait]
pub trait TensorTransport: std::fmt::Debug + Send + Sync {
    fn capabilities(&self) -> &TransportCapabilities;

    async fn submit(&self, transfer: TensorTransfer) -> Result<TransferCompletion, TransportError>;
}

/// Metadata-only implementation used to validate planner/runtime contracts.
#[derive(Debug)]
pub struct InProcessTransport {
    capabilities: TransportCapabilities,
    next_completion: AtomicU64,
}

impl Default for InProcessTransport {
    fn default() -> Self {
        Self {
            capabilities: TransportCapabilities {
                kind: TransportKind::InProcess,
                source_devices: vec![DeviceKind::Cpu, DeviceKind::Cuda],
                destination_devices: vec![DeviceKind::Cpu, DeviceKind::Cuda],
                supports_device_direct: true,
                supports_cuda_stream: true,
            },
            next_completion: AtomicU64::new(1),
        }
    }
}

#[async_trait]
impl TensorTransport for InProcessTransport {
    fn capabilities(&self) -> &TransportCapabilities {
        &self.capabilities
    }

    async fn submit(&self, transfer: TensorTransfer) -> Result<TransferCompletion, TransportError> {
        transfer.validate()?;
        Ok(TransferCompletion {
            completion_id: format!(
                "in-process-{}",
                self.next_completion.fetch_add(1, Ordering::Relaxed)
            ),
            transferred_bytes: transfer.bytes,
            kind: TransportKind::InProcess,
            cuda_event: transfer.cuda_stream,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quillcache_types::{DeviceKind, WorkerId};

    fn handle(bytes: u64) -> TensorHandle {
        TensorHandle {
            owner: WorkerId("worker".into()),
            device_kind: DeviceKind::Cuda,
            device_index: 0,
            address: 0x1000,
            bytes,
            registration_key: None,
            generation: 1,
        }
    }

    #[tokio::test]
    async fn validates_registered_bounds_before_submission() {
        let transport = InProcessTransport::default();
        let transfer = TensorTransfer {
            source: handle(16),
            source_offset: 8,
            destination: handle(16),
            destination_offset: 0,
            bytes: 16,
            cuda_stream: Some(7),
            deadline_unix_us: 100,
        };
        assert_eq!(
            transport.submit(transfer).await.unwrap_err(),
            TransportError::OutOfBounds
        );
    }
}
