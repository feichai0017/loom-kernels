//! Tensor-oriented transport API.
//!
//! Unlike the removed byte-store transport, operations refer to registered
//! memory and completion events. Implementations can map this contract to CUDA
//! IPC, NCCL, NIXL, UCX, or GPUDirect RDMA without host serialization.

use crate::types::{DeviceKind, TensorHandle};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DeviceKind, WorkerId};

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

    #[test]
    fn rejects_out_of_bounds_tensor_range() {
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
            transfer.validate().unwrap_err(),
            TransportError::OutOfBounds
        );
    }
}
