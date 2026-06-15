//! Master-metadata **OpLog** (Mooncake's `ha/oplog`) — an append-only, per-entry
//! checksummed log of the durable mutations to the master's state, replayed on
//! recovery. Mirrors Mooncake's HA model: each entry is framed
//! `[u32 len][u32 crc][payload]` and `fsync`ed on append (the durability point);
//! a torn or corrupt tail (a crash mid-append) stops replay cleanly instead of
//! returning garbage — the same discipline as the SSD-tier WAL. Paired with a
//! periodic [`crate::MasterService::snapshot`] (its `.tmp`+rename atomic publish)
//! for compaction, this is the standard durable state-machine: snapshot + log.
//!
//! vs the existing snapshot-only recovery: the snapshot is a full point-in-time
//! dump; the OpLog records each committed change incrementally, so a master can
//! recover to the *last fsynced op* rather than the last snapshot.

use crate::replica::Replica;
use crate::types::{ObjectKey, SegmentName};
use quillcache_core::IdentityScope;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

/// A durable mutation to the master's recoverable state. Only *committed* changes
/// are logged (an in-flight `PutStart` is not durable until `PutEnd`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OpLogEntry {
    SegmentMounted {
        name: SegmentName,
        capacity: u64,
    },
    SegmentUnmounted {
        name: SegmentName,
    },
    /// An object became durable (`PutEnd`) — its replicas + identity + pins, enough
    /// to rebuild it on replay.
    PutCommitted {
        key: ObjectKey,
        identity: IdentityScope,
        replicas: Vec<Replica>,
        soft_pinned: bool,
        hard_pinned: bool,
    },
    Removed {
        key: ObjectKey,
    },
}

/// An append-only, checksummed log file.
#[derive(Debug)]
pub struct OpLog {
    file: File,
}

impl OpLog {
    /// Open `path` for appending (creating it if absent). Existing entries are
    /// kept — `append` adds to the end.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file })
    }

    /// Append one entry and `fsync` — the single durability point. Frame layout:
    /// `[u32 len][u32 crc][payload]`, all little-endian.
    pub fn append(&mut self, entry: &OpLogEntry) -> std::io::Result<()> {
        let payload = serde_json::to_vec(entry)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let crc = crc32fast::hash(&payload);
        let mut frame = Vec::with_capacity(8 + payload.len());
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(&crc.to_le_bytes());
        frame.extend_from_slice(&payload);
        self.file.write_all(&frame)?;
        self.file.sync_data()?;
        Ok(())
    }

    /// Replay every intact entry from `path` (a missing file → empty). Stops at
    /// the first torn frame (short read — a crash mid-append) or CRC mismatch
    /// (corruption), so a half-written tail is never decoded.
    pub fn replay(path: impl AsRef<Path>) -> std::io::Result<Vec<OpLogEntry>> {
        let mut bytes = Vec::new();
        match File::open(path) {
            Ok(mut f) => {
                f.read_to_end(&mut bytes)?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        }
        let mut entries = Vec::new();
        let mut pos = 0;
        while pos + 8 <= bytes.len() {
            let len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            let crc = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap());
            let start = pos + 8;
            let Some(end) = start.checked_add(len) else {
                break;
            };
            if end > bytes.len() {
                break; // torn tail: frame claims more bytes than are present
            }
            let payload = &bytes[start..end];
            if crc32fast::hash(payload) != crc {
                break; // corrupt frame: stop (never decode past it)
            }
            match serde_json::from_slice::<OpLogEntry>(payload) {
                Ok(entry) => entries.push(entry),
                Err(_) => break,
            }
            pos = end;
        }
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "quillcache-oplog-test-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn mounted(n: &str, cap: u64) -> OpLogEntry {
        OpLogEntry::SegmentMounted {
            name: n.into(),
            capacity: cap,
        }
    }

    #[test]
    fn append_then_replay_roundtrips_in_order() {
        let path = tmp("roundtrip");
        {
            let mut log = OpLog::open(&path).unwrap();
            log.append(&mounted("seg-0", 100)).unwrap();
            log.append(&OpLogEntry::Removed { key: "k1".into() })
                .unwrap();
            log.append(&mounted("seg-1", 200)).unwrap();
        }
        let replayed = OpLog::replay(&path).unwrap();
        assert_eq!(
            replayed,
            vec![
                mounted("seg-0", 100),
                OpLogEntry::Removed { key: "k1".into() },
                mounted("seg-1", 200),
            ]
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn replay_of_missing_file_is_empty() {
        let path = tmp("missing");
        assert!(OpLog::replay(&path).unwrap().is_empty());
    }

    #[test]
    fn reopen_appends_not_truncates() {
        let path = tmp("reopen");
        OpLog::open(&path)
            .unwrap()
            .append(&mounted("seg-0", 1))
            .unwrap();
        // A second open + append must keep the first entry (append mode).
        OpLog::open(&path)
            .unwrap()
            .append(&mounted("seg-1", 2))
            .unwrap();
        assert_eq!(OpLog::replay(&path).unwrap().len(), 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn torn_tail_stops_replay_at_the_crash_point() {
        let path = tmp("torn");
        {
            let mut log = OpLog::open(&path).unwrap();
            log.append(&mounted("seg-0", 100)).unwrap();
            log.append(&mounted("seg-1", 200)).unwrap();
        }
        // Simulate a crash mid-append: chop the last few bytes off the file.
        let full = std::fs::metadata(&path).unwrap().len();
        let f = OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(full - 3).unwrap();
        drop(f);
        // The intact first entry survives; the torn second is dropped.
        assert_eq!(OpLog::replay(&path).unwrap(), vec![mounted("seg-0", 100)]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn corrupt_frame_stops_replay() {
        let path = tmp("corrupt");
        {
            let mut log = OpLog::open(&path).unwrap();
            log.append(&mounted("seg-0", 100)).unwrap();
            log.append(&mounted("seg-1", 200)).unwrap();
        }
        // Flip a byte inside the first frame's payload (past the 8-byte header).
        let mut bytes = std::fs::read(&path).unwrap();
        let i = 10;
        bytes[i] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        // CRC mismatch on the first frame → replay stops immediately (empty).
        assert!(OpLog::replay(&path).unwrap().is_empty());
        std::fs::remove_file(&path).ok();
    }
}
