//! RDMA backend (Mooncake's `RdmaTransport`) — **one-sided RDMA**, Mooncake's
//! primary data path and core IP: RDMA READ/WRITE move bytes directly between
//! registered memory by `(remote_addr, rkey)`, with no remote CPU in the loop —
//! exactly QuillCache's `(segment, offset)` model, kernel-bypass.
//!
//! The real verbs path lives behind `--features rdma` (raw `ibverbs-sys`; needs
//! libibverbs at build and an RDMA NIC *or* SoftRoCE/rxe to run). [`one_sided_roundtrip`]
//! is the verified core: register memory, connect an RC queue-pair, and do an
//! RDMA WRITE then READ against a remote MR — proven over SoftRoCE on commodity
//! hardware (see the `#[ignore]` test). The `RdmaTransport` Transport-trait
//! wiring (TCP-side-channel handshake + a per-endpoint QP pool) builds on this
//! proven mechanism and is the remaining increment; without the feature it stays
//! an `Unsupported` stub so the default build needs no NIC.

use super::{LinkClass, TransferError, Transport};
use async_trait::async_trait;
use bytes::Bytes;

/// One-sided RoCE/IB RDMA transport. Holds the local device + GID index used to
/// open queue-pairs; `read_remote`/`write_remote` connect to a remote
/// [`serve_rdma_segment`] over a TCP side-channel and do one-sided RDMA READ /
/// WRITE by `(addr + offset, rkey)`. Real under `--features rdma`; an
/// `Unsupported` stub otherwise, so the default build needs no NIC.
/// Per-endpoint pool of reusable RDMA connections (one RC QP each; ops serialized
/// by the inner mutex). Reusing the QP turns a transfer into register + post +
/// poll instead of a full handshake every call.
#[cfg(feature = "rdma")]
type RdmaPool = std::sync::Arc<
    std::sync::Mutex<
        std::collections::HashMap<String, std::sync::Arc<std::sync::Mutex<verbs::RdmaConnection>>>,
    >,
>;

#[derive(Clone)]
pub struct RdmaTransport {
    pub device: String,
    pub gid_index: u8,
    #[cfg(feature = "rdma")]
    pool: RdmaPool,
}

impl std::fmt::Debug for RdmaTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RdmaTransport")
            .field("device", &self.device)
            .field("gid_index", &self.gid_index)
            .finish()
    }
}

/// Get the pooled connection for `endpoint`, connecting (one handshake) on miss.
#[cfg(feature = "rdma")]
fn pool_connect(
    pool: &RdmaPool,
    endpoint: &str,
    device: &str,
    gid_index: u8,
) -> Result<std::sync::Arc<std::sync::Mutex<verbs::RdmaConnection>>, String> {
    let mut map = pool.lock().map_err(|_| "rdma pool poisoned".to_string())?;
    if let Some(conn) = map.get(endpoint) {
        return Ok(conn.clone());
    }
    let conn = std::sync::Arc::new(std::sync::Mutex::new(verbs::RdmaConnection::connect(
        endpoint, device, gid_index,
    )?));
    map.insert(endpoint.to_string(), conn.clone());
    Ok(conn)
}

impl Default for RdmaTransport {
    fn default() -> Self {
        // rxe0 / GID index 1 = RoCEv2 IPv4 on a SoftRoCE box; override for a NIC.
        Self {
            device: "rxe0".into(),
            gid_index: 1,
            #[cfg(feature = "rdma")]
            pool: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }
}

#[cfg(not(feature = "rdma"))]
const NEEDS_NIC: &str =
    "RDMA needs --features rdma + an ibverbs device (a real NIC or SoftRoCE/rxe)";

#[async_trait]
impl Transport for RdmaTransport {
    fn name(&self) -> &str {
        "rdma"
    }

    fn link_class(&self) -> LinkClass {
        LinkClass::RdmaRoce
    }

    async fn read_remote(
        &self,
        endpoint: &str,
        offset: u64,
        length: u64,
    ) -> Result<Bytes, TransferError> {
        #[cfg(feature = "rdma")]
        {
            let (ep, device, gid, pool) = (
                endpoint.to_string(),
                self.device.clone(),
                self.gid_index,
                self.pool.clone(),
            );
            let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
                let conn = pool_connect(&pool, &ep, &device, gid)?;
                let conn = conn.lock().map_err(|_| "rdma conn poisoned".to_string())?;
                conn.read(offset, length as usize)
            })
            .await
            .map_err(|e| TransferError::Io(e.to_string()))?
            .map_err(TransferError::Io)?;
            Ok(Bytes::from(bytes))
        }
        #[cfg(not(feature = "rdma"))]
        {
            let _ = (endpoint, offset, length);
            Err(TransferError::Unsupported(NEEDS_NIC))
        }
    }

    async fn write_remote(
        &self,
        endpoint: &str,
        offset: u64,
        data: Bytes,
    ) -> Result<(), TransferError> {
        #[cfg(feature = "rdma")]
        {
            let (ep, device, gid, pool) = (
                endpoint.to_string(),
                self.device.clone(),
                self.gid_index,
                self.pool.clone(),
            );
            tokio::task::spawn_blocking(move || -> Result<(), String> {
                let conn = pool_connect(&pool, &ep, &device, gid)?;
                let conn = conn.lock().map_err(|_| "rdma conn poisoned".to_string())?;
                conn.write(offset, &data)
            })
            .await
            .map_err(|e| TransferError::Io(e.to_string()))?
            .map_err(TransferError::Io)?;
            Ok(())
        }
        #[cfg(not(feature = "rdma"))]
        {
            let _ = (endpoint, offset, data);
            Err(TransferError::Unsupported(NEEDS_NIC))
        }
    }
}

// =============================================================================
// Real one-sided RDMA over ibverbs (feature `rdma`). Verified over SoftRoCE.
// =============================================================================

#[cfg(feature = "rdma")]
pub use verbs::{
    one_sided_roundtrip, rdma_read, rdma_write, serve_rdma_segment, RdmaConnection, RdmaSegment,
};

#[cfg(feature = "rdma")]
mod verbs {
    use ibverbs_sys as ffi;
    use std::ffi::{c_int, c_void, CStr};
    use std::ptr;

    /// One queue-pair's wire identity, exchanged to connect two RC QPs.
    #[derive(Clone, Copy)]
    struct QpEndpoint {
        qpn: u32,
        psn: u32,
        gid: ffi::ibv_gid,
    }

    fn errno() -> i32 {
        std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
    }

    /// Open an ibverbs device by name (e.g. "rxe0"); first device if `None`.
    unsafe fn open_device(want: Option<&str>) -> Result<*mut ffi::ibv_context, String> {
        let mut num: c_int = 0;
        let list = ffi::ibv_get_device_list(&mut num as *mut _);
        if list.is_null() || num == 0 {
            return Err("no RDMA devices (is rxe0 up? `rdma link show`)".into());
        }
        let mut chosen: *mut ffi::ibv_device = ptr::null_mut();
        for i in 0..num as isize {
            let dev = *list.offset(i);
            match want {
                None => {
                    chosen = dev;
                    break;
                }
                Some(name) => {
                    let dname = CStr::from_ptr(ffi::ibv_get_device_name(dev));
                    if dname.to_string_lossy() == name {
                        chosen = dev;
                        break;
                    }
                }
            }
        }
        if chosen.is_null() {
            ffi::ibv_free_device_list(list);
            return Err(format!("RDMA device {want:?} not found"));
        }
        let ctx = ffi::ibv_open_device(chosen);
        ffi::ibv_free_device_list(list);
        if ctx.is_null() {
            return Err("ibv_open_device failed".into());
        }
        Ok(ctx)
    }

    /// The RoCEv2 GID at `gid_index` on `port` (index 1 = RoCEv2 IPv4 on rxe).
    unsafe fn query_gid(
        ctx: *mut ffi::ibv_context,
        port: u8,
        gid_index: c_int,
    ) -> Result<ffi::ibv_gid, String> {
        let mut gid: ffi::ibv_gid = std::mem::zeroed();
        if ffi::ibv_query_gid(ctx, port, gid_index, &mut gid as *mut _) != 0 {
            return Err(format!(
                "ibv_query_gid(port={port}, index={gid_index}) failed"
            ));
        }
        Ok(gid)
    }

    /// Post a single one-sided RDMA WR (WRITE or READ) and wait for its completion.
    #[allow(clippy::too_many_arguments)]
    unsafe fn post_and_wait(
        qp: *mut ffi::ibv_qp,
        cq: *mut ffi::ibv_cq,
        opcode: ffi::ibv_wr_opcode::Type,
        local_addr: u64,
        lkey: u32,
        len: u32,
        remote_addr: u64,
        rkey: u32,
    ) -> Result<(), String> {
        let mut sge = ffi::ibv_sge {
            addr: local_addr,
            length: len,
            lkey,
        };
        let mut wr: ffi::ibv_send_wr = std::mem::zeroed();
        wr.wr_id = 1;
        wr.next = ptr::null_mut();
        wr.sg_list = &mut sge as *mut _;
        wr.num_sge = 1;
        wr.opcode = opcode;
        wr.send_flags = ffi::ibv_send_flags::IBV_SEND_SIGNALED.0;
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey = rkey;

        let mut bad: *mut ffi::ibv_send_wr = ptr::null_mut();
        let ctx = (*qp).context;
        let ops = &mut (*ctx).ops;
        let rc = ops.post_send.as_mut().unwrap()(qp, &mut wr as *mut _, &mut bad as *mut _);
        if rc != 0 {
            return Err(format!("ibv_post_send rc={rc} errno={}", errno()));
        }

        // Poll the CQ until the completion lands.
        let cqctx = (*cq).context;
        let cqops = &mut (*cqctx).ops;
        let mut wc: ffi::ibv_wc = std::mem::zeroed();
        loop {
            let n = cqops.poll_cq.as_mut().unwrap()(cq, 1, &mut wc as *mut _);
            if n < 0 {
                return Err("ibv_poll_cq failed".into());
            }
            if n == 0 {
                continue;
            }
            if let Some((status, vendor_err)) = wc.error() {
                return Err(format!(
                    "RDMA op failed: wc.status={status} vendor_err={vendor_err}"
                ));
            }
            return Ok(());
        }
    }

    /// Bring an RC QP INIT → RTR → RTS, connected to `remote` over RoCEv2.
    unsafe fn connect_qp(
        qp: *mut ffi::ibv_qp,
        local_psn: u32,
        remote: &QpEndpoint,
        port: u8,
        gid_index: u8,
    ) -> Result<(), String> {
        use ffi::ibv_qp_attr_mask as M;

        // INIT
        let mut attr: ffi::ibv_qp_attr = std::mem::zeroed();
        attr.qp_state = ffi::ibv_qp_state::IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = port;
        attr.qp_access_flags = ffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
            | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0
            | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ.0;
        let mask = (M::IBV_QP_STATE.0
            | M::IBV_QP_PKEY_INDEX.0
            | M::IBV_QP_PORT.0
            | M::IBV_QP_ACCESS_FLAGS.0) as c_int;
        if ffi::ibv_modify_qp(qp, &mut attr as *mut _, mask) != 0 {
            return Err(format!("modify_qp INIT failed errno={}", errno()));
        }

        // RTR — RoCE requires global routing (GRH) with the remote's GID.
        let mut attr: ffi::ibv_qp_attr = std::mem::zeroed();
        attr.qp_state = ffi::ibv_qp_state::IBV_QPS_RTR;
        attr.path_mtu = ffi::IBV_MTU_1024;
        attr.dest_qp_num = remote.qpn;
        attr.rq_psn = remote.psn;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.dlid = 0;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = port;
        attr.ah_attr.grh.dgid = remote.gid;
        attr.ah_attr.grh.sgid_index = gid_index;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.grh.traffic_class = 0;
        attr.ah_attr.grh.flow_label = 0;
        let mask = (M::IBV_QP_STATE.0
            | M::IBV_QP_AV.0
            | M::IBV_QP_PATH_MTU.0
            | M::IBV_QP_DEST_QPN.0
            | M::IBV_QP_RQ_PSN.0
            | M::IBV_QP_MAX_DEST_RD_ATOMIC.0
            | M::IBV_QP_MIN_RNR_TIMER.0) as c_int;
        if ffi::ibv_modify_qp(qp, &mut attr as *mut _, mask) != 0 {
            return Err(format!("modify_qp RTR failed errno={}", errno()));
        }

        // RTS
        let mut attr: ffi::ibv_qp_attr = std::mem::zeroed();
        attr.qp_state = ffi::ibv_qp_state::IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = local_psn;
        attr.max_rd_atomic = 1;
        let mask = (M::IBV_QP_STATE.0
            | M::IBV_QP_TIMEOUT.0
            | M::IBV_QP_RETRY_CNT.0
            | M::IBV_QP_RNR_RETRY.0
            | M::IBV_QP_SQ_PSN.0
            | M::IBV_QP_MAX_QP_RD_ATOMIC.0) as c_int;
        if ffi::ibv_modify_qp(qp, &mut attr as *mut _, mask) != 0 {
            return Err(format!("modify_qp RTS failed errno={}", errno()));
        }
        Ok(())
    }

    /// Real one-sided RDMA round-trip over the given device (e.g. "rxe0"): two RC
    /// queue-pairs in one process, connected RoCEv2. RDMA-WRITE `payload` from a
    /// source MR into a destination MR (`(dst.addr, dst.rkey)`), then RDMA-READ it
    /// back from the destination into a third MR — proving both one-sided verbs.
    /// Returns the read-back bytes (the caller asserts they equal `payload`).
    ///
    /// This is the exact mechanism QuillCache's transfer engine moves KV with on
    /// real RDMA — no remote CPU touches the data; the HCA does the copy by
    /// `(remote_addr, rkey)`. Verified over SoftRoCE; a real NIC is a drop-in.
    pub fn one_sided_roundtrip(device: &str, payload: &[u8]) -> Result<Vec<u8>, String> {
        const PORT: u8 = 1;
        const GID_INDEX: u8 = 1; // RoCEv2 IPv4 on rxe (see `rdma link`/gid_attrs)
        const PSN_A: u32 = 0;
        const PSN_B: u32 = 0;
        let len = payload.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        unsafe {
            let ctx = open_device(Some(device))?;
            let pd = ffi::ibv_alloc_pd(ctx);
            if pd.is_null() {
                return Err("ibv_alloc_pd failed".into());
            }
            let gid = query_gid(ctx, PORT, GID_INDEX as c_int)?;

            // Three host buffers + MRs in the one PD: source, destination, readback.
            let mut src = payload.to_vec();
            let mut dst = vec![0u8; len];
            let mut back = vec![0u8; len];
            let access = (ffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0
                | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ.0)
                as c_int;
            let reg = |buf: &mut [u8]| -> *mut ffi::ibv_mr {
                ffi::ibv_reg_mr(pd, buf.as_mut_ptr() as *mut c_void, buf.len(), access)
            };
            let mr_src = reg(&mut src);
            let mr_dst = reg(&mut dst);
            let mr_back = reg(&mut back);
            if mr_src.is_null() || mr_dst.is_null() || mr_back.is_null() {
                return Err("ibv_reg_mr failed".into());
            }

            // Two RC queue-pairs (A active, B passive), each with its own CQ.
            let cq_a = ffi::ibv_create_cq(ctx, 16, ptr::null_mut(), ptr::null_mut(), 0);
            let cq_b = ffi::ibv_create_cq(ctx, 16, ptr::null_mut(), ptr::null_mut(), 0);
            if cq_a.is_null() || cq_b.is_null() {
                return Err("ibv_create_cq failed".into());
            }
            let mk_qp = |cq: *mut ffi::ibv_cq| -> *mut ffi::ibv_qp {
                let mut ia: ffi::ibv_qp_init_attr = std::mem::zeroed();
                ia.send_cq = cq;
                ia.recv_cq = cq;
                ia.qp_type = ffi::ibv_qp_type::IBV_QPT_RC;
                ia.cap.max_send_wr = 16;
                ia.cap.max_recv_wr = 16;
                ia.cap.max_send_sge = 1;
                ia.cap.max_recv_sge = 1;
                ffi::ibv_create_qp(pd, &mut ia as *mut _)
            };
            let qp_a = mk_qp(cq_a);
            let qp_b = mk_qp(cq_b);
            if qp_a.is_null() || qp_b.is_null() {
                return Err("ibv_create_qp failed".into());
            }

            let ep_a = QpEndpoint {
                qpn: (*qp_a).qp_num,
                psn: PSN_A,
                gid,
            };
            let ep_b = QpEndpoint {
                qpn: (*qp_b).qp_num,
                psn: PSN_B,
                gid,
            };
            connect_qp(qp_a, PSN_A, &ep_b, PORT, GID_INDEX)?;
            connect_qp(qp_b, PSN_B, &ep_a, PORT, GID_INDEX)?;

            // 1) RDMA WRITE: src -> dst (one-sided; qp_b's CPU is not involved).
            post_and_wait(
                qp_a,
                cq_a,
                ffi::ibv_wr_opcode::IBV_WR_RDMA_WRITE,
                (*mr_src).addr as u64,
                (*mr_src).lkey,
                len as u32,
                (*mr_dst).addr as u64,
                (*mr_dst).rkey,
            )?;
            if dst != payload {
                return Err("RDMA WRITE did not land the payload in dst".into());
            }

            // 2) RDMA READ: dst -> back (read remote memory by (addr, rkey)).
            post_and_wait(
                qp_a,
                cq_a,
                ffi::ibv_wr_opcode::IBV_WR_RDMA_READ,
                (*mr_back).addr as u64,
                (*mr_back).lkey,
                len as u32,
                (*mr_dst).addr as u64,
                (*mr_dst).rkey,
            )?;

            // Teardown (best-effort).
            ffi::ibv_destroy_qp(qp_a);
            ffi::ibv_destroy_qp(qp_b);
            ffi::ibv_destroy_cq(cq_a);
            ffi::ibv_destroy_cq(cq_b);
            ffi::ibv_dereg_mr(mr_src);
            ffi::ibv_dereg_mr(mr_dst);
            ffi::ibv_dereg_mr(mr_back);
            ffi::ibv_dealloc_pd(pd);
            ffi::ibv_close_device(ctx);

            Ok(back)
        }
    }

    // ---- Cross-node transport: a registered segment served over a TCP
    // side-channel QP handshake, and a client doing one-sided RDMA against it.
    // Same verbs as one_sided_roundtrip, split across two ibv contexts that
    // exchange QP identities + the segment's (addr, rkey) over TCP. ----

    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    const PORT: u8 = 1;
    const GID_INDEX: u8 = 1; // RoCEv2 IPv4 on rxe

    fn access_flags() -> c_int {
        (ffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
            | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0
            | ffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ.0) as c_int
    }

    /// Create an RC queue-pair on `pd` whose completions land in `cq`.
    unsafe fn create_rc_qp(
        pd: *mut ffi::ibv_pd,
        cq: *mut ffi::ibv_cq,
    ) -> Result<*mut ffi::ibv_qp, String> {
        let mut ia: ffi::ibv_qp_init_attr = std::mem::zeroed();
        ia.send_cq = cq;
        ia.recv_cq = cq;
        ia.qp_type = ffi::ibv_qp_type::IBV_QPT_RC;
        ia.cap.max_send_wr = 16;
        ia.cap.max_recv_wr = 16;
        ia.cap.max_send_sge = 1;
        ia.cap.max_recv_sge = 1;
        let qp = ffi::ibv_create_qp(pd, &mut ia as *mut _);
        if qp.is_null() {
            return Err("ibv_create_qp failed".into());
        }
        Ok(qp)
    }

    // Wire: client→server = 24B {qpn(4) psn(4) gid(16)};
    //       server→client = 24B endpoint + mr_addr(8) + rkey(4) + capacity(8) = 44B.
    unsafe fn endpoint_to_bytes(ep: &QpEndpoint) -> [u8; 24] {
        let mut b = [0u8; 24];
        b[0..4].copy_from_slice(&ep.qpn.to_be_bytes());
        b[4..8].copy_from_slice(&ep.psn.to_be_bytes());
        b[8..24].copy_from_slice(&ep.gid.raw);
        b
    }
    unsafe fn endpoint_from_bytes(b: &[u8]) -> QpEndpoint {
        let qpn = u32::from_be_bytes(b[0..4].try_into().unwrap());
        let psn = u32::from_be_bytes(b[4..8].try_into().unwrap());
        let mut gid: ffi::ibv_gid = std::mem::zeroed();
        gid.raw.copy_from_slice(&b[8..24]);
        QpEndpoint { qpn, psn, gid }
    }

    /// A registered RAM segment served to RDMA clients (Mooncake's storage-node
    /// segment, RDMA form): clients RDMA one-sidedly into/out of its MR by
    /// `(addr, rkey)` — this node's CPU never touches the bytes.
    pub struct RdmaSegment {
        ctx: *mut ffi::ibv_context,
        pd: *mut ffi::ibv_pd,
        mr: *mut ffi::ibv_mr,
        buf: *mut u8,
        capacity: usize,
        len: AtomicUsize,
        gid: ffi::ibv_gid,
    }
    // The verbs objects + buffer are owned for the segment's lifetime and only
    // mutated by the (one-sided) HCA + setup, so sharing across accept threads is
    // sound for this usage.
    unsafe impl Send for RdmaSegment {}
    unsafe impl Sync for RdmaSegment {}

    impl RdmaSegment {
        /// Open `device` (e.g. "rxe0") and register `capacity` bytes of RAM as an
        /// MR with remote read/write, ready to be RDMA'd by clients.
        pub fn new(device: &str, capacity: usize) -> Result<Self, String> {
            unsafe {
                let ctx = open_device(Some(device))?;
                let pd = ffi::ibv_alloc_pd(ctx);
                if pd.is_null() {
                    return Err("ibv_alloc_pd failed".into());
                }
                let gid = query_gid(ctx, PORT, GID_INDEX as c_int)?;
                let mut boxed = vec![0u8; capacity].into_boxed_slice();
                let buf = boxed.as_mut_ptr();
                std::mem::forget(boxed); // freed in Drop via buf/capacity
                let mr = ffi::ibv_reg_mr(pd, buf as *mut c_void, capacity, access_flags());
                if mr.is_null() {
                    return Err("ibv_reg_mr failed".into());
                }
                Ok(Self {
                    ctx,
                    pd,
                    mr,
                    buf,
                    capacity,
                    len: AtomicUsize::new(0),
                    gid,
                })
            }
        }

        /// Preload bytes into the segment (local copy); returns the offset (0).
        pub fn register(&self, data: &[u8]) -> Result<u64, String> {
            if data.len() > self.capacity {
                return Err("data exceeds segment capacity".into());
            }
            unsafe { ptr::copy_nonoverlapping(data.as_ptr(), self.buf, data.len()) };
            self.len.store(data.len(), Ordering::SeqCst);
            Ok(0)
        }

        /// Read bytes back from the segment's memory (what a client RDMA-wrote).
        pub fn read(&self, offset: usize, len: usize) -> Result<Vec<u8>, String> {
            if offset.checked_add(len).is_none_or(|e| e > self.capacity) {
                return Err("read out of segment bounds".into());
            }
            let mut out = vec![0u8; len];
            unsafe { ptr::copy_nonoverlapping(self.buf.add(offset), out.as_mut_ptr(), len) };
            Ok(out)
        }

        /// Logical length (bytes preloaded via [`Self::register`]).
        pub fn len(&self) -> usize {
            self.len.load(Ordering::SeqCst)
        }

        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        /// Serve one client: exchange QP identities + the MR handle, connect the
        /// RC QP, signal ready, then hold the QP alive while the client RDMAs
        /// (until it closes the side-channel).
        fn serve_one(&self, mut stream: TcpStream) -> Result<(), String> {
            unsafe {
                let cq = ffi::ibv_create_cq(self.ctx, 16, ptr::null_mut(), ptr::null_mut(), 0);
                if cq.is_null() {
                    return Err("ibv_create_cq failed".into());
                }
                let qp = create_rc_qp(self.pd, cq)?;

                let mut cbuf = [0u8; 24];
                stream.read_exact(&mut cbuf).map_err(|e| e.to_string())?;
                let client = endpoint_from_bytes(&cbuf);

                let server = QpEndpoint {
                    qpn: (*qp).qp_num,
                    psn: 0,
                    gid: self.gid,
                };
                let mut sbuf = [0u8; 44];
                sbuf[0..24].copy_from_slice(&endpoint_to_bytes(&server));
                sbuf[24..32].copy_from_slice(&(self.buf as u64).to_be_bytes());
                sbuf[32..36].copy_from_slice(&(*self.mr).rkey.to_be_bytes());
                sbuf[36..44].copy_from_slice(&(self.capacity as u64).to_be_bytes());
                stream.write_all(&sbuf).map_err(|e| e.to_string())?;

                connect_qp(qp, 0, &client, PORT, GID_INDEX)?;
                stream.write_all(&[1u8]).map_err(|e| e.to_string())?; // "ready"

                // Block until the client signals done / closes — its one-sided
                // RDMA runs against our MR meanwhile, with no CPU of ours involved.
                let mut done = [0u8; 1];
                let _ = stream.read(&mut done);

                ffi::ibv_destroy_qp(qp);
                ffi::ibv_destroy_cq(cq);
                Ok(())
            }
        }
    }

    impl Drop for RdmaSegment {
        fn drop(&mut self) {
            unsafe {
                ffi::ibv_dereg_mr(self.mr);
                ffi::ibv_dealloc_pd(self.pd);
                ffi::ibv_close_device(self.ctx);
                drop(Box::from_raw(std::slice::from_raw_parts_mut(
                    self.buf,
                    self.capacity,
                )));
            }
        }
    }

    /// Serve an [`RdmaSegment`] to clients on `listener` (one QP per connection).
    pub fn serve_rdma_segment(segment: Arc<RdmaSegment>, listener: TcpListener) {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { return };
            let segment = segment.clone();
            std::thread::spawn(move || {
                let _ = segment.serve_one(stream);
            });
        }
    }

    /// A reusable RDMA connection to a remote [`RdmaSegment`]: the RC queue-pair
    /// (and the side-channel that keeps the server's QP alive) stay open across
    /// many one-sided ops. Pooled per endpoint (see [`super::RdmaTransport`]) so a
    /// transfer costs only register + post + poll, not a full QP handshake per call.
    pub struct RdmaConnection {
        ctx: *mut ffi::ibv_context,
        pd: *mut ffi::ibv_pd,
        cq: *mut ffi::ibv_cq,
        qp: *mut ffi::ibv_qp,
        stream: TcpStream,
        server_addr: u64,
        server_rkey: u32,
    }
    // Held behind a Mutex in the per-endpoint pool; ops are serialized, so sharing
    // the raw verbs handles across the pool's worker threads is sound.
    unsafe impl Send for RdmaConnection {}

    impl RdmaConnection {
        /// Open + connect an RC QP to `endpoint`'s served segment (one handshake);
        /// learns the segment's `(addr, rkey)` and keeps the side-channel open.
        pub fn connect(endpoint: &str, device: &str, gid_index: u8) -> Result<Self, String> {
            let mut stream = TcpStream::connect(endpoint).map_err(|e| e.to_string())?;
            unsafe {
                let ctx = open_device(Some(device))?;
                let pd = ffi::ibv_alloc_pd(ctx);
                if pd.is_null() {
                    return Err("ibv_alloc_pd failed".into());
                }
                let gid = query_gid(ctx, PORT, gid_index as c_int)?;
                let cq = ffi::ibv_create_cq(ctx, 16, ptr::null_mut(), ptr::null_mut(), 0);
                if cq.is_null() {
                    return Err("ibv_create_cq failed".into());
                }
                let qp = create_rc_qp(pd, cq)?;

                let client = QpEndpoint {
                    qpn: (*qp).qp_num,
                    psn: 0,
                    gid,
                };
                stream
                    .write_all(&endpoint_to_bytes(&client))
                    .map_err(|e| e.to_string())?;
                let mut sbuf = [0u8; 44];
                stream.read_exact(&mut sbuf).map_err(|e| e.to_string())?;
                let server = endpoint_from_bytes(&sbuf[0..24]);
                let server_addr = u64::from_be_bytes(sbuf[24..32].try_into().unwrap());
                let server_rkey = u32::from_be_bytes(sbuf[32..36].try_into().unwrap());

                connect_qp(qp, 0, &server, PORT, gid_index)?;
                let mut ready = [0u8; 1];
                stream.read_exact(&mut ready).map_err(|e| e.to_string())?;

                Ok(Self {
                    ctx,
                    pd,
                    cq,
                    qp,
                    stream,
                    server_addr,
                    server_rkey,
                })
            }
        }

        /// One one-sided op over the reused QP: register the buffer, post, poll.
        fn op(
            &self,
            opcode: ffi::ibv_wr_opcode::Type,
            buf: &mut [u8],
            offset: u64,
        ) -> Result<(), String> {
            unsafe {
                let mr = ffi::ibv_reg_mr(
                    self.pd,
                    buf.as_mut_ptr() as *mut c_void,
                    buf.len(),
                    access_flags(),
                );
                if mr.is_null() {
                    return Err("ibv_reg_mr failed".into());
                }
                let res = post_and_wait(
                    self.qp,
                    self.cq,
                    opcode,
                    buf.as_ptr() as u64,
                    (*mr).lkey,
                    buf.len() as u32,
                    self.server_addr + offset,
                    self.server_rkey,
                );
                ffi::ibv_dereg_mr(mr);
                res
            }
        }

        /// RDMA-WRITE `data` into the remote segment at `offset` (reuses the QP).
        pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), String> {
            let mut buf = data.to_vec();
            self.op(ffi::ibv_wr_opcode::IBV_WR_RDMA_WRITE, &mut buf, offset)
        }

        /// RDMA-READ `len` bytes from the remote segment at `offset` (reuses the QP).
        pub fn read(&self, offset: u64, len: usize) -> Result<Vec<u8>, String> {
            let mut buf = vec![0u8; len];
            self.op(ffi::ibv_wr_opcode::IBV_WR_RDMA_READ, &mut buf, offset)?;
            Ok(buf)
        }
    }

    impl Drop for RdmaConnection {
        fn drop(&mut self) {
            unsafe {
                let _ = self.stream.write_all(&[1u8]); // tell the server to free its QP
                ffi::ibv_destroy_qp(self.qp);
                ffi::ibv_destroy_cq(self.cq);
                ffi::ibv_dealloc_pd(self.pd);
                ffi::ibv_close_device(self.ctx);
            }
        }
    }

    /// RDMA-WRITE `data` into the remote segment at `offset` (one-shot connect).
    pub fn rdma_write(
        endpoint: &str,
        offset: u64,
        data: &[u8],
        device: &str,
        gid_index: u8,
    ) -> Result<(), String> {
        RdmaConnection::connect(endpoint, device, gid_index)?.write(offset, data)
    }

    /// RDMA-READ `len` bytes from the remote segment at `offset` (one-shot connect).
    pub fn rdma_read(
        endpoint: &str,
        offset: u64,
        len: usize,
        device: &str,
        gid_index: u8,
    ) -> Result<Vec<u8>, String> {
        RdmaConnection::connect(endpoint, device, gid_index)?.read(offset, len)
    }
}

#[cfg(all(test, feature = "rdma"))]
mod tests {
    use super::*;

    // Needs an ibverbs device — a real RDMA NIC or SoftRoCE (rxe). Set it up with
    // `sudo modprobe rdma_rxe && sudo rdma link add rxe0 type rxe netdev <iface>`,
    // then: `cargo test -p quillcache-transfer-engine --features rdma -- --ignored`.
    #[test]
    #[ignore = "requires an ibverbs device (real NIC or SoftRoCE/rxe)"]
    fn one_sided_rdma_write_then_read_roundtrip() {
        let payload: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let back = one_sided_roundtrip("rxe0", &payload).expect("one-sided RDMA round-trip");
        assert_eq!(
            back, payload,
            "RDMA WRITE then READ must round-trip the bytes"
        );
    }

    // The full Transport over the wire: a served `RdmaSegment` + the async
    // `RdmaTransport` client — two ibv contexts handshaking over localhost TCP and
    // moving bytes one-sidedly by (addr, rkey) over SoftRoCE. Run on a box with
    // rxe up (see above): `cargo test -p quillcache-transfer-engine --features rdma
    // -- --ignored`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires an ibverbs device (real NIC or SoftRoCE/rxe)"]
    async fn rdma_transport_write_then_read_over_the_wire() {
        use std::sync::Arc;
        let segment = Arc::new(RdmaSegment::new("rxe0", 1 << 20).expect("RdmaSegment"));
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind side-channel");
        let endpoint = listener.local_addr().unwrap().to_string();
        let server = segment.clone();
        std::thread::spawn(move || serve_rdma_segment(server, listener));

        let t = RdmaTransport::default(); // rxe0, gid index 1
        let payload: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();

        // Client RDMA-WRITEs into the server's segment (one-sided)...
        t.write_remote(&endpoint, 0, Bytes::from(payload.clone()))
            .await
            .expect("write_remote");
        // ...the server sees the bytes in its own MR (the WRITE truly landed)...
        assert_eq!(
            segment.read(0, 4096).unwrap(),
            payload,
            "RDMA WRITE landed in the served segment"
        );
        // ...and the client RDMA-READs them straight back.
        let got = t
            .read_remote(&endpoint, 0, 4096)
            .await
            .expect("read_remote");
        assert_eq!(
            &got[..],
            &payload[..],
            "write_remote then read_remote must round-trip over RDMA"
        );
    }

    // QP pooling pays off: reusing one connected queue-pair across N ops vs a full
    // handshake + teardown per op. Prints `QC-RDMA-POOL per_call=.. pooled=..
    // speedup=..` and asserts the reuse wins. Needs rxe (see above).
    #[test]
    #[ignore = "requires an ibverbs device (real NIC or SoftRoCE/rxe)"]
    fn rdma_qp_pool_beats_per_call_connect() {
        use std::sync::Arc;
        use std::time::Instant;
        let segment = Arc::new(RdmaSegment::new("rxe0", 1 << 20).expect("RdmaSegment"));
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind side-channel");
        let endpoint = listener.local_addr().unwrap().to_string();
        let server = segment.clone();
        std::thread::spawn(move || serve_rdma_segment(server, listener));
        std::thread::sleep(std::time::Duration::from_millis(100));

        let payload = vec![7u8; 4096];
        let iters = 200u32;

        // Per-call: a full QP handshake + teardown on every op.
        let t = Instant::now();
        for _ in 0..iters {
            rdma_write(&endpoint, 0, &payload, "rxe0", 1).expect("per-call write");
        }
        let per_call_us = t.elapsed().as_micros() as f64 / f64::from(iters);

        // Pooled: one connected QP, reused for every op (register + post + poll).
        let conn = RdmaConnection::connect(&endpoint, "rxe0", 1).expect("connect");
        let t = Instant::now();
        for _ in 0..iters {
            conn.write(0, &payload).expect("pooled write");
        }
        let pooled_us = t.elapsed().as_micros() as f64 / f64::from(iters);

        eprintln!(
            "QC-RDMA-POOL per_call={per_call_us:.1}us/op pooled={pooled_us:.1}us/op speedup={:.1}x",
            per_call_us / pooled_us
        );
        assert!(
            pooled_us < per_call_us,
            "QP reuse must beat per-call connect (pooled {pooled_us} vs per-call {per_call_us})"
        );
    }
}
