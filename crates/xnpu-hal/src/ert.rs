//! ERT command packets: build a DPU-start command in a CMD BO, submit it
//! with EXEC_CMD, and wait on the context syncobj timeline.
//!
//! Wire contract (captured from XRT's own exec buf with an LD_PRELOAD ioctl
//! tracer, and cross-checked against the in-tree kernel):
//! the CMD BO holds `ert_start_kernel_cmd { header, cu_mask, data[] }` where
//! `count` (header bits [22:12]) counts the u32 words after the header,
//! *including* the cu-mask word. XRT sends opcode **0** (`ERT_START_CU`) with
//! type 3 (`ERT_CU`) and *no* `ert_npu_data` preamble: the payload right
//! after the cu mask is the register map itself. The kernel's
//! `aie2_cmdlist_fill_npu_cf` then tags the chain slot
//! `EXEC_NPU_TYPE_NON_ELF` and copies the payload verbatim into the firmware
//! exec args. (Opcode 20 / `ERT_START_NPU` instead takes the PARTIAL_ELF
//! path, which expects ELF ctrl code and silently no-ops on raw txn bins.)
//!
//! Regmap for the mlir_aie DPU pseudo-kernel, by convention:
//! `opcode u64 = 3, instr u64 (ctrl-code BO heap address), ninstr u32,
//! bo0..boN u64`. Instruction address is the ctrl BO's heap `xdna_addr`;
//! tensor addresses are the SHMEM BOs' *user* VAs (fw walks host page
//! tables via SVM/PASID — heap addresses there are silently ignored).
//! `ninstr` counts 32-bit WORDS, not bytes (P20b: mlir_aie's host runtime
//! passes `len(np.frombuffer(data, np.uint32))`; bytes here made the fw
//! pull 4x the ctrl code — 48KB of heap garbage past the quad's 16528B
//! ctrl BO — the intermittent corrupt/hang on the engine path while the
//! pyxrt path stayed bit-stable).

use std::io;

use crate::accel::Device;
use crate::bo::{BoType, BufferObject, Mapping};
use crate::hwctx::HwContext;
use crate::ioctl::{amdxdna_ioctl, raw_ioctl, AmdxdnaCmd, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT};

/// enum ert_cmd_state.
pub const ERT_CMD_STATE_NEW: u32 = 1;
pub const ERT_CMD_STATE_COMPLETED: u32 = 4;
pub const ERT_CMD_STATE_ERROR: u32 = 5;
pub const ERT_CMD_STATE_ABORT: u32 = 6;

/// enum ert_cmd_opcode: XRT's NPU2 exec packets use ERT_START_CU (0), which
/// the kernel maps to the NON_ELF chain slot (raw txn ctrl code).
const ERT_START_CU: u32 = 0;
/// enum ert_cmd_type.
const ERT_CU: u32 = 3;

/// The KMQ shim's exec BO size.
const MAX_EXEC_BO_SIZE: usize = 4096;

/// Offsets within the CMD BO, in bytes (single cu mask word).
const OFF_CU_MASK: usize = 4;
const OFF_REGMAP: usize = 8;

fn make_header(state: u32, count_words: u32, opcode: u32, ty: u32) -> u32 {
    // state:4 | stat_enabled:1 | unused:5 | extra_cu_masks:2 | count:11 |
    // opcode:5 | type:4  (ert_start_kernel_cmd, ert.h)
    (state & 0xf)
        | ((count_words & 0x7ff) << 12)
        | ((opcode & 0x1f) << 23)
        | ((ty & 0xf) << 28)
}

/// A CMD BO holding one ERT_START_NPU packet under construction.
///
/// Build order mirrors the shim's kernel.cpp: set the CU, set the ctrl
/// (instruction) BO once, then append register-map arguments in the order the
/// DPU control code expects them.
pub struct StartNpuCmd {
    bo: BufferObject,
    map: Mapping,
    /// u32 words after the header (cu mask + npu data + regmap).
    count: u32,
    /// Regmap words appended so far.
    regmap: u32,
}

impl StartNpuCmd {
    pub fn new(device: &Device) -> io::Result<StartNpuCmd> {
        let bo = BufferObject::new(device, BoType::Cmd, MAX_EXEC_BO_SIZE)?;
        let mut map = bo.map_owned()?;
        map.as_mut_slice().fill(0);
        let mut pkt = StartNpuCmd {
            bo,
            map,
            count: 1, // one word for the cu mask
            regmap: 0,
        };
        pkt.write_header();
        Ok(pkt)
    }

    fn write_header(&mut self) {
        let h = make_header(ERT_CMD_STATE_NEW, self.count, ERT_START_CU, ERT_CU);
        self.map.as_mut_slice()[0..4].copy_from_slice(&h.to_le_bytes());
    }

    fn room(&self, extra_words: u32) -> io::Result<()> {
        let end = OFF_REGMAP + ((self.regmap + extra_words) as usize) * 4;
        if end > self.map.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ERT packet regmap overflow",
            ));
        }
        Ok(())
    }

    /// Select the CU to run on (cu_mask = 1 << idx). The kernel picks the
    /// first set bit across the mask words as cu_idx.
    pub fn set_cu(&mut self, idx: u32) {
        self.map.as_mut_slice()[OFF_CU_MASK..OFF_CU_MASK + 4]
            .copy_from_slice(&(1u32 << idx).to_le_bytes());
    }

    /// Point the packet at the ctrl-code (instruction) BO. In XRT's wire
    /// format the instruction buffer/size are ordinary regmap entries, so
    /// this is a no-op kept for call-site symmetry.
    pub fn set_ctrl(&mut self, _instr: u64, _size: u32) -> io::Result<()> {
        self.room(0)?;
        Ok(())
    }

    /// Append a u32 register-map argument.
    pub fn arg32(&mut self, v: u32) -> io::Result<()> {
        self.room(1)?;
        let off = OFF_REGMAP + (self.regmap as usize) * 4;
        self.map.as_mut_slice()[off..off + 4].copy_from_slice(&v.to_le_bytes());
        self.regmap += 1;
        self.count += 1;
        self.write_header();
        Ok(())
    }

    /// Append a u64 register-map argument (two words, low first).
    pub fn arg64(&mut self, v: u64) -> io::Result<()> {
        self.room(2)?;
        let off = OFF_REGMAP + (self.regmap as usize) * 4;
        self.map.as_mut_slice()[off..off + 4].copy_from_slice(&(v as u32).to_le_bytes());
        self.map.as_mut_slice()[off + 4..off + 8]
            .copy_from_slice(&((v >> 32) as u32).to_le_bytes());
        self.regmap += 2;
        self.count += 2;
        self.write_header();
        Ok(())
    }

    /// The packet's current state field (ERT_CMD_STATE_*).
    pub fn state(&self) -> u32 {
        let mut w = [0u8; 4];
        w.copy_from_slice(&self.map.as_slice()[0..4]);
        u32::from_le_bytes(w) & 0xf
    }

    /// Snapshot of the packet's leading words (header + cu mask + start of
    /// the regmap), to detect the kernel health path overwriting the regmap
    /// with firmware health data (aie2_ctx_cmd_health_data memcpys into
    /// cmd->data, i.e. offset 4 onward).
    pub fn pkt_header_words(&self) -> [u32; 8] {
        let mut out = [0u32; 8];
        for (i, w) in out.iter_mut().enumerate() {
            let b = &self.map.as_slice()[i * 4..i * 4 + 4];
            *w = u32::from_le_bytes(b.try_into().unwrap());
        }
        out
    }

    /// Submit via AMDXDNA_EXEC_CMD. `arg_handles` are the BO handles the
    /// command touches (the driver pins them and records them on the job);
    /// the register map itself already carries their addresses. Returns the
    /// sequence number for [`syncobj_timeline_wait`].
    pub fn submit(
        &mut self,
        device: &Device,
        ctx: &HwContext<'_>,
        arg_handles: &[u32],
    ) -> io::Result<u64> {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        struct ExecCmdArg {
            ext: u64,
            ext_flags: u64,
            hwctx: u32,
            ty: u32, // AMDXDNA_CMD_SUBMIT_EXEC_BUF
            cmd_handles: u64, // single handle value (cmd_count == 1)
            args: u64,        // pointer to u32 arg-handle array
            cmd_count: u32,
            arg_count: u32,
            seq: u64,
        }
        let mut arg = ExecCmdArg {
            ext: 0,
            ext_flags: 0,
            hwctx: ctx.handle,
            ty: 0,
            cmd_handles: self.bo.handle() as u64,
            args: arg_handles.as_ptr() as u64,
            cmd_count: 1,
            arg_count: arg_handles.len() as u32,
            seq: 0,
        };
        let cmd = amdxdna_ioctl(AmdxdnaCmd::ExecCmd, std::mem::size_of::<ExecCmdArg>());
        // SAFETY: exact-size UAPI struct; args points at the handle slice for
        // the duration of the call.
        unsafe {
            raw_ioctl(
                device.raw_fd(),
                cmd,
                &mut arg as *mut ExecCmdArg as *mut u8,
            )?;
        }
        Ok(arg.seq)
    }
}

/// Wait for `point` on the context's syncobj timeline (the seq returned by
/// `StartNpuCmd::submit`). `timeout_ns` is relative; returns Err(ETIME-like)
/// on timeout. After the wait, read the packet's state() to judge success —
/// the driver only signals COMPLETED there.
pub fn syncobj_timeline_wait(
    device: &Device,
    syncobj: u32,
    point: u64,
    timeout_ns: u64,
) -> io::Result<()> {
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    struct TimelineWaitArg {
        handles: u64,
        points: u64,
        timeout_nsec: i64, // absolute CLOCK_MONOTONIC
        count_handles: u32,
        flags: u32,
        first_signaled: u32,
        pad: u32,
    }
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: plain timespec out-param.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    let now = ts.tv_sec * 1_000_000_000 + ts.tv_nsec;
    let handles = [syncobj];
    let points = [point];
    let mut arg = TimelineWaitArg {
        handles: handles.as_ptr() as u64,
        points: points.as_ptr() as u64,
        timeout_nsec: now + timeout_ns as i64,
        count_handles: 1,
        flags: 0, // wait for all
        first_signaled: 0,
        pad: 0,
    };
    // SAFETY: exact-size DRM struct; pointers valid for the call.
    unsafe {
        raw_ioctl(
            device.raw_fd(),
            DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT,
            &mut arg as *mut TimelineWaitArg as *mut u8,
        )
    }
}
