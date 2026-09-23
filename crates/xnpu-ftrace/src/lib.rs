//! xnpu-ftrace: LD_PRELOAD interceptor for the XRT C++ API (M5c, 方案 A).
//!
//! FLM's closed-source kernel DLLs (lib{hunyuan,q4_npu_eXpress,gemm,dequant,
//! mha,lm_head}_npu.so) call libxrt_coreutil.so.2 through the PLT, so
//! LD_PRELOAD can interpose the mangled C++ symbols — the M1 xdump technique
//! one level up (API calls instead of raw DRM ioctls). Every interceptor is a
//! dlsym(RTLD_NEXT) passthrough that logs a timestamped line:
//!
//!   <t_ns> <tid> <event> key=value ...
//!
//! ABI note: all intercepted signatures pass args in integer registers
//! (pointers/scalars, no by-value structs, no variadics, no sret returns in
//! the set), so declaring every return as u64 and forwarding RAX verbatim is
//! exact — void functions leave RAX undefined and callers of void ignore it.
//!
//! Op identity comes from xrt::ext::kernel's ctor, whose third arg is the
//! kernel NAME as a const std::string& (libstdc++ layout read inline, SSO
//! included); xrt::run::run(kernel&) then joins run objects to names by
//! address (single-inheritance upcast keeps them equal).
//!
//! Enable with FTRACE_OUT=<path> (unset = pure passthrough, no logging).

use std::ffi::c_void;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::sync::Mutex;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// logging core
// ---------------------------------------------------------------------------

fn log_file() -> &'static Mutex<Option<std::fs::File>> {
    static LOG: OnceLock<Mutex<Option<std::fs::File>>> = OnceLock::new();
    LOG.get_or_init(|| {
        let f = std::env::var("FTRACE_OUT")
            .ok()
            .and_then(|p| OpenOptions::new().create(true).append(true).open(p).ok());
        Mutex::new(f)
    })
}

fn now_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &mut ts) };
    (ts.tv_sec as u64) * 1_000_000_000 + ts.tv_nsec as u64
}

fn tid() -> u32 {
    unsafe { libc::gettid() as u32 }
}

/// One log line per event; unbuffered (crash-safe tail), fine at op rate.
fn ftrace(line: &str) {
    let mut g = log_file().lock().unwrap();
    if let Some(f) = g.as_mut() {
        let _ = writeln!(f, "{} {} {}", now_ns(), tid(), line);
    }
}

/// Read a libstdc++ std::string at `p` (SSO-aware). Best-effort: any layout
/// surprise yields "?" rather than a crash.
fn cxx_string(p: *const c_void) -> String {
    if p.is_null() {
        return "?".into();
    }
    let p = p as usize;
    unsafe {
        let dataptr = *(p as *const usize);
        let len = *((p + 8) as *const usize);
        // SSO iff the data pointer IS the embedded buffer at +16; the
        // length field at +8 is live in BOTH modes (first version wrongly
        // read a local-buffer byte as the SSO length).
        let (src, n) = if dataptr == p + 16 {
            (p + 16, len)
        } else if dataptr != 0 && len < 8192 {
            (dataptr, len)
        } else {
            return "?".into();
        };
        let bytes = std::slice::from_raw_parts(src as *const u8, n);
        String::from_utf8_lossy(bytes).into_owned()
    }
}

/// Cached dlsym(RTLD_NEXT) per symbol — one relaxed load on the hot path
/// after the first call. Panics loudly if the symbol is missing (the
/// interceptor would otherwise silently break the app).
macro_rules! xrt_real {
    ($getter:ident, $sym:literal, $sig:ty) => {
        unsafe fn $getter() -> $sig {
            static P: OnceLock<usize> = OnceLock::new();
            let p = *P.get_or_init(|| unsafe {
                let c = concat!($sym, "\0");
                libc::dlsym(libc::RTLD_NEXT, c.as_ptr().cast()) as usize
            });
            assert!(p != 0, "xnpu-ftrace: {} not found via RTLD_NEXT", $sym);
            unsafe { std::mem::transmute::<usize, $sig>(p) }
        }
    };
}

// ---------------------------------------------------------------------------
// interceptors
// ---------------------------------------------------------------------------

xrt_real!(r_kern_new, "_ZN3xrt3ext6kernelC1ERKNS_10hw_contextERKNS_6moduleERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE",
          unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, *const c_void) -> u64);
xrt_real!(r_run_new, "_ZN3xrt3runC1ERKNS_6kernelE",
          unsafe extern "C" fn(*mut c_void, *const c_void) -> u64);
xrt_real!(r_run_del, "_ZN3xrt3runD1Ev",
          unsafe extern "C" fn(*mut c_void) -> u64);
xrt_real!(r_run_start, "_ZN3xrt3run5startEv",
          unsafe extern "C" fn(*mut c_void) -> u64);
xrt_real!(r_run_wait, "_ZNK3xrt3run4waitERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE",
          unsafe extern "C" fn(*const c_void, *const c_void) -> u64);
xrt_real!(r_run_arg_ptr, "_ZN3xrt3run16set_arg_at_indexEiPKvm",
          unsafe extern "C" fn(*mut c_void, i32, *const c_void, u64) -> u64);
xrt_real!(r_run_arg_bo, "_ZN3xrt3run16set_arg_at_indexEiRKNS_2boE",
          unsafe extern "C" fn(*mut c_void, i32, *const c_void) -> u64);
xrt_real!(r_rl_new, "_ZN3xrt7runlistC1ERKNS_10hw_contextE",
          unsafe extern "C" fn(*mut c_void, *const c_void) -> u64);
xrt_real!(r_rl_add, "_ZN3xrt7runlist3addEONS_3runE",
          unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64);
xrt_real!(r_rl_reset, "_ZN3xrt7runlist5resetEv",
          unsafe extern "C" fn(*mut c_void) -> u64);
xrt_real!(r_rl_exec, "_ZN3xrt7runlist7executeEv",
          unsafe extern "C" fn(*mut c_void) -> u64);
xrt_real!(r_rl_wait, "_ZNK3xrt7runlist4waitERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE",
          unsafe extern "C" fn(*const c_void, *const c_void) -> u64);
xrt_real!(r_bo_new, "_ZN3xrt3ext2boC1ERKNS_6deviceEm",
          unsafe extern "C" fn(*mut c_void, *const c_void, u64) -> u64);
xrt_real!(r_bo_map, "_ZN3xrt2bo3mapEv",
          unsafe extern "C" fn(*mut c_void) -> u64);
xrt_real!(r_bo_sync, "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm",
          unsafe extern "C" fn(*mut c_void, i32, u64, u64) -> u64);
xrt_real!(r_xclbin_new, "_ZN3xrt6xclbinC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE",
          unsafe extern "C" fn(*mut c_void, *const c_void) -> u64);
xrt_real!(r_hwctx_new, "_ZN3xrt10hw_contextC1ERKNS_6deviceERKNS_4uuidENS0_11access_modeE",
          unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, i32) -> u64);
xrt_real!(r_module_new, "_ZN3xrt6moduleC1ERKNS_3elfE",
          unsafe extern "C" fn(*mut c_void, *const c_void) -> u64);
xrt_real!(r_elf_new, "_ZN3xrt3elfC1EPKvm",
          unsafe extern "C" fn(*mut c_void, *const c_void, u64) -> u64);

#[unsafe(export_name = "_ZN3xrt3ext6kernelC1ERKNS_10hw_contextERKNS_6moduleERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE")]
pub extern "C" fn kern_new(
    this: *mut c_void,
    ctx: *const c_void,
    module: *const c_void,
    name: *const c_void,
) -> u64 {
    unsafe {
        let r = r_kern_new()(this, ctx, module, name);
        ftrace(&format!("kern_new this={:x} ctx={:x} name={}", this as usize, ctx as usize, cxx_string(name)));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt3runC1ERKNS_6kernelE")]
pub extern "C" fn run_new(this: *mut c_void, kern: *const c_void) -> u64 {
    unsafe {
        let r = r_run_new()(this, kern);
        ftrace(&format!("run_new this={:x} kern={:x}", this as usize, kern as usize));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt3runD1Ev")]
pub extern "C" fn run_del(this: *mut c_void) -> u64 {
    unsafe {
        let r = r_run_del()(this);
        ftrace(&format!("run_del this={:x}", this as usize));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt3run5startEv")]
pub extern "C" fn run_start(this: *mut c_void) -> u64 {
    unsafe {
        let t = now_ns();
        let r = r_run_start()(this);
        ftrace(&format!("run_start run={:x} t0={}", this as usize, t));
        r
    }
}

#[unsafe(export_name = "_ZNK3xrt3run4waitERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE")]
pub extern "C" fn run_wait(this: *const c_void, dur: *const c_void) -> u64 {
    unsafe {
        let t = now_ns();
        let r = r_run_wait()(this, dur);
        ftrace(&format!("run_wait run={:x} t0={} ret={}", this as usize, t, r));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt3run16set_arg_at_indexEiPKvm")]
pub extern "C" fn run_arg_ptr(this: *mut c_void, idx: i32, value: *const c_void, bytes: u64) -> u64 {
    unsafe {
        let r = r_run_arg_ptr()(this, idx, value, bytes);
        // `value` points AT the argument payload; log up to the first 8 bytes
        // so scalars (sizes, instruction words) are visible, buffer args keep
        // their pointer as the fingerprint.
        let payload = if !value.is_null() && bytes <= 8 {
            let mut v = [0u8; 8];
            std::ptr::copy_nonoverlapping(value.cast(), v.as_mut_ptr(), bytes as usize);
            Some(u64::from_le_bytes(v))
        } else {
            None
        };
        ftrace(&format!(
            "run_arg_ptr run={:x} idx={} val={:x} bytes={} {}",
            this as usize,
            idx,
            value as usize,
            bytes,
            payload.map(|v| format!("data={v:x}")).unwrap_or_default(),
        ));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt3run16set_arg_at_indexEiRKNS_2boE")]
pub extern "C" fn run_arg_bo(this: *mut c_void, idx: i32, bo: *const c_void) -> u64 {
    unsafe {
        let r = r_run_arg_bo()(this, idx, bo);
        ftrace(&format!("run_arg_bo run={:x} idx={} bo={:x}", this as usize, idx, bo as usize));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt7runlistC1ERKNS_10hw_contextE")]
pub extern "C" fn rl_new(this: *mut c_void, ctx: *const c_void) -> u64 {
    unsafe {
        let r = r_rl_new()(this, ctx);
        ftrace(&format!("rl_new this={:x}", this as usize));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt7runlist3addEONS_3runE")]
pub extern "C" fn rl_add(this: *mut c_void, run: *mut c_void) -> u64 {
    unsafe {
        let r = r_rl_add()(this, run);
        ftrace(&format!("rl_add list={:x} run={:x}", this as usize, run as usize));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt7runlist5resetEv")]
pub extern "C" fn rl_reset(this: *mut c_void) -> u64 {
    unsafe {
        let r = r_rl_reset()(this);
        ftrace(&format!("rl_reset list={:x}", this as usize));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt7runlist7executeEv")]
pub extern "C" fn rl_exec(this: *mut c_void) -> u64 {
    unsafe {
        let t = now_ns();
        let r = r_rl_exec()(this);
        ftrace(&format!("rl_exec list={:x} t0={}", this as usize, t));
        r
    }
}

#[unsafe(export_name = "_ZNK3xrt7runlist4waitERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE")]
pub extern "C" fn rl_wait(this: *const c_void, dur: *const c_void) -> u64 {
    unsafe {
        let t = now_ns();
        let r = r_rl_wait()(this, dur);
        ftrace(&format!("rl_wait list={:x} t0={} ret={}", this as usize, t, r));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt3ext2boC1ERKNS_6deviceEm")]
pub extern "C" fn bo_new(this: *mut c_void, dev: *const c_void, size: u64) -> u64 {
    unsafe {
        let r = r_bo_new()(this, dev, size);
        ftrace(&format!("bo_new this={:x} size={}", this as usize, size));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt2bo3mapEv")]
pub extern "C" fn bo_map(this: *mut c_void) -> u64 {
    unsafe {
        let r = r_bo_map()(this);
        ftrace(&format!("bo_map bo={:x} va={:x}", this as usize, r));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm")]
pub extern "C" fn bo_sync(this: *mut c_void, dir: i32, size: u64, offset: u64) -> u64 {
    unsafe {
        let t = now_ns();
        let r = r_bo_sync()(this, dir, size, offset);
        ftrace(&format!(
            "bo_sync bo={:x} dir={} size={} off={} t0={} ret={}",
            this as usize,
            dir,
            size,
            offset,
            t,
            r
        ));
        r
    }
}

// All FLM kernels are named "MLIR_AIE" (the mlir_aie default), so op/graph
// identity comes from these joins: xclbin ctor logs the FILE path, hw_context
// ctor logs its uuid, kern_new logs its ctx -> kernel -> context -> xclbin
// separates the decode graph from the fused-prefill graph; within a graph,
// runs are grouped by argument fingerprints in the decoder.

#[unsafe(export_name = "_ZN3xrt6xclbinC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE")]
pub extern "C" fn xclbin_new(this: *mut c_void, path: *const c_void) -> u64 {
    unsafe {
        let r = r_xclbin_new()(this, path);
        ftrace(&format!("xclbin_new this={:x} path={}", this as usize, cxx_string(path)));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt10hw_contextC1ERKNS_6deviceERKNS_4uuidENS0_11access_modeE")]
pub extern "C" fn hwctx_new(
    this: *mut c_void,
    dev: *const c_void,
    uuid: *const c_void,
    mode: i32,
) -> u64 {
    unsafe {
        let r = r_hwctx_new()(this, dev, uuid, mode);
        let u = if uuid.is_null() {
            [0u8; 16]
        } else {
            let mut b = [0u8; 16];
            std::ptr::copy_nonoverlapping(uuid.cast(), b.as_mut_ptr(), 16);
            b
        };
        ftrace(&format!(
            "hwctx_new this={:x} uuid={:02x}{:02x}{:02x}{:02x} mode={}",
            this as usize, u[0], u[1], u[2], u[3], mode
        ));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt6moduleC1ERKNS_3elfE")]
pub extern "C" fn module_new(this: *mut c_void, elf: *const c_void) -> u64 {
    unsafe {
        let r = r_module_new()(this, elf);
        ftrace(&format!("module_new this={:x} elf={:x}", this as usize, elf as usize));
        r
    }
}

#[unsafe(export_name = "_ZN3xrt3elfC1EPKvm")]
pub extern "C" fn elf_new(this: *mut c_void, data: *const c_void, size: u64) -> u64 {
    unsafe {
        let r = r_elf_new()(this, data, size);
        ftrace(&format!("elf_new this={:x} data={:x} size={}", this as usize, data as usize, size));
        r
    }
}
