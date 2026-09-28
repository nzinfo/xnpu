//! xdump2: LD_PRELOAD ioctl tracer for amdxdna (M1 xdump, rebuilt for P20b).
//!
//! Intercepts ioctl + mmap, tracks CREATE_BO -> GET_BO_INFO -> mmap so BO
//! handles resolve to host VAs, and dumps every EXEC_CMD: the UAPI struct,
//! the arg-handle list, and the CMD-BO packet bytes (header + regmap).
//! Full hex for the first few execs, then a stable hash so per-exec packet
//! variation still shows without flooding the log.
//!
//! Enable with XDUMP_OUT=<path> (unset = pure passthrough).

use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::sync::Mutex;
use std::sync::OnceLock;

fn log_file() -> &'static Mutex<Option<std::fs::File>> {
    static LOG: OnceLock<Mutex<Option<std::fs::File>>> = OnceLock::new();
    LOG.get_or_init(|| {
        let f = std::env::var("XDUMP_OUT")
            .ok()
            .and_then(|p| OpenOptions::new().create(true).append(true).open(p).ok());
        Mutex::new(f)
    })
}

fn line(s: &str) {
    let mut g = log_file().lock().unwrap();
    if let Some(f) = g.as_mut() {
        let _ = writeln!(f, "{s}");
    }
}

fn req(nr: u32, size: usize) -> libc::c_ulong {
    ((3u64 << 30) | ((size as u64) << 16) | (0x64 << 8) | (0x40 + nr as u64) as u64) as libc::c_ulong
}

struct Bo {
    map_offset: u64,
    vaddr: u64,
    xdna_addr: u64,
    size: u64,
}

struct State {
    handle2off: HashMap<u32, u64>, // handle -> map_offset (from GET_BO_INFO)
    off2va: HashMap<u64, (u64, u64)>, // map_offset -> (va, len) from mmap
    hdl2size: HashMap<u32, (u32, u64)>, // handle -> (bo_type, size) from CREATE_BO
    hdl2uva: HashMap<u32, (u64, u64)>, // userptr BO: handle -> (user VA, size)
    xdna2hdl: HashMap<u64, u32>, // DEV BO xdna addr -> handle (from GET_BO_INFO)
    hdl2xdna: HashMap<u32, u64>, // handle -> xdna addr (from GET_BO_INFO)
    exec_count: u64,
    subdumps: u64,
}

fn state() -> &'static Mutex<State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(|| {
        Mutex::new(State {
            handle2off: HashMap::new(),
            off2va: HashMap::new(),
            hdl2size: HashMap::new(),
            hdl2uva: HashMap::new(),
            xdna2hdl: HashMap::new(),
            hdl2xdna: HashMap::new(),
            exec_count: 0,
            subdumps: 0,
        })
    })
}

fn tstamp() -> u128 {
    static T0: OnceLock<std::time::Instant> = OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_micros() as u128
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn resolve(handle: u32, st: &State) -> Option<(u64, u64)> {
    let off = st.handle2off.get(&handle)?;
    st.off2va.get(off).copied()
}

unsafe fn read_u64(p: *const u8) -> u64 {
    let mut b = [0u8; 8];
    std::ptr::copy_nonoverlapping(p, b.as_mut_ptr(), 8);
    u64::from_le_bytes(b)
}

unsafe fn read_u32(p: *const u8) -> u32 {
    let mut b = [0u8; 4];
    std::ptr::copy_nonoverlapping(p, b.as_mut_ptr(), 4);
    u32::from_le_bytes(b)
}

#[no_mangle]
pub extern "C" fn mmap(
    addr: *mut c_void,
    length: usize,
    prot: libc::c_int,
    flags: libc::c_int,
    fd: libc::c_int,
    offset: libc::c_long,
) -> *mut c_void {
    let real: extern "C" fn(*mut c_void, usize, libc::c_int, libc::c_int, libc::c_int, libc::c_long) -> *mut c_void =
        unsafe { std::mem::transmute(dlsym_next(b"mmap\0")) };
    let r = real(addr, length, prot, flags, fd, offset);
    if log_file().lock().unwrap().is_some() && !r.is_null() && fd >= 0 {
        let mut st = state().lock().unwrap();
        if let Some((va, len)) = st.off2va.get(&(offset as u64)) {
            let _ = (va, len);
        }
        st.off2va.insert(offset as u64, (r as u64, length as u64));
    }
    r
}

#[no_mangle]
pub extern "C" fn ioctl(fd: libc::c_int, request: libc::c_ulong, arg: *mut c_void) -> libc::c_int {
    let real: extern "C" fn(libc::c_int, libc::c_ulong, *mut c_void) -> libc::c_int =
        unsafe { std::mem::transmute(dlsym_next(b"ioctl\0")) };
    let enabled = log_file().lock().unwrap().is_some();

    let r = real(fd, request, arg);
    if !enabled {
        return r;
    }

    unsafe {
        let p = arg as *const u8;
        if arg.is_null() {
            return r;
        }
        // CREATE_BO {flags,vaddr,size,type,handle} 32B — handle out @28.
        if request == req(3, 32) && r == 0 {
            let h = read_u32(p.add(28));
            let size = read_u64(p.add(16));
            let ty = read_u32(p.add(24));
            let uva = read_u64(p.add(8));
            let st = &mut *state().lock().unwrap();
            st.hdl2size.insert(h, (ty, size));
            if uva != 0 {
                st.hdl2uva.insert(h, (uva, size));
            }
            line(&format!(
                "create_bo hdl={h} type={ty} size={size} uva={:x}",
                uva
            ));
        }
        // GET_BO_INFO {ext,ext_flags,handle,pad,map_offset,vaddr,xdna} 48B.
        else if request == req(4, 48) && r == 0 {
            let h = read_u32(p.add(16));
            let st = &mut *state().lock().unwrap();
            let off = read_u64(p.add(24));
            let va = read_u64(p.add(32));
            let xd = read_u64(p.add(40));
            st.handle2off.insert(h, off);
            if xd != u64::MAX {
                st.xdna2hdl.insert(xd, h);
                st.hdl2xdna.insert(h, xd);
            }
            line(&format!("bo_info hdl={h} map_off=0x{off:x} va=0x{va:x} xdna=0x{xd:x}"));
        }
        // SYNC_BO {handle,direction,offset,size} 24B.
        else if request == req(5, 24) {
            let h = read_u32(p);
            let dir = read_u32(p.add(4));
            let off = read_u64(p.add(8));
            let sz = read_u64(p.add(16));
            line(&format!("sync_bo hdl={h} dir={dir} off={off} size={sz}"));
        }
        // EXEC_CMD {ext,ext_flags,hwctx,ty,cmd_handles,args,cmd_count,
        //           arg_count,seq} 56B.
        else if request == req(6, 56) {
            let hwctx = read_u32(p.add(16));
            let ty = read_u32(p.add(20));
            let cmd_handles = read_u64(p.add(24));
            let args = read_u64(p.add(32));
            let cmd_count = read_u32(p.add(40));
            let arg_count = read_u32(p.add(44));
            let seq = read_u64(p.add(48));
            let mut arglist = Vec::new();
            if args != 0 && arg_count > 0 && arg_count < 1024 {
                for i in 0..arg_count {
                    arglist.push(read_u32((args as *const u8).add(4 * i as usize)));
                }
            }
            let st = &mut *state().lock().unwrap();
            st.exec_count += 1;
            let n = st.exec_count;
            // cmd BO: cmd_count==1 -> cmd_handles is the inline handle value.
            let mut pkt = String::new();
            if cmd_count == 1 {
                if let Some((va, len)) = resolve(cmd_handles as u32, st) {
                    let blen = len.min(256) as usize;
                    let bytes = std::slice::from_raw_parts(va as *const u8, blen);
                    let hdr = read_u32(bytes.as_ptr());
                    let state_f = hdr & 0xf;
                    let count = (hdr >> 12) & 0x7ff;
                    let opcode_f = (hdr >> 23) & 0x1f;
                    if n <= 5 {
                        let hex: String = bytes[..96]
                            .iter()
                            .map(|b| format!("{b:02x}"))
                            .collect::<Vec<_>>()
                            .join("");
                        pkt = format!(
                            " opcode={opcode_f} pkt96={hex} hash={:016x} hdr_state={state_f} hdr_count={count}",
                            fnv(bytes)
                        );
                    } else {
                        pkt = format!(
                            " opcode={opcode_f} hash={:016x} hdr_state={state_f} hdr_count={count}",
                            fnv(bytes)
                        );
                    }
                    // ERT_CMD_CHAIN: data[i] are sub-command BO handles —
                    // resolve and dump every real start-kernel packet.
                    if opcode_f == 19 {
                        let ccount = read_u32(bytes.as_ptr().add(4));
                        pkt.push_str(&format!(" ccount={ccount}"));
                        for i in 0..ccount.min(256) {
                            let boh = read_u32(bytes.as_ptr().add(28 + 8 * i as usize));
                            if let Some((sva, slen)) = resolve(boh, st) {
                                // XDUMP_SUBDUMP=<n>: write the first <n> sub BOs
                                // of every chain to $XDUMP_SUBDUMP_DIR/sub_*.bin.
                                if let Ok(cap) = std::env::var("XDUMP_SUBDUMP") {
                                    if let Ok(dir) = std::env::var("XDUMP_SUBDUMP_DIR") {
                                        let capn: u64 = cap.parse().unwrap_or(0);
                                        let mut cnt = st.subdumps;
                                        if cnt < capn {
                                            cnt += 1;
                                            st.subdumps = cnt;
                                            let _ = std::fs::write(
                                                format!("{dir}/e{n}_s{i}_h{boh}.bin"),
                                                std::slice::from_raw_parts(
                                                    sva as *const u8,
                                                    slen.min(1 << 20) as usize,
                                                ),
                                            );
                                        }
                                    }
                                }
                                // Also dump the TARGET ctrl-code BO the wrapper
                                // points at (w2 xdna addr), when enabled.
                                if let Ok(dir) = std::env::var("XDUMP_SUBDUMP_DIR") {
                                    let sp = sva as *const u8;
                                    let w2f = read_u32(sp.add(8)) as u64;
                                    let w4f = read_u32(sp.add(16)) as u64;
                                    let hopt = st.xdna2hdl.get(&(w2f & !0xfff)).copied();
                                    let vaopt = hopt.and_then(|h| {
                                        st.hdl2uva
                                            .get(&h)
                                            .copied()
                                            .or_else(|| resolve(h, st))
                                    }).or_else(|| {
                                        // DEV BO carved from the 64MB heap: host
                                        // VA = heap_map_va + (xdna - heap_xdna).
                                        let heap = st.hdl2size.iter().find(|(_, (t, _))| *t == 2)?;
                                        let (hh, (_, _hs)) = (heap.0, heap.1);
                                        let base = *st.hdl2xdna.get(hh)?;
                                        let (hva, hlen) = resolve(*hh, st)?;
                                        let off = w2f.checked_sub(base)?;
                                        if off + 4 <= hlen {
                                            Some((hva + off, hlen - off))
                                        } else {
                                            None
                                        }
                                    });
                                    if let Some((cva, clen)) = vaopt {
                                        if w4f > 0 && w4f < (1 << 20) {
                                            let _ = std::fs::write(
                                                format!("{dir}/ctrl_e{n}_s{i}_{w4f}B.bin"),
                                                std::slice::from_raw_parts(
                                                    cva as *const u8,
                                                    w4f.min(clen) as usize,
                                                ),
                                            );
                                        }
                                    }
                                }
                                let sb = std::slice::from_raw_parts(
                                    sva as *const u8,
                                    slen.min(192) as usize,
                                );
                                let shdr = read_u32(sb.as_ptr());
                                // w2/w3 = 64-bit xdna addr of the ctrl-code
                                // DEV BO; w4/w5 = candidate instr word count.
                                let w2 = read_u32(sb.as_ptr().add(8));
                                let w4 = read_u32(sb.as_ptr().add(16));
                                let ctrl_sz = st
                                    .xdna2hdl
                                    .get(&((w2 as u64) & !0xfff))
                                    .and_then(|h| st.hdl2size.get(h))
                                    .map(|(_, s)| *s)
                                    .unwrap_or(0);
                                let hex = if n <= 2 && i < 8 {
                                    let hx: String = sb[..128]
                                        .iter()
                                        .map(|b| format!("{b:02x}"))
                                        .collect::<Vec<_>>()
                                        .join("");
                                    format!(" hex128={hx}")
                                } else {
                                    String::new()
                                };
                                pkt.push_str(&format!(
                                    "\n  sub#{i} hdl={boh} opcode={} count={} \
                                     ctrl_addr=0x{w2:x} ctrl_sz={ctrl_sz} w4=0x{w4:x} \
                                     hash={:016x}{hex}",
                                    (shdr >> 23) & 0x1f,
                                    (shdr >> 12) & 0x7ff,
                                    fnv(sb)
                                ));
                            }
                        }
                    }
                } else {
                    pkt = " pkt=?unresolved".into();
                }
            }
            line(&format!(
                "exec#{n} t={} fd={fd} hwctx={hwctx} ty={ty} cmd_count={cmd_count} \
                 cmd_handles=0x{cmd_handles:x} args={arglist:?} arg_count={arg_count} \
                 seq={seq} ret={r}{pkt}",
                tstamp()
            ));
        }
    }
    r
}

fn dlsym_next(name: &[u8]) -> *mut c_void {
    unsafe { libc::dlsym(libc::RTLD_NEXT, name.as_ptr() as *const libc::c_char) }
}
