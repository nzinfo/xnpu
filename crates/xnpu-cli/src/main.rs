//! xnpu-cli: probe / diagnose tools for the amdxdna driver via xnpu-hal.

use std::process::ExitCode;

use xnpu_perf::{trace_json, MachineModel, Mode, OpMeta, Recorder};

use xnpu_hal::{
    syncobj_timeline_wait, BoType, BufferObject, Device, HwContext, Mapping, StartNpuCmd,
    SyncDirection, ERT_CMD_STATE_COMPLETED,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("info") => cmd_info(),
        Some("ctx-probe") => {
            let max: usize = args
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(32);
            let cols: u32 = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);
            cmd_ctx_probe(max, cols)
        }
        Some("run-add") => {
            let prj = args.get(1).cloned().unwrap_or_else(|| {
                "/home/nzinfo/qwen/xnpu/build/add_1c_2ch_2048_2048t.mlir.prj".to_string()
            });
            let dtype = args.get(2).cloned().unwrap_or_else(|| "bf16".to_string());
            cmd_run_add(&prj, &dtype)
        }
        Some("run-gemm") => {
            // Trailing dtype token (bf16 default); everything numeric between
            // the project dir and it is M K N.
            let mut rest: Vec<String> = args.get(2..).map(|r| r.to_vec()).unwrap_or_default();
            let dtype = match rest.last().map(String::as_str) {
                Some("bf16") | Some("i8") => rest.pop().unwrap(),
                _ => "bf16".to_string(),
            };
            let prj = args.get(1).cloned().unwrap_or_else(|| {
                if dtype == "i8" {
                    "/home/nzinfo/qwen/xnpu/IRON/build/gemm_192x384x64_48x96x16_0_0_i8_i32.mlir.prj".to_string()
                } else {
                    "/home/nzinfo/qwen/xnpu/build/gemm_192x384x64_48x96x16_0_0.mlir.prj".to_string()
                }
            });
            let dims: Vec<usize> = rest.iter().filter_map(|s| s.parse().ok()).collect();
            let (m, k, n) = match dims.as_slice() {
                [m, k, n] => (*m, *k, *n),
                [] => (192, 384, 64),
                _ => {
                    eprintln!("run-gemm: expected M K N");
                    return ExitCode::FAILURE;
                }
            };
            cmd_run_gemm(&prj, m, k, n, &dtype)
        }
        Some("run-multi") => {
            let add_prj = args.get(1).cloned().unwrap_or_else(|| {
                "/home/nzinfo/qwen/xnpu/build/add_1c_2ch_2048_2048t.mlir.prj".to_string()
            });
            let gemm_prj = args.get(2).cloned().unwrap_or_else(|| {
                "/home/nzinfo/qwen/xnpu/build/gemm_192x384x64_48x96x16_0_0.mlir.prj".to_string()
            });
            let dims: Vec<usize> = args
                .get(3..)
                .map(|rest| rest.iter().filter_map(|s| s.parse().ok()).collect())
                .unwrap_or_default();
            let (m, k, n) = match dims.as_slice() {
                [m, k, n] => (*m, *k, *n),
                [] => (192, 384, 64),
                _ => {
                    eprintln!("run-multi: expected M K N");
                    return ExitCode::FAILURE;
                }
            };
            cmd_run_multi(&add_prj, &gemm_prj, m, k, n)
        }
        Some("run-pipe") => {
            let prj = args.get(1).cloned().unwrap_or_else(|| {
                "/home/nzinfo/qwen/xnpu/build/gemm_192x384x64_48x96x16_0_0.mlir.prj".to_string()
            });
            let nums: Vec<usize> = args
                .get(2..)
                .map(|rest| rest.iter().filter_map(|s| s.parse().ok()).collect())
                .unwrap_or_default();
            let (m, k, n, iters) = match nums.as_slice() {
                [m, k, n, iters] => (*m, *k, *n, *iters),
                [m, k, n] => (*m, *k, *n, 32),
                [] => (192, 384, 64, 32),
                _ => {
                    eprintln!("run-pipe: expected [M K N] [iters]");
                    return ExitCode::FAILURE;
                }
            };
            cmd_run_pipe(&prj, m, k, n, iters)
        }
        Some("run-chain") => {
            let add_prj = args.get(1).cloned().unwrap_or_else(|| {
                "/home/nzinfo/qwen/xnpu/build/add_1c_2ch_2048_2048t.mlir.prj".to_string()
            });
            let gemm_prj = args.get(2).cloned().unwrap_or_else(|| {
                "/home/nzinfo/qwen/xnpu/build/gemm_192x384x64_48x96x16_0_0.mlir.prj".to_string()
            });
            let nums: Vec<usize> = args
                .get(3..)
                .map(|rest| rest.iter().filter_map(|s| s.parse().ok()).collect())
                .unwrap_or_default();
            let (m, k, n, reps) = match nums.as_slice() {
                [m, k, n, reps] => (*m, *k, *n, *reps),
                [m, k, n] => (*m, *k, *n, 8),
                [] => (192, 384, 64, 8),
                _ => {
                    eprintln!("run-chain: expected [M K N] [reps]");
                    return ExitCode::FAILURE;
                }
            };
            cmd_run_chain(&add_prj, &gemm_prj, m, k, n, reps)
        }
        Some("run-q8") => {
            let gemm_prj = args.get(1).cloned().unwrap_or_else(|| {
                "/home/nzinfo/qwen/xnpu/IRON/build/gemm_192x384x64_48x96x16_0_0_i8_i32.mlir.prj"
                    .to_string()
            });
            let rescale_prj = args.get(2).cloned().unwrap_or_else(|| {
                "/home/nzinfo/qwen/xnpu/build/rescale_192x64_32t.mlir.prj".to_string()
            });
            let nums: Vec<usize> = args
                .get(3..)
                .map(|rest| rest.iter().filter_map(|s| s.parse().ok()).collect())
                .unwrap_or_default();
            let (m, k, n, tile_m, reps) = match nums.as_slice() {
                [m, k, n, tile_m, reps] => (*m, *k, *n, *tile_m, *reps),
                [m, k, n, tile_m] => (*m, *k, *n, *tile_m, 8),
                [] => (192, 384, 64, 32, 8),
                _ => {
                    eprintln!("run-q8: expected [M K N tile_m] [reps]");
                    return ExitCode::FAILURE;
                }
            };
            cmd_run_q8(&gemm_prj, &rescale_prj, m, k, n, tile_m, reps)
        }
        Some("run-w4layer") => {
            let w4dir = args.get(1).cloned().unwrap_or_else(|| {
                "/home/nzinfo/qwen/xnpu/build/w4".to_string()
            });
            let nums: Vec<usize> = args
                .get(2..)
                .map(|rest| rest.iter().filter_map(|s| s.parse().ok()).collect())
                .unwrap_or_default();
            let (layers, iters) = match nums.as_slice() {
                [layers, iters] => (*layers, *iters),
                [layers] => (*layers, 3),
                [] => (42, 3),
                _ => {
                    eprintln!("run-w4layer: expected [layers] [iters]");
                    return ExitCode::FAILURE;
                }
            };
            cmd_run_w4layer(&w4dir, layers, iters)
        }
        Some("run-w4ulayer") => {
            let w4dir = args.get(1).cloned().unwrap_or_else(|| {
                "/home/nzinfo/qwen/xnpu/build/w4u".to_string()
            });
            let nums: Vec<usize> = args
                .get(2..)
                .map(|rest| rest.iter().filter_map(|s| s.parse().ok()).collect())
                .unwrap_or_default();
            let (layers, iters) = match nums.as_slice() {
                [layers, iters] => (*layers, *iters),
                [layers] => (*layers, 3),
                [] => (42, 3),
                _ => {
                    eprintln!("run-w4ulayer: expected [layers] [iters]");
                    return ExitCode::FAILURE;
                }
            };
            cmd_run_w4ulayer(&w4dir, layers, iters)
        }
        Some("run-fkprobe") => {
            let prj = args.get(1).cloned().unwrap_or_else(|| {
                "/home/nzinfo/qwen/xnpu/build/flowkv_decode_16h_2kv_128d_1024s_32cs_2col.mlir.prj".to_string()
            });
            let iters = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
            cmd_run_fkprobe(&prj, iters)
        }
        Some("run-decode") => {
            // Positional [dec-dir] [w4-dir] [iters] plus flag tokens anywhere:
            // "hy" selects the hy-mt2 arch profile (M5c), "cpu" keeps the
            // Rust scalar attention (A/B path; default is NPU attention).
            let is_hy = args.iter().any(|s| matches!(s.as_str(), "hy" | "hy-mt2"));
            let arch: &DecArch = if is_hy { &DEC_HY } else { &DEC_MINICPM };
            let pos_args: Vec<&String> = args[1..]
                .iter()
                .filter(|s| !matches!(s.as_str(), "cpu" | "hy" | "hy-mt2"))
                .collect();
            let decdir = pos_args
                .first()
                .map(|s| s.to_string())
                .unwrap_or_else(|| arch.decdir.to_string());
            let w4dir = pos_args
                .get(1)
                .map(|s| s.to_string())
                .unwrap_or_else(|| arch.w4dir.to_string());
            let iters = pos_args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(5);
            let npu_attn = !args.iter().any(|s| s.as_str() == "cpu");
            cmd_run_decode(arch, &decdir, &w4dir, iters, npu_attn)
        }
        Some("run-w4gemv") => {
            let prj = args.get(1).cloned().unwrap_or_else(|| {
                "/home/nzinfo/qwen/xnpu/build/fused_dequant_gemv_2048x2048_1tsi_512tso_4col_g32.mlir.prj"
                    .to_string()
            });
            let nums: Vec<usize> = args
                .get(2..)
                .map(|rest| rest.iter().filter_map(|s| s.parse().ok()).collect())
                .unwrap_or_default();
            let (m, k, group, tsi, iters) = match nums.as_slice() {
                [m, k, group, tsi, iters] => (*m, *k, *group, *tsi, *iters),
                [m, k, group, tsi] => (*m, *k, *group, *tsi, 32),
                [m, k, group] => (*m, *k, *group, 1, 32),
                [m, k] => (*m, *k, 32, 1, 32),
                [] => (2048, 2048, 32, 1, 32),
                _ => {
                    eprintln!("run-w4gemv: expected [M K] [group] [tsi] [iters]");
                    return ExitCode::FAILURE;
                }
            };
            cmd_run_w4gemv(&prj, m, k, group, tsi, iters)
        }
        _ => {
            eprintln!(
                "usage: xnpu-cli <info | ctx-probe [max] [cols] | run-add [prj-dir] [bf16|i8] | run-gemm [prj-dir] [M K N] [bf16|i8] | run-multi [add-prj] [gemm-prj] [M K N] | run-pipe [prj-dir] [M K N] [iters] | run-chain [add-prj] [gemm-prj] [M K N] [reps] | run-q8 [gemm-prj] [rescale-prj] [M K N tile_m] [reps] | run-w4gemv [prj-dir] [M K] [group] [tsi] [iters] | run-w4layer [w4-dir] [layers] [iters] | run-w4ulayer [w4u-dir] [layers] [iters] | run-decode [dec-dir] [w4u-dir] [iters] [hy] [cpu]>"
            );
            ExitCode::FAILURE
        }
    }
}

/// f32 -> bf16 bits, round-to-nearest-even.
fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    let bias = 0x7fff + ((b >> 16) & 1);
    ((b.wrapping_add(bias)) >> 16) as u16
}

/// Element type of the add fixture under test.
#[derive(Clone, Copy, PartialEq)]
enum DType {
    Bf16,
    I8,
}

impl DType {
    fn parse(s: &str) -> Option<DType> {
        match s {
            "bf16" => Some(DType::Bf16),
            "i8" => Some(DType::I8),
            _ => None,
        }
    }

    fn itemsize(self) -> usize {
        match self {
            DType::Bf16 => 2,
            DType::I8 => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            DType::Bf16 => "bf16",
            DType::I8 => "int8",
        }
    }

    /// Pack one element (the fixture values are small integers, exactly
    /// representable in both types) at `i`-th position of a tensor buffer.
    fn pack_at(self, slice: &mut [u8], i: usize, v: f32) {
        match self {
            DType::Bf16 => {
                let b = f32_to_bf16(v).to_le_bytes();
                slice[i * 2..i * 2 + 2].copy_from_slice(&b);
            }
            DType::I8 => slice[i] = v as i8 as u8,
        }
    }

    /// Decode the i-th element of an output buffer as i64 for comparison,
    /// alongside the expected a+b value (same small-integer values, so the
    /// int8 add cannot wrap).
    fn unpack(self, buf: &[u8], i: usize, a: f32, b: f32) -> (i64, i64) {
        match self {
            DType::Bf16 => {
                let g = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
                (g as i64, f32_to_bf16(a + b) as i64)
            }
            DType::I8 => (
                buf[i] as i8 as i64,
                ((a as i32 + b as i32) as i8) as i64,
            ),
        }
    }
}

/// Read a fixture's PDI and ctrl-code, plus the partition width recorded in
/// main_aie_partition.json. IRON reserves the full array ("column_width": 8)
/// regardless of how many columns the design actually uses, so this just
/// formalizes the M1 constant.
fn load_fixture(prj: &str) -> Option<(Vec<u8>, Vec<u8>, u32)> {
    let pdi_path = format!("{prj}/main.pdi");
    // The ctrl-code .bin sits next to the .mlir.prj dir (both in the IRON
    // build dir), named after the shared stem.
    let stem = prj.trim_end_matches(".mlir.prj");
    let instr_path = match stem.rsplit_once('/') {
        Some((dir, name)) => format!("{dir}/{name}.bin"),
        None => format!("{stem}.bin"),
    };
    let pdi = std::fs::read(&pdi_path).ok()?;
    let instr = std::fs::read(&instr_path).ok()?;
    let cols = std::fs::read_to_string(format!("{prj}/main_aie_partition.json"))
        .ok()
        .and_then(|s| {
            let key = "\"column_width\"";
            let rest = &s[s.find(key)? + key.len()..];
            let rest = &rest[rest.find(':')? + 1..];
            rest.trim()
                .split(|c: char| !c.is_ascii_digit())
                .next()?
                .parse()
                .ok()
        })
        .unwrap_or(8);
    Some((pdi, instr, cols))
}

/// End-to-end first operator: load the IRON add_1c_2ch fixture (1 core, 2048
/// elements per buffer, bf16 or int8) over raw DRM and verify the AIE output.
///
/// Register-map ABI (main_kernels.json / mlir_aie DPU pseudo-kernel):
/// opcode u64 = 3, instr u64 (ctrl-code VA), ninstr u32, then bo0.. u64 VAs.
fn cmd_run_add(prj: &str, dtype: &str) -> ExitCode {
    let dt = match DType::parse(dtype) {
        Some(d) => d,
        None => {
            eprintln!("unknown dtype '{dtype}' (expected bf16 or i8)");
            return ExitCode::FAILURE;
        }
    };
    let (pdi, instr, cols) = match load_fixture(prj) {
        Some(f) => f,
        None => {
            eprintln!("load fixture {prj} failed");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "fixture: pdi {} B, ctrl-code {} B, partition {} cols",
        pdi.len(),
        instr.len(),
        cols
    );

    let dev = match Device::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open amdxdna device: {e}");
            return ExitCode::FAILURE;
        }
    };
    let md = match dev.aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("AIE metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let num_tiles = cols * md.core.row_count as u32;

    let mut ctx = match HwContext::create(&dev, num_tiles) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "hwctx: handle={} syncobj={} ({} cols / {} tiles)",
        ctx.handle, ctx.syncobj_handle, cols, num_tiles
    );

    if let Err(e) = ctx.configure_cu(&pdi, 0) {
        eprintln!("configure_cu (PDI load): {e}");
        return ExitCode::FAILURE;
    }
    println!("PDI loaded, CU configured");

    // Ctrl-code BO is a DEV BO carved from the device heap: the instruction
    // regmap field must carry its heap xdna_addr (see the tensor comment below
    // for the opposite, user-VA rule that applies to tensor BOs).
    let n: usize = 2048;
    let bytes = n * dt.itemsize();

    let ctrl_bo = match BufferObject::new(&dev, BoType::Dev, instr.len()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ctrl BO: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = dev.write_dev_bo(&ctrl_bo, &instr) {
        eprintln!("write ctrl BO: {e}");
        return ExitCode::FAILURE;
    }
    if let Err(e) = ctrl_bo.sync(SyncDirection::ToDevice, 0, ctrl_bo.size() as u64) {
        eprintln!("sync ctrl BO: {e}");
        return ExitCode::FAILURE;
    }
    let ctrl_addr = ctrl_bo.xdna_addr();
    println!(
        "ctrl BO: hdl={} xdna=0x{:x}",
        ctrl_bo.handle(),
        ctrl_addr
    );

    // in1[i] = (i % 7) - 3, in2[i] = (i / 7) % 5 -> sums stay small integers,
    // exact in both bf16 and int8.
    //
    // Tensors must be SHMEM BOs whose regmap entries carry the *user VA*: the
    // firmware dereferences them through SVM/PASID (it walks the host page
    // tables of the pinned arg BOs). Heap device addresses here are silently
    // ignored — the command completes but the AIE never writes, which is how
    // the first all-DEV attempt failed. Only the ctrl code (instruction
    // buffer) needs a heap xdna_addr.
    let tensor =
        |name: &str, f: &dyn Fn(usize) -> f32| -> Option<(BufferObject, Mapping)> {
            let bo = BufferObject::new(&dev, BoType::Shmem, bytes).ok()?;
            let mut map = bo.map_owned().ok()?;
            let slice = map.as_mut_slice();
            for i in 0..n {
                dt.pack_at(slice, i, f(i));
            }
            bo.sync(SyncDirection::ToDevice, 0, bo.size() as u64).ok()?;
            println!(
                "{} BO: hdl={} va=0x{:x}",
                name,
                bo.handle(),
                map.as_ptr() as u64
            );
            Some((bo, map))
        };
    let (in1_bo, in1_map) = match tensor("in1", &|i| (i % 7) as f32 - 3.0) {
        Some(t) => t,
        None => {
            eprintln!("in1 BO failed");
            return ExitCode::FAILURE;
        }
    };
    let (in2_bo, in2_map) = match tensor("in2", &|i| ((i / 7) % 5) as f32) {
        Some(t) => t,
        None => {
            eprintln!("in2 BO failed");
            return ExitCode::FAILURE;
        }
    };
    let (out_bo, out_map) = match tensor("out", &|_| 0.0) {
        Some(t) => t,
        None => {
            eprintln!("out BO failed");
            return ExitCode::FAILURE;
        }
    };
    let in1_addr = in1_map.as_ptr() as u64;
    let in2_addr = in2_map.as_ptr() as u64;
    let out_addr = out_map.as_ptr() as u64;

    // ERT_START_NPU packet: regmap = opcode, instr, ninstr, bo0..bo2. All
    // addresses are the DEV BOs' xdna_addr values.
    let mut pkt = match StartNpuCmd::new(&dev) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("cmd BO: {e}");
            return ExitCode::FAILURE;
        }
    };
    pkt.set_cu(0);
    let build = pkt
        .set_ctrl(ctrl_addr, instr.len() as u32)
        .and_then(|_| pkt.arg64(3)) // opcode: DPU txn start
        .and_then(|_| pkt.arg64(ctrl_addr))
        .and_then(|_| pkt.arg32(instr.len() as u32))
        .and_then(|_| pkt.arg64(in1_addr))
        .and_then(|_| pkt.arg64(in2_addr))
        .and_then(|_| pkt.arg64(out_addr));
    if let Err(e) = build {
        eprintln!("build packet: {e}");
        return ExitCode::FAILURE;
    }

    // The driver pins every arg BO for the job's lifetime; XRT passes the
    // ctrl BO in the arg list too.
    let arg_handles = [
        ctrl_bo.handle(),
        in1_bo.handle(),
        in2_bo.handle(),
        out_bo.handle(),
    ];
    let t0 = std::time::Instant::now();
    let seq = match pkt.submit(&dev, &ctx, &arg_handles) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("exec submit: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("submitted, seq={seq}");
    if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000) {
        eprintln!("wait seq {seq}: {e}");
        return ExitCode::FAILURE;
    }
    let state = pkt.state();
    println!("wait done in {:?}, packet state={}", t0.elapsed(), state);
    if state != ERT_CMD_STATE_COMPLETED {
        eprintln!("command did not complete (state {state})");
        return ExitCode::FAILURE;
    }

    // Drop stale cache lines before reading the AIE's output. Direction 0
    // (ToDevice) is deliberate: the ioctl clflushes regardless of direction,
    // while direction 1 additionally takes the fw debug-BO path, which fails
    // EINVAL without a debug BO registered.
    if let Err(e) = out_bo.sync(SyncDirection::ToDevice, 0, out_bo.size() as u64) {
        eprintln!("flush out: {e}");
        return ExitCode::FAILURE;
    }
    let s: Vec<u8> = out_map.as_slice().to_vec();
    let _ = std::fs::write("/tmp/out.bin", &s);
    let _ = std::fs::write("/tmp/in1.bin", in1_map.as_slice());
    let _ = std::fs::write("/tmp/in2.bin", in2_map.as_slice());
    let (mismatches, first_bad) = check_add(&s, dt, n);
    match first_bad {
        Some((i, got, want)) => println!(
            "first mismatch @[{i}]: got {} ({:#06x}) want {} ({:#06x})",
            got,
            got as u16,
            want,
            want as u16
        ),
        None => println!(
            "out[0..8] = {:?}",
            (0..8)
                .map(|i| dt.unpack(&s, i, (i % 7) as f32 - 3.0, ((i / 7) % 5) as f32).0)
                .collect::<Vec<_>>()
        ),
    }
    if mismatches == 0 {
        println!("VERIFY: PASS ({n} elements, {} exact)", dt.name());
        ExitCode::SUCCESS
    } else {
        eprintln!("VERIFY: FAIL ({mismatches}/{n} mismatches)");
        ExitCode::FAILURE
    }
}

fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// Verify an add-fixture output against the CLI's fixed inputs
/// (in1[i] = (i%7)-3, in2[i] = (i/7)%5 — small integers, exact in bf16 and
/// int8, no wraparound). Returns (mismatch count, first diff).
fn check_add(out: &[u8], dt: DType, n: usize) -> (usize, Option<(usize, i64, i64)>) {
    let mut mismatches = 0usize;
    let mut first: Option<(usize, i64, i64)> = None;
    for i in 0..n {
        let (got, want) = dt.unpack(out, i, (i % 7) as f32 - 3.0, ((i / 7) % 5) as f32);
        if got != want {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, got, want));
            }
        }
    }
    (mismatches, first)
}

/// Host reference for the small-integer GEMM fixtures: decode A/B to f32,
/// accumulate in f32, round once to bf16 — bit-exact for f32-accumulation
/// fixtures (bf16_f32_ONLY), a few ULP off for bf16-accumulation ones.
/// Returns (mismatch count, max ulp, first diff).
fn check_gemm(
    a: &[u8],
    b: &[u8],
    c: &[u8],
    m: usize,
    k: usize,
    n: usize,
) -> (usize, i64, Option<(usize, usize, u16, u16, i64)>) {
    let decode = |bytes: &[u8], elems: usize| -> Vec<f32> {
        (0..elems)
            .map(|e| bf16_to_f32(u16::from_le_bytes([bytes[e * 2], bytes[e * 2 + 1]])))
            .collect()
    };
    let a_f = decode(a, m * k);
    let b_f = decode(b, k * n);
    let ulp = |g: u16, w: u16| -> i64 {
        if (g >> 15) == (w >> 15) {
            (g as i16 as i64 - w as i16 as i64).abs()
        } else {
            1 << 24
        }
    };
    let mut mismatches = 0usize;
    let mut max_ulp: i64 = 0;
    let mut first: Option<(usize, usize, u16, u16, i64)> = None;
    for i in 0..m {
        for l in 0..n {
            let mut acc: f32 = 0.0;
            for j in 0..k {
                acc += a_f[i * k + j] * b_f[j * n + l];
            }
            let want = f32_to_bf16(acc);
            let got = u16::from_le_bytes([c[(i * n + l) * 2], c[(i * n + l) * 2 + 1]]);
            if got != want {
                let d = ulp(got, want);
                mismatches += 1;
                max_ulp = max_ulp.max(d);
                if first.is_none() {
                    first = Some((i, l, got, want, d));
                }
            }
        }
    }
    (mismatches, max_ulp, first)
}

/// M2: first GEMM over raw DRM — the IRON gemm fixture computes
/// C[M,N] = A[M,K] @ B[K,N], all row-major, either bf16 inputs with f32
/// accumulation (bf16_f32_ONLY, per gemm/test.py) or int8 inputs with an
/// i32 accumulator (i8_i32_ONLY — the q8 route-A fixture).
///
/// Input values are small integers: in the bf16 case every f32 partial sum
/// is an exact integer well under 2^24 regardless of summation order and the
/// final conversion is plain RNE; in the int8 case the i32 accumulator is
/// exact integer math outright. Both references are host-reproducible
/// bit-exactly.
fn cmd_run_gemm(prj: &str, m: usize, k: usize, n: usize, dtype: &str) -> ExitCode {
    let is_i8 = match dtype {
        "bf16" => false,
        "i8" => true,
        _ => {
            eprintln!("unknown gemm dtype '{dtype}' (expected bf16 or i8)");
            return ExitCode::FAILURE;
        }
    };
    let (pdi, instr, cols) = match load_fixture(prj) {
        Some(f) => f,
        None => {
            eprintln!("load fixture {prj} failed");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "fixture: pdi {} B, ctrl-code {} B, partition {} cols",
        pdi.len(),
        instr.len(),
        cols
    );
    println!(
        "problem: C[{m}x{n}] = A[{m}x{k}] @ B[{k}x{n}], {}",
        if is_i8 { "i8/i32-acc" } else { "bf16/f32-acc" }
    );

    let dev = match Device::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open amdxdna device: {e}");
            return ExitCode::FAILURE;
        }
    };
    let md = match dev.aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("AIE metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let num_tiles = cols * md.core.row_count as u32;
    let mut ctx = match HwContext::create(&dev, num_tiles) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "hwctx: handle={} syncobj={} ({} cols / {} tiles)",
        ctx.handle, ctx.syncobj_handle, cols, num_tiles
    );
    if let Err(e) = ctx.configure_cu(&pdi, 0) {
        eprintln!("configure_cu (PDI load): {e}");
        return ExitCode::FAILURE;
    }
    println!("PDI loaded, CU configured");

    // Ctrl-code DEV BO (instruction field takes the heap xdna_addr).
    let ctrl_bo = match BufferObject::new(&dev, BoType::Dev, instr.len()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ctrl BO: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = dev.write_dev_bo(&ctrl_bo, &instr) {
        eprintln!("write ctrl BO: {e}");
        return ExitCode::FAILURE;
    }
    if let Err(e) = ctrl_bo.sync(SyncDirection::ToDevice, 0, ctrl_bo.size() as u64) {
        eprintln!("sync ctrl BO: {e}");
        return ExitCode::FAILURE;
    }
    let ctrl_addr = ctrl_bo.xdna_addr();
    println!("ctrl BO: hdl={} xdna=0x{:x}", ctrl_bo.handle(), ctrl_addr);

    // A[i,j] = ((i+j)%7)-3, B[j,l] = ((5j+3l)%7): small integers, exact in
    // bf16 and int8. Tensors are SHMEM BOs carrying user VAs in the regmap
    // (same contract as run-add).
    let c_elems = m * n;
    let a_bytes = if is_i8 {
        pack_i8(m, k, |i, j| ((i + j) % 7) as i32 - 3)
    } else {
        pack_bf16(m, k, |i, j| ((i + j) % 7) as i32 - 3)
    };
    let b_bytes = if is_i8 {
        pack_i8(k, n, |j, l| ((5 * j + 3 * l) % 7) as i32)
    } else {
        pack_bf16(k, n, |j, l| ((5 * j + 3 * l) % 7) as i32)
    };

    let tensor = |name: &str, data: &[u8]| -> Option<(BufferObject, Mapping)> {
        let bo = BufferObject::new(&dev, BoType::Shmem, data.len()).ok()?;
        let mut map = bo.map_owned().ok()?;
        map.as_mut_slice().copy_from_slice(data);
        bo.sync(SyncDirection::ToDevice, 0, bo.size() as u64).ok()?;
        println!(
            "{} BO: hdl={} va=0x{:x} ({} B)",
            name,
            bo.handle(),
            map.as_ptr() as u64,
            data.len()
        );
        Some((bo, map))
    };
    let (a_bo, a_map) = match tensor("A", &a_bytes) {
        Some(t) => t,
        None => {
            eprintln!("A BO failed");
            return ExitCode::FAILURE;
        }
    };
    let (b_bo, b_map) = match tensor("B", &b_bytes) {
        Some(t) => t,
        None => {
            eprintln!("B BO failed");
            return ExitCode::FAILURE;
        }
    };
    let (c_bo, c_map) = match tensor("C", &vec![0u8; c_elems * if is_i8 { 4 } else { 2 }]) {
        Some(t) => t,
        None => {
            eprintln!("C BO failed");
            return ExitCode::FAILURE;
        }
    };

    let mut pkt = match StartNpuCmd::new(&dev) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("cmd BO: {e}");
            return ExitCode::FAILURE;
        }
    };
    pkt.set_cu(0);
    let build = pkt
        .arg64(3)
        .and_then(|_| pkt.arg64(ctrl_addr))
        .and_then(|_| pkt.arg32(instr.len() as u32))
        .and_then(|_| pkt.arg64(a_map.as_ptr() as u64))
        .and_then(|_| pkt.arg64(b_map.as_ptr() as u64))
        .and_then(|_| pkt.arg64(c_map.as_ptr() as u64));
    if let Err(e) = build {
        eprintln!("build packet: {e}");
        return ExitCode::FAILURE;
    }

    let arg_handles = [
        ctrl_bo.handle(),
        a_bo.handle(),
        b_bo.handle(),
        c_bo.handle(),
    ];
    // Best-effort warm-up run: the first exec after PDI load pays one-time
    // firmware/queue setup; time the second for the throughput figure.
    if let Err(e) = pkt.submit(&dev, &ctx, &arg_handles) {
        eprintln!("exec submit (warmup): {e}");
        return ExitCode::FAILURE;
    }
    if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, 0, 10_000_000_000) {
        eprintln!("wait (warmup): {e}");
        return ExitCode::FAILURE;
    }
    let t0 = std::time::Instant::now();
    let seq = match pkt.submit(&dev, &ctx, &arg_handles) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("exec submit: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("submitted, seq={seq}");
    if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000) {
        eprintln!("wait seq {seq}: {e}");
        return ExitCode::FAILURE;
    }
    let state = pkt.state();
    let elapsed = t0.elapsed();
    println!("wait done in {elapsed:?}, packet state={state}");
    if state != ERT_CMD_STATE_COMPLETED {
        eprintln!("command did not complete (state {state})");
        return ExitCode::FAILURE;
    }
    let gflops = 2.0 * (m * k * n) as f64 / elapsed.as_secs_f64() / 1e9;
    println!("throughput: {gflops:.1} GFLOP/s (incl. submit overhead)");

    if let Err(e) = c_bo.sync(SyncDirection::ToDevice, 0, c_bo.size() as u64) {
        eprintln!("flush C: {e}");
        return ExitCode::FAILURE;
    }
    let c_bytes: Vec<u8> = c_map.as_slice().to_vec();

    if is_i8 {
        let (mismatches, first_bad) = check_gemm_i8(&a_bytes, &b_bytes, &c_bytes, m, k, n);
        if mismatches == 0 {
            println!(
                "C[0][0..4] = {:?}",
                (0..4.min(n))
                    .map(|l| i32::from_le_bytes(
                        c_bytes[l * 4..l * 4 + 4].try_into().unwrap()
                    ))
                    .collect::<Vec<_>>()
            );
            println!("VERIFY: PASS ({c_elems} elements, i8/i32-acc bit-exact)");
            return ExitCode::SUCCESS;
        }
        if let Some((i, l, got, want)) = first_bad {
            println!("first diff @C[{i}][{l}]: got {got} want {want}");
        }
        eprintln!("VERIFY: FAIL ({mismatches}/{c_elems} mismatches)");
        return ExitCode::FAILURE;
    }

    // Host reference: pre-decode A/B to f32, integer-exact accumulation in
    // f32 with one RNE round to bf16 — bit-exact for f32-accum fixtures
    // (bf16_f32_ONLY, e.g. gemm/test.py); the R2 swiglu-style bf16-accum
    // fixtures drift a few ULPs and pass the ULP tier below.
    let (mismatches, max_ulp, first_bad) = check_gemm(&a_bytes, &b_bytes, &c_bytes, m, k, n);
    if mismatches == 0 {
        println!(
            "C[0][0..4] = {:?}",
            (0..4.min(n))
                .map(|l| {
                    bf16_to_f32(u16::from_le_bytes([c_bytes[l * 2], c_bytes[l * 2 + 1]]))
                })
                .collect::<Vec<_>>()
        );
        println!("VERIFY: PASS ({c_elems} elements, bf16/f32-acc bit-exact)");
        ExitCode::SUCCESS
    } else {
        if let Some((i, l, got, want, d)) = first_bad {
            println!(
                "first diff @C[{i}][{l}]: got bf16 0x{got:04x} ({}) want 0x{want:04x} ({}) [{d} ulp]",
                bf16_to_f32(got),
                bf16_to_f32(want)
            );
        }
        if max_ulp <= 64 {
            // Consistent with a bf16-accumulation fixture measured against an
            // f32-exact reference; the compute path itself is sound.
            println!(
                "VERIFY: PASS ({c_elems} elements, max {max_ulp} ulp vs f32-acc reference — bf16-accum fixture)"
            );
            ExitCode::SUCCESS
        } else {
            eprintln!(
                "VERIFY: FAIL ({mismatches}/{c_elems} diffs, max {max_ulp} ulp — beyond accumulation noise)"
            );
            ExitCode::FAILURE
        }
    }
}

/// One operator's exec state on a shared context: its ctrl-code BO (the
/// instruction regmap slot takes the heap xdna_addr), SHMEM tensor BOs
/// (regmap carries user VAs), and the ERT packet. Everything stays alive for
/// the whole session so repeated submissions reuse identical addresses.
struct OpState {
    name: &'static str,
    cu: u32,
    instr_len: u32,
    pkt: StartNpuCmd,
    ctrl_bo: BufferObject,
    bos: Vec<BufferObject>,
    maps: Vec<Mapping>,
}

impl OpState {
    fn new(dev: &Device, name: &'static str, cu: u32, instr: &[u8]) -> Result<OpState, String> {
        let ctrl_bo = BufferObject::new(dev, BoType::Dev, instr.len())
            .map_err(|e| format!("{name} ctrl BO: {e}"))?;
        dev.write_dev_bo(&ctrl_bo, instr)
            .map_err(|e| format!("{name} write ctrl: {e}"))?;
        ctrl_bo
            .sync(SyncDirection::ToDevice, 0, ctrl_bo.size() as u64)
            .map_err(|e| format!("{name} sync ctrl: {e}"))?;
        let pkt = StartNpuCmd::new(dev).map_err(|e| format!("{name} cmd BO: {e}"))?;
        Ok(OpState {
            name,
            cu,
            instr_len: instr.len() as u32,
            pkt,
            ctrl_bo,
            bos: Vec::new(),
            maps: Vec::new(),
        })
    }

    /// Create one SHMEM tensor, fill it, sync it, park it; returns its VA.
    fn add_tensor(
        &mut self,
        dev: &Device,
        bytes: usize,
        label: &str,
        fill: impl FnOnce(&mut [u8]),
    ) -> Result<u64, String> {
        let bo = BufferObject::new(dev, BoType::Shmem, bytes)
            .map_err(|e| format!("{label} BO: {e}"))?;
        let mut map = bo.map_owned().map_err(|e| format!("{label} map: {e}"))?;
        fill(map.as_mut_slice());
        bo.sync(SyncDirection::ToDevice, 0, bo.size() as u64)
            .map_err(|e| format!("{label} sync: {e}"))?;
        println!("{label} BO: hdl={} va=0x{:x}", bo.handle(), map.as_ptr() as u64);
        let va = map.as_ptr() as u64;
        self.bos.push(bo);
        self.maps.push(map);
        Ok(va)
    }

    /// Emit the DPU regmap: [opcode=3][instr VA][ninstr][tensor VAs...].
    fn build(&mut self, tensor_vas: &[u64]) -> Result<(), String> {
        let ctrl_addr = self.ctrl_bo.xdna_addr();
        println!(
            "{} ctrl BO: hdl={} xdna=0x{:x}",
            self.name,
            self.ctrl_bo.handle(),
            ctrl_addr
        );
        let pkt = &mut self.pkt;
        pkt.set_cu(self.cu);
        let mut r = pkt
            .set_ctrl(ctrl_addr, self.instr_len)
            .and_then(|_| pkt.arg64(3)) // opcode: DPU txn start
            .and_then(|_| pkt.arg64(ctrl_addr))
            .and_then(|_| pkt.arg32(self.instr_len));
        for va in tensor_vas {
            r = r.and_then(|_| pkt.arg64(*va));
        }
        r.map_err(|e| format!("{} packet: {e}", self.name))
    }

    /// Submit and wait; returns (packet state, wall time).
    fn exec(&mut self, dev: &Device, ctx: &HwContext<'_>) -> Result<(u32, std::time::Duration), String> {
        let mut handles = Vec::with_capacity(self.bos.len() + 1);
        handles.push(self.ctrl_bo.handle());
        handles.extend(self.bos.iter().map(|b| b.handle()));
        let t0 = std::time::Instant::now();
        let seq = self
            .pkt
            .submit(dev, ctx, &handles)
            .map_err(|e| format!("{} submit: {e}", self.name))?;
        syncobj_timeline_wait(dev, ctx.syncobj_handle, seq, 10_000_000_000)
            .map_err(|e| format!("{} wait seq {seq}: {e}", self.name))?;
        Ok((self.pkt.state(), t0.elapsed()))
    }

    /// Flush and snapshot the output tensor (always bos/maps index 2 here).
    fn take_output(&self) -> Vec<u8> {
        let bo = &self.bos[2];
        let _ = bo.sync(SyncDirection::ToDevice, 0, bo.size() as u64);
        self.maps[2].as_slice().to_vec()
    }
}

/// Row-major bf16 grid of `rows*cols` cells holding small integers.
fn pack_bf16(rows: usize, cols: usize, f: impl Fn(usize, usize) -> i32) -> Vec<u8> {
    let mut v = Vec::with_capacity(rows * cols * 2);
    for i in 0..rows {
        for j in 0..cols {
            v.extend_from_slice(&f32_to_bf16(f(i, j) as f32).to_le_bytes());
        }
    }
    v
}

/// Row-major int8 grid — the q8-route-A GEMM fixture input format.
fn pack_i8(rows: usize, cols: usize, f: impl Fn(usize, usize) -> i32) -> Vec<u8> {
    let mut v = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            v.push(f(i, j) as i8 as u8);
        }
    }
    v
}

/// Reference for the int8 GEMM fixtures: exact i64 accumulation of i8*i8
/// products, matching the kernel's i32 accumulator bit-for-bit (values are
/// small enough that no i32 overflow is possible). Returns (mismatches,
/// first diff as (i, l, got, want)).
fn check_gemm_i8(
    a: &[u8],
    b: &[u8],
    c: &[u8],
    m: usize,
    k: usize,
    n: usize,
) -> (usize, Option<(usize, usize, i32, i32)>) {
    let mut mismatches = 0usize;
    let mut first: Option<(usize, usize, i32, i32)> = None;
    for i in 0..m {
        for l in 0..n {
            let mut acc: i64 = 0;
            for j in 0..k {
                acc += (a[i * k + j] as i8 as i64) * (b[j * n + l] as i8 as i64);
            }
            let want = acc as i32;
            let off = (i * n + l) * 4;
            let got = i32::from_le_bytes([c[off], c[off + 1], c[off + 2], c[off + 3]]);
            if got != want {
                mismatches += 1;
                if first.is_none() {
                    first = Some((i, l, got, want));
                }
            }
        }
    }
    (mismatches, first)
}

fn build_add_op(dev: &Device, instr: &[u8], cu: u32) -> Result<OpState, String> {
    let mut op = OpState::new(dev, "add", cu, instr)?;
    let n = 2048usize;
    let in1 = op.add_tensor(dev, n * 2, "add.in1", |s| {
        for i in 0..n {
            DType::Bf16.pack_at(s, i, (i % 7) as f32 - 3.0);
        }
    })?;
    let in2 = op.add_tensor(dev, n * 2, "add.in2", |s| {
        for i in 0..n {
            DType::Bf16.pack_at(s, i, ((i / 7) % 5) as f32);
        }
    })?;
    let out = op.add_tensor(dev, n * 2, "add.out", |s| s.fill(0))?;
    op.build(&[in1, in2, out])?;
    Ok(op)
}

/// Returns the op plus the packed A/B bytes (kept for host-side reference).
fn build_gemm_op(
    dev: &Device,
    instr: &[u8],
    cu: u32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(OpState, Vec<u8>, Vec<u8>), String> {
    let mut op = OpState::new(dev, "gemm", cu, instr)?;
    let a_bytes = pack_bf16(m, k, |i, j| ((i + j) % 7) as i32 - 3);
    let b_bytes = pack_bf16(k, n, |j, l| ((5 * j + 3 * l) % 7) as i32);
    let a = op.add_tensor(dev, a_bytes.len(), "gemm.A", |s| s.copy_from_slice(&a_bytes))?;
    let b = op.add_tensor(dev, b_bytes.len(), "gemm.B", |s| s.copy_from_slice(&b_bytes))?;
    let c = op.add_tensor(dev, m * n * 2, "gemm.C", |s| s.fill(0))?;
    op.build(&[a, b, c])?;
    Ok((op, a_bytes, b_bytes))
}

/// M2 follow-up: two operators — the 2048-element bf16 add and an MxKxN GEMM
/// — sharing ONE hardware context as two CUs, both PDIs attached in a single
/// CONFIG_HWCTX (the driver refuses any re-config). Each exec packet picks
/// its CU via the cu_mask bit.
///
/// Open question this answers: both fixtures program the full array, so does
/// running one CU clobber the other's tiles? Schedule: add -> gemm -> add ->
/// gemm; a broken round-2 op is the clobber signature.
fn cmd_run_multi(add_prj: &str, gemm_prj: &str, m: usize, k: usize, n: usize) -> ExitCode {
    let (add_pdi, add_instr, add_cols) = match load_fixture(add_prj) {
        Some(f) => f,
        None => {
            eprintln!("load fixture {add_prj} failed");
            return ExitCode::FAILURE;
        }
    };
    let (gemm_pdi, gemm_instr, gemm_cols) = match load_fixture(gemm_prj) {
        Some(f) => f,
        None => {
            eprintln!("load fixture {gemm_prj} failed");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "fixtures: add pdi {} B / ctrl {} B ({} cols), gemm pdi {} B / ctrl {} B ({} cols)",
        add_pdi.len(),
        add_instr.len(),
        add_cols,
        gemm_pdi.len(),
        gemm_instr.len(),
        gemm_cols
    );
    println!("gemm problem: C[{m}x{n}] = A[{m}x{k}] @ B[{k}x{n}], bf16/f32-acc");

    let dev = match Device::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open amdxdna device: {e}");
            return ExitCode::FAILURE;
        }
    };
    let md = match dev.aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("AIE metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    // IRON reserves the full array in every fixture (partition json always
    // says column_width=8), so one full-width context serves both CUs.
    let cols = add_cols.max(gemm_cols);
    let num_tiles = cols * md.core.row_count as u32;
    let mut ctx = match HwContext::create(&dev, num_tiles) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "hwctx: handle={} syncobj={} ({} cols / {} tiles)",
        ctx.handle, ctx.syncobj_handle, cols, num_tiles
    );

    // The experiment proper: CU 0 = add, CU 1 = gemm, in one CONFIG_HWCTX.
    if let Err(e) = ctx.configure_cus(&[(&add_pdi, 0), (&gemm_pdi, 0)]) {
        eprintln!("configure_cus (2 PDIs, one config): {e}");
        return ExitCode::FAILURE;
    }
    println!("2 CUs attached in one CONFIG_HWCTX (cu 0 = add, cu 1 = gemm)");

    let mut add_op = match build_add_op(&dev, &add_instr, 0) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let (mut gemm_op, a_bytes, b_bytes) = match build_gemm_op(&dev, &gemm_instr, 1, m, k, n) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    // Scheduling comparison, the run-pipe follow-up: the pipelined run showed
    // this gemm takes ~36us of device time when executed back-to-back on its
    // own CU, but ~1.2ms when alternating with the add CU — the firmware
    // reloads a CU's PDI on every cu_mask change. Burst (same-CU runs) vs
    // interleaved (alternating CUs) here quantifies that switch cost with the
    // submit+wait overhead common to both sides.
    let reps = 8usize;
    let exec_ok = |op: &mut OpState| -> Option<std::time::Duration> {
        match op.exec(&dev, &ctx) {
            Ok((state, dt)) if state == ERT_CMD_STATE_COMPLETED => Some(dt),
            Ok((state, _)) => {
                eprintln!("{} did not complete (state {state})", op.name);
                None
            }
            Err(e) => {
                eprintln!("{}: {e}", op.name);
                None
            }
        }
    };

    // Burst: reps consecutive add execs, then reps consecutive gemm execs.
    let mut all_ok = true;
    let mut add_burst = std::time::Duration::ZERO;
    let mut gemm_burst = std::time::Duration::ZERO;
    for _ in 0..reps {
        match exec_ok(&mut add_op) {
            Some(dt) => add_burst += dt,
            None => all_ok = false,
        }
    }
    let out = add_op.take_output();
    let (mm, _) = check_add(&out, DType::Bf16, 2048);
    all_ok &= mm == 0;
    for _ in 0..reps {
        match exec_ok(&mut gemm_op) {
            Some(dt) => gemm_burst += dt,
            None => all_ok = false,
        }
    }
    let c = gemm_op.take_output();
    let (mm, max_ulp, _) = check_gemm(&a_bytes, &b_bytes, &c, m, k, n);
    all_ok &= mm == 0 || max_ulp <= 64;
    println!(
        "burst:      add {:>10?}/op   gemm {:>10?}/op (verify {})",
        add_burst / reps as u32,
        gemm_burst / reps as u32,
        if all_ok { "PASS" } else { "FAIL" }
    );

    // Interleaved: reps alternating add -> gemm pairs.
    let mut add_alt = std::time::Duration::ZERO;
    let mut gemm_alt = std::time::Duration::ZERO;
    for _ in 0..reps {
        match exec_ok(&mut add_op) {
            Some(dt) => add_alt += dt,
            None => all_ok = false,
        }
        match exec_ok(&mut gemm_op) {
            Some(dt) => gemm_alt += dt,
            None => all_ok = false,
        }
    }
    let out = add_op.take_output();
    let (mm, _) = check_add(&out, DType::Bf16, 2048);
    all_ok &= mm == 0;
    let c = gemm_op.take_output();
    let (mm, max_ulp, _) = check_gemm(&a_bytes, &b_bytes, &c, m, k, n);
    all_ok &= mm == 0 || max_ulp <= 64;
    println!(
        "interleaved: add {:>10?}/op   gemm {:>10?}/op (verify {})",
        add_alt / reps as u32,
        gemm_alt / reps as u32,
        if mm == 0 || max_ulp <= 64 { "PASS" } else { "FAIL" }
    );
    println!(
        "CU-switch cost: add +{:?}, gemm +{:?} per alternation",
        (add_alt - add_burst) / reps as u32,
        (gemm_alt - gemm_burst) / reps as u32
    );
    if all_ok {
        println!("MULTI-CU: PASS — both operators verified under burst and interleaved scheduling");
        ExitCode::SUCCESS
    } else {
        eprintln!("MULTI-CU: FAIL — see per-op results above");
        ExitCode::FAILURE
    }
}

/// M2 finale: command-queue pipelining — the decode-bottleneck question.
///
/// The R3 Python stack lost to CPU (2.26 vs 4.55 tok/s) because every
/// operator paid a submit+wait round trip (42 layers x ~8 ops per forward).
/// Here N packets go to one CU back-to-back with no intervening waits; the
/// queue executes them in order anyway, so only true device time remains.
/// Sequential vs pipelined totals put a number on the recoverable overhead.
///
/// Every exec's returned timeline seq is also printed — the M1 "seq always 0"
/// observation came from runs that only ever submitted once, so the first
/// point 0 was also the last. A pipelined run should walk the timeline.
fn cmd_run_pipe(gemm_prj: &str, m: usize, k: usize, n: usize, iters: usize) -> ExitCode {
    let (pdi, instr, cols) = match load_fixture(gemm_prj) {
        Some(f) => f,
        None => {
            eprintln!("load fixture {gemm_prj} failed");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "fixture: pdi {} B, ctrl-code {} B, partition {} cols",
        pdi.len(),
        instr.len(),
        cols
    );
    println!(
        "problem: C[{m}x{n}] = A[{m}x{k}] @ B[{k}x{n}], {} iters",
        iters
    );

    let dev = match Device::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open amdxdna device: {e}");
            return ExitCode::FAILURE;
        }
    };
    let md = match dev.aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("AIE metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let num_tiles = cols * md.core.row_count as u32;
    let mut ctx = match HwContext::create(&dev, num_tiles) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "hwctx: handle={} syncobj={} ({} cols / {} tiles)",
        ctx.handle, ctx.syncobj_handle, cols, num_tiles
    );
    if let Err(e) = ctx.configure_cu(&pdi, 0) {
        eprintln!("configure_cu (PDI load): {e}");
        return ExitCode::FAILURE;
    }

    let (mut op, a_bytes, b_bytes) = match build_gemm_op(&dev, &instr, 0, m, k, n) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    // Packet pool: every in-flight exec needs its own cmd BO (the firmware
    // consumes them asynchronously); the regmap is identical across the pool
    // since the tensors are shared.
    let ctrl_addr = op.ctrl_bo.xdna_addr();
    let tensor_vas: Vec<u64> = op.maps.iter().map(|mp| mp.as_ptr() as u64).collect();
    let mut pkts: Vec<StartNpuCmd> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut p = match StartNpuCmd::new(&dev) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("cmd BO: {e}");
                return ExitCode::FAILURE;
            }
        };
        p.set_cu(0);
        let mut r = p
            .set_ctrl(ctrl_addr, instr.len() as u32)
            .and_then(|_| p.arg64(3))
            .and_then(|_| p.arg64(ctrl_addr))
            .and_then(|_| p.arg32(instr.len() as u32));
        for va in &tensor_vas {
            r = r.and_then(|_| p.arg64(*va));
        }
        if let Err(e) = r {
            eprintln!("build packet: {e}");
            return ExitCode::FAILURE;
        }
        pkts.push(p);
    }
    let mut handles = Vec::with_capacity(op.bos.len() + 1);
    handles.push(op.ctrl_bo.handle());
    handles.extend(op.bos.iter().map(|b| b.handle()));

    // One warm-up exec (firmware/queue setup) through the op's own packet.
    match op.exec(&dev, &ctx) {
        Ok((state, _)) if state == ERT_CMD_STATE_COMPLETED => {}
        Ok((state, _)) => {
            eprintln!("warmup did not complete (state {state})");
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("warmup: {e}");
            return ExitCode::FAILURE;
        }
    }

    // Phase 1 — sequential: the R3 pattern, submit+wait per op.
    let mut seqs: Vec<u64> = Vec::with_capacity(iters);
    let t0 = std::time::Instant::now();
    for p in pkts.iter_mut() {
        let seq = match p.submit(&dev, &ctx, &handles) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("submit (sequential): {e}");
                return ExitCode::FAILURE;
            }
        };
        seqs.push(seq);
        if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000) {
            eprintln!("wait seq {seq}: {e}");
            return ExitCode::FAILURE;
        }
        let state = p.state();
        if state != ERT_CMD_STATE_COMPLETED {
            eprintln!("sequential exec did not complete (state {state})");
            return ExitCode::FAILURE;
        }
    }
    let seq_elapsed = t0.elapsed();
    let c1 = op.take_output();
    let (mm, _, _) = check_gemm(&a_bytes, &b_bytes, &c1, m, k, n);
    println!(
        "sequential: {:?} total, {:?}/op (verify {})",
        seq_elapsed,
        seq_elapsed / iters as u32,
        if mm == 0 { "PASS" } else { "FAIL" }
    );
    println!("seqs walked: first={:?} last={:?}", seqs.first(), seqs.last());

    // Phase 2 — pipelined: submit everything, then poll the cmd BOs' state
    // fields for completion. Same tensors throughout, so the in-order queue
    // leaves a correct result in C regardless of depth.
    let mut seqs2: Vec<u64> = Vec::with_capacity(iters);
    let t0 = std::time::Instant::now();
    for p in pkts.iter_mut() {
        match p.submit(&dev, &ctx, &handles) {
            Ok(s) => seqs2.push(s),
            Err(e) => {
                eprintln!("submit (pipelined, depth {}): {e}", seqs2.len() + 1);
                return ExitCode::FAILURE;
            }
        }
    }
    let submit_elapsed = t0.elapsed();
    let t1 = std::time::Instant::now();
    let mut spins = 0u64;
    loop {
        if pkts.iter().all(|p| p.state() == ERT_CMD_STATE_COMPLETED) {
            break;
        }
        spins += 1;
        if spins > 40_000_000 {
            eprintln!("pipelined drain timed out");
            return ExitCode::FAILURE;
        }
        std::thread::sleep(std::time::Duration::from_micros(50));
    }
    let drain_elapsed = t1.elapsed();
    let pipe_elapsed = t0.elapsed();
    let c2 = op.take_output();
    let (mm2, _, _) = check_gemm(&a_bytes, &b_bytes, &c2, m, k, n);
    println!(
        "pipelined:  {:?} total, {:?}/op (submit loop {:?}, drain {:?}, {} polls, verify {})",
        pipe_elapsed,
        pipe_elapsed / iters as u32,
        submit_elapsed,
        drain_elapsed,
        spins,
        if mm2 == 0 { "PASS" } else { "FAIL" }
    );
    println!("seqs walked: first={:?} last={:?}", seqs2.first(), seqs2.last());
    println!(
        "speedup: {:.2}x (recoverable per-op overhead {:?})",
        seq_elapsed.as_secs_f64() / pipe_elapsed.as_secs_f64(),
        seq_elapsed.saturating_sub(pipe_elapsed) / iters as u32
    );
    if mm == 0 && mm2 == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Create one SHMEM tensor, fill it, sync it, park the (BO, mapping) pair in
/// `live` so both outlive the experiment; returns its VA for the regmap.
fn chain_tensor(
    dev: &Device,
    live: &mut Vec<(BufferObject, Mapping)>,
    label: &str,
    data: &[u8],
) -> Option<u64> {
    let bo = BufferObject::new(dev, BoType::Shmem, data.len()).ok()?;
    let mut map = bo.map_owned().ok()?;
    map.as_mut_slice().copy_from_slice(data);
    bo.sync(SyncDirection::ToDevice, 0, bo.size() as u64).ok()?;
    println!("{label} BO: hdl={} va=0x{:x}", bo.handle(), map.as_ptr() as u64);
    let va = map.as_ptr() as u64;
    live.push((bo, map));
    Some(va)
}

/// One op's packet + its ctrl-code BO, kept together so both stay alive.
struct ChainOp {
    pkt: StartNpuCmd,
    ctrl_bo: BufferObject,
}

fn chain_op(
    dev: &Device,
    label: &str,
    instr: &[u8],
    cu: u32,
    tensor_vas: &[u64],
) -> Option<ChainOp> {
    let ctrl_bo = BufferObject::new(dev, BoType::Dev, instr.len()).ok()?;
    dev.write_dev_bo(&ctrl_bo, instr).ok()?;
    ctrl_bo.sync(SyncDirection::ToDevice, 0, ctrl_bo.size() as u64).ok()?;
    let ctrl_addr = ctrl_bo.xdna_addr();
    println!("{label} ctrl BO: hdl={} xdna=0x{:x}", ctrl_bo.handle(), ctrl_addr);
    let mut pkt = StartNpuCmd::new(dev).ok()?;
    pkt.set_cu(cu);
    let mut r = pkt
        .set_ctrl(ctrl_addr, instr.len() as u32)
        .and_then(|_| pkt.arg64(3))
        .and_then(|_| pkt.arg64(ctrl_addr))
        .and_then(|_| pkt.arg32(instr.len() as u32));
    for va in tensor_vas {
        r = r.and_then(|_| pkt.arg64(*va));
    }
    r.ok()?;
    Some(ChainOp { pkt, ctrl_bo })
}

/// M3 first experiment: a cross-CU data-dependency chain on one context —
/// the shape of one decode layer. E = (A @ B) + D where the gemm's output C
/// and the add's first input are THE SAME buffer object: the regmap just
/// carries the same VA in both packets, the in-order queue provides the
/// ordering, and no host round trip or copy happens between the ops.
///
/// Sequential (wait between ops) vs chained (submit both, wait once)
/// timings measure what layer-level batching recovers, now with a real
/// cross-CU dependency in flight — including the PDI-reload switch cost
/// that run-multi attributed to every cu_mask change.
fn cmd_run_chain(
    add_prj: &str,
    gemm_prj: &str,
    m: usize,
    k: usize,
    n: usize,
    reps: usize,
) -> ExitCode {
    let (add_pdi, add_instr, add_cols) = match load_fixture(add_prj) {
        Some(f) => f,
        None => {
            eprintln!("load fixture {add_prj} failed");
            return ExitCode::FAILURE;
        }
    };
    let (gemm_pdi, gemm_instr, gemm_cols) = match load_fixture(gemm_prj) {
        Some(f) => f,
        None => {
            eprintln!("load fixture {gemm_prj} failed");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "chain: E[0..2048] = (A[{m}x{k}] @ B[{k}x{n}])[0..2048] + D, gemm(cu1) -> add(cu0), shared C, {reps} reps"
    );

    let dev = match Device::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open amdxdna device: {e}");
            return ExitCode::FAILURE;
        }
    };
    let md = match dev.aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("AIE metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let cols = add_cols.max(gemm_cols);
    let num_tiles = cols * md.core.row_count as u32;
    let mut ctx = match HwContext::create(&dev, num_tiles) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = ctx.configure_cus(&[(&add_pdi, 0), (&gemm_pdi, 0)]) {
        eprintln!("configure_cus: {e}");
        return ExitCode::FAILURE;
    }
    println!("2 CUs attached (cu 0 = add, cu 1 = gemm)");

    // A/B gemm inputs; C shared gemm-out/add-in (full gemm size — the add
    // only touches its first 2048 elements); D add input 2; E add output.
    let a_bytes = pack_bf16(m, k, |i, j| ((i + j) % 7) as i32 - 3);
    let b_bytes = pack_bf16(k, n, |j, l| ((5 * j + 3 * l) % 7) as i32);
    let mut d_bytes = vec![0u8; 2048 * 2];
    for i in 0..2048 {
        let bits = f32_to_bf16(((i / 7) % 5) as f32).to_le_bytes();
        d_bytes[i * 2..i * 2 + 2].copy_from_slice(&bits);
    }
    let c_zero = vec![0u8; m * n * 2];
    let e_zero = vec![0u8; 2048 * 2];
    let mut live: Vec<(BufferObject, Mapping)> = Vec::new();
    let vas: Vec<u64> = [
        ("A", a_bytes.as_slice()),
        ("B", b_bytes.as_slice()),
        ("C", c_zero.as_slice()),
        ("D", d_bytes.as_slice()),
        ("E", e_zero.as_slice()),
    ]
    .iter()
    .filter_map(|(label, data)| chain_tensor(&dev, &mut live, label, data))
    .collect();
    if vas.len() != 5 {
        eprintln!("tensor allocation failed");
        return ExitCode::FAILURE;
    }
    let [a_va, b_va, c_va, d_va, e_va] = [vas[0], vas[1], vas[2], vas[3], vas[4]];

    let mut gemm = match chain_op(&dev, "gemm", &gemm_instr, 1, &[a_va, b_va, c_va]) {
        Some(o) => o,
        None => {
            eprintln!("gemm op setup failed");
            return ExitCode::FAILURE;
        }
    };
    let mut add = match chain_op(&dev, "add", &add_instr, 0, &[c_va, d_va, e_va]) {
        Some(o) => o,
        None => {
            eprintln!("add op setup failed");
            return ExitCode::FAILURE;
        }
    };
    let gemm_handles = [gemm.ctrl_bo.handle(), live[0].0.handle(), live[1].0.handle(), live[2].0.handle()];
    let add_handles = [add.ctrl_bo.handle(), live[2].0.handle(), live[3].0.handle(), live[4].0.handle()];

    // Host reference for E: C from the f32-accumulation reference, then the
    // add's exact bf16 sum (both fixtures are small-integer exact).
    let e_ref: Vec<u16> = {
        let dec = |bytes: &[u8], e: usize| bf16_to_f32(u16::from_le_bytes([bytes[e * 2], bytes[e * 2 + 1]]));
        (0..2048)
            .map(|i| {
                // C_ref[i]: row i/n, col i%n of the reference product.
                let (row, col) = (i / n, i % n);
                let mut acc = 0f32;
                for j in 0..k {
                    let av = dec(&a_bytes, row * k + j);
                    let bv = dec(&b_bytes, j * n + col);
                    acc += av * bv;
                }
                f32_to_bf16(bf16_to_f32(f32_to_bf16(acc)) + dec(&d_bytes, i))
            })
            .collect()
    };

    let submit_wait = |op: &mut ChainOp, handles: &[u32]| -> Option<std::time::Duration> {
        let t0 = std::time::Instant::now();
        let seq = op.pkt.submit(&dev, &ctx, handles).ok()?;
        syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000).ok()?;
        if op.pkt.state() != ERT_CMD_STATE_COMPLETED {
            return None;
        }
        Some(t0.elapsed())
    };

    // Warmup chain.
    if submit_wait(&mut gemm, &gemm_handles).is_none()
        || submit_wait(&mut add, &add_handles).is_none()
    {
        eprintln!("warmup chain failed");
        return ExitCode::FAILURE;
    }

    // Sequential: the R3 pattern, one wait per op.
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        if submit_wait(&mut gemm, &gemm_handles).is_none()
            || submit_wait(&mut add, &add_handles).is_none()
        {
            eprintln!("sequential chain failed");
            return ExitCode::FAILURE;
        }
    }
    let seq_total = t0.elapsed();

    // Interim check after the sequential phase only — separates a wrong
    // reference (fails here too) from a broken cross-CU dependency under
    // chained submission (fails only below). The final add rounds ties
    // differently than host RNE — AIE's bf16 conversion only rounds
    // half-to-even with -DROUND_CONV_EVEN, which the add fixture predates —
    // so exact matches come first and <=1 ULP tie diffs form the pass tier.
    let check_e = |phase: &str| -> (usize, i64) {
        let e_bo = &live[4].0;
        let _ = e_bo.sync(SyncDirection::ToDevice, 0, e_bo.size() as u64);
        let s = live[4].1.as_slice();
        let mut mm = 0usize;
        let mut max_ulp: i64 = 0;
        let mut first: Option<(usize, u16, u16)> = None;
        for i in 0..2048 {
            let got = u16::from_le_bytes([s[i * 2], s[i * 2 + 1]]);
            if got != e_ref[i] {
                let d = if (got >> 15) == (e_ref[i] >> 15) {
                    (got as i16 as i64 - e_ref[i] as i16 as i64).abs()
                } else {
                    1 << 24
                };
                mm += 1;
                max_ulp = max_ulp.max(d);
                if first.is_none() {
                    first = Some((i, got, e_ref[i]));
                }
            }
        }
        if let Some((i, got, want)) = first {
            println!(
                "{phase}: first diff @[{i}]: got 0x{got:04x} ({}) want 0x{want:04x} ({})",
                bf16_to_f32(got),
                bf16_to_f32(want)
            );
        }
        println!(
            "{phase}: E verify {} ({mm}/2048 diffs, max {max_ulp} ulp — tie-rounding tier)",
            if max_ulp <= 1 { "PASS" } else { "FAIL" }
        );
        (mm, max_ulp)
    };
    let (_mm_seq, ulp_seq) = check_e("sequential-phase");

    // Chained: submit gemm+add pairs back-to-back (in-order queue preserves
    // the dependency), drain by polling both packets, repeat.
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        if let Err(e) = gemm.pkt.submit(&dev, &ctx, &gemm_handles) {
            eprintln!("chained gemm submit: {e}");
            return ExitCode::FAILURE;
        }
        if let Err(e) = add.pkt.submit(&dev, &ctx, &add_handles) {
            eprintln!("chained add submit: {e}");
            return ExitCode::FAILURE;
        }
        let mut polls = 0u32;
        loop {
            let g = gemm.pkt.state() == ERT_CMD_STATE_COMPLETED;
            let a = add.pkt.state() == ERT_CMD_STATE_COMPLETED;
            if g && a {
                break;
            }
            polls += 1;
            if polls > 40_000 {
                eprintln!("chained drain timed out");
                return ExitCode::FAILURE;
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }
    let chain_total = t0.elapsed();
    let (_mm_chain, ulp_chain) = check_e("chained-phase");

    println!(
        "sequential: {:?}/chain ({:?}/op avg)",
        seq_total / reps as u32,
        seq_total / (reps as u32 * 2)
    );
    println!(
        "chained:    {:?}/chain — submit-only between ops, one drain per chain",
        chain_total / reps as u32
    );
    println!(
        "speedup: {:.2}x",
        seq_total.as_secs_f64() / chain_total.as_secs_f64()
    );
    if ulp_seq <= 1 && ulp_chain <= 1 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// q8 route A closed loop: D = bf16(f32(A_i8 @ B_i8) × sa[M] × sw[N]).
/// i8 GEMM (cu1) produces an exact int32 accumulation C; rescale (cu0)
/// multiplies it by per-row × per-column scales and narrows to bf16. C is
/// ONE SHMEM BO whose VA rides in both packets — the shared-BO dependency
/// run-chain validated, now with both ends integer-exact: the only rounding
/// in flight is the rescale's conv_even bf16 narrowing, which should match
/// host RNE bit-exactly (the ≤1 ULP tier only guards an unexpected f32-mul
/// tie, mirroring run-chain's check_e).
///
/// The rescale fixture is compiled for one (M, N, tile_m) shape, so M and N
/// must match it exactly (unlike the add fixture, which just touches a
/// prefix of the gemm output).
fn cmd_run_q8(
    gemm_prj: &str,
    rescale_prj: &str,
    m: usize,
    k: usize,
    n: usize,
    tile_m: usize,
    reps: usize,
) -> ExitCode {
    let (gemm_pdi, gemm_instr, gemm_cols) = match load_fixture(gemm_prj) {
        Some(f) => f,
        None => {
            eprintln!("load fixture {gemm_prj} failed");
            return ExitCode::FAILURE;
        }
    };
    let (rs_pdi, rs_instr, rs_cols) = match load_fixture(rescale_prj) {
        Some(f) => f,
        None => {
            eprintln!("load fixture {rescale_prj} failed");
            return ExitCode::FAILURE;
        }
    };
    let num_blocks = m / tile_m;
    if m % tile_m != 0 {
        eprintln!("run-q8: M={m} not a multiple of tile_m={tile_m}");
        return ExitCode::FAILURE;
    }
    println!(
        "q8 loop: D[{m}x{n}] = bf16(f32(A[{m}x{k}] @i8 B[{k}x{n}]) × sa[{m}] × sw[{n}]), gemm(cu1) -> rescale(cu0), shared C, {reps} reps"
    );

    let dev = match Device::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open amdxdna device: {e}");
            return ExitCode::FAILURE;
        }
    };
    let md = match dev.aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("AIE metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let cols = rs_cols.max(gemm_cols);
    let num_tiles = cols * md.core.row_count as u32;
    let mut ctx = match HwContext::create(&dev, num_tiles) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = ctx.configure_cus(&[(&rs_pdi, 0), (&gemm_pdi, 0)]) {
        eprintln!("configure_cus: {e}");
        return ExitCode::FAILURE;
    }
    println!("2 CUs attached (cu 0 = rescale, cu 1 = gemm)");

    // Same integer grids as the i8 run-gemm fixture (exact in i8 and in the
    // i32 accumulator; |acc| ≤ 3·6·k ≪ 2^24 so f32 widening is exact too).
    let a_bytes = pack_i8(m, k, |i, j| ((i + j) % 7) as i32 - 3);
    let b_bytes = pack_i8(k, n, |j, l| ((5 * j + 3 * l) % 7) as i32);
    // Scale grids from the rescale reference: every value of
    // ((r%8)-3)/8 and ((c%5)+1)/16 is exact in bf16, so the f32 the host
    // ships is bit-identical to the bf16 quantization the model would use.
    let sa = |row: usize| (((row % 8) as i32 - 3) as f32) / 8.0;
    let sw = |col: usize| (((col % 5) as i32 + 1) as f32) / 16.0;
    // One concatenated scale block per row block: [tile_m row | N col] f32.
    let mut s_bytes = Vec::with_capacity((tile_m + n) * num_blocks * 4);
    for b in 0..num_blocks {
        for r in 0..tile_m {
            s_bytes.extend_from_slice(&sa(b * tile_m + r).to_le_bytes());
        }
        for l in 0..n {
            s_bytes.extend_from_slice(&sw(l).to_le_bytes());
        }
    }
    let c_zero = vec![0u8; m * n * 4];
    let d_zero = vec![0u8; m * n * 2];
    let mut live: Vec<(BufferObject, Mapping)> = Vec::new();
    let vas: Vec<u64> = [
        ("A", a_bytes.as_slice()),
        ("B", b_bytes.as_slice()),
        ("C", c_zero.as_slice()),
        ("S", s_bytes.as_slice()),
        ("D", d_zero.as_slice()),
    ]
    .iter()
    .filter_map(|(label, data)| chain_tensor(&dev, &mut live, label, data))
    .collect();
    if vas.len() != 5 {
        eprintln!("tensor allocation failed");
        return ExitCode::FAILURE;
    }
    let [a_va, b_va, c_va, s_va, d_va] = [vas[0], vas[1], vas[2], vas[3], vas[4]];

    let mut gemm = match chain_op(&dev, "gemm", &gemm_instr, 1, &[a_va, b_va, c_va]) {
        Some(o) => o,
        None => {
            eprintln!("gemm op setup failed");
            return ExitCode::FAILURE;
        }
    };
    // rescale runlist order: (input, scales, output).
    let mut rescale = match chain_op(&dev, "rescale", &rs_instr, 0, &[c_va, s_va, d_va]) {
        Some(o) => o,
        None => {
            eprintln!("rescale op setup failed");
            return ExitCode::FAILURE;
        }
    };
    let gemm_handles = [gemm.ctrl_bo.handle(), live[0].0.handle(), live[1].0.handle(), live[2].0.handle()];
    let rs_handles = [
        rescale.ctrl_bo.handle(),
        live[2].0.handle(),
        live[3].0.handle(),
        live[4].0.handle(),
    ];

    // Host reference: exact i32 dot products, then the kernel's multiply
    // order ((acc×sa)×sw) and RNE narrowing — conv_even should match RNE
    // bit-for-bit.
    let d_ref: Vec<u16> = (0..m * n)
        .map(|i| {
            let (row, col) = (i / n, i % n);
            let mut acc: i32 = 0;
            for j in 0..k {
                acc += (a_bytes[row * k + j] as i8 as i32) * (b_bytes[j * n + col] as i8 as i32);
            }
            f32_to_bf16((acc as f32 * sa(row)) * sw(col))
        })
        .collect();

    let submit_wait = |op: &mut ChainOp, handles: &[u32]| -> Option<std::time::Duration> {
        let t0 = std::time::Instant::now();
        let seq = op.pkt.submit(&dev, &ctx, handles).ok()?;
        syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000).ok()?;
        if op.pkt.state() != ERT_CMD_STATE_COMPLETED {
            return None;
        }
        Some(t0.elapsed())
    };

    if submit_wait(&mut gemm, &gemm_handles).is_none()
        || submit_wait(&mut rescale, &rs_handles).is_none()
    {
        eprintln!("warmup loop failed");
        return ExitCode::FAILURE;
    }

    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        if submit_wait(&mut gemm, &gemm_handles).is_none()
            || submit_wait(&mut rescale, &rs_handles).is_none()
        {
            eprintln!("sequential loop failed");
            return ExitCode::FAILURE;
        }
    }
    let seq_total = t0.elapsed();

    // Two-tier check after the sequential phase (wrong reference vs broken
    // dependency, same split as run-chain).
    let check_d = |phase: &str| -> (usize, i64) {
        let d_bo = &live[4].0;
        let _ = d_bo.sync(SyncDirection::ToDevice, 0, d_bo.size() as u64);
        let s = live[4].1.as_slice();
        let total = m * n;
        let mut mm = 0usize;
        let mut max_ulp: i64 = 0;
        let mut first: Option<(usize, u16, u16)> = None;
        // Normalize signed zero: rows whose scale is 0 multiply out to ±0 and
        // the AIE f32 mul encodes the negative case as +0 (only the zero
        // operand triggers this — all non-zero magnitudes and signs matched
        // bit-exactly). ±0 compares equal in every inference-relevant sense.
        let norm = |b: u16| if b & 0x7fff == 0 { 0u16 } else { b };
        for i in 0..total {
            let got = norm(u16::from_le_bytes([s[i * 2], s[i * 2 + 1]]));
            if got != norm(d_ref[i]) {
                let d = if (got >> 15) == (d_ref[i] >> 15) {
                    (got as i16 as i64 - d_ref[i] as i16 as i64).abs()
                } else {
                    1 << 24
                };
                mm += 1;
                max_ulp = max_ulp.max(d);
                if first.is_none() {
                    first = Some((i, got, d_ref[i]));
                }
            }
        }
        if let Some((i, got, want)) = first {
            println!(
                "{phase}: first diff @[{i}] (row {}, col {}): got 0x{got:04x} ({}) want 0x{want:04x} ({})",
                i / n,
                i % n,
                bf16_to_f32(got),
                bf16_to_f32(want)
            );
        }
        println!(
            "{phase}: D verify {} ({mm}/{total} diffs, max {max_ulp} ulp)",
            if max_ulp <= 1 { "PASS" } else { "FAIL" }
        );
        (mm, max_ulp)
    };
    let (_mm_seq, ulp_seq) = check_d("sequential-phase");

    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        if let Err(e) = gemm.pkt.submit(&dev, &ctx, &gemm_handles) {
            eprintln!("chained gemm submit: {e}");
            return ExitCode::FAILURE;
        }
        if let Err(e) = rescale.pkt.submit(&dev, &ctx, &rs_handles) {
            eprintln!("chained rescale submit: {e}");
            return ExitCode::FAILURE;
        }
        let mut polls = 0u32;
        loop {
            let g = gemm.pkt.state() == ERT_CMD_STATE_COMPLETED;
            let r = rescale.pkt.state() == ERT_CMD_STATE_COMPLETED;
            if g && r {
                break;
            }
            polls += 1;
            if polls > 40_000 {
                eprintln!("chained drain timed out");
                return ExitCode::FAILURE;
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }
    let chain_total = t0.elapsed();
    let (_mm_chain, ulp_chain) = check_d("chained-phase");

    let gemm_ops = (2 * m * k * n) as f64 / 1e9;
    println!(
        "sequential: {:?}/loop ({:?}/op avg), chained: {:?}/loop",
        seq_total / reps as u32,
        seq_total / (reps as u32 * 2),
        chain_total / reps as u32
    );
    println!(
        "q8 gemm throughput (chained): {:.0} GOP/s effective over the loop",
        gemm_ops / (chain_total.as_secs_f64() / reps as f64)
    );
    if ulp_seq <= 1 && ulp_chain <= 1 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// W4 GEMV through the w4gemv2 fixture: persistent workers (infinite fifo
/// loop, no core restart between launches) stream packed SIGNED int4
/// weights + per-group-32 bf16 scales, dequantize in-register (the int4
/// unpack sign-extends; nibbles are two's-complement [-8,7]) and produce
/// C[i] = bf16(Σ_k w_dequant(i,k)·x[k]) with a conv_even narrowing.
///
/// Test data keeps every intermediate exact (the M1 recipe, now for the
/// quantized-weight path): scales are dyadics ((g%4)+1)/16, nibbles in
/// [-8,7], so w_dequant = nibble×scale is exact in bf16; x is small
/// integers; each product is a multiple of 2⁻⁴ with ≤ 19 significant bits
/// summed over K terms — exact in f32 regardless of the kernel's lane split
/// and reduce_add order. The only rounding in flight is the final bf16
/// narrowing, so host RNE must match bit-for-bit.
fn cmd_run_w4gemv(
    prj: &str,
    m: usize,
    k: usize,
    group: usize,
    tsi: usize,
    iters: usize,
) -> ExitCode {
    let (pdi, instr, cols) = match load_fixture(prj) {
        Some(f) => f,
        None => {
            eprintln!("load fixture {prj} failed");
            return ExitCode::FAILURE;
        }
    };
    // Fixture geometry: `cols` AIE columns each owning M/cols rows; each
    // tile covers tsi rows as [tsi*K/2 nibble bytes | tsi*(K/group) bf16
    // scales], tiles stacked per column (column 0 first).
    let ncols = cols as usize;
    let tile_bytes = tsi * (k / 2) + tsi * (k / group) * 2;
    let a_bytes_len = ncols * (m / ncols / tsi) * tile_bytes;
    println!(
        "w4 gemv: C[{m}] = W_packed[{m}x{k}] @ x[{k}], signed int4/g{group} scales, {cols} cols, tile {tile_bytes} B, weights {} KB, {iters} iters",
        a_bytes_len / 1024
    );

    let dev = match Device::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open amdxdna device: {e}");
            return ExitCode::FAILURE;
        }
    };
    let md = match dev.aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("AIE metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let num_tiles = cols * md.core.row_count as u32;
    let mut ctx = match HwContext::create(&dev, num_tiles) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = ctx.configure_cu(&pdi, 0) {
        eprintln!("configure_cu (PDI load): {e}");
        return ExitCode::FAILURE;
    }
    println!("PDI loaded, CU configured ({cols} cols / {num_tiles} tiles)");

    // Deterministic data: nibble(i,k) = ((13i+7k)%16)-8 ∈ [-8,7] (signed
    // two's-complement, exercising both signs), scale(g) = ((g%4)+1)/16
    // (exact bf16 dyadic), x[k] = (k%7)-3 (small integer).
    let nibble = |i: usize, kk: usize| (((13 * i + 7 * kk) % 16) as i32 - 8) as i8;
    let scale = |g: usize| (((g % 4) + 1) as f32) / 16.0;
    let groups_per_row = k / group;
    let mut a_bytes = vec![0u8; a_bytes_len];
    let rows_per_col = m / ncols;
    let nibble_bytes = tsi * k / 2;
    for i in 0..m {
        // Row i lives in column i/rows_per_col, tile (i%rows_per_col)/tsi,
        // local row (i%rows_per_col)%tsi within the tile.
        let col = i / rows_per_col;
        let r = i % rows_per_col;
        let off = (col * (rows_per_col / tsi) + r / tsi) * tile_bytes;
        let rr = r % tsi;
        for p in 0..k / 2 {
            // Low nibble first; & 0xF stores the two's-complement pattern.
            a_bytes[off + rr * (k / 2) + p] =
                (nibble(i, 2 * p) as u8 & 0x0F) | ((nibble(i, 2 * p + 1) as u8 & 0x0F) << 4);
        }
        for g in 0..groups_per_row {
            let bits = f32_to_bf16(scale(g));
            a_bytes[off + nibble_bytes + (rr * groups_per_row + g) * 2
                ..off + nibble_bytes + (rr * groups_per_row + g) * 2 + 2]
                .copy_from_slice(&bits.to_le_bytes());
        }
    }
    let x_bits: Vec<u16> = (0..k).map(|kk| f32_to_bf16(((kk % 7) as i32 - 3) as f32)).collect();
    let mut b_bytes = vec![0u8; k * 2];
    for (kk, b) in x_bits.iter().enumerate() {
        b_bytes[kk * 2..kk * 2 + 2].copy_from_slice(&b.to_le_bytes());
    }
    let c_zero = vec![0u8; m * 2];

    let mut live: Vec<(BufferObject, Mapping)> = Vec::new();
    let vas: Vec<u64> = [
        ("W", a_bytes.as_slice()),
        ("x", b_bytes.as_slice()),
        ("C", c_zero.as_slice()),
    ]
    .iter()
    .filter_map(|(label, data)| chain_tensor(&dev, &mut live, label, data))
    .collect();
    if vas.len() != 3 {
        eprintln!("tensor allocation failed");
        return ExitCode::FAILURE;
    }
    let [w_va, x_va, c_va] = [vas[0], vas[1], vas[2]];

    let mut op = match chain_op(&dev, "w4gemv", &instr, 0, &[w_va, x_va, c_va]) {
        Some(o) => o,
        None => {
            eprintln!("w4gemv op setup failed");
            return ExitCode::FAILURE;
        }
    };
    let mut handles = vec![op.ctrl_bo.handle()];
    handles.extend(live.iter().map(|(b, _)| b.handle()));

    // Host reference: exact f32 dot per row (see doc comment), one RNE
    // narrowing — must match the kernel bit-for-bit.
    let c_ref: Vec<u16> = (0..m)
        .map(|i| {
            let mut acc = 0f32;
            for kk in 0..k {
                let w = (nibble(i, kk) as f32) * scale(kk / group);
                acc += w * bf16_to_f32(x_bits[kk]);
            }
            f32_to_bf16(acc)
        })
        .collect();
    // Note: w = nibble × scale is exact in bf16 (dyadic scale, small int),
    // and the kernel multiplies bf16(nibble) × bf16(scale) in its mac — same
    // exact product, so the reference needs no intermediate rounding.

    // Warmup.
    let t0 = std::time::Instant::now();
    let seq = match op.pkt.submit(&dev, &ctx, &handles) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("warmup submit: {e}");
            return ExitCode::FAILURE;
        }
    };
    if syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000).is_err()
        || op.pkt.state() != ERT_CMD_STATE_COMPLETED
    {
        eprintln!("warmup did not complete");
        return ExitCode::FAILURE;
    }
    println!("warmup: {:?}", t0.elapsed());

    let check_c = |label: &str| -> usize {
        let c_bo = &live[2].0;
        let _ = c_bo.sync(SyncDirection::ToDevice, 0, c_bo.size() as u64);
        let s = live[2].1.as_slice();
        let mut mm = 0usize;
        let mut first: Option<(usize, u16, u16)> = None;
        for i in 0..m {
            let got = u16::from_le_bytes([s[i * 2], s[i * 2 + 1]]);
            if got != c_ref[i] {
                mm += 1;
                if first.is_none() {
                    first = Some((i, got, c_ref[i]));
                }
            }
        }
        match first {
            Some((i, got, want)) => println!(
                "{label}: first diff @[{i}]: got 0x{got:04x} ({}) want 0x{want:04x} ({})",
                bf16_to_f32(got),
                bf16_to_f32(want)
            ),
            None => println!("{label}: verify PASS ({mm}/{m} diffs)"),
        }
        mm
    };
    let mm = check_c("warmup");
    if mm != 0 {
        return ExitCode::FAILURE;
    }

    // Sequential: submit+wait per op (the R3 engine pattern).
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let seq = match op.pkt.submit(&dev, &ctx, &handles) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("sequential submit: {e}");
                return ExitCode::FAILURE;
            }
        };
        if syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000).is_err()
            || op.pkt.state() != ERT_CMD_STATE_COMPLETED
        {
            eprintln!("sequential exec did not complete");
            return ExitCode::FAILURE;
        }
    }
    let seq_elapsed = t0.elapsed();
    let mm_seq = check_c("sequential");

    // Pipelined: a pool of in-flight cmd BOs, submit all, drain by state
    // (same tensors, in-order queue → last result wins).
    let ctrl_addr = op.ctrl_bo.xdna_addr();
    let mut pkts: Vec<StartNpuCmd> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut p = match StartNpuCmd::new(&dev) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("cmd BO: {e}");
                return ExitCode::FAILURE;
            }
        };
        p.set_cu(0);
        let mut r = p
            .set_ctrl(ctrl_addr, instr.len() as u32)
            .and_then(|_| p.arg64(3))
            .and_then(|_| p.arg64(ctrl_addr))
            .and_then(|_| p.arg32(instr.len() as u32));
        for va in &[w_va, x_va, c_va] {
            r = r.and_then(|_| p.arg64(*va));
        }
        if let Err(e) = r {
            eprintln!("build packet: {e}");
            return ExitCode::FAILURE;
        }
        pkts.push(p);
    }
    let t0 = std::time::Instant::now();
    let mut submits = 0u32;
    for p in pkts.iter_mut() {
        if let Err(e) = p.submit(&dev, &ctx, &handles) {
            eprintln!("pipelined submit (depth {}): {e}", submits + 1);
            return ExitCode::FAILURE;
        }
        submits += 1;
    }
    let submit_elapsed = t0.elapsed();
    let t1 = std::time::Instant::now();
    let mut spins = 0u64;
    loop {
        if pkts.iter().all(|p| p.state() == ERT_CMD_STATE_COMPLETED) {
            break;
        }
        spins += 1;
        if spins > 4_000_000 {
            eprintln!("pipelined drain timed out");
            return ExitCode::FAILURE;
        }
        std::thread::sleep(std::time::Duration::from_micros(50));
    }
    let pipe_elapsed = t0.elapsed();
    let mm_pipe = check_c("pipelined");

    // Weight bytes streamed per op: nibbles + scales (the vector and output
    // are noise at this size) — the number that bounds decode throughput.
    let weight_bytes = a_bytes_len as f64;
    let seq_per = seq_elapsed.as_secs_f64() / iters as f64;
    let pipe_per = pipe_elapsed.as_secs_f64() / iters as f64;
    println!(
        "sequential: {:?}/op ({:.0} GB/s weight stream)",
        seq_elapsed / iters as u32,
        weight_bytes / seq_per / 1e9
    );
    println!(
        "pipelined:  {:?}/op total, submit loop {:?}, drain {:?} ({:.0} GB/s weight stream)",
        pipe_elapsed / iters as u32,
        submit_elapsed,
        t1.elapsed(),
        weight_bytes / pipe_per / 1e9
    );
    println!(
        "per-token projection (MiniCPM5 42 layers: qkv+gate_up+down = 1.80 G weights → 1.02 GB w4 incl. scales): {:.2} ms sequential / {:.2} ms pipelined",
        1.02e9 / (weight_bytes / seq_per) * 1e3,
        1.02e9 / (weight_bytes / pipe_per) * 1e3
    );
    if mm_seq == 0 && mm_pipe == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// One MiniCPM5 decode projection shape: fixture stem + packed geometry.
/// Mirrors tools/w4_import.py's SHAPES table (tsi per the 64 KB tile-memory
/// budget; see notes §13).
struct W4Shape {
    name: &'static str,
    m: usize,
    k: usize,
    tsi: usize,
    fixture: &'static str,
}

const W4_SHAPES: [W4Shape; 4] = [
    W4Shape { name: "qkv", m: 2560, k: 2048, tsi: 16, fixture: "w4gemv2_2560x2048_16tsi_320tso_8col_g32" },
    W4Shape { name: "o", m: 2048, k: 2048, tsi: 16, fixture: "w4gemv2_2048x2048_16tsi_256tso_8col_g32" },
    W4Shape { name: "gateup", m: 12288, k: 2048, tsi: 16, fixture: "w4gemv2_12288x2048_16tsi_1536tso_8col_g32" },
    W4Shape { name: "down", m: 2048, k: 6144, tsi: 4, fixture: "w4gemv2_2048x6144_4tsi_256tso_8col_g32" },
];

/// Parse one golden_{shape}.bin written by the importer:
/// u32 nrows, u32 K, rows u32[nrows], x bits u16[K], ref bits u16[nrows].
fn read_golden(path: &std::path::Path) -> Option<(Vec<usize>, Vec<u16>, Vec<u16>)> {
    let d = std::fs::read(path).ok()?;
    let rd_u32 = |o: usize| u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) as usize;
    let nrows = rd_u32(0);
    let k = rd_u32(4);
    if d.len() != 8 + 4 * nrows + 2 * k + 2 * nrows {
        return None;
    }
    let rows: Vec<usize> = (0..nrows).map(|i| rd_u32(8 + 4 * i)).collect();
    let rd_u16 = |o: usize| u16::from_le_bytes([d[o], d[o + 1]]);
    let x = (0..k).map(|i| rd_u16(8 + 4 * nrows + 2 * i)).collect();
    let ref_bits = (0..nrows)
        .map(|i| rd_u16(8 + 4 * nrows + 2 * k + 2 * i))
        .collect();
    Some((rows, x, ref_bits))
}

/// M3a: one full decode step's projection chain over REAL w4-imported
/// weights — 42 layers x 4 GEMV shapes (qkv, o, gate_up, down) on 4 CUs of
/// one context. The real dataflow has mha/swiglu between the GEMVs; here
/// every GEMV reads a fixed activation, so the run measures exactly what
/// the engine's projection stream costs (weight bandwidth + CU switching),
/// which the four scheduling modes slice differently:
///   per-op    — submit+wait each op (the R3 pattern, worst case);
///   per-layer — 4 submits per layer, drain at layer end;
///   pipelined — all 168 submits, one drain (graph-executor target);
///   grouped   — same-CU runs batched (42 qkv, then 42 o, ...): breaks the
///               layer dependency on paper, quantifies the PDI-reload cost
///               the layer-order modes pay on every CU change.
fn cmd_run_w4layer(w4dir: &str, nlayers: usize, iters: usize) -> ExitCode {
    let build = "/home/nzinfo/qwen/xnpu/build";
    println!(
        "w4 layer chain: {nlayers} layers x 4 GEMV (qkv/o/gateup/down), {iters} iters/mode, weights {w4dir}"
    );

    // Fixtures: PDI + ctrl-code per shape.
    let fixtures: Vec<(Vec<u8>, Vec<u8>, u32)> = W4_SHAPES
        .iter()
        .map(|s| match load_fixture(&format!("{build}/{}.mlir.prj", s.fixture)) {
            Some(f) => f,
            None => {
                eprintln!("load fixture {} failed", s.fixture);
                std::process::exit(2);
            }
        })
        .collect();
    for (s, f) in W4_SHAPES.iter().zip(&fixtures) {
        println!(
            "  {:>7}: pdi {} B, ctrl {} B (M={}, K={}, tsi={})",
            s.name, f.0.len(), f.1.len(), s.m, s.k, s.tsi
        );
    }

    let dev = match Device::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open amdxdna device: {e}");
            return ExitCode::FAILURE;
        }
    };
    let md = match dev.aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("AIE metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let cols = fixtures.iter().map(|f| f.2).max().unwrap_or(8);
    let num_tiles = cols * md.core.row_count as u32;
    let mut ctx = match HwContext::create(&dev, num_tiles) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    let pdis: Vec<(&[u8], u8)> = fixtures.iter().map(|f| (f.0.as_slice(), 0)).collect();
    if let Err(e) = ctx.configure_cus(&pdis) {
        eprintln!("configure_cus (4 PDIs): {e}");
        return ExitCode::FAILURE;
    }
    println!("4 CUs attached (0=qkv 1=o 2=gateup 3=down), {} cols / {} tiles", cols, num_tiles);

    // Activations (fixed) and outputs. x is shared per K; c is per shape.
    let mut live: Vec<(BufferObject, Mapping)> = Vec::new();
    let mut x_va: [u64; 2] = [0; 2]; // [K=2048, K=6144]
    let x2048 = vec![0u8; 2048 * 2];
    let x6144 = vec![0u8; 6144 * 2];
    x_va[0] = match chain_tensor(&dev, &mut live, "x2048", &x2048) {
        Some(v) => v,
        None => {
            eprintln!("x2048 BO failed");
            return ExitCode::FAILURE;
        }
    };
    x_va[1] = match chain_tensor(&dev, &mut live, "x6144", &x6144) {
        Some(v) => v,
        None => {
            eprintln!("x6144 BO failed");
            return ExitCode::FAILURE;
        }
    };
    let mut c_va = [0u64; 4];
    for (si, s) in W4_SHAPES.iter().enumerate() {
        c_va[si] = match chain_tensor(&dev, &mut live, &format!("c_{}", s.name), &vec![0u8; s.m * 2]) {
            Some(v) => v,
            None => {
                eprintln!("c_{} BO failed", s.name);
                return ExitCode::FAILURE;
            }
        };
    }

    // Real weights: one SHMEM BO per (layer, shape), all preloaded so no
    // host copy lands inside the timed loops. Ctrl BOs are shared per
    // shape (the ctrl code is read-only; run-pipe validated the pattern).
    let mut w_va = vec![0u64; nlayers * 4];
    let mut total_w = 0usize;
    for n in 0..nlayers {
        for (si, s) in W4_SHAPES.iter().enumerate() {
            let data = match std::fs::read(format!("{w4dir}/layer{n:02}_{}.bin", s.name)) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("read layer{n:02}_{}: {e}", s.name);
                    return ExitCode::FAILURE;
                }
            };
            let expect = 8 * (s.m / 8) * (s.tsi * (s.k / 2) + s.tsi * (s.k / 32) * 2) / s.tsi;
            if data.len() != expect {
                eprintln!(
                    "layer{n:02}_{}: {} B, expected {expect} B (stale import?)",
                    s.name,
                    data.len()
                );
                return ExitCode::FAILURE;
            }
            total_w += data.len();
            w_va[n * 4 + si] = match chain_tensor(&dev, &mut live, &format!("w{n:02}.{}", s.name), &data) {
                Some(v) => v,
                None => {
                    eprintln!("w{n:02}.{} BO failed", s.name);
                    return ExitCode::FAILURE;
                }
            };
        }
    }
    println!("weights resident: {total_w}/1e6 MB in {} BOs", live.len() - 6);

    let mut ops: Vec<ChainOp> = Vec::with_capacity(nlayers * 4);
    for n in 0..nlayers {
        for (si, s) in W4_SHAPES.iter().enumerate() {
            let xv = x_va[if s.k == 6144 { 1 } else { 0 }];
            let op = chain_op(
                &dev,
                &format!("{}L{n:02}", s.name),
                &fixtures[si].1,
                si as u32,
                &[w_va[n * 4 + si], xv, c_va[si]],
            );
            match op {
                Some(o) => ops.push(o),
                None => {
                    eprintln!("op setup layer{n:02} {} failed", s.name);
                    return ExitCode::FAILURE;
                }
            }
        }
    }
    // Per-op arg handles: [ctrl(shape), w(layer,shape), x, c]. live layout:
    // [x2048, x6144, c_qkv, c_o, c_gateup, c_down, w00.qkv, w00.o, ...].
    let x_hdl = [live[0].0.handle(), live[1].0.handle()];
    let c_hdl: Vec<u32> = (2..6).map(|i| live[i].0.handle()).collect();
    let mut op_handles: Vec<Vec<u32>> = Vec::with_capacity(ops.len());
    for n in 0..nlayers {
        for si in 0..4 {
            let wi = 6 + n * 4 + si;
            op_handles.push(vec![
                ops[n * 4 + si].ctrl_bo.handle(),
                live[wi].0.handle(),
                x_hdl[if W4_SHAPES[si].k == 6144 { 1 } else { 0 }],
                c_hdl[si],
            ]);
        }
    }

    // Warmup + golden verification on layer 0 (real weights, bf16 output,
    // f32 accumulation order differs -> tolerance tier, not bit-exact).
    let mut golden_ok = true;
    {
        for (si, s) in W4_SHAPES.iter().enumerate() {
            let (rows, x_bits, ref_bits) = match read_golden(std::path::Path::new(&format!(
                "{w4dir}/golden_{}.bin", s.name
            ))) {
                Some(g) => g,
                None => {
                    eprintln!("read golden_{} failed", s.name);
                    return ExitCode::FAILURE;
                }
            };
            {
                let xi = if s.k == 6144 { 1 } else { 0 };
                let (bo, map) = &mut live[xi];
                for (i, b) in x_bits.iter().enumerate() {
                    map.as_mut_slice()[i * 2..i * 2 + 2].copy_from_slice(&b.to_le_bytes());
                }
                bo.sync(SyncDirection::ToDevice, 0, bo.size() as u64).ok();
            }
            let op = &mut ops[si];
            let seq = match op.pkt.submit(&dev, &ctx, &op_handles[si]) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("golden submit {}: {e}", s.name);
                    return ExitCode::FAILURE;
                }
            };
            if syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000).is_err()
                || op.pkt.state() != ERT_CMD_STATE_COMPLETED
            {
                eprintln!("golden exec {} did not complete", s.name);
                return ExitCode::FAILURE;
            }
            let (c_bo, c_map) = &live[2 + si];
            let _ = c_bo.sync(SyncDirection::ToDevice, 0, c_bo.size() as u64);
            let cs = c_map.as_slice();
            let mut worst = 0f32;
            let mut bad = 0usize;
            for (ri, row) in rows.iter().enumerate() {
                let got = u16::from_le_bytes([cs[row * 2], cs[row * 2 + 1]]);
                let g = bf16_to_f32(got);
                let w = bf16_to_f32(ref_bits[ri]);
                let err = (g - w).abs();
                if err > 0.01 + 0.01 * w.abs() {
                    bad += 1;
                }
                worst = worst.max(err / (w.abs() + 1e-6));
            }
            println!(
                "  golden {:>7}: {} bad rows (worst rel err {:.2e}) -> {}",
                s.name,
                bad,
                worst,
                if bad == 0 { "PASS" } else { "FAIL" }
            );
            golden_ok &= bad == 0;
        }
    }
    if !golden_ok {
        eprintln!("GOLDEN: FAIL — real-weight outputs disagree with the importer reference");
        return ExitCode::FAILURE;
    }
    println!("GOLDEN: PASS — real-weight GEMV outputs match the importer reference (bf16 tolerance)");

    // Timed scheduling modes. op index = layer*4 + shape.
    let nops = nlayers * 4;
    let wait_op = |ops: &mut [ChainOp], i: usize| -> bool {
        let op = &mut ops[i];
        let seq = match op.pkt.submit(&dev, &ctx, &op_handles[i]) {
            Ok(s) => s,
            Err(_) => return false,
        };
        syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000).is_ok()
            && op.pkt.state() == ERT_CMD_STATE_COMPLETED
    };
    let drain = |ops: &[ChainOp]| -> bool {
        let mut polls = 0u64;
        loop {
            if ops.iter().all(|o| o.pkt.state() == ERT_CMD_STATE_COMPLETED) {
                return true;
            }
            polls += 1;
            if polls > 4_000_000 {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    };

    let report = |label: &str, total: std::time::Duration| {
        println!(
            "  {label:>10}: {:>10.2?} /token ({:.1} tok/s, {:.1} GB/s weight stream)",
            total / iters as u32,
            1e3 / (total / iters as u32).as_secs_f64() / 1e3,
            total_w as f64 / (total.as_secs_f64() / iters as f64) / 1e9
        );
    };

    // per-op sync
    let mut all_ok = true;
    let mut t0 = std::time::Instant::now();
    for _ in 0..iters {
        for i in 0..nops {
            all_ok &= wait_op(&mut ops, i);
        }
    }
    if !all_ok {
        eprintln!("per-op mode failure");
        return ExitCode::FAILURE;
    }
    report("per-op", t0.elapsed());

    // per-layer: 4 submits, drain, next layer.
    t0 = std::time::Instant::now();
    for _ in 0..iters {
        for n in 0..nlayers {
            for j in 0..4 {
                let i = n * 4 + j;
                if ops[i].pkt.submit(&dev, &ctx, &op_handles[i]).is_err() {
                    eprintln!("per-layer submit failed");
                    return ExitCode::FAILURE;
                }
            }
            if !drain(&ops[n * 4..n * 4 + 4]) {
                eprintln!("per-layer drain timeout");
                return ExitCode::FAILURE;
            }
        }
    }
    report("per-layer", t0.elapsed());

    // pipelined: all submits, one drain.
    t0 = std::time::Instant::now();
    for _ in 0..iters {
        for i in 0..nops {
            if ops[i].pkt.submit(&dev, &ctx, &op_handles[i]).is_err() {
                eprintln!("pipelined submit failed");
                return ExitCode::FAILURE;
            }
        }
        if !drain(&ops) {
            eprintln!("pipelined drain timeout");
            return ExitCode::FAILURE;
        }
    }
    report("pipelined", t0.elapsed());

    // grouped: all layers' ops of shape 0, then shape 1, ... one drain.
    t0 = std::time::Instant::now();
    for _ in 0..iters {
        for si in 0..4 {
            for n in 0..nlayers {
                let i = n * 4 + si;
                if ops[i].pkt.submit(&dev, &ctx, &op_handles[i]).is_err() {
                    eprintln!("grouped submit failed");
                    return ExitCode::FAILURE;
                }
            }
        }
        if !drain(&ops) {
            eprintln!("grouped drain timeout");
            return ExitCode::FAILURE;
        }
    }
    report("grouped", t0.elapsed());

    ExitCode::SUCCESS
}

/// One MiniCPM5 decode projection shape for the UNIVERSAL w4gemvu kernel:
/// fixture stem + B-stream geometry. Mirrors IRON w4gemvu/op.py (notes §14).
struct W4UShape {
    name: &'static str,
    m: usize,
    k: usize,
    /// B fifo elements = tiles_per_col / 16 (tiles_per_col = M / 32): the
    /// multi-element B fill that keeps shim BD usage at one per channel.
    f: usize,
}

const W4U_SHAPES: [W4UShape; 4] = [
    W4UShape { name: "qkv", m: 2560, k: 2048, f: 5 },
    W4UShape { name: "o", m: 2048, k: 2048, f: 4 },
    W4UShape { name: "gateup", m: 12288, k: 2048, f: 24 },
    W4UShape { name: "down", m: 2048, k: 6144, f: 4 },
];

/// Decode-chain architecture profile (run-decode): everything the glue
/// needs beyond the shared w4gemvu shapes. hy rope base = theta·alpha^
/// (d/(d−2)) — FLM's "dynamic" scaling is a STATIC alpha rescale of the
/// base (libhunyuan_npu.so disasm; beta_fast/beta_slow never parsed).
struct DecArch {
    name: &'static str,
    layers: usize,
    heads: usize,
    kv: usize,
    qk_norm: bool,
    rope_base: f32,
    /// qkv fused shape: (M, F). o/gateup/down are shared with MiniCPM5.
    qkv_m: usize,
    qkv_f: usize,
    /// flowkv_decode fixture stem (per-arch compiled KV-head geometry).
    fk_fixture: &'static str,
    decdir: &'static str,
    w4dir: &'static str,
}

const DEC_MINICPM: DecArch = DecArch {
    name: "minicpm",
    layers: 42,
    heads: 16,
    kv: 2,
    qk_norm: false,
    rope_base: 5e6,
    qkv_m: 2560,
    qkv_f: 5,
    fk_fixture: "flowkv_decode_16h_2kv_128d_1024s_32cs_2col",
    decdir: "/home/nzinfo/qwen/xnpu/build/dec",
    w4dir: "/home/nzinfo/qwen/xnpu/build/w4u",
};

const DEC_HY: DecArch = DecArch {
    name: "hy-mt2",
    layers: 32,
    heads: 16,
    kv: 4,
    qk_norm: true, // per-head q/k rms AFTER rope (hunyuan_npu.hpp)
    // theta·alpha^(d/(d−2)) = 10000·1000^(128/126) = 11158839.925 — f32
    // rounding shifts it 7e-9 relative, far below rope-angle noise.
    rope_base: 11158840.0,
    qkv_m: 3072, // cat(q 2048, k 512, v 512), 16Q/4KV GQA
    qkv_f: 6,
    fk_fixture: "flowkv_decode_16h_4kv_128d_1024s_32cs_4col",
    decdir: "/home/nzinfo/qwen/xnpu/build/dec_hy",
    w4dir: "/home/nzinfo/qwen/xnpu/build/w4u_hy",
};

impl DecArch {
    fn shapes(&self) -> [W4UShape; 4] {
        [
            W4UShape { name: "qkv", m: self.qkv_m, k: 2048, f: self.qkv_f },
            W4UShape { name: "o", m: 2048, k: 2048, f: 4 },
            W4UShape { name: "gateup", m: 12288, k: 2048, f: 24 },
            W4UShape { name: "down", m: 2048, k: 6144, f: 4 },
        ]
    }
}

/// One padded max-K layout-v2 slot (nibbles + hole + scales + K at the tail).
const W4U_ELEM: usize = 13840;
const W4U_K_MAX: usize = 6144;

/// M3b: the same 42-layer projection chain on the UNIVERSAL w4gemvu kernel.
/// All four shapes share one PDI (the kernel reads K from the slot tail at
/// runtime), so the whole chain runs on ONE CU and differs only in ctrl
/// code — the ~650us-per-switch PDI reload run-w4layer pays 168 times per
/// token is gone BY CONSTRUCTION, not by scheduling. Differences from v1:
///   - fixture stems w4gemvu_{M}x{K}; the four PDIs must be byte-identical
///     (asserted — that identity is the entire premise of the single CU);
///   - packed layout v2: 8 cols x (M/32) x 13840-byte slots;
///   - the vector buffer is the multi-element B stream: F K_MAX-wide
///     slots each holding x zero-padded — two shared x BOs (xu2048 with
///     24 slots serves every F in {5,4,24}; xu6144 with 4 slots serves down).
/// Expected convergence: all four scheduling modes at the device floor
/// (~2.3 ms/layer), where v1 spanned 75-234 ms/token on scheduling alone.
fn cmd_run_w4ulayer(w4dir: &str, nlayers: usize, iters: usize) -> ExitCode {
    let build = "/home/nzinfo/qwen/xnpu/build";
    println!(
        "w4 UNIVERSAL layer chain: {nlayers} layers x 4 GEMV on ONE CU, {iters} iters/mode, weights {w4dir}"
    );

    // Fixtures: per-shape ctrl code, one shared PDI.
    let fixtures: Vec<(Vec<u8>, Vec<u8>, u32)> = W4U_SHAPES
        .iter()
        .map(|s| {
            match load_fixture(&format!("{build}/w4gemvu_{}x{}.mlir.prj", s.m, s.k)) {
                Some(f) => f,
                None => {
                    eprintln!(
                        "load fixture w4gemvu_{}x{} failed (run the w4gemvu pytest first)",
                        s.m, s.k
                    );
                    std::process::exit(2);
                }
            }
        })
        .collect();
    for (s, f) in W4U_SHAPES.iter().zip(&fixtures) {
        println!(
            "  {:>7}: ctrl {} B, {} cols (M={}, K={}, F={})",
            s.name, f.1.len(), f.2, s.m, s.k, s.f
        );
    }
    for (s, f) in W4U_SHAPES.iter().zip(&fixtures).skip(1) {
        if f.0 != fixtures[0].0 {
            eprintln!(
                "PDI for {} differs from {} — stale fixtures, rebuild",
                s.name,
                W4U_SHAPES[0].name
            );
            return ExitCode::FAILURE;
        }
    }
    println!(
        "PDI identical across all 4 shapes ({} B) — universal kernel confirmed",
        fixtures[0].0.len()
    );

    let dev = match Device::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open amdxdna device: {e}");
            return ExitCode::FAILURE;
        }
    };
    let md = match dev.aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("AIE metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let cols = fixtures.iter().map(|f| f.2).max().unwrap_or(8);
    let num_tiles = cols * md.core.row_count as u32;
    let mut ctx = match HwContext::create(&dev, num_tiles) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = ctx.configure_cus(&[(fixtures[0].0.as_slice(), 0)]) {
        eprintln!("configure_cus (1 PDI): {e}");
        return ExitCode::FAILURE;
    }
    println!("1 CU attached (cu0 = w4gemvu universal), {} cols / {} tiles", cols, num_tiles);

    // Replicated activation BOs: xu2048 (24 slots) serves every K=2048 shape
    // (each op's B fill reads an F-slot prefix); xu6144 (4 slots) serves down.
    let mut live: Vec<(BufferObject, Mapping)> = Vec::new();
    let mut x_va: [u64; 2] = [0; 2]; // [xu2048 (24 slots), xu6144 (4 slots)]
    let xu2048 = vec![0u8; 24 * W4U_K_MAX * 2];
    let xu6144 = vec![0u8; 4 * W4U_K_MAX * 2];
    x_va[0] = match chain_tensor(&dev, &mut live, "xu2048", &xu2048) {
        Some(v) => v,
        None => {
            eprintln!("xu2048 BO failed");
            return ExitCode::FAILURE;
        }
    };
    x_va[1] = match chain_tensor(&dev, &mut live, "xu6144", &xu6144) {
        Some(v) => v,
        None => {
            eprintln!("xu6144 BO failed");
            return ExitCode::FAILURE;
        }
    };
    let mut c_va = [0u64; 4];
    for (si, s) in W4U_SHAPES.iter().enumerate() {
        c_va[si] = match chain_tensor(&dev, &mut live, &format!("c_{}", s.name), &vec![0u8; s.m * 2]) {
            Some(v) => v,
            None => {
                eprintln!("c_{} BO failed", s.name);
                return ExitCode::FAILURE;
            }
        };
    }

    // Real weights, layout v2: one SHMEM BO per (layer, shape), all preloaded.
    let mut w_va = vec![0u64; nlayers * 4];
    let mut total_w = 0usize;
    for n in 0..nlayers {
        for (si, s) in W4U_SHAPES.iter().enumerate() {
            let data = match std::fs::read(format!("{w4dir}/layer{n:02}_{}.bin", s.name)) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("read layer{n:02}_{}: {e}", s.name);
                    return ExitCode::FAILURE;
                }
            };
            let expect = 8 * (s.m / 32) * W4U_ELEM;
            if data.len() != expect {
                eprintln!(
                    "layer{n:02}_{}: {} B, expected {expect} B (stale import? use --layout v2)",
                    s.name,
                    data.len()
                );
                return ExitCode::FAILURE;
            }
            total_w += data.len();
            w_va[n * 4 + si] = match chain_tensor(&dev, &mut live, &format!("w{n:02}.{}", s.name), &data) {
                Some(v) => v,
                None => {
                    eprintln!("w{n:02}.{} BO failed", s.name);
                    return ExitCode::FAILURE;
                }
            };
        }
    }
    let useful_w: usize = W4U_SHAPES
        .iter()
        .map(|s| s.m * s.k / 2 + s.m * (s.k / 32) * 2)
        .sum::<usize>()
        * nlayers;
    println!(
        "weights resident: {:.2} GB slot stream / {:.2} GB useful in {} BOs",
        total_w as f64 / 1e9,
        useful_w as f64 / 1e9,
        live.len() - 6
    );

    // M5a xnpu-perf: per-shape 字节/FLOP 计数（roofline 分母来源）+ 全程
    // 时间线记录器。metas 按形状聚合（跨层共享），event 名 = 形状名。
    let metas: Vec<OpMeta> = W4U_SHAPES
        .iter()
        .map(|s| {
            let mut m = OpMeta::new(
                s.name,
                "w4gemvu",
                0,
                (s.m * s.k / 2 + s.m * (s.k / 32) * 2 + s.k * 2) as u64, // w4 + scales + x
                (s.m * 2) as u64, // c out
                (2 * s.m * s.k) as u64,
            );
            m.bytes_stream = Some((8 * (s.m / 32) * W4U_ELEM) as u64); // padded slot 流
            m
        })
        .collect();
    let mut rec = Recorder::new();
    let mut rec_seq = 0u64;

    let mut ops: Vec<ChainOp> = Vec::with_capacity(nlayers * 4);
    for n in 0..nlayers {
        for (si, s) in W4U_SHAPES.iter().enumerate() {
            let xv = x_va[if s.k == 6144 { 1 } else { 0 }];
            let op = chain_op(
                &dev,
                &format!("{}L{n:02}", s.name),
                &fixtures[si].1,
                0, // ONE CU for every shape — that is the whole point.
                &[w_va[n * 4 + si], xv, c_va[si]],
            );
            match op {
                Some(o) => ops.push(o),
                None => {
                    eprintln!("op setup layer{n:02} {} failed", s.name);
                    return ExitCode::FAILURE;
                }
            }
        }
    }
    // Per-op arg handles: [ctrl(shape), w(layer,shape), x, c]. live layout:
    // [xu2048, xu6144, c_qkv, c_o, c_gateup, c_down, w00.qkv, w00.o, ...].
    let x_hdl = [live[0].0.handle(), live[1].0.handle()];
    let c_hdl: Vec<u32> = (2..6).map(|i| live[i].0.handle()).collect();
    let mut op_handles: Vec<Vec<u32>> = Vec::with_capacity(ops.len());
    for n in 0..nlayers {
        for si in 0..4 {
            let wi = 6 + n * 4 + si;
            op_handles.push(vec![
                ops[n * 4 + si].ctrl_bo.handle(),
                live[wi].0.handle(),
                x_hdl[if W4U_SHAPES[si].k == 6144 { 1 } else { 0 }],
                c_hdl[si],
            ]);
        }
    }

    // Warmup + golden verification on layer 0 (real weights). The x BOs get
    // the golden activations replicated F-slots-wide — a bare (K,) write
    // would leave slots 2..F reading the creation-time zeros. Goldens are
    // preloaded for EVERY layer: the scheduling modes below can then verify
    // a layer's outputs right after they drain, before later layers
    // overwrite the shared c buffers — the deep-queue modes are only
    // meaningful if their execution is actually checked.
    let mut goldens: Vec<(Vec<usize>, Vec<u16>, Vec<u16>)> = Vec::with_capacity(nlayers * 4);
    for n in 0..nlayers {
        for s in W4U_SHAPES.iter() {
            match read_golden(std::path::Path::new(&format!(
                "{w4dir}/golden_L{n:02}_{}.bin",
                s.name
            ))) {
                Some(g) => goldens.push(g),
                None => {
                    eprintln!("read golden_L{n:02}_{} failed", s.name);
                    return ExitCode::FAILURE;
                }
            }
        }
    }
    for (xi, slots) in [(0usize, 24usize), (1usize, 4usize)] {
        let x_bits = &goldens[if xi == 0 { 0 } else { 3 }].1; // qkv / down
        let (bo, map) = &mut live[xi];
        let bytes = map.as_mut_slice();
        for s in 0..slots {
            let base = s * W4U_K_MAX * 2;
            for (i, b) in x_bits.iter().enumerate() {
                bytes[base + i * 2..base + i * 2 + 2].copy_from_slice(&b.to_le_bytes());
            }
        }
        bo.sync(SyncDirection::ToDevice, 0, bo.size() as u64).ok();
    }
    let mut golden_ok = true;
    for (si, s) in W4U_SHAPES.iter().enumerate() {
        let (rows, _x_bits, ref_bits) = &goldens[si];
        let op = &mut ops[si];
        let seq = match op.pkt.submit(&dev, &ctx, &op_handles[si]) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("golden submit {}: {e}", s.name);
                return ExitCode::FAILURE;
            }
        };
        if syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000).is_err()
            || op.pkt.state() != ERT_CMD_STATE_COMPLETED
        {
            eprintln!("golden exec {} did not complete", s.name);
            return ExitCode::FAILURE;
        }
        let (c_bo, c_map) = &live[2 + si];
        let _ = c_bo.sync(SyncDirection::ToDevice, 0, c_bo.size() as u64);
        let cs = c_map.as_slice();
        let mut worst = 0f32;
        let mut bad = 0usize;
        for (ri, row) in rows.iter().enumerate() {
            let got = u16::from_le_bytes([cs[row * 2], cs[row * 2 + 1]]);
            let g = bf16_to_f32(got);
            let w = bf16_to_f32(ref_bits[ri]);
            let err = (g - w).abs();
            if err > 0.01 + 0.01 * w.abs() {
                bad += 1;
            }
            worst = worst.max(err / (w.abs() + 1e-6));
        }
        println!(
            "  golden {:>7}: {} bad rows (worst rel err {:.2e}) -> {}",
            s.name,
            bad,
            worst,
            if bad == 0 { "PASS" } else { "FAIL" }
        );
        golden_ok &= bad == 0;
    }
    if !golden_ok {
        eprintln!("GOLDEN: FAIL — real-weight outputs disagree with the importer reference");
        return ExitCode::FAILURE;
    }
    println!("GOLDEN: PASS — universal CU real-weight outputs match (bf16 tolerance)");

    // Spot-verify layer n's four outputs against the importer goldens.
    // MUST run while no later layer has overwritten the c buffers.
    let verify_layer = |n: usize| -> bool {
        let mut ok = true;
        for si in 0..4 {
            let (rows, _x_bits, ref_bits) = &goldens[n * 4 + si];
            let (c_bo, c_map) = &live[2 + si];
            let _ = c_bo.sync(SyncDirection::FromDevice, 0, c_bo.size() as u64);
            let cs = c_map.as_slice();
            for (ri, row) in rows.iter().enumerate() {
                let got = u16::from_le_bytes([cs[row * 2], cs[row * 2 + 1]]);
                let g = bf16_to_f32(got);
                let w = bf16_to_f32(ref_bits[ri]);
                if (g - w).abs() > 0.01 + 0.01 * w.abs() {
                    ok = false;
                }
            }
        }
        ok
    };

    // Timed scheduling modes. op index = layer*4 + shape; cu never changes.
    let nops = nlayers * 4;
    let wait_op = |ops: &mut [ChainOp], i: usize| -> bool {
        let op = &mut ops[i];
        let seq = match op.pkt.submit(&dev, &ctx, &op_handles[i]) {
            Ok(s) => s,
            Err(_) => return false,
        };
        syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000).is_ok()
            && op.pkt.state() == ERT_CMD_STATE_COMPLETED
    };
    let drain = |ops: &[ChainOp]| -> bool {
        let mut polls = 0u64;
        loop {
            if ops.iter().all(|o| o.pkt.state() == ERT_CMD_STATE_COMPLETED) {
                return true;
            }
            polls += 1;
            if polls > 4_000_000 {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    };
    // On failure, identify what the c buffer actually holds: another
    // layer's golden (reordered/stale output) or garbage (interleaved
    // ctrl-code execution corrupting the run).
    let diagnose = |n: usize| -> String {
        let mut parts = Vec::new();
        for si in 0..4 {
            let (c_bo, c_map) = &live[2 + si];
            let _ = c_bo.sync(SyncDirection::FromDevice, 0, c_bo.size() as u64);
            let cs = c_map.as_slice();
            let rd = |row: usize| u16::from_le_bytes([cs[row * 2], cs[row * 2 + 1]]);
            let mut hits: Vec<usize> = Vec::new();
            for m in 0..nlayers {
                let (rows, _x, ref_bits) = &goldens[m * 4 + si];
                if rows.iter().zip(ref_bits).all(|(row, rb)| {
                    let g = bf16_to_f32(rd(*row));
                    let w = bf16_to_f32(*rb);
                    (g - w).abs() <= 0.01 + 0.01 * w.abs()
                }) {
                    hits.push(m);
                }
            }
            let name = W4U_SHAPES[si].name;
            parts.push(match hits.first() {
                Some(&m) if m != n => format!("{name}=L{m:02}"),
                Some(_) => format!("{name}=ok"),
                None => {
                    let (rows, _x, ref_bits) = &goldens[n * 4 + si];
                    let mut worst = 0f32;
                    for (row, rb) in rows.iter().zip(ref_bits) {
                        worst = worst
                            .max((bf16_to_f32(rd(*row)) - bf16_to_f32(*rb)).abs());
                    }
                    format!("{name}=garbage(worst {worst:.1})")
                }
            });
        }
        parts.join(" ")
    };
    let report = |label: &str, total: std::time::Duration| {
        let per = total / iters as u32;
        println!(
            "  {label:>10}: {:>10.2?} /token ({:.1} tok/s, {:.1} GB/s useful / {:.1} GB/s slot stream)",
            per,
            1e3 / per.as_secs_f64() / 1e3,
            useful_w as f64 / per.as_secs_f64() / 1e9,
            total_w as f64 / per.as_secs_f64() / 1e9
        );
    };

    // per-op sync (verified at each layer boundary)
    let mut all_ok = true;
    let mut t0 = std::time::Instant::now();
    for it in 0..iters {
        for n in 0..nlayers {
            for j in 0..4 {
                let ts = std::time::Instant::now();
                all_ok &= wait_op(&mut ops, n * 4 + j);
                rec.solo(&metas[j], it as u32, ts, rec_seq);
                rec_seq += 1;
            }
            all_ok &= verify_layer(n);
        }
    }
    if !all_ok {
        eprintln!("per-op mode failure (exec or verification)");
        return ExitCode::FAILURE;
    }
    report("per-op", t0.elapsed());

    // per-layer: 4 submits, wait on the batch's LAST syncobj point, verify.
    // State-poll drains return before the final host writes are visible —
    // the syncobj timeline is the trustworthy completion signal (per-op mode
    // with syncobj waits verified 126/126; state drains showed write-
    // visibility lag failures that a 2 ms retry cleared).
    t0 = std::time::Instant::now();
    let mut pl_bad: Vec<String> = Vec::new();
    for it in 0..iters {
        let tb = std::time::Instant::now();
        for n in 0..nlayers {
            let mut last_seq = 0u64;
            for j in 0..4 {
                let i = n * 4 + j;
                match ops[i].pkt.submit(&dev, &ctx, &op_handles[i]) {
                    Ok(s) => {
                        last_seq = s;
                        rec.burst_submit(&metas[j], it as u32, rec_seq);
                        rec_seq += 1;
                    }
                    Err(_) => {
                        eprintln!("per-layer submit failed");
                        return ExitCode::FAILURE;
                    }
                }
            }
            if syncobj_timeline_wait(&dev, ctx.syncobj_handle, last_seq, 10_000_000_000)
                .is_err()
            {
                eprintln!("per-layer wait timeout");
                return ExitCode::FAILURE;
            }
            if !verify_layer(n) {
                pl_bad.push(format!("{it}/{n:02}:{}", diagnose(n)));
            }
        }
        rec.burst_done("per-layer", Mode::Burst, it as u32, tb, nops as u32, 1);
    }
    report("per-layer", t0.elapsed());
    println!(
        "    per-layer verified layers: {}/{} bad {}",
        iters * nlayers - pl_bad.len(),
        iters * nlayers,
        if pl_bad.is_empty() {
            "-".to_string()
        } else {
            pl_bad.join(",")
        }
    );

    // pipelined: all submits, one drain. Intermediate outputs are
    // overwritten in flight, so only the final layer is verifiable.
    t0 = std::time::Instant::now();
    for it in 0..iters {
        let tb = std::time::Instant::now();
        for i in 0..nops {
            match ops[i].pkt.submit(&dev, &ctx, &op_handles[i]) {
                Ok(_) => {
                    rec.burst_submit(&metas[i % 4], it as u32, rec_seq);
                    rec_seq += 1;
                }
                Err(_) => {
                    eprintln!("pipelined submit failed");
                    return ExitCode::FAILURE;
                }
            }
        }
        if !drain(&ops) {
            eprintln!("pipelined drain timeout");
            return ExitCode::FAILURE;
        }
        rec.burst_done("pipelined", Mode::Burst, it as u32, tb, nops as u32, 1);
    }
    report("pipelined", t0.elapsed());
    if !verify_layer(nlayers - 1) {
        println!("    pipelined final-layer check: FAIL — {}", diagnose(nlayers - 1));
    } else {
        println!("    pipelined final-layer check: PASS");
    }

    // grouped: same-CU batching is now the SAME order as per-layer — kept to
    // confirm the two coincide (zero switches means order no longer matters).
    t0 = std::time::Instant::now();
    for it in 0..iters {
        let tb = std::time::Instant::now();
        for si in 0..4 {
            for n in 0..nlayers {
                let i = n * 4 + si;
                match ops[i].pkt.submit(&dev, &ctx, &op_handles[i]) {
                    Ok(_) => {
                        rec.burst_submit(&metas[si], it as u32, rec_seq);
                        rec_seq += 1;
                    }
                    Err(_) => {
                        eprintln!("grouped submit failed");
                        return ExitCode::FAILURE;
                    }
                }
            }
        }
        if !drain(&ops) {
            eprintln!("grouped drain timeout");
            return ExitCode::FAILURE;
        }
        rec.burst_done("grouped", Mode::Burst, it as u32, tb, nops as u32, 1);
    }
    report("grouped", t0.elapsed());

    // Queue-depth sweep: submit in N-op chunks, wait each batch out on the
    // syncobj timeline — the only trustworthy completion signal (state-poll
    // drains return before the final host writes are visible, and past a
    // packet's first execution its state field is stale anyway: submit never
    // resets it). Verify the batch's LAST layer only — earlier layers'
    // outputs are legitimately overwritten by later ops of the same batch.
    // Per-iteration times expose the variance the old totals were hiding.
    for chunk in [4usize, 6, 8, 12, 16, 24, 32, 64, nops] {
        let mut iter_ms: Vec<f64> = Vec::with_capacity(iters);
        let mut exec_ok = 0usize;
        let mut bad = String::new();
        for it in 0..iters {
            t0 = std::time::Instant::now();
            let mut start = 0;
            let mut ok = true;
            while start < nops {
                let end = (start + chunk).min(nops);
                let mut last_seq = 0u64;
                for i in start..end {
                    match ops[i].pkt.submit(&dev, &ctx, &op_handles[i]) {
                        Ok(s) => {
                            last_seq = s;
                            rec.burst_submit(&metas[i % 4], it as u32, rec_seq);
                            rec_seq += 1;
                        }
                        Err(_) => {
                            eprintln!("chunk{chunk} submit failed");
                            return ExitCode::FAILURE;
                        }
                    }
                }
                if syncobj_timeline_wait(&dev, ctx.syncobj_handle, last_seq, 10_000_000_000)
                    .is_err()
                {
                    eprintln!("chunk{chunk} wait timeout");
                    return ExitCode::FAILURE;
                }
                let last_layer = end / 4 - 1;
                if !verify_layer(last_layer) {
                    ok = false;
                    bad = format!("it{it} L{last_layer:02} holds {}", diagnose(last_layer));
                }
                start = end;
            }
            iter_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            exec_ok += ok as usize;
            rec.burst_done(
                &format!("chunk{chunk}"),
                Mode::Burst,
                it as u32,
                t0,
                nops as u32,
                1,
            );
        }
        iter_ms.sort_by(|a, b| a.total_cmp(b));
        let med = iter_ms[iter_ms.len() / 2];
        println!(
            "  chunk{:>3}: median {:>7.2} ms/token ({:>5.1} tok/s, min {:>6.2}, max {:>6.2}) verified {}/{} {}",
            chunk,
            med,
            1e3 / med,
            iter_ms[0],
            iter_ms[iter_ms.len() - 1],
            exec_ok,
            iters,
            &bad
        );
    }

    // ---- M5a xnpu-perf 报告：per-op 表 + 链式块对比 + roofline 判定 ----
    let model = MachineModel::default();
    let title = format!("run-w4ulayer: {nlayers} layers x 4 shapes, {iters} iters/mode");
    let (md, summary) = xnpu_perf::render_markdown(&rec, &metas, &model, &title);
    println!("\n{md}");
    let dir = "/home/nzinfo/qwen/xnpu/build/perf";
    if std::fs::create_dir_all(dir).is_ok() {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let stem = format!("{dir}/w4ulayer_{nlayers}L_{ts}");
        let json = trace_json(&rec, &model, &title, &summary);
        match std::fs::write(format!("{stem}.json"), json)
            .and_then(|()| std::fs::write(format!("{stem}.md"), &md))
        {
            Ok(()) => println!("perf trace written: {stem}.json / .md"),
            Err(e) => eprintln!("perf trace write failed: {e}"),
        }
    }

    ExitCode::SUCCESS
}

/// ---- M3b decode-chain CPU glue (mirrors tools/decode_export.py exactly:
/// f32 compute, one bf16 rounding per op boundary) ----

fn rms_norm_bf16(x: &[u16], w: &[u16], out: &mut [u16]) {
    let mut sum = 0f32;
    for b in x.iter() {
        let v = bf16_to_f32(*b);
        sum += v * v;
    }
    let inv = 1f32 / (sum / x.len() as f32 + 1e-5).sqrt();
    for i in 0..x.len() {
        out[i] = f32_to_bf16(bf16_to_f32(x[i]) * inv * bf16_to_f32(w[i]));
    }
}

fn add_bf16(a: &[u16], b: &[u16], out: &mut [u16]) {
    for i in 0..a.len() {
        out[i] = f32_to_bf16(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
    }
}

/// silu(gate)*up for a fused [gate | up] vector.
fn swiglu_bf16(x: &[u16], out: &mut [u16]) {
    let half = x.len() / 2;
    for i in 0..half {
        let g = bf16_to_f32(x[i]);
        let u = bf16_to_f32(x[half + i]);
        out[i] = f32_to_bf16(g * (1.0 / (1.0 + (-g).exp())) * u);
    }
}

/// Llama rotate-half rope tables at one position (inv_freq = base^(-j/64)).
fn rope_table(pos: usize, base: f32) -> ([f32; 64], [f32; 64]) {
    let mut c = [0f32; 64];
    let mut s = [0f32; 64];
    for j in 0..64 {
        let inv = base.powf(-(j as f32) / 64.0);
        let a = pos as f32 * inv;
        c[j] = a.cos();
        s[j] = a.sin();
    }
    (c, s)
}

fn rope_apply(x: &[u16], c: &[f32; 64], s: &[f32; 64], heads: usize, out: &mut [u16]) {
    for h in 0..heads {
        let o = h * 128;
        for j in 0..64 {
            let x1 = bf16_to_f32(x[o + j]);
            let x2 = bf16_to_f32(x[o + 64 + j]);
            out[o + j] = f32_to_bf16(x1 * c[j] - x2 * s[j]);
            out[o + 64 + j] = f32_to_bf16(x2 * c[j] + x1 * s[j]);
        }
    }
}

/// Per-head RMS norm over head_dim (hy qk-norm, applied AFTER rope):
/// bf16(x_f32 * rsqrt(mean(x^2)+1e-5) * w_f32), weights [128] shared by
/// all heads (mirrors decode_export.py's qk_rms).
fn qk_rms_bf16(x: &[u16], w: &[u16], out: &mut [u16]) {
    for h in 0..x.len() / 128 {
        let o = h * 128;
        let mut sum = 0f32;
        for j in 0..128 {
            let v = bf16_to_f32(x[o + j]);
            sum += v * v;
        }
        let inv = 1f32 / (sum / 128.0 + 1e-5).sqrt();
        for j in 0..128 {
            out[o + j] = f32_to_bf16(bf16_to_f32(x[o + j]) * inv * bf16_to_f32(w[j]));
        }
    }
}

/// GQA decode attention: q (heads, roped) against a (kv, CACHE_SEQ, 128)
/// cache; q head h reads kv head h/(heads/kv). Scores/softmax/PV in f32,
/// one bf16 rounding at the output.
fn attention_bf16(
    q: &[u16],
    kc: &[u16],
    vc: &[u16],
    pos: usize,
    cache_seq: usize,
    out: &mut [u16],
    heads: usize,
    nkv: usize,
) {
    let s = pos + 1;
    let group = heads / nkv;
    let mut sc = vec![0f32; s];
    for h in 0..heads {
        let kv = h / group;
        let mut mx = f32::NEG_INFINITY;
        for t in 0..s {
            let off = (kv * cache_seq + t) * 128;
            let mut d = 0f32;
            for j in 0..128 {
                d += bf16_to_f32(kc[off + j]) * bf16_to_f32(q[h * 128 + j]);
            }
            sc[t] = d / (128f32).sqrt();
            mx = mx.max(sc[t]);
        }
        let mut sum = 0f32;
        for v in sc[..s].iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        for j in 0..128 {
            let mut acc = 0f32;
            for t in 0..s {
                let off = (kv * cache_seq + t) * 128;
                acc += sc[t] * bf16_to_f32(vc[off + j]);
            }
            out[h * 128 + j] = f32_to_bf16(acc / sum);
        }
    }
}

/// M3b/M4a: one full 42-layer decode step over real weights — the engine
/// skeleton. Projections run on the universal w4gemvu CU (run-w4ulayer
/// machinery); rope, rms-norm, swiglu and the residuals run in Rust (f32
/// math, bf16 boundaries — exactly what tools/decode_export.py's reference
/// computes), so the final hidden must reproduce golden_hidden.bin to
/// accumulation-order noise. M4a adds the NPU decode attention: the
/// flowkv_decode CU (streaming online-softmax attention, runtime S from a
/// Q element header — one compiled binary per cache slot) fed from
/// per-layer interleaved KV cache BOs. Pass the trailing "cpu" arg to keep
/// the Rust scalar attention as the A/B reference path.
fn cmd_run_decode(
    arch: &DecArch,
    decdir: &str,
    w4dir: &str,
    iters: usize,
    npu_attn_req: bool,
) -> ExitCode {
    let build = "/home/nzinfo/qwen/xnpu/build";
    let layers = arch.layers;
    let shapes = arch.shapes();
    let npu_attn = npu_attn_req;
    println!(
        "decode chain: {name} {layers} layers, w4gemvu projections on CU0 + {} attention{}, {iters} iters",
        if npu_attn { "flowkv NPU (CU1)" } else { "Rust scalar" },
        if npu_attn { "" } else { " (cpu mode)" },
        name = arch.name,
    );

    // Fixtures + PDI identity (same contract as run-w4ulayer).
    let fixtures: Vec<(Vec<u8>, Vec<u8>, u32)> = shapes
        .iter()
        .map(|s| {
            match load_fixture(&format!("{build}/w4gemvu_{}x{}.mlir.prj", s.m, s.k)) {
                Some(f) => f,
                None => {
                    eprintln!("load fixture w4gemvu_{}x{} failed", s.m, s.k);
                    std::process::exit(2);
                }
            }
        })
        .collect();
    for (s, f) in shapes.iter().zip(&fixtures).skip(1) {
        if f.0 != fixtures[0].0 {
            eprintln!("PDI for {} differs — stale fixtures", s.name);
            return ExitCode::FAILURE;
        }
    }

    // M4a flowkv fixture: same 8-col QoS partition, its own CU (func 1).
    let fk_fixture = if npu_attn {
        match load_fixture(&format!("{build}/{}.mlir.prj", arch.fk_fixture)) {
            Some(f) => {
                println!("flowkv: pdi {} B, ctrl-code {} B", f.0.len(), f.1.len());
                f
            }
            None => {
                eprintln!("load flowkv fixture failed (run the flowkv pytest first)");
                return ExitCode::FAILURE;
            }
        }
    } else {
        (Vec::new(), Vec::new(), 0)
    };

    let dev = match Device::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open amdxdna device: {e}");
            return ExitCode::FAILURE;
        }
    };
    let md = match dev.aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("AIE metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let cols = 8u32;
    let num_tiles = cols * md.core.row_count as u32;
    let mut ctx = match HwContext::create(&dev, num_tiles) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    // Both PDIs take cu_func 0 — the CU slot is the index in this list and
    // is selected per op via set_cu (the run-multi pattern; func != 0 makes
    // the fw look up a DPU function the PDI doesn't have and the op never
    // runs).
    let cus: Vec<(&[u8], u8)> = if npu_attn {
        vec![(fixtures[0].0.as_slice(), 0), (fk_fixture.0.as_slice(), 0)]
    } else {
        vec![(fixtures[0].0.as_slice(), 0)]
    };
    if let Err(e) = ctx.configure_cus(&cus) {
        eprintln!("configure_cus: {e}");
        return ExitCode::FAILURE;
    }

    // Device buffers: replicated-x BOs, per-shape outputs, per-(layer,shape)
    // weights. live layout: [xu2048, xu6144, c_qkv, c_o, c_gateup, c_down,
    // w00.qkv, w00.o, ...].
    let mut live: Vec<(BufferObject, Mapping)> = Vec::new();
    if chain_tensor(&dev, &mut live, "xu2048", &vec![0u8; 24 * W4U_K_MAX * 2]).is_none()
        || chain_tensor(&dev, &mut live, "xu6144", &vec![0u8; 4 * W4U_K_MAX * 2]).is_none()
    {
        eprintln!("x BO failed");
        return ExitCode::FAILURE;
    }
    for s in shapes.iter() {
        if chain_tensor(&dev, &mut live, &format!("c_{}", s.name), &vec![0u8; s.m * 2])
            .is_none()
        {
            eprintln!("c_{} BO failed", s.name);
            return ExitCode::FAILURE;
        }
    }
    for n in 0..layers {
        for s in shapes.iter() {
            let data = match std::fs::read(format!("{w4dir}/layer{n:02}_{}.bin", s.name)) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("read layer{n:02}_{}: {e}", s.name);
                    return ExitCode::FAILURE;
                }
            };
            if data.len() != 8 * (s.m / 32) * W4U_ELEM {
                eprintln!("layer{n:02}_{}: stale import (use --layout v2)", s.name);
                return ExitCode::FAILURE;
            }
            if chain_tensor(&dev, &mut live, &format!("w{n:02}.{}", s.name), &data).is_none() {
                eprintln!("w{n:02}.{} BO failed", s.name);
                return ExitCode::FAILURE;
            }
        }
    }
    let mut ops: Vec<ChainOp> = Vec::with_capacity(layers * 4);
    for n in 0..layers {
        for (si, s) in shapes.iter().enumerate() {
            let xv = live[if s.k == 6144 { 1 } else { 0 }].1.as_ptr() as u64;
            match chain_op(
                &dev,
                &format!("{}L{n:02}", s.name),
                &fixtures[si].1,
                0,
                &[live[6 + n * 4 + si].1.as_ptr() as u64, xv, live[2 + si].1.as_ptr() as u64],
            ) {
                Some(o) => ops.push(o),
                None => {
                    eprintln!("op setup layer{n:02} {} failed", s.name);
                    return ExitCode::FAILURE;
                }
            }
        }
    }
    let x_hdl = [live[0].0.handle(), live[1].0.handle()];
    let c_hdl: Vec<u32> = (2..6).map(|i| live[i].0.handle()).collect();
    let mut op_handles: Vec<Vec<u32>> = Vec::with_capacity(layers * 4);
    for n in 0..layers {
        for si in 0..4 {
            op_handles.push(vec![
                ops[n * 4 + si].ctrl_bo.handle(),
                live[6 + n * 4 + si].0.handle(),
                x_hdl[if shapes[si].k == 6144 { 1 } else { 0 }],
                c_hdl[si],
            ]);
        }
    }

    // Host state: norms, caches, x0, goldens.
    let rd_u16file = |p: &str| -> Option<Vec<u16>> {
        let d = std::fs::read(p).ok()?;
        Some(d.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    };
    let norms = match rd_u16file(&format!("{decdir}/norms.bin")) {
        Some(v) if v.len() == (2 * layers + 1) * 2048 => v,
        _ => {
            eprintln!("read {decdir}/norms.bin failed");
            return ExitCode::FAILURE;
        }
    };
    let qknorms = if arch.qk_norm {
        match rd_u16file(&format!("{decdir}/qknorms.bin")) {
            Some(v) if v.len() == layers * 256 => v,
            _ => {
                eprintln!("read {decdir}/qknorms.bin failed");
                return ExitCode::FAILURE;
            }
        }
    } else {
        Vec::new()
    };
    let x0 = match rd_u16file(&format!("{decdir}/x0.bin")) {
        Some(v) if v.len() == 2048 => v,
        _ => {
            eprintln!("read {decdir}/x0.bin failed");
            return ExitCode::FAILURE;
        }
    };
    let cache_seq = 1024usize;
    let mut kcache = match rd_u16file(&format!("{decdir}/kcache.bin")) {
        Some(v) if v.len() == layers * arch.kv * cache_seq * 128 => v,
        _ => {
            eprintln!("read kcache failed");
            return ExitCode::FAILURE;
        }
    };
    let mut vcache = match rd_u16file(&format!("{decdir}/vcache.bin")) {
        Some(v) if v.len() == layers * arch.kv * cache_seq * 128 => v,
        _ => {
            eprintln!("read vcache failed");
            return ExitCode::FAILURE;
        }
    };
    let golden_hidden = match rd_u16file(&format!("{decdir}/golden_hidden.bin")) {
        Some(v) if v.len() == 2048 => v,
        _ => {
            eprintln!("read golden_hidden failed");
            return ExitCode::FAILURE;
        }
    };
    let pos = 100usize;
    let (rc, rs) = rope_table(pos, arch.rope_base);

    // M4a NPU-attention state. Layouts must match
    // iron/operators/flowkv_decode: KV cache = (kv heads, 1024 pos, [K|V],
    // 128) interleaved per layer; Q element = [Q_group | angles (128
    // interleaved cos/sin) | hdr(16, runtime S u32 in the first two bf16
    // bit patterns)] per KV group, header at the element TAIL (peano
    // anchor, notes §16).
    //
    // ANGLES ARE IDENTITY (cos=1, sin=0) BY CONTRACT: the kernel ropes Q
    // on the fly (reference.py: O = softmax(rope(Q)·K)·V, K pre-roped), so
    // shipping real angles alongside the host-roped Q double-rotates —
    // R(a)·R(a)=R(2a), i.e. Q at 2·pos (the latent M4a bug the hy port
    // exposed; minicpm's NPU path had it too). Identity is exact in bf16
    // (x·1−y·0 = x), so the host's roped (+qk-normed, hy) Q passes through
    // bit-exact and ALL of rope/qk-norm stays host-side for both archs.
    const FK_CAP: usize = 1024;
    // Q element stride: one GQA group's Q heads + angles + runtime-S hdr
    // (op.py pack_q_with_angles). minicpm 8h/group=1176, hy 4h/group=656.
    let fk_group = arch.heads / arch.kv;
    let fk_stride = fk_group * 128 + 128 + 16;
    struct FkState {
        ops: Vec<ChainOp>,
        handles: Vec<Vec<u32>>,
        kv: Vec<(BufferObject, Mapping)>,
        q: (BufferObject, Mapping),
        o: (BufferObject, Mapping),
    }
    let mut fk: Option<FkState> = if npu_attn {
        let q_bytes_total = arch.kv * fk_stride * 2;
        let o_bytes_total = 16 * 128 * 2;
        let mut qdata = vec![0u16; arch.kv * fk_stride];
        for g in 0..arch.kv {
            let abase = g * fk_stride + fk_group * 128;
            for j in 0..64 {
                qdata[abase + 2 * j] = f32_to_bf16(1.0); // identity angles
                qdata[abase + 2 * j + 1] = f32_to_bf16(0.0);
            }
            // hdr[0..2] = S as u32 (little-endian halves); rest stays 0.
            qdata[abase + 128] = (pos + 1) as u16;
        }
        let q_bo = match BufferObject::new(&dev, BoType::Shmem, q_bytes_total) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("flowkv q BO: {e}");
                return ExitCode::FAILURE;
            }
        };
        let mut q_map = match q_bo.map_owned() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("flowkv q map: {e}");
                return ExitCode::FAILURE;
            }
        };
        q_map
            .as_mut_slice()
            .copy_from_slice(&qdata.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        if let Err(e) = q_bo.sync(SyncDirection::ToDevice, 0, q_bytes_total as u64) {
            eprintln!("flowkv q sync: {e}");
            return ExitCode::FAILURE;
        }
        let o_bo = match BufferObject::new(&dev, BoType::Shmem, o_bytes_total) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("flowkv o BO: {e}");
                return ExitCode::FAILURE;
            }
        };
        let o_map = match o_bo.map_owned() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("flowkv o map: {e}");
                return ExitCode::FAILURE;
            }
        };
        let q_va = q_map.as_ptr() as u64;
        let o_va = o_map.as_ptr() as u64;

        // Per-layer interleaved KV caches: history rows 0..pos from the
        // exported kcache/vcache; the row at `pos` is written per step.
        let mut kv = Vec::with_capacity(layers);
        for n in 0..layers {
            let mut inter = vec![0u16; arch.kv * FK_CAP * 2 * 128];
            for kvh in 0..arch.kv {
                for p in 0..pos {
                    let src = (n * arch.kv + kvh) * cache_seq * 128 + p * 128;
                    let dst = (kvh * FK_CAP * 2 + p * 2) * 128;
                    inter[dst..dst + 128].copy_from_slice(&kcache[src..src + 128]);
                    inter[dst + 128..dst + 256].copy_from_slice(&vcache[src..src + 128]);
                }
            }
            let bo = match BufferObject::new(&dev, BoType::Shmem, inter.len() * 2) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("flowkv kv{n:02} BO: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let mut map = match bo.map_owned() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("flowkv kv{n:02} map: {e}");
                    return ExitCode::FAILURE;
                }
            };
            map.as_mut_slice()
                .copy_from_slice(&inter.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            if let Err(e) = bo.sync(SyncDirection::ToDevice, 0, bo.size() as u64) {
                eprintln!("flowkv kv{n:02} sync: {e}");
                return ExitCode::FAILURE;
            }
            kv.push((bo, map));
        }

        let mut ops = Vec::with_capacity(layers);
        let mut handles = Vec::with_capacity(layers);
        for n in 0..layers {
            let h = kv[n].0.handle();
            match chain_op(
                &dev,
                &format!("fkL{n:02}"),
                &fk_fixture.1,
                1,
                &[kv[n].1.as_ptr() as u64, q_va, o_va],
            ) {
                Some(op) => {
                    handles.push(vec![op.ctrl_bo.handle(), h, q_bo.handle(), o_bo.handle()]);
                    ops.push(op);
                }
                None => {
                    eprintln!("flowkv op layer{n} setup failed");
                    return ExitCode::FAILURE;
                }
            }
        }
        println!(
            "flowkv: {} kv cache BOs ({} KiB each), q {} B, o {} B, {} ops on CU1",
            layers,
            arch.kv * FK_CAP * 2 * 128 * 2 / 1024,
            q_bytes_total,
            o_bytes_total,
            layers
        );
        Some(FkState {
            ops,
            handles,
            kv,
            q: (q_bo, q_map),
            o: (o_bo, o_map),
        })
    } else {
        None
    };

    // M5a xnpu-perf: per-shape OpMeta（同 run-w4ulayer）+ flowkv（S=pos+1
    // 运行时；stream 口径给满编译容量的 KV 全量）。it=0 的 checked step
    // 含逐层 golden 磁盘读，不记录，避免污染 solo 分布。
    let metas: Vec<OpMeta> = shapes
        .iter()
        .map(|s| {
            let mut m = OpMeta::new(
                s.name,
                "w4gemvu",
                0,
                (s.m * s.k / 2 + s.m * (s.k / 32) * 2 + s.k * 2) as u64,
                (s.m * 2) as u64,
                (2 * s.m * s.k) as u64,
            );
            m.bytes_stream = Some((8 * (s.m / 32) * W4U_ELEM) as u64);
            m
        })
        .chain(if npu_attn {
            Some({
                let s_pos = (pos + 1) as u64;
                let nkv = arch.kv as u64;
                let mut m = OpMeta::new(
                    "flowkv",
                    "attn",
                    1,
                    nkv * s_pos * 2 * 128 * 2 + nkv * fk_stride as u64 * 2, // S 行 KV + q
                    (16 * 128 * 2) as u64,
                    (16 * s_pos * 128 * 2 * 2) as u64,
                );
                // 内核按编译容量流 KV（S 是运行时 header）；stream 口径给全量
                m.bytes_stream =
                    Some(nkv * FK_CAP as u64 * 2 * 128 * 2 + nkv * fk_stride as u64 * 2);
                m
            })
        } else {
            None
        }
        .into_iter())
        .collect();
    let mut rec = Recorder::new();
    let mut rec_seq = 0u64;

    // One w4gemvu call: replicate x into the vector BO, submit, wait, read c.
    let mut gemv = |ops: &mut [ChainOp],
                    i: usize,
                    si: usize,
                    x: &[u16],
                    it: u32,
                    rec: &mut Recorder,
                    rec_seq: &mut u64|
     -> Option<Vec<u16>> {
        let k = shapes[si].k;
        let m = shapes[si].m;
        let vi = if k == 6144 { 1 } else { 0 };
        let slots = if vi == 0 { 24 } else { 4 };
        {
            let (bo, map) = &mut live[vi];
            let bytes = map.as_mut_slice();
            for s in 0..slots {
                let base = s * W4U_K_MAX * 2;
                for (j, b) in x.iter().enumerate() {
                    bytes[base + j * 2..base + j * 2 + 2].copy_from_slice(&b.to_le_bytes());
                }
            }
            if let Err(e) = bo.sync(SyncDirection::ToDevice, 0, bo.size() as u64) {
                eprintln!("gemv x sync (op {i}): {e}");
                return None;
            }
        }
        let ts = std::time::Instant::now();
        let op = &mut ops[i];
        let seq = match op.pkt.submit(&dev, &ctx, &op_handles[i]) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("gemv submit (op {i}): {e}");
                return None;
            }
        };
        if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000) {
            eprintln!("gemv wait (op {i}, seq {seq}): {e}");
            return None;
        }
        if it > 0 {
            rec.solo(&metas[si], it, ts, *rec_seq);
            *rec_seq += 1;
        }
        let (c_bo, c_map) = &live[2 + si];
        // SHMEM is coherent; the direction-1 ioctl needs a debug BO (M1) —
        // best-effort only, like every other read path here.
        let _ = c_bo.sync(SyncDirection::FromDevice, 0, c_bo.size() as u64);
        let cs = c_map.as_slice();
        Some((0..m).map(|r| u16::from_le_bytes([cs[r * 2], cs[r * 2 + 1]])).collect())
    };

    // The decode step. Scratch lives outside the closure (passed per call)
    // so the final-norm block can reuse xn after the last call.
    struct Scratch {
        xn: Vec<u16>,
        qr: Vec<u16>,
        kr: Vec<u16>,
        attn: Vec<u16>,
        gu: Vec<u16>,
        sw: Vec<u16>,
    }
    let mut sc = Scratch {
        xn: vec![0u16; 2048],
        qr: vec![0u16; 2048],
        kr: vec![0u16; arch.kv * 128],
        attn: vec![0u16; 2048],
        gu: vec![0u16; 12288],
        sw: vec![0u16; 6144],
    };
    let mut x = x0.clone();
    let mut first_bad: Option<usize> = None;
    let mut decode_step = |ops: &mut [ChainOp],
                       kcache: &mut [u16],
                       vcache: &mut [u16],
                       x: &mut Vec<u16>,
                       sc: &mut Scratch,
                       fk: &mut Option<FkState>,
                       check: bool,
                       it: u32,
                       rec: &mut Recorder,
                       rec_seq: &mut u64|
     -> bool {
        for n in 0..layers {
            let Scratch { xn, qr, kr, attn, gu, sw } = sc;
            rms_norm_bf16(x, &norms[n * 2 * 2048..][..2048], xn);
            let qkv = match gemv(ops, n * 4, 0, xn, it, rec, rec_seq) {
                Some(v) => v,
                None => return false,
            };
            let kdim = arch.kv * 128;
            rope_apply(&qkv[..2048], &rc, &rs, arch.heads, qr);
            rope_apply(&qkv[2048..2048 + kdim], &rc, &rs, arch.kv, kr);
            if arch.qk_norm {
                // hy: per-head rms AFTER rope. qk_rms can't alias x and out,
                // so bounce through attn as scratch (kr is only kv·128 long).
                let qlen = qr.len();
                qk_rms_bf16(qr, &qknorms[n * 256..][..128], attn);
                qr.copy_from_slice(&attn[..qlen]);
                qk_rms_bf16(kr, &qknorms[n * 256 + 128..][..128], &mut attn[..kdim]);
                kr.copy_from_slice(&attn[..kdim]);
            }
            for kv in 0..arch.kv {
                let off = (n * arch.kv + kv) * cache_seq * 128 + pos * 128;
                kcache[off..off + 128].copy_from_slice(&kr[kv * 128..(kv + 1) * 128]);
                vcache[off..off + 128].copy_from_slice(
                    &qkv[2048 + kdim + kv * 128..2048 + kdim + (kv + 1) * 128],
                );
            }
            match fk.as_mut() {
                Some(fk) => {
                    // Append this step's rotated K row + raw V row into the
                    // layer's interleaved device cache (K first, V next —
                    // 512 contiguous bytes per KV head).
                    {
                        let (bo, map) = &mut fk.kv[n];
                        let bytes = map.as_mut_slice();
                        for kvh in 0..arch.kv {
                            let off = ((kvh * FK_CAP * 2 + pos * 2) * 128) * 2;
                            for j in 0..128 {
                                bytes[off + j * 2..off + j * 2 + 2]
                                    .copy_from_slice(&kr[kvh * 128 + j].to_le_bytes());
                                // V rows start after Q + all K heads: 2048 +
                                // kv*128 (2304 was the MiniCPM kv=2 constant;
                                // hy kv=4 needs 2560 — the current token's V
                                // was K-head-2/3 data, ~12% E2E rms).
                                bytes[off + 256 + j * 2..off + 256 + j * 2 + 2]
                                    .copy_from_slice(&qkv[2048 + kdim + kvh * 128 + j].to_le_bytes());
                            }
                            if let Err(e) = bo.sync(SyncDirection::ToDevice, off as u64, 512) {
                                eprintln!("kv{n:02} row sync: {e}");
                                return false;
                            }
                        }
                    }
                    // Refresh the Q heads in the shared Q element (angles +
                    // header were packed once at setup — only Q changes).
                    {
                        let (qbo, qmap) = &mut fk.q;
                        let bytes = qmap.as_mut_slice();
                        let gq = fk_group * 128; // Q heads per element
                        for g in 0..arch.kv {
                            let base = g * fk_stride * 2;
                            for j in 0..gq {
                                bytes[base + j * 2..base + j * 2 + 2]
                                    .copy_from_slice(&qr[g * gq + j].to_le_bytes());
                            }
                        }
                        if let Err(e) = qbo.sync(SyncDirection::ToDevice, 0, qbo.size() as u64) {
                            eprintln!("q sync (layer {n}): {e}");
                            return false;
                        }
                    }
                    let op = &mut fk.ops[n];
                    // Same first-exec O-read-race guard as run-fkprobe: flush
                    // the o BO's cache lines before submit so the post-wait
                    // read sees the DMA writes.
                    let _ = fk.o.0.sync(SyncDirection::ToDevice, 0, fk.o.0.size() as u64);
                    let ts = std::time::Instant::now();
                    let seq = match op.pkt.submit(&dev, &ctx, &fk.handles[n]) {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("flowkv submit (layer {n}): {e}");
                            return false;
                        }
                    };
                    if let Err(e) =
                        syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000)
                    {
                        eprintln!("flowkv wait (layer {n}, seq {seq}): {e}");
                        return false;
                    }
                    if it > 0 {
                        rec.solo(&metas[4], it, ts, *rec_seq);
                        *rec_seq += 1;
                    }
                    // SHMEM is coherent; the direction-1 ioctl is best-effort.
                    let _ = fk.o.0.sync(SyncDirection::FromDevice, 0, fk.o.0.size() as u64);
                    let os = fk.o.1.as_slice();
                    for i in 0..2048 {
                        attn[i] = u16::from_le_bytes([os[i * 2], os[i * 2 + 1]]);
                    }
                }
                None => {
                    let klo = n * arch.kv * cache_seq * 128;
                    attention_bf16(
                        qr,
                        &kcache[klo..],
                        &vcache[klo..],
                        pos,
                        cache_seq,
                        attn,
                        arch.heads,
                        arch.kv,
                    );
                }
            }
            let o = match gemv(ops, n * 4 + 1, 1, attn, it, rec, rec_seq) {
                Some(v) => v,
                None => return false,
            };
            add_bf16(x, &o, xn); // x = x + o (reuse xn as scratch)
            std::mem::swap(x, xn);
            rms_norm_bf16(x, &norms[(n * 2 + 1) * 2048..][..2048], xn);
            *gu = match gemv(ops, n * 4 + 2, 2, xn, it, rec, rec_seq) {
                Some(v) => v,
                None => return false,
            };
            swiglu_bf16(gu, sw);
            let d = match gemv(ops, n * 4 + 3, 3, sw, it, rec, rec_seq) {
                Some(v) => v,
                None => return false,
            };
            add_bf16(x, &d, xn);
            std::mem::swap(x, xn);
            if check {
                let g = match rd_u16file(&format!("{decdir}/golden_L{n:02}.bin")) {
                    Some(v) => v,
                    None => return false,
                };
                // Same tolerance form as the gemv golden tier (plain rel err
                // explodes on near-zero goldens).
                let mut bad = 0usize;
                let mut worst_rel = 0f32;
                for i in 0..2048 {
                    let gv = bf16_to_f32(x[i]);
                    let e = (gv - bf16_to_f32(g[i])).abs();
                    if e > 0.01 + 0.02 * gv.abs() {
                        bad += 1;
                    }
                    worst_rel = worst_rel.max(e / (gv.abs() + 1e-6));
                }
                if bad > 16 && first_bad.is_none() {
                    first_bad = Some(n);
                    println!(
                        "  first divergence at layer {n}: {bad}/2048 outside tolerance (worst rel {worst_rel:.3})"
                    );
                }
            }
        }
        true
    };

    let mut t0 = std::time::Instant::now();
    {
        let mut ops2 = std::mem::take(&mut ops);
        let ok = decode_step(
            &mut ops2, &mut kcache, &mut vcache, &mut x, &mut sc, &mut fk, true, 0,
            &mut rec, &mut rec_seq,
        );
        ops = ops2;
        if !ok {
            eprintln!("decode step failed");
            return ExitCode::FAILURE;
        }
    }
    let step1 = t0.elapsed();

    // Final norm + verification.
    rms_norm_bf16(&x, &norms[2 * layers * 2048..][..2048], &mut sc.xn);
    let _ = std::fs::write(
        "/tmp/dec_hidden.bin",
        (0..2048).flat_map(|i| sc.xn[i].to_le_bytes()).collect::<Vec<u8>>(),
    );
    let mut worst_abs = 0f32;
    let mut sum_sq = 0f32;
    let mut g_sq = 0f32;
    let mut top: Vec<(usize, f32, f32)> = (0..2048)
        .map(|i| {
            let g = bf16_to_f32(sc.xn[i]);
            let w = bf16_to_f32(golden_hidden[i]);
            (i, (g - w).abs(), w)
        })
        .collect();
    for i in 0..2048 {
        let g = bf16_to_f32(sc.xn[i]);
        let w = bf16_to_f32(golden_hidden[i]);
        let e = (g - w).abs();
        worst_abs = worst_abs.max(e);
        sum_sq += e * e;
        g_sq += w * w;
    }
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    let rms = (sum_sq / 2048.0).sqrt();
    let grms = (g_sq / 2048.0).sqrt();
    println!(
        "final hidden vs reference: rms {:.4} ({:.1}% of golden rms {:.3}), max abs {:.4} -> {}",
        rms,
        100.0 * rms / grms,
        grms,
        worst_abs,
        // Gate on rms relative to the reference's own magnitude — the honest
        // drift metric (max-rel is meaningless on near-zero goldens).
        if rms < 0.05 * grms { "PASS" } else { "FAIL" }
    );
    for (i, e, w) in top.iter().take(3) {
        println!("  largest diff @[{i}]: |err| {e:.4} on golden {w:.4} ({:.1}%)", 100.0 * e / w.abs().max(1e-9));
    }

    // Timed iterations (steady state; caches are idempotent at fixed pos).
    t0 = std::time::Instant::now();
    for it in 1..=iters as u32 {
        let tb = std::time::Instant::now();
        let mut ops2 = std::mem::take(&mut ops);
        let ok = decode_step(
            &mut ops2, &mut kcache, &mut vcache, &mut x, &mut sc, &mut fk, false, it,
            &mut rec, &mut rec_seq,
        );
        ops = ops2;
        if !ok {
            eprintln!("timed decode step failed");
            return ExitCode::FAILURE;
        }
        rec.burst_done(
            "decode-step",
            Mode::Solo,
            it,
            tb,
            (layers * 4 + if npu_attn { layers } else { 0 }) as u32,
            1,
        );
    }
    let per = t0.elapsed() / iters as u32;
    println!(
        "decode step: first (checked) {:?}, steady {:.2?} /token ({:.1} tok/s)",
        step1,
        per,
        1e3 / per.as_secs_f64() / 1e3
    );
    println!(
        "  (CPU: rope+{}+norms+swiglu{}; NPU: {} w4gemvu{})",
        if npu_attn {
            "kv-row-append"
        } else {
            "GQA-attention"
        },
        if arch.qk_norm { "+qk-norm" } else { "" },
        layers * 4,
        if npu_attn {
            format!(" on CU0 + {} flowkv attention on CU1", layers)
        } else {
            " on 1 CU".to_string()
        },
    );

    // ---- M5a xnpu-perf 报告 ----
    let model = MachineModel::default();
    let title = format!(
        "run-decode: {} {}L, {} attention, {iters} timed iters",
        arch.name,
        layers,
        if npu_attn { "NPU flowkv" } else { "CPU scalar" }
    );
    let (md, summary) = xnpu_perf::render_markdown(&rec, &metas, &model, &title);
    println!("\n{md}");
    let dir = "/home/nzinfo/qwen/xnpu/build/perf";
    if std::fs::create_dir_all(dir).is_ok() {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let stem = format!(
            "{dir}/decode_{}_{}it_{ts}",
            if npu_attn { "npu" } else { "cpu" },
            iters
        );
        let json = trace_json(&rec, &model, &title, &summary);
        match std::fs::write(format!("{stem}.json"), json)
            .and_then(|()| std::fs::write(format!("{stem}.md"), &md))
        {
            Ok(()) => println!("perf trace written: {stem}.json / .md"),
            Err(e) => eprintln!("perf trace write failed: {e}"),
        }
    }

    ExitCode::SUCCESS
}

/// M4a probe: drive the flowkv_decode fixture STANDALONE over raw DRM —
/// no w4gemvu CU, fresh context. Isolates "fixture vs raw submit path"
/// from "persistent-kernel PDI switching" when the decode chain's flowkv
/// op times out. Data is the M1 small-int recipe: Q/K constant, cos=1
/// sin=0 (RoPE identity), zero header (full capacity) — every score is
/// equal, so O must equal the per-column mean of V exactly (to bf16
/// noise). Pass bar: op completes AND O matches the uniform-mix golden.
fn cmd_run_fkprobe(prj: &str, iters: usize) -> ExitCode {
    let (pdi, instr, cols) = match load_fixture(prj) {
        Some(f) => {
            println!(
                "flowkv probe: pdi {} B, ctrl-code {} B, partition {} cols",
                f.0.len(),
                f.1.len(),
                f.2
            );
            f
        }
        None => {
            eprintln!("load fixture {prj} failed");
            return ExitCode::FAILURE;
        }
    };
    let dev = match Device::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open amdxdna device: {e}");
            return ExitCode::FAILURE;
        }
    };
    let md = match dev.aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("AIE metadata: {e}");
            return ExitCode::FAILURE;
        }
    };
    let num_tiles = cols * md.core.row_count as u32;
    let mut ctx = match HwContext::create(&dev, num_tiles) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = ctx.configure_cus(&[(pdi.as_slice(), 0)]) {
        eprintln!("configure_cus: {e}");
        return ExitCode::FAILURE;
    }

    // Buffers: KV interleaved (2, 1024, [K|V], 128); Q element
    // [Q_group(1024) | angles(128) | hdr(16)] x 2, header zero = full S.
    const CAP: usize = 1024;
    const STRIDE: usize = 8 * 128 + 128 + 16;
    let kv_elems = 2 * CAP * 2 * 128;
    let q_elems = 2 * STRIDE;
    let o_elems = 16 * 128;
    let mut kv_data = vec![0u16; kv_elems];
    let mut q_data = vec![0u16; q_elems];
    // Sharp-random mode (the real decode fingerprint): LCG bf16 data, K
    // scaled up so softmax is peaked — a stale C_c/F read then shows up as
    // a shrunken implied weight sum (O = Y/l losing mass), which the
    // uniform recipe is blind to (all corrections are 1 there).
    let lcg = |s: &mut u64| -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut kr_rows = vec![0f32; 2 * CAP * 128];
    for h in 0..2usize {
        for t in 0..CAP {
            let krow = (h * CAP * 2 + t * 2) * 128;
            for j in 0..128 {
                let v = 3.0 * lcg(&mut seed);
                kr_rows[(h * CAP + t) * 128 + j] = v;
                kv_data[krow + j] = f32_to_bf16(v);
            }
            let vrow = krow + 128;
            for j in 0..128 {
                let v = lcg(&mut seed);
                kv_data[vrow + j] = f32_to_bf16(v);
            }
        }
    }
    let mut q_rows = vec![0f32; 16 * 128];
    for h in 0..16usize {
        for j in 0..128 {
            q_rows[h * 128 + j] = lcg(&mut seed);
        }
    }
    for g in 0..2usize {
        let base = g * STRIDE;
        for j in 0..1024 {
            q_data[base + j] = f32_to_bf16(q_rows[g * 1024 + j]);
        }
        let abase = base + 1024;
        for p in 0..64 {
            q_data[abase + 2 * p] = f32_to_bf16(1.0); // cos = 1
            q_data[abase + 2 * p + 1] = f32_to_bf16(0.0); // sin = 0
        }
        // header stays zero: full compiled capacity
    }
    // f32 golden per head over the same bf16-rounded inputs.
    let b16 = |x: f32| -> f32 { bf16_to_f32(f32_to_bf16(x)) };
    let mut golden = vec![0f32; o_elems];
    for h in 0..16usize {
        let kvh = h / 8;
        let mut sc = vec![0f32; CAP];
        let mut mx = f32::NEG_INFINITY;
        for t in 0..CAP {
            let mut d = 0f32;
            for j in 0..128 {
                d += b16(kr_rows[(kvh * CAP + t) * 128 + j]) * b16(q_rows[h * 128 + j]);
            }
            sc[t] = d / (128f32).sqrt();
            mx = mx.max(sc[t]);
        }
        let mut sum = 0f32;
        for v in sc.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        for j in 0..128 {
            let mut acc = 0f32;
            for t in 0..CAP {
                let vrow = (kvh * CAP * 2 + t * 2) * 128 + 128;
                acc += sc[t] * bf16_to_f32(kv_data[vrow + j]);
            }
            golden[h * 128 + j] = acc / sum;
        }
    }

    let kv_bo = match BufferObject::new(&dev, BoType::Shmem, kv_elems * 2) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("kv BO: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut kv_map = match kv_bo.map_owned() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("kv map: {e}");
            return ExitCode::FAILURE;
        }
    };
    kv_map.as_mut_slice()
        .copy_from_slice(&kv_data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    if let Err(e) = kv_bo.sync(SyncDirection::ToDevice, 0, kv_bo.size() as u64) {
        eprintln!("kv sync: {e}");
        return ExitCode::FAILURE;
    }
    let q_bo = match BufferObject::new(&dev, BoType::Shmem, q_elems * 2) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("q BO: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut q_map = match q_bo.map_owned() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("q map: {e}");
            return ExitCode::FAILURE;
        }
    };
    q_map.as_mut_slice()
        .copy_from_slice(&q_data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    if let Err(e) = q_bo.sync(SyncDirection::ToDevice, 0, q_bo.size() as u64) {
        eprintln!("q sync: {e}");
        return ExitCode::FAILURE;
    }
    let o_bo = match BufferObject::new(&dev, BoType::Shmem, o_elems * 2) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("o BO: {e}");
            return ExitCode::FAILURE;
        }
    };
    let o_map = match o_bo.map_owned() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("o map: {e}");
            return ExitCode::FAILURE;
        }
    };
    let op = match chain_op(
        &dev,
        "fk",
        &instr,
        0,
        &[kv_map.as_ptr() as u64, q_map.as_ptr() as u64, o_map.as_ptr() as u64],
    ) {
        Some(o) => o,
        None => {
            eprintln!("op setup failed");
            return ExitCode::FAILURE;
        }
    };
    let mut op = op;
    let handles = vec![op.ctrl_bo.handle(), kv_bo.handle(), q_bo.handle(), o_bo.handle()];

    // M5a xnpu-perf: flowkv 单 op 的字节/FLOP 计数（S header=0 → 满编译
    // 容量，KV 全量 1MB 流量）。flops = 16 头 × 1024 pos × (QK+PV 各 128) × 2。
    let mut fk_meta = OpMeta::new(
        "flowkv",
        "attn",
        0,
        (kv_elems + q_elems) as u64 * 2, // kv + q 输入字节
        (o_elems * 2) as u64,            // O 输出（字节）
        (16 * CAP * 128 * 2 * 2) as u64,
    );
    fk_meta.bytes_stream = None; // 无 slot padding，stream 与 useful 同口径
    let metas = [fk_meta.clone()];
    let mut rec = Recorder::new();
    let mut rec_seq = 0u64;

    let mut t0 = std::time::Instant::now();
    let mut first = std::time::Duration::ZERO;
    let mut bad_iters = 0usize;
    for it in 0..iters {
        // First-exec O read race (notes §17): the O drain can lag the
        // completion syncobj on the very first exec after PDI load, leaving
        // partial/zero output in host RAM. XRT masks this by syncing the
        // output BO ToDevice (clflush) before every submit — mirror that.
        let _ = o_bo.sync(SyncDirection::ToDevice, 0, o_bo.size() as u64);
        let ts = std::time::Instant::now();
        let seq = match op.pkt.submit(&dev, &ctx, &handles) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("submit (iter {it}): {e}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000) {
            eprintln!("wait (iter {it}, seq {seq}): {e}");
            return ExitCode::FAILURE;
        }
        rec.solo(&fk_meta, it as u32, ts, rec_seq);
        rec_seq += 1;
        if it == 0 {
            first = t0.elapsed();
            t0 = std::time::Instant::now();
        }
        // Per-iteration check: is EVERY op corrupted, or only some? The
        // corruption fingerprint (O ≈ a·golden, a != 1) reduces to one number
        // per head: the lstsq scale a. Print per-head a per iteration.
        let _ = o_bo.sync(SyncDirection::FromDevice, 0, o_bo.size() as u64);
        let os = o_map.as_slice();
        let mut msg = String::new();
        let mut iter_bad = false;
        for h in 0..16usize {
            let mut sxy = 0f64;
            let mut sxx = 0f64;
            for j in 0..128 {
                let v = bf16_to_f32(u16::from_le_bytes([os[(h * 128 + j) * 2], os[(h * 128 + j) * 2 + 1]])) as f64;
                let g = golden[h * 128 + j] as f64;
                sxy += v * g;
                sxx += g * g;
            }
            let a = sxy / sxx.max(1e-12);
            if (a - 1.0).abs() > 0.02 {
                iter_bad = true;
            }
            if h < 8 {
                msg.push_str(&format!("{a:.2} "));
            }
        }
        if iter_bad {
            bad_iters += 1;
        }
        println!("iter {it}: group0 scales [{msg}]{}", if iter_bad { " BAD" } else { " ok" });
    }
    let per = t0.elapsed() / (iters - 1).max(1) as u32;
    let _ = o_bo.sync(SyncDirection::FromDevice, 0, o_bo.size() as u64);
    let os = o_map.as_slice();
    // Per-head rms vs the f32 golden — the per-head view catches sporadic
    // single-head corruption that a global max would average away.
    let mut worst = 0f32;
    let mut nonfinite = 0usize;
    let mut head_rms = vec![0f32; 16];
    for h in 0..16usize {
        let mut s = 0f64;
        for j in 0..128 {
            let v = bf16_to_f32(u16::from_le_bytes([os[(h * 128 + j) * 2], os[(h * 128 + j) * 2 + 1]]));
            if !v.is_finite() {
                nonfinite += 1;
            }
            let e = (v - golden[h * 128 + j]).abs();
            worst = worst.max(e);
            s += (e * e) as f64;
        }
        head_rms[h] = (s / 128.0).sqrt() as f32;
    }
    println!(
        "first op {first:?}, steady {per:?}/op over {iters} iters; worst |O-golden| {worst:.4}, nonfinite {nonfinite}, bad-scale iters {bad_iters}"
    );
    let _ = std::fs::write(
        "/tmp/fkp_o.bin",
        os.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<u16>>()
            .iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    );
    let _ = std::fs::write(
        "/tmp/fkp_g.bin",
        golden.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    );
    println!("per-head rms: {:?}", head_rms);
    let clean = nonfinite == 0 && bad_iters == 0 && head_rms.iter().all(|r| *r < 0.02);
    println!("-> {}", if clean { "PASS" } else { "FAIL" });

    // M5a burst 相位：同 op 连发不等待，drain 于最后一个 syncobj——纯设备
    // 时间口径（solo 含 submit/wait 往返，两者差 = overhead 判定输入）。
    const BURST_N: usize = 24;
    let tb = std::time::Instant::now();
    let mut last_seq = 0u64;
    for _ in 0..BURST_N {
        let _ = o_bo.sync(SyncDirection::ToDevice, 0, o_bo.size() as u64);
        match op.pkt.submit(&dev, &ctx, &handles) {
            Ok(s) => {
                last_seq = s;
                rec.burst_submit(&fk_meta, 0, rec_seq);
                rec_seq += 1;
            }
            Err(e) => {
                eprintln!("burst submit: {e}");
                break;
            }
        }
    }
    if last_seq > 0 {
        if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, last_seq, 10_000_000_000) {
            eprintln!("burst wait: {e}");
        } else {
            rec.burst_done("flowkv", Mode::Burst, 0, tb, BURST_N as u32, BURST_N as u32);
        }
    }

    let model = MachineModel::default();
    let title = format!("run-fkprobe: flowkv 16h/2kv d128 S=1024, {iters} solo + {BURST_N} burst");
    let (md, summary) = xnpu_perf::render_markdown(&rec, &metas, &model, &title);
    println!("\n{md}");
    let dir = "/home/nzinfo/qwen/xnpu/build/perf";
    if std::fs::create_dir_all(dir).is_ok() {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let stem = format!("{dir}/fkprobe_{ts}");
        let json = trace_json(&rec, &model, &title, &summary);
        match std::fs::write(format!("{stem}.json"), json)
            .and_then(|()| std::fs::write(format!("{stem}.md"), &md))
        {
            Ok(()) => println!("perf trace written: {stem}.json / .md"),
            Err(e) => eprintln!("perf trace write failed: {e}"),
        }
    }
    ExitCode::SUCCESS
}

fn cmd_info() -> ExitCode {
    let dev = match Device::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open amdxdna device: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("device: {}", dev.path.display());

    let md = match dev.aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("AIE metadata query: {e} (not an amdxdna device?)");
            return ExitCode::FAILURE;
        }
    };
    println!("aie: {}x{} cols={:?}", md.cols, md.rows, md.version);
    println!(
        "col_size={} core_rows={} (start {}) mem_rows={} (start {}) shim_rows={} (start {})",
        md.col_size,
        md.core.row_count,
        md.core.row_start,
        md.mem.row_count,
        md.mem.row_start,
        md.shim.row_count,
        md.shim.row_start,
    );
    println!(
        "locks/tile={} dma_ch/tile={} events/tile={}",
        md.core.lock_count, md.core.dma_channel_count, md.core.event_reg_count,
    );

    if let Ok((npu, h)) = dev.clock_metadata() {
        println!("clocks: {}={}MHz {}={}MHz", npu.0, npu.1, h.0, h.1);
    }
    if let Ok(v) = dev.firmware_version() {
        println!("firmware: {}.{}.{}.{}", v.0, v.1, v.2, v.3);
    }
    if let Ok(p) = dev.power_mode() {
        println!("power mode: {p}");
    }
    ExitCode::SUCCESS
}

/// Create bare HW contexts until the driver refuses, then drop them all.
/// Measures the firmware's concurrent-context budget (per docs/06 this is
/// not queryable, only probe-able).
///
/// `cols` is the column span requested per context. The kernel derives
/// num_col = num_tiles / core.row_count and requires 1 <= num_col <= total,
/// so num_tiles is cols * core rows (mem rows don't count).
fn cmd_ctx_probe(max: usize, cols: u32) -> ExitCode {
    let dev = match Device::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("open amdxdna device: {e}");
            return ExitCode::FAILURE;
        }
    };
    let md = match dev.aie_metadata() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("AIE metadata query: {e}");
            return ExitCode::FAILURE;
        }
    };
    if cols < 1 || cols > md.cols as u32 {
        eprintln!("cols must be in 1..={}", md.cols);
        return ExitCode::FAILURE;
    }
    let num_tiles = cols * md.core.row_count as u32;
    println!(
        "device: {} grid cols={} core={} mem={} => num_tiles={} (num_col={})",
        dev.path.display(),
        md.cols,
        md.core.row_count,
        md.mem.row_count,
        num_tiles,
        cols
    );

    let mut ctxs: Vec<HwContext<'_>> = Vec::new();
    let mut reached = 0usize;
    for i in 0..max {
        match HwContext::create(&dev, num_tiles) {
            Ok(c) => {
                println!(
                    "ctx[{i}]: handle={} doorbell=0x{:x} syncobj={}",
                    c.handle, c.umq_doorbell, c.syncobj_handle
                );
                ctxs.push(c);
                reached += 1;
            }
            Err(e) => {
                println!("ctx[{i}]: CREATE failed: {e} (os error {})", e.raw_os_error().unwrap_or(0));
                break;
            }
        }
    }
    println!("concurrent HW context budget: {reached} (probe max {max})");
    drop(ctxs);
    println!("all contexts destroyed");
    ExitCode::SUCCESS
}
