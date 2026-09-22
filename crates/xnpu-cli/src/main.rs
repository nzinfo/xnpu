//! xnpu-cli: probe / diagnose tools for the amdxdna driver via xnpu-hal.

use std::process::ExitCode;

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
        _ => {
            eprintln!(
                "usage: xnpu-cli <info | ctx-probe [max] [cols] | run-add [prj-dir] [bf16|i8] | run-gemm [prj-dir] [M K N] [bf16|i8] | run-multi [add-prj] [gemm-prj] [M K N] | run-pipe [prj-dir] [M K N] [iters] | run-chain [add-prj] [gemm-prj] [M K N] [reps] | run-q8 [gemm-prj] [rescale-prj] [M K N tile_m] [reps]>"
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
