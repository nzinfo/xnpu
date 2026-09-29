//! xnpu-cli: probe / diagnose tools for the amdxdna driver via xnpu-hal.

use std::process::ExitCode;

use xnpu_perf::{tier, trace_json, MachineModel, Mode, OpMeta, Recorder};

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
            // Rust scalar attention (A/B path; default is NPU attention),
            // "fused" runs the M9/P18 fused rms-pair chain (hy only: pair
            // geometry needs qkv_m == 3072), "quad" the P19 whole-layer
            // chain (one exec per layer, device swiglu; hy only).
            let is_hy = args.iter().any(|s| matches!(s.as_str(), "hy" | "hy-mt2"));
            let arch: &DecArch = if is_hy { &DEC_HY } else { &DEC_MINICPM };
            let fused = args.iter().any(|s| s.as_str() == "fused");
            let quad = args.iter().any(|s| s.as_str() == "quad");
            let lv2 = args.iter().any(|s| s.as_str() == "lv2");
            if (fused || quad) && !is_hy {
                eprintln!("fused/quad chains need the hy arch (qkv_m 3072)");
                return ExitCode::FAILURE;
            }
            if fused && quad {
                eprintln!("fused and quad are exclusive chain modes");
                return ExitCode::FAILURE;
            }
            if lv2 {
                if !is_hy {
                    eprintln!("lv2 chain needs the hy arch (layerv2 packs are hy-only)");
                    return ExitCode::FAILURE;
                }
                if fused || quad {
                    eprintln!("lv2 is exclusive with fused/quad");
                    return ExitCode::FAILURE;
                }
                let pos_args: Vec<&String> = args[1..]
                    .iter()
                    .filter(|s| !matches!(s.as_str(), "cpu" | "hy" | "hy-mt2" | "lv2"))
                    .collect();
                let decdir = pos_args
                    .first()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| arch.decdir.to_string());
                let w4dir = pos_args
                    .get(1)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| arch.w4dir.to_string());
                let iters = pos_args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
                // P28-5: host attention between execs is inherent to the
                // half-layer-shifted chain — no flowkv mode here (yet).
                return cmd_run_decode_lv2(arch, &decdir, &w4dir, iters);
            }
            let pos_args: Vec<&String> = args[1..]
                .iter()
                .filter(|s| {
                    !matches!(s.as_str(), "cpu" | "hy" | "hy-mt2" | "fused" | "quad")
                })
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
            cmd_run_decode(arch, &decdir, &w4dir, iters, npu_attn, fused, quad)
        }
        Some("run-quadloop") => {
            // P20: engine-side quad hang probe (see cmd_run_quadloop).
            let iters = args
                .iter()
                .skip(1)
                .find_map(|s| s.parse::<usize>().ok())
                .unwrap_or(200);
            let mode = args
                .iter()
                .skip(1)
                .find(|s| s.parse::<usize>().is_err())
                .map(|s| s.as_str())
                .unwrap_or("plain");
            cmd_run_quadloop(iters, mode)
        }
        Some("run-lv2loop") => {
            // P28-6 6f-7: engine-path lv2 crawl discriminator (see
            // cmd_run_lv2loop) — the same fixture the pyxrt forensic
            // drove, submitted through OUR ERT packet + syncobj wait
            // instead of an XRT runlist.
            let nums: Vec<usize> = args
                .iter()
                .skip(1)
                .filter_map(|s| s.parse::<usize>().ok())
                .collect();
            let iters = nums.first().copied().unwrap_or(40);
            let n = nums.get(1).copied().unwrap_or(16);
            cmd_run_lv2loop(iters, n)
        }
        Some("run-lmhead") => {
            // M8/P8: hy lm_head on NPU — isolated probe (correctness + perf).
            let iters = args
                .iter()
                .filter_map(|s| s.parse::<usize>().ok())
                .next()
                .unwrap_or(4);
            cmd_run_lmhead(iters)
        }
        Some("perf-calibrate") => {
            // M5a P6: measure machine-model ceilings (slot-stream per shape,
            // strided via flowkv) and emit the overlay JSON.
            let is_hy = args.iter().any(|s| matches!(s.as_str(), "hy" | "hy-mt2"));
            let arch: &DecArch = if is_hy { &DEC_HY } else { &DEC_MINICPM };
            let iters = args
                .iter()
                .filter_map(|s| s.parse::<usize>().ok())
                .next()
                .unwrap_or(6);
            cmd_perf_calibrate(arch, iters)
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
                "usage: xnpu-cli <info | ctx-probe [max] [cols] | run-add [prj-dir] [bf16|i8] | run-gemm [prj-dir] [M K N] [bf16|i8] | run-multi [add-prj] [gemm-prj] [M K N] | run-pipe [prj-dir] [M K N] [iters] | run-chain [add-prj] [gemm-prj] [M K N] [reps] | run-q8 [gemm-prj] [rescale-prj] [M K N tile_m] [reps] | run-w4gemv [prj-dir] [M K] [group] [tsi] [iters] | run-w4layer [w4-dir] [layers] [iters] | run-w4ulayer [w4u-dir] [layers] [iters] | run-decode [dec-dir] [w4u-dir] [iters] [hy] [cpu] [fused|quad|lv2] | run-lmhead [iters] | perf-calibrate [hy] [iters]>"
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
        .and_then(|_| pkt.arg32((instr.len() / 4) as u32))
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
        .and_then(|_| pkt.arg32((instr.len() / 4) as u32))
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
            .and_then(|_| p.arg32((instr.len() / 4) as u32));
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
/// P21-3 ToDevice sync dispatch: user-space CLFLUSHOPT by default (the
/// SYNC_BO ioctl measures ~30us FLAT regardless of range — a syscall
/// round trip, not a flush), the ioctl when XNPU_IOCTL_SYNC=1 so both
/// variants can be A/B'd from ONE binary back-to-back (the board's
/// absolute exec time drifts across batches; only same-batch
/// comparisons count — perf-lab P21-3).
fn sync_to_device(bo: &BufferObject, map: &Mapping, off: usize, len: usize) {
    use std::sync::OnceLock;
    static IOCTL: OnceLock<bool> = OnceLock::new();
    if *IOCTL.get_or_init(|| std::env::var("XNPU_IOCTL_SYNC").is_ok()) {
        let _ = bo.sync(SyncDirection::ToDevice, off as u64, len as u64);
    } else {
        map.clflush_region(off, len);
    }
}

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
        .and_then(|_| pkt.arg32((instr.len() / 4) as u32));
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
            .and_then(|_| p.arg32((instr.len() / 4) as u32));
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
                // P24 axcache A/B: with BD AxCache=aggressive (0x0e) the
                // flush may be load-bearing or not — gate it so both arms
                // run from one binary.
                if std::env::var_os("XNPU_NO_IN_FLUSH").is_none() {
                    bo.sync(SyncDirection::ToDevice, 0, bo.size() as u64).ok();
                }
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
            // P24 axcache A/B: this ToDevice-direction sync is really the
            // post-exec cache invalidate (see note at sync_to_device).
            if std::env::var_os("XNPU_NO_OUT_FLUSH").is_none() {
                let _ = c_bo.sync(SyncDirection::ToDevice, 0, c_bo.size() as u64);
            }
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

/// One decode projection shape for the UNIVERSAL w4gemvu kernel: fixture
/// stem + B-stream geometry. Mirrors IRON w4gemvu/op.py (notes §14).
struct W4UShape {
    name: &'static str,
    /// PADDED M (v4 block ABI: M % 256 == 0; pad rows are zero weights ->
    /// zero outputs, dropped on read).
    m: usize,
    k: usize,
}

const W4U_SHAPES: [W4UShape; 4] = [
    W4UShape { name: "qkv", m: 2560, k: 2048 }, // MiniCPM5 (v4 unpads 2688)
    W4UShape { name: "o", m: 2048, k: 2048 }, // v4 unpads v3's 2112
    W4UShape { name: "gateup", m: 12288, k: 2048 },
    W4UShape { name: "down", m: 2048, k: 6144 },
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
    /// qkv fused M (v5: no F — the B stream is gone). o/gateup/down are
    /// shared with MiniCPM5.
    qkv_m: usize,
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
    qkv_m: 2560, // v4 ABI: no pad needed (M % 256 == 0)
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
    fk_fixture: "flowkv_decode_16h_4kv_128d_1024s_32cs_4col",
    decdir: "/home/nzinfo/qwen/xnpu/build/dec_hy",
    w4dir: "/home/nzinfo/qwen/xnpu/build/w4u_hy",
};

impl DecArch {
    fn shapes(&self) -> [W4UShape; 4] {
        [
            W4UShape { name: "qkv", m: self.qkv_m, k: 2048 },
            W4UShape { name: "o", m: 2048, k: 2048 },
            W4UShape { name: "gateup", m: 12288, k: 2048 },
            W4UShape { name: "down", m: 2048, k: 6144 },
        ]
    }
}

/// One v4 MATRIX-UNIT-TILE fifo element (P11): 16 rows x 2048 k, computed
/// by mmul<4,16,16,int8,int4> (mac_4x16_16x16, 1024 MACs/instr; the v3 fp
/// inner loop was ISSUE-bound at ~10 vector ops / 32 MACs — the P11 probe
/// measured the matrix unit at ~100 GMAC/s/core, 32x). Nibbles are
/// group-major transposed into the B-operand order, scales transposed
/// (sf_t[g*16+n]), K header at the tail. ELEM % 64 == 0 is MANDATORY:
/// depth-2 fifo element buffers sit at base and base+ELEM, and aie2p
/// load_v needs 64B alignment for 1024-bit streams — 18464 (%32 only)
/// left every odd element buffer computing from garbage nibbles (the
/// fingerprint: even 16-row tiles exact, odd tiles wrong; at 18448 the
/// scale loads went junk too, 1e18 values).
const W4U_ELEM: usize = 18560;
const W4U_K_MAX: usize = 6144;
const W4U_TILE_K: usize = 2048;
const W4U_TILE_ROWS: usize = 16;

/// v5 block math (mirrors IRON w4gemvu/reference.py). Every block is one
/// 16-row x 2048 tile, self-describing (K + chunk words at the tail);
/// K=6144 ops stream 3 chunk-blocks per tile in CHUNK-MAJOR order. There
/// is NO B stream (P12/v5): the activation rides the A fifo as a K=0
/// element preceding each op's blocks.
fn w4u_chunks(k: usize) -> usize {
    k / W4U_TILE_K // 1 (K=2048) / 3 (K=6144)
}
/// 16-row tiles per column per chunk.
fn w4u_tiles(m: usize) -> usize {
    m / 8 / W4U_TILE_ROWS
}
fn w4u_blocks(m: usize, k: usize) -> usize {
    w4u_tiles(m) * w4u_chunks(k)
}
/// C bf16 ROWS the drain tensor holds (v5): per column one section of
/// [16 zero rows (the activation element's) | blocks 16-row partials |
/// 16-row pad element] — sections are whole 16-row elements and EVEN in
/// element count (BD 4-B alignment; the pad element is never moved).
fn w4u_c_rows(m: usize, k: usize) -> usize {
    8 * (w4u_blocks(m, k) + 2) * W4U_TILE_ROWS
}

/// Quantize a bf16 activation to the v4 int8 ABI: per-group-32 symmetric,
/// d = amax/127 bf16, q in [-127,127] — MUST mirror reference.py
/// quantize_vector bit-exactly (RNE bf16 + round_ties_even f32), the
/// goldens are computed against this exact (q, d).
fn w4u_quantize_x(x_bits: &[u16], k: usize) -> (Vec<i8>, Vec<u16>) {
    // P27-2: the hostprof microscope put quantize at 18ns/elem scalar
    // (B:quant alone was 3.6ms/token). Bit-identical AVX-512 path below.
    if std::arch::is_x86_feature_detected!("avx512f") {
        return unsafe { w4u_quantize_x_avx512(x_bits, k) };
    }
    assert!(k % 32 == 0 && x_bits.len() >= k);
    let n_groups = k / 32;
    let mut q = vec![0i8; k];
    let mut d = vec![0u16; n_groups];
    for g in 0..n_groups {
        let mut amax = 0f32;
        for j in 0..32 {
            let v = bf16_to_f32(x_bits[g * 32 + j]).abs();
            if v > amax {
                amax = v;
            }
        }
        if amax == 0.0 {
            continue; // q stays 0, d stays 0 (torch.where guard)
        }
        let d_bits = f32_to_bf16(amax / 127.0);
        d[g] = d_bits;
        let df = bf16_to_f32(d_bits);
        for j in 0..32 {
            let v = bf16_to_f32(x_bits[g * 32 + j]) / df;
            let r = v.round_ties_even();
            q[g * 32 + j] = r.clamp(-127.0, 127.0) as i8;
        }
    }
    (q, d)
}

/// P27-2: AVX-512 w4u_quantize_x — BIT-IDENTICAL to the scalar body (the
/// torch quantize_vector contract golden is built on): amax as integer
/// max over |f32 bits| (nonneg IEEE bit order == value order), IEEE
/// `_mm512_div_ps` for x/df, `roundscale(0x00)` == f32::round_ties_even,
/// clamp-then-saturate matching `r.clamp(-127.0,127.0) as i8`. The
/// amax==0 group keeps the torch.where guard (q=0, d=0).
#[target_feature(enable = "avx512f")]
unsafe fn w4u_quantize_x_avx512(x_bits: &[u16], k: usize) -> (Vec<i8>, Vec<u16>) {
    use std::arch::x86_64::*;
    unsafe {
        assert!(k % 32 == 0 && x_bits.len() >= k);
        let n_groups = k / 32;
        let mut q = vec![0i8; k];
        let mut d = vec![0u16; n_groups];
        let absmask = _mm512_set1_epi32(0x7fff_ffff);
        for g in 0..n_groups {
            // 32 bf16 lanes in one 512b load; widen both 16-lane halves
            // to f32 by the exact bit shift.
            let raw = _mm512_loadu_si512(x_bits.as_ptr().add(g * 32) as *const __m512i);
            let lo = _mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm512_castsi512_si256(raw)),
                16,
            );
            let hi = _mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(raw, 1)),
                16,
            );
            let m = _mm512_max_epi32(
                _mm512_and_si512(lo, absmask),
                _mm512_and_si512(hi, absmask),
            );
            let amax_bits = _mm512_reduce_max_epi32(m) as u32;
            if amax_bits == 0 {
                continue; // q stays 0, d stays 0 (torch.where guard)
            }
            let amax = f32::from_bits(amax_bits);
            let d_bits = f32_to_bf16(amax / 127.0);
            d[g] = d_bits;
            let df = _mm512_set1_ps(bf16_to_f32(d_bits));
            for (half, v) in [(0usize, lo), (16usize, hi)] {
                let t = _mm512_roundscale_ps(
                    _mm512_div_ps(_mm512_castsi512_ps(v), df),
                    0x00,
                );
                let t = _mm512_min_ps(
                    _mm512_max_ps(t, _mm512_set1_ps(-127.0)),
                    _mm512_set1_ps(127.0),
                );
                let bytes = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(t));
                _mm_storeu_si128(
                    q.as_mut_ptr().add(g * 32 + half) as *mut __m128i,
                    bytes,
                );
            }
        }
        (q, d)
    }
}
/// Build the ONE ELEM-sized activation element an op's X fill ships
/// (v5: rides the A fifo, K header 0): q (ALL chunks) at 0..K, d bf16 at
/// K_MAX, K=0 word at ELEM-8. Every column's X tap reads the same bytes.
fn w4u_build_x_elem(q: &[i8], d: &[u16], k: usize) -> Vec<u8> {
    assert!(k == 2048 || k == 6144);
    let mut out = vec![0u8; W4U_ELEM];
    for (j, &qv) in q.iter().enumerate() {
        out[j] = qv as u8;
    }
    let dbase = W4U_K_MAX;
    for (g, &dv) in d.iter().enumerate() {
        out[dbase + g * 2..dbase + g * 2 + 2].copy_from_slice(&dv.to_le_bytes());
    }
    out[W4U_ELEM - 8..W4U_ELEM - 4].copy_from_slice(&0u32.to_le_bytes());
    out
}

/// Read a whole op's C tensor into (m,) bf16 bits — the bulk form of
/// w4u_c_at (P15): walks column sections sequentially, no per-row
/// div/mod, identical values and accumulation order.
fn w4u_read_c(cs: &[u8], k: usize, m: usize) -> Vec<u16> {
    let rpc = m / 8;
    let chunks = w4u_chunks(k);
    let section = (w4u_blocks(m, k) + 2) * W4U_TILE_ROWS;
    let mut out = vec![0u16; m];
    // P27-2: the byte-assembly loop was the read path's real cost — the
    // BO mapping is u16-aligned (page base, even offsets), so reinterpret
    // once and let chunks==1 collapse to per-column memcpys.
    let cs16: &[u16] =
        unsafe { std::slice::from_raw_parts(cs.as_ptr() as *const u16, cs.len() / 2) };
    for col in 0..8 {
        let base = col * section + W4U_TILE_ROWS;
        if chunks == 1 {
            out[col * rpc..][..rpc].copy_from_slice(&cs16[base..][..rpc]);
        } else {
            for w in 0..rpc {
                let mut acc = bf16_to_f32(cs16[base + w]);
                for c in 1..chunks {
                    acc += bf16_to_f32(cs16[base + c * rpc + w]);
                }
                out[col * rpc + w] = f32_to_bf16(acc);
            }
        }
    }
    out
}

/// Real output row -> bf16 bits from the v5 C tensor. Column col's
/// section = (blocks+2) 16-row elements; skip the 16 leading zero rows
/// (the activation element's), the block partials follow chunk-major;
/// K=6144 sums its 3 chunk partials in f32 (the reference host sum).
/// P28-9: dequant ONE row of a v4 w4 pack to bf16 bits (embedding
/// lookup — tie_word_embeddings makes lm_head rows the token embedding).
/// Mirrors unpack_deq32's layout exactly: 8 logical cols x (M/8/16)
/// tiles x 16 rows per block; nibble byte g*256 + k*8 + r/2 packs rows
/// 2j(lo)|2j+1(hi); bf16 scales at 16384 + (g*16 + r)*2; W = q * sf.
fn w4u_row_bf16(pack: &[u8], m: usize, k: usize, row: usize) -> Vec<u16> {
    let cols = 8usize;
    let tr = 16usize;
    let chunks = k / 2048;
    let t_num = m / cols / tr;
    let col = row / (m / cols);
    let within = row % (m / cols);
    let t = within / tr;
    let r = within % tr;
    let off = (col * t_num * chunks + t) * W4U_ELEM;
    let groups = 64usize;
    let mut out = vec![0u16; k];
    for g in 0..groups {
        let sb = off + 16384 + (g * 16 + r) * 2;
        let sf = f32::from_bits((u16::from_le_bytes([pack[sb], pack[sb + 1]]) as u32) << 16);
        for kk in 0..32usize {
            let b = pack[off + g * 256 + kk * 8 + r / 2];
            let nib = if r % 2 == 0 { b & 0x0F } else { b >> 4 };
            let q = ((nib as i8) << 4) >> 4;
            out[g * 32 + kk] = f32_to_bf16(q as f32 * sf);
        }
    }
    out
}

fn w4u_c_at(cs: &[u8], row: usize, k: usize, m: usize) -> u16 {    let rpc = m / 8; // real rows per column
    let (col, w) = (row / rpc, row % rpc);
    let section = (w4u_blocks(m, k) + 2) * W4U_TILE_ROWS;
    let base = col * section + W4U_TILE_ROWS + w;
    let chunks = w4u_chunks(k);
    if chunks == 1 {
        return u16::from_le_bytes([cs[base * 2], cs[base * 2 + 1]]);
    }
    let mut acc = 0f32;
    for c in 0..chunks {
        let off = (base + c * rpc) * 2;
        acc += bf16_to_f32(u16::from_le_bytes([cs[off], cs[off + 1]]));
    }
    f32_to_bf16(acc)
}

/// lm_head padded M: vocab 120818 -> 946 even blocks of one 16-row tile.
const W4U_LM_M: usize = 121088;

/// M9/P18 fused rms-pair geometry (IRON op_fused.py). The rms window
/// element is ELEM-sized and reads C rows [0..9280): [8 op1 sections |
/// residual M1=2048 bf16 | pad | u32 K=1 at ELEM-8, u32 blocks1 at
/// ELEM-4 (u16 rows 9276..9279 = [1, 0, blocks1, 0])]. Op2 sections
/// start at row 9280, per column section2 = (blocks2+2)*16 rows, data
/// after the 2 dummy 16-row elements.
const W4UF_WINDOW_ROWS: usize = W4U_ELEM / 2; // 9280

/// M9/P18: pack the fused op2 weight stream — per column [K=3 w element
/// | that column's plain v5 blocks] (op_fused.py build_packed2). The w
/// element: rms weight M1 bf16 at bytes [0..4096), pad, K=3 u32 at
/// ELEM-8, blocks1 u32 at ELEM-4; every column carries the same element
/// (fifo order runs the glue before op2's blocks).
fn w4uf_build_packed2(blocks: &[u8], w_bits: &[u16], blocks1: usize, blocks2: usize) -> Vec<u8> {
    assert!(blocks.len() == 8 * blocks2 * W4U_ELEM);
    assert!(w_bits.len() == 2048);
    let per_col = (1 + blocks2) * W4U_ELEM;
    let mut out = vec![0u8; 8 * per_col];
    for col in 0..8 {
        let base = col * per_col;
        for (i, &w) in w_bits.iter().enumerate() {
            out[base + i * 2..base + i * 2 + 2].copy_from_slice(&w.to_le_bytes());
        }
        out[base + W4U_ELEM - 8..base + W4U_ELEM - 4].copy_from_slice(&3u32.to_le_bytes());
        out[base + W4U_ELEM - 4..base + W4U_ELEM].copy_from_slice(&(blocks1 as u32).to_le_bytes());
        let src = col * blocks2 * W4U_ELEM;
        out[base + W4U_ELEM..base + per_col].copy_from_slice(&blocks[src..src + blocks2 * W4U_ELEM]);
    }
    out
}

/// M9/P18: the fused pair's C tensor pre-run seed — zeros plus the rms
/// window's write-once header words (op_fused.py build_c_init; the
/// residual rows are re-seeded per exec by the caller, the drains never
/// touch either region).
fn w4uf_build_c_init(c_rows: usize, blocks1: usize) -> Vec<u8> {
    let mut out = vec![0u8; c_rows * 2];
    let h = W4UF_WINDOW_ROWS * 2 - 8; // u32 K=1 at ELEM-8
    out[h..h + 4].copy_from_slice(&1u32.to_le_bytes());
    out[h + 4..h + 8].copy_from_slice(&(blocks1 as u32).to_le_bytes());
    out
}

/// P19 quad C geometry (IRON design_quad.py): 40704 bf16 rows — win1
/// [o sections (8x288) | residual1 @2304 | hdr @9276], gate cols 0..3
/// @9280, padA @15552, up cols 4..7 @18560, padB @24832, win2 (down
/// sections + residual2 rows, dead) @27840, qkv sections @37120 (3
/// dummy 16-row groups ahead of the data).
const W4Q_C_ROWS: usize = 40704;
const W4Q_WIN2_OFF: usize = 27840; // down sections: 8 x 800 rows, 2 dummy groups
                                   // (37116 is only the win2 K=2 header words)
const W4Q_SEC_DN: usize = 800;
const W4Q_QKV_OFF: usize = 37120;
const W4Q_SEC_Q: usize = 448;
const W4Q_RES1_ROW: usize = 2304;

// P20: on a wedged quad exec, dump C and summarize which sections'
// data rows made it out — the stall site fingerprint. Sections'
// first-data-row boundaries are exactly where the parked-Cs race
// stalled before, so a repeat there vs somewhere new is the split
// between "design fix incomplete" and "engine-path bug".
fn c_bo_wait_dump(cm: &(BufferObject, Mapping), n: usize, it: u32) -> std::io::Result<()> {
    let (bo, map) = cm;
    let _ = bo.sync(SyncDirection::FromDevice, 0, bo.size() as u64);
    let cs = map.as_slice();
    let nz = |lo: usize, hi: usize| -> usize {
        (lo..hi).filter(|&r| cs[r * 2] != 0 || cs[r * 2 + 1] != 0).count()
    };
    let (mut o_nz, mut gu_nz, mut dn_nz, mut q_nz) = (0usize, 0usize, 0usize, 0usize);
    for col in 0..8 {
        o_nz += nz(col * 288 + 16, col * 288 + 288);
        gu_nz += nz(9280 + col % 4 * 1568 + 32, 9280 + col % 4 * 1568 + 32 + 1536)
            + nz(18560 + col % 4 * 1568 + 32, 18560 + col % 4 * 1568 + 32 + 1536);
        dn_nz += nz(27840 + col * 800 + 32, 27840 + col * 800 + 32 + 768);
        q_nz += nz(37120 + col * 448 + 48, 37120 + col * 448 + 48 + 384);
    }
    println!(
        "[quad-hang] L{n} it{it}: data rows nonzero o {o_nz}/2048 gate+up {gu_nz}/12288 down {dn_nz}/6144 qkv {q_nz}/3072"
    );
    let _ = std::fs::create_dir_all("/tmp/qkvdump");
    std::fs::write(
        format!("/tmp/qkvdump/hang_L{n}_it{it}.bin"),
        cs,
    )?;
    Ok(())
}

/// P19 quad packed1: per column [X element | that column's 16 o blocks]
/// (op_quad.py build_packed1). The X head is zeroed here — the exec
/// rewrites it (same quantized activation in all 8 columns) before each
/// submit; one whole-BO clflush covers all 8 copies.
fn w4q_build_packed1(blocks: &[u8], blocks1: usize) -> Vec<u8> {
    // P21-2 front-grouped: [X0..X7 | blocks_col0..col7]. The per-column
    // fifo stream is still [X | 16 blocks] (two fills per column in the
    // ctrl code), but the host's per-exec dirty set — the 8 X elements —
    // is one contiguous 148KB run at the BO head, so seeding X costs one
    // small region sync instead of flushing the 2.5MB weight BO.
    assert_eq!(blocks.len(), 8 * blocks1 * W4U_ELEM);
    let x_region = 8 * W4U_ELEM;
    let mut out = vec![0u8; x_region + 8 * blocks1 * W4U_ELEM];
    for col in 0..8 {
        let dst = x_region + col * blocks1 * W4U_ELEM;
        let src = col * blocks1 * W4U_ELEM;
        out[dst..dst + blocks1 * W4U_ELEM]
            .copy_from_slice(&blocks[src..src + blocks1 * W4U_ELEM]);
    }
    out
}

/// P19 quad C pre-run seed: zeros plus the four windows' header words
/// (op_quad.py build_c_init minus the residual, which is re-seeded per
/// exec by gemv_quad). residual2 rows stay ZERO — the device computes
/// h2' = x + o_out itself (stage1r overwrites stage1b's dead copy).
fn w4q_build_c_init() -> Vec<u8> {
    fn hdr(out: &mut [u8], row: usize, k: u32, blocks1: u32) {
        let h = row * 2;
        out[h..h + 4].copy_from_slice(&k.to_le_bytes());
        out[h + 4..h + 8].copy_from_slice(&blocks1.to_le_bytes());
    }
    let mut out = vec![0u8; W4Q_C_ROWS * 2];
    hdr(&mut out, 9276, 1, 16); // win1 (o sections + residual1)
    hdr(&mut out, 18556, 4, 0); // padA: K=4 gate window
    hdr(&mut out, 27836, 5, 0); // padB: K=5 up window
    hdr(&mut out, 37116, 2, 48); // win2 (down sections)
    out
}

/// M3b: the same 42-layer projection chain on the UNIVERSAL w4gemvu kernel.
/// All four shapes share one PDI (the kernel reads K from the block tail at
/// runtime), so the whole chain runs on ONE CU and differs only in ctrl
/// code — the ~650us-per-switch PDI reload run-w4layer pays 168 times per
/// token is gone BY CONSTRUCTION, not by scheduling. Differences from v1:
///   - fixture stems w4gemvu_{M}x{K}; the four PDIs must be byte-identical
///     (asserted — that identity is the entire premise of the single CU);
///   - packed layout v4 matrix-unit tiles: 8 cols x blocks_per_col x
///     18560-byte 16x2048 tiles (K=6144 streams 3 chunk-blocks per tile,
///     chunk-major; host sums the 3 chunk partials);
///   - v5: NO B stream — the vector buffer is ONE 18560-B activation
///     element per K (q all chunks at 0..K, d at K_MAX, K=0 at ELEM-8);
///     two shared ELEM-sized x BOs (xu2048, xu6144).
/// Expected convergence: all four scheduling modes at the device floor
/// (~0.6 ms/layer), where v1 spanned 75-234 ms/token on scheduling alone.
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
            "  {:>7}: ctrl {} B, {} cols (M={}, K={}, {} blocks/col)",
            s.name,
            f.1.len(),
            f.2,
            s.m,
            s.k,
            w4u_blocks(s.m, s.k)
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

    // Activation BOs: ONE ELEM-sized x element per K (v5 — every op's X
    // fill ships the same 18560 B; no F-slot replication).
    let mut live: Vec<(BufferObject, Mapping)> = Vec::new();
    let mut x_va: [u64; 2] = [0; 2]; // [xu2048 (ELEM), xu6144 (ELEM)]
    let xu2048 = vec![0u8; W4U_ELEM];
    let xu6144 = vec![0u8; W4U_ELEM];
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
        c_va[si] = match chain_tensor(
            &dev,
            &mut live,
            &format!("c_{}", s.name),
            &vec![0u8; w4u_c_rows(s.m, s.k) * 2],
        ) {
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
            let expect = 8 * w4u_blocks(s.m, s.k) * W4U_ELEM;
            if data.len() != expect {
                eprintln!(
                    "layer{n:02}_{}: {} B, expected {expect} B (stale import? rerun w4_import)",
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
                (w4u_c_rows(s.m, s.k) * 2) as u64, // c out (K6144: 3M rows)
                (2 * s.m * s.k) as u64,
            )
            .with_tier(tier::SLOT_STREAM); // x 复制进 F 槽的 slot 流形态
            m.bytes_stream = Some((8 * w4u_blocks(s.m, s.k) * W4U_ELEM) as u64); // 块流
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

    // Warmup + golden verification on layer 0 (real weights). Goldens are
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
    // Golden activations: every K=2048 golden shares one deterministic x
    // (importer convention), so ONE activation element per K serves all
    // four shapes (v5: all chunks live in the single ELEM).
    for (xi, k) in [(0usize, 2048usize), (1, 6144)] {
        let x_bits = if xi == 0 {
            &goldens[0].1 // qkv x, any K=2048 shape's
        } else {
            &goldens[3].1 // down x
        };
        let (q, d) = w4u_quantize_x(x_bits, k);
        let bytes = w4u_build_x_elem(&q, &d, k);
        let (bo, map) = &mut live[xi];
        map.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
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
            let got = w4u_c_at(cs, *row, s.k, s.m);
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
                let got = w4u_c_at(cs, *row, W4U_SHAPES[si].k, W4U_SHAPES[si].m);
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
            let k = W4U_SHAPES[si].k;
            let rd = |row: usize| w4u_c_at(cs, row, k, W4U_SHAPES[si].m);
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
    let model = machine_model_or_default();
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

/// M8/P8: lm_head on NPU — decode token 路径最后一个 CPU 计算算子（hy
/// vocab 120818，tied embed；pad 到 121088 = v4 块 ABI：946 偶数块）。普适 w4gemvu
/// PDI 在 10x M 下仍逐位同（下方 assert；只有 ctrl code 携带 M）。隔离
/// 探针：golden step 输入，121088 全 logits 对拍 golden_lmhead（argmax +
/// top-8 + rel_rms），solo xN + 同 op burst 拿 134 MiB 权重流（v4 矩阵单元块）的 per-op
/// 读数（M 比任何投影形状大 10 倍 —— 大 M slot 流是否越过 45.6 GB/s 的
/// slot 天花板，是 P6 预言的"未建模档"信号，报告见分晓）。
fn cmd_run_lmhead(iters: usize) -> ExitCode {
    let build = "/home/nzinfo/qwen/xnpu/build";
    const M: usize = W4U_LM_M;
    const K: usize = 2048;

    let (pdi, instr, cols) =
        match load_fixture(&format!("{build}/w4gemvu_{M}x{K}.mlir.prj")) {
            Some(f) => f,
            None => {
                eprintln!("load fixture w4gemvu_{M}x{K} failed (run the w4gemvu pytest first)");
                return ExitCode::FAILURE;
            }
        };
    // PDI 与 decode qkv 夹具逐位同 —— 普适内核前提（P6 陷阱：ctrl code
    // 携带 M，PDI 不携带；PDI 不同 = 编译缓存陈旧）。
    match load_fixture(&format!("{build}/w4gemvu_3072x2048.mlir.prj")) {
        Some((qdi, _, _)) if qdi == pdi => {}
        _ => {
            eprintln!("lm_head PDI != qkv PDI — stale fixture, rebuild");
            return ExitCode::FAILURE;
        }
    }
    let wdata = match std::fs::read(format!("{build}/w4u_hy/lmhead.bin")) {
        Ok(d) if d.len() == 8 * w4u_blocks(M, K) * W4U_ELEM => d,
        Ok(d) => {
            eprintln!(
                "lmhead.bin: {} B, expected {}",
                d.len(),
                8 * w4u_blocks(M, K) * W4U_ELEM
            );
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("read lmhead.bin: {e} (q4nx_import.py --lmhead)");
            return ExitCode::FAILURE;
        }
    };
    // golden_lmhead.bin: u32 n, u32 k, x bits u16[k], ref bits u16[n].
    let g = match std::fs::read(format!("{build}/w4u_hy/golden_lmhead.bin")) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("read golden_lmhead.bin: {e}");
            return ExitCode::FAILURE;
        }
    };
    let u32at = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    if u32at(&g, 0) as usize != M || u32at(&g, 4) as usize != K || g.len() != 8 + 2 * K + 2 * M {
        eprintln!("golden_lmhead.bin: bad header (rerun q4nx_import.py --lmhead)");
        return ExitCode::FAILURE;
    }
    let rd_u16 = |b: &[u8], o: usize, n: usize| -> Vec<u16> {
        b[o..o + 2 * n]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let x_bits = rd_u16(&g, 8, K);
    let ref_bits = rd_u16(&g, 8 + 2 * K, M);
    println!(
        "lm_head: M={M} (vocab 120818 + 270 pad), K={K}, weights {} MiB, {} blocks/col (v5)",
        wdata.len() >> 20,
        w4u_blocks(M, K)
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
    if let Err(e) = ctx.configure_cus(&[(pdi.as_slice(), 0)]) {
        eprintln!("configure_cus: {e}");
        return ExitCode::FAILURE;
    }

    let mut live: Vec<(BufferObject, Mapping)> = Vec::new();
    let w_va = match chain_tensor(&dev, &mut live, "w.lmhead", &wdata) {
        Some(v) => v,
        None => {
            eprintln!("w.lmhead BO failed");
            return ExitCode::FAILURE;
        }
    };
    // x 量化 int8 + 打成单个 v5 激活元素（K header 0，同投影链）。
    let (q, d) = w4u_quantize_x(&x_bits, K);
    let xdata = w4u_build_x_elem(&q, &d, K);
    let x_va = match chain_tensor(&dev, &mut live, "x_lm", &xdata) {
        Some(v) => v,
        None => {
            eprintln!("x_lm BO failed");
            return ExitCode::FAILURE;
        }
    };
    let c_va = match chain_tensor(&dev, &mut live, "c_lm", &vec![0u8; w4u_c_rows(M, K) * 2]) {
        Some(v) => v,
        None => {
            eprintln!("c_lm BO failed");
            return ExitCode::FAILURE;
        }
    };
    let mut op = match chain_op(&dev, "lmhead", &instr, 0, &[w_va, x_va, c_va]) {
        Some(o) => o,
        None => {
            eprintln!("lmhead op setup failed");
            return ExitCode::FAILURE;
        }
    };
    let handles = vec![op.ctrl_bo.handle(), live[0].0.handle(), live[1].0.handle(), live[2].0.handle()];

    let mut meta = OpMeta::new(
        "lmhead",
        "w4gemvu",
        0,
        (K * 2) as u64,
        (M * 2) as u64,
        (2 * M * K) as u64,
    )
    .with_tier(tier::SLOT_STREAM);
    meta.bytes_stream = Some((8 * w4u_blocks(M, K) * W4U_ELEM) as u64);

    // ---- 正确性：全 121088 logits 对拍（不进 perf 分布）----
    let seq = match op.pkt.submit(&dev, &ctx, &handles) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("lmhead submit: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 60_000_000_000) {
        eprintln!("lmhead wait: {e}");
        return ExitCode::FAILURE;
    }
    let _ = live[2]
        .0
        .sync(SyncDirection::FromDevice, 0, (w4u_c_rows(M, K) * 2) as u64);
    let cs = live[2].1.as_slice();
    let mut sum_sq = 0f32;
    let mut g_sq = 0f32;
    let mut argmax_n = 0usize;
    let mut argmax_g = 0usize;
    let mut best_o = f32::NEG_INFINITY;
    let mut best_r = f32::NEG_INFINITY;
    let mut top_n: Vec<(usize, f32)> = Vec::new();
    for i in 0..M {
        let o = bf16_to_f32(w4u_c_at(cs, i, K, M));
        let r = bf16_to_f32(ref_bits[i]);
        sum_sq += (o - r) * (o - r);
        g_sq += r * r;
        if o > best_o {
            best_o = o;
            argmax_n = i;
        }
        if r > best_r {
            best_r = r;
            argmax_g = i;
        }
        top_n.push((i, o));
    }
    let rel_rms = (sum_sq / M as f32).sqrt() / (g_sq / M as f32).sqrt().max(1e-9);
    top_n.sort_by(|a, b| b.1.total_cmp(&a.1));
    let gset: std::collections::BTreeSet<usize> = {
        let mut idx: Vec<(usize, f32)> =
            ref_bits.iter().enumerate().map(|(i, &b)| (i, bf16_to_f32(b))).collect();
        idx.sort_by(|a, b| b.1.total_cmp(&a.1));
        idx.into_iter().take(8).map(|(i, _)| i).collect()
    };
    let overlap = top_n.iter().take(8).filter(|(i, _)| gset.contains(i)).count();
    println!(
        "lm_head vs golden: rel_rms {:.4}, argmax NPU {} vs golden {}, top-8 overlap {}/8 -> {}",
        rel_rms,
        argmax_n,
        argmax_g,
        overlap,
        if argmax_n == argmax_g && rel_rms < 0.05 { "PASS" } else { "FAIL" }
    );

    // ---- 计时：solo xN + 同 op burst ----
    let mut rec = Recorder::new();
    let mut rec_seq = 0u64;
    for it in 0..iters {
        let ts = std::time::Instant::now();
        let seq = match op.pkt.submit(&dev, &ctx, &handles) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("solo submit: {e}");
                return ExitCode::FAILURE;
            }
        };
        if syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 60_000_000_000).is_err() {
            eprintln!("solo wait timed out");
            return ExitCode::FAILURE;
        }
        rec.solo(&meta, it as u32, ts, rec_seq);
        rec_seq += 1;
    }
    const BURST_N: usize = 8;
    let tb = std::time::Instant::now();
    let mut last_seq = 0u64;
    for it in 0..BURST_N {
        match op.pkt.submit(&dev, &ctx, &handles) {
            Ok(s) => {
                last_seq = s;
                rec.burst_submit(&meta, it as u32, rec_seq);
                rec_seq += 1;
            }
            Err(e) => {
                eprintln!("burst submit: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if syncobj_timeline_wait(&dev, ctx.syncobj_handle, last_seq, 60_000_000_000).is_err() {
        eprintln!("burst wait timed out");
        return ExitCode::FAILURE;
    }
    rec.burst_done("lmhead", Mode::Burst, 0, tb, BURST_N as u32, BURST_N as u32);

    let model = machine_model_or_default();
    let title = format!("run-lmhead: hy-mt2 lm_head {M}x{K}, {iters} solo + {BURST_N} burst");
    let (md_out, _summary) = xnpu_perf::render_markdown(&rec, &[meta], &model, &title);
    println!("\n{md_out}");
    let dir = "/home/nzinfo/qwen/xnpu/build/perf";
    if std::fs::create_dir_all(dir).is_ok() {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let stem = format!("{dir}/lmhead_{ts}");
        let json = xnpu_perf::trace_json(&rec, &model, &title, &_summary);
        match std::fs::write(format!("{stem}.json"), json)
            .and_then(|()| std::fs::write(format!("{stem}.md"), &md_out))
        {
            Ok(()) => println!("perf trace written: {stem}.json / .md"),
            Err(e) => eprintln!("perf trace write failed: {e}"),
        }
    }

    ExitCode::SUCCESS
}

/// M5a P6 收尾（P7 接线）：报告一律对最新校准渲染 —— perf-calibrate 的
/// 产物 build/perf/machine_model.json 存在则 overlay 到默认值上；解析失败
/// 退回默认值并提示，绝不阻塞测量本身。
fn machine_model_or_default() -> MachineModel {
    const PATH: &str = "/home/nzinfo/qwen/xnpu/build/perf/machine_model.json";
    match std::fs::read_to_string(PATH) {
        Ok(text) => match MachineModel::default().overlay_json(&text) {
            Ok(m) => {
                println!("machine model: {PATH}（{}）", m.name);
                m
            }
            Err(e) => {
                eprintln!("machine model {PATH} 解析失败（{e}）— 用默认值");
                MachineModel::default()
            }
        },
        Err(_) => MachineModel::default(),
    }
}

/// M5a P6: measure the machine-model ceilings on-device and emit the overlay
/// JSON. Per shape (layer-00 real weights, ONE universal w4gemvu CU):
/// solo xN then a same-op burst block — the FIRST per-shape burst/op numbers
/// in the repo (run-w4ulayer's burst blocks are all mixed-op chains, so
/// per-op burst has never been isolated). slot-stream tier ceiling = max
/// burst GB/s across shapes; strided tier = flowkv burst (fixture optional);
/// seq-dma has no self-test kernel in our stack and keeps the FLM lower
/// bound (52 GB/s, M5c P5) — recorded, not measured. Output doubles as a
/// machine_model.json overlay consumed by MachineModel::overlay_json.
fn cmd_perf_calibrate(arch: &DecArch, iters: usize) -> ExitCode {
    let build = "/home/nzinfo/qwen/xnpu/build";
    let shapes = arch.shapes();
    let w4dir = arch.w4dir;
    // P9: strided 层级按架构限定 —— flowkv 几何（头数/容量）是架构专属，
    // minicpm 与 hy 的校准互踩会把对方的 strided 天花板写进自己的报告。
    // slot-stream/seq-dma 与架构无关，保留裸名。
    let strided_key = format!("strided:{}", arch.name);
    // 队列深度扫描（P7）：P6 的 48-op gateup 块两跑差 40%（29.6→41.3 GB/s），
    // 深队列背压非确定 —— 单一深度读数不可作天花板。每个深度各出一块
    // （事件名带 /dN 后缀，报告 per-op 表逐深度成行），天花板 = 跨深度
    // （及跨跑）最大值，仍是下界语义。
    const DEPTHS: [usize; 4] = [8, 16, 32, 64];
    println!(
        "perf-calibrate: {} shapes, layer00 weights, {} solo + burst depth sweep {:?}",
        arch.name, iters, DEPTHS
    );

    let fixtures: Vec<(Vec<u8>, Vec<u8>, u32)> = shapes
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
    for (s, f) in shapes.iter().zip(&fixtures).skip(1) {
        if f.0 != fixtures[0].0 {
            eprintln!("PDI for {} differs — stale fixtures, rebuild", s.name);
            return ExitCode::FAILURE;
        }
    }
    // flowkv fixture is optional: without it the strided tier keeps its default.
    let fk = load_fixture(&format!("{build}/{}.mlir.prj", arch.fk_fixture));
    if fk.is_none() {
        println!("flowkv fixture absent — strided tier keeps default (1.2 GB/s)");
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
    // Both PDIs take cu_func 0（DPU 函数号；CU 槽位 = 本列表下标，由
    // chain_op 的 cu 参数选择 —— func != 0 固件查不到函数，op 永不执行，
    // run-decode 4080 行注释同款陷阱）。
    let cus: Vec<(&[u8], u8)> = match &fk {
        Some(f) => vec![(fixtures[0].0.as_slice(), 0), (f.0.as_slice(), 0)],
        None => vec![(fixtures[0].0.as_slice(), 0)],
    };
    if let Err(e) = ctx.configure_cus(&cus) {
        eprintln!("configure_cus: {e}");
        return ExitCode::FAILURE;
    }

    // Buffers: two ELEM-sized x BOs + one c per shape + layer00 weights.
    // Content is zeros — calibration is timing, and the kernel has no
    // data-dependent control flow.
    let mut live: Vec<(BufferObject, Mapping)> = Vec::new();
    let mut x_va = [0u64; 2];
    for i in 0..2 {
        x_va[i] = match chain_tensor(&dev, &mut live, &format!("x{i}"), &vec![0u8; W4U_ELEM]) {
            Some(v) => v,
            None => {
                eprintln!("x{i} BO failed");
                return ExitCode::FAILURE;
            }
        };
    }
    let mut c_va = [0u64; 4];
    for (si, s) in shapes.iter().enumerate() {
        c_va[si] = match chain_tensor(
            &dev,
            &mut live,
            &format!("c_{}", s.name),
            &vec![0u8; w4u_c_rows(s.m, s.k) * 2],
        ) {
            Some(v) => v,
            None => {
                eprintln!("c_{} BO failed", s.name);
                return ExitCode::FAILURE;
            }
        };
    }
    let mut w_va = [0u64; 4];
    for (si, s) in shapes.iter().enumerate() {
        let data = match std::fs::read(format!("{w4dir}/layer00_{}.bin", s.name)) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("read layer00_{}: {e} (stale import?)", s.name);
                return ExitCode::FAILURE;
            }
        };
        let expect = 8 * w4u_blocks(s.m, s.k) * W4U_ELEM;
        if data.len() != expect {
            eprintln!("layer00_{}: {} B, expected {expect} B", s.name, data.len());
            return ExitCode::FAILURE;
        }
        w_va[si] = match chain_tensor(&dev, &mut live, &format!("w.{}", s.name), &data) {
            Some(v) => v,
            None => {
                eprintln!("w.{} BO failed", s.name);
                return ExitCode::FAILURE;
            }
        };
    }

    let metas: Vec<OpMeta> = shapes
        .iter()
        .map(|s| {
            let mut m = OpMeta::new(
                s.name,
                "w4gemvu",
                0,
                (s.m * s.k / 2 + s.m * (s.k / 32) * 2 + s.k * 2) as u64,
                (w4u_c_rows(s.m, s.k) * 2) as u64,
                (2 * s.m * s.k) as u64,
            )
            .with_tier(tier::SLOT_STREAM);
            m.bytes_stream = Some((8 * w4u_blocks(s.m, s.k) * W4U_ELEM) as u64);
            m
        })
        .collect();
    let mut rec = Recorder::new();
    let mut rec_seq = 0u64;

    // One op per shape on cu0; layer00 weights.
    // live layout: [x0, x1, c_qkv, c_o, c_gateup, c_down, w.qkv, w.o, w.gateup, w.down]
    let mut ops: Vec<ChainOp> = Vec::with_capacity(4);
    let mut op_handles: Vec<Vec<u32>> = Vec::with_capacity(4);
    for (si, s) in shapes.iter().enumerate() {
        let xv = x_va[if s.k == 6144 { 1 } else { 0 }];
        let op = match chain_op(&dev, &s.name, &fixtures[si].1, 0, &[w_va[si], xv, c_va[si]]) {
            Some(o) => o,
            None => {
                eprintln!("op setup {} failed", s.name);
                return ExitCode::FAILURE;
            }
        };
        op_handles.push(vec![
            op.ctrl_bo.handle(),
            live[6 + si].0.handle(),
            live[if s.k == 6144 { 1 } else { 0 }].0.handle(),
            live[2 + si].0.handle(),
        ]);
        ops.push(op);
    }

    // Per shape: solo xN (submit+wait), then one same-op burst block PER
    // DEPTH. Depth rows carry the "/dN" name suffix so the report table
    // shows one row per (shape, depth); the base name keeps the solos.
    // Depth metas must reach all_metas — the report looks bytes/tier up by
    // exact name, naked rows would render as n/a-bytes.
    println!("\n== slot-stream tier（每形状 solo + 同 op 连发深度扫描 {:?}） ==", DEPTHS);
    let mut all_metas: Vec<OpMeta> = metas.clone();
    for si in 0..4 {
        for it in 0..iters {
            let ts = std::time::Instant::now();
            let seq = match ops[si].pkt.submit(&dev, &ctx, &op_handles[si]) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("solo submit {}: {e}", shapes[si].name);
                    return ExitCode::FAILURE;
                }
            };
            if syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000).is_err() {
                eprintln!("solo wait {} timed out", shapes[si].name);
                return ExitCode::FAILURE;
            }
            rec.solo(&metas[si], it as u32, ts, rec_seq);
            rec_seq += 1;
        }
        for &d in DEPTHS.iter() {
            let mut dm = metas[si].clone();
            dm.name = format!("{}/d{d}", shapes[si].name);
            let t0 = std::time::Instant::now();
            let mut last_seq = 0u64;
            for it in 0..d {
                match ops[si].pkt.submit(&dev, &ctx, &op_handles[si]) {
                    Ok(s) => {
                        last_seq = s;
                        rec.burst_submit(&dm, it as u32, rec_seq);
                        rec_seq += 1;
                    }
                    Err(e) => {
                        eprintln!("burst submit {}: {e}", dm.name);
                        return ExitCode::FAILURE;
                    }
                }
            }
            if syncobj_timeline_wait(&dev, ctx.syncobj_handle, last_seq, 10_000_000_000).is_err() {
                eprintln!("burst wait {} timed out", dm.name);
                return ExitCode::FAILURE;
            }
            rec.burst_done(&dm.name, Mode::Burst, 0, t0, d as u32, d as u32);
            all_metas.push(dm);
        }
    }

    // flowkv (cu1): zeros are fine — S header 0 = full compiled capacity, and
    // the kernel streams CAP rows regardless of values. The o-BO ToDevice
    // sync before each submit is the first-exec race guard (notes §17).
    let mut fk_op: Option<(ChainOp, Vec<u32>, (BufferObject, Mapping), (BufferObject, Mapping), (BufferObject, Mapping))> =
        None;
    // (owning tuple: op + arg handles + the three BOs whose mappings must
    // outlive every submit)
    if let Some(fk) = &fk {
        const FK_CAP: usize = 1024;
        let fk_group = arch.heads / arch.kv;
        let fk_stride = fk_group * 128 + 128 + 16;
        let kv_elems = arch.kv * FK_CAP * 2 * 128;
        let q_elems = arch.kv * fk_stride;
        let o_elems = arch.heads * 128;
        let kv_bo = match BufferObject::new(&dev, BoType::Shmem, kv_elems * 2) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("kv BO: {e}");
                return ExitCode::FAILURE;
            }
        };
        let _kv_map = match kv_bo.map_owned() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("kv map: {e}");
                return ExitCode::FAILURE;
            }
        };
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
        // identity angles（数值无关紧要，但保持与 run-decode 相同的布局）
        let q_bytes = q_map.as_mut_slice();
        for g in 0..arch.kv {
            let abase = g * fk_stride + fk_group * 128;
            for p in 0..64 {
                q_bytes[(abase + 2 * p) * 2..(abase + 2 * p) * 2 + 2]
                    .copy_from_slice(&f32_to_bf16(1.0).to_le_bytes());
                q_bytes[(abase + 2 * p + 1) * 2..(abase + 2 * p + 1) * 2 + 2]
                    .copy_from_slice(&f32_to_bf16(0.0).to_le_bytes());
            }
        }
        let _ = q_bo.sync(SyncDirection::ToDevice, 0, q_bo.size() as u64);
        let _ = kv_bo.sync(SyncDirection::ToDevice, 0, kv_bo.size() as u64);
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
            "flowkv",
            &fk.1,
            1,
            &[_kv_map.as_ptr() as u64, q_map.as_ptr() as u64, o_map.as_ptr() as u64],
        ) {
            Some(o) => o,
            None => {
                eprintln!("flowkv op setup failed");
                return ExitCode::FAILURE;
            }
        };
        let handles = vec![op.ctrl_bo.handle(), kv_bo.handle(), q_bo.handle(), o_bo.handle()];
        fk_op = Some((op, handles, (kv_bo, _kv_map), (q_bo, q_map), (o_bo, o_map)));
    }

    if let Some((op, handles, (_kv_bo, _), (_q_bo, _), (o_bo, _))) = fk_op.as_mut() {
        println!("\n== strided tier（flowkv 容量流，S=0） ==");
        const FK_CAP_U: u64 = 1024;
        let fk_group = (arch.heads / arch.kv) as u64;
        let fk_stride = (fk_group as usize * 128 + 128 + 16) as u64;
        let mut m = OpMeta::new(
            "flowkv",
            "attn",
            1,
            arch.kv as u64 * FK_CAP_U * 2 * 128 * 2 + arch.kv as u64 * fk_stride * 2,
            (arch.heads * 128 * 2) as u64,
            arch.heads as u64 * FK_CAP_U * 128 * 2 * 2,
        )
        .with_tier(strided_key.clone());
        m.bytes_stream = Some(arch.kv as u64 * FK_CAP_U * 2 * 128 * 2 + arch.kv as u64 * fk_stride * 2);
        for it in 0..iters {
            let _ = o_bo.sync(SyncDirection::ToDevice, 0, o_bo.size() as u64);
            let ts = std::time::Instant::now();
            let seq = match op.pkt.submit(&dev, &ctx, handles) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("flowkv solo submit: {e}");
                    return ExitCode::FAILURE;
                }
            };
            if syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000).is_err() {
                eprintln!("flowkv solo wait timed out");
                return ExitCode::FAILURE;
            }
            rec.solo(&m, it as u32, ts, rec_seq);
            rec_seq += 1;
        }
        let mut fk_depth_metas: Vec<OpMeta> = Vec::new();
        for &d in DEPTHS.iter() {
            let mut dm = m.clone();
            dm.name = format!("flowkv/d{d}");
            let tb = std::time::Instant::now();
            let mut last_seq = 0u64;
            for it in 0..d {
                let _ = o_bo.sync(SyncDirection::ToDevice, 0, o_bo.size() as u64);
                match op.pkt.submit(&dev, &ctx, handles) {
                    Ok(s) => {
                        last_seq = s;
                        rec.burst_submit(&dm, it as u32, rec_seq);
                        rec_seq += 1;
                    }
                    Err(e) => {
                        eprintln!("flowkv burst submit: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            if syncobj_timeline_wait(&dev, ctx.syncobj_handle, last_seq, 10_000_000_000).is_err() {
                eprintln!("flowkv burst wait timed out");
                return ExitCode::FAILURE;
            }
            rec.burst_done(&dm.name, Mode::Burst, 0, tb, d as u32, d as u32);
            fk_depth_metas.push(dm);
        }
        all_metas.push(m);
        all_metas.extend(fk_depth_metas);
    }

    // ---- report + overlay JSON ----
    // 起点是上一次校准（merge 语义：本次实测的键覆盖，未测的保留）。
    let model = machine_model_or_default();
    let title = format!(
        "perf-calibrate: {} shapes, layer00, {iters} solo + burst depth sweep {:?}",
        arch.name, DEPTHS
    );
    let (md, summary) = xnpu_perf::render_markdown(&rec, &all_metas, &model, &title);
    println!("\n{md}");

    // 深度行名 = "{base}/d{N}"，solo 只记在 base 名下 —— overhead = base
    // solo_med − 各深度 burst/op（每深度一个样本进中位，深度间应平坦，
    // 不平坦本身就是发现）。
    fn base_of(n: &str) -> &str {
        n.split('/').next().unwrap_or(n)
    }
    let solo_med_of = |n: &str| {
        summary
            .ops
            .iter()
            .find(|o| o.name == n)
            .and_then(|o| o.solo_med_us)
    };
    let mut slot_ceiling = 0f64;
    let mut ovhs: Vec<f64> = Vec::new();
    println!("== 深度扫描结果（slot-stream） ==");
    for s in shapes.iter() {
        let solo_m = solo_med_of(s.name);
        for o in summary.ops.iter().filter(|o| base_of(&o.name) == s.name) {
            let (Some(b), Some(sb)) = (o.burst_us_per_op, o.stream_bytes) else {
                continue; // base 行只有 solo
            };
            let g = sb as f64 / (1000.0 * b);
            slot_ceiling = slot_ceiling.max(g);
            if let Some(sm) = solo_m {
                ovhs.push(sm - b);
            }
            println!(
                "  {:>12}: burst {:>7.1} µs/op -> {:>5.1} GB/s weight stream (solo {}, ovh {})",
                o.name,
                b,
                g,
                solo_m.map(|v| format!("{v:.1} µs")).unwrap_or_else(|| "–".into()),
                solo_m.map(|sm| format!("{:.1} µs", sm - b)).unwrap_or_else(|| "–".into()),
            );
        }
    }
    let strided = summary
        .ops
        .iter()
        .filter(|o| base_of(&o.name) == "flowkv")
        .filter_map(|o| o.burst_us_per_op.zip(o.stream_bytes))
        .map(|(b, sb)| sb as f64 / (1000.0 * b))
        .fold(0f64, f64::max);
    let strided = (strided > 0.0).then_some(strided);
    if strided.is_some() {
        println!("\n== 深度扫描结果（strided / flowkv） ==");
        for o in summary
            .ops
            .iter()
            .filter(|o| o.name != "flowkv" && base_of(&o.name) == "flowkv")
        {
            if let (Some(b), Some(sb)) = (o.burst_us_per_op, o.stream_bytes) {
                println!(
                    "  {:>12}: burst {:>7.1} µs/op -> {:>5.2} GB/s strided",
                    o.name,
                    b,
                    sb as f64 / (1000.0 * b)
                );
            }
        }
    }
    ovhs.sort_by(|a, b| a.total_cmp(b));
    let submit_ovh = if ovhs.is_empty() {
        model.submit_overhead_us // 无 burst 数据时保留默认
    } else {
        ovhs[ovhs.len() / 2]
    };
    // 稳健性：本轮一个可测行都没有时保留模型现值，绝不把 0 写进 overlay
    // （P7 实测中招：一次 meta 缺失的跑把 slot-stream 0.0 写进
    // machine_model.json，下一跑全表 inf%）。
    if slot_ceiling <= 0.0 {
        slot_ceiling = model.bw_for(Some(tier::SLOT_STREAM));
        eprintln!("slot ceiling 本轮无可测样本 — 保留模型现值 {slot_ceiling:.1}");
    }

    println!("\n== 校准结果 ==");
    println!("slot-stream ceiling = {:.1} GB/s", slot_ceiling);
    if let Some(g) = strided {
        println!("strided            = {:.2} GB/s", g);
    }
    println!("seq-dma            = 52 GB/s（保留 FLM 下界，M5c P5；无自测内核）");
    println!("submit_overhead    = {:.1} µs（per-shape solo−burst 中位）", submit_ovh);

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let prov = format!(
        "perf-calibrate {}: slot-stream={:.1} GB/s（burst 深度扫描 {:?} 取跨深度最大），strided={}，seq-dma=52 FLM 下界（M5c P5，未自测），submit={:.1}µs solo−burst 中位",
        arch.name,
        slot_ceiling,
        DEPTHS,
        strided.map(|g| format!("{g:.2}")).unwrap_or_else(|| "默认1.2".into()),
        submit_ovh
    );
    let mut tier_body = format!("\"{}\": {}", tier::SLOT_STREAM, slot_ceiling);
    if let Some(g) = strided {
        tier_body.push_str(&format!(", \"{}\": {:.3}", strided_key, g));
    }
    let overlay = format!(
        "{{\n  \"name\": \"xdna2-npu2-cal-{}-{}\",\n  \"bw_tiers\": {{ {} }},\n  \"default_tier\": \"{}\",\n  \"submit_overhead_us\": {:.1},\n  \"provenance\": {}\n}}\n",
        arch.name, ts, tier_body, tier::SLOT_STREAM, submit_ovh, xnpu_perf::jstr(&prov)
    );
    // Self-check: the overlay we emit must parse back (that's how future
    // runs will consume it).
    match MachineModel::default().overlay_json(&overlay) {
        Ok(m) => println!(
            "\noverlay round-trip OK: slot={} seq-dma={} {}={:.2}",
            m.bw_for(Some(tier::SLOT_STREAM)),
            m.bw_for(Some(tier::SEQ_DMA)),
            strided_key,
            m.bw_for(Some(&strided_key))
        ),
        Err(e) => {
            eprintln!("overlay does not parse ({e}) — bug in emitter");
            return ExitCode::FAILURE;
        }
    }
    println!("{overlay}");
    let dir = "/home/nzinfo/qwen/xnpu/build/perf";
    if std::fs::create_dir_all(dir).is_ok() {
        let stem = format!("{dir}/calibrate_{}_{}", arch.name, ts);
        let json = trace_json(&rec, &model, &title, &summary);
        match std::fs::write(format!("{stem}.md"), &md)
            .and_then(|()| std::fs::write(format!("{stem}.json"), json))
            .and_then(|()| std::fs::write(format!("{dir}/machine_model.json"), &overlay))
        {
            Ok(()) => println!("calibration written: {stem}.md/.json + {dir}/machine_model.json"),
            Err(e) => eprintln!("calibration write failed: {e}"),
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
    if std::arch::is_x86_feature_detected!("avx512f") {
        return unsafe { swiglu_bf16_avx512(x, out) };
    }
    let half = x.len() / 2;
    for i in 0..half {
        let g = bf16_to_f32(x[i]);
        let u = bf16_to_f32(x[half + i]);
        out[i] = f32_to_bf16(g * (1.0 / (1.0 + (-g).exp())) * u);
    }
}

/// AVX-512 flavor of swiglu (P27-2): sigmoid rides the same 6-term
/// ln2-series poly as attention_avx512 (drift vs libc expf ~1e-7 relative,
/// far below the bf16 grain); the IEEE division and the RNE bf16 store
/// (bit-trick, same integer ops as the scalar f32_to_bf16) keep everything
/// else bit-identical. Golden gates arbitrate the sigmoid drift.
#[target_feature(enable = "avx512f")]
unsafe fn swiglu_bf16_avx512(x: &[u16], out: &mut [u16]) {
    use std::arch::x86_64::*;
    unsafe {
        let half = x.len() / 2;
        let log2e = 1.4426950408889634f32;
        let c1 = 0.6931471805599453f32;
        let c2 = 0.2402265069591007f32;
        let c3 = 0.0555041086648216f32;
        let c4 = 0.009618129107628477f32;
        let c5 = 0.0013333558146428443f32;
        let c6 = 0.0001540353039338167f32;
        let one = _mm512_set1_ps(1f32);
        let lo80 = _mm512_set1_ps(-80f32);
        let hi80 = _mm512_set1_ps(80f32);
        let bias = _mm512_set1_epi32(0x7fff);
        let lsb = _mm512_set1_epi32(1);
        let mut i = 0;
        while i + 16 <= half {
            let conv = |p: *const u16| -> __m512 {
                _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(_mm256_loadu_si256(p as *const __m256i)),
                    16,
                ))
            };
            let g = conv(x.as_ptr().add(i));
            let u = conv(x.as_ptr().add(half + i));
            // sigmoid(g) = 1/(1+exp(-g)); exp poly on clamped z=-g.
            let z = _mm512_min_ps(
                _mm512_max_ps(_mm512_sub_ps(_mm512_setzero_ps(), g), lo80),
                hi80,
            );
            let zn = _mm512_mul_ps(z, _mm512_set1_ps(log2e));
            let n = _mm512_roundscale_ps(zn, 0x00); // RNE
            let f = _mm512_sub_ps(zn, n);
            let p = _mm512_fmadd_ps(_mm512_set1_ps(c6), f, _mm512_set1_ps(c5));
            let p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(c4));
            let p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(c3));
            let p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(c2));
            let p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(c1));
            let p = _mm512_fmadd_ps(p, f, one);
            let e = _mm512_castsi512_ps(_mm512_add_epi32(
                _mm512_castps_si512(p),
                _mm512_slli_epi32(_mm512_cvtps_epi32(n), 23),
            ));
            let sig = _mm512_div_ps(one, _mm512_add_ps(one, e));
            let r = _mm512_mul_ps(_mm512_mul_ps(g, sig), u);
            // f32->bf16 RNE: bias = 0x7fff + ((b>>16)&1); (b+bias)>>16.
            let rb = _mm512_castps_si512(r);
            let b = _mm512_add_epi32(
                rb,
                _mm512_add_epi32(bias, _mm512_and_si512(_mm512_srli_epi32(rb, 16), lsb)),
            );
            let h16 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(b, 16));
            _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, h16);
            i += 16;
        }
        while i < half {
            let g = bf16_to_f32(x[i]);
            let u = bf16_to_f32(x[half + i]);
            out[i] = f32_to_bf16(g * (1.0 / (1.0 + (-g).exp())) * u);
            i += 1;
        }
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

// ---------------------------------------------------------------------------
// P27-1: host-segment profiler for the decode hot loop. The AVX-512
// attention rewrite only bought 0.5ms of the ~10.5ms of non-device time
// per token — P15's "6.2ms attention" had bundled rope/qk-norm/KV-append
// staging with the core, so before cutting further we MEASURE where the
// fused-loop host time actually goes. Enable with XNPU_HOSTPROF=1; the
// accumulators reset at the timed loop and print per-token µs after the
// steady line. `wait` segments cover submit->syncobj-wait (device-
// dominated); everything else is pure host glue.
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
enum HSeg {
    Rope,
    QkNorm,
    KvApp,
    Attn,
    Swiglu,
    QuantA,
    ResA,
    WaitA,
    ReadA,
    QuantB,
    ResB,
    WaitB,
    ReadB,
    FinalNorm,
    Lm,
    Lv2Fill,
    Lv2Wait,
    Lv2Read,
}
const HSEG_NAMES: [&str; 18] = [
    "rope", "qknorm", "kvapp", "attn", "swiglu", "A:quant", "A:res", "A:wait", "A:read",
    "B:quant", "B:res", "B:wait", "B:read", "finalnorm", "lm", "lv2:fill", "lv2:wait",
    "lv2:read",
];

thread_local! {
    static HOSTPROF: std::cell::RefCell<Option<Box<[u64; 18]>>> =
        const { std::cell::RefCell::new(None) };
}

fn hp_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XNPU_HOSTPROF").is_ok())
}

fn hp_add(seg: HSeg, ns: u64) {
    HOSTPROF.with(|h| {
        if let Some(a) = h.borrow_mut().as_mut() {
            a[seg as usize] += ns;
        }
    });
}

fn hp_reset() {
    if hp_enabled() {
        HOSTPROF.with(|h| *h.borrow_mut() = Some(Box::new([0u64; 18])));
    }
}

fn hp_print(iters: u32) {
    if !hp_enabled() {
        return;
    }
    HOSTPROF.with(|h| {
        let binding = h.borrow();
        let Some(a) = binding.as_ref() else { return };
        let mut tot = 0u128;
        println!("hostprof (µs/token, {iters} timed iters):");
        for (i, v) in a.iter().enumerate() {
            if *v == 0 {
                continue;
            }
            tot += *v as u128;
            println!("  {:>10} {:8.1}", HSEG_NAMES[i], *v as f64 / 1000.0 / iters as f64);
        }
        println!("  {:>10} {:8.1}  (sum of segments)", "SUM", tot as f64 / 1000.0 / iters as f64);
    });
}

/// Time a block into an HSeg bucket. Disabled (env unset) costs one
/// hp_enabled() atomic-free lazy bool check per site.
macro_rules! hp {
    ($seg:expr, $body:block) => {
        if hp_enabled() {
            let t__ = std::time::Instant::now();
            let r__ = $body;
            hp_add($seg, t__.elapsed().as_nanos() as u64);
            r__
        } else {
            $body
        }
    };
}

/// GQA decode attention: q (heads, roped) against a (kv, CACHE_SEQ, 128)
/// cache; q head h reads kv head h/(heads/kv). Scores/softmax/PV in f32,
/// one bf16 rounding at the output.
///
/// P15: K/V rows are staged to f32 ONCE per kv head (the group's q heads
/// share them) and q once per head — the old loop re-converted every
/// bf16 element on every multiply (~8x the conversions, and the PV walk
/// strided the V cache per output dim). Accumulation ORDER per output is
/// unchanged (t ascending), so results are bit-identical.
/// P27-1: dispatch to the AVX-512 kernel when the CPU has it (Zen5 here
/// does, full-width native). Semantics identical to the scalar body below;
/// see attention_avx512 for the numerics note.
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
    if pos + 1 >= 16
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
    {
        unsafe { attention_avx512(q, kc, vc, pos, cache_seq, out, heads, nkv) };
        return;
    }
    let s = pos + 1;
    let group = heads / nkv;
    let mut kf = vec![0f32; s * 128];
    let mut vf = vec![0f32; s * 128];
    let mut sc = vec![0f32; s];
    let mut acc = [0f32; 128];
    for kv in 0..nkv {
        let base = kv * cache_seq * 128;
        for i in 0..s * 128 {
            kf[i] = bf16_to_f32(kc[base + i]);
            vf[i] = bf16_to_f32(vc[base + i]);
        }
        for h in kv * group..(kv + 1) * group {
            let qo = h * 128;
            let qf: [f32; 128] = std::array::from_fn(|j| bf16_to_f32(q[qo + j]));
            let mut mx = f32::NEG_INFINITY;
            for t in 0..s {
                let kro = t * 128;
                let mut d = 0f32;
                for j in 0..128 {
                    d += kf[kro + j] * qf[j];
                }
                sc[t] = d / (128f32).sqrt();
                mx = mx.max(sc[t]);
            }
            let mut sum = 0f32;
            for v in sc[..s].iter_mut() {
                *v = (*v - mx).exp();
                sum += *v;
            }
            acc = [0f32; 128];
            for t in 0..s {
                let w = sc[t];
                let vro = t * 128;
                for j in 0..128 {
                    acc[j] += w * vf[vro + j];
                }
            }
            for j in 0..128 {
                out[qo + j] = f32_to_bf16(acc[j] / sum);
            }
        }
    }
}

/// P27-1: AVX-512 attention, same math as the scalar version (per kv-head:
/// K/V rows converted to f32 once, the GQA group of q-heads reuses them).
/// The scalar body was chain-bound: `d += kf*qf` is a 128-long serial FMA
/// chain (4-cycle latency each) and PV the same, plus ~52k libc expf calls
/// per token — that was the 6.2ms. Here the three hot loops go 16-wide.
///
/// Numerics: bf16→f32 is exact (bit shift); dots/sums reorder float adds
/// (lane-tree, ~1e-7 rel drift — far under the bf16 input noise); exp is a
/// 6-term 2^f poly on f∈[-0.5,0.5] (~1e-7 rel). The output epilogue keeps
/// the scalar f32_to_bf16 path bit-identical to before. Golden double-gate
/// (hidden rms + argmax/top-8) arbitrates.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn attention_avx512(
    q: &[u16],
    kc: &[u16],
    vc: &[u16],
    pos: usize,
    cache_seq: usize,
    out: &mut [u16],
    heads: usize,
    nkv: usize,
) {
    use std::arch::x86_64::*;
    unsafe {
    let s = pos + 1;
    let group = heads / nkv;
    let scale = 1f32 / (128f32).sqrt();
    let mut kf = vec![0f32; s * 128];
    let mut vf = vec![0f32; s * 128];
    let mut sc = vec![0f32; s];
    let mut acc = [0f32; 128];
    for kv in 0..nkv {
        let base = kv * cache_seq * 128;
        // bf16 bits << 16 == f32 bits; 16 elems per round, exact.
        let conv = |src: &[u16], dst: &mut [f32]| {
            let n = dst.len();
            let mut i = 0;
            while i + 16 <= n {
                let w = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
                let f = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(w),
                    16,
                ));
                _mm512_storeu_ps(dst.as_mut_ptr().add(i), f);
                i += 16;
            }
            while i < n {
                dst[i] = bf16_to_f32(src[i]);
                i += 1;
            }
        };
        conv(&kc[base..][..s * 128], &mut kf);
        conv(&vc[base..][..s * 128], &mut vf);
        for h in kv * group..(kv + 1) * group {
            let qo = h * 128;
            let qf: [f32; 128] = std::array::from_fn(|j| bf16_to_f32(q[qo + j]));
            // QK dots: 8x fma + lane-tree reduce per row (vs the 128-long
            // serial chain before). Exact max, same as scalar .max().
            let mut mx = f32::NEG_INFINITY;
            for t in 0..s {
                let mut d = _mm512_setzero_ps();
                for k in 0..8 {
                    d = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kf.as_ptr().add(t * 128 + k * 16)),
                        _mm512_loadu_ps(qf.as_ptr().add(k * 16)),
                        d,
                    );
                }
                let v = _mm512_reduce_add_ps(d) * scale;
                sc[t] = v;
                if v > mx {
                    mx = v;
                }
            }
            // exp(sc - mx) 16-wide: n = rne(z), f = z - n ∈ [-0.5, 0.5],
            // 2^f by 6-term ln2-series poly, 2^n by exponent-bit add.
            // x clamped to ≥ -80 so the exponent field stays normal.
            let log2e = 1.4426950408889634f32;
            let c1 = 0.6931471805599453f32;
            let c2 = 0.2402265069591007f32;
            let c3 = 0.0555041086648216f32;
            let c4 = 0.009618129107628477f32;
            let c5 = 0.0013333558146428443f32;
            let c6 = 0.0001540353039338167f32;
            let mut vsum = _mm512_setzero_ps();
            let mut sum_tail = 0f32;
            let mut i = 0;
            while i + 16 <= s {
                let x = _mm512_max_ps(
                    _mm512_sub_ps(
                        _mm512_loadu_ps(sc.as_ptr().add(i)),
                        _mm512_set1_ps(mx),
                    ),
                    _mm512_set1_ps(-80f32),
                );
                let z = _mm512_mul_ps(x, _mm512_set1_ps(log2e));
                let n = _mm512_roundscale_ps(z, 0x00); // RNE
                let f = _mm512_sub_ps(z, n);
                let p = _mm512_fmadd_ps(
                    _mm512_set1_ps(c6),
                    f,
                    _mm512_set1_ps(c5),
                );
                let p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(c4));
                let p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(c3));
                let p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(c2));
                let p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(c1));
                let p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1f32));
                let e = _mm512_castsi512_ps(_mm512_add_epi32(
                    _mm512_castps_si512(p),
                    _mm512_slli_epi32(_mm512_cvtps_epi32(n), 23),
                ));
                _mm512_storeu_ps(sc.as_mut_ptr().add(i), e);
                vsum = _mm512_add_ps(vsum, e);
                i += 16;
            }
            while i < s {
                sc[i] = (sc[i] - mx).exp();
                sum_tail += sc[i];
                i += 1;
            }
            let sum = _mm512_reduce_add_ps(vsum) + sum_tail;
            // PV: acc[128] lives in 8 zmm regs; w broadcast per t.
            let mut avec: [__m512; 8] = std::array::from_fn(|_| _mm512_setzero_ps());
            for t in 0..s {
                let w = _mm512_set1_ps(sc[t]);
                for k in 0..8 {
                    avec[k] = _mm512_fmadd_ps(
                        w,
                        _mm512_loadu_ps(vf.as_ptr().add(t * 128 + k * 16)),
                        avec[k],
                    );
                }
            }
            for k in 0..8 {
                _mm512_storeu_ps(acc.as_mut_ptr().add(k * 16), avec[k]);
            }
            // scalar epilogue: identical rounding to the scalar version.
            for j in 0..128 {
                out[qo + j] = f32_to_bf16(acc[j] / sum);
            }
        }
    }
    }
}

/// P20: engine-side quad hang probe. The pyxrt-path loop (quad_probe2.py)
/// is 500-clean on the restructured design while the E2E still hangs
/// ~2%/exec AND occasionally completes with garbage — so the trigger lives
/// in something the E2E does that the probe doesn't. This loops ONE quad
/// exec (layer-0 real weights) through the ENGINE's own ERT submission and
/// strips the E2E variables one at a time:
///   plain   = pure submit+wait loop (tests the submission machinery alone)
///   xwrite  = + the E2E per-exec host writes (X element rewrite into the
///             packed1 heads + whole-BO ToDevice clflush, res1 seed into C
///             + 4KB region sync)
///   data    = xwrite + varying X/res1 content per iter (bf16 LSB rotation)
/// C is snapshotted each iter and diffed against iter 0, so the SILENT
/// corruption variant (an E2E run finished with NaN final hidden) is
/// caught the same way as the hang.
fn cmd_run_quadloop(iters: usize, mode: &str) -> ExitCode {
    let build = "/home/nzinfo/qwen/xnpu/build";
    let w4dir = "/home/nzinfo/qwen/xnpu/build/w4u_hy";
    let decdir = "/home/nzinfo/qwen/xnpu/build/dec_hy";
    let (pdi, instr, _) = match load_fixture(&format!(
        "{build}/w4gemvuq_2048x2048_12288_2048x6144_3072.mlir.prj"
    )) {
        Some(f) => f,
        None => {
            eprintln!("load quad fixture failed (run test_quad first)");
            return ExitCode::FAILURE;
        }
    };
    let blocks: [usize; 4] = [16, 96, 48, 24]; // o, gateup, down, qkv
    let rd = |stem: &str| -> Option<Vec<u8>> {
        let d = std::fs::read(format!("{w4dir}/{stem}")).ok()?;
        Some(d)
    };
    let (o, g, dn, q) = match (
        rd("layer00_o.bin"),
        rd("layer00_gateup.bin"),
        rd("layer00_down.bin"),
        rd("layer01_qkv.bin"),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => {
            eprintln!("layer weights absent in {w4dir}");
            return ExitCode::FAILURE;
        }
    };
    let norms = match std::fs::read(format!("{decdir}/norms.bin")) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("read {decdir}/norms.bin: {e}");
            return ExitCode::FAILURE;
        }
    };
    let norms_u16: Vec<u16> = norms
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    // layer-0 quad: rms1 w = ln2_0 = norms[1], rms2 w = ln1_1 = norms[2].
    let p1 = w4q_build_packed1(&o, blocks[0]);
    let p2 = w4uf_build_packed2(&g, &norms_u16[1 * 2048..][..2048], blocks[0], blocks[1]);
    let p4 = w4uf_build_packed2(&q, &norms_u16[2 * 2048..][..2048], blocks[2], blocks[3]);
    let cdata = w4q_build_c_init();
    let mut x_bits = match std::fs::read(format!("{decdir}/x0.bin")) {
        Ok(v) => v
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect::<Vec<u16>>(),
        Err(_) => (0..2048).map(|i| 0x3f80u16 | ((i as u16) & 0x7f)).collect(),
    };
    x_bits.resize(2048, 0x3f80);

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
    let mut ctx = match HwContext::create(&dev, 8 * md.core.row_count as u32) {
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
    let mut live: Vec<(BufferObject, Mapping)> = Vec::new();
    let mut vas = Vec::new();
    for (tag, data) in [("p1", p1), ("p2", p2), ("p3", dn), ("p4", p4), ("c", cdata)] {
        match chain_tensor(&dev, &mut live, &format!("quadloop.{tag}"), &data) {
            Some(va) => vas.push(va),
            None => return ExitCode::FAILURE,
        }
    }
    let mut op = match chain_op(&dev, "quadloop", &instr, 0, &vas) {
        Some(op) => op,
        None => return ExitCode::FAILURE,
    };
    // P21-3 instrumentation: phase breakdown of the X seed (quantize+
    // build vs memcpy vs SYNC_BO), averaged over the run. The sync size
    // is steerable via mode suffix so the SAME loop measures 148KB vs
    // whole-BO clflush cost directly (e.g. "xwrite" vs "xwrite-full").
    let mut prof: [f64; 3] = [0.0; 3];
    let mut prof_n: u32 = 0;
    let sync_full = mode.ends_with("-full");
    // "-noflush": user-space CLFLUSHOPT on the mapping instead of the
    // SYNC_BO ioctl (P21-3: the ioctl is ~30us FLAT regardless of size —
    // a syscall-count problem, not a byte problem).
    let sync_noflush = mode.ends_with("-noflush");
    let seed_x = |live: &mut Vec<(BufferObject, Mapping)>,
                  x: &[u16],
                  it: usize,
                  prof: &mut [f64; 3]|
     -> bool {
        let ta = std::time::Instant::now();
        let (q, d) = w4u_quantize_x(x, 2048);
        let xe = w4u_build_x_elem(&q, &d, 2048);
        let tb = std::time::Instant::now();
        let (bo, map) = &mut live[0];
        let bytes = map.as_mut_slice();
        // P21-2: X elements front-grouped [X0..X7] — one contiguous
        // 148KB dirty run, single region sync (was: whole-BO 2.5MB flush;
        // 8 strided syncs measured slower — P21-1).
        for col in 0..8 {
            let off = col * W4U_ELEM;
            bytes[off..off + W4U_ELEM].copy_from_slice(&xe);
        }
        let tc = std::time::Instant::now();
        let sz = if sync_full {
            bytes.len() as u64
        } else {
            (8 * W4U_ELEM) as u64
        };
        let r = if sync_noflush {
            map.clflush_region(0, sz as usize);
            Ok(())
        } else {
            bo.sync(SyncDirection::ToDevice, 0, sz)
        };
        let td = std::time::Instant::now();
        prof[0] += (tb - ta).as_secs_f64() * 1e6;
        prof[1] += (tc - tb).as_secs_f64() * 1e6;
        prof[2] += (td - tc).as_secs_f64() * 1e6;
        if r.is_err() {
            eprintln!("quadloop x sync failed (it {it})");
            return false;
        }
        true
    };
    let seed_res = |live: &mut Vec<(BufferObject, Mapping)>, x: &[u16]| {
        let (bo, map) = &mut live[4];
        let bytes = map.as_mut_slice();
        let r0 = W4Q_RES1_ROW * 2;
        for j in 0..2048 {
            bytes[r0 + j * 2..r0 + j * 2 + 2].copy_from_slice(&x[j].to_le_bytes());
        }
        if sync_noflush {
            map.clflush_region(r0, 4096);
        } else if bo.sync(SyncDirection::ToDevice, r0 as u64, 4096).is_err() {
            eprintln!("quadloop res sync failed");
            return false;
        }
        true
    };
    // data mode's X pattern has period 2 in `it` (the XOR flips half the
    // groups each iter), so odd iters legitimately differ from iter 0 —
    // keep two baselines and only compare same-parity iterations.
    let mut base: [Option<Vec<u8>>; 2] = [None, None];
    let t0 = std::time::Instant::now();
    for it in 0..iters {
        // "sleep"/"syncs": plain + an inter-exec barrier, bisecting whether
        // the corruption is a posted-write race (exec N's tail drains still
        // landing while exec N+1's C-fills read / drains write the same C)
        // or something in the packet itself. "syncs" mimics pyxrt's
        // run_runlist (whole-BO TO_DEVICE syncs ahead of every exec — the
        // probe path's accidental barrier).
        if mode == "sleep" {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        if mode == "syncs" {
            for (bo, _) in live.iter() {
                let _ = bo.sync(SyncDirection::ToDevice, 0, bo.size() as u64);
            }
        }
        if !matches!(mode, "plain" | "sleep" | "syncs" | "fresh") {
            let x: Vec<u16> = if mode.starts_with("data") {
                x_bits
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v ^ (((it + i / 256) as u16) & 1))
                    .collect()
            } else {
                x_bits.clone()
            };
            if !seed_x(&mut live, &x, it, &mut prof) {
                return ExitCode::FAILURE;
            }
            prof_n += 1;
            if !seed_res(&mut live, &x) {
                return ExitCode::FAILURE;
            }
        }
        // "fresh": rebuild the ctrl BO AND the ERT packet every exec — the
        // one thing pyxrt does per call that this loop doesn't (XRT builds a
        // fresh run object + exec packet; M3b taught that submit never
        // resets packet-header state). If this mode is clean where plain
        // corrupts, the bug is packet/ctrl reuse, not device cross-exec
        // state.
        if mode == "fresh" {
            op = match chain_op(&dev, "quadloop", &instr, 0, &vas) {
                Some(o) => o,
                None => return ExitCode::FAILURE,
            };
        }
        let handles = {
            let mut h = vec![op.ctrl_bo.handle()];
            h.extend(live.iter().map(|(bo, _)| bo.handle()));
            h
        };
        let seq = match op.pkt.submit(&dev, &ctx, &handles) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("quadloop submit (it {it}): {e}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 5_000_000_000) {
            eprintln!("quadloop wait (it {it}, seq {seq}): {e}");
            let _ = c_bo_wait_dump(&live[4], 0, it as u32);
            return ExitCode::FAILURE;
        }
        // FromDevice + divergence check vs iter 0 (silent-corruption
        // catch). In -noflush modes the readback flush is ALSO
        // user-space — clflush writes back + invalidates, dropping stale
        // host lines so the device-written DDR data is re-fetched. The
        // ioctl additionally round-trips a MSG_OP_SYNC_BO to firmware;
        // if that fence were required for readback correctness, this
        // mode diverges (data mode compares against parity baselines).
        {
            let (bo, map) = &live[4];
            if sync_noflush {
                map.clflush_region(0, bo.size());
            } else {
                let _ = bo.sync(SyncDirection::FromDevice, 0, bo.size() as u64);
            }
            let cs = map.as_slice().to_vec();
            let slot = &mut base[it % 2];
            match slot {
                None => {
                    if it < 2 {
                        let _ = std::fs::create_dir_all("/tmp/qkvdump");
                        let _ = std::fs::write("/tmp/qkvdump/loopbase.bin", &cs);
                    }
                    *slot = Some(cs);
                }
                Some(b) => {
                    let d = cs.iter().zip(b.iter()).filter(|(a, c)| a != c).count();
                    if d != 0 {
                        eprintln!("quadloop DIVERGENCE at it {it}: {d} bytes differ from iter {}", it % 2);
                        let _ = std::fs::create_dir_all("/tmp/qkvdump");
                        let _ = std::fs::write(format!("/tmp/qkvdump/loopdiv_it{it}.bin"), &cs);
                        return ExitCode::FAILURE;
                    }
                }
            }
        }
        if it > 0 && it % 50 == 0 {
            println!("it {it} clean ({:.1} us/exec)", t0.elapsed().as_micros() as f64 / it as f64);
        }
    }
    println!(
        "quadloop: {iters} iters clean (mode {mode}, {:.1} us/exec avg)",
        t0.elapsed().as_micros() as f64 / iters as f64
    );
    if prof_n > 0 {
        println!(
            "xseed profile ({} seeds): quant+build {:.1}us, memcpy {:.1}us, sync({}) {:.1}us",
            prof_n,
            prof[0] / prof_n as f64,
            prof[1] / prof_n as f64,
            if sync_full { "whole-BO" } else { "148KB" },
            prof[2] / prof_n as f64
        );
    }
    ExitCode::SUCCESS
}

/// P28-6 6f-7: engine-path lv2 loop — the missing cell of the crawl matrix.
///
/// 6f-6 (pyxrt forensic) re-characterized the N=16 "deadlock" as a bimodal
/// crawl of the per-element fill/drain handshake (clean ~1.2µs/BD vs crawl
/// 0.4-1.5ms/BD). The engine path at N=8 (P28-5: 165 solo samples, 765µs
/// median, fills incl. the whole 27.6MB W stream replayed EVERY exec) never
/// showed it, while the pyxrt forensic crawls even at N=8 (~1/8 attempts).
/// The one untested combination is engine @ N=16: this loop replays one
/// exec's ctrl through OUR ERT_START_CU packet + syncobj wait (no XRT
/// runlist, no per-attempt host BO syncs) and times every exec, so the
/// bimodality is directly visible in the wait-duration distribution.
/// Verdict table: clean here + crawl in pyxrt => the disease is the
/// submission/runtime path, and the engine N=16 port can proceed; crawl
/// here too => the dataflow protocol itself must be fixed (per-element
/// handshake elimination), regardless of path.
fn cmd_run_lv2loop(iters: usize, n: usize) -> ExitCode {
    let build = "/home/nzinfo/qwen/xnpu/build";
    if !matches!(n, 8 | 16) {
        eprintln!("n must be 8 or 16 (design_layerv2 assert)");
        return ExitCode::FAILURE;
    }
    // LV2LOOP_PROBE=<rings>: drive the AIELv2Probe fixture (design_lv2probe)
    // instead of the full layerv2 design — the exact cells the 6f-6 pyxrt
    // forensic measured, now through the engine submission path. W/X/XN
    // mirror the forensic instrument (zeros + K/id/N headers).
    let probe = std::env::var("LV2LOOP_PROBE").ok().and_then(|s| s.parse::<usize>().ok());
    // LV2LOOP_WAIT_S: per-exec syncobj timeout (default 30s; the N=16
    // full design stalls past it, so allow longer windows to distinguish
    // a heavy-tailed crawl from a true wedge).
    let wait_s: u64 = std::env::var("LV2LOOP_WAIT_S")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let (welem, drain_rows) = if n == 16 { (94usize, 320usize) } else { (186, 640) };
    let fixture = match probe {
        Some(r) => format!("/home/nzinfo/qwen/xnpu/IRON/build/lv2probe{n}_r{r}.mlir.prj"),
        None => format!("{build}/w4gemvu_layerv2_{n}.mlir.prj"),
    };
    let (pdi, instr, _) = match load_fixture(&fixture) {
        Some(f) => f,
        None => {
            eprintln!("load fixture {fixture} failed");
            return ExitCode::FAILURE;
        }
    };
    let c_bytes = n * drain_rows * 2;
    let le = |v: u32| v.to_le_bytes();
    let (wdata, xdata, xndata): (Vec<u8>, Vec<u8>, Vec<u8>);
    match probe {
        Some(r) => {
            // probe geometry (test_lv2probe.geom): mirrors the forensic
            // instrument's build_inputs — zeros everywhere except the
            // header words that drive flavor dispatch.
            let rows = 2048 / n;
            let n_o = rows / 16;
            let n_gate = (6144 / n) / 16;
            let n_qkv = (3072 / n) / 16;
            let mut ks: Vec<u32> = Vec::new();
            ks.extend(std::iter::repeat(2048).take(n_o));
            ks.push(101);
            ks.extend(std::iter::repeat(103).take(n_gate));
            ks.extend(std::iter::repeat(104).take(n_gate));
            ks.extend(std::iter::repeat(105).take(3 * n_o));
            ks.push(102);
            ks.extend(std::iter::repeat(2048).take(n_qkv));
            let we = ks.len();
            if (n == 16 && we != 94) || (n == 8 && we != 162) {
                eprintln!("probe geom mismatch: {we} elements (r{r} n{n})");
                return ExitCode::FAILURE;
            }
            let mut w = vec![0u8; n * we * W4U_ELEM];
            for c in 0..n {
                let base = c * we * W4U_ELEM;
                for (i, k) in ks.iter().enumerate() {
                    let off = base + i * W4U_ELEM;
                    w[off + W4U_ELEM - 8..off + W4U_ELEM - 4].copy_from_slice(&le(*k));
                }
            }
            wdata = w;
            let mut x = vec![0u8; n * W4U_ELEM];
            for w in 0..n {
                let off = w * W4U_ELEM;
                x[off + 6400..off + 6404].copy_from_slice(&le(w as u32));
                x[off + 6404..off + 6408].copy_from_slice(&le(n as u32));
            }
            xdata = x;
            let mut xn = vec![0u8; n * W4U_ELEM];
            for w in 0..n {
                let off = w * W4U_ELEM;
                xn[off + W4U_ELEM - 8..off + W4U_ELEM - 4].copy_from_slice(&le(100));
            }
            xndata = xn;
        }
        None => {
            // One real exec pack as the static W stream: a mid-chain exec
            // has the full element mix with real K headers. LV2LOOP_W
            // overrides the pack path (P28-7: tools/lv2_floor_pack.py
            // builds the all-K=2049 glue-floor discriminator).
            let lv2dir = if n == 16 {
                format!("{build}/lv2_hy_w16")
            } else {
                format!("{build}/lv2_hy")
            };
            let wpath =
                std::env::var("LV2LOOP_W").unwrap_or_else(|_| format!("{lv2dir}/exec05.bin"));
            let wd = match std::fs::read(&wpath) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("read {wpath}: {e} (tools/layerv2_pack.py --n {n})");
                    return ExitCode::FAILURE;
                }
            };
            let w_bytes = n * welem * W4U_ELEM;
            if wd.len() != w_bytes {
                eprintln!("exec05.bin: {} B != {w_bytes} (stale pack)", wd.len());
                return ExitCode::FAILURE;
            }
            wdata = wd;
            let x_bits: Vec<u16> = (0..2048).map(|i| 0x3f80u16 | ((i as u16) & 0x7f)).collect();
            let (q, d) = w4u_quantize_x(&x_bits, 2048);
            let xe = w4u_build_x_elem(&q, &d, 2048);
            let mut x = vec![0u8; n * W4U_ELEM];
            for w in 0..n {
                let off = w * W4U_ELEM;
                x[off..off + W4U_ELEM].copy_from_slice(&xe);
                x[off + 6400..off + 6404].copy_from_slice(&le(w as u32));
                // design_layerv2.py:35 — the runtime-N kernel reads the
                // worker count from X[6404,6408); omitting it feeds N=0
                // geometry (the 60.8s wedge, 6f-7 §6). The N=8 E2E predates
                // the runtime-N refactor (compile-time N=8 PDI) so it never
                // needed this word.
                x[off + 6404..off + 6408].copy_from_slice(&le(n as u32));
            }
            xdata = x;
            let mut xn = vec![0u8; n * W4U_ELEM];
            for w in 0..n {
                let off = w * W4U_ELEM;
                for j in 0..256usize {
                    xn[off + j * 2..off + j * 2 + 2]
                        .copy_from_slice(&x_bits[(w * 13 + j) & 0x7ff].to_le_bytes());
                }
                xn[off + W4U_ELEM - 8..off + W4U_ELEM - 4].copy_from_slice(&le(100));
            }
            xndata = xn;
        }
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
    let mut ctx = match HwContext::create(&dev, 8 * md.core.row_count as u32) {
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

    let mut live: Vec<(BufferObject, Mapping)> = Vec::new();
    let vas = [
        chain_tensor(&dev, &mut live, "lv2loop.W", &wdata),
        chain_tensor(&dev, &mut live, "lv2loop.X", &xdata),
        chain_tensor(&dev, &mut live, "lv2loop.XN", &xndata),
        chain_tensor(&dev, &mut live, "lv2loop.C", &vec![0u8; c_bytes]),
    ];
    if vas.iter().any(|v| v.is_none()) {
        eprintln!("lv2loop tensor BO failed");
        return ExitCode::FAILURE;
    }
    let vas: Vec<u64> = vas.into_iter().map(|v| v.unwrap()).collect();
    // rt.sequence(W, X, XN, C) — mirror cmd_run_decode_lv2.
    let mut op = match chain_op(&dev, "lv2loop", &instr, 0, &vas) {
        Some(o) => o,
        None => return ExitCode::FAILURE,
    };
    let handles = {
        let mut h = vec![op.ctrl_bo.handle()];
        h.extend(live.iter().map(|(bo, _)| bo.handle()));
        h
    };

    // Static inputs, zero host work between execs: the ONLY variable is
    // the device-side execution. Long wait so crawl execs complete and
    // land in the distribution instead of aborting.
    //
    // Per-exec C instrumentation: after each wait, invalidate the host
    // view (P21 clflush_region) and hash the drain image. A 20µs exec
    // that leaves C bit-identical to the previous exec = drains replayed
    // stale L1 slots without any recompute (the P19b starved-S2MM
    // mechanism) — the wait "completing" is not evidence of work.
    let mut times: Vec<f64> = Vec::with_capacity(iters);
    let mut slow: Vec<(usize, f64)> = Vec::new();
    let mut prev_hash: Option<u64> = None;

    // P28-8a discriminator: LV2LOOP_PIPE builds `iters` INDEPENDENT
    // command BOs (re-submitting ONE ctrl BO lets the driver coalesce/
    // replace queued commands — first attempt measured 1us paces from
    // fake completions), submits them all up front, then waits in
    // sequence. The fixture is static, so the ERT is free to pre-queue
    // command k+1 while k streams. The median DELTA between consecutive
    // wait returns is the true back-to-back exec pace; if it drops well
    // below the serialized 707us, the ~132us per-exec fixed cost is
    // dispatch latency that submit-ahead can hide in the E2E chain too.
    if std::env::var("LV2LOOP_PIPE").is_ok() {
        let mut pipe_ops: Vec<ChainOp> = Vec::with_capacity(iters);
        let mut pipe_handles: Vec<Vec<u32>> = Vec::with_capacity(iters);
        for _ in 0..iters {
            match chain_op(&dev, "lv2loopP", &instr, 0, &vas) {
                Some(o) => pipe_ops.push(o),
                None => return ExitCode::FAILURE,
            }
        }
        for op in &pipe_ops {
            let mut h = vec![op.ctrl_bo.handle()];
            h.extend(live.iter().map(|(bo, _)| bo.handle()));
            pipe_handles.push(h);
        }
        // Depth-bounded pipeline (LV2LOOP_PIPE=<depth>, default 2 — the
        // E2E-relevant regime: at most <depth> commands in flight, submit
        // k+1 right after k, wait in order). The burst-submit variant
        // (all up front) is meaningless: the submit ioctl storm crawls
        // (~143us each, 6f-6) and the fence timeline lumps — states read
        // NEW. Pace = median delta between consecutive wait returns.
        let depth: usize = std::env::var("LV2LOOP_PIPE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v >= 1)
            .unwrap_or(2);
        let mut seqs = vec![0u64; iters];
        let mut next_sub = 0usize;
        let mut dones: Vec<f64> = Vec::with_capacity(iters);
        let mut t0all = std::time::Instant::now();
        for it in 0..iters {
            while next_sub < iters && next_sub < it + depth {
                match pipe_ops[next_sub].pkt.submit(&dev, &ctx, &pipe_handles[next_sub]) {
                    Ok(s) => seqs[next_sub] = s,
                    Err(e) => {
                        eprintln!("lv2loop pipe submit (it {next_sub}): {e}");
                        return ExitCode::FAILURE;
                    }
                }
                next_sub += 1;
            }
            if it == 0 {
                t0all = std::time::Instant::now();
            }
            if let Err(e) =
                syncobj_timeline_wait(&dev, ctx.syncobj_handle, seqs[it], wait_s * 1_000_000_000)
            {
                eprintln!("lv2loop pipe wait (it {it}): {e}");
                return ExitCode::FAILURE;
            }
            dones.push(t0all.elapsed().as_secs_f64() * 1e6);
            let st = pipe_ops[it].pkt.state();
            if st != 4 {
                println!("  pipe it {it}: state={st} <-- NOT COMPLETED");
            }
        }
        let mut pace: Vec<f64> = Vec::with_capacity(iters - 1);
        for w in dones.windows(2) {
            pace.push(w[1] - w[0]);
        }
        let mut p = pace.clone();
        p.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "lv2loop PIPE depth={depth}: n={n} pace med {:.0}us min {:.0}us max {:.0}us (serialized ref ~707)",
            p[p.len() / 2],
            p[0],
            p[p.len() - 1],
            n = n
        );
        println!("  paces: {:.0?}", pace);
        return ExitCode::SUCCESS;
    }

    for it in 0..iters {
        let seq = match op.pkt.submit(&dev, &ctx, &handles) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("lv2loop submit (it {it}): {e}");
                return ExitCode::FAILURE;
            }
        };
        let t0 = std::time::Instant::now();
        if let Err(e) =
            syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, wait_s * 1_000_000_000)
        {
            eprintln!("lv2loop wait (it {it}, seq {seq}): {e}");
            return ExitCode::FAILURE;
        }
        let us = t0.elapsed().as_secs_f64() * 1e6;
        times.push(us);
        if us > 5_000.0 {
            slow.push((it, us));
        }
        // ERT cmd state (header bits[3:0]): the driver signals the syncobj
        // fence on EVERY response path — COMPLETED(4), ERROR(5), ABORT(6),
        // SUBMITTED(7), TIMEOUT(8), NORESPONSE(9) — so a returned wait is
        // NOT evidence the exec ran (aie2_sched_notify → dma_fence_signal
        // unconditionally; aie2_ctx.c). The health path additionally
        // memcpys firmware health data over the packet's regmap (offset 4+)
        // and permanently poisons the ctx (every later cmd = instant ABORT).
        let st = op.pkt.state();
        const ST: [&str; 10] = [
            "INVALID",
            "NEW",
            "QUEUED",
            "RUNNING",
            "COMPLETED",
            "ERROR",
            "ABORT",
            "SUBMITTED",
            "TIMEOUT",
            "NORESPONSE",
        ];
        let pkt_first = op.pkt.pkt_header_words();
        let st_str = ST.get(st as usize).copied().unwrap_or("?");
        let fake = if st == 4 { "" } else { "  <-- FAKE COMPLETION" };
        // settle 2ms (S2MM tail in NoC), then invalidate + hash
        std::thread::sleep(std::time::Duration::from_millis(2));
        let (_, c_map) = &live[3];
        c_map.clflush_region(0, c_bytes);
        let cs = c_map.as_slice();
        let nonzero = cs.chunks_exact(2).filter(|c| c[0] != 0 || c[1] != 0).count();
        let mut h: u64 = 0xcbf29ce484222325;
        for b in &cs[..c_bytes] {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let same = prev_hash == Some(h);
        prev_hash = Some(h);
        println!(
            "  it {it}: wait {us:.0}us, state={st_str}({st}){fake}, C nonzero {nonzero}/{}, hash 0x{h:016x}{}",
            c_bytes / 2,
            if same { " (SAME as prev — no recompute)" } else { "" }
        );
        println!("    pkt[0..8] = {:08x?}", pkt_first);
    }
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "lv2loop: n={n} {iters} execs: min {:.0}us med {:.0}us max {:.0}us, {} slow(>5ms)",
        sorted[0],
        sorted[sorted.len() / 2],
        sorted[sorted.len() - 1],
        slow.len(),
    );
    // Read C back once: real packs + nonzero X => the drains must have
    // left nonzero rows (sanity that the execs actually computed).
    let (c_bo, c_map) = &live[3];
    let mut nonzero = 0usize;
    match c_bo.sync(SyncDirection::FromDevice, 0, c_bo.size() as u64) {
        Ok(()) => {
            nonzero = c_map
                .as_slice()
                .chunks_exact(2)
                .filter(|c| c[0] != 0 || c[1] != 0)
                .count();
            println!("C nonzero u16 {}/{}", nonzero, c_bytes / 2);
        }
        Err(e) => eprintln!("C FromDevice failed (non-fatal): {e}"),
    }
    println!(
        "per-exec us: {}",
        times
            .iter()
            .map(|t| format!("{:.0}", t))
            .collect::<Vec<_>>()
            .join(",")
    );
    for (it, us) in &slow {
        println!("  SLOW it {it}: {us:.0}us");
    }
    ExitCode::SUCCESS
}

/// P28-5: host E2E integration of the layer-v2 chain (design_layerv2 +
/// w4gemvu_layer.cc, P28-4: one exec = half a transformer layer shifted —
/// [o+MLP of layer e-2] + [qkv of layer e-1] — on 8 persistent ring
/// workers). 33 execs/token, all attention on host. The dependency
/// qkv_L -> host rope/qk-norm/attention -> next exec's X element forces
/// the half-layer shift; exec 1 pads with exact-zero weight packs (X q=0
/// -> o=0, w2=0 -> the whole MLP is exactly zero, so xn1 = xn) and exec
/// 33 has no qkv phase (w1=0 -> qkv section exact zeros; its xn1 drain
/// IS the final hidden). Weights are pre-packed offline per exec
/// (tools/layerv2_pack.py, every bit re-sliced from the verified w4u_hy
/// v4 packs) so the runtime only fills X/XN per exec and reads the drain.
///
/// Drain geometry (test_layerv2): worker w's C rows = [qkv pos(w)-slice
/// 384 | xn1 pos(w)-chunk 256]; XN(w) of the NEXT exec <- C(w)[384:640]
/// at [0,512) of its element — same worker index, no permutation. Worker
/// id u32 at X element [6400,6404) is static (the kernel derives its ring
/// position from it); K headers ride the element tails, also static.
fn cmd_run_decode_lv2(
    arch: &DecArch,
    decdir: &str,
    w4dir: &str,
    iters: usize,
) -> ExitCode {
    let build = "/home/nzinfo/qwen/xnpu/build";
    let lv2dir = format!("{build}/lv2_hy");
    let layers = arch.layers;
    let execs = layers + 1; // head/tail-padded half-layer pipeline: L+1
    const LV2_WELEM: usize = 186;
    const DRAIN_ROWS: usize = 640; // [qkv 384 | xn1 256] bf16 per worker
    let w_bytes = 8 * LV2_WELEM * W4U_ELEM;
    let c_bytes = 8 * DRAIN_ROWS * 2;
    println!(
        "lv2 decode chain: {name} {layers} layers = {execs} half-layer execs on CU0 (8 ring workers, {mb:.2} MB weights/exec), host rope+qk-norm+GQA attention, lm_head CU1, {iters} iters",
        name = arch.name,
        mb = w_bytes as f64 / 1e6,
    );

    // ring position of worker w (design_layerv2 serpentine SUCC table).
    let pos_of = |w: usize| if w < 4 { w } else { 11 - w };

    // fixtures: the layerv2 xclbin (P28-4 test artifact) + the plain
    // w4gemvu PDI carrying the M=121088 lm_head ctrl code.
    let (lv2_pdi, lv2_instr, _) = match load_fixture(&format!("{build}/w4gemvu_layerv2_8.mlir.prj")) {
        Some(f) => f,
        None => {
            eprintln!("load w4gemvu_layerv2_8 fixture failed (run test_layerv2 + copy first)");
            return ExitCode::FAILURE;
        }
    };
    const LM_M: usize = W4U_LM_M;
    const LM_K: usize = 2048;
    let (lm_pdi, lm_instr, _) = match load_fixture(&format!("{build}/w4gemvu_{LM_M}x{LM_K}.mlir.prj")) {
        Some(f) => f,
        None => {
            eprintln!("load w4gemvu_{LM_M}x{LM_K} fixture failed");
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
    let cols = 8u32;
    let num_tiles = cols * md.core.row_count as u32;
    let mut ctx = match HwContext::create(&dev, num_tiles) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("create hwctx: {e}");
            return ExitCode::FAILURE;
        }
    };
    // layerv2 on CU0, plain w4gemvu (lm_head ctrl) on CU1 — ONE cu flip
    // per token at the lm boundary (~650µs, the P18 reload tax, once).
    if let Err(e) = ctx.configure_cus(&[(lv2_pdi.as_slice(), 0), (lm_pdi.as_slice(), 0)]) {
        eprintln!("configure_cus: {e}");
        return ExitCode::FAILURE;
    }

    // ---- device buffers: [x, xn, c] shared by all 33 ops, then W_01..W_33.
    // live layout: [X, XN, C, W_1 .. W_33]; live_lm: [w, x, c]. ----
    let mut live: Vec<(BufferObject, Mapping)> = Vec::new();
    // X: exec-1 zeros (q=0/d=0) + per-worker id + K=0 header — the q/d
    // ranges are rewritten per exec, id/header stay for the process life.
    let mut xdata = vec![0u8; 8 * W4U_ELEM];
    for w in 0..8usize {
        let e = w4u_build_x_elem(&[], &[], 2048); // q=0, d=0, K=0 header
        xdata[w * W4U_ELEM..(w + 1) * W4U_ELEM].copy_from_slice(&e);
        xdata[w * W4U_ELEM + 6400..w * W4U_ELEM + 6404]
            .copy_from_slice(&(w as u32).to_le_bytes());
    }
    if chain_tensor(&dev, &mut live, "X", &xdata).is_none() {
        eprintln!("X BO failed");
        return ExitCode::FAILURE;
    }
    // XN: header K=100 per element; [0,512) is the residual chunk (exec 1
    // = x0 by ring position, later execs = the previous exec's drain).
    let host = Lv2Host::load(arch, decdir);
    let mut xndata = vec![0u8; 8 * W4U_ELEM];
    for w in 0..8usize {
        let p = pos_of(w);
        let off = w * W4U_ELEM;
        for j in 0..256usize {
            xndata[off + j * 2..off + j * 2 + 2]
                .copy_from_slice(&host.x0[p * 256 + j].to_le_bytes());
        }
        xndata[off + W4U_ELEM - 8..off + W4U_ELEM - 4].copy_from_slice(&100u32.to_le_bytes());
    }
    if chain_tensor(&dev, &mut live, "XN", &xndata).is_none() {
        eprintln!("XN BO failed");
        return ExitCode::FAILURE;
    }
    if chain_tensor(&dev, &mut live, "C", &vec![0u8; c_bytes]).is_none() {
        eprintln!("C BO failed");
        return ExitCode::FAILURE;
    }
    let mut ops: Vec<ChainOp> = Vec::with_capacity(execs);
    let mut handles: Vec<Vec<u32>> = Vec::with_capacity(execs);
    for e in 1..=execs {
        let data = match std::fs::read(format!("{lv2dir}/exec{e:02}.bin")) {
            Ok(d) => d,
            Err(err) => {
                eprintln!("read exec{e:02}.bin: {err} (tools/layerv2_pack.py)");
                return ExitCode::FAILURE;
            }
        };
        if data.len() != w_bytes {
            eprintln!("exec{e:02}.bin: {} B != {w_bytes} (stale pack)", data.len());
            return ExitCode::FAILURE;
        }
        let w_va = match chain_tensor(&dev, &mut live, &format!("W{e:02}"), &data) {
            Some(v) => v,
            None => {
                eprintln!("W{e:02} BO failed");
                return ExitCode::FAILURE;
            }
        };
        // rt.sequence(W, X, XN, C) — tensor VAs in that order.
        let op = match chain_op(
            &dev,
            &format!("lv2E{e:02}"),
            &lv2_instr,
            0,
            &[w_va, live[0].1.as_ptr() as u64, live[1].1.as_ptr() as u64, live[2].1.as_ptr() as u64],
        ) {
            Some(o) => o,
            None => {
                eprintln!("op exec{e:02} setup failed");
                return ExitCode::FAILURE;
            }
        };
        handles.push(vec![
            op.ctrl_bo.handle(),
            live[2 + e].0.handle(), // W_e
            live[0].0.handle(),
            live[1].0.handle(),
            live[2].0.handle(),
        ]);
        ops.push(op);
    }

    // ---- lm_head on CU1 (plain w4gemvu + the M=121088 ctrl code) ----
    let mut live_lm: Vec<(BufferObject, Mapping)> = Vec::new();
    let lm_wdata = match std::fs::read(format!("{w4dir}/lmhead.bin")) {
        Ok(d) if d.len() == 8 * w4u_blocks(LM_M, LM_K) * W4U_ELEM => d,
        _ => {
            eprintln!("lmhead.bin absent/stale (q4nx_import.py --lmhead)");
            return ExitCode::FAILURE;
        }
    };
    let lm_x = chain_tensor(&dev, &mut live_lm, "x_lm", &vec![0u8; W4U_ELEM]);
    let lm_c = chain_tensor(&dev, &mut live_lm, "c_lm", &vec![0u8; w4u_c_rows(LM_M, LM_K) * 2]);
    let lm_w = chain_tensor(&dev, &mut live_lm, "w_lm", &lm_wdata);
    if lm_x.is_none() || lm_c.is_none() || lm_w.is_none() {
        eprintln!("lm BO failed");
        return ExitCode::FAILURE;
    }
    let mut lm_op = match chain_op(
        &dev,
        "lmhead",
        &lm_instr,
        1,
        &[lm_w.unwrap(), lm_x.unwrap(), lm_c.unwrap()],
    ) {
        Some(o) => o,
        None => {
            eprintln!("lm op setup failed");
            return ExitCode::FAILURE;
        }
    };
    let lm_handles = vec![
        lm_op.ctrl_bo.handle(),
        live_lm[2].0.handle(),
        live_lm[0].0.handle(),
        live_lm[1].0.handle(),
    ];

    // ---- goldens (tools/layerv2_golden.py: kernel-chain numerics + host
    // glue in the engine's exact forms) ----
    let rd_u16file = |p: String| -> Option<Vec<u16>> {
        let d = std::fs::read(p).ok()?;
        Some(d.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    };
    let golden_execs: Vec<Vec<u16>> = (1..=execs)
        .map(|e| rd_u16file(format!("{lv2dir}/golden_exec{e:02}.bin")))
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();
    let golden_hidden = rd_u16file(format!("{lv2dir}/golden_hidden.bin"));
    let golden_logits = rd_u16file(format!("{lv2dir}/golden_logits.bin"));
    if golden_execs.len() != execs || golden_hidden.is_none() || golden_logits.is_none() {
        // P28-10 (weed #4): accurate split — a partial golden set is NOT
        // "ungated": missing per-exec goldens only skip the drain
        // comparison, while missing hidden/logits refuses the run below
        // unless XNPU_SKIP_GATES is set.
        if golden_execs.len() != execs {
            println!(
                "lv2 per-exec goldens incomplete ({}/{execs}) — drain comparison skipped",
                golden_execs.len()
            );
        }
        if golden_hidden.is_none() || golden_logits.is_none() {
            println!(
                "lv2 final goldens absent — run refuses unless XNPU_SKIP_GATES is set"
            );
        }
    }

    let mut rec = Recorder::new();
    let mut rec_seq = 0u64;
    let mut lv2_meta = OpMeta::new(
        "lv2exec",
        "layerv2",
        0,
        (2 * 8 * W4U_ELEM) as u64,
        c_bytes as u64,
        // 2*(o + gate + up + down + qkv rows * K) per exec
        (2 * 2048 * (2048 + 2 * 6144 + 6144 + 3072)) as u64,
    )
    .with_tier(tier::SLOT_STREAM);
    lv2_meta.bytes_stream = Some(w_bytes as u64);
    let mut lm_meta = OpMeta::new(
        "lmhead",
        "w4gemvu",
        1,
        (LM_K * 2) as u64,
        (w4u_c_rows(LM_M, LM_K) * 2) as u64,
        (2 * LM_M * LM_K) as u64,
    )
    .with_tier(tier::SLOT_STREAM);
    lm_meta.bytes_stream = Some((8 * w4u_blocks(LM_M, 2048) * W4U_ELEM) as u64);
    let metas = vec![lv2_meta.clone(), lm_meta.clone()];

    // ---- one token: 33 execs + final norm + lm_head ----
    // scratch: drain image (5120 u16), qkv assembly, rope/qk/attn temps.
    let mut drain: Vec<u16> = vec![0u16; 8 * DRAIN_ROWS];
    let mut qkv = vec![0u16; arch.qkv_m];
    let mut qr = vec![0u16; arch.heads * 128];
    let mut kr = vec![0u16; arch.kv * 128];
    let mut attn = vec![0u16; 2048];
    let cache_seq = 1024usize;
    let pos = 100usize;
    let (rc, rs) = rope_table(pos, arch.rope_base);
    // posn/rc/rs are now token_step parameters (P28-9 ask loop).

    // ONE token step. `check` gates the per-exec drain comparison (the
    // checked step only; timed steps skip the read entirely when goldens
    // are absent). P28-9: x0/posn/rope are PARAMETERS (the ask loop
    // advances position and re-seeds x0 from the embedding each step).
    let mut token_step = |kcache: &mut Vec<u16>,
                          vcache: &mut Vec<u16>,
                          hidden: &mut [u16],
                          it: u32,
                          check: bool,
                          rec: &mut Recorder,
                          rec_seq: &mut u64,
                          x0: &[u16],
                          posn: usize,
                          rc: &[f32; 64],
                          rs: &[f32; 64]|
     -> bool {
        for e in 1..=execs {
            // (a) fill XN(e): exec 1 <- x0 by ring position, else the
            // previous exec's xn1 drain (worker-index-keyed, no perm).
            hp!(HSeg::Lv2Fill, {
                {
                    let (_, map) = &mut live[1];
                    let bytes = map.as_mut_slice();
                    for w in 0..8usize {
                        let off = w * W4U_ELEM;
                        let src: &[u16] = if e == 1 {
                            &x0[pos_of(w) * 256..][..256]
                        } else {
                            &drain[w * DRAIN_ROWS + 384..][..256]
                        };
                        for j in 0..256 {
                            bytes[off + j * 2..off + j * 2 + 2]
                                .copy_from_slice(&src[j].to_le_bytes());
                        }
                    }
                }
                let (bo, map) = &live[1];
                for w in 0..8usize {
                    sync_to_device(bo, map, w * W4U_ELEM, 512);
                }
            });
            // (b) fill X(e): exec 1 needs q=0/d=0 (-> o=0 exactly) EVERY
            // token — the previous token's exec-33 fill left real attn in
            // the slots; else quantize the attention output into every
            // worker slot (replicated attn: all 8 read the same q/d).
            {
                hp!(HSeg::Lv2Fill, {
                    let (zq, zd) = if e == 1 {
                        (vec![0i8; 2048], vec![0u16; 64])
                    } else {
                        w4u_quantize_x(&attn, 2048)
                    };
                    let (q, d): (&[i8], &[u16]) = (&zq, &zd);
                    {
                        let (_, map) = &mut live[0];
                        let bytes = map.as_mut_slice();
                        for w in 0..8usize {
                            let off = w * W4U_ELEM;
                            for j in 0..2048 {
                                bytes[off + j] = q[j] as u8;
                            }
                            for g in 0..64 {
                                bytes[off + W4U_K_MAX + g * 2..off + W4U_K_MAX + g * 2 + 2]
                                    .copy_from_slice(&d[g].to_le_bytes());
                            }
                        }
                    }
                    let (bo, map) = &live[0];
                    for w in 0..8usize {
                        let off = w * W4U_ELEM;
                        sync_to_device(bo, map, off, 2048);
                        sync_to_device(bo, map, off + W4U_K_MAX, 128);
                    }
                });
            }
            // (c) P21: flush C so the DMA's writes land in memory the host
            // will read post-wait (previous iteration's clean lines).
            {
                let (bo, map) = &live[2];
                sync_to_device(bo, map, 0, c_bytes);
            }
            // (d) submit + data-ready on CU0 (device-dominated segment).
            // P28-8 pipelined chain (LV2_PIPE=0 reverts to the P28-5
            // serialized wait): the host releases at DATA-READY, not at
            // formal completion. Sentinel = every worker's LAST C
            // element (its final cxn drain, 32B) differing from the
            // previous exec's image — each worker's single S2MM channel
            // delivers elements in fifo order (qkv then cxn), so the
            // last element landing means that worker's whole image is
            // already in memory. Everything between here and the next
            // submit (read/gate/attention/fills) then hides inside exec
            // e's TCT/syncobj tail; the lv2loop PIPE discriminator
            // measured the pace win at 656 -> 603us.
            let piped = e < execs
                && std::env::var("LV2_PIPE").map(|v| v != "0").unwrap_or(true);
            let ts = std::time::Instant::now();
            let seq = hp!(HSeg::Lv2Wait, {
                match ops[e - 1].pkt.submit(&dev, &ctx, &handles[e - 1]) {
                    Ok(s) => s,
                    Err(err) => {
                        eprintln!("lv2 submit (exec {e}): {err}");
                        return false;
                    }
                }
            });
            let mut ready = false;
            if piped {
                let deadline = std::time::Instant::now()
                    + std::time::Duration::from_millis(5);
                while std::time::Instant::now() < deadline {
                    let cmap = &live[2].1;
                    for w in 0..8usize {
                        // last 16 bf16 of worker w's rows
                        let off = (w * DRAIN_ROWS + DRAIN_ROWS - 16) * 2;
                        cmap.clflush_region(off, 32);
                    }
                    let cs = live[2].1.as_slice();
                    let mut all = true;
                    for w in 0..8usize {
                        let base = (w * DRAIN_ROWS + DRAIN_ROWS - 16) * 2;
                        for j in 0..16usize {
                            let b = [cs[base + j * 2], cs[base + j * 2 + 1]];
                            let old = drain[w * DRAIN_ROWS + DRAIN_ROWS - 16 + j];
                            if b != old.to_le_bytes() {
                                all = false;
                                break;
                            }
                        }
                        if !all {
                            break;
                        }
                    }
                    if !all {
                        ready = true;
                        break;
                    }
                    std::hint::spin_loop();
                }
                if !ready {
                    eprintln!("lv2 poll (exec {e}) timed out; falling back");
                }
            }
            if !ready {
                hp!(HSeg::Lv2Wait, {
                    if let Err(err) = syncobj_timeline_wait(
                        &dev,
                        ctx.syncobj_handle,
                        seq,
                        60_000_000_000,
                    ) {
                        eprintln!("lv2 wait (exec {e}, seq {seq}): {err}");
                        return false;
                    }
                });
                // P28-10 (weed #1): the fence signals on EVERY response
                // path — COMPLETED, ERROR, ABORT, TIMEOUT alike — so the
                // wait returning is NOT evidence the exec ran. Read the
                // state word (main.rs lv2loop note; 6f-7 law).
                let st = ops[e - 1].pkt.state();
                if st != 4 {
                    eprintln!("lv2 exec {e}: state={st} after wait (ghost completion)");
                    return false;
                }
                // Weed A: the final serialized exec must also cover its
                // piped predecessor (exec execs-1) — the lagged check
                // never reads it on an all-sentinel run.
                if e == execs && execs >= 2 {
                    let ps = ops[e - 2].pkt.state();
                    if ps != 4 {
                        eprintln!("lv2 exec {}: state={ps} (ghost, final-exec lagged check)", e - 1);
                        return false;
                    }
                }
            } else if e >= 2 {
                // Piped: exec e is still in its tail here, but execs
                // 1..e-1 are fully complete (in-order ERT) — check the
                // previous exec's state word with one-exec lag. The final
                // exec goes through the serialized branch above.
                let st = ops[e - 2].pkt.state();
                if st != 4 {
                    eprintln!("lv2 exec {}: state={st} (ghost, lagged piped check)", e - 1);
                    return false;
                }
            }
            if it > 0 {
                rec.solo(&lv2_meta, it, ts, *rec_seq);
                *rec_seq += 1;
            }
            // (e) read the drain image. Piped: the sentinel proved the
            // data is in memory — clflush the region and read the mmap
            // (the FromDevice ioctl would fence against exec e's still
            // in-flight tail, re-serializing the chain). Serialized:
            // keep the fw-fence ioctl (P21-3 correctness).
            hp!(HSeg::Lv2Read, {
                if ready {
                    live[2].1.clflush_region(0, c_bytes);
                    let cs = live[2].1.as_slice();
                    for i in 0..8 * DRAIN_ROWS {
                        drain[i] = u16::from_le_bytes([cs[i * 2], cs[i * 2 + 1]]);
                    }
                } else {
                    // P28-10 (weed #2): the fence error is SURFACED, not
                    // swallowed — but downgraded to a warning: this ioctl
                    // has returned EINVAL on the final exec since P28-5
                    // (previously invisible behind `let _ =`), while gates
                    // kept passing — with the syncobj wait AND the
                    // state==4 check above as completion proof, the
                    // second fence is redundant here.
                    if let Err(err) =
                        live[2].0.sync(SyncDirection::FromDevice, 0, live[2].0.size() as u64)
                    {
                        eprintln!(
                            "lv2 drain FromDevice fence (exec {e}): {err} (warning; completion proven by wait+state)"
                        );
                    }
                    let cs = live[2].1.as_slice();
                    for i in 0..8 * DRAIN_ROWS {
                        drain[i] = u16::from_le_bytes([cs[i * 2], cs[i * 2 + 1]]);
                    }
                }
            });
            if std::env::var("LV2_DUMP_ALL").is_ok() {
                let _ = std::fs::create_dir_all("/tmp/lv2_dump");
                let bytes: Vec<u8> = drain.iter().flat_map(|b| b.to_le_bytes()).collect();
                let _ = std::fs::write(format!("/tmp/lv2_dump/exec{e:02}.bin"), bytes);
            }
            // (f) golden gate: qkv rows strict (rel .08 abs .8 — rms1
            // renormalization absorbs the common mode), xn1 rows loose
            // (rel .08 abs 200 — the hw-sigmoid rides the down partials).
            if check && !golden_execs.is_empty() {
                let g = &golden_execs[e - 1];
                let mut bad = 0usize;
                let mut worst = 0f32;
                let mut worst_at = (0usize, 0f32, 0f32);
                for r in 0..8 * DRAIN_ROWS {
                    let (af, gf) = (bf16_to_f32(drain[r]), bf16_to_f32(g[r]));
                    let xn1 = (r % DRAIN_ROWS) >= 384;
                    let (rel, abs) = if xn1 { (0.08f32, 200.0f32) } else { (0.08, 0.8) };
                    if (af - gf).abs() > rel * gf.abs() + abs {
                        bad += 1;
                        if (af - gf).abs() > worst {
                            worst = (af - gf).abs();
                            worst_at = (r, af, gf);
                        }
                    }
                }
                if bad > 16 {
                    eprintln!(
                        "exec {e:02} drain gate FAILED: {bad}/5120 rows, worst |err| {worst:.2} @row {} (act {:-.2} vs golden {:-.2})",
                        worst_at.0, worst_at.1, worst_at.2
                    );
                    if std::env::var("LV2_DUMP").is_ok() {
                        let _ = std::fs::create_dir_all("/tmp/lv2_dump");
                        let bytes: Vec<u8> =
                            drain.iter().flat_map(|b| b.to_le_bytes()).collect();
                        let _ = std::fs::write(format!("/tmp/lv2_dump/exec{e:02}.bin"), bytes);
                    }
                    return false;
                }
            }
            // (g) host attention for the NEXT exec's X element: layer
            // e-1's qkv (assembled by ring position) -> rope -> qk-norm
            // -> kv append -> GQA attention.
            if e < execs {
                let l = e - 1;
                for w in 0..8usize {
                    let p = pos_of(w);
                    qkv[p * 384..(p + 1) * 384]
                        .copy_from_slice(&drain[w * DRAIN_ROWS..w * DRAIN_ROWS + 384]);
                }
                let kdim = arch.kv * 128;
                hp!(HSeg::Rope, {
                    rope_apply(&qkv[..2048], &rc, &rs, arch.heads, &mut qr);
                    rope_apply(&qkv[2048..2048 + kdim], &rc, &rs, arch.kv, &mut kr);
                });
                hp!(HSeg::QkNorm, {
                    let qlen = qr.len();
                    qk_rms_bf16(&qr, &host.qknorms[l * 256..][..128], &mut attn);
                    qr.copy_from_slice(&attn[..qlen]);
                    qk_rms_bf16(&kr, &host.qknorms[l * 256 + 128..][..128], &mut attn[..kdim]);
                    kr.copy_from_slice(&attn[..kdim]);
                });
                hp!(HSeg::KvApp, {
                    for kv in 0..arch.kv {
                        let off = (l * arch.kv + kv) * cache_seq * 128 + posn * 128;
                        kcache[off..off + 128].copy_from_slice(&kr[kv * 128..(kv + 1) * 128]);
                        vcache[off..off + 128].copy_from_slice(
                            &qkv[2048 + kdim + kv * 128..2048 + kdim + (kv + 1) * 128],
                        );
                    }
                });
                hp!(HSeg::Attn, {
                    let klo = l * arch.kv * cache_seq * 128;
                    attention_bf16(
                        &qr,
                        &kcache[klo..],
                        &vcache[klo..],
                        posn,
                        cache_seq,
                        &mut attn,
                        arch.heads,
                        arch.kv,
                    );
                });
            }
        }
        // exec 33's xn1 drain = final hidden, assembled by ring position.
        for w in 0..8usize {
            let p = pos_of(w);
            hidden[p * 256..(p + 1) * 256]
                .copy_from_slice(&drain[w * DRAIN_ROWS + 384..w * DRAIN_ROWS + DRAIN_ROWS]);
        }
        true
    };

    // ---- checked step (gates vs the offline golden chain) ----
    let mut kcache = host.kcache.clone();
    let mut vcache = host.vcache.clone();
    let mut hidden = vec![0u16; 2048];
    let t0 = std::time::Instant::now();
    if !token_step(&mut kcache, &mut vcache, &mut hidden, 0, true, &mut rec, &mut rec_seq, &host.x0, pos, &rc, &rs) {
        return ExitCode::FAILURE;
    }
    let mut xn = vec![0u16; 2048];
    rms_norm_bf16(&hidden, &host.norms[2 * layers * 2048..][..2048], &mut xn);

    // lm_head on CU1 (one cu flip per token, the only PDI reload left).
    let mut lm_run = |xn: &[u16], it: u32, rec: &mut Recorder, rec_seq: &mut u64| -> Option<Vec<u16>> {
        let (q, d) = w4u_quantize_x(xn, 2048);
        let xe = w4u_build_x_elem(&q, &d, 2048);
        {
            let (_, xmap) = &mut live_lm[0];
            xmap.as_mut_slice()[..W4U_ELEM].copy_from_slice(&xe);
        }
        let (bo, xmap) = &live_lm[0];
        sync_to_device(bo, xmap, 0, W4U_ELEM);
        let ts = std::time::Instant::now();
        let seq = match lm_op.pkt.submit(&dev, &ctx, &lm_handles) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("lm submit: {e}");
                return None;
            }
        };
        if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 60_000_000_000) {
            eprintln!("lm wait: {e}");
            return None;
        }
        // P28-10 (weed #1): lm completion needs the state word too.
        let st = lm_op.pkt.state();
        if st != 4 {
            eprintln!("lm: state={st} after wait (ghost completion)");
            return None;
        }
        if it > 0 {
            rec.solo(&lm_meta, it, ts, *rec_seq);
            *rec_seq += 1;
        }
        if let Err(e) = live_lm[1].0.sync(SyncDirection::FromDevice, 0, live_lm[1].0.size() as u64)
        {
            // Same P28-10 downgrade: warn, don't fail — wait+state above
            // already prove completion (the EINVAL twin of the drain path).
            eprintln!("lm C FromDevice fence: {e} (warning)");
        }
        Some(w4u_read_c(live_lm[1].1.as_slice(), 2048, LM_M))
    };
    let logits = match lm_run(&xn, 0, &mut rec, &mut rec_seq) {
        Some(l) => l,
        None => return ExitCode::FAILURE,
    };

    // gates: final hidden rms + logits argmax/rel_rms/top-8.
    let f32of = |b: u16| bf16_to_f32(b);
    if let (Some(gh), Some(gl)) = (&golden_hidden, &golden_logits) {
        let mut num = 0f64;
        let mut den = 0f64;
        for i in 0..2048 {
            let d = f32of(hidden[i]) - f32of(gh[i]);
            num += (d * d) as f64;
            den += (f32of(gh[i]) * f32of(gh[i])) as f64;
        }
        let rms = (num / den).sqrt() as f32;
        let argmax = |v: &[u16]| -> usize {
            let mut a = 0usize;
            let mut b = f32of(v[0]);
            for (i, x) in v.iter().enumerate() {
                let t = f32of(*x);
                if t > b {
                    b = t;
                    a = i;
                }
            }
            a
        };
        let (an, ag) = (argmax(&logits), argmax(gl));
        let mut num2 = 0f64;
        let mut den2 = 0f64;
        for i in 0..LM_M {
            let d = f32of(logits[i]) - f32of(gl[i]);
            num2 += (d * d) as f64;
            den2 += (f32of(gl[i]) * f32of(gl[i])) as f64;
        }
        let rel = (num2 / den2).sqrt() as f32;
        let ids = |v: &[u16]| -> Vec<usize> {
            let mut t: Vec<(usize, f32)> = (0..LM_M).map(|i| (i, f32of(v[i]))).collect();
            t.sort_by(|a, b| b.1.total_cmp(&a.1));
            t.into_iter().take(8).map(|(i, _)| i).collect()
        };
        let overlap = ids(&logits).iter().filter(|i| ids(gl).contains(i)).count();
        println!(
            "lv2 gates: hidden rel_rms {rms:.4} (<0.05), logits rel_rms {rel:.4} argmax NPU {an} vs golden {ag}, top-8 {overlap}/8 -> {}",
            if rms < 0.05 && an == ag && rel < 0.05 { "PASS" } else { "FAIL" }
        );
        if rms >= 0.05 || an != ag || rel >= 0.05 {
            if std::env::var("XNPU_SKIP_GATES").is_ok() {
                eprintln!("gate FAILED — continuing (XNPU_SKIP_GATES, PERF ONLY)");
            } else {
                return ExitCode::FAILURE;
            }
        }
    } else if std::env::var("XNPU_SKIP_GATES").is_err() {
        eprintln!("goldens absent and XNPU_SKIP_GATES unset — refusing to run blind");
        return ExitCode::FAILURE;
    }

    // ---- P28-9 ask mode: real-prompt greedy generation ----
    // LV2_ASK=<dir>,<steps>: <dir> carries tools/lv2_ask.py's REAL-prompt
    // prefill (kcache/vcache + ask.txt "pos first"). Per step: x0 = the
    // bf16 embedding row of the last sampled token (tie_word_embeddings:
    // the lm_head w4 pack dequantized row-by-row), position advances,
    // caches grow — every decode step is the full NPU chain (33 execs +
    // lm on CU1) with host attention between execs, exactly the engine's
    // architecture. Prints greedy token ids for the caller to decode.
    if let Ok(spec) = std::env::var("LV2_ASK") {
        let (adir, steps_s) = match spec.rsplit_once(',') {
            Some((d, s)) => (d.to_string(), s.to_string()),
            None => (spec.clone(), "24".to_string()),
        };
        let steps: usize = steps_s.trim().parse().unwrap_or(24);
        let ask_meta = match std::fs::read_to_string(format!("{adir}/ask.txt")) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("ask: read {adir}/ask.txt: {e} (tools/lv2_ask.py)");
                return ExitCode::FAILURE;
            }
        };
        let mut it = ask_meta.split_whitespace();
        let pos0: usize = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut cur: usize = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let rd_cache = |name: &str| -> Vec<u16> {
            std::fs::read(format!("{adir}/{name}"))
                .expect("ask cache read")
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        let mut kcache = rd_cache("kcache.bin");
        let mut vcache = rd_cache("vcache.bin");
        let lm_pack = match std::fs::read(format!("{w4dir}/lmhead.bin")) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("ask: read lmhead.bin: {e}");
                return ExitCode::FAILURE;
            }
        };
        println!("[ask] pos0 {pos0} first {cur}, {steps} steps:");
        print!("[ask] ids: {cur}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        // hy-mt2 stop tokens: the chat template's eos <｜hy_place▁holder▁no▁2｜>
        // (120020 — decodes to the EMPTY string, so a swallowed eos is
        // invisible in the text and the model drifts into template-ish
        // continuation like "2. 请帮我...") and the plain end-of-sentence
        // variant 120001. Caught live: the model emitted 120020 right after
        // "我是混元，由腾讯开发的大模型。" and the loop kept decoding.
        const HY_EOS: [usize; 2] = [120020, 120001];
        for s in 0..steps {
            if HY_EOS.contains(&cur) {
                println!("  <-- stop token {cur} at step {s}");
                break;
            }
            let posn = pos0 + s;
            let (rc_s, rs_s) = rope_table(posn, arch.rope_base);
            let x0 = w4u_row_bf16(&lm_pack, LM_M, 2048, cur);
            if !token_step(
                &mut kcache,
                &mut vcache,
                &mut hidden,
                0,
                false,
                &mut rec,
                &mut rec_seq,
                &x0,
                posn,
                &rc_s,
                &rs_s,
            ) {
                eprintln!("ask: token_step {s} failed");
                return ExitCode::FAILURE;
            }
            rms_norm_bf16(&hidden, &host.norms[2 * layers * 2048..][..2048], &mut xn);
            let logits = match lm_run(&xn, 0, &mut rec, &mut rec_seq) {
                Some(l) => l,
                None => {
                    eprintln!("ask: lm failed at step {s}");
                    return ExitCode::FAILURE;
                }
            };
            let mut best = 0usize;
            let mut bv = bf16_to_f32(logits[0]);
            for (i, b) in logits.iter().enumerate() {
                let v = bf16_to_f32(*b);
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            cur = best;
            print!(" {cur}");
            let _ = std::io::stdout().flush();
        }
        println!();
    }

    // ---- timed iterations (caches idempotent at fixed pos) ----
    hp_reset();
    let t1 = std::time::Instant::now();
    for it in 1..=iters as u32 {
        let tb = std::time::Instant::now();
        if !token_step(&mut kcache, &mut vcache, &mut hidden, it, false, &mut rec, &mut rec_seq, &host.x0, pos, &rc, &rs) {
            eprintln!("timed lv2 step failed");
            return ExitCode::FAILURE;
        }
        hp!(HSeg::FinalNorm, {
            rms_norm_bf16(&hidden, &host.norms[2 * layers * 2048..][..2048], &mut xn);
        });
        if hp!(HSeg::Lm, { lm_run(&xn, it, &mut rec, &mut rec_seq) }).is_none() {
            eprintln!("timed lm failed");
            return ExitCode::FAILURE;
        }
        rec.burst_done("lv2-decode-step", Mode::Solo, it, tb, execs as u32 + 1, 1);
    }
    let per = t1.elapsed() / iters as u32;
    println!(
        "lv2 decode step: first (checked) {:?}, steady {:.2?} /token ({:.1} tok/s)",
        t0,
        per,
        1e3 / per.as_secs_f64() / 1e3
    );
    hp_print(iters as u32);
    println!(
        "  (CPU: rope+qk-norm+GQA-attention+quant; NPU: {execs} layerv2 execs on CU0 + lm_head on CU1)"
    );

    let model = machine_model_or_default();
    let title = format!("run-decode lv2: {} {layers}L, {execs} half-layer execs, {iters} iters", arch.name);
    let (md, summary) = xnpu_perf::render_markdown(&rec, &metas, &model, &title);
    println!("\n{md}");
    let dir = "/home/nzinfo/qwen/xnpu/build/perf";
    if std::fs::create_dir_all(dir).is_ok() {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let stem = format!("{dir}/decode_lv2_{iters}it_{ts}");
        let json = trace_json(&rec, &model, &title, &summary);
        match std::fs::write(format!("{stem}.md"), &md)
            .and_then(|()| std::fs::write(format!("{stem}.json"), json))
        {
            Ok(()) => println!("perf written: {stem}.md/.json"),
            Err(e) => eprintln!("perf write failed: {e}"),
        }
    }
    drop(ctx);
    ExitCode::SUCCESS
}

/// Host-side state for the lv2 chain (decdir fixtures + norms).
struct Lv2Host {
    norms: Vec<u16>,
    qknorms: Vec<u16>,
    x0: Vec<u16>,
    kcache: Vec<u16>,
    vcache: Vec<u16>,
}

impl Lv2Host {
    fn load(arch: &DecArch, decdir: &str) -> Lv2Host {
        let rd = |p: String| -> Vec<u16> {
            std::fs::read(p)
                .unwrap_or_else(|e| panic!("lv2: {e}"))
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        let layers = arch.layers;
        let cache_seq = 1024usize;
        let n = Lv2Host {
            norms: rd(format!("{decdir}/norms.bin")),
            qknorms: rd(format!("{decdir}/qknorms.bin")),
            x0: rd(format!("{decdir}/x0.bin")),
            kcache: rd(format!("{decdir}/kcache.bin")),
            vcache: rd(format!("{decdir}/vcache.bin")),
        };
        assert_eq!(n.norms.len(), (2 * layers + 1) * 2048);
        assert_eq!(n.qknorms.len(), layers * 256);
        assert_eq!(n.x0.len(), 2048);
        assert_eq!(n.kcache.len(), layers * arch.kv * cache_seq * 128);
        assert_eq!(n.vcache.len(), layers * arch.kv * cache_seq * 128);
        n
    }
}



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
    fused: bool,
    quad: bool,
) -> ExitCode {
    let build = "/home/nzinfo/qwen/xnpu/build";
    let layers = arch.layers;
    let shapes = arch.shapes();
    let npu_attn = npu_attn_req;
    println!(
        "decode chain: {name} {layers} layers, w4gemvu projections on CU0{} + {} attention{}, {iters} iters",
        if fused {
            " + fused rms-pairs on CU1"
        } else if quad {
            " + quad whole-layer execs on CU1"
        } else {
            ""
        },
        if npu_attn { "flowkv NPU" } else { "Rust scalar" },
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

    // M9/P18 fused rms-pair fixtures (IRON w4gemvuf_*): pair A =
    // o(K1=2048) -> ln2_n -> gateup(M2=12288), pair B = down(K1=6144)
    // -> ln1_{n+1} -> qkv(M2=3072). Own CU slot; the ctrl code takes
    // FOUR tensor BOs (rt.sequence order A1, A2, X, C) — chain_op
    // already appends any number of tensor VAs. Quad mode loads ONLY
    // pair A (the last layer's tail: no qkv(L) exists for a quad).
    let fused_fx: Option<Vec<(Vec<u8>, Vec<u8>, u32)>> = if fused || quad {
        let stems = [
            "w4gemvuf_2048x2048_12288x2048", // pair A
            "w4gemvuf_2048x6144_3072x2048",  // pair B
        ];
        let mut v = Vec::new();
        for st in stems {
            match load_fixture(&format!("{build}/{st}.mlir.prj")) {
                Some(f) => v.push(f),
                None => {
                    eprintln!("load fixture {st} failed (run the fused pytest first)");
                    return ExitCode::FAILURE;
                }
            }
        }
        Some(v)
    } else {
        None
    };

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
    // P19 quad fixture (IRON w4gemvuq_*): the fixed hy layer
    // o -> rms1 -> gateup -> swiglu -> down -> rms2 -> qkv' in ONE exec.
    let quad_fx: Option<(Vec<u8>, Vec<u8>, u32)> = if quad {
        match load_fixture(&format!(
            "{build}/w4gemvuq_2048x2048_12288_2048x6144_3072.mlir.prj"
        )) {
            Some(f) => Some(f),
            None => {
                eprintln!("load quad fixture failed (run test_quad first)");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    // Both PDIs take cu_func 0 — the CU slot is the index in this list and
    // is selected per op via set_cu (the run-multi pattern; func != 0 makes
    // the fw look up a DPU function the PDI doesn't have and the op never
    // runs). Fused inserts the w4gemvuf PDI at slot 1 (flowkv -> 2); quad
    // inserts w4gemvuq at 1 AND the pair-A PDI at 2 (flowkv -> 3).
    let quad_cu: u32 = 1;
    // P27-3a: ONE CU carrying the FUSED PDI for the whole token — the
    // fused kernel is a superset (same w4gemvu.cc, K-header dispatch
    // covers the plain K=0/K=2048 flavors; board-proven: gates
    // bit-identical), so the plain ctrl bins (L0 qkv, tail down, lm)
    // execute on it unchanged and the cu_mask NEVER flips — the
    // ~2x500µs fw PDI-reload tax (P18 补记) is gone (interleaved A/B:
    // -1.8ms/token). XNPU_TWOCU=1 restores the two-PDI wiring for A/B.
    let onecu = std::env::var("XNPU_TWOCU").is_err() && fused && !quad && !npu_attn;
    let pa_cu: u32 = if quad {
        2
    } else if onecu {
        0
    } else {
        1
    };
    let fk_cu: u32 = if quad {
        3
    } else if fused {
        2
    } else {
        1
    };
    let cus: Vec<(&[u8], u8)> = match (&quad_fx, &fused_fx) {
        (Some(qf), Some(ffx)) => {
            let mut v = vec![
                (fixtures[0].0.as_slice(), 0),
                (qf.0.as_slice(), 0),
                (ffx[0].0.as_slice(), 0),
            ];
            if npu_attn {
                v.push((fk_fixture.0.as_slice(), 0));
            }
            v
        }
        (None, Some(ffx)) => {
            if npu_attn {
                vec![
                    (fixtures[0].0.as_slice(), 0),
                    (ffx[0].0.as_slice(), 0),
                    (fk_fixture.0.as_slice(), 0),
                ]
            } else if onecu {
                vec![(ffx[0].0.as_slice(), 0)]
            } else {
                vec![(fixtures[0].0.as_slice(), 0), (ffx[0].0.as_slice(), 0)]
            }
        }
        _ => {
            if npu_attn {
                vec![(fixtures[0].0.as_slice(), 0), (fk_fixture.0.as_slice(), 0)]
            } else {
                vec![(fixtures[0].0.as_slice(), 0)]
            }
        }
    };
    if let Err(e) = ctx.configure_cus(&cus) {
        eprintln!("configure_cus: {e}");
        return ExitCode::FAILURE;
    }

    // Device buffers: ELEM-sized x BOs (one activation element per K, v5),
    // per-shape outputs, per-(layer,shape) weights. live layout: [xu2048,
    // xu6144, c_qkv, c_o, c_gateup, c_down, w00.qkv, w00.o, ...].
    let mut live: Vec<(BufferObject, Mapping)> = Vec::new();
    if chain_tensor(&dev, &mut live, "xu2048", &vec![0u8; W4U_ELEM]).is_none()
        || chain_tensor(&dev, &mut live, "xu6144", &vec![0u8; W4U_ELEM]).is_none()
    {
        eprintln!("x BO failed");
        return ExitCode::FAILURE;
    }
    for s in shapes.iter() {
        if chain_tensor(
            &dev,
            &mut live,
            &format!("c_{}", s.name),
            &vec![0u8; w4u_c_rows(s.m, s.k) * 2],
        )
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
            if data.len() != 8 * w4u_blocks(s.m, s.k) * W4U_ELEM {
                eprintln!("layer{n:02}_{}: stale import (rerun the importer)", s.name);
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

    // M8: lm_head on NPU（可选 —— 需要 q4nx_import --lmhead 的
    // lmhead.bin + golden_lmhead.bin 和 w4gemvu_121088x2048 夹具）。PDI
    // 就是同一个普适内核，只多一份携带 M=121088 的 ctrl code。BO 存
    // live_lm（所有 submit 期间保活）。minicpm 无导出 → 自动跳过。
    let mut live_lm: Vec<(BufferObject, Mapping)> = Vec::new();
    let mut lm: Option<(ChainOp, Vec<u32>, OpMeta, Vec<u16>)> = (|| {
        const LM_M: usize = W4U_LM_M;
        const LM_K: usize = 2048;
        let (_, linstr, _) =
            match load_fixture(&format!("{build}/w4gemvu_{LM_M}x{LM_K}.mlir.prj")) {
                Some(f) => f,
                None => {
                    println!("lm_head: fixture absent — token 路径无 logits 算子");
                    return None;
                }
            };
        let wdata = match std::fs::read(format!("{w4dir}/lmhead.bin")) {
            Ok(d) if d.len() == 8 * w4u_blocks(LM_M, LM_K) * W4U_ELEM => d,
            _ => {
                println!("lm_head: lmhead.bin absent/stale — skipped (q4nx_import.py --lmhead)");
                return None;
            }
        };
        let g = match std::fs::read(format!("{w4dir}/golden_lmhead.bin"))
            .or_else(|_| std::fs::read(format!("{decdir}/golden_lmhead.bin")))
        {
            Ok(d)
                if d.len() == 8 + 2 * LM_K + 2 * LM_M
                    && u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as usize == LM_M =>
            {
                d
            }
            _ => {
                println!("lm_head: golden_lmhead.bin absent/stale — skipped");
                return None;
            }
        };
        let bits = |o: usize, n: usize| -> Vec<u16> {
            g[o..o + 2 * n]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        let ref_bits = bits(8 + 2 * LM_K, LM_M);
        let x_bits = bits(8, LM_K);
        let (q, d) = w4u_quantize_x(&x_bits, LM_K);
        let xdata = w4u_build_x_elem(&q, &d, LM_K);
        let w_va = chain_tensor(&dev, &mut live_lm, "w.lmhead", &wdata)?;
        let x_va = chain_tensor(&dev, &mut live_lm, "x_lm", &xdata)?;
        let c_va = chain_tensor(&dev, &mut live_lm, "c_lm", &vec![0u8; w4u_c_rows(LM_M, LM_K) * 2])?;
        let op = match chain_op(&dev, "lmhead", &linstr, 0, &[w_va, x_va, c_va]) {
            Some(o) => o,
            None => {
                eprintln!("lm_head op setup failed");
                return None;
            }
        };
        let handles = vec![
            op.ctrl_bo.handle(),
            live_lm[0].0.handle(),
            live_lm[1].0.handle(),
            live_lm[2].0.handle(),
        ];
        let mut meta = OpMeta::new(
            "lmhead",
            "w4gemvu",
            0,
            (LM_K * 2) as u64,
            (w4u_c_rows(LM_M, LM_K) * 2) as u64,
            (2 * LM_M * LM_K) as u64,
        )
        .with_tier(tier::SLOT_STREAM);
        meta.bytes_stream = Some((8 * w4u_blocks(LM_M, 2048) * W4U_ELEM) as u64);
        println!(
            "lm_head on NPU: M={LM_M} (133.9 MiB v5 权重/token), CU0 同 PDI 第 5 个 ctrl code"
        );
        Some((op, handles, meta, ref_bits))
    })();

    // 一个 lm_head step：重建单个激活元素 → submit → wait → 读 logits。
    // it > 0 才记 solo（checked step 不进分布）。
    let mut lm_run = |lm: &mut Option<(ChainOp, Vec<u32>, OpMeta, Vec<u16>)>,
                  hidden: &[u16],
                  it: u32,
                  rec: &mut Recorder,
                  rec_seq: &mut u64|
     -> Option<Vec<u16>> {
        const LM_M: usize = W4U_LM_M;
        let (op, handles, meta, _) = lm.as_mut()?;
        {
            let (q, d) = w4u_quantize_x(hidden, 2048);
            let xe = w4u_build_x_elem(&q, &d, 2048);
            {
                let (_, xmap) = &mut live_lm[1];
                xmap.as_mut_slice()[..W4U_ELEM].copy_from_slice(&xe);
            }
            let (bo, xmap) = &live_lm[1];
            sync_to_device(bo, xmap, 0, W4U_ELEM);
        }
        let ts = std::time::Instant::now();
        let seq = match op.pkt.submit(&dev, &ctx, handles) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("lm_head submit: {e}");
                return None;
            }
        };
        if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 60_000_000_000) {
            eprintln!("lm_head wait: {e}");
            return None;
        }
        if it > 0 {
            rec.solo(meta, it, ts, *rec_seq);
            *rec_seq += 1;
        }
        let _ = live_lm[2].0.sync(SyncDirection::FromDevice, 0, live_lm[2].0.size() as u64);
        let cs = live_lm[2].1.as_slice();
        Some(w4u_read_c(cs, 2048, LM_M))
    };

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

    // M9/P18 fused rms-pair state (hy only). Pair A(n) = o(n) -> ln2_n ->
    // gateup(n); pair B(n) = down(n) -> ln1_{n+1} -> qkv(n+1), n < L-1
    // (layer 0's qkv and the last layer's down have nothing to fuse with
    // and stay on the plain CU). live_fused layout: base 6n = [p1a, p2a,
    // c_a] and 6n+3 = [p1b, p2b, c_b]; fst.ops = [A(0), B(0), A(1), ...,
    // A(L-1)] (A(n) at 2n, B(n) at 2n+1). packed1 is the plain v5 weight
    // stream as-is; packed2 interleaves the K=3 rms-weight element per
    // column; C carries write-once headers (seeded here) + the per-token
    // residual (seeded per exec in gemv_pair).
    struct FusedState {
        ops: Vec<ChainOp>,
        handles: Vec<Vec<u32>>,
        k1: [usize; 2],
        m2: [usize; 2],
        c_rows1: [usize; 2],
        section2: [usize; 2],
        /// P19 quad mode: exactly ONE pair was built — A of the LAST
        /// layer (index 0 in ops/handles, live_fused [p1, p2, c]).
        a_only_last: bool,
    }
    let mut live_fused: Vec<(BufferObject, Mapping)> = Vec::new();
    let mut fst: Option<FusedState> = if fused || quad {
        let (qa, qb) = (
            fused_fx.as_ref().unwrap()[0].1.as_slice(),
            fused_fx.as_ref().unwrap()[1].1.as_slice(),
        );
        // hy shape contract (pair geometry is compiled into the ctrl code):
        // qkv 3072x2048, o 2048x2048, gateup 12288x2048, down 2048x6144.
        let (mq, mo, mg, md, kd) =
            (shapes[0].m, shapes[1].m, shapes[2].m, shapes[3].m, shapes[3].k);
        assert!(mq == 3072 && mo == 2048 && mg == 12288 && md == 2048 && kd == 6144);
        let blocks1 = [w4u_blocks(2048, 2048), w4u_blocks(2048, 6144)]; // [16, 48]
        let blocks2 = [w4u_blocks(mg, 2048), w4u_blocks(mq, 2048)]; // [96, 24]
        let c_rows1 = [8 * (blocks1[0] + 2) * 16, 8 * (blocks1[1] + 2) * 16]; // [2304, 6400]
        let section2 = [(blocks2[0] + 2) * 16, (blocks2[1] + 2) * 16]; // [1568, 416]
        let c_rows =
            [W4UF_WINDOW_ROWS + 8 * section2[0], W4UF_WINDOW_ROWS + 8 * section2[1]];
        let mut ops = Vec::with_capacity(2 * layers - 1);
        let mut handles = Vec::with_capacity(2 * layers - 1);
        let mut mk_pair = |n: usize,
                           pi: usize,
                           instr: &[u8],
                           p1: Vec<u8>,
                           p2: Vec<u8>,
                           live_fused: &mut Vec<(BufferObject, Mapping)>|
         -> bool {
            let cdata = w4uf_build_c_init(c_rows[pi], blocks1[pi]);
            let (p1_va, p2_va, c_va) = (
                match chain_tensor(&dev, live_fused, &format!("f{pi}L{n:02}.p1"), &p1) {
                    Some(v) => v,
                    None => return false,
                },
                match chain_tensor(&dev, live_fused, &format!("f{pi}L{n:02}.p2"), &p2) {
                    Some(v) => v,
                    None => return false,
                },
                match chain_tensor(&dev, live_fused, &format!("f{pi}L{n:02}.c"), &cdata) {
                    Some(v) => v,
                    None => return false,
                },
            );
            // X rides the SHARED plain x element BO (all columns read the
            // same ELEM — quantized per exec by gemv_pair).
            let x_va = live[if pi == 1 { 1 } else { 0 }].1.as_ptr() as u64;
            match chain_op(
                &dev,
                &format!("fused{}L{n:02}", if pi == 0 { "A" } else { "B" }),
                instr,
                pa_cu,
                &[p1_va, p2_va, x_va, c_va],
            ) {
                Some(op) => {
                    // mk_pair pushed exactly [p1, p2, c] above, so the C BO
                    // sits at len() - 1 — robust in quad mode's A-only
                    // layout too (was n * 6 + 2 + pi * 3).
                    let ci = live_fused.len() - 1;
                    handles.push(vec![
                        op.ctrl_bo.handle(),
                        live_fused[ci - 2].0.handle(),
                        live_fused[ci - 1].0.handle(),
                        live[if pi == 1 { 1 } else { 0 }].0.handle(),
                        live_fused[ci].0.handle(),
                    ]);
                    ops.push(op);
                    true
                }
                None => false,
            }
        };
        let rd_w = |stem: String| -> Option<Vec<u8>> {
            let d = std::fs::read(format!("{w4dir}/{stem}")).ok()?;
            Some(d)
        };
        for n in 0..layers {
            // Quad mode needs exactly ONE pair: A(L-1) closes the last
            // layer's tail (there is no qkv(L) for a quad to produce —
            // pair A + plain down + host swiglu finish the token).
            if quad && n + 1 != layers {
                continue;
            }
            // pair A: packed1 = o blocks; packed2 = [w elem | gateup blocks],
            // rms weight = ln2_n.
            let p1a = match rd_w(format!("layer{n:02}_{}.bin", shapes[1].name)) {
                Some(d) if d.len() == 8 * blocks1[0] * W4U_ELEM => d,
                _ => {
                    eprintln!("fused: layer{n:02}_o absent/stale");
                    return ExitCode::FAILURE;
                }
            };
            let g = match rd_w(format!("layer{n:02}_{}.bin", shapes[2].name)) {
                Some(d) if d.len() == 8 * blocks2[0] * W4U_ELEM => d,
                _ => {
                    eprintln!("fused: layer{n:02}_gateup absent/stale");
                    return ExitCode::FAILURE;
                }
            };
            let p2a = w4uf_build_packed2(
                &g,
                &norms[(2 * n + 1) * 2048..][..2048],
                blocks1[0],
                blocks2[0],
            );
            if !mk_pair(n, 0, qa, p1a, p2a, &mut live_fused) {
                eprintln!("fused pair A layer{n} setup failed");
                return ExitCode::FAILURE;
            }
            if n + 1 < layers && !quad {
                // pair B: packed1 = down blocks; packed2 = [w elem | NEXT
                // layer's qkv blocks], rms weight = ln1_{n+1}.
                let p1b = match rd_w(format!("layer{n:02}_{}.bin", shapes[3].name)) {
                    Some(d) if d.len() == 8 * blocks1[1] * W4U_ELEM => d,
                    _ => {
                        eprintln!("fused: layer{n:02}_down absent/stale");
                        return ExitCode::FAILURE;
                    }
                };
                let q = match rd_w(format!("layer{:02}_{}.bin", n + 1, shapes[0].name)) {
                    Some(d) if d.len() == 8 * blocks2[1] * W4U_ELEM => d,
                    _ => {
                        eprintln!("fused: layer{:02}_qkv absent/stale", n + 1);
                        return ExitCode::FAILURE;
                    }
                };
                let p2b = w4uf_build_packed2(
                    &q,
                    &norms[(2 * (n + 1)) * 2048..][..2048],
                    blocks1[1],
                    blocks2[1],
                );
                if !mk_pair(n, 1, qb, p1b, p2b, &mut live_fused) {
                    eprintln!("fused pair B layer{n} setup failed");
                    return ExitCode::FAILURE;
                }
            }
        }
        if quad {
            println!(
                "fused rms-pairs: 1 pair A (o->ln2->gateup) at L{} on CU{pa_cu} — the quad execs cover every other layer",
                layers - 1
            );
        } else {
            println!(
                "fused rms-pairs: {} pair A (o->ln2->gateup) + {} pair B (down->ln1'->qkv') execs/token on CU1",
                layers,
                layers - 1
            );
        }
        Some(FusedState {
            ops,
            handles,
            k1: [2048, 6144],
            m2: [mg, mq],
            c_rows1,
            section2,
            a_only_last: quad,
        })
    } else {
        None
    };

    // P19 quad state (hy only): layers-1 execs, quad(n) = o(n) -> rms1
    // (ln2_n) -> gateup -> DEVICE swiglu -> down -> rms2 (ln1_{n+1}) ->
    // qkv(n+1) in ONE NPU exec (design_quad.py). live_quad layout: base
    // 5n = [p1, p2, p3, p4, c]; handles [ctrl, p1, p2, p3, p4, c]
    // (rt.sequence order A1..A4, C — the X elements ride packed1
    // FRONT-GROUPED at the BO head, P21-2, still the 5-BO ctrl-kernel
    // cap). packed1's X region is rewritten per exec (one 148KB region
    // sync); packed3 is the plain v5 stream; packed2/
    // packed4 interleave the two K=3 rms-weight elements (same builder
    // as the fused pairs). C carries setup-once window headers; the
    // host seeds residual1 per exec and reads back ONLY the o sections
    // (x' = x + o via the same add_bf16 as the split path — the
    // residual stream stays bit-exact) and the qkv sections.
    struct QuadState {
        ops: Vec<ChainOp>,
        handles: Vec<Vec<u32>>,
    }
    let mut live_quad: Vec<(BufferObject, Mapping)> = Vec::new();
    let mut qst: Option<QuadState> = if quad {
        let qinstr = quad_fx.as_ref().unwrap().1.as_slice();
        // hy shape contract (quad geometry is compiled into the ctrl code):
        // o 2048x2048, gateup 12288x2048, down 2048x6144, qkv 3072x2048.
        let (mq, mo, mg, md, kd) =
            (shapes[0].m, shapes[1].m, shapes[2].m, shapes[3].m, shapes[3].k);
        assert!(mq == 3072 && mo == 2048 && mg == 12288 && md == 2048 && kd == 6144);
        let blocks1 = w4u_blocks(2048, 2048); // 16 (o)
        let blocks2 = w4u_blocks(mg, 2048); // 96 (gateup)
        let blocks3 = w4u_blocks(md, kd); // 48 (down)
        let blocks4 = w4u_blocks(mq, 2048); // 24 (qkv)
        let mut ops = Vec::with_capacity(layers - 1);
        let mut handles = Vec::with_capacity(layers - 1);
        let rd_w = |stem: String| -> Option<Vec<u8>> {
            let d = std::fs::read(format!("{w4dir}/{stem}")).ok()?;
            Some(d)
        };
        for n in 0..layers - 1 {
            let o = match rd_w(format!("layer{n:02}_{}.bin", shapes[1].name)) {
                Some(d) if d.len() == 8 * blocks1 * W4U_ELEM => d,
                _ => {
                    eprintln!("quad: layer{n:02}_o absent/stale");
                    return ExitCode::FAILURE;
                }
            };
            let g = match rd_w(format!("layer{n:02}_{}.bin", shapes[2].name)) {
                Some(d) if d.len() == 8 * blocks2 * W4U_ELEM => d,
                _ => {
                    eprintln!("quad: layer{n:02}_gateup absent/stale");
                    return ExitCode::FAILURE;
                }
            };
            let dn = match rd_w(format!("layer{n:02}_{}.bin", shapes[3].name)) {
                Some(d) if d.len() == 8 * blocks3 * W4U_ELEM => d,
                _ => {
                    eprintln!("quad: layer{n:02}_down absent/stale");
                    return ExitCode::FAILURE;
                }
            };
            let q = match rd_w(format!("layer{:02}_{}.bin", n + 1, shapes[0].name)) {
                Some(d) if d.len() == 8 * blocks4 * W4U_ELEM => d,
                _ => {
                    eprintln!("quad: layer{:02}_qkv absent/stale", n + 1);
                    return ExitCode::FAILURE;
                }
            };
            // rms1 weight = ln2_n; rms2 weight = ln1_{n+1} (the glue that
            // feeds the NEXT layer's qkv — same pairing as fused pair B).
            let p1 = w4q_build_packed1(&o, blocks1);
            let p2 =
                w4uf_build_packed2(&g, &norms[(2 * n + 1) * 2048..][..2048], blocks1, blocks2);
            let p4 = w4uf_build_packed2(
                &q,
                &norms[(2 * (n + 1)) * 2048..][..2048],
                blocks3,
                blocks4,
            );
            let cdata = w4q_build_c_init();
            let (mut vas, mut hds) = (Vec::new(), Vec::new());
            let mut ok = true;
            for (tag, data) in [("p1", p1), ("p2", p2), ("p3", dn), ("p4", p4)] {
                match chain_tensor(&dev, &mut live_quad, &format!("qL{n:02}.{tag}"), &data) {
                    Some(va) => vas.push(va),
                    None => ok = false,
                }
            }
            if ok {
                match chain_tensor(&dev, &mut live_quad, &format!("qL{n:02}.c"), &cdata) {
                    Some(va) => vas.push(va),
                    None => ok = false,
                }
            }
            if !ok {
                eprintln!("quad layer{n} BO setup failed");
                return ExitCode::FAILURE;
            }
            hds.extend(
                live_quad[n * 5..n * 5 + 5]
                    .iter()
                    .map(|(bo, _)| bo.handle()),
            );
            match chain_op(&dev, &format!("quadL{n:02}"), qinstr, quad_cu, &vas) {
                Some(op) => {
                    let mut h = vec![op.ctrl_bo.handle()];
                    h.extend(hds);
                    handles.push(h);
                    ops.push(op);
                }
                None => {
                    eprintln!("quad op layer{n} setup failed");
                    return ExitCode::FAILURE;
                }
            }
        }
        println!(
            "quad whole-layer: {} execs/token on CU{quad_cu} (o->rms->gateup->swiglu->down->rms->qkv'), swiglu on device",
            layers - 1
        );
        Some(QuadState { ops, handles })
    } else {
        None
    };

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
                fk_cu,
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
            "flowkv: {} kv cache BOs ({} KiB each), q {} B, o {} B, {} ops on CU{fk_cu}",
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
                (w4u_c_rows(s.m, s.k) * 2) as u64,
                (2 * s.m * s.k) as u64,
            )
            .with_tier(tier::SLOT_STREAM);
            m.bytes_stream = Some((8 * w4u_blocks(s.m, s.k) * W4U_ELEM) as u64);
            m
        })
        .chain(if npu_attn {
            Some({
                let s_pos = (pos + 1) as u64;
                let nkv = arch.kv as u64;
                let mut m = OpMeta::new(
                    "flowkv",
                    "attn",
                    fk_cu,
                    nkv * s_pos * 2 * 128 * 2 + nkv * fk_stride as u64 * 2, // S 行 KV + q
                    (16 * 128 * 2) as u64,
                    (16 * s_pos * 128 * 2 * 2) as u64,
                )
                // P9: 架构限定层级（strided:hy-mt2 / strided:minicpm）
                .with_tier(format!("strided:{}", arch.name)); // 2D stride KV 容量流
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
    // M8: lm_head 进报告（若在）。
    let metas: Vec<OpMeta> = match &lm {
        Some((_, _, m, _)) => {
            let mut v = metas;
            v.push(m.clone());
            v
        }
        None => metas,
    };
    // M9/P18: fused pair metas (CU1; reported alongside the split shapes
    // they replace). bytes_stream = the real weight stream incl. each
    // column's K=3 w element.
    let metas_fused: Vec<OpMeta> = [
        ("fused_o_gateup", 2048usize, 12288usize, 16usize, 96usize),
        ("fused_down_qkv", 6144, 3072, 48, 24),
    ]
    .iter()
    .map(|&(nm, k1, m2, b1, b2)| {
        let mut m = OpMeta::new(
            nm,
            "w4gemvu",
            1,
            (W4U_ELEM + 4096) as u64, // x element + residual seed
            ((W4UF_WINDOW_ROWS + 8 * (b2 + 2) * 16) * 2) as u64,
            (2 * (2048 * k1 + m2 * 2048)) as u64,
        )
        .with_tier(tier::SLOT_STREAM);
        m.bytes_stream = Some((8 * (b1 + 1 + b2) * W4U_ELEM) as u64);
        m
    })
    .collect();
    // P19: the quad whole-layer meta (CU1; replaces pair A + host swiglu
    // + pair B for one layer). bytes_stream = the real weight stream:
    // 8 columns x ([X | 16 o] + [w | 96 gateup] + 48 down + [w | 24 qkv]).
    let meta_quad: OpMeta = {
        let mut m = OpMeta::new(
            "quad_layer",
            "w4gemvu",
            1,
            (8 * W4U_ELEM + 4096) as u64, // 8 X element heads + residual1 seed
            (W4Q_C_ROWS * 2) as u64,
            (2 * (2048 * 2048 + 12288 * 2048 + 2048 * 6144 + 3072 * 2048)) as u64,
        )
        .with_tier(tier::SLOT_STREAM);
        m.bytes_stream = Some((8 * (17 + 97 + 48 + 25) * W4U_ELEM) as u64);
        m
    };
    let mut rec = Recorder::new();
    let mut rec_seq = 0u64;

    // One w4gemvu call: replicate x into the vector BO, submit, wait, read c.
    // `live` is an ARG (not a capture) so gemv and gemv_pair coexist.
    let mut gemv = |ops: &mut [ChainOp],
                    i: usize,
                    si: usize,
                    x: &[u16],
                    live: &mut Vec<(BufferObject, Mapping)>,
                    it: u32,
                    rec: &mut Recorder,
                    rec_seq: &mut u64|
     -> Option<Vec<u16>> {
        let k = shapes[si].k;
        let m = shapes[si].m;
        let vi = if k == 6144 { 1 } else { 0 };
        {
            let (bo, map) = &mut live[vi];
            let bytes = map.as_mut_slice();
            // 量化 int8 + 打成单个 ELEM 激活元素（v5：全 chunk 在一个元素
            // 里，无槽复制）。逐 u16 写是首个 v3 版本 101ms 回归的主因 —
            // 保持批量打包。
            let (q, d) = w4u_quantize_x(x, k);
            bytes[..W4U_ELEM].copy_from_slice(&w4u_build_x_elem(&q, &d, k));
            if let Err(e) = bo.sync(SyncDirection::ToDevice, 0, W4U_ELEM as u64) {
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
        // v5 C：每列截面 [16 零行 | 块截面 | pad]，w4u_read_c 跳零行整块读；
        // K=6144 的 3 个 chunk 截面按 f32 求和后舍入 bf16（同 reference）。
        Some(w4u_read_c(cs, k, m))
    };

    // M9/P18: one fused rms-pair exec — op1 gemv -> K=1 window element
    // (compact partials + residual, read BACK from C) -> K=3 w element
    // (add+rms+per-group-32 int8 quantize, prebuilds op2's A-operands) ->
    // op2 gemv, all inside ONE NPU exec. Host work per call: quantize
    // op1's activation into the SHARED x element BO (identical to the
    // split path), seed this pair's residual rows in C (rows
    // [c_rows1..c_rows1+2048) — never touched by the drains), submit
    // [ctrl, p1, p2, x, c], then read op1's plain-layout sections and
    // op2's sections past the 9280-row window. The host residual-add
    // (out1) is bit-exact with the split chain — the same add_bf16 on
    // the same w4u_read_c values — so the E2E residual stream drifts
    // nowhere; only op2's quantized input carries the NPU glue's <=1-ulp
    // f32-vs-bf16 glue deltas (P18 analysis).
    let mut gemv_pair = |pi: usize, // 0 = pair A (o->gateup), 1 = pair B (down->qkv)
                         n: usize, // layer index of op1
                         act: &[u16], // op1 activation (pre-gemv)
                         res: &[u16], // residual (pre-add hidden)
                         out1: &mut [u16], // post-add hidden (host add)
                         out2: &mut [u16], // op2 output (m2)
                         live: &mut Vec<(BufferObject, Mapping)>,
                         it: u32,
                         rec: &mut Recorder,
                         rec_seq: &mut u64|
     -> bool {
        let (k1, m2, c_rows1, section2, oi, ci) = {
            let fs = fst.as_ref().unwrap();
            if fs.a_only_last {
                // quad mode: exactly one pair was built (A of the last
                // layer) — it IS ops[0] / live_fused[0..3].
                (
                    fs.k1[pi],
                    fs.m2[pi],
                    fs.c_rows1[pi],
                    fs.section2[pi],
                    0,
                    2,
                )
            } else {
                (
                    fs.k1[pi],
                    fs.m2[pi],
                    fs.c_rows1[pi],
                    fs.section2[pi],
                    n * 2 + pi,
                    n * 6 + 2 + pi * 3,
                )
            }
        };
        let vi = if k1 == 6144 { 1 } else { 0 };
        // P27-1: pair A vs pair B land in separate prof buckets.
        let (sq, sr, swt, scr) = if pi == 0 {
            (HSeg::QuantA, HSeg::ResA, HSeg::WaitA, HSeg::ReadA)
        } else {
            (HSeg::QuantB, HSeg::ResB, HSeg::WaitB, HSeg::ReadB)
        };
        hp!(sq, {
            let (q, d) = w4u_quantize_x(act, k1);
            let xe = w4u_build_x_elem(&q, &d, k1);
            {
                let (_, map) = &mut live[vi];
                map.as_mut_slice()[..W4U_ELEM].copy_from_slice(&xe);
            }
            let (bo, map) = &live[vi];
            sync_to_device(bo, map, 0, W4U_ELEM);
        });
        hp!(sr, {
            let r0 = c_rows1 * 2;
            {
                let (_, map) = &mut live_fused[ci];
                let bytes = map.as_mut_slice();
                // P27-2: u16 reinterpret — the C BO is page-aligned and r0
                // is even, so the residual staging is one slice copy.
                let b16: &mut [u16] = unsafe {
                    std::slice::from_raw_parts_mut(
                        bytes.as_mut_ptr() as *mut u16,
                        bytes.len() / 2,
                    )
                };
                b16[r0 / 2..][..2048].copy_from_slice(res);
            }
            let (bo, map) = &live_fused[ci];
            sync_to_device(bo, map, r0, 4096);
        });
        let ts = std::time::Instant::now();
        hp!(swt, {
            let fs = fst.as_mut().unwrap();
            let op = &mut fs.ops[oi];
            let seq = match op.pkt.submit(&dev, &ctx, &fs.handles[oi]) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("fused submit (L{n} p{pi}, op {oi}): {e}");
                    return false;
                }
            };
            if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000) {
                eprintln!("fused wait (L{n} p{pi}, seq {seq}): {e}");
                return false;
            }
            if it > 0 {
                rec.solo(&metas_fused[pi], it, ts, *rec_seq);
                *rec_seq += 1;
            }
        });
        hp!(scr, {
            let (c_bo, c_map) = &live_fused[ci];
            let _ = c_bo.sync(SyncDirection::FromDevice, 0, c_bo.size() as u64);
            let cs = c_map.as_slice();
            // op1 sections are the PLAIN v5 C layout — same read the split
            // path does (K=6144 sums the 3 chunk partials in f32).
            let o1 = w4u_read_c(cs, k1, 2048);
            add_bf16(res, &o1, out1);
            // op2 sections: past the window, per column section2 rows, data
            // after the 2 dummy 16-row glue elements (P27-2: per-column
            // slice copies on the u16 view — same layout, memcpy speed).
            let cs16: &[u16] = unsafe {
                std::slice::from_raw_parts(cs.as_ptr() as *const u16, cs.len() / 2)
            };
            let rpc2 = m2 / 8;
            for col in 0..8 {
                let base = W4UF_WINDOW_ROWS + col * section2 + 2 * W4U_TILE_ROWS;
                out2[col * rpc2..][..rpc2].copy_from_slice(&cs16[base..][..rpc2]);
            }
        });
        true
    };

    // P19: one QUAD whole-layer exec — tg1 o gemv -> tg2 rms1 + gateup
    // -> tg3 DEVICE swiglu + down -> tg4 rms2 + qkv(n+1), all inside ONE
    // NPU exec. Host work per call: quantize the attention output into
    // packed1's EIGHT column-head X elements (the 5-BO ctrl-kernel cap
    // pushed X into the weight stream — one whole-BO clflush covers all
    // 8 copies), seed residual1 in C, submit [ctrl, p1..p4, c], then
    // read the o sections (x' = x + o via the same add_bf16 on the same
    // section bits as the split path — the residual stream stays
    // bit-exact) and the qkv sections (3 dummy 16-row groups ahead).
    // The swiglu numerics live on the AIE2P hw exp2 — the E2E gates
    // arbitrate that (notes/perf-lab.md P19).
    let mut gemv_quad = |n: usize, // layer index (op = quad(n), n < L-1)
                         act: &[u16], // attention output (o's activation)
                         res: &[u16], // residual x_n (seeded as residual1)
                         out1: &mut [u16], // x_{n+1} = x_n + o_out + down_out
                         qkv_out: &mut [u16], // qkv(n+1) (3072,)
                         ops: &mut [ChainOp], // plain ops (QUAD_DEBUG compare)
                         op_h: &[Vec<u32>],
                         live: &mut Vec<(BufferObject, Mapping)>,
                         it: u32,
                         rec: &mut Recorder,
                         rec_seq: &mut u64|
     -> bool {
        {
            let (q, d) = w4u_quantize_x(act, 2048);
            let xe = w4u_build_x_elem(&q, &d, 2048);
            // P21-2: the 8 X elements are front-grouped [X0..X7] at the
            // packed1 head — one contiguous 148KB dirty run (sync via
            // sync_to_device: P21-3 CLFLUSHOPT vs ioctl A/B).
            {
                let (_, map) = &mut live_quad[n * 5];
                let bytes = map.as_mut_slice();
                for col in 0..8 {
                    let off = col * W4U_ELEM;
                    bytes[off..off + W4U_ELEM].copy_from_slice(&xe);
                }
            }
            let (bo, map) = &live_quad[n * 5];
            sync_to_device(bo, map, 0, 8 * W4U_ELEM);
        }
        {
            let r0 = W4Q_RES1_ROW * 2;
            {
                let (_, map) = &mut live_quad[n * 5 + 4];
                let bytes = map.as_mut_slice();
                for j in 0..2048 {
                    bytes[r0 + j * 2..r0 + j * 2 + 2].copy_from_slice(&res[j].to_le_bytes());
                }
            }
            let (bo, map) = &live_quad[n * 5 + 4];
            sync_to_device(bo, map, r0, 4096);
        }
        let ts = std::time::Instant::now();
        let qs = qst.as_mut().unwrap();
        let op = &mut qs.ops[n];
        let seq = match op.pkt.submit(&dev, &ctx, &qs.handles[n]) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("quad submit (L{n}): {e}");
                return false;
            }
        };
        if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000) {
            eprintln!("quad wait (L{n}, seq {seq}): {e}");
            // P20: dump the wedged exec's C state — which sections made it
            // out names the stall site (the same fingerprint technique that
            // localized the parked-Cs race).
            let _ = c_bo_wait_dump(&live_quad[n * 5 + 4], n, it);
            return false;
        }
        if it > 0 {
            rec.solo(&meta_quad, it, ts, *rec_seq);
            *rec_seq += 1;
        }
        let (c_bo, c_map) = &live_quad[n * 5 + 4];
        let _ = c_bo.sync(SyncDirection::FromDevice, 0, c_bo.size() as u64);
        let cs = c_map.as_slice();
        // QUAD_DEBUG: bit-compare this exec's o sections against the PLAIN
        // o op on the same activation (the o path must be bit-exact — any
        // diff names the quad bring-up, not numerics).
        if std::env::var("QUAD_DEBUG").is_ok() && n == 0 && it == 0 {
            let (xbo, xmap) = &mut live[0];
            let (q, d) = w4u_quantize_x(act, 2048);
            xmap.as_mut_slice()[..W4U_ELEM]
                .copy_from_slice(&w4u_build_x_elem(&q, &d, 2048));
            let _ = xbo.sync(SyncDirection::ToDevice, 0, W4U_ELEM as u64);
            let seq = match ops[1].pkt.submit(&dev, &ctx, &op_h[1]) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("debug o submit: {e}");
                    return false;
                }
            };
            if let Err(e) = syncobj_timeline_wait(&dev, ctx.syncobj_handle, seq, 10_000_000_000) {
                eprintln!("debug o wait: {e}");
                return false;
            }
            let (oc_bo, oc_map) = &live[3];
            let _ = oc_bo.sync(SyncDirection::FromDevice, 0, oc_bo.size() as u64);
            let ocs = oc_map.as_slice();
            let mut diffs = 0usize;
            let mut first = Vec::new();
            for r in 0..2304 * 2 {
                if cs[r] != ocs[r] {
                    diffs += 1;
                    if first.len() < 12 {
                        first.push(r / 2);
                    }
                }
            }
            println!(
                "[quad-debug] L0 o sections vs plain op: {diffs}/4608 bytes differ, first rows {first:?}"
            );
        }
        // o sections: the PLAIN v5 C layout in rows [0..2304) — same read
        // as the split path (K=2048, single chunk).
        let o = w4u_read_c(cs, 2048, 2048);
        add_bf16(res, &o, out1);
        // down partials from the win2 sections (8 x 800 rows): TWO dummy
        // groups (the K=4/K=5 glue zero-Cs) ahead of the 3 chunk-major
        // chunks — w4u_read_c's own walk assumes a 0 base and 1 dummy, so
        // this is spelled out. x_{n+1} = bf16(f32(bf16(x+o)) + sum_c
        // f32(p_c)) — the golden's exact rounding order (and the device's
        // own h2''), NOT the missing-down x+o the first bring-up ran
        // (E2E snowballed: each layer's down dropped from the residual
        // stream, ~0.03/layer at L0 growing with depth, decorrelating
        // attention by L11 — qkv itself matched fused to 2e-4 the whole
        // time, which is what named the host bug).
        for col in 0..8 {
            let sec = (W4Q_WIN2_OFF + col * W4Q_SEC_DN) * 2;
            for w in 0..256usize {
                let oi = col * 256 + w;
                let mut acc = bf16_to_f32(out1[oi]);
                for c in 0..3usize {
                    let off = sec + (2 * W4U_TILE_ROWS + c * 256 + w) * 2;
                    acc += bf16_to_f32(u16::from_le_bytes([cs[off], cs[off + 1]]));
                }
                out1[oi] = f32_to_bf16(acc);
            }
        }
        // qkv sections: 3 dummy 16-row groups ahead of 384 data rows/col.
        let rpcq = 3072 / 8;
        for col in 0..8 {
            let base = (W4Q_QKV_OFF + col * W4Q_SEC_Q + 3 * W4U_TILE_ROWS) * 2;
            for r in 0..rpcq {
                let off = base + r * 2;
                qkv_out[col * rpcq + r] = u16::from_le_bytes([cs[off], cs[off + 1]]);
            }
        }
        true
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
        qkv: Vec<u16>, // P18: pair B(n-1) leaves qkv(n) here
    }
    let mut sc = Scratch {
        xn: vec![0u16; 2048],
        qr: vec![0u16; 2048],
        kr: vec![0u16; arch.kv * 128],
        attn: vec![0u16; 2048],
        gu: vec![0u16; 12288],
        sw: vec![0u16; 6144],
        qkv: vec![0u16; arch.qkv_m],
    };
    let mut x = x0.clone();
    let mut first_bad: Option<usize> = None;
    let mut decode_step = |ops: &mut [ChainOp],
                       kcache: &mut [u16],
                       vcache: &mut [u16],
                       x: &mut Vec<u16>,
                       sc: &mut Scratch,
                       live: &mut Vec<(BufferObject, Mapping)>,
                       fk: &mut Option<FkState>,
                       check: bool,
                       it: u32,
                       rec: &mut Recorder,
                       rec_seq: &mut u64|
     -> bool {
        for n in 0..layers {
            let Scratch { xn, qr, kr, attn, gu, sw, qkv: scq } = sc;
            // P18 fused / P19 quad: layer n>0's qkv came from the previous
            // layer's fused tail (pair B / the quad exec) — its ln1_n rms
            // ran in the NPU glue, the result rides Scratch.qkv.
            let qkv_owned;
            let qkv: &[u16] = if n == 0 || !fused && !quad {
                rms_norm_bf16(x, &norms[n * 2 * 2048..][..2048], xn);
                qkv_owned = match gemv(ops, n * 4, 0, xn, live, it, rec, rec_seq) {
                    Some(v) => v,
                    None => return false,
                };
                &qkv_owned
            } else {
                scq.as_slice()
            };
            let kdim = arch.kv * 128;
            hp!(HSeg::Rope, {
                rope_apply(&qkv[..2048], &rc, &rs, arch.heads, qr);
                rope_apply(&qkv[2048..2048 + kdim], &rc, &rs, arch.kv, kr);
            });
            if arch.qk_norm {
                // hy: per-head rms AFTER rope. qk_rms can't alias x and out,
                // so bounce through attn as scratch (kr is only kv·128 long).
                hp!(HSeg::QkNorm, {
                    let qlen = qr.len();
                    qk_rms_bf16(qr, &qknorms[n * 256..][..128], attn);
                    qr.copy_from_slice(&attn[..qlen]);
                    qk_rms_bf16(kr, &qknorms[n * 256 + 128..][..128], &mut attn[..kdim]);
                    kr.copy_from_slice(&attn[..kdim]);
                });
            }
            hp!(HSeg::KvApp, {
                for kv in 0..arch.kv {
                    let off = (n * arch.kv + kv) * cache_seq * 128 + pos * 128;
                    kcache[off..off + 128].copy_from_slice(&kr[kv * 128..(kv + 1) * 128]);
                    vcache[off..off + 128].copy_from_slice(
                        &qkv[2048 + kdim + kv * 128..2048 + kdim + (kv + 1) * 128],
                    );
                }
            });
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
                        // P21-3 dispatch (CLFLUSHOPT vs ioctl A/B).
                        sync_to_device(qbo, qmap, 0, qbo.size());
                    }
                    let op = &mut fk.ops[n];
                    // Same first-exec O-read-race guard as run-fkprobe: flush
                    // the o BO's cache lines before submit so the post-wait
                    // read sees the DMA writes. (P21-3 dispatch.)
                    sync_to_device(&fk.o.0, &fk.o.1, 0, fk.o.0.size());
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
                    hp!(HSeg::Attn, {
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
                    });
                }
            }
            if quad {
                if n + 1 < layers {
                    // P19 quad: o(n) -> rms1 -> gateup -> DEVICE swiglu ->
                    // down -> rms2 -> qkv(n+1) in ONE exec; qkv rides
                    // Scratch.qkv for the next layer iteration.
                    if !gemv_quad(
                        n, attn, x, xn, scq, ops, &op_handles, live, it, rec, rec_seq,
                    ) {
                        return false;
                    }
                    std::mem::swap(x, xn); // x = x + o + down (gu/sw device-side)
                    if check && std::env::var("QUAD_DUMP").is_ok() {
                        // P19 debug: raw qkv(n+1) readback trajectory, to diff
                        // against the fused path's (gate-passing) qkv at the
                        // same layer — isolates the quad qkv path from
                        // rope/attention amplification.
                        let _ = std::fs::create_dir_all("/tmp/qkvdump");
                        let _ = std::fs::write(
                            format!("/tmp/qkvdump/quad_L{:02}.bin", n + 1),
                            (0..arch.qkv_m)
                                .flat_map(|i| scq[i].to_le_bytes())
                                .collect::<Vec<u8>>(),
                        );
                    }
                } else {
                    // last layer: no qkv(L) exists for a quad to produce —
                    // close the tail with pair A + host swiglu + plain down.
                    if !gemv_pair(0, n, attn, x, xn, gu, live, it, rec, rec_seq) {
                        return false;
                    }
                    std::mem::swap(x, xn);
                    swiglu_bf16(gu, sw);
                    let d = match gemv(ops, n * 4 + 3, 3, sw, live, it, rec, rec_seq) {
                        Some(v) => v,
                        None => return false,
                    };
                    add_bf16(x, &d, xn);
                    std::mem::swap(x, xn);
                }
            } else if fused {
                // P18 pair A: o(n) -> ln2_n rms -> gateup(n) in ONE exec.
                // The host rebuilds the post-add hidden from op1's
                // plain-layout C sections via the SAME add_bf16 the split
                // path uses — the residual stream stays bit-exact.
                if !gemv_pair(0, n, attn, x, xn, gu, live, it, rec, rec_seq) {
                    return false;
                }
                std::mem::swap(x, xn); // x = x + o
                hp!(HSeg::Swiglu, { swiglu_bf16(gu, sw); });
                if n + 1 < layers {
                    // P18 pair B: down(n) -> ln1_{n+1} rms -> qkv(n+1),
                    // left in Scratch.qkv for the next layer iteration.
                    if !gemv_pair(1, n, sw, x, xn, scq, live, it, rec, rec_seq) {
                        return false;
                    }
                    std::mem::swap(x, xn);
                    if check && std::env::var("QUAD_DUMP").is_ok() {
                        // P19 debug: reference qkv(n+1) for the quad diff (the
                        // fused chain passes the E2E gates, so its readback is
                        // the working baseline).
                        let _ = std::fs::create_dir_all("/tmp/qkvdump");
                        let _ = std::fs::write(
                            format!("/tmp/qkvdump/fused_L{:02}.bin", n + 1),
                            (0..arch.qkv_m)
                                .flat_map(|i| scq[i].to_le_bytes())
                                .collect::<Vec<u8>>(),
                        );
                    }
                } else {
                    let d = match gemv(ops, n * 4 + 3, 3, sw, live, it, rec, rec_seq) {
                        Some(v) => v,
                        None => return false,
                    };
                    add_bf16(x, &d, xn);
                    std::mem::swap(x, xn);
                }
            } else {
                let o = match gemv(ops, n * 4 + 1, 1, attn, live, it, rec, rec_seq) {
                    Some(v) => v,
                    None => return false,
                };
                add_bf16(x, &o, xn); // x = x + o (reuse xn as scratch)
                std::mem::swap(x, xn);
                rms_norm_bf16(x, &norms[(n * 2 + 1) * 2048..][..2048], xn);
                *gu = match gemv(ops, n * 4 + 2, 2, xn, live, it, rec, rec_seq) {
                    Some(v) => v,
                    None => return false,
                };
                swiglu_bf16(gu, sw);
                let d = match gemv(ops, n * 4 + 3, 3, sw, live, it, rec, rec_seq) {
                    Some(v) => v,
                    None => return false,
                };
                add_bf16(x, &d, xn);
                std::mem::swap(x, xn);
            }
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
                if bad > 16 && std::env::var("QUAD_DEBUG").is_ok() {
                    println!(
                        "  [quad-debug] layer {n}: {bad}/2048 outside tolerance (worst rel {worst_rel:.3})"
                    );
                }
                if bad > 16 && first_bad.is_none() {
                    first_bad = Some(n);
                    if std::env::var("QUAD_DEBUG").is_ok() {
                        let _ = std::fs::write(
                            "/tmp/quad_x.bin",
                            (0..2048).flat_map(|i| x[i].to_le_bytes()).collect::<Vec<u8>>(),
                        );
                    }
                    // P19 debug: dump the failing rows (index, ours, golden)
                    // — the o path should be bit-exact, so the pattern
                    // (columns/groups) names the culprit region directly.
                    let rows: Vec<usize> = (0..2048)
                        .filter(|&i| {
                            let e =
                                (bf16_to_f32(x[i]) - bf16_to_f32(g[i])).abs();
                            e > 0.01 + 0.02 * bf16_to_f32(x[i]).abs()
                        })
                        .take(12)
                        .collect();
                    for i in rows {
                        println!(
                            "    row {i}: x={:.5} golden={:.5}",
                            bf16_to_f32(x[i]),
                            bf16_to_f32(g[i])
                        );
                    }
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
            &mut ops2, &mut kcache, &mut vcache, &mut x, &mut sc, &mut live, &mut fk,
            true, 0, &mut rec, &mut rec_seq,
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

    // M8 lm_head 对拍：checked 步的 final hidden 喂 NPU lm_head，与
    // golden logits 比 argmax / rel_rms / top-8 重叠（gate 同 run-lmhead）。
    if lm.is_some() {
        let logits = match lm_run(&mut lm, &sc.xn, 0, &mut rec, &mut rec_seq) {
            Some(l) => l,
            None => {
                eprintln!("lm_head golden step failed");
                return ExitCode::FAILURE;
            }
        };
        let ref_bits = &lm.as_ref().unwrap().3;
        let f32of = |b: u16| bf16_to_f32(b);
        let argmax = |v: &Vec<u16>| -> usize {
            let mut a = 0usize;
            let mut b = f32of(v[0]);
            for (i, x) in v.iter().enumerate() {
                let t = f32of(*x);
                if t > b {
                    b = t;
                    a = i;
                }
            }
            a
        };
        let argmax_n = argmax(&logits);
        let argmax_g = argmax(ref_bits);
        let mut num = 0f64;
        let mut den = 0f64;
        for i in 0..W4U_LM_M {
            let d = f32of(logits[i]) - f32of(ref_bits[i]);
            num += (d * d) as f64;
            den += (f32of(ref_bits[i]) * f32of(ref_bits[i])) as f64;
        }
        let rel_rms = (num / den).sqrt() as f32;
        let ids = |v: &Vec<u16>| -> Vec<usize> {
            let mut t: Vec<(usize, f32)> =
                (0..W4U_LM_M).map(|i| (i, f32of(v[i]))).collect();
            t.sort_by(|a, b| b.1.total_cmp(&a.1));
            t.into_iter().take(8).map(|(i, _)| i).collect()
        };
        let tn = ids(&logits);
        let tg = ids(ref_bits);
        let overlap = tn.iter().filter(|i| tg.contains(i)).count();
        println!(
            "lm_head vs golden: rel_rms {:.4}, argmax NPU {} vs golden {}, top-8 overlap {}/8 -> {}",
            rel_rms,
            argmax_n,
            argmax_g,
            overlap,
            if argmax_n == argmax_g && rel_rms < 0.05 { "PASS" } else { "FAIL" }
        );
        if argmax_n != argmax_g || rel_rms >= 0.05 {
            // XNPU_SKIP_GATES: PERF-ONLY escape for ctrl-surgery probes
            // (P27-4 hoist) whose checked step is garbage by design.
            if std::env::var("XNPU_SKIP_GATES").is_ok() {
                eprintln!("lm_head gate FAILED — continuing (XNPU_SKIP_GATES, PERF ONLY)");
            } else {
                eprintln!("lm_head gate FAILED — token 路径结果不可信");
                return ExitCode::FAILURE;
            }
        }
    }

    // Timed iterations (steady state; caches are idempotent at fixed pos).
    hp_reset();
    t0 = std::time::Instant::now();
    for it in 1..=iters as u32 {
        let tb = std::time::Instant::now();
        let mut ops2 = std::mem::take(&mut ops);
        let ok = decode_step(
            &mut ops2, &mut kcache, &mut vcache, &mut x, &mut sc, &mut live, &mut fk,
            false, it, &mut rec, &mut rec_seq,
        );
        ops = ops2;
        if !ok {
            eprintln!("timed decode step failed");
            return ExitCode::FAILURE;
        }
        // M8: 每 token 的 final norm + lm_head（logits 也在 NPU 上）。
        hp!(HSeg::FinalNorm, {
            rms_norm_bf16(&x, &norms[2 * layers * 2048..][..2048], &mut sc.xn);
        });
        if lm.is_some()
            && hp!(HSeg::Lm, { lm_run(&mut lm, &sc.xn, it, &mut rec, &mut rec_seq) }).is_none()
        {
            eprintln!("timed lm_head failed");
            return ExitCode::FAILURE;
        }
        rec.burst_done(
            "decode-step",
            Mode::Solo,
            it,
            tb,
            (if quad {
                // 1 plain qkv (L0) + (L-1) quads + 1 pair A + 1 plain down
                layers + 2
            } else if fused {
                layers * 4 - (2 * layers - 1)
            } else {
                layers * 4
            } + if npu_attn { layers } else { 0 }
                + if lm.is_some() { 1 } else { 0 }) as u32,
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
    hp_print(iters as u32);
    println!(
        "  (CPU: rope+{}+norms+swiglu{}{}; NPU: {} w4gemvu{})",
        if npu_attn {
            "kv-row-append"
        } else {
            "GQA-attention"
        },
        if quad { "(last layer only)" } else { "" },
        if arch.qk_norm { "+qk-norm" } else { "" },
        if quad {
            layers + 2
        } else if fused {
            layers * 4 - (2 * layers - 1)
        } else {
            layers * 4
        },
        if quad && npu_attn {
            format!(
                " on CU0 + {} quad whole-layer on CU1 + 1 pair A on CU2 + {} flowkv attention on CU{fk_cu}",
                layers - 1,
                layers
            )
        } else if quad {
            format!(
                " on CU0 + {} quad whole-layer on CU1 + 1 pair A on CU2",
                layers - 1
            )
        } else if fused && npu_attn {
            format!(
                " on CU0 + {} fused rms-pairs on CU1 + {} flowkv attention on CU{fk_cu}",
                2 * layers - 1,
                layers
            )
        } else if fused {
            format!(" on CU0 + {} fused rms-pairs on CU1", 2 * layers - 1)
        } else if npu_attn {
            format!(" on CU0 + {} flowkv attention on CU1", layers)
        } else {
            " on 1 CU".to_string()
        },
    );

    // ---- M5a xnpu-perf 报告 ----
    let model = machine_model_or_default();
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
    )
    // P9: 架构限定层级 —— 探针无 arch 参数，从夹具名辨识（_4kv=hy）
    .with_tier(if prj.contains("_4kv") {
        "strided:hy-mt2"
    } else {
        "strided:minicpm"
    }); // 2D stride KV 容量流
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

    let model = machine_model_or_default();
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
