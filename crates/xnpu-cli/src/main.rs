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
        _ => {
            eprintln!(
                "usage: xnpu-cli <info | ctx-probe [max] [cols] | run-add [prj-dir] [bf16|i8]>"
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
    let pdi_path = format!("{prj}/main.pdi");
    // The ctrl-code .bin sits next to the .mlir.prj dir (both in the IRON
    // build dir), named after the shared stem.
    let stem = prj.trim_end_matches(".mlir.prj");
    let instr_path = match stem.rsplit_once('/') {
        Some((dir, name)) => format!("{dir}/{name}.bin"),
        None => format!("{stem}.bin"),
    };
    let pdi = match std::fs::read(&pdi_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {pdi_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let instr = match std::fs::read(&instr_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {instr_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "fixture: pdi {} B, ctrl-code {} B",
        pdi.len(),
        instr.len()
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
    // The fixture's partition reserves the full array (column_width 8).
    let cols: u32 = 8;
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
    let mut mismatches = 0usize;
    let mut first_bad: Option<(usize, i64, i64)> = None;
    for i in 0..n {
        let v1 = (i % 7) as f32 - 3.0;
        let v2 = ((i / 7) % 5) as f32;
        let (got, want) = dt.unpack(&s, i, v1, v2);
        if got != want {
            mismatches += 1;
            if first_bad.is_none() {
                first_bad = Some((i, got, want));
            }
        }
    }
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
