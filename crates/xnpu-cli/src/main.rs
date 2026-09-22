//! xnpu-cli: probe / diagnose tools for the amdxdna driver via xnpu-hal.

use std::process::ExitCode;

use xnpu_hal::{Device, HwContext};

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
        _ => {
            eprintln!("usage: xnpu-cli <info|ctx-probe [max] [cols]>");
            ExitCode::FAILURE
        }
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
