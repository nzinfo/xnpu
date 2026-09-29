# P28-4 research artifacts (layer-v2 numerics root-cause)

`sim.py` — the offline hypothesis machine: replays the kernel-side chain
in numpy (golden chain verbatim + kernel rounding mirrors + the f32-dequant
device-faithful model) against the board dumps. This is the file that
cornered the d3-scale scramble: every other artifact in the chain (ring
shifts, qkv path, x', q2, q3, fixtures, sigmoid) was proven clean one by
one until lv_sw's scale area was the only unverified piece left.

- `dumps/` — board echoes from the TEMP lv_cxn debug variants (the fixed
  16-element C drain reused as a 512B/probe window; see perf-lab P28-4).
  dbg_act_bits = last-iteration device output bits; dbg_exp_bits = golden.
- `w4gemvu_layer_fixed.o` — the post-fix kernel object (three 16B
  vector ops in lv_fr2/lv_st2 scale copies; verify with
  llvm-aie/bin/llvm-objdump -d -j .text.lv_fr2).
- `cdo_fail.log` — the PMEM-overflow failure that started the soft-float
  purge (kept as the law's origin artifact).
- `final.dis` / `frames.txt` / `la_lines.txt` — PMEM-size archaeology
  (per-function .text + paddxm frames during the 16KB squeeze).

Large fixtures (dbg_w_fixture.npy 27MB = the exact weight bytes the device
consumed; dbg_x/xn_fixture) are regenerable from seed 42 via
generate_layerv2_reference and were NOT committed.

Disassembler note: system objdump says "architecture UNKNOWN" and
llvm-objdump-18/21 "can't find target" — the working tool is
/home/nzinfo/.venvs/npu314/lib/python3.14/site-packages/llvm-aie/bin/llvm-objdump
(the mlir_aie/peano toolchain install).

## p28-6f/（2026-09-29 追加）
- lv2_forensic.py：6f-6 取证仪器（AIELv2Probe 独立驱动，C 哨兵 0xAAAA
  + 2ms 轮询 drain 掩码）——发现 pyxrt 提交路径爬行双峰。
- lv2_dump_exec01-33.tgz：E2E LV2_DUMP_ALL 调试期的 33 exec C dump
  （10KB each，root 属主原样打包）。
