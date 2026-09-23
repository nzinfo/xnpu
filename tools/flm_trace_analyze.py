#!/usr/bin/env python3
"""M5c: analyze an xnpu-ftrace log (LD_PRELOAD XRT C++ API intercept) of FLM
into the per-op / per-token structure, rendered xnpu-perf-style.

Usage: python3 flm_trace_analyze.py [trace.log]

Structure the shim reveals (hy-mt2 1.8B, layer + fused_prefill xclbins):
  setup     xclbin loads + weight BO staging (bo_new/map histogram)
  prefill   per-layer 8-op cycle, per-op run_start/wait (fused_prefill graph)
  decode    per token: 33 run_new (1 fused kernel/layer + 1) -> one runlist
            rl_exec + ONE rl_wait (~19.5ms batch) + one 2.6ms long run
            (lm_head) -> next token. ~20.8ms/token = 44.3 tok/s.

Per-run device time INSIDE a runlist is not observable at the API level
(no per-run wait) — the batch total is. Per-op ioctl-level timestamps
would need the M1 raw-EXEC_BO technique (future work).
"""

import sys
from collections import defaultdict

# hy-mt2 1.8B weight geometry (config.json + q4nx import), int4 + g32 bf16
# scale ~ 4.5 bit/param effective.
HY_LAYER_B = 2048 * (3072 + 2048 + 12288 + 6144) * 4.5 / 8  # ~= 27.2 MB
HY_LMHEAD_B = 120818 * 2048 * 4.5 / 8  # ~= 138 MB (tie embeds, still run)


def parse(path):
    evs = []
    for l in open(path):
        p = l.rstrip("\n").split(None, 3)
        if len(p) < 4:
            continue
        kv = dict(q.split("=", 1) for q in p[3].split() if "=" in q)
        evs.append((int(p[0]), p[2], kv))
    return evs


def median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else 0.0


def main():
    evs = parse(sys.argv[1] if len(sys.argv) > 1 else "/tmp/flm_trace.log")
    t0 = evs[0][0]

    # ---- setup ----------------------------------------------------------
    xclbins, bo_new, bo_sync_ct = [], 0, 0
    bo_sizes = defaultdict(int)
    for _, ev, kv in evs:
        if ev == "xclbin_new":
            xclbins.append(kv["path"].rsplit("/", 1)[-1])
        elif ev == "bo_new":
            bo_new += 1
            bo_sizes[int(kv["size"])] += 1
        elif ev == "bo_sync":
            bo_sync_ct += 1
    tot_mb = sum(s * n for s, n in bo_sizes.items()) / 2**20
    print(f"== setup ==")
    print(f"xclbins: {xclbins}")
    print(f"BOs: {bo_new} allocs, {tot_mb:.0f} MiB total; top sizes: "
          + ", ".join(f"{s/2**20:.0f}MiB x{n}" for s, n in
                      sorted(bo_sizes.items(), key=lambda x: -x[0] * x[1])[:5]))

    # ---- per-run submit->wait durations ---------------------------------
    waits = defaultdict(list)
    for t, ev, kv in evs:
        if ev == "run_wait":
            waits[kv["run"]].append((int(kv["t0"]), t))
    starts = [(t, kv["run"]) for t, ev, kv in evs if ev == "run_start"]
    durs, t_first_start = [], None
    for t, run in starts:
        if t_first_start is None:
            t_first_start = t
        d = next((w for w in waits.get(run, []) if w[0] >= t), None)
        if d:
            durs.append(((d[1] - d[0]) / 1e3, t, run))

    # ---- phases: prefill = starts before first rl_exec; decode = after ---
    execs = [t for t, ev, _ in evs if ev == "rl_exec"]
    rl_waits = [(t, int(kv["t0"])) for t, ev, kv in evs if ev == "rl_wait"]
    rl_adds = [(t, kv["list"], kv["run"]) for t, ev, kv in evs if ev == "rl_add"]

    if execs:
        t_dec = execs[0]
    else:
        t_dec = evs[-1][0]
    pre = [d for d, t, _ in durs if t < t_dec]
    # decode "long runs": start/wait pairs after t_dec (the non-runlist runs)
    dec = [d for d, t, _ in durs if t >= t_dec]

    print(f"\n== prefill (single-op start+wait path, fused_prefill graph) ==")
    pre_span = ((t_dec - t_first_start) / 1e6) if pre else 0
    print(f"ops: {len(pre)}, sum {sum(pre)/1e3:.1f} ms, span {pre_span:.1f} ms, "
          f"med {median(pre):.0f} us, p90 {sorted(pre)[int(len(pre)*0.9)] if pre else 0:.0f} us")
    if pre:
        print(f"layer cycle: ~{len(pre)/32:.1f} ops/layer, {pre_span/32:.2f} ms/layer")

    print(f"\n== decode (runlist-batched) ==")
    n_tokens = len(execs)
    if n_tokens:
        # blocking wait (call->return) vs exec->return brackets the device
        # batch: FLM preps the NEXT token's runlist between exec and wait.
        wait_block = [(t_ret - t_sub) / 1e3 for t_ret, t_sub in rl_waits]
        exec_t = sorted(execs)
        wait_ret = sorted(t for t, _ in rl_waits)
        exec_to_ret = [(w - e) / 1e3 for e, w in zip(exec_t, wait_ret) if w > e]
        # runs per token: rl_adds inside each [exec_i, exec_{i+1}) window
        adds_win = []
        for i in range(n_tokens):
            hi = exec_t[i + 1] if i + 1 < n_tokens else evs[-1][0]
            adds_win.append(sum(1 for t, _, _ in rl_adds if exec_t[i] <= t < hi))
        adds_per_tok = median(adds_win)
        span = (execs[-1] - execs[0]) / 1e6
        ms_tok = span / max(n_tokens - 1, 1)
        med_block, med_ret = median(wait_block), median(exec_to_ret)
        med_long = median(dec) if dec else 0
        print(f"tokens: {n_tokens}, span {span:.1f} ms -> {ms_tok:.2f} ms/token "
              f"({1000/ms_tok:.1f} tok/s)")
        print(f"per token: {adds_per_tok:.0f} runs in ONE rl_exec; wait-block "
              f"med {med_block/1e3:.2f} ms, exec->ret med {med_ret/1e3:.2f} ms "
              f"(brackets device batch); long-run (lm_head) med {med_long/1e3:.2f} ms")
        n_layer = max(adds_per_tok - 1, 1)
        for label, bms in (("wait-block", med_block), ("exec->ret", med_ret)):
            per_layer_ms = bms / 1e3 / n_layer
            print(f"  [{label}] {per_layer_ms:.3f} ms/layer -> "
                  f"~{HY_LAYER_B/1e6/per_layer_ms:.0f} GB/s weight stream "
                  f"({HY_LAYER_B/2**20:.1f} MiB/layer)")
        if med_long:
            print(f"  lm_head {med_long/1e3:.2f} ms -> "
                  f"~{HY_LMHEAD_B/1e6/(med_long/1e3):.0f} GB/s "
                  f"(assumes full {HY_LMHEAD_B/2**20:.0f} MiB lm_head stream)")

    print(f"\n== vs our engine (run-decode hy, cpu-attention path) ==")
    print("| engine | ms/token | structure |")
    print("|---|---|---|")
    print(f"| FLM | {ms_tok if execs else float('nan'):.1f} | "
          f"1 fused kernel/layer, 33-op runlist, 1 wait/token |")
    print("| ours | 78.4 | 4 w4gemvu/layer + cpu glue, 160 per-op syncobj waits |")


if __name__ == "__main__":
    main()
