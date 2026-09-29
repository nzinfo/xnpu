#!/usr/bin/env python3
"""P28-6 forensics v2: N=16 relay "deadlock" is a pathological CRAWL.

v1 finding (perf-lab 6f-6): rings=1 does not hard-deadlock in a bare
harness -- execs COMPLETE with all 320 C elements drained, but the drain
timeline crawls: per-element arrival ~2-6ms, heavy tails to 790ms total,
and slow starts (first drain at 2-13ms). pytest's 0/5 was the ERT 30s
timeout catching the tail. Kernel symbols alone do not explain it
(dumb/slow-dumb pass), so the open question is whether the crawl is
UNIVERSAL at N=16 (fills/drains themselves, 2 workers/shim) or
correlates with real kernels.

Usage: lv2_forensic.py [rings] [attempts] [poll_s]
  rings=1  real lv kernels; 11 all-dumb; 15 slow-dumb; ...
Per attempt: submit, poll C (sentinel 0xAAAA) every 2ms, wait, report
one summary line + first-drain/last-drain times. Full timeline printed
only for the worst attempt.
"""

import sys
import time
import struct

import numpy as np

sys.path.insert(0, "/home/nzinfo/qwen/xnpu/IRON")

from iron.common import AIEContext
from iron.operators.w4gemvu.test_lv2probe import AIELv2Probe, geom

COLS = int(__import__("os").environ.get("FCOLS", "16"))
RINGS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ATTEMPTS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
POLL_S = float(sys.argv[3]) if len(sys.argv) > 3 else 26.0
ELEM = 18560
SENT = 0xAAAA
if COLS == 16:
    NELEM = 20  # 12 qkv + 8 cxn
else:
    NELEM = 40  # N=8: 24 qkv + 16 cxn


def u32(v):
    return np.frombuffer(struct.pack("<I", v), dtype=np.uint8)


def build_inputs(g):
    x = np.zeros(COLS * ELEM, dtype=np.uint8)
    for w in range(COLS):
        x[w * ELEM + 6400 : w * ELEM + 6404] = u32(w)
        x[w * ELEM + 6404 : w * ELEM + 6408] = u32(COLS)
    xn = np.zeros(COLS * ELEM, dtype=np.uint8)
    for w in range(COLS):
        xn[w * ELEM + ELEM - 8 : w * ELEM + ELEM - 4] = u32(100)
    ks = ([2048] * g["N_O"] + [101] + [103] * g["N_GATE"] +
          [104] * g["N_GATE"] + [105] * g["N_DOWN"] + [102] +
          [2048] * g["N_QKV"])
    wbuf = np.zeros(COLS * g["N_WELEM"] * ELEM, dtype=np.uint8)
    for c in range(COLS):
        base = c * g["N_WELEM"] * ELEM
        for i, k in enumerate(ks):
            off = base + i * ELEM
            wbuf[off + ELEM - 8 : off + ELEM - 4] = u32(k)
    return {"W": wbuf, "X": x, "XN": xn}


def drain_counts(cview):
    a = cview[: COLS * NELEM * 16].reshape(COLS, NELEM, 16)
    still = (a == SENT).all(axis=2)
    return NELEM - still.sum(axis=1)


def main():
    g = geom(COLS)
    ctx = AIEContext()
    op = AIELv2Probe(cols=COLS, rings=RINGS, context=ctx)
    ctx.compile_all()
    ctx.prepare_runtime()
    for name, arr in build_inputs(g).items():
        op.write_buffer(name, arr)
    c_bo = op.get_bo("C")
    cview = np.frombuffer(c_bo.map(), dtype=np.uint16)
    import pyxrt
    for name in ("W", "X", "XN"):
        op.get_bo(name).sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    worst = (None, -1)
    totals = []
    for attempt in range(1, ATTEMPTS + 1):
        cview[:] = SENT
        c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t0 = time.perf_counter()
        op.xrt_runlist.execute()
        events = []
        last = np.zeros(COLS, dtype=np.int64)
        first_ms = None
        while time.perf_counter() - t0 < POLL_S:
            c = drain_counts(cview)
            if c.sum() > 0 and first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1e3
            if not np.array_equal(c, last):
                now = (time.perf_counter() - t0) * 1e3
                for w in np.nonzero(c != last)[0]:
                    events.append((now, int(w), int(c[w])))
                last = c.copy()
            if int(c.sum()) == COLS * NELEM:
                break
            time.sleep(0.002)
        drained = int(last.sum())
        t_poll = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        werr = None
        try:
            op.xrt_runlist.wait()
        except Exception as e:
            werr = str(e)
        t_wait = (time.perf_counter() - t1) * 1e3
        done = drained == COLS * NELEM and t_wait < 1000 and werr is None
        last_ms = max((e[0] for e in events), default=0.0)
        print(f"[r{RINGS} att{attempt}] done={done} drained={drained}/{COLS*NELEM} "
              f"first={None if first_ms is None else round(first_ms,1)}ms "
              f"last={round(last_ms,1)}ms wait={round(t_wait)}ms err={werr}",
              flush=True)
        totals.append((attempt, drained, t_poll, t_wait, done))
        if len(events) > worst[1]:
            worst = (events, len(events))

    print(f"\n== r{RINGS} x{ATTEMPTS}: {sum(1 for t in totals if t[4])} done, "
          f"poll_ms={[round(t[2]) for t in totals]} ==")
    ev = worst[0] or []
    print(f"worst attempt timeline ({len(ev)} events):")
    for t, w, c in ev[:80]:
        print(f"  {t:9.2f}ms w{w:2d} {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
