#!/usr/bin/env python3
"""Aggregate true streamed bytes of an aie2 ctrl blob (P26 analysis).

ctrl_decode.py's choreography view lists BD fills, but buffer_length alone
undercounts when BDs carry 2D dims / iteration counts / next_bd chains and
queue pushes carry repeat counters. This tool walks the same proven per-op
size rules and reports, per column and per direction:

  BD fills    : bd_id, len, dims(d0/d1/d2), iter, nextbd
  pushes      : reg, bd, repeat, issue_token
  byte totals : naive = sum(buffer_length); iter-amplified = len * iter;
                push-amplified = len * repeat of the bd it pushes

Usage: stream_bytes.py <blob.bin> [--bd] [--push]
"""
import sys, struct
from collections import Counter, defaultdict

def u32(b, i): return struct.unpack_from("<I", b, i)[0]

def decode(blob, want_bd=False, want_push=False):
    magic, _, ninstr, total = struct.unpack_from("<4I", blob, 0)
    assert magic == 0x06040100, "not an aie2 ctrl blob"
    end = total if 0 < total <= len(blob) else len(blob)
    off = 16
    bds = {}          # (col, bd_id) -> fill dict (later fills overwrite)
    bd_order = []
    pushes = []
    while off + 16 <= end:
        op = u32(blob, off) & 0xFF
        if op in (1, 2):
            sz = u32(blob, off + 12)
            addr = u32(blob, off + 8)
            a = addr & 0xFFFFF
            if op == 1 and 0x1D000 <= a < 0x1D200 and sz >= 44:
                col = addr >> 25
                bd_id = (a - 0x1D000) >> 5
                w = [u32(blob, off + 4 * i) for i in range(sz // 4)]
                p = w[4:]                      # payload words
                d0, d1, d2, it, lk = p[3], p[4], p[5], p[6], p[7]
                nxt = (lk >> 27) & 0xF if (lk >> 26) & 1 else None
                fill = dict(col=col, bd=bd_id, len=p[0], boff=p[1],
                            d0=d0, d1=d1, d2=d2, iter=(it >> 20) & 0x3FF,
                            iter_st=it & 0xFFFFF, nextbd=nxt)
                bds[(col, bd_id)] = fill
                bd_order.append((col, bd_id))
            off += sz
        elif op in (0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 4):
            off += u32(blob, off + 4)
        elif op == 0:
            reg = u32(blob, off + 8)
            val = u32(blob, off + 16)
            r = reg & 0xFFFFF
            if 0x1D200 <= r < 0x1D400:
                pushes.append(dict(col=reg >> 25, reg=r,
                                   dir="MM2S" if r & 0x10 else "S2MM",
                                   ch=(r >> 3) & 1, bd=val & 0xF,
                                   rep=(val >> 16) & 0xFF,
                                   tok=(val >> 31) & 1))
            off += 24
        elif op == 3:
            off += 28
        else:
            raise SystemExit(f"unwalkable op {op} at 0x{off:x}")
    return bds, bd_order, pushes

if __name__ == "__main__":
    path = sys.argv[1]
    blob = open(path, "rb").read()
    bds, order, pushes = decode(blob)
    print(f"{path}: {len(bds)} BDs, {len(pushes)} queue pushes")

    print("\nBD fills (nonzero dims/iter only shown in full):")
    for (col, bd_id) in order[:0]:
        pass
    bycol = defaultdict(list)
    for (col, bd_id) in order:
        bycol[col].append(bds[(col, bd_id)])
    for col in sorted(bycol):
        fills = bycol[col]
        naive = sum(f["len"] for f in fills)
        ited = sum(f["len"] * max(f["iter"], 1) for f in fills)
        print(f" col{col}: {len(fills)} fills, naive={naive}B, "
              f"iter-amp={ited}B")
        if len(sys.argv) > 2 and "--bd" in sys.argv:
            for f in fills:
                print(f"   bd{f['bd']}: len={f['len']} boff=0x{f['boff']:x} "
                      f"d0=0x{f['d0']:08x} d1=0x{f['d1']:08x} "
                      f"d2=0x{f['d2']:08x} iter={f['iter']}@0x{f['iter_st']:x} "
                      f"nextbd={f['nextbd']}")

    print("\nqueue pushes:")
    pc = Counter((p["col"], p["dir"]) for p in pushes)
    reps = Counter(p["rep"] for p in pushes)
    toks = sum(p["tok"] for p in pushes)
    print(f" by (col,dir): {dict(sorted(pc.items()))}")
    print(f" repeat hist: {dict(sorted(reps.items()))}  issue_tokens: {toks}")
    if len(sys.argv) > 2 and "--push" in sys.argv:
        for p in pushes[:40]:
            print(f"   c{p['col']} {p['dir']} ch{p['ch']} bd={p['bd']} "
                  f"rep={p['rep']} tok={p['tok']}")

    # push-amplified byte total: each push streams len(bd) * max(rep,1)
    # (repeat counter re-triggers the same BD)
    amp = defaultdict(int)
    for p in pushes:
        f = bds.get((p["col"], p["bd"]))
        if f:
            amp[(p["col"], p["dir"])] += f["len"] * max(p["rep"], 1)
    tot_mm2s = sum(v for (c, d), v in amp.items() if d == "MM2S")
    tot_s2mm = sum(v for (c, d), v in amp.items() if d == "S2MM")
    print(f"\npush-amplified bytes: MM2S={tot_mm2s} ({tot_mm2s/1e6:.2f}MB) "
          f"S2MM={tot_s2mm} ({tot_s2mm/1e6:.2f}MB)")
    for (c, d) in sorted(amp):
        print(f"  c{c} {d}: {amp[(c,d)]}B")
