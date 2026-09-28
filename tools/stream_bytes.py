#!/usr/bin/env python3
"""Aggregate true streamed bytes of an aie2 ctrl blob (P26/P26b analysis).

Walks the P23-calibrated per-op size rules and reports, per column and
direction, the bytes actually DMA'd by queue pushes.

Two laws learned the hard way (P26b erratum — first version undercounted
4x and corrupted re-filled slots):

1. BD word 4 `buffer_length` is in 32-bit WORDS, not bytes. Triangulated:
   FLM layer blob: 4 compute cols x (68x18432 + 8x55296) words x 4B/word
   = 27,131,904B = 48,234,496 params x 0.5625 B/param exactly; IRON quad
   bd0 = 4640 words = 18,560B = the v5 fifo ELEM size; FLM DDR_PATCH
   stride 0x12000 = 73,728B = 4 x 18,432. (Header npu_cmd_write_dma.hpp
   documents the field but not its unit; unit pinned empirically.)
2. Slots are re-filled: the same (col, bd_id) is BLOCKWRITE'n many times
   (fill -> DDR_PATCH -> push -> TCT loop). Accounting must be
   event-sourced — capture each fill AT FILL TIME and resolve pushes
   against the slot state when the push fires.

Also per npu_cmd_write_dma.hpp: iter word stores iter_size - 1; D1's top
bits 0xc0000000 are a constant burst marker, not a size.

Usage: stream_bytes.py <blob.bin> [--bd] [--push]
"""
import sys, struct
from collections import Counter, defaultdict

def u32(b, i): return struct.unpack_from("<I", b, i)[0]

def decode(blob):
    magic, _, ninstr, total = struct.unpack_from("<4I", blob, 0)
    assert magic == 0x06040100, "not an aie2 ctrl blob"
    end = total if 0 < total <= len(blob) else len(blob)
    off = 16
    state = {}       # (col, bd_id) -> fill dict, CURRENT at each point
    fills = []       # every fill captured at fill time
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
                fill = dict(col=col, bd=bd_id, len=p[0], bytes=p[0] * 4,
                            boff=p[1], d0=d0, d1=d1, d2=d2,
                            itraw=(it >> 20) & 0x3FF, itst=it & 0xFFFFF,
                            nextbd=nxt)
                state[(col, bd_id)] = fill
                fills.append(fill)
            off += sz
        elif op in (0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 4):
            off += u32(blob, off + 4)
        elif op == 0:
            reg = u32(blob, off + 8)
            val = u32(blob, off + 16)
            r = reg & 0xFFFFF
            if 0x1D200 <= r < 0x1D400:
                col = reg >> 25
                push = dict(col=col, reg=r,
                            dir="MM2S" if r & 0x10 else "S2MM",
                            ch=(r >> 3) & 1, bd=val & 0xF,
                            rep=(val >> 16) & 0xFF,
                            tok=(val >> 31) & 1)
                # amplify against slot state AT PUSH TIME, following the
                # next_bd chain (rep counter re-triggers the whole BD)
                tot, bd, seen = 0, push["bd"], set()
                while bd is not None and bd not in seen:
                    seen.add(bd)
                    f = state.get((col, bd))
                    if f is None:
                        break
                    tot += f["bytes"] * (f["itraw"] + 1)
                    bd = f["nextbd"]
                push["bytes"] = tot * max(push["rep"], 1)
                pushes.append(push)
            off += 24
        elif op == 3:
            off += 28
        else:
            raise SystemExit(f"unwalkable op {op} at 0x{off:x}")
    return fills, pushes

if __name__ == "__main__":
    path = sys.argv[1]
    blob = open(path, "rb").read()
    fills, pushes = decode(blob)
    print(f"{path}: {len(fills)} BD fills, {len(pushes)} queue pushes")

    print("\nBD fill fingerprints ((len_words, iter_size, 2d) x count):")
    fp = Counter((f["col"], f["len"], f["itraw"] + 1, f["d0"] != 0)
                 for f in fills)
    for col, ln, it, twod in sorted(fp):
        print(f"  col{col}: len={ln}w ({ln*4}B) iter={it} 2d={twod}"
              f"  x{fp[(col, ln, it, twod)]}")
    bycol = defaultdict(int)
    for f in fills:
        bycol[f["col"]] += f["bytes"]
    print("\nper-col static BD content (fill-time, bytes):")
    for col in sorted(bycol):
        print(f"  col{col}: {bycol[col]:,}B")

    reps = Counter(p["rep"] for p in pushes)
    toks = sum(p["tok"] for p in pushes)
    pc = Counter((p["col"], p["dir"]) for p in pushes)
    print(f"\npushes by (col,dir): {dict(sorted(pc.items()))}")
    print(f"repeat hist: {dict(sorted(reps.items()))}  issue_tokens: {toks}")

    amp = defaultdict(int)
    for p in pushes:
        amp[(p["col"], p["dir"])] += p["bytes"]
    tot_mm2s = sum(v for (c, d), v in amp.items() if d == "MM2S")
    tot_s2mm = sum(v for (c, d), v in amp.items() if d == "S2MM")
    print(f"\npush-amplified bytes (event-sourced, x4B/word): "
          f"MM2S={tot_mm2s:,} ({tot_mm2s/1e6:.2f}MB) "
          f"S2MM={tot_s2mm:,} ({tot_s2mm/1e6:.2f}MB)")
    for (c, d) in sorted(amp):
        print(f"  c{c} {d}: {amp[(c,d)]:,}B")
