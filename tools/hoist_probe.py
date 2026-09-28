#!/usr/bin/env python3
"""P27-4: hoist the pair's tg2 weight fills before the tg1 sync (PERF ONLY).

P27-3b closed the model: pair latency = 2 task-group fixed costs + stream at
the 55 GB/s wall; the tg2 weight stream cannot start until [tg1 stream tail
-> core compute -> C drain -> DDR -> TCT -> window fill] completes, because
the ctrl program emits tg2's BD writes/pushes only after tg1's npu.syncs.

This patcher reorders the compiled ctrl bin so tg2's MM2S machinery (BD
descriptor writes, DDR patches, queue pushes) is issued BEFORE tg1's TCT
waits. Channel queues are in-order, so the per-core fifo still receives
[X, A1 blocks, window, A2 blocks] -- the semantic element order is intact;
only the window's SOURCE data is stale (tg1's C hasn't drained yet), so the
run produces garbage values with identical bytes, fills, drains and kernel
work. PERF ONLY -- never assert goldens against a hoisted bin.

Slot collision law: tg2's BLOCKWRITEs reuse MM2S slots 0-3, which tg1's
still-active descriptors occupy; overwriting a live descriptor mid-transfer
is exactly what FLM never does (they rewrite only after the TCT). Hoisted
fills are renumbered +8 (slots 8-11, within the 16-slot BD space, clear of
tg1's 0-3 and the S2MM drain law slots 4/5, P16).

Keep/hoist classification inside the tg2 region:
  BLOCKWRITE -> BD space slot<=3   : hoist, addr += 0x100
  DDR_PATCH  -> bd_reg slot<=3     : hoist, bd_reg += 0x100
  WRITE      -> MM2S queue, bd<=3  : hoist, bd nibble += 8
  everything else (drain BDs 4/5, S2MM pushes, MASKWRITE issue tokens): stay.

P27-4a board result: input-side hoist alone = NULL (pair latency unchanged,
numeric degradation proves the early window read fired). The serializer is
downstream: tg2's S2MM drain BD+push still execute only after tg1's TCT
chain, so the core stalls once the depth-2 C fifo fills and the A2 stream
backs up behind it. --drains additionally hoists the drain machinery
(slots 4/5 -> 12/13, same live-descriptor law; MASKWRs ride along) and
writes <blob>.hs2. Element-order safety: the S2MM channel executes BDs in
queue order, and the core produces C1 elements strictly before C2 elements
(fifo input order [X, A1..., window, A2...] with the one-C-per-A-element
contract), so the earlier-queued drain just changes WHEN elements can flow,
not which BD consumes which element.

Usage: hoist_probe.py [--drains] <blob.bin> [more.bin ...]
       writes <blob>.hs (or .hs2 with --drains)
"""
import sys, struct

OPS_SZ_AT = {1: 12, 2: 12, 0: 20, 3: 24}  # byte offset of size word, else w1

def u32(b, i): return struct.unpack_from("<I", b, i)[0]
def p32(b, i, v): struct.pack_into("<I", b, i, v)

def walk(blob):
    """-> list of (off, op, size). Raises on unwalkable (P23-calibrated rules)."""
    magic, _, ninstr, total = struct.unpack_from("<4I", blob, 0)
    assert magic == 0x06040100, "not an aie2 ctrl blob"
    end = total if 0 < total <= len(blob) else len(blob)
    out, off = [], 16
    while off + 4 <= end:
        op = u32(blob, off) & 0xFF
        if op in (1, 2):
            sz = u32(blob, off + 12)
        elif op in (0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 4):
            sz = u32(blob, off + 4)
        elif op == 0:
            sz = u32(blob, off + 20)
        elif op == 3:
            sz = u32(blob, off + 24)
        else:
            raise SystemExit(f"unwalkable op {op} at 0x{off:x}")
        assert sz and sz % 4 == 0 and off + sz <= end, f"bad size {sz} at 0x{off:x}"
        out.append((off, op, sz))
        off += sz
    assert off == end, f"walk ended 0x{off:x} != total 0x{end:x}"
    return out

def classify(blob, off, op, sz, drains=False):
    """-> None (keep) | kind string for tg2 MM2S (+ S2MM w/ drains) instrs."""
    if op == 1:
        addr = u32(blob, off + 8)
        a = addr & 0xFFFFF
        if 0x1D000 <= a < 0x1D200:
            slot = (a - 0x1D000) >> 5
            if slot <= 3 or (drains and 4 <= slot <= 5):
                return "fill"
    elif op == 0x81:
        reg = u32(blob, off + 24)  # w6 = bd_reg
        r = reg & 0xFFFFF
        if 0x1D000 <= r < 0x1D200:
            slot = (r - 0x1D000) >> 5
            if slot <= 3 or (drains and 4 <= slot <= 5):
                return "patch"
    elif op == 0:
        reg = u32(blob, off + 8)
        r = reg & 0xFFFFF
        val = u32(blob, off + 16)
        if 0x1D200 <= r < 0x1D400:
            bd = val & 0xF
            if (r & 0x10) and bd <= 3:            # MM2S queue push
                return "push"
            if drains and not (r & 0x10) and 4 <= bd <= 5:  # S2MM drain push
                return "push"
    elif op == 3 and drains:
        reg = u32(blob, off + 8)
        r = reg & 0xFFFFF
        if 0x1D200 <= r < 0x1D400:
            return "maskwr"            # issue token, rides with its drain
    return None

def renumber(blob, off, op, sz):
    if op == 1:
        p32(blob, off + 8, u32(blob, off + 8) + 0x100)      # BD slot +8
    elif op == 0x81:
        p32(blob, off + 24, u32(blob, off + 24) + 0x100)    # patch bd_reg +8
    elif op == 0:
        v = u32(blob, off + 16)
        p32(blob, off + 16, (v & ~0xF) | ((v & 0xF) + 8))   # push bd nibble +8
    # op == 3 (MASKWRITE): no BD slot fields, byte-identical

def hoist(blob, drains=False):
    instrs = walk(blob)
    tcts = [i for i in instrs if i[1] == 0x80]
    assert len(tcts) == 16, f"expected 16 TCTs (2 groups x 8), got {len(tcts)}"
    t8 = tcts[7]                       # last tg1 sync
    t9 = tcts[8]                       # first tg2 sync
    tg2_start, tg2_end = t8[0] + t8[2], t9[0]

    hoisted, kept = bytearray(), bytearray()
    n_hoist = n_keep = 0
    for off, op, sz in instrs:
        if not (tg2_start <= off < tg2_end):
            continue
        chunk = bytearray(blob[off:off + sz])
        if classify(blob, off, op, sz, drains):
            renumber(chunk, 0, op, sz)
            hoisted += chunk
            n_hoist += 1
        else:
            kept += chunk
            n_keep += 1

    first_tct = tcts[0][0]
    out = (bytearray(blob[:first_tct]) + hoisted
           + bytearray(blob[first_tct:tg2_start]) + kept
           + bytearray(blob[tg2_end:]))
    assert len(out) == len(blob), "reassembly changed size"
    walk(bytes(out))                   # re-walk sanity
    return bytes(out), n_hoist, n_keep

if __name__ == "__main__":
    args = sys.argv[1:]
    drains = False
    if args and args[0] == "--drains":
        drains = True
        args = args[1:]
    suffix = ".hs2" if drains else ".hs"
    for path in args:
        blob = open(path, "rb").read()
        out, nh, nk = hoist(blob, drains)
        open(path + suffix, "wb").write(out)
        what = "MM2S+S2MM instrs" if drains else "MM2S instrs"
        print(f"{path}: hoisted {nh} {what} before tg1 sync, "
              f"kept {nk} in place -> {path}{suffix}")
