#!/usr/bin/env python3
"""P24: patch aie2 ctrl-code BD AxCACHE attribute (normal->aggressive).

FLM's decode graphs set the shim BD AxCache field to aggressive_cache
(0x0e << 24) on every big-stream BD and issue ZERO SYNC_BO ioctls for the
whole run; IRON-compiled bins use normal_cache (0x02 << 24) and our engine
pays the P21 ToDevice clflush. This tool rewrites the field in a compiled
ctrl bin so the A/B (XNPU_NO_IN_FLUSH / XNPU_NO_OUT_FLUSH in xnpu-cli's
run-w4layer golden loop) can test whether coherence follows the descriptor.

BD payload word 5 (instruction word 9) of each BLOCKWRITE to shim BD space
((addr & 0xFFFFF) in [0x1D000, 0x1D200)) is the AxCache word. Walk rule per
tools/ctrl_decode.py: op_size<<2 = total instruction bytes, at w[3] for
BLOCKWRITE. Originals are kept as <bin>.ax02 (restore with --restore).

Usage: axcache_patch.py [--restore] <blob.bin> [more.bin ...]
"""
import sys, struct

def u32(b, i): return struct.unpack_from("<I", b, i)[0]

def patch(blob, old, new):
    data = bytearray(blob)
    magic, _, ninstr, total = struct.unpack_from("<4I", data, 0)
    assert magic == 0x06040100, "not an aie2 ctrl blob"
    end = total if 0 < total <= len(data) else len(data)
    off, n, seen = 16, 0, 0
    while off + 16 <= end:
        op = u32(data, off) & 0xFF
        sz = u32(data, off + 12) if op in (1, 2) else None
        if sz is None or sz == 0 or sz % 4 or off + sz > end:
            # not a BD BLOCKWRITE; step by the generic per-op size rules
            if op in (0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 4):
                off += u32(data, off + 4)
            elif op == 0:
                off += 24
            elif op == 3:
                off += 28
            else:
                raise SystemExit(f"unwalkable op {op} at 0x{off:x}")
            continue
        addr = u32(data, off + 8)
        if 0x1D000 <= (addr & 0xFFFFF) < 0x1D200 and sz >= 44:
            w9 = off + 36  # instruction word 9 = payload word 5 (axcache)
            cur = u32(data, w9)
            if cur == old:
                struct.pack_into("<I", data, w9, new)
                n += 1
            seen += 1
        off += sz
    return bytes(data), n, seen

if __name__ == "__main__":
    args = sys.argv[1:]
    restore = False
    if args and args[0] == "--restore":
        restore = True
        args = args[1:]
    for path in args:
        if restore:
            import shutil
            shutil.copy(path + ".ax02", path)
            print(f"{path}: restored from .ax02")
            continue
        blob = open(path, "rb").read()
        out, n, seen = patch(blob, 0x02000000, 0x0e000000)
        if n == 0:
            print(f"{path}: no BDs patched ({seen} BD fills seen)")
            continue
        open(path + ".ax02", "wb").write(blob)
        open(path, "wb").write(out)
        print(f"{path}: patched {n}/{seen} BD fills 0x02->0x0e (backup .ax02)")
