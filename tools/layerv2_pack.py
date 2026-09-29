#!/usr/bin/env python3
"""P28-5/P28-6 layerv2 offline weight packer (hy-mt2 1.8b -> 33 exec W tensors).

The layerv2 design (IRON design_layerv2.py) computes, per exec on N ring
workers (N = 8/16; the kernel reads N from the X element, P28-6), HALF A
LAYER SHIFTED: o_proj+MLP of layer mlp plus qkv of layer qkv_l, chained by
the residual the previous exec drained. The dependency qkv_L -> host
attention -> next exec's X element forces that shift; the head/tail execs
pad the pipeline with exact-zero weight packs:

  exec 1    : mlp=None (o/gate/up/down/w2 all zeros; X element q=0 makes
              o=0 regardless, and w2=0 zeroes q2 -> gate/up=0 -> sw=0 ->
              q3=0 -> down=0 EXACTLY, so xn1 = xn and only the qkv phase
              is live) + qkv_l=0.
  exec e    : mlp = e-2, qkv_l = e-1   (e = 2..32)
  exec 33   : mlp = 31, qkv_l=None (zero qkv pack + zero w1: rms1 sees
              w1=0 -> q4=0 -> C's qkv section is exact zeros, ignored).
  exec 33's xn1 drain = the FINAL hidden (after model.norm on host it
              feeds lm_head).

Every bit is RE-SLICED from the engine's verified w4u_hy v4 packs -- no
requantization anywhere.

SUB-COLUMN SLICING LAW (P28-6, empirically pinned against the packer):
the v4 packs are 8-column (a column spans M/8 rows), so a ring position's
blocks are the contiguous t-run of ONE pack column -- only N=8 coincides
with whole columns, and a flat p*count run silently scrambles gate/up/down
for N=16. The concat gate_up pack (12288 rows) carries gate in columns
0-3 and up in columns 4-7; position p's gate rows start at concat row
p*jpw (up: 6144 + p*jpw). Down (K=6144) tiles are stored 3 chunk-major
blocks per 16 rows (block b -> col b//48, chunk (b%48)//16, tile
(b%48)%16), so a column spans 256 rows regardless and chunks multiply the
block stride, never the row span.

Per exec bin = the test_layerv2 `weights` tensor: worker w's N_WELEM-element
run in fill order [o xN_O | w2(K=101) | gate xN_GATE(K=103) | up
xN_GATE(K=104) | down 3*N_O(K=105) | w1(K=102) | qkv xN_QKV], sliced by
RING POSITION pos(w) (N=8 serpentine / N=16 Hamiltonian -- see pos());
COLUMN w carries position pos(w) blocks. Packing column p with position p
mirrors the odd columns of the N=8 serpentine (pos is an involution there)
and scrambles the ring2 all-gather (board-caught P28-5, N=8 case).

Run: npu314 python tools/layerv2_pack.py                 # N=8 -> build/lv2_hy
     npu314 python tools/layerv2_pack.py --n 16          # -> build/lv2_hy_w16
     npu314 python tools/layerv2_pack.py --execs 1 33
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ELEM = 18560
HIDDEN, INTER, QKV_M = 2048, 6144, 3072
LAYERS = 32

SRC = Path("/home/nzinfo/qwen/xnpu/build/w4u_hy")
OUT = Path("/home/nzinfo/qwen/xnpu/build/lv2_hy")


def geom(n):
    """Per-N geometry mirror of design_layerv2.my_layerv2 / test_layerv2.geom."""
    rows = HIDDEN // n
    N_O = rows // 16
    N_GATE = (INTER // n) // 16
    return {
        "n": n, "rows": rows, "jpw": INTER // n,
        "N_O": N_O, "N_GATE": N_GATE, "N_UP": N_GATE,
        "N_DOWN": 3 * N_O, "N_QKV": (QKV_M // n) // 16, "N_CXN": N_O,
        "N_WELEM": N_O + 2 * N_GATE + 3 * N_O + (QKV_M // n) // 16 + 2,
        "drain_rows": QKV_M // n + rows,
    }


def pos(w, n=8):
    """Worker id -> ring position (matches the kernel's lv_xelem and the
    design's ring_tables). N=8: the two-column serpentine (an involution).
    N=16: the all-adjacent Hamiltonian cycle order 0,1,2,3,7,6,5,9,10,11,
    15,14,13,12,8,4 (CORE<->CORE OBJECTFIFO LAW -- non-adjacent edges
    split into core mem-DMA channels, 2 per npu2 core; the serpentine
    wrap needed 3 on worker 12). The cycle's inverse is piecewise: col 0
    = r, col 3 = 13-r, col 1/2 bottom worker = 15/14, else col 1 = 7-r,
    col 2 = 6+r. NOT an involution at N=16."""
    if n != 16:
        c = w >> 2
        return w if c % 2 == 0 else 8 * c + 3 - w
    c, r = w >> 2, w & 3
    if c == 0:
        return r
    if c == 3:
        return 13 - r
    if r == 0:
        return 15 if c == 1 else 14
    return 7 - r if c == 1 else 6 + r


HAM16 = [0, 1, 2, 3, 7, 6, 5, 9, 10, 11, 15, 14, 13, 12, 8, 4]


def u32(x):
    return np.frombuffer(struct.pack("<I", x), dtype=np.uint8)


def sub_blocks(mm, M, p, rows_per_pos, chunks=1, chunk=0, r0_extra=0):
    """SUB-COLUMN SLICING LAW: position p's contiguous 16-row-block run of
    pack column col (the 8-column pack; rows_per_col = M/8 regardless of
    K-chunking). r0_extra shifts into e.g. the up half of the concat pack."""
    rows_per_col = M // 8
    r0 = r0_extra + p * rows_per_pos
    col, t0 = r0 // rows_per_col, (r0 % rows_per_col) // 16
    T = rows_per_col // 16
    base = (col * chunks + chunk) * T + t0
    cnt = rows_per_pos // 16
    return mm[base * ELEM : (base + cnt) * ELEM]


class LayerBins:
    """Per-layer packed blocks + norm weights, loaded once, sliced by
    ring position. All arrays are read-only views into the big mmap."""

    def __init__(self, src: Path, g):
        self.g = g
        self.o = {}      # layer -> mmap of the (2048, 2048) pack
        self.dn = {}     # (2048, 6144) pack (chunk-major blocks)
        self.qkv = {}    # (3072, 2048) pack
        self.gu = {}     # concat gate_up (12288, 2048) pack
        self.ln_in = {}  # input_layernorm bf16 (2048,)
        self.ln_post = {}
        with safe_open(str(src / "bf16.safetensors"), framework="pt") as f:
            for n in range(LAYERS):
                for shape, store in (("o", self.o), ("gateup", self.gu),
                                     ("down", self.dn), ("qkv", self.qkv)):
                    path = src / f"layer{n:02d}_{shape}.bin"
                    store[n] = np.memmap(path, dtype=np.uint8, mode="r")
                self.ln_in[n] = f.get_tensor(
                    f"model.layers.{n}.input_layernorm.weight").to(torch.bfloat16)
                self.ln_post[n] = f.get_tensor(
                    f"model.layers.{n}.post_attention_layernorm.weight").to(torch.bfloat16)

    # -- per-position block runs (sub-column law; zeros for padded execs) --
    def o_blocks(self, layer, p):
        if layer is None:
            return zeros_blocks(self.g["N_O"])
        return sub_blocks(self.o[layer], 2048, p, self.g["rows"])

    def qkv_blocks(self, layer, p):
        if layer is None:
            return zeros_blocks(self.g["N_QKV"])
        return sub_blocks(self.qkv[layer], 3072, p, QKV_M // self.g["n"])

    def down_blocks(self, layer, p, c):
        """Chunk c's blocks for position p (the fixture lays them out
        c-major: chunk c's N_O blocks in a row)."""
        if layer is None:
            return zeros_blocks(self.g["N_O"])
        return sub_blocks(self.dn[layer], 2048, p, self.g["rows"],
                          chunks=3, chunk=c)

    def gate_blocks(self, layer, p):
        if layer is None:
            return zeros_blocks(self.g["N_GATE"])
        return sub_blocks(self.gu[layer], 12288, p, self.g["jpw"])

    def up_blocks(self, layer, p):
        if layer is None:
            return zeros_blocks(self.g["N_GATE"])
        return sub_blocks(self.gu[layer], 12288, p, self.g["jpw"], r0_extra=6144)


_ZERO_CACHE = {}


def zeros_blocks(n):
    """n zero weight blocks with valid K=2048/chunk=0 headers (q=0,
    scale=0 -> exact zero partials)."""
    if n not in _ZERO_CACHE:
        b = np.zeros(n * ELEM, dtype=np.uint8)
        for i in range(n):
            off = i * ELEM
            b[off + ELEM - 8 : off + ELEM - 4] = u32(2048)
        _ZERO_CACHE[n] = b
    return _ZERO_CACHE[n]


def patch_k(blocks, k, n):
    """Patch the K header of every ELEM-sized block in a mutable copy."""
    out = np.array(blocks, dtype=np.uint8, copy=True).reshape(n * ELEM)
    for i in range(n):
        off = i * ELEM
        out[off + ELEM - 8 : off + ELEM - 4] = u32(k)
    return out


def norm_element(wgt_bf16, k):
    """K=101/102 element: ln weight bf16[2048] at [0,4096)."""
    e = np.zeros(ELEM, dtype=np.uint8)
    e[0:4096] = wgt_bf16.view(torch.uint16).numpy().view(np.uint8)
    e[ELEM - 8 : ELEM - 4] = u32(k)
    return e


def zero_norm_element(k):
    e = np.zeros(ELEM, dtype=np.uint8)
    e[ELEM - 8 : ELEM - 4] = u32(k)
    return e


def build_exec(lb: LayerBins, mlp, qkv_l):
    """One exec's W tensor: N x N_WELEM x ELEM. COLUMN w carries RING
    POSITION pos(w) blocks (test_layerv2's build_worker_weights(pos(w))
    contract -- the kernel derives its ring position from the worker id
    and stages sw slices / xn chunks by it; pos is an involution, so
    packing column p with position p mirrors the odd columns and the
    ring2 all-gather assembles a scrambled sw -> every down partial
    consumes wrong K-chunks. Board-caught P28-5 (N=8 mirror case)."""
    g = lb.g
    n, N_O, N_GATE = g["n"], g["N_O"], g["N_GATE"]
    N_DOWN, N_QKV, N_WELEM = g["N_DOWN"], g["N_QKV"], g["N_WELEM"]
    w = np.zeros(n * N_WELEM * ELEM, dtype=np.uint8)
    blk = ELEM
    w2 = lb.ln_post[mlp] if mlp is not None else None
    w1 = lb.ln_in[qkv_l] if qkv_l is not None else None
    for col in range(n):
        p = pos(col, n)
        base = col * N_WELEM * blk
        g0 = base + (N_O + 1) * blk
        u0 = g0 + N_GATE * blk
        d0 = u0 + N_GATE * blk
        w10 = d0 + N_DOWN * blk
        q0 = w10 + blk
        w[base : base + N_O * blk] = lb.o_blocks(mlp, p)
        w[base + N_O * blk : g0] = (
            norm_element(w2, 101) if w2 is not None else zero_norm_element(101))
        w[g0 : g0 + N_GATE * blk] = patch_k(lb.gate_blocks(mlp, p), 103, N_GATE)
        w[u0 : u0 + N_GATE * blk] = patch_k(lb.up_blocks(mlp, p), 104, N_GATE)
        # down: c-major chunk blocks, each chunk's run sliced by the law
        for c in range(3):
            dc = d0 + c * N_O * blk
            w[dc : dc + N_O * blk] = patch_k(lb.down_blocks(mlp, p, c), 105, N_O)
        w[w10 : w10 + blk] = (
            norm_element(w1, 102) if w1 is not None else zero_norm_element(102))
        w[q0 : q0 + N_QKV * blk] = lb.qkv_blocks(qkv_l, p)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, choices=(8, 16),
                    help="worker count (16 = P28-6 widened ring; 32 is past "
                         "the npu2 shim channel budget)")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=None,
                    help="default: build/lv2_hy (n=8) / build/lv2_hy_w16 (n=16)")
    ap.add_argument("--execs", type=int, nargs="+", default=None,
                    help="exec ids 1..33 (default: all)")
    args = ap.parse_args()
    g = geom(args.n)
    src = Path(args.src)
    out = Path(args.out) if args.out else (
        OUT if args.n == 8 else OUT.with_name(f"lv2_hy_w{args.n}"))
    out.mkdir(parents=True, exist_ok=True)
    lb = LayerBins(src, g)
    ids = args.execs if args.execs else list(range(1, LAYERS + 2))
    manifest = []
    for e in ids:
        mlp = e - 2 if e >= 2 else None
        qkv_l = e - 1 if e - 1 < LAYERS else None
        w = build_exec(lb, mlp, qkv_l)
        f = out / f"exec{e:02d}.bin"
        w.tofile(f)
        manifest.append({"exec": e, "file": f.name, "mlp": mlp, "qkv": qkv_l})
        print(f"exec {e:2d}: mlp={mlp} qkv={qkv_l} -> {f} "
              f"({len(w) / 1e6:.2f} MB)", flush=True)
    order = list(HAM16) if args.n == 16 else [
        w for c in range(args.n // 4)
        for w in (([c * 4 + r for r in range(4)] if c % 2 == 0
                   else [c * 4 + r for r in reversed(range(4))]))]
    meta = {
        "arch": "hy-mt2-1.8b", "elem": ELEM, "n": args.n,
        "n_welem": g["N_WELEM"], "layers": LAYERS,
        "geometry": {k: g[k] for k in
                     ("rows", "jpw", "N_O", "N_GATE", "N_DOWN", "N_QKV",
                      "N_CXN", "drain_rows")},
        "order": order,
        "pos_table": [pos(w, args.n) for w in range(args.n)],  # w -> ring pos
        "drain_rows_per_worker": g["drain_rows"],  # [qkv 3072/n | xn1 2048/n]
        "execs": manifest,
    }
    with open(out / "meta.json", "w") as fp:
        json.dump(meta, fp, indent=1)
    print(f"done -> {out} ({len(ids)} execs, N={args.n}, "
          f"{g['N_WELEM']} elem/worker, drain {g['drain_rows']} rows/worker)")


if __name__ == "__main__":
    main()
