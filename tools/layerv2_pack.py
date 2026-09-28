#!/usr/bin/env python3
"""P28-5 layerv2 offline weight packer (hy-mt2 1.8b -> 33 exec W tensors).

The layerv2 design (IRON design_layerv2.py) computes, per exec on 8 ring
workers, HALF A LAYER SHIFTED: o_proj+MLP of layer mlp plus qkv of layer
qkv_l, chained by the residual the previous exec drained. The dependency
qkv_L -> host attention -> next exec's X element forces that shift; the
head/tail execs pad the pipeline with exact-zero weight packs:

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
requantization anywhere. The concat gate_up pack (12288 rows) spreads
positions across columns: position p's gate rows live in concat column
p//2 (blocks (p%2)*48..+48) and up in column 4+p//2 -- verified bit-equal
against separately-packed gate/up (tools/p28 archive, P28-5 note).

Per exec bin = the test_layerv2 `weights` tensor: worker w's 186-element
run in fill order [o x16 | w2(K=101) | gate x48(K=103) | up x48(K=104) |
down x48(K=105) | w1(K=102) | qkv x24], sliced by RING POSITION p(w) =
(w<4) ? w : 11-w -- COLUMN w carries position p(w) blocks (the mirror of
the naive p->p packing scrambles the ring gathers; see build_exec).

Run: npu314 python tools/layerv2_pack.py            # all 33 execs (~913MB)
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
COLS = 8
N_O, N_GATE, N_UP, N_DOWN, N_QKV = 16, 48, 48, 48, 24
N_WELEM = N_O + 1 + N_GATE + N_UP + N_DOWN + 1 + N_QKV  # 186
LAYERS = 32

SRC = Path("/home/nzinfo/qwen/xnpu/build/w4u_hy")
OUT = Path("/home/nzinfo/qwen/xnpu/build/lv2_hy")


def u32(x):
    return np.frombuffer(struct.pack("<I", x), dtype=np.uint8)


def pos(w):
    """Worker column -> ring position (serpentine SUCC, an involution)."""
    return w if w < 4 else 11 - w


class LayerBins:
    """Per-layer packed blocks + norm weights, loaded once, sliced by
    ring position. All arrays are read-only views into the big mmap."""

    def __init__(self, src: Path):
        self.o = {}      # pos -> (offset_blocks, buf) layer packs
        self.dn = {}
        self.qkv = {}
        self.gu = {}     # concat gate_up packs
        self.ln_in = {}  # input_layernorm bf16 (2048,)
        self.ln_post = {}
        with safe_open(str(src / "bf16.safetensors"), framework="pt") as f:
            keys = set(f.keys())
            for n in range(LAYERS):
                for shape, store in (("o", self.o), ("gateup", self.gu),
                                     ("down", self.dn), ("qkv", self.qkv)):
                    path = src / f"layer{n:02d}_{shape}.bin"
                    store[n] = np.memmap(path, dtype=np.uint8, mode="r")
                self.ln_in[n] = f.get_tensor(
                    f"model.layers.{n}.input_layernorm.weight").to(torch.bfloat16)
                self.ln_post[n] = f.get_tensor(
                    f"model.layers.{n}.post_attention_layernorm.weight").to(torch.bfloat16)

    # -- per-position block views (16B-granular slices of the mmap) --
    def o_blocks(self, layer, p):
        if layer is None:
            return zeros_blocks(N_O)
        return self.o[layer][p * N_O * ELEM : (p + 1) * N_O * ELEM]

    def qkv_blocks(self, layer, p):
        if layer is None:
            return zeros_blocks(N_QKV)
        return self.qkv[layer][p * N_QKV * ELEM : (p + 1) * N_QKV * ELEM]

    def down_blocks(self, layer, p):
        if layer is None:
            return zeros_blocks(N_DOWN)
        return self.dn[layer][p * N_DOWN * ELEM : (p + 1) * N_DOWN * ELEM]

    def gate_blocks(self, layer, p):
        # concat column p//2, second half of its 96 blocks when p is odd
        return self._gu_slice(layer, p // 2, (p % 2) * N_GATE)

    def up_blocks(self, layer, p):
        return self._gu_slice(layer, 4 + p // 2, (p % 2) * N_UP)

    def _gu_slice(self, layer, col, off):
        if layer is None:
            return zeros_blocks(N_GATE)
        base = (col * 96 + off) * ELEM
        return self.gu[layer][base : base + N_GATE * ELEM]


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
    """One exec's W tensor: cols x 186 x ELEM. COLUMN w carries RING
    POSITION pos(w) = (w<4) ? w : 11-w blocks (test_layerv2's
    build_worker_weights(pos(w)) contract — the kernel derives its ring
    position from the worker id and stages sw slices / xn chunks by it;
    pos is an involution, so packing column p with position p mirrors
    columns 4-7 and the ring2 all-gather assembles sw as
    [s0,s1,s2,s3,s7,s6,s5,s4] -> every down partial consumes mirrored
    K-chunks. Board-caught P28-5: exec01 workers 0-3 exact, 4-7 carried
    position 11-w's qkv)."""
    w = np.zeros(COLS * N_WELEM * ELEM, dtype=np.uint8)
    blk = ELEM
    w2 = lb.ln_post[mlp] if mlp is not None else None
    w1 = lb.ln_in[qkv_l] if qkv_l is not None else None
    for col in range(COLS):
        p = pos(col)
        base = col * N_WELEM * blk
        g0 = base + (N_O + 1) * blk
        u0 = g0 + N_GATE * blk
        d0 = u0 + N_UP * blk
        w10 = d0 + N_DOWN * blk
        q0 = w10 + blk
        w[base : base + N_O * blk] = lb.o_blocks(mlp, p)
        w[base + N_O * blk : g0] = (
            norm_element(w2, 101) if w2 is not None else zero_norm_element(101))
        w[g0 : g0 + N_GATE * blk] = patch_k(lb.gate_blocks(mlp, p), 103, N_GATE)
        w[u0 : u0 + N_UP * blk] = patch_k(lb.up_blocks(mlp, p), 104, N_UP)
        w[d0 : d0 + N_DOWN * blk] = patch_k(lb.down_blocks(mlp, p), 105, N_DOWN)
        w[w10 : w10 + blk] = (
            norm_element(w1, 102) if w1 is not None else zero_norm_element(102))
        w[q0 : q0 + N_QKV * blk] = lb.qkv_blocks(qkv_l, p)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--execs", type=int, nargs="+", default=None,
                    help="exec ids 1..33 (default: all)")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lb = LayerBins(src)
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
    meta = {
        "arch": "hy-mt2-1.8b", "elem": ELEM, "cols": COLS,
        "n_welem": N_WELEM, "layers": LAYERS,
        "order": "[o x16 | w2(101) | gate x48(103) | up x48(104) | "
                 "down x48(105) | w1(102) | qkv x24]",
        "succ": {"0": 1, "1": 2, "2": 3, "3": 7, "7": 6, "6": 5, "5": 4, "4": 0},
        "pos": "p = (w < 4) ? w : 11 - w",
        "drain_rows_per_worker": 640,  # [qkv 384 | xn1 256] bf16
        "execs": manifest,
    }
    with open(out / "meta.json", "w") as fp:
        json.dump(meta, fp, indent=1)
    print(f"done -> {out} ({len(ids)} execs)")


if __name__ == "__main__":
    main()
