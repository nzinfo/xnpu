#!/usr/bin/env python3
"""P28-5 layerv2 E2E golden generator (hy-mt2, one deterministic decode
step at pos from build/dec_hy).

Mirrors the 33-exec layerv2 runtime chain EXACTLY -- kernel numerics in
the test_layerv2 proven form (device-faithful f32 weight dequant
_deq_f32 semantics, sequential sumsq, kernel rsqrt, accurate sigmoid,
g32 activation quantize) and host glue in the decode_export form (rope
with the static NTK-alpha rescaled base, qk-norm AFTER rope, f32
attention) -- over the ENGINE'S OWN v4 packs:

  exec 1    : X q=0 -> o=0; mlp=None (w2=0 zeroes the MLP exactly) ->
              xn1 = x; qkv_0 via rms1(x, ln_in[0])
  exec e    : o=Wo@attn; x'=bf16(x+o); rms2 chain -> gate/up -> sw ->
              q3; down 3 partials f32-summed; xn1=bf16(x'+dacc);
              qkv_{e-1} via rms1(xn1, ln_in[e-1])
  exec 33   : mlp of layer 31 only; qkv section exact zeros (ignored)

Per exec the golden C image is the full drain BO: worker w's 640 bf16
rows = [qkv pos(w)*384..+384 | xn1 pos(w)*256..+256]. Final hidden =
exec 33's xn1; logits = bf16(lm_deq32 @ deq(quantize(rms(x, final_w)))).

Run: ironenv/bin/python tools/layerv2_golden.py            # all 33
     ironenv/bin/python tools/layerv2_golden.py --execs 1 2 --skip-lm
"""

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ELEM = 18560
COLS = 8
N_O, N_GATE, N_UP, N_DOWN, N_QKV = 16, 48, 48, 48, 24
N_WELEM = 186
LAYERS = 32
HIDDEN, HEAD_DIM, HEADS, KV = 2048, 128, 16, 4
EPS = 1e-5
ROPE_BASE = 10000.0 * 1000.0 ** (128.0 / 126.0)

W4DIR = Path("/home/nzinfo/qwen/xnpu/build/w4u_hy")
DEC = Path("/home/nzinfo/qwen/xnpu/build/dec_hy")
OUT = Path("/home/nzinfo/qwen/xnpu/build/lv2_hy")

_spec = importlib.util.spec_from_file_location(
    "w4ref", "/home/nzinfo/qwen/xnpu/IRON/iron/operators/w4gemvu/reference.py")
_w4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_w4)


def pos(w):
    return w if w < 4 else 11 - w


# ---- kernel-chain helpers (test_layerv2 forms, verbatim) ----

def _f32(b):
    return (np.asarray(b, dtype=np.uint32) << 16).view(np.float32).copy()


def _bf16(f32):
    u = np.asarray(f32, dtype=np.float32).view(np.uint32)
    rounded = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _kernel_rsqrt(s):
    v = np.float32(s)
    magic = np.array(0x5F3759DF, dtype=np.uint32)
    x = (magic - (v.view(np.uint32) >> np.uint32(1))).view(np.float32)
    for _ in range(3):
        x = np.float32(x * (np.float32(2.0) - v * x * x))
    return x


def _sumsq_sequential(h2f):
    sumsq = np.float32(0)
    for v in h2f:
        sumsq = np.float32(sumsq + np.float32(v * v))
    return sumsq


def _quantize_kernel(xn_f32, m):
    v32 = xn_f32.reshape(m // 32, 32)
    amax = np.abs(v32).max(axis=1)
    d_bits = _bf16((amax / np.float32(127.0)).astype(np.float32))
    invd = (np.float32(1.0) / _f32(d_bits.astype(np.uint32)))[:, None]
    v = v32 * invd
    v = np.where(amax[:, None] == 0, 0.0, v)
    q = np.round(v)
    q = np.clip(q, -127, 127).astype(np.int8).reshape(m)
    return q, d_bits


def _dequant(q, d_bits):
    return q.astype(np.float32) * np.repeat(_f32(d_bits.astype(np.uint32)), 32)


def _quantize_vector_bf16(x_bf16_bits):
    """Host-glue g32 quantize on a bf16 vector (torch.round ties-even ==
    Rust round_ties_even; reference.quantize_vector contract)."""
    x = torch.from_numpy(_f32(x_bf16_bits))
    q, d, x_deq = _w4.quantize_vector(x)
    return q.numpy(), d.view(torch.uint16).numpy(), x_deq.numpy().astype(np.float32)


# ---- host-glue helpers (decode_export forms) ----

def rms_weighted_bits(x_bits, w_bits, eps=EPS):
    """Kernel rms chain: h2=x bf16, sumsq sequential f32, kernel rsqrt,
    y = x*inv*w UNROUNDED (feeds the g32 quantize)."""
    xr = _f32(x_bits)
    inv = _kernel_rsqrt(np.float32(_sumsq_sequential(xr) / np.float32(2048) + np.float32(eps)))
    return (xr * inv * _f32(w_bits)).astype(np.float32)


def rope_pairs(xf, p):
    j = np.arange(HEAD_DIM // 2, dtype=np.float32)
    ang = np.float32(p) * (np.float32(ROPE_BASE) ** (-j / np.float64(HEAD_DIM // 2))).astype(np.float32)
    cos, sin = np.cos(ang), np.sin(ang)
    out = np.empty_like(xf)
    out[..., :64] = xf[..., :64] * cos - xf[..., 64:] * sin
    out[..., 64:] = xf[..., 64:] * cos + xf[..., :64] * sin
    return out


def qk_rms_bits(x_bits, w_bits):
    """Per-head rms over 128 after rope, bf16 out."""
    x = _f32(x_bits).reshape(-1, HEAD_DIM)
    w = _f32(w_bits)
    out = np.zeros_like(x)
    for h in range(x.shape[0]):
        v = x[h]
        ms = np.float32(0)
        for t in v:
            ms = np.float32(ms + np.float32(t * t))
        inv = _kernel_rsqrt(np.float32(ms / np.float32(HEAD_DIM) + np.float32(EPS)))
        out[h] = v * inv * w
    return _bf16(out).reshape(-1)


def attention_bits(q_bits, kc_bits, vc_bits, p):
    """GQA attention, f32 scores/softmax/PV, bf16 out. caches: (KV,S,128)
    bf16 bits with row p already written."""
    S = p + 1
    q = _f32(q_bits).reshape(HEADS, HEAD_DIM)
    k = _f32(kc_bits[:, :S])  # (KV,S,128)
    v = _f32(vc_bits[:, :S])
    group = HEADS // KV
    out = np.zeros(HEADS * HEAD_DIM, dtype=np.uint16)
    for h in range(HEADS):
        kk, vv = k[h // group], v[h // group]
        scores = kk @ q[h] / np.float32(HEAD_DIM ** 0.5)
        scores = scores - scores.max()
        e = np.exp(scores)
        p_ = e / e.sum()
        out[h * HEAD_DIM:(h + 1) * HEAD_DIM] = _bf16(p_ @ vv)
    return out


# ---- v4 pack -> exact f32 dequant (bit-faithful device weights) ----

def unpack_deq32(packed_path, M, K):
    """Reverse of reference.quantize_and_pack: int4 q and bf16 sf from the
    v4 blocks -> W = q * sf exactly (f32), row-major (M, K)."""
    buf = np.memmap(packed_path, dtype=np.uint8, mode="r")
    cols, tr, chunks = 8, 16, K // 2048
    T = M // cols // tr
    W = np.zeros((M, K), dtype=np.float32)
    groups = 2048 // 32
    for col in range(cols):
        rows0 = col * (M // cols)
        for c in range(chunks):
            for t in range(T):
                off = ((col * T * chunks) + c * T + t) * ELEM
                r0 = rows0 + t * tr
                # nibbles: (64 groups, 32 k, 16 rows), byte j packs rows
                # 2j (lo) | 2j+1 (hi) at k = j//8
                nb = np.frombuffer(buf[off: off + tr * 2048 // 2],
                                   dtype=np.uint8).reshape(groups, 32 * tr // 2)
                lo = (nb & 0x0F).astype(np.int8)
                hi = (nb >> 4).astype(np.int8)
                lo = np.where(lo > 7, lo - 16, lo).astype(np.float32)
                hi = np.where(hi > 7, hi - 16, hi).astype(np.float32)
                q = np.empty((groups, 32, tr), dtype=np.float32)
                q[..., 0::2] = lo.reshape(groups, 32, tr // 2)
                q[..., 1::2] = hi.reshape(groups, 32, tr // 2)
                # scales: bf16[64][16] at g*16+n -> (rows, groups)
                sfb = np.frombuffer(buf[off + tr * 2048 // 2:
                                        off + tr * 2048 // 2 + tr * groups * 2],
                                    dtype=np.uint16).reshape(groups, tr)
                sf = _f32(sfb.T.astype(np.uint32))  # (tr, groups)
                # W[r, c*2048 + g*32 + k] = q[g, k, r] * sf[r, g]
                W[r0:r0 + tr, c * 2048:(c + 1) * 2048] = (
                    q.transpose(2, 0, 1).reshape(tr, groups * 32) * np.repeat(sf, 32, axis=1))
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execs", type=int, nargs="+", default=None)
    ap.add_argument("--skip-lm", action="store_true")
    args = ap.parse_args()
    out = OUT
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((DEC / "meta.json").read_text())
    p = meta["pos"]
    S = meta["cache_seq"]

    x_bits = np.fromfile(DEC / "x0.bin", dtype=np.uint16)
    kc = np.fromfile(DEC / "kcache.bin", dtype=np.uint16).reshape(LAYERS, KV, S, HEAD_DIM).copy()
    vc = np.fromfile(DEC / "vcache.bin", dtype=np.uint16).reshape(LAYERS, KV, S, HEAD_DIM).copy()

    with safe_open(str(W4DIR / "bf16.safetensors"), framework="pt") as f:
        g = lambda n: f.get_tensor(n).to(torch.bfloat16).view(torch.uint16).numpy()
        ln_in = [g(f"model.layers.{n}.input_layernorm.weight") for n in range(LAYERS)]
        ln_post = [g(f"model.layers.{n}.post_attention_layernorm.weight") for n in range(LAYERS)]
        qn = [g(f"model.layers.{n}.self_attn.q_norm.weight") for n in range(LAYERS)]
        kn = [g(f"model.layers.{n}.self_attn.k_norm.weight") for n in range(LAYERS)]
        final_w = g("model.norm.weight")

    cache = {}

    def W(kind, layer):
        key = (kind, layer)
        if key not in cache:
            shape = {"o": (2048, 2048), "gate": (6144, 2048), "up": (6144, 2048),
                     "down": (2048, 6144), "qkv": (3072, 2048)}[kind]
            if kind in ("gate", "up"):
                # slice the concat gateup pack's columns (verified layout)
                Wc = unpack_deq32(W4DIR / f"layer{layer:02d}_gateup.bin", 12288, 2048)
                half = 6144
                cache[key] = Wc[:half] if kind == "gate" else Wc[half:]
                cache[("up" if kind == "gate" else "gate", layer)] = (
                    Wc[half:] if kind == "gate" else Wc[:half])
            else:
                cache[key] = unpack_deq32(
                    W4DIR / f"layer{layer:02d}_{kind}.bin", *shape)
            if len(cache) > 8:  # keep it bounded
                for k in list(cache)[:len(cache) - 8]:
                    if k != key:
                        cache.pop(k, None)
        return cache[key]

    def gemv_bf16(Wm, x_deq):
        return _bf16(Wm @ x_deq)

    def mlp_chain(xn_bits, w2_bits, layer):
        """rms2 -> gate/up -> sw -> q3; returns q3, d3 (device semantics)."""
        xn2_f = rms_weighted_bits(xn_bits, w2_bits)
        q2, d2 = _quantize_kernel(xn2_f, 2048)
        x_deq2 = _dequant(q2, d2)
        g_f = _f32(gemv_bf16(W("gate", layer), x_deq2))
        u_f = _f32(gemv_bf16(W("up", layer), x_deq2))
        sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-g_f.astype(np.float32)))
        sw_bits = _bf16(g_f * sig * u_f)
        q3, d3 = _quantize_kernel(_f32(sw_bits.astype(np.uint32)), 6144)
        return q3, d3

    def down_chain(q3, d3, layer):
        x_deq3 = _dequant(q3, d3)
        Wd = W("down", layer)
        acc = np.zeros(2048, dtype=np.float32)
        for c in range(3):
            part = gemv_bf16(Wd[:, c * 2048:(c + 1) * 2048],
                             x_deq3[c * 2048:(c + 1) * 2048])
            acc += _f32(part.astype(np.uint32))
        return acc

    def qkv_chain(xn1_bits, w1_bits, layer):
        xn_f = rms_weighted_bits(xn1_bits, w1_bits)
        q4, d4 = _quantize_kernel(xn_f, 2048)
        return gemv_bf16(W("qkv", layer), _dequant(q4, d4))

    def host_attn(qkv_bits, layer, p):
        q_bits = qkv_bits[:2048].copy()
        k_bits = qkv_bits[2048:2048 + 512].copy()
        v_bits = qkv_bits[2048 + 512:].copy()
        q_bits = _bf16(rope_pairs(_f32(q_bits).reshape(HEADS, HEAD_DIM), p).reshape(-1))
        k_bits = _bf16(rope_pairs(_f32(k_bits).reshape(KV, HEAD_DIM), p).reshape(-1))
        q_bits = qk_rms_bits(q_bits, qn[layer])
        k_bits = qk_rms_bits(k_bits, kn[layer])
        kc[layer][:, p] = k_bits.reshape(KV, HEAD_DIM)
        vc[layer][:, p] = v_bits.reshape(KV, HEAD_DIM)
        return attention_bits(q_bits, kc[layer], vc[layer], p)

    def drain_image(qkv_bits, xn1_bits):
        """Full C BO image: worker w = [qkv pos(w) slice | xn1 pos(w) chunk]."""
        img = np.zeros(COLS * 640, dtype=np.uint16)
        for w in range(COLS):
            pp = pos(w)
            img[w * 640 : w * 640 + 384] = qkv_bits[pp * 384 : (pp + 1) * 384]
            img[w * 640 + 384 : (w + 1) * 640] = xn1_bits[pp * 256 : (pp + 1) * 256]
        return img

    ids = args.execs if args.execs else list(range(1, LAYERS + 2))
    # state walked in exec order (chain must run sequentially even for
    # subset runs; only the requested execs are WRITTEN)
    xn_bits = x_bits.copy()
    attn_bits = np.zeros(2048, dtype=np.uint16)  # exec 1: q=0 -> o=0
    hidden_bits = None
    for e in range(1, LAYERS + 2):
        mlp = e - 2 if e >= 2 else None
        ql = e - 1 if e - 1 < LAYERS else None
        # o + residual
        if mlp is None:
            xp_bits = xn_bits  # o == 0 exactly (X q=0)
        else:
            qX, dX, x_deq1 = _quantize_vector_bf16(attn_bits)
            o_bits = gemv_bf16(W("o", mlp), x_deq1)
            xp_bits = _bf16(_f32(xn_bits) + _f32(o_bits))
        # mlp
        if mlp is None:
            xn1_bits = xp_bits  # w2=0 -> exact zero MLP
        else:
            q3, d3 = mlp_chain(xp_bits, ln_post[mlp], mlp)
            xn1_bits = _bf16(_f32(xp_bits) + down_chain(q3, d3, mlp))
        # qkv
        qkv_bits = (np.zeros(3072, dtype=np.uint16) if ql is None
                    else qkv_chain(xn1_bits, ln_in[ql], ql))
        if e in ids:
            (out / f"golden_exec{e:02d}.bin").write_bytes(
                drain_image(qkv_bits, xn1_bits).tobytes())
            print(f"exec {e:2d}: mlp={mlp} qkv={ql} golden written", flush=True)
        # host attention for the NEXT exec's X element
        if ql is not None:
            attn_bits = host_attn(qkv_bits, ql, p)
        else:
            hidden_bits = xn1_bits
        xn_bits = xn1_bits
        if e == max(ids):
            break  # later execs only matter if their goldens are wanted

    if not args.skip_lm and hidden_bits is not None:
        hf = rms_weighted_bits(hidden_bits, final_w)
        q5, d5 = _quantize_kernel(hf, 2048)
        Wlm = unpack_deq32(W4DIR / "lmhead.bin", 121088, 2048)
        logits = gemv_bf16(Wlm, _dequant(q5, d5))
        (out / "golden_hidden.bin").write_bytes(hidden_bits.tobytes())
        (out / "golden_logits.bin").write_bytes(logits.tobytes())
        lf = _f32(logits)
        print(f"final: hidden + logits written; argmax {int(np.argmax(lf))} "
              f"top3 {np.argsort(lf)[-3:][::-1].tolist()}")
    meta_out = {"pos": p, "cache_seq": S, "bands": {"qkv": [0.08, 0.8],
               "xn1": [0.08, 200.0]}, "drain_rows": 640}
    with open(out / "golden_meta.json", "w") as fp:
        json.dump(meta_out, fp, indent=1)


if __name__ == "__main__":
    main()
