#!/usr/bin/env python3
"""P28-9 ask mode: REAL-prompt prefill for the lv2 E2E (hy-mt2).

The E2E fixtures (build/dec_hy) are SYNTHETIC (deterministic x0/caches,
no text). This script runs the golden-chain math (layerv2_golden forms,
verbatim) over a REAL tokenized prompt to produce what `run-decode ...
LV2_ASK` needs: genuine per-layer KV caches, the decode start position,
and the first greedy token. The decode steps themselves run on the NPU
(33 execs + lm per token, host attention between execs — the engine's
actual architecture). Embedding = lm_head rows (tie_word_embeddings).

Outputs <out>/: kcache.bin, vcache.bin ([L][KV][1024][128] bf16, prompt
entries 0..P-1, rest zero), meta.json {pos: P, first: t0, steps}.
The first sampled token's logits come from the reference chain here so
the Rust loop can start from it.

Run: ironenv/bin/python tools/lv2_ask.py "你是谁？" --steps 24
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

ELEM = 18560
LAYERS, HIDDEN, HEAD_DIM, HEADS, KV = 32, 2048, 128, 16, 4
QKV_M = 3072
EPS = 1e-5
ROPE_BASE = 11158840.0
CACHE_SEQ = 1024
MODEL_DIR = Path.home() / ".config/flm/models/Hy-MT2-1.8B-NPU2"
W4DIR = Path("/home/nzinfo/qwen/xnpu/build/w4u_hy")
BF16 = Path("/home/nzinfo/qwen/xnpu/build/w4u_hy/bf16.safetensors")


# ---- kernel-chain helpers (layerv2_golden forms, verbatim) ----
def _f32(b):
    return (np.asarray(b, dtype=np.uint32) << 16).view(np.float32).copy()


def _bf16(f32):
    u = np.asarray(f32, dtype=np.float32).view(np.uint32)
    rounded = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _kernel_rsqrt(s):
    v = np.float32(s)
    x = (np.array(0x5F3759DF, dtype=np.uint32) - (v.view(np.uint32) >> np.uint32(1))).view(np.float32)
    for _ in range(3):
        x = np.float32(x * (np.float32(2.0) - v * x * x))
    return x


def _sumsq_seq(h2f):
    s = np.float32(0)
    for v in h2f:
        s = np.float32(s + np.float32(v * v))
    return s


def _quantize_kernel(xn_f32, m):
    v32 = xn_f32.reshape(m // 32, 32)
    amax = np.abs(v32).max(axis=1)
    d = _bf16((amax / np.float32(127.0)).astype(np.float32))
    invd = (np.float32(1.0) / _f32(d.astype(np.uint32)))[:, None]
    v = np.where(amax[:, None] == 0, 0.0, v32 * invd)
    q = np.clip(np.round(v), -127, 127).astype(np.int8).reshape(m)
    return q, d


def _dequant(q, d):
    return q.astype(np.float32) * np.repeat(_f32(d.astype(np.uint32)), 32)


def rms_w(x_bits, w_bits):
    xr = _f32(x_bits)
    inv = _kernel_rsqrt(np.float32(_sumsq_seq(xr) / np.float32(2048) + EPS))
    return (xr * inv * _f32(w_bits)).astype(np.float32)


def rope_pairs(xf, p):
    j = np.arange(HEAD_DIM // 2, dtype=np.float32)
    ang = np.float32(p) * (np.float32(ROPE_BASE) ** (-j / np.float64(HEAD_DIM // 2))).astype(np.float32)
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(xf)
    out[..., :64] = xf[..., :64] * c - xf[..., 64:] * s
    out[..., 64:] = xf[..., 64:] * c + xf[..., :64] * s
    return out


def qk_rms(x_bits, w_bits):
    x = _f32(x_bits).reshape(-1, HEAD_DIM)
    w = _f32(w_bits)
    out = np.zeros_like(x)
    for h in range(x.shape[0]):
        v = x[h]
        ms = np.float32(0)
        for t in v:
            ms = np.float32(ms + np.float32(t * t))
        inv = _kernel_rsqrt(np.float32(ms / HEAD_DIM + EPS))
        out[h] = v * inv * w
    return _bf16(out).reshape(-1)


def attention(q_bits, kc, vc, p):
    S = p + 1
    q = _f32(q_bits).reshape(HEADS, HEAD_DIM)
    k = _f32(kc[:, :S])
    v = _f32(vc[:, :S])
    grp = HEADS // KV
    out = np.zeros(HEADS * HEAD_DIM, dtype=np.uint16)
    for h in range(HEADS):
        kk, vv = k[h // grp], v[h // grp]
        sc = kk @ q[h] / np.float32(HEAD_DIM ** 0.5)
        sc = sc - sc.max()
        e = np.exp(sc)
        out[h * HEAD_DIM:(h + 1) * HEAD_DIM] = _bf16((e / e.sum()) @ vv)
    return out


def unpack_deq32(path, M, K):
    buf = np.memmap(path, dtype=np.uint8, mode="r")
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
                nb = np.frombuffer(buf[off: off + tr * 2048 // 2], dtype=np.uint8).reshape(groups, 32 * tr // 2)
                lo = (nb & 0x0F).astype(np.int8)
                hi = (nb >> 4).astype(np.int8)
                lo = np.where(lo > 7, lo - 16, lo).astype(np.float32)
                hi = np.where(hi > 7, hi - 16, hi).astype(np.float32)
                q = np.empty((groups, 32, tr), dtype=np.float32)
                q[..., 0::2] = lo.reshape(groups, 32, tr // 2)
                q[..., 1::2] = hi.reshape(groups, 32, tr // 2)
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
    ap.add_argument("question")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--out", default="/home/nzinfo/qwen/xnpu/build/dec_ask")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    text = f"<｜hy_begin▁of▁sentence｜><｜hy_User｜>{args.question}<｜hy_Assistant｜>"
    ids = tok.encode(text).ids
    P = len(ids)
    print(f"prompt tokens ({P}): {ids}")

    with safe_open(str(BF16), framework="pt") as f:
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
            shape = {"o": (2048, 2048), "down": (2048, 6144), "qkv": (QKV_M, 2048)}.get(kind, (12288, 2048))
            if kind in ("gate", "up"):
                Wc = unpack_deq32(W4DIR / f"layer{layer:02d}_gateup.bin", 12288, 2048)
                half = 6144
                cache[key] = Wc[:half] if kind == "gate" else Wc[half:]
                cache[("up" if kind == "gate" else "gate", layer)] = Wc[half:] if kind == "gate" else Wc[:half]
            else:
                cache[key] = unpack_deq32(W4DIR / f"layer{layer:02d}_{kind}.bin", *shape)
            if len(cache) > 4:
                for k in list(cache)[:len(cache) - 4]:
                    cache.pop(k, None)
        return cache[key]

    print("loading lm_head (embedding table)...", flush=True)
    Wlm = unpack_deq32(W4DIR / "lmhead.bin", 121088, 2048)
    Wlm_bits = _bf16(Wlm)  # bf16 embedding rows

    kc = np.zeros((LAYERS, KV, CACHE_SEQ, HEAD_DIM), dtype=np.uint16)
    vc = np.zeros_like(kc)

    def layer_step(x_bits, p):
        for l in range(LAYERS):
            xf = rms_w(x_bits, ln_in[l])
            q4, d4 = _quantize_kernel(xf, 2048)
            qkv = _bf16(W("qkv", l) @ _dequant(q4, d4))
            q = _bf16(rope_pairs(_f32(qkv[:2048]).reshape(HEADS, HEAD_DIM), p).reshape(-1))
            k = _bf16(rope_pairs(_f32(qkv[2048:2048 + 512]).reshape(KV, HEAD_DIM), p).reshape(-1))
            q = qk_rms(q, qn[l])
            k = qk_rms(k, kn[l])
            kc[l][:, p] = k.reshape(KV, HEAD_DIM)
            vc[l][:, p] = qkv[2048 + 512:].reshape(KV, HEAD_DIM)
            a = attention(q, kc[l], vc[l], p)
            qX, dX = _quantize_kernel(_f32(a), 2048)
            o = _bf16(W("o", l) @ (qX.astype(np.float32) * np.repeat(_f32(dX.astype(np.uint32)), 32)))
            xp = _bf16(_f32(x_bits) + _f32(o))
            x2 = rms_w(xp, ln_post[l])
            q2, d2 = _quantize_kernel(x2, 2048)
            g_ = _f32(_bf16(W("gate", l) @ _dequant(q2, d2)))
            u_ = _f32(_bf16(W("up", l) @ _dequant(q2, d2)))
            sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-g_))
            sw = _bf16(g_ * sig * u_)
            q3, d3 = _quantize_kernel(_f32(sw), 6144)
            Wd = W("down", l)
            acc = np.zeros(2048, dtype=np.float32)
            xq3 = _dequant(q3, d3)
            for c in range(3):
                part = _bf16(Wd[:, c * 2048:(c + 1) * 2048] @ xq3[c * 2048:(c + 1) * 2048])
                acc += _f32(part)
            x_bits = _bf16(_f32(xp) + acc)
        return x_bits

    print("prefill...", flush=True)
    x = Wlm_bits[ids[0]].copy()
    for p in range(P):
        x = layer_step(Wlm_bits[ids[p]].copy(), p)
        print(f"  pos {p} done", flush=True)

    hf = rms_w(x, final_w)
    q5, d5 = _quantize_kernel(hf, 2048)
    logits = _f32(_bf16(Wlm @ _dequant(q5, d5)))
    first = int(np.argmax(logits))
    top = np.argsort(logits)[-3:][::-1].tolist()
    print(f"first token {first} (top3 {top})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kc.tofile(out / "kcache.bin")
    vc.tofile(out / "vcache.bin")
    with open(out / "ask.txt", "w") as fp:
        fp.write(f"{P} {first}\n")
    with open(out / "meta.json", "w") as fp:
        json.dump({"pos": P, "first": first, "steps": args.steps,
                   "question": args.question}, fp, indent=1)
    print(f"written {out}/kcache.bin vcache.bin ask.txt meta.json (pos={P}, first={first})")


if __name__ == "__main__":
    main()
