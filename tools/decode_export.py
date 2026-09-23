#!/usr/bin/env python3
"""Offline exporter for the M3b decode-chain harness (tools + golden).

Builds ONE deterministic decode step for MiniCPM5-2B and everything the
Rust `run-decode` command needs to execute and verify it:

  norms.bin      85 x 2048 bf16  (L0.in, L0.post, L1.in, ... final norm)
  x0.bin         2048 bf16       the "embedding" input vector
  kcache.bin     [42][2][S][128] bf16   deterministic KV cache (2 real heads;
  vcache.bin                            Rust expands h//8 like the app)
  golden_hidden.bin  2048 bf16   final hidden (after model.norm) — reference
  golden_layers.bin  42 x 2048   hidden after each layer's final residual

Reference math mirrors the Rust glue EXACTLY (f32 compute, one bf16
rounding per op boundary) and uses the DEQUANTIZED w4 weights, so the
NPU/Rust pipeline should reproduce it to accumulation-order noise:
  rmsnorm(x,w): y = bf16(x_f32 * rsqrt(mean(x^2)+1e-5) * w_f32)
  gemv:         y = bf16(W_deq_f32 @ x_f32)
  rope:         llama rotate-half, inv_freq = 5e6^(-j/64), j=0..63, f32
  attention:    GQA h//8, scores/softmax/PV in f32, out bf16
  add/swiglu:   f32 -> bf16 (silu(x) = x*sigmoid(x))

Run: ironenv/bin/python tools/decode_export.py [--out build/dec] [--pos 100]
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

_spec = importlib.util.spec_from_file_location(
    "w4uref", "/home/nzinfo/qwen/xnpu/IRON/iron/operators/w4gemvu/reference.py"
)
_w4uref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_w4uref)
quantize_and_pack = _w4uref.quantize_and_pack

MODEL = Path("/home/nzinfo/qwen/xnpu/models/minicpm5-2b/model.safetensors")

# name -> (row tensors in concat order, M, K) — must match W4U_SHAPES in Rust
SHAPES = {
    "qkv": ([("q_proj", 2048), ("k_proj", 256), ("v_proj", 256)], 2560, 2048),
    "o": ([("o_proj", 2048)], 2048, 2048),
    "gateup": ([("gate_proj", 6144), ("up_proj", 6144)], 12288, 2048),
    "down": ([("down_proj", 2048)], 2048, 6144),
}

N_LAYERS = 42
HIDDEN = 2048
N_HEADS = 16
N_KV = 2
HEAD_DIM = 128
EPS = 1e-5
ROPE_BASE = 5e6


def lcg_vec(n, mod=13, div=4.0, seed=12345):
    """Deterministic bf16 pattern: ((lcg % mod) - mod//2) / div."""
    out = np.empty(n, dtype=np.float32)
    s = seed
    for i in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & ((1 << 63) - 1)
        out[i] = ((s >> 33) % mod - mod // 2) / div
    return torch.from_numpy(out).to(torch.bfloat16)


def rms_norm(x, w):
    xf = x.to(torch.float32)
    ms = xf.pow(2).mean()
    y = xf * torch.rsqrt(ms + EPS) * w.to(torch.float32)
    return y.to(torch.bfloat16)


def rope(x, pos):
    """x: (heads, head_dim) bf16 -> rotate-half rope at position pos, bf16."""
    xf = x.to(torch.float32)
    j = torch.arange(HEAD_DIM // 2, dtype=torch.float32)
    inv = ROPE_BASE ** (-j / (HEAD_DIM // 2))
    ang = pos * inv
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = xf[..., : HEAD_DIM // 2], xf[..., HEAD_DIM // 2:]
    out = torch.empty_like(xf)
    out[..., : HEAD_DIM // 2] = x1 * cos - x2 * sin
    out[..., HEAD_DIM // 2:] = x2 * cos + x1 * sin
    return out.to(torch.bfloat16)


def attention(q, k_cache, v_cache, pos):
    """One decode step. q: (16,128) bf16 (already roped); k/v_cache:
    (2, S, 128) bf16. GQA: q head h reads kv head h // (16//2). f32 math."""
    group = N_HEADS // N_KV
    S = pos + 1
    out = torch.empty(N_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    for h in range(N_HEADS):
        kv = h // group
        k = k_cache[kv, :S].to(torch.float32)  # (S,128)
        v = v_cache[kv, :S].to(torch.float32)
        scores = (k @ q[h].to(torch.float32)) / (HEAD_DIM**0.5)
        scores = scores - scores.max()
        p = torch.softmax(scores, dim=0)
        out[h] = (p @ v).to(torch.bfloat16)
    return out


def swiglu(x):
    """x: (12288,) bf16 [gate | up] -> silu(gate)*up, bf16."""
    g = x[: x.numel() // 2].to(torch.float32)
    u = x[x.numel() // 2 :].to(torch.float32)
    return (g * torch.sigmoid(g) * u).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/nzinfo/qwen/xnpu/build/dec")
    ap.add_argument("--pos", type=int, default=100)
    ap.add_argument("--cache-seq", type=int, default=1024)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pos, S = args.pos, args.cache_seq

    from safetensors import safe_open

    x = lcg_vec(HIDDEN, mod=17, div=4.0)  # the decode-step input
    x0 = x.clone()  # the loop below consumes x — keep the step input
    kcache = torch.zeros(N_LAYERS, N_KV, S, HEAD_DIM, dtype=torch.bfloat16)
    vcache = torch.zeros(N_LAYERS, N_KV, S, HEAD_DIM, dtype=torch.bfloat16)
    for n in range(N_LAYERS):
        for kv in range(N_KV):
            kcache[n, kv, : pos + 1] = lcg_vec((pos + 1) * HEAD_DIM,
                                               seed=1000 * n + 7 * kv + 1).view(pos + 1, HEAD_DIM)
            vcache[n, kv, : pos + 1] = lcg_vec((pos + 1) * HEAD_DIM,
                                               seed=1000 * n + 7 * kv + 2).view(pos + 1, HEAD_DIM)

    # Layer weights: dequantized w4 for the reference, per layer/shape.
    norms = []
    with safe_open(str(MODEL), framework="pt") as f:
        for n in range(N_LAYERS):
            wdeq = {}
            for shape, (parts, m, k) in SHAPES.items():
                W = torch.cat(
                    [f.get_tensor(f"model.layers.{n}.self_attn.{p}.weight")
                     if "proj" in p and p in ("q_proj", "k_proj", "v_proj", "o_proj")
                     else f.get_tensor(f"model.layers.{n}.mlp.{p}.weight")
                     for p, _ in parts],
                    dim=0,
                ).to(torch.float32)
                assert W.shape == (m, k)
                _, w_dequant = quantize_and_pack(
                    W.numpy(), group_size=32, m_input=4, cols=8
                )
                wdeq[shape] = w_dequant.to(torch.float32)
            n1 = f.get_tensor(f"model.layers.{n}.input_layernorm.weight")
            n2 = f.get_tensor(f"model.layers.{n}.post_attention_layernorm.weight")
            norms.append((n1, n2))

            # --- the reference decode step (mirrors the Rust glue) ---
            xn = rms_norm(x, n1)
            qkv = (wdeq["qkv"] @ xn.to(torch.float32)).to(torch.bfloat16)
            q, k, v = qkv[:2048], qkv[2048:2304], qkv[2304:2560]
            q = rope(q.view(N_HEADS, HEAD_DIM), pos)
            k = rope(k.view(N_KV, HEAD_DIM), pos)
            kcache[n, :, pos] = k
            vcache[n, :, pos] = v.view(N_KV, HEAD_DIM)
            attn = attention(q, kcache[n], vcache[n], pos)  # (16,128)
            o = (wdeq["o"] @ attn.reshape(-1).to(torch.float32)).to(torch.bfloat16)
            x = (x.to(torch.float32) + o.to(torch.float32)).to(torch.bfloat16)
            h = rms_norm(x, n2)
            gu = (wdeq["gateup"] @ h.to(torch.float32)).to(torch.bfloat16)
            sw = swiglu(gu)
            d = (wdeq["down"] @ sw.to(torch.float32)).to(torch.bfloat16)
            x = (x.to(torch.float32) + d.to(torch.float32)).to(torch.bfloat16)
            (out / f"golden_L{n:02}.bin").write_bytes(
                x.view(torch.uint16).numpy().tobytes()
            )
            print(f"layer {n}: reference step done", flush=True)

        final_w = f.get_tensor("model.norm.weight")
    hidden = rms_norm(x, final_w)

    norms_flat = torch.cat([t for pair in norms for t in pair] + [final_w])
    (out / "norms.bin").write_bytes(
        norms_flat.contiguous().view(torch.uint16).numpy().tobytes())
    (out / "x0.bin").write_bytes(x0.view(torch.uint16).numpy().tobytes())
    # Zero the pos slice in the exported caches: the Rust side must write its
    # OWN roped k / v there (a rope bug then diverges from the reference,
    # which attended with the correct values).
    kcache[:, :, pos, :] = 0
    (out / "kcache.bin").write_bytes(
        kcache.contiguous().view(torch.uint16).numpy().tobytes())
    (out / "vcache.bin").write_bytes(
        vcache.contiguous().view(torch.uint16).numpy().tobytes())
    (out / "golden_hidden.bin").write_bytes(
        hidden.view(torch.uint16).numpy().tobytes())
    with open(out / "meta.json", "w") as fp:
        json.dump({"pos": pos, "cache_seq": S, "layers": N_LAYERS,
                   "hidden": HIDDEN, "heads": N_HEADS, "kv_heads": N_KV,
                   "head_dim": HEAD_DIM, "eps": EPS, "rope_base": ROPE_BASE},
                  fp, indent=1)
    print(f"done -> {out} (golden_hidden + 42 layer goldens + caches)")


if __name__ == "__main__":
    main()
