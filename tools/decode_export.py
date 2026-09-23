#!/usr/bin/env python3
"""Offline exporter for the M3b/M5c decode-chain harness (tools + golden).

Builds ONE deterministic decode step and everything the Rust `run-decode`
command needs to execute and verify it, for either supported arch:

  --arch minicpm  MiniCPM5-2B (default): 42L, 16Q/2KV, no qk-norm,
                  rope base 5e6, weights from HF safetensors.
  --arch hy       hy-mt2 1.8B (M5c): 32L, 16Q/4KV, qk-norm AFTER rope
                  (hunyuan_npu.hpp), rope base_eff = theta*alpha^(d/(d-2))
                  = 10000*1000^(128/126) (libhunyuan_npu.so disasm: static
                  NTK-alpha rescale; beta_fast/beta_slow unused in binary),
                  weights decoded from FLM model.q4nx (tools/q4nx_import).

  norms.bin     (2L+1) x 2048 bf16  (L0.in, L0.post, L1.in, ... final norm)
  qknorms.bin   hy only: L x 256 bf16  ([q_norm 128 | k_norm 128] per layer)
  x0.bin        2048 bf16       the "embedding" input vector
  kcache.bin    [L][KV][S][128] bf16   deterministic KV cache (Rust expands
  vcache.bin                            GQA group like the app)
  golden_hidden.bin  2048 bf16   final hidden (after model.norm) — reference
  golden_layers.bin  L x 2048   hidden after each layer's final residual

Reference math mirrors the Rust glue EXACTLY (f32 compute, one bf16
rounding per op boundary) and uses the DEQUANTIZED w4 weights, so the
NPU/Rust pipeline should reproduce it to accumulation-order noise:
  rmsnorm(x,w): y = bf16(x_f32 * rsqrt(mean(x^2)+1e-5) * w_f32)
  gemv:         y = bf16(W_deq_f32 @ x_f32)
  rope:         llama rotate-half, angles[i] = base^(-i/(d/2)), f32
  qk-norm (hy): per head, bf16(x * rsqrt(mean(x^2)+1e-5) * w) AFTER rope
  attention:    GQA h//group, scores/softmax/PV in f32, out bf16
  add/swiglu:   f32 -> bf16 (silu(x) = x*sigmoid(x))

Run: ironenv/bin/python tools/decode_export.py [--arch hy] [--out build/dec_hy] [--pos 100]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

_spec = importlib.util.spec_from_file_location(
    "w4uref", "/home/nzinfo/qwen/xnpu/IRON/iron/operators/w4gemvu/reference.py"
)
_w4uref = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_w4uref)
quantize_and_pack = _w4uref.quantize_and_pack

# name -> (row tensors in concat order, M, K) — must match W4U_SHAPES in Rust
MINICPM_SHAPES = {
    "qkv": ([("q_proj", 2048), ("k_proj", 256), ("v_proj", 256)], 2560, 2048),
    "o": ([("o_proj", 2048)], 2048, 2048),
    "gateup": ([("gate_proj", 6144), ("up_proj", 6144)], 12288, 2048),
    "down": ([("down_proj", 2048)], 2048, 6144),
}
HY_SHAPES = {
    "qkv": ([("q_proj", 2048), ("k_proj", 512), ("v_proj", 512)], 3072, 2048),
    "o": ([("o_proj", 2048)], 2048, 2048),
    "gateup": ([("gate_proj", 6144), ("up_proj", 6144)], 12288, 2048),
    "down": ([("down_proj", 2048)], 2048, 6144),
}

EPS = 1e-5
HEAD_DIM = 128
HIDDEN = 2048

ARCHS = {
    "minicpm": {
        "layers": 42, "heads": 16, "kv": 2,
        "rope_base": 5e6, "qk_norm": False,
        "shapes": MINICPM_SHAPES,
        "model": Path("/home/nzinfo/qwen/xnpu/models/minicpm5-2b/model.safetensors"),
    },
    "hy": {
        "layers": 32, "heads": 16, "kv": 4,
        # FLM hunyuan dynamic rope = static alpha rescale of the base
        "rope_base": 10000.0 * 1000.0 ** (128.0 / 126.0), "qk_norm": True,
        "shapes": HY_SHAPES,
        "model": Path("/home/nzinfo/.config/flm/models/Hy-MT2-1.8B-NPU2/model.q4nx"),
    },
}


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


def qk_rms(x, w):
    """Per-head RMS norm over head_dim (hy qk-norm, applied AFTER rope)."""
    xf = x.to(torch.float32)
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    y = xf * torch.rsqrt(ms + EPS) * w.to(torch.float32)
    return y.to(torch.bfloat16)


def rope(x, pos, base):
    """x: (heads, head_dim) bf16 -> rotate-half rope at position pos, bf16."""
    xf = x.to(torch.float32)
    j = torch.arange(HEAD_DIM // 2, dtype=torch.float32)
    inv = base ** (-j / (HEAD_DIM // 2))
    ang = pos * inv
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = xf[..., : HEAD_DIM // 2], xf[..., HEAD_DIM // 2:]
    out = torch.empty_like(xf)
    out[..., : HEAD_DIM // 2] = x1 * cos - x2 * sin
    out[..., HEAD_DIM // 2:] = x2 * cos + x1 * sin
    return out.to(torch.bfloat16)


def attention(q, k_cache, v_cache, pos, heads, kv):
    """One decode step. q: (heads,128) bf16 (roped [+qk-normed]); k/v_cache:
    (kv, S, 128) bf16. GQA: q head h reads kv head h // (heads//kv). f32."""
    group = heads // kv
    S = pos + 1
    out = torch.empty(heads, HEAD_DIM, dtype=torch.bfloat16)
    for h in range(heads):
        kvc = h // group
        k = k_cache[kvc, :S].to(torch.float32)  # (S,128)
        v = v_cache[kvc, :S].to(torch.float32)
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


def hy_weights(arch):
    """Weight source for hy: decode q4nx directly (same W the closed FLM
    engine executes — the M5c comparison is apples-to-apples by design)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from q4nx_import import Q4nx

    q4 = Q4nx(arch["model"])
    weights, layer_norms, qk_norms = {}, {}, {}
    for n in range(arch["layers"]):
        weights[n] = {
            p: q4.i8_matrix(f"model.layers.{n}.self_attn.{p}.weight")
            for p in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        weights[n].update(
            {p: q4.i8_matrix(f"model.layers.{n}.mlp.{p}.weight")
             for p in ("gate_proj", "up_proj", "down_proj")}
        )
        layer_norms[n] = (
            torch.from_numpy(q4.bf16(f"model.layers.{n}.input_layernorm.weight")).to(torch.bfloat16),
            torch.from_numpy(q4.bf16(f"model.layers.{n}.post_attention_layernorm.weight")).to(torch.bfloat16),
        )
        qk_norms[n] = (
            torch.from_numpy(q4.bf16(f"model.layers.{n}.self_attn.q_norm.weight")).to(torch.bfloat16),
            torch.from_numpy(q4.bf16(f"model.layers.{n}.self_attn.k_norm.weight")).to(torch.bfloat16),
        )
    final_w = torch.from_numpy(q4.bf16("model.norm.weight")).to(torch.bfloat16)
    return weights, layer_norms, qk_norms, final_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(ARCHS), default="minicpm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pos", type=int, default=100)
    ap.add_argument("--cache-seq", type=int, default=1024)
    args = ap.parse_args()
    arch = ARCHS[args.arch]
    nl, heads, kv = arch["layers"], arch["heads"], arch["kv"]
    base = arch["rope_base"]
    out = Path(args.out) if args.out else Path(
        f"/home/nzinfo/qwen/xnpu/build/{'dec_hy' if args.arch == 'hy' else 'dec'}")
    pos, S = args.pos, args.cache_seq
    out.mkdir(parents=True, exist_ok=True)

    x = lcg_vec(HIDDEN, mod=17, div=4.0)  # the decode-step input
    x0 = x.clone()  # the loop below consumes x — keep the step input
    kcache = torch.zeros(nl, kv, S, HEAD_DIM, dtype=torch.bfloat16)
    vcache = torch.zeros(nl, kv, S, HEAD_DIM, dtype=torch.bfloat16)
    for n in range(nl):
        for kvi in range(kv):
            kcache[n, kvi, : pos + 1] = lcg_vec((pos + 1) * HEAD_DIM,
                                                seed=1000 * n + 7 * kvi + 1).view(pos + 1, HEAD_DIM)
            vcache[n, kvi, : pos + 1] = lcg_vec((pos + 1) * HEAD_DIM,
                                                seed=1000 * n + 7 * kvi + 2).view(pos + 1, HEAD_DIM)

    # Layer weights: dequantized w4 for the reference, per layer/shape.
    norms = []
    if args.arch == "hy":
        weights, layer_norms, qk_norms, final_w = hy_weights(arch)
    else:
        from safetensors import safe_open

        weights, layer_norms, qk_norms = {}, None, None
        src = safe_open(str(arch["model"]), framework="pt")

        def get(name):
            return src.get_tensor(name)

        final_w = get("model.norm.weight")
    for n in range(nl):
        wdeq = {}
        for shape, (parts, m, k) in arch["shapes"].items():
            W = torch.cat(
                [torch.from_numpy(weights[n][p]).to(torch.float32)
                 if args.arch == "hy"
                 else get(f"model.layers.{n}.self_attn.{p}.weight"
                          if p in ("q_proj", "k_proj", "v_proj", "o_proj")
                          else f"model.layers.{n}.mlp.{p}.weight").to(torch.float32)
                 for p, _ in parts],
                dim=0,
            )
            assert W.shape == (m, k)
            _, w_dequant = quantize_and_pack(
                W.numpy(), group_size=32, m_input=4, cols=8
            )
            wdeq[shape] = w_dequant.to(torch.float32)
        if args.arch == "hy":
            n1, n2 = layer_norms[n]
        else:
            n1 = get(f"model.layers.{n}.input_layernorm.weight")
            n2 = get(f"model.layers.{n}.post_attention_layernorm.weight")
        norms.append((n1, n2))

        # --- the reference decode step (mirrors the Rust glue) ---
        xn = rms_norm(x, n1)
        qkv = (wdeq["qkv"] @ xn.to(torch.float32)).to(torch.bfloat16)
        kdim = kv * HEAD_DIM
        q, k, v = qkv[:2048], qkv[2048 : 2048 + kdim], qkv[2048 + kdim :]
        q = rope(q.view(heads, HEAD_DIM), pos, base)
        k = rope(k.view(kv, HEAD_DIM), pos, base)
        if arch["qk_norm"]:  # hy: per-head rms AFTER rope
            qw, kw = qk_norms[n]
            q = qk_rms(q, qw)
            k = qk_rms(k, kw)
        kcache[n, :, pos] = k
        vcache[n, :, pos] = v.view(kv, HEAD_DIM)
        attn = attention(q, kcache[n], vcache[n], pos, heads, kv)  # (16,128)
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

    hidden = rms_norm(x, final_w)

    norms_flat = torch.cat([t for pair in norms for t in pair] + [final_w])
    (out / "norms.bin").write_bytes(
        norms_flat.contiguous().view(torch.uint16).numpy().tobytes())
    if arch["qk_norm"]:
        qk_flat = torch.cat([torch.cat([qw, kw]) for qw, kw in qk_norms.values()])
        (out / "qknorms.bin").write_bytes(
            qk_flat.contiguous().view(torch.uint16).numpy().tobytes())
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
    meta = {"arch": args.arch, "pos": pos, "cache_seq": S, "layers": nl,
            "hidden": HIDDEN, "heads": heads, "kv_heads": kv,
            "head_dim": HEAD_DIM, "eps": EPS, "rope_base": base,
            "qk_norm": arch["qk_norm"]}
    with open(out / "meta.json", "w") as fp:
        json.dump(meta, fp, indent=1)
    print(f"done -> {out} (golden_hidden + {nl} layer goldens + caches)")


if __name__ == "__main__":
    main()
