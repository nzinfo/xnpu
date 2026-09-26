#!/usr/bin/env python3
"""Q4NX weight importer for the xnpu engine (M5c, hy-mt2 1.8B).

Source is the FLM model directory (~/.config/flm/models/Hy-MT2-1.8B-NPU2)
— model.q4nx is a self-contained local weight source, and using FLM's own
quantized weights keeps the per-op comparison apples-to-apples (same W the
closed engine executes).

Q4NX layout (reverse-engineered, see notes/perf-lab.md P2):
  file     = u64 LE manifest_len + JSON manifest + blob (safetensors-style)
  I8 tensor = ceil(M/32)*ceil(K/256) tiles, 4608 B each, t = tr*ntc + tc
  tile     = 32 rows x 256 cols:
    [0,512)    256 bf16 scales, flat idx = group*32 + row  ([g][lr])
    [512,4608) int4 nibbles: byte = 512 + col*16 + row//2
               (even row = low nibble, odd = high), signed two's complement
  w = q * d with d = max(group)/-8 (llama.cpp Q4_0 style, no zero point)

The manifest carries no logical dims — the projection table below supplies
M and K per tensor (validated: every I8 shape[0] matches ceil(M/32)*ceil(K/256)).

Output contract matches w4_import.py exactly (layer{n:02d}_{shape}.bin in
the w4gemvu v2 universal-slot layout, golden_{...}.bin spot checks,
meta.json) so the Rust loader needs no format change — only the qkv shape
2560 -> 3072 is new (16Q/4KV GQA instead of 16/2).

hy-mt2 decode shapes per layer:
  qkv     3072x2048  cat(q 2048, k 512, v 512)
  o       2048x2048  o_proj
  gate_up 12288x2048 cat(gate 6144, up 6144)
  down    2048x6144  down_proj

bf16.safetensors next to the .bins carries the norms (input/post/q_norm/
k_norm/final) for decode_export; --with-embed adds the 495 MB tied
embedding table. Tie verification: decoded lm_head vs the bf16 embed table
is a full-tensor check of the decoder (30208 tiles, padding rows included).

Run: ironenv/bin/python tools/q4nx_import.py --layers 0 1   # bring-up
     ironenv/bin/python tools/q4nx_import.py                # all 32
"""

import argparse
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

_SPEC_V2 = importlib.util.spec_from_file_location(
    "w4uref", "/home/nzinfo/qwen/xnpu/IRON/iron/operators/w4gemvu/reference.py"
)
_w4uref = importlib.util.module_from_spec(_SPEC_V2)
assert _SPEC_V2.loader is not None
_SPEC_V2.loader.exec_module(_w4uref)
quantize_and_pack = _w4uref.quantize_and_pack

MODEL_DIR = Path("/home/nzinfo/.config/flm/models/Hy-MT2-1.8B-NPU2")

# logical dims per projection tensor: manifest-name suffix -> (M, K); hy-mt2
PROJ = {
    "q_proj": (2048, 2048), "k_proj": (512, 2048), "v_proj": (512, 2048),
    "o_proj": (2048, 2048),
    "gate_proj": (6144, 2048), "up_proj": (6144, 2048),
    "down_proj": (2048, 6144),
    "lm_head": (120818, 2048),
}

# fused shapes: (row tensors in concat order, M, K) — must match Rust W4U_SHAPES
SHAPES = {
    "qkv": ([("q_proj", 2048), ("k_proj", 512), ("v_proj", 512)], 3072, 2048),
    "o": ([("o_proj", 2048)], 2048, 2048),
    "gateup": ([("gate_proj", 6144), ("up_proj", 6144)], 12288, 2048),
    "down": ([("down_proj", 2048)], 2048, 6144),
}
N_LAYERS = 32
SLOT_V2 = 13840  # v2 packed size = 8 * (M/32) * SLOT_V2 per shape


class Q4nx:
    """mmap-backed reader for model.q4nx (manifest + blob)."""

    def __init__(self, path):
        self.f = np.memmap(path, dtype=np.uint8, mode="r")
        n = int(struct.unpack("<Q", self.f[:8].tobytes())[0])
        self.man = json.loads(bytes(self.f[8 : 8 + n]))
        self.blob = 8 + n

    def bf16(self, name):
        """Raw BF16 tensor -> float32 (norms, embed)."""
        t = self.man[name]
        off, end = t["data_offsets"]
        raw = self.f[self.blob + off : self.blob + end]
        u16 = raw.view(np.uint16).astype(np.uint32)
        return (u16 << 16).view(np.float32).reshape(t["shape"])

    def i8_matrix(self, name, m=None, k=None):
        """Decode an I8 tensor to float32 [M, K] (row padding dropped).

        Logical dims come from PROJ unless given (lm_head embed tie check
        passes them explicitly).
        """
        if m is None or k is None:
            suffix = name.split(".")[-2]  # ...self_attn.q_proj.weight -> q_proj
            m, k = PROJ[suffix]
        t = self.man[name]
        off, end = t["data_offsets"]
        n_tiles = t["shape"][0]
        raw = self.f[self.blob + off : self.blob + end].reshape(n_tiles, 4608)

        ntr, ntc = (m + 31) // 32, (k + 255) // 256
        assert ntr * ntc == n_tiles, (
            f"{name}: {n_tiles} tiles vs {ntr}x{ntc} from ({m},{k})"
        )

        # scales: 256 bf16, flat idx = g*32 + lr -> per (lr, col): S[col//32, lr]
        sc = raw[:, :512].view(np.uint16).astype(np.uint32)
        sc = (sc << 16).view(np.float32).reshape(n_tiles, 8, 32)  # (T, g, lr)
        scm = np.repeat(sc.transpose(0, 2, 1), 32, axis=2)  # (T, lr, col)

        # nibbles: byte = col*16 + lr//2; even lr = low nibble, odd = high
        nb = raw[:, 512:].reshape(n_tiles, 256, 16)  # (T, col, byte)
        q = np.empty((n_tiles, 256, 32), dtype=np.int8)  # (T, col, lr)
        # lr = 2i -> byte i low nibble; lr = 2i+1 -> byte i high nibble
        q[:, :, 0::2] = (nb & 0xF).astype(np.int8)
        q[:, :, 1::2] = (nb >> 4).astype(np.int8)
        q[q >= 8] -= 16
        Wt = q.transpose(0, 2, 1).astype(np.float32) * scm  # (T, lr, col)

        # tiles t = tr*ntc + tc -> [ntr*32, ntc*256]
        W = Wt.reshape(ntr, ntc, 32, 256).transpose(0, 2, 1, 3)
        return W.reshape(ntr * 32, ntc * 256)[:m, :k]


def spot_rows(m, k, w_dequant, x_bf16):
    """Reference outputs for a few rows: f32 dot of dequant W against the
    deterministic x, one bf16 rounding — same tolerance tier as w4_import."""
    rows = sorted({0, 1, m // 2, m - 2, m - 1})
    ref = (w_dequant[rows].to(torch.float32) @ x_bf16.to(torch.float32)).to(
        torch.bfloat16
    )
    return rows, ref.view(torch.uint16).numpy()


def write_golden(out, shape, k, rows, ref_bits, x_bits):
    """One raw file per shape (Rust-friendly, no zip/npy parsing):
    u32 nrows, u32 K, rows u32[nrows], x bits u16[K], ref bits u16[nrows]."""
    buf = bytearray()
    buf += struct.pack("<II", len(rows), k)
    buf += np.asarray(rows, dtype="<u4").tobytes()
    buf += np.ascontiguousarray(x_bits, dtype="<u2").tobytes()
    buf += np.ascontiguousarray(ref_bits, dtype="<u2").tobytes()
    (out / f"golden_{shape}.bin").write_bytes(bytes(buf))


def verify_tie(q4, verbose=True):
    """Full-decoder check: lm_head (I8, 30208 tiles incl. 14 pad rows)
    must equal the tied bf16 embed table to int4 quantization error."""
    emb = q4.bf16("model.embed_tokens.weight")  # (120818, 2048) f32
    head = q4.i8_matrix("lm_head.weight")
    assert head.shape == emb.shape, f"{head.shape} vs {emb.shape}"
    n = emb.shape[0]
    num = np.mean((head[:n] - emb) ** 2)
    den = np.mean(emb**2)
    rel = float(np.sqrt(num / den))
    if verbose:
        print(f"tie check lm_head vs embed: rel_rms={rel:.4f} "
              f"({'OK' if rel < 0.2 else 'FAIL'})")
    return rel


def export_lmhead(q4, out, decdir):
    """hy lm_head (tied embed, vocab 120818 -> padded 120832) as w4gemvu v2
    packed weights + golden logits for the decode golden step.

    Same treatment as every layer weight: FLM int4 -> f32 -> OUR amax/7
    int4 ABI via the unchanged IRON packer (zero rows pad quantize to
    q=0/scale=0). Golden x = decdir/golden_hidden.bin (the reference final
    hidden decode_export emits) — ref = dequant(W) @ x, f32 dot, one bf16
    rounding (spot_rows convention).

    File: lmhead.bin (8*(120832/32)*13840 = 418,078,720 B = 399 MiB);
    golden_lmhead.bin = u32 n, u32 k, x bits u16[k], ref bits u16[n].
    """
    head = q4.i8_matrix("lm_head.weight")  # (120818, 2048) f32
    m_padded = 120832
    assert head.shape == (120818, 2048)
    W = np.zeros((m_padded, 2048), dtype=np.float32)
    W[:120818] = head
    del head
    packed, w_dequant = quantize_and_pack(
        np.ascontiguousarray(W), group_size=32, m_input=4, cols=8
    )
    assert len(packed) == 8 * (m_padded // 32) * SLOT_V2
    packed_bytes = len(packed)
    (out / "lmhead.bin").write_bytes(packed.tobytes())
    del packed

    x_bits = np.fromfile(decdir / "golden_hidden.bin", dtype="<u2")
    assert x_bits.size == 2048, f"golden_hidden: {x_bits.size}"
    x = torch.from_numpy((x_bits.astype(np.uint32) << 16).view(np.float32))
    ref = (w_dequant.to(torch.float32) @ x).to(torch.bfloat16)
    buf = bytearray()
    buf += struct.pack("<II", m_padded, 2048)
    buf += x_bits.astype("<u2").tobytes()
    buf += ref.view(torch.uint16).numpy().astype("<u2").tobytes()
    (out / "golden_lmhead.bin").write_bytes(bytes(buf))
    print(f"lm_head: packed {packed_bytes} B -> lmhead.bin + golden_lmhead.bin "
          f"(x = golden_hidden)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/nzinfo/qwen/xnpu/build/w4u_hy")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="layer indices to import (default: all 32)")
    ap.add_argument("--with-embed", action="store_true",
                    help="also export the 495 MB bf16 embedding table")
    ap.add_argument("--lmhead", action="store_true",
                    help="also export lm_head (hy): lmhead.bin (399 MiB v2) "
                         "+ golden_lmhead.bin for the decode golden step")
    ap.add_argument("--skip-tie-check", action="store_true")
    args = ap.parse_args()

    q4 = Q4nx(MODEL_DIR / "model.q4nx")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = args.layers if args.layers is not None else list(range(N_LAYERS))

    if not args.skip_tie_check:
        verify_tie(q4)

    if args.lmhead:
        export_lmhead(q4, out, Path("/home/nzinfo/qwen/xnpu/build/dec_hy"))

    meta = {
        "group_size": 32, "cols": 8, "layout": "v2", "layers": layers,
        "model": "hy-mt2-1.8b", "shapes": {},
        # architecture facts decode_export needs (from FLM config.json)
        "arch": {"n_layers": 32, "hidden": 2048, "inter": 6144,
                 "n_heads": 16, "n_kv_heads": 4, "head_dim": 128,
                 "qk_norm": True, "rope_base": 10000.0,
                 "tie_word_embeddings": True, "vocab": 120818},
    }

    xs = {2048: (torch.arange(2048) % 7 - 3).to(torch.bfloat16),
          6144: (torch.arange(6144) % 7 - 3).to(torch.bfloat16)}

    for n in layers:
        tensors = {
            p: q4.i8_matrix(f"model.layers.{n}.self_attn.{p}.weight")
            for p in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        tensors.update(
            {p: q4.i8_matrix(f"model.layers.{n}.mlp.{p}.weight")
             for p in ("gate_proj", "up_proj", "down_proj")}
        )
        for shape, (parts, m, k) in SHAPES.items():
            for p, rows in parts:
                got = tensors[p].shape
                assert got == (rows, k), f"L{n} {shape}/{p}: {got} != {(rows, k)}"
            W = np.concatenate([tensors[p] for p, _ in parts], axis=0)
            assert W.shape == (m, k)
            # f32 in, re-quantized to OUR w4 ABI (amax/7): we inherit FLM's
            # int4 weights as ground truth — the comparison target IS FLM.
            packed, w_dequant = quantize_and_pack(
                np.ascontiguousarray(W), group_size=32, m_input=4, cols=8
            )
            assert len(packed) == 8 * (m // 32) * SLOT_V2, (
                f"layer {n} {shape}: v2 packed is {len(packed)} bytes"
            )
            packed.tofile(out / f"layer{n:02d}_{shape}.bin")
            rows, ref = spot_rows(m, k, w_dequant, xs[k])
            write_golden(out, f"L{n:02d}_{shape}", k, rows, ref,
                         xs[k].view(torch.uint16).numpy())
            if n == layers[0]:
                meta["shapes"][shape] = {"M": m, "K": k, "bytes": len(packed)}
                write_golden(out, shape, k, rows, ref,
                             xs[k].view(torch.uint16).numpy())
        print(f"layer {n}: 4 shapes written", flush=True)

    # norms (and optionally embed) for decode_export, as bf16 safetensors
    def to_bf16_t(a):
        return torch.from_numpy(a).to(torch.bfloat16)

    sd = {}
    for n in layers:
        for src, dst in (
            (f"model.layers.{n}.input_layernorm.weight",
             f"model.layers.{n}.input_layernorm.weight"),
            (f"model.layers.{n}.post_attention_layernorm.weight",
             f"model.layers.{n}.post_attention_layernorm.weight"),
            (f"model.layers.{n}.self_attn.q_norm.weight",
             f"model.layers.{n}.self_attn.q_norm.weight"),
            (f"model.layers.{n}.self_attn.k_norm.weight",
             f"model.layers.{n}.self_attn.k_norm.weight"),
        ):
            sd[dst] = to_bf16_t(q4.bf16(src))
    sd["model.norm.weight"] = to_bf16_t(q4.bf16("model.norm.weight"))
    if args.with_embed:
        sd["model.embed_tokens.weight"] = to_bf16_t(
            q4.bf16("model.embed_tokens.weight"))
    save_file(sd, out / "bf16.safetensors")

    with open(out / "meta.json", "w") as fp:
        json.dump(meta, fp, indent=1)
    total = sum(s["bytes"] for s in meta["shapes"].values()) * len(layers)
    print(f"done: {len(layers)} layers, {total/1e6:.1f} MB total, "
          f"golden spot checks for layer {layers[0]}")


if __name__ == "__main__":
    main()
