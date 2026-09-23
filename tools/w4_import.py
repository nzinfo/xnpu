#!/usr/bin/env python3
"""Offline w4 weight importer for the xnpu engine (M3a).

Reads MiniCPM5-2B safetensors, quantizes each decode projection to the
w4gemv2 signed-int4 ABI (per-group-32 symmetric, scale = amax/7 bf16,
nibbles two's-complement low-first, tile column-major DDR layout — the
same packer the IRON test uses) and writes one .bin per layer per fused
shape plus a meta.json and a golden spot-check file.

Shapes per layer (all 8 AIE columns; tsi per the fixture memory budget):
  qkv     2560x2048  cat(q 2048, k 256, v 256)   tsi=16   fixture w4gemv2_2560x2048_16tsi_320tso_8col_g32
  o       2048x2048  o_proj                      tsi=16   fixture w4gemv2_2048x2048_16tsi_256tso_8col_g32
  gate_up 12288x2048 cat(gate 6144, up 6144)     tsi=16   fixture w4gemv2_12288x2048_16tsi_1536tso_8col_g32
  down    2048x6144  down_proj                   tsi=4    fixture w4gemv2_2048x6144_4tsi_256tso_8col_g32

Golden spot check: a deterministic bf16 vector x[k] = (k%7)-3 per shape
and reference rows (f32 dot of the DEQUANTIZED weights against x, rounded
to bf16) so the Rust harness can verify real-weight outputs within bf16
tolerance (real scales are not dyadic, so bit-exactness is not expected).

Run:  ironenv/bin/python tools/w4_import.py --layers 0 1   # subset for bring-up
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Load the packer straight from its file: importing through the iron
# package pulls in iron.common -> mlir_aie host runtime, which opens the
# NPU device on import (fails under a plain user shell).
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "w4ref", "/home/nzinfo/qwen/xnpu/IRON/iron/operators/w4gemv2/reference.py"
)
_w4ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_w4ref)
quantize_and_pack = _w4ref.quantize_and_pack

MODEL = Path("/home/nzinfo/qwen/xnpu/models/minicpm5-2b/model.safetensors")

# name -> (row tensors in concat order with expected row counts, M, K, tsi)
SHAPES = {
    "qkv": ([("q_proj", 2048), ("k_proj", 256), ("v_proj", 256)], 2560, 2048, 16),
    "o": ([("o_proj", 2048)], 2048, 2048, 16),
    "gateup": ([("gate_proj", 6144), ("up_proj", 6144)], 12288, 2048, 16),
    "down": ([("down_proj", 2048)], 2048, 6144, 4),
}


def spot_rows(m, k, w_dequant, x_bf16):
    """Reference outputs for a few rows: f32 dot of dequant W against the
    deterministic x, one bf16 rounding — same tolerance tier as the IRON
    test (bf16 accumulation-order noise only)."""
    rows = sorted({0, 1, m // 2, m - 2, m - 1})
    ref = (w_dequant[rows].to(torch.float32) @ x_bf16.to(torch.float32)).to(
        torch.bfloat16
    )
    return rows, ref.view(torch.uint16).numpy()


def write_golden(out, shape, k, rows, ref_bits, x_bits):
    """One raw file per shape (Rust-friendly, no zip/npy parsing):
    u32 nrows, u32 K, rows u32[nrows], x bits u16[K], ref bits u16[nrows]."""
    import struct

    buf = bytearray()
    buf += struct.pack("<II", len(rows), k)
    buf += np.asarray(rows, dtype="<u4").tobytes()
    buf += np.ascontiguousarray(x_bits, dtype="<u2").tobytes()
    buf += np.ascontiguousarray(ref_bits, dtype="<u2").tobytes()
    (out / f"golden_{shape}.bin").write_bytes(bytes(buf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/nzinfo/qwen/xnpu/build/w4")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="layer indices to import (default: all 42)")
    args = ap.parse_args()
    layers = args.layers if args.layers is not None else list(range(42))

    from safetensors import safe_open

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"group_size": 32, "cols": 8, "layers": layers, "shapes": {}}

    # Deterministic activation per K (bf16 small integers, exact).
    xs = {2048: (torch.arange(2048) % 7 - 3).to(torch.bfloat16),
          6144: (torch.arange(6144) % 7 - 3).to(torch.bfloat16)}

    with safe_open(str(MODEL), framework="pt") as f:
        for n in layers:
            tensors = {
                name: f.get_tensor(f"model.layers.{n}.self_attn.{name}.weight")
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            }
            tensors.update(
                {name: f.get_tensor(f"model.layers.{n}.mlp.{name}.weight")
                 for name in ("gate_proj", "up_proj", "down_proj")}
            )
            for shape, (parts, m, k, tsi) in SHAPES.items():
                for p, rows in parts:
                    assert tensors[p].shape == (rows, k), (
                        f"layer {n} {shape}: {p} is {tuple(tensors[p].shape)}, "
                        f"expected ({rows}, {k})"
                    )
                W = torch.cat([tensors[p] for p, _ in parts], dim=0)
                assert W.shape == (m, k)
                # bf16 -> f32 is lossless; the packer quantizes from f32.
                packed, w_dequant = quantize_and_pack(
                    W.to(torch.float32).numpy(), group_size=32, m_input=tsi, cols=8
                )
                path = out / f"layer{n:02d}_{shape}.bin"
                packed.tofile(path)
                if n == layers[0]:
                    meta["shapes"][shape] = {
                        "M": m, "K": k, "tsi": tsi, "bytes": len(packed),
                    }
                    rows, ref = spot_rows(m, k, w_dequant, xs[k])
                    write_golden(out, shape, k, rows, ref,
                                 xs[k].view(torch.uint16).numpy())
            print(f"layer {n}: 4 shapes written", flush=True)

    with open(out / "meta.json", "w") as fp:
        json.dump(meta, fp, indent=1)
    total = sum(s["bytes"] for s in meta["shapes"].values()) * len(layers)
    print(f"done: {len(layers)} layers, {total/1e6:.1f} MB total, "
          f"golden spot checks for layer {layers[0]}")


if __name__ == "__main__":
    main()
