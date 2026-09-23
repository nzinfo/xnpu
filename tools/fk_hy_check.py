#!/usr/bin/env python3
"""M5c 判别实验：16h_4kv flowkv op 用 E2E 真实 hy 数据（qk-norm 后的 q/k，
S=101，identity angles）经 XRT 单独跑，逐头 diff f32 标量 attention。

目的：把「内核在 hy 数据尺度下的数值问题」和「Rust 侧 BO 打包 bug」分开。
run-decode hy cpu PASS（1.0%）而 npu FAIL（12.1%），差异只在 attention 来源；
此脚本在 XRT 路径上复刻 Rust 的打包（identity angles + 预 rope+qk-norm 的
Q + interleave cache + S=101 hdr），如果这里也对不上 → 内核/数据尺度问题；
如果这里严丝合缝 → Rust raw-DRM 打包差异。

Run: ironenv/bin/python tools/fk_hy_check.py [--layer 0]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

IRON = "/home/nzinfo/qwen/xnpu/IRON"
sys.path.insert(0, IRON)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from iron.common import AIEContext
from iron.operators.flowkv_decode.op import AIEFlowKVDecode, pack_q_with_angles
from iron.operators.flowkv_decode.reference import (
    interleave_kv_cache,
    make_rope_angles_interleaved,
)

_spec = importlib.util.spec_from_file_location(
    "de", "/home/nzinfo/qwen/xnpu/tools/decode_export.py"
)
de = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(de)

_spec = importlib.util.spec_from_file_location(
    "w4uref", f"{IRON}/iron/operators/w4gemvu/reference.py"
)
w4uref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w4uref)

DEC = Path("/home/nzinfo/qwen/xnpu/build/dec_hy")
MODEL = Path("/home/nzinfo/.config/flm/models/Hy-MT2-1.8B-NPU2/model.q4nx")
POS = 100
CACHE_SEQ = 1024


def rd_bf16(p):
    return torch.from_numpy(
        np.frombuffer(Path(p).read_bytes(), dtype=np.uint16).copy()
    ).view(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()
    n = args.layer
    heads, kv, hd = 16, 4, 128
    base = 10000.0 * 1000.0 ** (128.0 / 126.0)

    from q4nx_import import Q4nx

    q4 = Q4nx(MODEL)
    x0 = rd_bf16(DEC / "x0.bin").to(torch.bfloat16)
    norms = rd_bf16(DEC / "norms.bin").to(torch.bfloat16).view(-1, 2048)
    qkn = rd_bf16(DEC / "qknorms.bin").to(torch.bfloat16).view(-1, 256)
    kcache = rd_bf16(DEC / "kcache.bin").to(torch.bfloat16).view(32, kv, CACHE_SEQ, hd)
    vcache = rd_bf16(DEC / "vcache.bin").to(torch.bfloat16).view(32, kv, CACHE_SEQ, hd)

    # Layer-n forward to the attention inputs (decode_export math verbatim).
    n1 = norms[n * 2]
    xn = de.rms_norm(x0 if n == 0 else rd_bf16(DEC / f"golden_L{n-1:02}.bin"), n1)
    parts = []
    for p, rows in (("q_proj", 2048), ("k_proj", 512), ("v_proj", 512)):
        W = torch.from_numpy(q4.i8_matrix(f"model.layers.{n}.self_attn.{p}.weight")).to(torch.float32)
        assert W.shape == (rows, 2048)
        parts.append(W)
    W = torch.cat(parts, dim=0)
    _, w_deq = w4uref.quantize_and_pack(W.numpy(), group_size=32, m_input=4, cols=8)
    w_deq = w_deq.to(torch.float32)
    qkv = (w_deq @ xn.to(torch.float32)).to(torch.bfloat16)
    kdim = kv * hd
    q, k, v = qkv[:2048], qkv[2048 : 2048 + kdim], qkv[2048 + kdim :]
    q = de.rope(q.view(heads, hd), POS, base)
    k = de.rope(k.view(kv, hd), POS, base)
    qw, kw = qkn[n, :hd], qkn[n, hd:]
    q = de.qk_rms(q, qw)
    k = de.qk_rms(k, kw)
    kc = kcache[n].clone()
    vc = vcache[n].clone()
    kc[:, POS] = k
    vc[:, POS] = v.view(kv, hd)

    o_ref = de.attention(q, kc, vc, POS, heads, kv)  # (16,128) bf16
    o_ref_f = o_ref.to(torch.float32)

    # bf16-compute simulation: round score storage, exp weights, and the
    # Y/F accumulation to bf16 the way the kernel's all-bf16 online softmax
    # does (chunked f32 mac, bf16 narrow at every tensor boundary).
    def attn_sim(q, kc, vc, pos, heads, kv, mode="plain"):
        out = torch.empty(heads, 128, dtype=torch.bfloat16)
        for h in range(heads):
            kvh = h // (heads // kv)
            if mode == "tail":
                # (B): kernel processes ceil(S/32)*32 rows; rows >= S are the
                # caller's zeros (K=0 -> score 0, V=0).
                pad = (-(pos + 1)) % 32
                K = torch.cat([kc[kvh, : pos + 1], torch.zeros(pad, 128)]).to(torch.float32)
                V = torch.cat([vc[kvh, : pos + 1], torch.zeros(pad, 128)]).to(torch.float32)
            else:
                K = kc[kvh, : pos + 1].to(torch.float32)
                V = vc[kvh, : pos + 1].to(torch.float32)
            s = (K @ q[h].to(torch.float32) / 128**0.5).to(torch.bfloat16).to(torch.float32)
            m = s.max()
            if mode == "exp2arg":
                # (A): the exp2 argument itself rounds to bf16 (test.py note).
                arg = ((s - m) * torch.log2(torch.tensor(2.718281828459045))).to(torch.bfloat16).to(torch.float32)
                w = torch.exp2(arg)
            else:
                w = torch.exp(s - m).to(torch.bfloat16).to(torch.float32)
            Y = torch.zeros(128)
            F = 0.0
            for t in range(0, K.shape[0], 32):  # chunked bf16 accumulation
                wc, Vc = w[t : t + 32], V[t : t + 32]
                Y = (Y + (wc[:, None] * Vc).to(torch.bfloat16).to(torch.float32).sum(0).to(torch.bfloat16).to(torch.float32)).to(torch.bfloat16).to(torch.float32)
                F = (F + wc.sum().to(torch.bfloat16).to(torch.float32)).to(torch.bfloat16).to(torch.float32)
            out[h] = (Y / F).to(torch.bfloat16)
        return out

    o_sim = attn_sim(q, kc, vc, POS, heads, kv)
    o_sim_f = o_sim.to(torch.float32)

    # EXACT kernel-numerics simulation — every rounding point as written in
    # aie_kernels/aie2p/flowkv.cc, chunk by chunk:
    #   s = bf16(f32 mac dot / sqrt(128)); chunk max over bf16 scores;
    #   C_c = bf16(exp2(bf16((m_old-m_new) * 1.4453125)));   # NOT log2(e)!
    #   f   = bf16(exp2(bf16((s - m_new) * 1.4453125)));     # arg bf16
    #   l   = f32 accumulation, but crosses to the value core as bf16;
    #   Y   = f32 accumulation (f32(f) * f32(V)), rescaled by f32(C_c);
    #   O   = bf16(Y * 1/f32(bf16(l))).
    def attn_exact(q, kc, vc, pos, heads, kv, log2e=1.4453125, vprod=False):
        bf = lambda t: t.to(torch.bfloat16).to(torch.float32)  # noqa: E731
        out = torch.empty(heads, 128, dtype=torch.bfloat16)
        CS = 32
        for h in range(heads):
            kvh = h // (heads // kv)
            K = kc[kvh, : pos + 1].to(torch.float32)
            V = vc[kvh, : pos + 1].to(torch.float32)
            m_old, l_old = -1e30, 0.0
            Y = torch.zeros(128)
            F_final = 1.0
            f_store = []
            for t0 in range(0, K.shape[0], CS):
                s = bf(K @ q[h].to(torch.float32) / 128**0.5)  # bf16 storage
                sc = s[t0 : t0 + CS]
                m_chunk = sc.max()
                m_new = max(m_chunk, m_old)
                c_c = bf(torch.exp2(torch.tensor(bf((m_old - m_new) * log2e))))
                l = c_c * l_old
                f = bf(torch.exp2(bf((sc - m_new) * log2e)))
                l = l + f.sum()
                f_store.append((f, c_c, bf(torch.tensor(l)), t0))
                m_old, l_old = m_new, l
            for f, c_c, l_bf, t0 in f_store:
                if vprod:
                    # value-core mul/add path carries ~bf16 grain per term:
                    # Y = bf16(Y*C_c); fv = bf16(f*V) each, f32 add
                    Y = bf(Y * c_c)
                    fv = bf(f[:, None] * V[t0 : t0 + f.shape[0]])
                    Y = Y + fv.sum(0)
                else:
                    Y = Y * c_c  # f32 rescale by bf16-quantized C_c
                    Y = Y + (f[:, None] * V[t0 : t0 + f.shape[0]]).sum(0)
                F_final = l_bf
            out[h] = (Y / F_final).to(torch.bfloat16)
        return out

    o_sim_x = attn_exact(q, kc, vc, POS, heads, kv).to(torch.float32)
    # variant with the true log2(e): isolates the 1.4453125 constant's share
    o_sim_x_true = attn_exact(q, kc, vc, POS, heads, kv, log2e=1.4426950408889634).to(
        torch.float32
    )
    # variant with bf16-grain value-core products (H3: aie::mul/add on the
    # f32 Y path round each f*V product and Y*C_c rescale to bf16)
    o_sim_x_vp = attn_exact(q, kc, vc, POS, heads, kv, vprod=True).to(torch.float32)

    # NPU run — the Rust packing mirrored exactly.
    angles = make_rope_angles_interleaved(
        torch.ones(hd // 2), torch.zeros(hd // 2)
    )  # identity: Q arrives pre-roped (+qk-normed)
    q_packed = pack_q_with_angles(q, angles, heads // kv, kv, seq_len_cur=POS + 1)
    kv_inter = interleave_kv_cache(kc, vc)

    ctx = AIEContext()
    op = AIEFlowKVDecode(
        num_heads=heads,
        num_kv_heads=kv,
        head_dim=hd,
        seq_len=CACHE_SEQ,
        chunk_size=32,
        num_cols=4,
        context=ctx,
    )
    ctx.compile_all()
    ctx.prepare_runtime()
    for _ in range(2):
        op.run_runlist()  # warmup configures the persistent workers
    op.write_buffer("output", np.zeros(op.buffers["output"], dtype=np.uint8))
    from iron.common.utils import torch_to_numpy

    op.write_buffer("kv_cache", torch_to_numpy(kv_inter))
    op.write_buffer("queries", torch_to_numpy(q_packed))
    for _ in range(3):
        op.run_runlist()
        out = op.read_buffer("output", (heads * hd,), dtype=np.dtype(bfloat16))
        o_npu = torch.from_numpy(out.astype(np.float32)).view(heads, hd)

        d = (o_npu - o_ref_f).abs()
        rms = d.pow(2).mean(dim=1).sqrt()
        # per-head lstsq scale o_npu ~ a * o_ref: a<1 => shrinkage (tail bug)
        a_scale = [
            ((o_npu[h] @ o_ref_f[h]) / (o_ref_f[h] @ o_ref_f[h] + 1e-12)).item()
            for h in range(heads)
        ]
        rms_to = lambda x: x.abs().pow(2).mean().sqrt().item()  # noqa: E731
        print(
            f"layer {n}: npu-vs-ref rms {rms_to(o_npu - o_ref_f):.5f} | "
            f"fit vs npu: plain {rms_to(o_sim_f - o_npu):.5f} "
            f"EXACT {rms_to(o_sim_x - o_npu):.5f} "
            f"EXACT-trueLog2e {rms_to(o_sim_x_true - o_npu):.5f} "
            f"EXACT-vprod {rms_to(o_sim_x_vp - o_npu):.5f}"
        )
        print(
            f"  EXACT vs ref {rms_to(o_sim_x - o_ref_f):.5f} "
            f"(trueLog2e vs ref {rms_to(o_sim_x_true - o_ref_f):.5f})"
        )
        print("  per-head lstsq scale (1.0 = no shrinkage):")
        for h0 in range(0, heads, 8):
            print(
                "   "
                + " ".join(f"h{h}:{a_scale[h]:.3f}" for h in range(h0, min(h0 + 8, heads)))
            )
        # score scale context: max |q·k|/sqrt(128) per head over live rows
        scale = []
        for h in range(heads):
            kk = kc[h // (heads // kv), : POS + 1].to(torch.float32)
            s = (kk @ q[h].to(torch.float32)) / (hd**0.5)
            scale.append(s.abs().max().item())
        print(
            f"  (ref rms {o_ref_f.pow(2).mean().sqrt():.3f}, max |diff| {d.max():.4f})"
        )
        for h in range(heads):
            print(
                f"  head {h:2d}: rms {rms[h]:.5f} max {d[h].max():.4f} "
                f"maxscore {scale[h]:.1f}"
            )


if __name__ == "__main__":
    main()
