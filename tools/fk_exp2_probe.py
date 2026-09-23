#!/usr/bin/env python3
"""M5c 判别实验 2c：score 幅值扫描——real-data 失配的最后嫌疑轴。

前序（2b，已证）：shuf-deep 贴合（online rescale 无辜）；dense ±1 数据上
模型大体贴合（npu-vs-ref 0.0014 ≈ sim-vs-ref 0.0014）。但 ±1 数据的 score
σ≈0.3，而 hy 真实数据 qk-norm 后 score 3.4–8.7（bf16 ULP 从 0.004 涨到
0.03–0.06）。本实验把 LCG K 幅值 ×1/×9/×15 推到真实量级：
  若 npu-vs-sim 在 ×9/×15 处爆开而 sim-vs-ref 不动 → score 幅值触发，
  内核在 |s|>4 处有超出 .cc 文档算术的噪声（bf16 指数段加倍？）；
  若同步增长 → 模型仍贴合，hy real 0.0028 是另有所在。
"""

import sys

import numpy as np
import torch
from ml_dtypes import bfloat16

IRON = "/home/nzinfo/qwen/xnpu/IRON"
sys.path.insert(0, IRON)

from iron.common import AIEContext
from iron.common.utils import torch_to_numpy
from iron.operators.flowkv_decode.op import AIEFlowKVDecode, pack_q_with_angles
from iron.operators.flowkv_decode.reference import (
    interleave_kv_cache,
    make_rope_angles_interleaved,
)

POS = 100
HEADS, KV, HD = 16, 4, 128
S = POS + 1
CS = 32
INV_SQRT_D = 0.08838834764831845
LOG2E_K = 1.4453125


def lcg(n, seed, scale=1.0):
    s = seed
    out = torch.zeros(n)
    for i in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out[i] = ((s >> 33) / (1 << 31) - 1.0) * scale
    return out


def bf(t):
    return t.to(torch.bfloat16).to(torch.float64)


def kernel_sim(q, kc, vc):
    """Chunk-exact kernel arithmetic sim (scores f32 dot → bf16; bf16 exp2
    args on the 1.4453125 pseudo-log2e; bf16 f/C_c; f32 l & Y; bf16 l cross;
    bf16 O). Mirrors aie_kernels/aie2p/flowkv.cc exactly."""
    out = torch.zeros(HEADS, HD)
    for h in range(HEADS):
        kvh = h // (HEADS // KV)
        K = kc[kvh, :S].to(torch.float64)
        V = vc[kvh, :S].to(torch.float64)
        s_all = bf((K @ q[h].to(torch.float64) / 128**0.5).to(torch.float32))
        m_old, l_old = -1e30, 0.0
        chunks = []
        for t0 in range(0, S, CS):
            sc = s_all[t0 : t0 + CS]
            m_new = max(sc.max().item(), m_old)
            c_c = bf(torch.exp2(torch.tensor((m_old - m_new) * LOG2E_K)))
            l = c_c * l_old
            f = bf(torch.exp2(bf((sc - m_new) * LOG2E_K)))
            l = l + f.sum()
            chunks.append((f, c_c, bf(torch.tensor(l)), t0))
            m_old, l_old = m_new, l
        Y = torch.zeros(HD)
        for f, c_c, l_bf, t0 in chunks:
            Y = Y * c_c
            Y = Y + (f[:, None] * V[t0 : t0 + f.shape[0]]).sum(0)
        out[h] = bf((Y / chunks[-1][2]).to(torch.float32))
    return out


def exact_ref(q, kc, vc):
    out = torch.zeros(HEADS, HD)
    for h in range(HEADS):
        kvh = h // (HEADS // KV)
        K = kc[kvh, :S].to(torch.float64)
        V = vc[kvh, :S].to(torch.float64)
        s = K @ q[h].to(torch.float64) / 128**0.5
        p = torch.softmax(s, dim=0)
        out[h] = (p[:, None] * V).sum(0)
    return out


def main():
    q = torch.zeros(HEADS, HD, dtype=torch.bfloat16)
    kc = torch.zeros(KV, 1024, HD, dtype=torch.bfloat16)
    vc = torch.zeros(KV, 1024, HD, dtype=torch.bfloat16)

    # deep sweep values, then a deterministic shuffle for kv1
    vals = [-0.16 * (t % 101) for t in range(S)]
    rng = np.random.default_rng(7)
    shuf = vals[:]
    rng.shuffle(shuf)

    for g, mode in enumerate(["mono-deep", "dense-x1", "dense-x9", "dense-x15"]):
        h0 = g * (HEADS // KV)
        if mode == "mono-deep":
            for h in range(h0, h0 + HEADS // KV):
                q[h, 0] = 8.0
            for t, s_t in enumerate(vals):
                k0 = s_t / (8.0 * INV_SQRT_D)
                kc[g, t, 0] = torch.tensor(k0, dtype=torch.bfloat16)
            for t in range(S):
                vc[g, t, t] = 1.0  # O directly exposes the weights
        else:
            scale = float(mode.split("-x")[1])
            qd = lcg(HEADS // KV * HD, 0x1234 + g)
            for j, v in enumerate(qd):
                q[h0, j % HD] = torch.tensor(v, dtype=torch.bfloat16).item()
            # every head in the group gets its own dense q
            for hh in range(h0 + 1, h0 + HEADS // KV):
                qd = lcg(HD, 0x9000 + hh)
                for j, v in enumerate(qd):
                    q[hh, j] = torch.tensor(v, dtype=torch.bfloat16).item()
            kd = lcg(S * HD, 0x5000 + g, scale=scale)
            for t in range(S):
                for j, v in enumerate(kd[t * HD : (t + 1) * HD]):
                    kc[g, t, j] = torch.tensor(v, dtype=torch.bfloat16).item()
            vd = lcg(S * HD, 0xA000 + g)
            for t in range(S):
                for j, v in enumerate(vd[t * HD : (t + 1) * HD]):
                    vc[g, t, j] = torch.tensor(v, dtype=torch.bfloat16).item()

    o_sim = kernel_sim(q, kc, vc)
    o_ref = exact_ref(q, kc, vc)

    angles = make_rope_angles_interleaved(torch.ones(HD // 2), torch.zeros(HD // 2))
    q_packed = pack_q_with_angles(q, angles, HEADS // KV, KV, seq_len_cur=S)
    kv_inter = interleave_kv_cache(kc, vc)

    ctx = AIEContext()
    op = AIEFlowKVDecode(
        num_heads=HEADS, num_kv_heads=KV, head_dim=HD,
        seq_len=1024, chunk_size=CS, num_cols=4, context=ctx,
    )
    ctx.compile_all()
    ctx.prepare_runtime()
    for _ in range(2):
        op.run_runlist()
    op.write_buffer("output", np.zeros(op.buffers["output"], dtype=np.uint8))
    op.write_buffer("kv_cache", torch_to_numpy(kv_inter))
    op.write_buffer("queries", torch_to_numpy(q_packed))
    op.run_runlist()
    out = op.read_buffer("output", (HEADS * HD,), dtype=np.dtype(bfloat16))
    o_npu = torch.from_numpy(out.astype(np.float32)).view(HEADS, HD).to(torch.float64)

    rms = lambda a: float(a.abs().pow(2).mean().sqrt())  # noqa: E731
    print(f"{'mode':>12} {'npu-vs-ref':>11} {'npu-vs-sim':>11} {'sim-vs-ref':>11}  per-head lstsq scale vs ref")
    for g, mode in enumerate(["mono-deep", "dense-x1", "dense-x9", "dense-x15"]):
        h0 = g * (HEADS // KV)
        hs = slice(h0, h0 + HEADS // KV)
        sc = [
            float((o_npu[h].double() @ o_ref[h].double())
                  / (o_ref[h].double() @ o_ref[h].double() + 1e-12))
            for h in range(h0, h0 + HEADS // KV)
        ]
        print(
            f"{mode:>12} {rms(o_npu[hs] - o_ref[hs]):>11.5f} "
            f"{rms(o_npu[hs] - o_sim[hs]):>11.5f} {rms(o_sim[hs] - o_ref[hs]):>11.5f}  "
            + " ".join(f"{s:.3f}" for s in sc)
        )


if __name__ == "__main__":
    main()
