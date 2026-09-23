#!/usr/bin/env python3
"""M5c 判别实验 2e：per-position 权重偏差的位置性 vs 取值性。

2d 发现（V one-hot 直接暴露 p_t）：coarse 扫描里除 max 外所有位置的
f 隐含值 ~+4%（Δarg≈+0.058），l 虚高 3.2%——远超 bf16 舍入，向量 rms
0.0005 的旧指标完全掩盖了它。两种解释：
  (a) 取值性：exp2 intrinsic 的按值误差（同一 arg 恒同误差）
  (b) 位置性：score/FIFO/累加路径按位置的误差（同一值不同位置不同误差）

判别：k 行成对复制（s, s, s', s', ...）→ 同 arg 两个位置。对内差异
>bf16 粒度 → (b)；完全一致 → (a)。附带检查：组内 4 个 Q 头输出应逐位
相同（q 相同、KV 相同）——不同则 value core 有按头问题。
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


def bf(t):
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    return t.to(torch.bfloat16).to(torch.float64)


def main():
    q = torch.zeros(HEADS, HD, dtype=torch.bfloat16)
    kc = torch.zeros(KV, 1024, HD, dtype=torch.bfloat16)
    vc = torch.zeros(KV, 1024, HD, dtype=torch.bfloat16)

    # group 0/1: paired scores, step 0.64/0.16 -> 50/50 pairs over 101 rows
    # group 2: control mono sweep 0.16; group 3: reversed (max LAST: online
    #          rescale active on every chunk) with step 0.16
    def fill_pairs(g, step):
        h0 = g * (HEADS // KV)
        for h in range(h0, h0 + HEADS // KV):
            q[h, 0] = 8.0
        for t in range(S):
            s_t = -step * (t // 2)
            kc[g, t, 0] = torch.tensor(s_t / (8.0 * INV_SQRT_D), dtype=torch.bfloat16)
            vc[g, t, t] = 1.0

    fill_pairs(0, 0.64)
    fill_pairs(1, 0.16)
    g, h0 = 2, 2 * (HEADS // KV)
    for h in range(h0, h0 + HEADS // KV):
        q[h, 0] = 8.0
    for t in range(S):
        kc[2, t, 0] = torch.tensor(-(0.16 * t) / (8.0 * INV_SQRT_D), dtype=torch.bfloat16)
        vc[2, t, t] = 1.0
    h0 = 3 * (HEADS // KV)
    for h in range(h0, h0 + HEADS // KV):
        q[h, 0] = 8.0
    for t in range(S):
        kc[3, t, 0] = torch.tensor(-(0.16 * (S - 1 - t)) / (8.0 * INV_SQRT_D), dtype=torch.bfloat16)
        vc[3, t, t] = 1.0

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

    for g, name in enumerate(["pairs-0.64", "pairs-0.16", "mono-0.16", "rev-0.16"]):
        hs = list(range(g * (HEADS // KV), (g + 1) * (HEADS // KV)))
        spread = max(
            float((o_npu[hs[0]] - o_npu[h]).abs().max()) for h in hs[1:]
        )
        h = hs[0]
        p_npu = o_npu[h, :S]
        print(f"[{name}] cross-head max|dO| = {spread:.6f}")
        if name.startswith("pairs"):
            diffs = [float((p_npu[2 * i] - p_npu[2 * i + 1]).abs()) for i in range(S // 2)]
            print(
                f"  pair |p_2i - p_2i+1|: max {max(diffs):.6f}, "
                f"mean {sum(diffs) / len(diffs):.6f};  p[0..7] = "
                + " ".join(f"{v:.5f}" for v in p_npu[:8].tolist())
            )
        # vs sim
        k0 = kc[g, :, 0].to(torch.float64)
        s_bf = bf((8.0 * k0 * INV_SQRT_D).to(torch.float32))[:S]
        f = torch.zeros(S)
        m_old, l = -1e30, 0.0
        fstore, cstore = [], []
        for t0 in range(0, S, CS):
            sc = s_bf[t0 : t0 + CS]
            m_new = max(sc.max().item(), m_old)
            c_c = bf(torch.exp2(torch.tensor(bf((m_old - m_new) * LOG2E_K))))
            fc = bf(torch.exp2(bf((sc - m_new) * LOG2E_K)))
            fstore.append((fc, c_c, t0))
            l = c_c * l + fc.sum()
            m_old = m_new
        Y = torch.zeros(S)
        for fc, c_c, t0 in fstore:
            Y = Y * c_c
            Y[t0 : t0 + fc.shape[0]] = Y[t0 : t0 + fc.shape[0]] + fc  # one-hot V
        p_sim = bf((Y / bf(torch.tensor(l))).to(torch.float32))
        rel = ((p_npu - p_sim).abs() / p_sim.clamp(min=1e-9))
        top = rel.argsort(descending=True)[:6]
        print(
            f"  vs sim: rms {float((p_npu - p_sim).pow(2).mean().sqrt()):.6f} "
            f"maxrel {float(rel.max()):.5f}  worst: "
            + " ".join(f"[t{i}:{float(rel[i]):.4f}]" for i in top.tolist())
        )


if __name__ == "__main__":
    main()
