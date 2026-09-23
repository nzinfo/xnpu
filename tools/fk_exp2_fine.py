#!/usr/bin/env python3
"""M5c 判别实验 2d：exp2 硬件 intrinsic 的按值误差扫描。

2c 结论：real-data 上 NPU 带每头 ±3–6% 的 common-mode（lstsq scale），而合成
数据完全没有；mono-deep（步长 0.16）模型贴合。→ 嫌疑集中到 exp2 intrinsic
的误差是「输入值」的函数：粗扫描没采到的 arg 值区域可能有坏点，同一头内
所有命中坏区的位置产生同向误差 → common-mode。

本实验：V one-hot（O[t] = w_t 直接暴露每个权重），score 步长 0.02/偏移
0.005/0.01/0.16 四组，逐位置比较 f_npu vs f_sim，打印最坏 |Δf| 与对应 arg。
若 fine 扫描出现 >0.5% 的坏点 → exp2 按值误差实锤；若干净 → exp2 彻底排除，
common-mode 在别处（q/k 输入侧）。
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

MODES = [("fine-a", 0.02, 0.0), ("fine-b", 0.02, 0.005), ("fine-c", 0.02, 0.01), ("coarse", 0.16, 0.0)]


def bf(t):
    return t.to(torch.bfloat16).to(torch.float64)


def main():
    q = torch.zeros(HEADS, HD, dtype=torch.bfloat16)
    kc = torch.zeros(KV, 1024, HD, dtype=torch.bfloat16)
    vc = torch.zeros(KV, 1024, HD, dtype=torch.bfloat16)

    for g, (name, step, off) in enumerate(MODES):
        h0 = g * (HEADS // KV)
        for h in range(h0, h0 + HEADS // KV):
            q[h, 0] = 8.0
        for t in range(S):
            s_t = -(step * t + off)
            kc[g, t, 0] = torch.tensor(s_t / (8.0 * INV_SQRT_D), dtype=torch.bfloat16)
            vc[g, t, t] = 1.0  # O[h, t] = w_t directly

    # kernel-arithmetic sim: identical rounding chain, one shared max m=bf16(0)
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

    print(f"{'mode':>8} {'maxRelErr':>9}  worst positions (arg, rel err)")
    for g, (name, step, off) in enumerate(MODES):
        h = g * (HEADS // KV)  # first head of group
        # scores after bf16(k) round-trip: recompute what the kernel stores
        k0 = kc[g, :, 0].to(torch.float64)
        s_bf = bf((8.0 * k0 * INV_SQRT_D).to(torch.float32))[:S]
        m = s_bf.max()
        f = bf(torch.exp2(bf(((s_bf - m) * LOG2E_K)).to(torch.float32)))
        # mono sweep: m never updates (C_c = 1), Y = f exact, O = bf16(f / bf16(l))
        l_bf = bf(torch.tensor(f.sum()))
        w_sim = bf((f / l_bf).to(torch.float32))
        w_npu = o_npu[h, :S]
        rel = ((w_npu - w_sim).abs() / w_sim.clamp(min=1e-6))
        top = rel.argsort(descending=True)[:4]
        print(
            f"{name:>8} {rel.max().item():>9.5f}  "
            + "  ".join(
                f"arg={((s_bf[i] - m) * LOG2E_K).item():+.3f} rel={rel[i].item():.5f}"
                for i in top
            )
        )
        # top-8 largest-p positions: error where the output mass actually is
        big = f.argsort(descending=True)[:8]
        print(
            "          top-p: "
            + " ".join(
                f"[arg{((s_bf[i] - m) * LOG2E_K).item():+6.2f}:{rel[i].item():.5f}]"
                for i in big
            )
        )
        if name == "coarse":
            print("          raw dump (t, arg, f_sim, p_sim, p_npu, f_implied=p_npu*l_sim):")
            for i in big:
                arg = ((s_bf[i] - m) * LOG2E_K).item()
                print(
                    f"            t={i:3d} arg={arg:+7.4f} f={f[i].item():.6f} "
                    f"p_sim={w_sim[i].item():.6f} p_npu={w_npu[i].item():.6f} "
                    f"f_impl={(w_npu[i] * f.sum()).item():.6f}"
                )
            print(f"          l_sim={f.sum().item():.4f} l_implied={1.0 / w_npu[big[0]].item():.4f}")


if __name__ == "__main__":
    main()
