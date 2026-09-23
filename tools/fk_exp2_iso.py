#!/usr/bin/env python3
"""M5c 判别实验 2f：硬件 exp2 intrinsic 误差曲线的隔离测量。

2e 已证偏差是「取值性、确定性、逐位可复现」（同 score 同输出；位置/头无关），
且经 online-rescale（C_c）路径放大 3×。嫌疑收敛到 ::exp2(accum<accfloat>)
硬件初等函数指令本身的按值误差。

隔离设计：每个 run 只有位置 t*=50 的 k 非零，其余 100 个位置 score 恰为 0
（arg 恒 0）。则 l = 100·f(0) + x，p_t* = x/(100+x)（设 f(0)=1，2d 已证），
可精确反解 x = 100p/(1-p)。对 ~100 个 arg 值（bf16 可表示）逐一测：
  err(arg) = f_npu / bf16(2^arg_exact) - 1
一次编译，循环写 buffer 复用 runlist。输出 err-arg 曲线 → 若 |err| 达 % 级
且随 arg 变化 → 硬件初等函数精度实锤（bf16 快速路径）；据此决定修法。
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
TSTAR = 50


def bf(t):
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    return t.to(torch.bfloat16).to(torch.float64)


def main():
    q = torch.zeros(HEADS, HD, dtype=torch.bfloat16)
    for h in range(HEADS):
        q[h, 0] = 8.0

    angles = make_rope_angles_interleaved(torch.ones(HD // 2), torch.zeros(HD // 2))
    q_packed = pack_q_with_angles(q, angles, HEADS // KV, KV, seq_len_cur=S)

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
    op.write_buffer("queries", torch_to_numpy(q_packed))

    # arg probe list: dense in [-24, 0], 4 per run (one per KV group)
    args_list = [round(-0.125 * i, 4) for i in range(0, 193)]  # 0 .. -24
    extra = [-0.001, -0.002, -0.004, -0.008, -0.016, -0.03, -0.06, 0.003]
    args_list = sorted(set(args_list + extra), reverse=True)

    results = []  # (arg_seen, f_npu, f_cr)
    for base in range(0, len(args_list), KV):
        batch = args_list[base : base + KV]
        kc = torch.zeros(KV, 1024, HD, dtype=torch.bfloat16)
        vc = torch.zeros(KV, 1024, HD, dtype=torch.bfloat16)
        for g, arg_target in enumerate(batch):
            s_star = arg_target / LOG2E_K
            kc[g, TSTAR, 0] = torch.tensor(s_star / (8.0 * INV_SQRT_D), dtype=torch.bfloat16)
            vc[g, TSTAR, TSTAR] = 1.0
        op.write_buffer("kv_cache", torch_to_numpy(interleave_kv_cache(kc, vc)))
        op.run_runlist()
        out = op.read_buffer("output", (HEADS * HD,), dtype=np.dtype(bfloat16))
        o_npu = torch.from_numpy(out.astype(np.float32)).view(HEADS, HD).to(torch.float64)
        for g, arg_target in enumerate(batch):
            h = g * (HEADS // KV)
            p = o_npu[h, TSTAR].item()
            if p <= 0 or p >= 1:
                results.append((arg_target, float("nan"), float("nan"), p))
                continue
            x = 100.0 * p / (1.0 - p)  # f_npu at the probe position
            # recompute the arg the hardware actually saw (k bf16 round-trip)
            k_bf = kc[g, TSTAR, 0].to(torch.float64).item()
            s_seen = bf(torch.tensor(8.0 * k_bf * INV_SQRT_D).to(torch.float32)).item()
            arg_seen = bf(torch.tensor(s_seen * LOG2E_K)).item() if s_seen < 0 else 0.0
            f_cr = bf(torch.exp2(torch.tensor(arg_seen)))  # correctly rounded ref
            results.append((arg_seen, x, f_cr.item(), p))

    print(f"{'arg':>9} {'f_npu':>12} {'f_CR':>12} {'relerr':>9}")
    nan_ct = 0
    for arg, x, f_cr, p in results:
        if x != x:
            nan_ct += 1
            continue
        rel = x / f_cr - 1.0
        print(f"{arg:>9.4f} {x:>12.6f} {f_cr:>12.6f} {rel:>+9.5f}")
    print(f"({nan_ct} probes had p==0 (underflow); total {len(results)})")

    rels = [x / f - 1.0 for _, x, f, _ in results if x == x and f > 0]
    if rels:
        print(
            f"relerr: max {max(rels):+.5f} min {min(rels):+.5f} "
            f"mean {sum(rels) / len(rels):+.5f} n={len(rels)}"
        )


if __name__ == "__main__":
    main()
