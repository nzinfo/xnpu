import sys, time, torch, numpy as np
sys.path.insert(0, "/home/nzinfo/qwen/xnpu/IRON")
from iron.operators.w4gemvu.op_quad import AIEW4GEMVUQuad
from iron.operators.w4gemvu.test_quad import generate_quad_reference
from iron.common.aie_context import AIEContext
from iron.common.test_utils import torch_to_numpy

M1,K1,M2,M3,K3,M4 = 2048,2048,12288,2048,6144,3072
g = generate_quad_reference(M1,K1,M2,M3,K3,M4)
ctx = AIEContext()
op = AIEW4GEMVUQuad(M1=M1,K1=K1,M2=M2,M3=M3,K3=K3,M4=M4,num_aie_columns=8,group_size=32,context=ctx)
inputs = {
    "packed1": op.build_packed1(torch.from_numpy(g["packed1"]), g["activation"]),
    "packed2": op.build_packed_w(torch.from_numpy(g["packed2_blocks"]), g["wgt1"], op.blocks1),
    "packed3": torch.from_numpy(g["packed3_blocks"]),
    "packed4": op.build_packed_w(torch.from_numpy(g["packed4_blocks"]), g["wgt2"], op.blocks3),
    "output": op.build_c_init(g["res1"]),
}
ctx.compile_all(); ctx.prepare_runtime()
op.write_buffer("output", np.zeros(op.buffers["output"], dtype=np.uint8))
for n, b in inputs.items():
    op.write_buffer(n, torch_to_numpy(b))
exp = g["output_raw"].view(torch.uint16).numpy()

REGIONS = [("o",0,2304),("res1",2304,4352),("win1pad",4352,9280),("gate0_3",9280,15552),
           ("padA",15552,18560),("up4_7",18560,24832),("padB",24832,27840),
           ("down",27840,34240),("res2pad",34240,37120),("qkv",37120,40704)]
for i in range(500):
    try:
        op.run_runlist()
    except RuntimeError as e:
        print(f"HANG at iter {i}: {e}")
        act = op.read_buffer("output", (op.c_total_rows,), dtype=np.uint16)
        for name, lo, hi in REGIONS:
            d = act[lo:hi] != exp[lo:hi]
            n = int(d.sum())
            if n:
                first = int(np.nonzero(d)[0][0]) + lo
                print(f"  {name:8s}: {n:5d}/{hi-lo} rows differ (first @{first})")
            else:
                print(f"  {name:8s}: 0 (complete/correct)")
        # partial-column detail for the first bad region
        for name, lo, hi, stride in (("down",27840,34240,800),("qkv",37120,40704,448)):
            d = act[lo:hi] != exp[lo:hi]
            per = [int(d[c*stride:(c+1)*stride].sum()) for c in range(8)]
            print(f"  {name} per-col diffs: {per}")
        sys.exit(1)
    if i % 50 == 0:
        print(f"iter {i}", flush=True)
print("no hang in 500")
