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

REGIONS = [("o",0,2304,288),("res1",2304,4352,0),("gate0_3",9280,15552,1568),
           ("up4_7",18560,24832,1568),("down",27840,34240,800),("qkv",37120,40704,448)]
def rd():
    return op.read_buffer("output", (op.c_total_rows,), dtype=np.uint16).copy()

base = None
for i in range(500):
    op.run_runlist()
    cur = rd()
    if base is None:
        base = cur
        exp = g["output_raw"].view(torch.uint16).numpy()
        print("baseline diff vs golden per region:")
        for name, lo, hi, _ in REGIONS:
            print(f"  {name:8s}: {int((base[lo:hi]!=exp[lo:hi]).sum())}/{hi-lo}")
        continue
    d = cur != base
    nd = int(d.sum())
    if nd:
        print(f"iter {i}: C differs from baseline in {nd} rows:")
        for name, lo, hi, stride in REGIONS:
            dd = cur[lo:hi] != base[lo:hi]
            if int(dd.sum()):
                rows = np.nonzero(dd)[0][:8] + lo
                print(f"  {name:8s}: {int(dd.sum())} rows, first {rows.tolist()}")
        sys.exit(0)
    if i % 25 == 0:
        print(f"iter {i} clean", flush=True)
print("no divergence in 500")
