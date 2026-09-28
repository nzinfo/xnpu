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

N = int(sys.argv[1]); SLEEP = float(sys.argv[2])
t0 = time.time()
for i in range(N):
    op.run_runlist()
    if SLEEP > 0:
        time.sleep(SLEEP)
    if i % 20 == 0:
        print(f"iter {i}: {time.time()-t0:.1f}s", flush=True)
print("DONE", N, "iters,", round(time.time()-t0,1), "s, sleep", SLEEP)
