import numpy as np
def load_u16(p):
    a = np.fromfile(p, dtype=np.uint16)
    return (a.astype(np.uint32) << 16).view(np.float32)
x = load_u16("/tmp/quad_x.bin")            # x0 + o(0)  (quad host trajectory @ L0)
g = load_u16("/home/nzinfo/qwen/xnpu/build/dec_hy/golden_L00.bin")  # x0+o+down
down = g - x
print(f"implied down(0): max|.|={np.abs(down).max():.4f}  mean|.|={np.abs(down).mean():.4f}")
for t in (0.005, 0.01, 0.02, 0.05, 0.1):
    print(f"  rows |down|>{t}: {(np.abs(down)>t).sum()}/2048")
print("sample rows:", np.round(down[:12], 4).tolist())
