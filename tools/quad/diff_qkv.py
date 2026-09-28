import numpy as np
def load(p):
    a = np.fromfile(p, dtype=np.uint16)
    return (a.astype(np.uint32) << 16).view(np.float32)
print(f"{'L':>3} {'q_bad':>6} {'q_maxrel':>8} {'k_bad':>6} {'k_maxrel':>8} {'v_bad':>6} {'v_maxrel':>8} {'v_maxabs':>8}")
for n in range(1, 13):
    try:
        q = load(f"/tmp/qkvdump/quad_L{n:02}.bin"); f = load(f"/tmp/qkvdump/fused_L{n:02}.bin")
    except FileNotFoundError:
        break
    for name, lo, hi in (("q",0,2048),("k",2048,2560),("v",2560,3072)):
        a, b = q[lo:hi], f[lo:hi]
        d = np.abs(a-b); rel = d/np.maximum(np.abs(b),1e-6)
        bad = int(((d > 0.005+0.005*np.abs(b))).sum())
        if name=="v": vma=d.max()
        print(f"{n:>3} {bad:>6} {rel.max():>8.3f}", end="")
    print(f" {vma:>8.4f}")
