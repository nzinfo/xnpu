import sys, torch, numpy as np
sys.path.insert(0, "/home/nzinfo/qwen/xnpu/IRON")
from iron.operators.w4gemvu.op_quad import AIEW4GEMVUQuad
from iron.operators.w4gemvu.test_quad import generate_quad_reference

M1,K1,M2,M3,K3,M4 = 2048,2048,12288,2048,6144,3072
g = generate_quad_reference(M1,K1,M2,M3,K3,M4)
from iron.common.aie_context import AIEContext
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
from iron.common.test_utils import torch_to_numpy
op.write_buffer("output", np.zeros(op.buffers["output"], dtype=np.uint8))
for n, b in inputs.items():
    op.write_buffer(n, torch_to_numpy(b))
op.run_runlist()
act = op.read_buffer("output", (op.c_total_rows,), dtype=np.uint16)
exp = g["output_raw"].view(torch.uint16).numpy()
a, e = act, exp
d = np.nonzero(a != e)[0]
print("bit-diff rows:", len(d), "min", d.min() if len(d) else None, "max", d.max() if len(d) else None)
import collections
regions = collections.Counter()
for r in d:
    if r < 2304: regions["o"] += 1
    elif r < 4352: regions["res1"] += 1
    elif r < 9280: regions["win1pad"] += 1
    elif r < 15552: regions["gate0_3"] += 1
    elif r < 18560: regions["padA"] += 1
    elif r < 24832: regions["up4_7"] += 1
    elif r < 27840: regions["padB"] += 1
    elif r < 34240: regions["down"] += 1
    elif r < 37120: regions["win2tail/res2"] += 1
    else: regions["qkv"] += 1
print(dict(regions))
def show(rows):
    for r in rows:
        print(f"{r:6d} act=0x{a[r]:04x} exp=0x{e[r]:04x}")
print("--- win2 col0 rows 0..63 (27840..27903)")
show(range(27840, 27876))
# where inside col0 section do diffs fall?
c0 = d[(d >= 27840) & (d < 28640)]
print("col0 diff row offsets within section:", (c0 - 27840)[:40].tolist())

print("--- gate col0 head rows 9280..9320")
show(range(9280, 9320))
print("--- up col0 head rows 18560..18600")
show(range(18560, 18600))
print("--- qkv col0 head rows 37120..37170 (every 4th)")
show(range(37120, 37172, 4))
print("--- o col0 head rows 0..40")
show(range(0, 36, 2))

print("--- identify group0 mystery rows")
m = a[27840:27856]
v0, v1 = int(m[0]), int(m[1])
hits = np.nonzero((a[:-1] == v0) & (a[1:] == v1))[0]
print("pair-match hits at rows:", hits[:20].tolist())
print("mystery[:4]   :", [hex(x) for x in m[:4]])

# ---- per-group classification: Z=zeros, .=exact exp, gk=matches exp group k, X=other
def classify(base_per_col, stride, n_groups, label, colrange=8):
    out = []
    for c in range(colrange):
        b = base_per_col + c * stride
        row = []
        for gi in range(n_groups):
            gact = a[b + 16*gi : b + 16*gi + 16]
            gexp = e[b + 16*gi : b + 16*gi + 16]
            if (gact == 0).all():
                row.append("Z")
            elif (gact == gexp).all():
                row.append(".")
            else:
                # find matching exp group
                found = None
                for gj in range(n_groups):
                    if (gact == e[b + 16*gj : b + 16*gj + 16]).all():
                        found = gj; break
                row.append(f"g{found}" if found is not None else "X")
        out.append("".join(row))
    for c, r in enumerate(out):
        print(f"  col{c}: {r}")

# gate cols 0..3 at 9280, stride 1568, 98 groups (2 dummy + 96)
print("=== gate sections (cols0-3) ===")
classify(9280, 1568, 98, "gate")
print("=== up sections (cols4-7) ===")
classify(18560, 1568, 98, "up")
print("=== down sections (all 8) ===")
classify(27840, 800, 50, "down")
print("=== qkv sections (all 8) ===")
classify(37120, 448, 28, "qkv")

# ---- tolerance-aware classification with shift detection ----
def f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)
def tol_ok(x, y):
    xf, yf = f32(x), f32(y)
    return np.all(np.abs(xf - yf) <= np.maximum(0.08 * np.abs(yf), 0.8))
def classify_tol(base, stride, n_groups, label, cols=8):
    print(f"=== {label} (tolerance) ===")
    for c in range(cols):
        b = base + c * stride
        row = []
        for gi in range(n_groups):
            gact = a[b + 16*gi : b + 16*gi + 16]
            if (gact == 0).all():
                row.append("Z"); continue
            tag = "X"
            for shift in (0, -1, +1, -2, +2):
                gj = gi + shift
                if 0 <= gj < n_groups and tol_ok(gact, e[b + 16*gj : b + 16*gj + 16]):
                    tag = "." if shift == 0 else { -1: "<", +1: ">", -2: "L", +2: "R" }[shift]
                    break
            row.append(tag)
        print(f"  col{c}: {''.join(row)}")
classify_tol(9280, 1568, 98, "gate cols0-3")
classify_tol(18560, 1568, 98, "up cols4-7")
classify_tol(27840, 800, 50, "down")
classify_tol(37120, 448, 28, "qkv")

# ---- down-partial deviation stats (hw sigmoid vs accurate-exp golden) ----
af = (a.astype(np.uint32) << 16).view(np.float32)
ef = (e.astype(np.uint32) << 16).view(np.float32)
m = np.zeros(len(a), bool)
mq = np.zeros(len(a), bool)
for c in range(8):
    b = 27840 + c*800
    m[b+32 : b+800] = True
    q = 37120 + c*448
    mq[q+48 : q+48+384] = True
for name, mask in (("down-partials", m), ("qkv", mq)):
    d = np.abs(af[mask] - ef[mask]); r = d / np.maximum(np.abs(ef[mask]), 1e-9)
    print(f"{name}: n={mask.sum()} max_abs={d.max():.3f} max_rel={r[d>0.8].max() if (d>0.8).any() else 0:.3f}")
    for band in ((0.25,2.5),(0.35,4.0),(0.5,6.0),(1.0,10.0)):
        ok = (d <= np.maximum(band[0]*np.abs(ef[mask]), band[1]))
        print(f"   band rel{band[0]}/abs{band[1]}: fail {int((~ok).sum())}")


# ---- decisive: does the glue's h2'' input (stored partials sum) match qkv cleanliness? ----
# logical model row n = col*256 + j; section rows per chunk: 32 + 256*c + j
stored_sum = np.zeros(2048, np.float32); gold_sum = np.zeros(2048, np.float32)
for col in range(8):
    for j in range(256):
        rs = [27840 + col*800 + 32 + 256*c + j for c in range(3)]
        stored_sum[col*256+j] = sum(af[r] for r in rs)
        gold_sum[col*256+j] = sum(ef[r] for r in rs)
dev = np.abs(stored_sum - gold_sum)
print("h2'' partial-sum deviation: max", dev.max(), "rows>2:", int((dev>2).sum()), "rows>10:", int((dev>10).sum()))
top = np.argsort(-dev)[:10]
for n in top:
    col, j = n//256, n%256
    rs = [27840+col*800+32+256*c+j for c in range(3)]
    print(f"  n={n} col{col} j={j}: stored={[round(float(af[r]),2) for r in rs]} gold={[round(float(ef[r]),2) for r in rs]}")
# permutation check: do bad act values exist in golden partials anywhere?
badrows = np.nonzero((np.abs(af-ef) > np.maximum(np.abs(ef),10)) & m)[0]
hits = 0
gv = ef[m]
for r in badrows[:50]:
    if np.any(np.abs(gv - af[r]) < 0.01): hits += 1
print("bad values found elsewhere in golden partials:", hits, "/", len(badrows))
