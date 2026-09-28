#!/usr/bin/env python3
"""P28-4 offline diagnosis: replay the kernel-side chain in numpy under
candidate-bug hypotheses and match against the dumped board output."""
import numpy as np
import torch
from ml_dtypes import bfloat16

torch.manual_seed(42)
COLS = 8
SUCC = {0: 1, 1: 2, 2: 3, 3: 7, 7: 6, 6: 5, 5: 4, 4: 0}
pos = lambda w: w if w < 4 else 11 - w

# ---- golden chain (verbatim from test_layerv2) ----
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "w4ref", "/home/nzinfo/qwen/xnpu/IRON/iron/operators/w4gemvu/reference.py")
_w4ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_w4ref)
quantize_and_pack = _w4ref.quantize_and_pack
quantize_vector = _w4ref.quantize_vector


def _f32(b):
    return (np.asarray(b, dtype=np.uint32) << 16).view(np.float32).copy()


def _bf16(f32):
    u = np.asarray(f32, dtype=np.float32).view(np.uint32)
    rounded = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _kernel_rsqrt(s):
    v = np.float32(s)
    magic = np.array(0x5F3759DF, dtype=np.uint32)
    x = (magic - (v.view(np.uint32) >> np.uint32(1))).view(np.float32)
    for _ in range(3):
        x = np.float32(x * (np.float32(2.0) - v * x * x))
    return x


def _sumsq_sequential(h2f):
    sumsq = np.float32(0)
    for v in h2f:
        sumsq = np.float32(sumsq + np.float32(v * v))
    return sumsq


def _quantize_kernel(xn_f32, m):
    v32 = xn_f32.reshape(m // 32, 32)
    amax = np.abs(v32).max(axis=1)
    d_bits = _bf16((amax / np.float32(127.0)).astype(np.float32))
    invd = (np.float32(1.0) / _f32(d_bits.astype(np.uint32)))[:, None]
    v = v32 * invd
    v = np.where(amax[:, None] == 0, 0.0, v)
    q = np.round(v)
    q = np.clip(q, -127, 127).astype(np.int8).reshape(m)
    return q, d_bits


def _dequant(q, d_bits):
    return q.astype(np.float32) * np.repeat(_f32(d_bits.astype(np.uint32)), 32)

W_o = (torch.rand(2048, 2048) * 2 - 1).numpy()
W_g = (torch.rand(6144, 2048) * 2 - 1).numpy()
W_u = (torch.rand(6144, 2048) * 2 - 1).numpy()
W_d = (torch.rand(2048, 6144) * 2 - 1).numpy()
W_q = (torch.rand(3072, 2048) * 2 - 1).numpy()
x_attn = (torch.rand(2048) * 2 - 1).to(torch.bfloat16)
x_n = (torch.rand(2048) * 2 - 1).to(torch.bfloat16)
wgt2 = (torch.rand(2048) * 2 - 1).to(torch.bfloat16)
wgt1 = (torch.rand(2048) * 2 - 1).to(torch.bfloat16)

_, W_o_dq = quantize_and_pack(W_o)
_, W_g_dq = quantize_and_pack(W_g)
_, W_u_dq = quantize_and_pack(W_u)
_, W_d_dq = quantize_and_pack(W_d)
_, W_q_dq = quantize_and_pack(W_q)
q1, d1, x_deq1 = quantize_vector(x_attn)

o_out = (W_o_dq.to(torch.float32) @ x_deq1).to(torch.bfloat16)
o_f = _f32(o_out.view(torch.uint16).numpy().astype(np.uint32))
xn_f32 = _f32(x_n.view(torch.uint16).numpy().astype(np.uint32))
xp_bits = _bf16(xn_f32 + o_f)
xp_r = _f32(xp_bits.astype(np.uint32))
w2f = _f32(wgt2.view(torch.uint16).numpy().astype(np.uint32))
w1f = _f32(wgt1.view(torch.uint16).numpy().astype(np.uint32))

# ---- kernel-side sim with permuted ring gathers ----
def sim(ring1_shift=0, ring2_shift=0, ring3_shift=0):
    # ringN_shift: worker 0's slot s ends up holding origin
    # (s + shift) & 7 instead of s. (slot s receives round r = (0 - s) & 7
    # -> origin (0 - r + shift) & 7 = (s + shift) & 7)
    def gather_w0(chunks, shift):
        out = chunks[(np.arange(COLS) + shift) & 7].copy()
        out[0] = chunks[0]  # own chunk always correct
        return out

    xp_per_p = xp_bits.reshape(COLS, 256)
    xp_full = gather_w0(xp_per_p, ring1_shift).reshape(2048)
    xp_full_r = _f32(xp_full.astype(np.uint32))

    inv1 = _kernel_rsqrt(np.float32(_sumsq_sequential(xp_full_r) / np.float32(2048) + np.float32(1e-5)))
    xn_f = xp_full_r * inv1 * w2f
    q2, d2 = _quantize_kernel(xn_f, 2048)
    x_deq2 = torch.from_numpy(_dequant(q2, d2))

    g_out = (W_g_dq.to(torch.float32) @ x_deq2).to(torch.bfloat16)
    u_out = (W_u_dq.to(torch.float32) @ x_deq2).to(torch.bfloat16)
    g_f = _f32(g_out.view(torch.uint16).numpy().astype(np.uint32))
    u_f = _f32(u_out.view(torch.uint16).numpy().astype(np.uint32))
    sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-g_f.astype(np.float32)))
    sw_bits = _bf16(g_f * sig * u_f)
    q3, d3 = _quantize_kernel(_f32(sw_bits.astype(np.uint32)), 6144)

    # ring2 as worker 0 sees it: slot s <- origin (s + shift) & 7
    q3g = q3.reshape(COLS, 768)
    d3g = d3.reshape(COLS, 24)
    qg = q3g[(np.arange(COLS) + ring2_shift) & 7].copy()
    dg = d3g[(np.arange(COLS) + ring2_shift) & 7].copy()
    qg[0] = q3g[0]; dg[0] = d3g[0]
    sw_full_q = qg.reshape(6144)
    sw_full_d = dg.reshape(192)
    x_deq3 = sw_full_q.astype(np.float32) * np.repeat(
        _f32(sw_full_d.astype(np.uint32)), 32)

    W_df = W_d_dq.to(torch.float32)
    p_dn = torch.empty(3, 2048, dtype=torch.bfloat16)
    for c in range(3):
        p_dn[c] = (W_df[:, c * 2048:(c + 1) * 2048]
                   @ torch.from_numpy(x_deq3[c * 2048:(c + 1) * 2048])).to(torch.bfloat16)
    dacc = _f32(p_dn.view(torch.uint16).numpy().astype(np.uint32)).sum(axis=0)
    # worker 0's own xn1 chunk (rows 0..256)
    xn1_bits_sim = _bf16(xp_r + dacc)

    xn1_per_p = xn1_bits_sim.reshape(COLS, 256)
    xn1_full = gather_w0(xn1_per_p, ring3_shift).reshape(2048)
    xn1_r = _f32(xn1_full.astype(np.uint32))

    inv2 = _kernel_rsqrt(np.float32(_sumsq_sequential(xn1_r) / np.float32(2048) + np.float32(1e-5)))
    xn2_f = xn1_r * inv2 * w1f
    q4, d4 = _quantize_kernel(xn2_f, 2048)
    x_deq4 = torch.from_numpy(_dequant(q4, d4))
    qkv = (W_q_dq.to(torch.float32) @ x_deq4).to(torch.bfloat16)
    return qkv, xn1_bits_sim, dacc


act = np.load("/tmp/p28/dbg_act_bits.npy")
act_f = _f32(act)
OUT_ROWS = 640

def score(qkv_sim, xn1_sim, tag):
    # sim is worker 0's view (own-slot exception at slot 0); only w=0 is
    # exact, other workers differ by their own-slot chunk placement.
    qf = _f32(qkv_sim.view(torch.uint16).numpy())
    xf = _f32(xn1_sim)
    parts = []
    for w in range(COLS):
        p = pos(w)
        base = w * OUT_ROWS
        dq = np.abs(act_f[base:base + 384] - qf[p * 384:(p + 1) * 384]).max()
        dx = np.abs(act_f[base + 384:base + OUT_ROWS] - xf[p * 256:(p + 1) * 256]).max()
        parts.append(f"w{w}:{dq:.0f}/{dx:.0f}")
    print(f"{tag:30s} " + "  ".join(parts))


print("   (each cell: qkv maxdiff / xn1 maxdiff vs board, per worker)")
_, _, dacc_g = sim(0, 0, 0)
for s1 in (0, 1, -1):
    for s2 in (0, 1, -1):
        qkv_sim, xn1_sim, _ = sim(ring1_shift=s1, ring2_shift=s2)
        score(qkv_sim, xn1_sim, f"ring1_shift={s1} ring2_shift={s2}")
for s3 in (1, -1):
    qkv_sim, xn1_sim, _ = sim(ring3_shift=s3)
    score(qkv_sim, xn1_sim, f"ring3_shift={s3}")

# ---- structural check on xn1: implied dacc tile permutation ----
print("\nimplied dacc (act_xn1 - xp) vs golden dacc tile match:")
for w in (0, 1):
    p = pos(w)
    base = w * OUT_ROWS
    act_x = act_f[base + 384:base + OUT_ROWS]
    xp_chunk = xp_r[p * 256:(p + 1) * 256]
    d_impl = act_x - xp_chunk
    dg = dacc_g[p * 256:(p + 1) * 256]
    for t in range(0, 4):
        d_tile = d_impl[t * 16:(t + 1) * 16]
        best = int(np.argmin([np.abs(dg[16 * tt:16 * tt + 16] - d_tile).max()
                              for tt in range(16)]))
        md = np.abs(dg[16 * best:16 * best + 16] - d_tile).max()
        print(f"  w{w} tile{t}: best golden tile {best} maxdiff {md:.2f} "
              f"(scale {np.abs(d_tile).max():.0f})")

# ---- scale stats + chain consistency: act_qkv vs chain(act_xn1) ----
print("\nscale stats (abs):")
exp_bits = np.load("/tmp/p28/dbg_exp_bits.npy")
ef = _f32(exp_bits)
q = ef[:COLS*640].reshape(COLS, 640)[0, :384]  # w0 qkv exp slice
print(f"  qkv exp: p50 {np.median(np.abs(q)):.1f} p90 {np.percentile(np.abs(q),90):.1f} max {np.abs(q).max():.1f}")
x = ef[384:640]
print(f"  xn1 exp: p50 {np.median(np.abs(x)):.1f} p90 {np.percentile(np.abs(x),90):.1f} max {np.abs(x).max():.1f}")

# rebuild the full actual xn1 vector (position p's chunk from worker at pos p)
xn1_act = np.zeros(2048, dtype=np.uint16)
for w in range(COLS):
    p = pos(w)
    xn1_act[p*256:(p+1)*256] = act[w*640+384:(w+1)*640]
xa = _f32(xn1_act)
inv2 = _kernel_rsqrt(np.float32(_sumsq_sequential(xa) / np.float32(2048) + np.float32(1e-5)))
xn2_f = xa * inv2 * w1f
q4, d4 = _quantize_kernel(xn2_f, 2048)
qkv_c = (W_q_dq.to(torch.float32) @ torch.from_numpy(_dequant(q4, d4))).to(torch.bfloat16)
qcf = _f32(qkv_c.view(torch.uint16).numpy())
print("\nchain consistency: qkv recomputed FROM actual xn1 vs actual qkv")
for w in (0, 1, 4):
    p = pos(w)
    base = w * 640
    a_q = act_f[base:base+384]
    d_direct = np.abs(a_q - qcf[p*384:(p+1)*384]).max()
    d_golden = np.abs(a_q - ef[base:base+384]).max()
    # also: golden xn1 -> qkv vs actual qkv (isolates xn1 error's effect)
    print(f"  w{w}: |act-chain(act_xn1)| {d_direct:8.1f}   |act-golden| {d_golden:8.1f}")

# ---- xn1 error structure: which elements are wrong, how ----
print("\nxn1 error structure per worker (err = act - exp, |err|>200 = wrong):")
for w in (0, 1, 4):
    p = pos(w)
    a_x = act_f[w*640+384:(w+1)*640]
    e_x = ef[w*640+384:(w+1)*640]
    err = a_x - e_x
    bad = np.abs(err) > 200
    print(f"  w{w}: wrong {bad.sum()}/256; err p50 {np.median(np.abs(err)):.0f} max {np.abs(err).max():.0f}"
          f"; ratio act/exp p50 {np.median(np.abs(a_x[bad]))/np.median(np.abs(e_x[bad])):.3f}")
    # 16-row tile wrong-count profile
    tiles = bad.reshape(16, 16).sum(axis=1)
    print(f"    per-16-row-tile wrong counts: {tiles.tolist()}")
    # sign correlation: does err correlate with exp or with act?
    c_exp = np.corrcoef(err, e_x)[0,1]
    print(f"    corr(err, exp) {c_exp:+.3f}  err/exp p50 {np.median(err[bad]/e_x[bad]):+.3f}")

# golden sw (recompute at module scope for the partial tests)
inv1g = _kernel_rsqrt(np.float32(_sumsq_sequential(xp_r) / np.float32(2048) + np.float32(1e-5)))
q2g, d2g = _quantize_kernel(xp_r * inv1g * w2f, 2048)
g_f = _f32((W_g_dq.to(torch.float32) @ torch.from_numpy(_dequant(q2g, d2g))).to(torch.bfloat16).view(torch.uint16).numpy().astype(np.uint32))
u_f = _f32((W_u_dq.to(torch.float32) @ torch.from_numpy(_dequant(q2g, d2g))).to(torch.bfloat16).view(torch.uint16).numpy().astype(np.uint32))
sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-g_f.astype(np.float32)))
q3, d3 = _quantize_kernel(_f32(_bf16(g_f * sig * u_f).astype(np.uint32)), 6144)
x_deq3g = q3.astype(np.float32) * np.repeat(_f32(d3.astype(np.uint32)), 32)
W_df = W_d_dq.to(torch.float32)
P = torch.empty(3, 2048, dtype=torch.bfloat16)
for c in range(3):
    P[c] = (W_df[:, c*2048:(c+1)*2048] @ torch.from_numpy(x_deq3g[c*2048:(c+1)*2048])).to(torch.bfloat16)
Pf = _f32(P.view(torch.uint16).numpy().astype(np.uint32))
combos = {"c0+c1+c2": Pf[0]+Pf[1]+Pf[2], "c1+c2": Pf[1]+Pf[2],
          "c0+c2": Pf[0]+Pf[2], "c0+c1": Pf[0]+Pf[1],
          "0.5c2": Pf[0]+Pf[1]+0.5*Pf[2]}
for w in (0, 1, 4):
    p = pos(w)
    a_x = act_f[w*640+384:(w+1)*640]
    xpc = xp_r[p*256:(p+1)*256]
    parts = []
    for name, dd in combos.items():
        md = np.abs(a_x - _f32(_bf16(xpc + dd[p*256:(p+1)*256]))).max()
        parts.append(f"{name}:{md:.0f}")
    print(f"  w{w}: " + "  ".join(parts))

# ---- dacc echo analysis (kernel lv_cxn TEMP variant: rows 384: = bf16(dacc)) ----
print("\ndacc echo vs golden dacc (maxdiff per worker, golden scale ~10k):")
dacc_gold = Pf.sum(axis=0)
dacc_act = np.load("/tmp/p28/dbg_act_bits.npy")
da_f = _f32(dacc_act)
for w in range(COLS):
    p = pos(w)
    a_d = da_f[w*640+384:(w+1)*640]
    g_d = dacc_gold[p*256:(p+1)*256]
    md = np.abs(a_d - g_d).max()
    rel = np.median(np.abs(a_d)) / np.median(np.abs(g_d))
    print(f"  w{w}: maxdiff {md:9.1f}  |act|/|gold| median scale ratio {rel:.3f}")

# ---- sw q echo analysis: lv_sw[0:512) vs golden q3[0:512) ----
print("\nsw q echo (lv_sw[0:512) = origin-0 q bytes) vs golden q3[0:512):")
qa = np.load("/tmp/p28/dbg_act_bits.npy")[384:640].view(np.uint8).view(np.int8)
qg = q3[:512]
nbad = int(np.sum(qa != qg))
print(f"  w0: mismatched bytes {nbad}/512")
if nbad:
    idx = np.nonzero(qa != qg)[0]
    print(f"  first mismatches at {idx[:16].tolist()}")
    for i in idx[:8]:
        print(f"    byte {i:4d} (group {i//32:2d} lane {i%32:2d}): act {qa[i]:4d} golden {qg[i]:4d}")
# per-group mismatch profile
gm = (qa != qg).reshape(16, 32).sum(axis=1)
print(f"  per-group wrong counts: {gm.tolist()}")

# ---- emulate the KERNEL sw chain (hw exp2 sigmoid, f32 div) vs echo ----
print("\nkernel-sigmoid emulation vs echo vs golden (q bytes 0:512):")
LOG2E = np.float32(1.4426950408889634)
def kernel_sw_q(g_f, u_f):
    xarg = (g_f * (-LOG2E)).astype(np.float32)
    t = _bf16(np.exp2(xarg))                       # hw exp2 -> bf16 result
    tf = _f32(t.astype(np.uint32))
    den = (np.float32(1.0) + tf)
    sig = (np.float32(1.0) / den)                  # aie::div ~ 1ulp
    gs = (g_f * sig).astype(np.float32)
    swb = _bf16((gs * u_f).astype(np.float32))
    return swb, _f32(swb.astype(np.uint32))

swb_k, temp_k = kernel_sw_q(g_f, u_f)
q3k, d3k = _quantize_kernel(temp_k, 6144)
qa = np.load("/tmp/p28/dbg_act_bits.npy")[384:640].view(np.uint8).view(np.int8)
print(f"  echo vs kernel-model : {int(np.sum(qa != q3k[:512]))}/512")
print(f"  echo vs golden       : {int(np.sum(qa != q3[:512]))}/512")
print(f"  kernel-model vs golden: {int(np.sum(q3k[:512] != q3[:512]))}/512")
# where do model and echo differ, if anywhere
d = np.nonzero(qa != q3k[:512])[0]
for i in d[:8]:
    print(f"    byte {i:4d}: echo {qa[i]:4d} model {q3k[i]:4d} golden {q3[i]:4d}")

# ---- arena2 echo: lv_arena[0:512) (replicated q2 groups 0-3) vs golden ----
print("\narena2 echo vs golden replicated q2 (groups 0-3):")
ar = np.load("/tmp/p28/dbg_act_bits.npy")[384:640].view(np.uint8).view(np.int8)
# golden replication: group g -> 128B pattern [x0 x1 x0 x1 / x0 x1 x0 x1]
rep = np.zeros(512, dtype=np.int8)
for g in range(4):
    xg = q2g[g*32:(g+1)*32]
    rep[g*128:g*128+64] = np.tile(xg[0:16], 4)
    rep[g*128+64:g*128+128] = np.tile(xg[16:32], 4)
nb = int(np.sum(ar != rep))
print(f"  w0: mismatched bytes {nb}/512")
if nb:
    idx = np.nonzero(ar != rep)[0]
    for i in idx[:10]:
        print(f"    byte {i:4d}: echo {ar[i]:4d} golden-rep {rep[i]:4d}")
    # compare against replicated q2k (kernel-model quantize) too

# ---- fixture bit-verification: X / xn / w2 / w1 elements vs golden ----
print("\nfixture input bit-verification (worker 0, position 0):")
ELEM = 18560
Wfx = np.load("/tmp/p28/dbg_w_fixture.npy")
Xfx = np.load("/tmp/p28/dbg_x_fixture.npy")
XNfx = np.load("/tmp/p28/dbg_xn_fixture.npy")
N_O, N_GATE, N_UP, N_DOWN, N_QKV = 16, 48, 48, 48, 24
# X element: q1 at [0,2048), d1 bf16 at [6144,6272)
xq = Xfx[0:ELEM]
ok_q = np.array_equal(xq[0:2048], q1.numpy().view(np.uint8))
d1_bits = d1.view(torch.uint16).numpy()
ok_d = np.array_equal(xq[6144:6272].view(np.uint16), d1_bits)
print(f"  X elem: q1 bytes equal {ok_q}, d1 bits equal {ok_d}")
# xn element: x_n[0:256] bf16 at [0,512)
xn_bits = x_n[0:256].view(torch.uint16).numpy()
ok_xn = np.array_equal(XNfx[0:512].view(np.uint16), xn_bits)
print(f"  xn elem: x_n chunk bits equal {ok_xn}")
# w2 element (index 16) / w1 element (index 169): wgt2/wgt1 bf16 [0,4096)
we = Wfx[16*ELEM:(17)*ELEM]
w2_bits = wgt2.view(torch.uint16).numpy()
w1_bits = wgt1.view(torch.uint16).numpy()
we1 = Wfx[169*ELEM:170*ELEM]
print(f"  w2 elem: wgt2 bits equal {np.array_equal(we[0:4096].view(np.uint16), w2_bits)}"
      f"  K hdr {np.frombuffer(we[ELEM-8:ELEM-4].tobytes(), dtype=np.uint32)[0]}")
print(f"  w1 elem: wgt1 bits equal {np.array_equal(we1[0:4096].view(np.uint16), w1_bits)}"
      f"  K hdr {np.frombuffer(we1[ELEM-8:ELEM-4].tobytes(), dtype=np.uint32)[0]}")
# o weight blocks: element 0..15 vs packed_o?  packed_o not dumped; but
# golden o_out was computed from W_o_dq -- verify blocks dequantize to
# W_o_dq rows 0:256 by re-deriving o_out per-block from fixture bytes.
def dequant_block(b, rows=16):
    nib = b[0:16384].reshape(1024, 16)   # 1024 nibble-pairs? use int4 hi/lo
    sf = b[16384:16640].view(np.uint16)  # 128 bf16 scales
    return nib, sf
b0 = Wfx[0:ELEM]
nib = np.zeros(2048, dtype=np.int8); lo = b0[0:2048]
# nibble layout: per group 32: 16 lo bytes then 16 hi (v5 ABI) -- infer:
nib[0:16] = (lo[0:16] & 0xF).astype(np.int8); nib[0:16][ (lo[0:16]&0xF) > 7 ] -= 16
print(f"  (o block 0 raw first bytes {b0[0:8].tolist()})")

# ---- full K-header profile of the 186-element W stream (worker 0) ----
hdrs = [int(np.frombuffer(Wfx[i*ELEM+ELEM-8:i*ELEM+ELEM-4].tobytes(), dtype=np.uint32)[0])
        for i in range(186)]
want = [2048]*16 + [101] + [103]*48 + [104]*48 + [105]*48 + [102] + [2048]*24
bad = [(i, hdrs[i], want[i]) for i in range(186) if hdrs[i] != want[i]]
print(f"\nK-header profile: {len(bad)} mismatches {bad[:10]}")
print(f"  w1@161 bits equal {np.array_equal(Wfx[161*ELEM:161*ELEM+4096].view(np.uint16), w1_bits)}")
# also verify worker 4 (position 7) profile
hdrs4 = [int(np.frombuffer(Wfx[(186*4+i)*ELEM+ELEM-8:(186*4+i)*ELEM+ELEM-4].tobytes(), dtype=np.uint32)[0])
         for i in range(186)]
bad4 = [(i, hdrs4[i], want[i]) for i in range(186) if hdrs4[i] != want[i]]
print(f"  worker 4: {len(bad4)} mismatches {bad4[:6]}")

# ---- x' echo: lv_shared slot-0 bits vs golden xp bits ----
print("\nx' echo (slot 0 bits) vs golden xp_bits[0:256] (position 0):")
xe = np.load("/tmp/p28/dbg_act_bits.npy")[384:640].view(np.uint16)
nb = int(np.sum(xe != xp_bits[0:256]))
print(f"  bit mismatches: {nb}/256")
if nb:
    idx = np.nonzero(xe != xp_bits[0:256])[0]
    for i in idx[:8]:
        print(f"    lane {i:3d}: act {_f32(np.array([xe[i]]))[0]:9.3f} "
              f"golden {_f32(np.array([xp_bits[i]]))[0]:9.3f}")

# ---- scale table + error accounting ----
print("\nscale table (medians of absolutes, golden chain):")
for name, v in [("o", o_f), ("x'", xp_r), ("xn_f", xn_f if 'xn_f' in dir() else xp_r*inv1g*w2f),
                ("gate g", g_f), ("up u", u_f)]:
    print(f"  {name:8s} {np.median(np.abs(v)):10.4f}  max {np.abs(v).max():10.2f}")
print(f"  d2 scale {np.median(_f32(d2g.astype(np.uint32))):8.4f}   d3 scale {np.median(_f32(d3.astype(np.uint32))):8.2f}")
print(f"  q3 |q| median {np.median(np.abs(q3.astype(np.int16))):.0f}")
print(f"  partial P medians {[f'{np.median(np.abs(Pf[c])):.0f}' for c in range(3)]}")
print(f"  dacc median {np.median(np.abs(dacc_gold)):.0f}")
# x' echo error magnitude (device vs golden) as fraction of o scale
xe_f = _f32(np.load("/tmp/p28/dbg_act_bits.npy")[384:640].view(np.uint16))
xn0 = _f32(xn_bits)  # position 0 chunk
o_impl = xe_f - xn0
o_err = o_impl - o_f[0:256]
print(f"\no error: median|err| {np.median(np.abs(o_err)):.4f} vs |o| median {np.median(np.abs(o_f[0:256])):.3f}"
      f" -> rel {np.median(np.abs(o_err))/np.median(np.abs(o_f[0:256]))*100:.2f}%")
print(f"  corr(o_err, o_golden) {np.corrcoef(o_err, o_f[0:256])[0,1]:+.3f}")

# ---- device-faithful chain: weights dequantized in f32 (no bf16 round) ----
print("\nf32-dequant (device-faithful) weight model, full chain:")
def deq_f32(W, gs=32):
    M, K = W.shape
    Wg = W.reshape(M*K//gs, gs)
    amax = np.abs(Wg).max(axis=1, keepdims=True)
    scale = _f32(_bf16((amax / 7.0).astype(np.float32)).astype(np.uint32))
    q = np.round(Wg / scale)
    q = np.clip(q, -8, 7)
    return (q * scale).reshape(M, K).astype(np.float32)

W_o_f, W_g_f = deq_f32(W_o), deq_f32(W_g)
W_u_f, W_d_f, W_q_f = deq_f32(W_u), deq_f32(W_d), deq_f32(W_q)
o32_bits = _bf16(W_o_f @ x_deq1.numpy().astype(np.float32))
xp32_bits = _bf16(_f32(o32_bits.astype(np.uint32)) + xn_f32)
xp32 = _f32(xp32_bits.astype(np.uint32))
inv_32 = _kernel_rsqrt(np.float32(_sumsq_sequential(xp32) / np.float32(2048) + np.float32(1e-5)))
q2_32, d2_32 = _quantize_kernel(xp32 * inv_32 * w2f, 2048)
x2_32 = _dequant(q2_32, d2_32)
g32 = _bf16(W_g_f @ x2_32); u32 = _bf16(W_u_f @ x2_32)
g32f = _f32(g32.astype(np.uint32)); u32f = _f32(u32.astype(np.uint32))
sig32 = np.float32(1.0) / (np.float32(1.0) + np.exp(-g32f.astype(np.float32)))
sw32 = _bf16(g32f * sig32 * u32f)
q3_32, d3_32 = _quantize_kernel(_f32(sw32.astype(np.uint32)), 6144)
x3_32 = _dequant(q3_32, d3_32)
P32 = np.stack([_bf16(W_d_f[:, c*2048:(c+1)*2048] @ x3_32[c*2048:(c+1)*2048]) for c in range(3)])
dacc32 = _f32(P32.view if False else np.stack([P32[c] for c in range(3)]).astype(np.uint32)).sum(axis=0) if False else _f32(np.stack(P32).astype(np.uint32)).sum(axis=0)
xn1_32 = _bf16(xp32 + dacc32)
inv2_32 = _kernel_rsqrt(np.float32(_sumsq_sequential(_f32(xn1_32.astype(np.uint32))) / np.float32(2048) + np.float32(1e-5)))
q4_32, d4_32 = _quantize_kernel(_f32(xn1_32.astype(np.uint32)) * inv2_32 * w1f, 2048)
qkv32 = _bf16(W_q_f @ _dequant(q4_32, d4_32))
# compare: f32-chain vs GOLDEN chain (both are models; device should
# match the f32 one)
xg = np.load("/tmp/p28/dbg_exp_bits.npy")
gf = _f32(xg)
print("  f32-chain vs golden-model (per worker, qkv/xn1 maxdiff):")
for w in (0, 1, 4):
    p = pos(w)
    base = w*640
    dq = np.abs(_f32(qkv32.astype(np.uint32))[p*384:(p+1)*384] - gf[base:base+384]).max()
    dx = np.abs(_f32(xn1_32.astype(np.uint32))[p*256:(p+1)*256] - gf[base+384:base+640]).max()
    print(f"    w{w}: qkv {dq:7.1f}  xn1 {dx:8.1f}")

# ---- x' echo vs f32-dequant model ----
xe2 = np.load("/tmp/p28/dbg_act_bits.npy")[384:640].view(np.uint16)
print(f"\nx' echo vs f32-model xp32 bits: {int(np.sum(xe2 != xp32_bits[0:256]))}/256 mismatches")
print(f"x' echo vs golden   xp bits: 101/256 (measured before)")

# ---- model-vs-model flip rates on the echoed regions ----
rep32 = np.zeros(512, dtype=np.int8)
for g in range(4):
    xg = q2_32[g*32:(g+1)*32]
    rep32[g*128:g*128+64] = np.tile(xg[0:16], 4)
    rep32[g*128+64:g*128+128] = np.tile(xg[16:32], 4)
repg = np.zeros(512, dtype=np.int8)
for g in range(4):
    xg = q2g[g*32:(g+1)*32]
    repg[g*128:g*128+64] = np.tile(xg[0:16], 4)
    repg[g*128+64:g*128+128] = np.tile(xg[16:32], 4)
print(f"\narena2 region: f32-q2 vs golden-q2 flips = {int(np.sum(rep32 != repg))}/512"
      f"  (device-vs-golden was 44/512)")
print(f"sw region: f32-q3 vs golden-q3 flips = {int(np.sum(q3_32[:512] != q3[:512]))}/512"
      f"  (device-vs-golden was 82/512)")

# ---- d3 scales echo: lv_sw[6144:6528) vs golden/f32 d3 bits ----
se = np.load("/tmp/p28/dbg_act_bits.npy")[384:640].view(np.uint16)  # 256 u16
echo_scales = se[:192]  # 192 scale slots (global group order)
d3_gold_bits = d3.astype(np.uint16)
d3_32_bits = d3_32.astype(np.uint16)
nb_g = int(np.sum(echo_scales != d3_gold_bits))
nb_m = int(np.sum(echo_scales != d3_32_bits))
print(f"\nd3 scale echo: vs golden {nb_g}/192, vs f32-model {nb_m}/192 bit mismatches")
if nb_m or nb_g:
    idx = np.nonzero(echo_scales != d3_32_bits)[0]
    for i in idx[:10]:
        print(f"    scale g={i:3d}: echo {_f32(np.array([echo_scales[i]]))[0]:9.4f}"
              f"  f32 {_f32(np.array([d3_32_bits[i]]))[0]:9.4f}"
              f"  golden {_f32(np.array([d3_gold_bits[i]]))[0]:9.4f}")

# ---- full scale permutation map ----
print("\nscale echo permutation map (echo[g] == f32[g']?):")
mism = np.nonzero(echo_scales != d3_32_bits)[0]
print(f"  mismatched g: {mism.tolist()}")
for g in mism:
    ev = echo_scales[g]
    src = np.nonzero(d3_32_bits == ev)[0]
    srcs = src.tolist() if len(src) < 8 else f"{len(src)} candidates"
    fv = _f32(np.array([d3_32_bits[g]]))[0]
    print(f"    g={g:3d}: echo {ev:04x} src-g {srcs}   (f32[g]={d3_32_bits[g]:04x})")
