#!/usr/bin/env python3
"""P28-7 glue-floor discriminator pack: rewrite every W element's K
header to 2049 (the kernel's PERF-ONLY dummy-compute flavor) so an exec
runs the full A-fifo fill + lv_compute stream + C drains with ZERO glue
phases (no rms/quant/arena/swiglu/gathers on the K path -- the design's
ring rendezvous still run in the real design; pair this with the r0
probe if you want those out too).

T(full pack) - T(floor pack) = the glue exposure the 6f-9 duty-cycle
model predicted (~250us/exec) -- the number that was never isolated
before, because every earlier probe kept the K chain intact.

Usage: lv2_floor_pack.py <in.bin> <out.bin> [n]   (n = ring width, 8/16)
"""

import sys
import numpy as np

ELEM = 18560
N_WELEM = {8: 186, 16: 94}


def main():
    src_path, dst_path = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    w = np.fromfile(src_path, dtype=np.uint8)
    expect = n * N_WELEM[n] * ELEM
    assert w.size == expect, f"{src_path}: {w.size} B != {expect} (stale pack?)"
    hdr = np.frombuffer(np.uint32(2049).tobytes(), dtype=np.uint8)
    for wk in range(n):
        base = wk * N_WELEM[n] * ELEM
        for e in range(N_WELEM[n]):
            off = base + e * ELEM + ELEM - 8  # K header @ element tail - 8
            w[off : off + 4] = hdr
    w.tofile(dst_path)
    print(f"{dst_path}: {n} x {N_WELEM[n]} K headers -> 2049 ({w.size} B)")


if __name__ == "__main__":
    main()
