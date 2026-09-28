#!/usr/bin/env python3
"""Decode an aie2 transaction ctrl-code blob (IRON aiebu / FLM npu_sequence).

Instruction format verified against BOTH:
  - IRON compiled bin (w4gemvuq...3072.bin, 17488B) + its npu_insts.mlir
  - FLM open headers npu_cmd_*.hpp (to_npu encoders)
Format law: every instruction is a whole number of 32-bit words; the word
`op_size << 2` = TOTAL instruction bytes appears at a per-op position:
  WRITE(0)=6w [op,0,addr,0,value,size]         size@w5
  BLOCKWRITE(1)/BLOCKSET(2)=[op,0,addr,size,payload...]  size@w3, payload=(size/4-4)w
  MASKWRITE(3)=7w [op,0,addr,0,value,mask,size] size@w6
  MASKPOLL(4)=5w? [op,0,addr,mask,...]         size@w4 (last)
  TCT(0x80)=4w [op,size,row/col/dir,ch|0x10100] size@w1
  DDR_PATCH(0x81)=12w [op,size,0,0,0,0,bdaddr,0,argidx,0,argoff,0] size@w1
BD space: BLOCKWRITE to (addr&0xFFFFF) in [0x1D000,0x1D200), bd_id=((a&0xFFFFF)-0x1D000)>>5
          (BD stride 0x20B = the 8-word payload)
Queue push: WRITE to reg in [0x1D200,0x1D400): S2MM=0x1D200+8*ch, MM2S=+0x10;
          value=bd_id|repeat<<16|issue_token<<31  (npu_cmd_write.hpp)
Usage: ctrl_decode.py <blob.bin> [-v N] [--bd] [--kicks]
"""
import sys, struct
from collections import Counter

OPS = {0:"WRITE",1:"BLOCKWRITE",2:"BLOCKSET",3:"MASKWRITE",4:"MASKPOLL",5:"NOOP",
       6:"PREEMPT",7:"MASKPOLL_BUSY",8:"LOADPDI",13:"CFG_SHIMDMA_BD",
       14:"CFG_SHIMDMA_DMABUF_BD",0x80:"TCT",0x81:"DDR_PATCH",0x82:"READ_REGS",
       0x83:"RECORD_TIMER",0x84:"MERGE_SYNC",0x85:"NEXT"}

def u32(b,i): return struct.unpack_from("<I",b,i)[0]

def decode_bd(payload):
    """8-word shim BD payload per npu_cmd_write_dma.hpp dump_cmd."""
    w = payload  # [len, off, pkt, D0, D1, D2, iter, lock/next] as w4..w11 of instr
    d = {}
    d["len"] = w[0]; d["boff"] = w[1]
    pkt = w[2]
    d["pkt"] = None if not (pkt>>30)&1 else f"id{(pkt>>19)&0x1f}"
    d0 = w[3]
    d["d0"] = "linear" if d0==0 else f"2d(sz{(d0>>20)&0x3ff},st0x{d0&0xfffff:x})"
    d["d1"] = f"sz{(w[4]>>20)&0x3ff},st0x{w[4]&0xfffff:x}" if w[4] else ""
    d["d2"] = f"sz{(w[5]>>20)&0x3ff},st0x{w[5]&0xfffff:x}" if w[5] else ""
    it = w[6]
    d["iter"] = f"{(it>>20)&0x3ff}x st0x{it&0xfffff:x}" if it else ""
    lk = w[7]
    nb = (lk>>27)&0xf
    d["nextbd"] = nb if (lk>>26)&1 else None
    d["lock"] = ""
    if (lk>>12)&1: d["lock"] += f"acq{(lk>>0)&0xf}={(lk>>5)&0xef}"
    if (lk>>13)&0xf or (lk>>18)&0xef:
        d["lock"] += f" rel{(lk>>13)&0xf}={(lk>>18)&0xef}"
    return d

def decode(blob, verbose=0):
    magic,h1,h2,h3 = struct.unpack_from("<4I",blob,0)
    print(f"hdr: magic=0x{magic:08x} w1={h1} w2={h2} w3={h3} (blob={len(blob)}B)")
    total = h3 if 0 < h3 <= len(blob) else len(blob)
    off = 16; n = 0
    hist = Counter(); instrs = []
    while off + 4 <= total:
        w0 = u32(blob,off); op = w0 & 0xff
        # size in bytes, read from the per-op size word
        try:
            if op in (1,2): sz = u32(blob,off+12)
            elif op in (0x80,0x81,0x82,0x83,0x84,0x85,4): sz = u32(blob,off+4)
            elif op == 0: sz = u32(blob,off+20)
            elif op == 3: sz = u32(blob,off+24)
            else: sz = 16
        except struct.error:
            print(f"  off=0x{off:x} read past end"); break
        if sz == 0 or sz % 4 or sz > total - off:
            print(f"  off=0x{off:x} bad size {sz} for op {OPS.get(op,hex(op))} — stop")
            break
        words = [u32(blob,off+4*i) for i in range(sz//4)]
        instrs.append((off,op,words)); hist[OPS.get(op,f"?{op}")] += 1
        off += sz; n += 1
        if verbose and n <= verbose:
            print(f"  #{n} off=0x{instrs[-1][0]:x} {OPS.get(op,hex(op))} {words}")
    print(f"instrs: {n}, walked to 0x{off:x} / total 0x{total:x} {'OK' if off==total else 'MISALIGN'}")
    print("histogram:", dict(hist))
    return instrs

def semantic(instrs):
    """Reduce raw instrs to the choreography: BD fills, DDR patches, queue pushes, waits."""
    bd_fills = []; patches = []; pushes = []; waits = []; polls = []; writes = []
    for off,op,w in instrs:
        if op == 1 and (w[2] & 0xFFFFF) >= 0x1D000 and (w[2] & 0xFFFFF) < 0x1D200:
            addr = w[2]; bd_id = ((addr & 0xFFFFF) - 0x1D000) >> 5
            bd_fills.append(dict(off=off, col=addr>>25, row=(addr>>20)&0x1f, bd=bd_id,
                                 **decode_bd(w[4:])))
        elif op == 1:
            bd_fills.append(dict(off=off, col=w[2]>>25, row=(w[2]>>20)&0x1f,
                                 bd="MEM", addr=w[2], len=w[4] if len(w)>4 else 0))
        elif op == 0x81:
            patches.append(dict(off=off, bd_reg=w[6], arg=w[8], argoff=w[10]))
        elif op == 0:
            reg = w[2] & 0xFFFFF
            if 0x1D200 <= reg < 0x1D400:
                direction = "MM2S" if reg & 0x10 else "S2MM"
                pushes.append(dict(off=off, col=w[2]>>25, row=(w[2]>>20)&0x1f, dir=direction,
                                   ch=(reg>>3)&1, bd=w[4]&0xf, rep=(w[4]>>16)&0xff,
                                   tok=(w[4]>>31)&1))
            else:
                writes.append(dict(off=off, addr=w[2], val=w[4]))
        elif op == 3:
            reg = w[2] & 0xFFFFF
            if 0x1D200 <= reg < 0x1D400:
                pushes.append(dict(off=off, col=w[2]>>25, row=(w[2]>>20)&0x1f, dir="TOKEN?",
                                   ch=(reg>>3)&1, val=w[4], mask=w[5]))
            else:
                writes.append(dict(off=off, addr=w[2], val=w[4], mask=w[5]))
        elif op == 4:
            polls.append(dict(off=off, addr=w[2], mask=w[3] if len(w)>3 else 0))
        elif op == 0x80:
            waits.append(dict(off=off, w2=w[2], w3=w[3]))
    return bd_fills, patches, pushes, waits, polls, writes

if __name__ == "__main__":
    path = sys.argv[1]
    verbose = int(sys.argv[sys.argv.index("-v")+1]) if "-v" in sys.argv else 0
    blob = open(path,"rb").read()
    instrs = decode(blob, verbose)
    bd,pat,push,wait,poll,wr = semantic(instrs)
    print(f"\nBD fills: {len(bd)}")
    for b in bd[:12]:
        print(f"  off=0x{b['off']:x} (c{b['col']},r{b['row']}) bd{b['bd']}: len={b.get('len')} boff={b.get("boff")} {b.get('d0','')} {b.get('d1','')} {b.get('d2','')} iter={b.get('iter','')} nextbd={b.get('nextbd')} lock={b.get('lock','')}")
    print(f"DDR patches: {len(pat)} (arg histogram: {dict(Counter(p['arg'] for p in pat))})")
    for p in pat[:6]: print(f"  off=0x{p['off']:x} bd_reg=0x{p['bd_reg']:x} arg={p['arg']} +0x{p['argoff']:x}")
    print(f"queue pushes: {len(push)}")
    for p in push[:16]: print(f"  off=0x{p['off']:x} (c{p['col']},r{p['row']}) {p.get('dir')} ch{p.get('ch')} bd={p.get('bd')} rep={p.get('rep')} tok={p.get('tok')}")
    print(f"TCT waits: {len(wait)}  MASKPOLLs: {len(poll)}  other writes: {len(wr)}")
    for w_ in wr[:10]: print(f"  off=0x{w_['off']:x} write 0x{w_['addr']:08x} = 0x{w_['val']:x}")
    # per-column push summary
    pc = Counter((p['col'], p.get('dir')) for p in push)
    print("push by (col,dir):", dict(sorted(pc.items())))
