# MHA d=128 数据损坏调查日志（已修复）

日期: 2026-09-22（第 4 会话，E2E 乱码定案；第 3 会话 σ 修复；第 2 会话结论修正）
项目: ~/qwen/xnpu/IRON — MiniCPM5-2B (42层, d=128) XDNA2 NPU 部署, R2b 阶段
结果: `test_mha[iter*-mha_2048_128_16_8_0]` **5/5 PASSED**（修复前 2,926,954 错误 / 59%）

## 0. 最终定案 (TL;DR)

1. **σ 只在 O 下行（join 写路径）**：偶数源列组交付两次、奇数组丢失的确定性置换
   `m2(n) = 16·((n//8) mod 8) + (n mod 8)` 是 **O-join 的 o_dims BD（MemTile col 6/7，
   各 4 条，D2=(wrap16, step4w)）被硬件执行成 (wrap8, step8w)** 造成的。
   Q/K/V 上行 MM2S BD 携带相同 D2 却执行**正确**（第 2 会话"MM2S 读侧失效"结论被
   奇组判别探针推翻）。
2. **判别探针** `/tmp/probe_oddgrp.py`：Q/K 质量全放奇数 d 组（col 9）→ P 仍精确
   one-hot ⇒ Q/K 交付完好。（注意：one-hot 点积探针对"Q/K 收缩维**相同**置换"天然
   盲——置换在点积中抵消；奇组探针只能证明一致性，正确性靠本探针的 group 级判定。）
3. **修复**（输入路径零改动）：
   - `design.py`: `vectorized and d > 64 → o_dims = None`（O 线性下行，其余 dims 不动）；
   - `op.py`: `read_buffer` 覆写对 "O" 施加 C-tile→行主序反摆序
     `reshape(H,S//64,8,d//8,8,8).transpose(0,1,2,4,3,5)`（对 buffer 级 harness
     `run_test` 也生效——它绕过 forward()，这正是第一版修复"探针过了测试仍挂"的原因）。
4. C-tile 块内四轴 (z:8 行组, j:d/8 列组, k:8 组内行, e:8)，flat = 1024z+64j+8k+e
   （rescale 的 `l*8*DIM_D + j*64 + k*8`）。反摆序置换已离线数值验证 = identity。
5. ~~d=64 多头配置在原版上游同样失败~~ **（第 4 会话推翻）**：那次 stash 验证只换
   op.py 源码、不触发 .o 缓存失效，跑的还是陈旧 objects（见 §8）。清缓存重测
   **25/25 PASSED**——上游 d=64 从未坏过。
6. 残留小项：one-hot 奇组探针里首 8 行内 ~4 行 score 配对异常（P 均匀化/互换，
   如 head3 j0=3 的 rows 1,2,4,5）；随机数据下不可见（max err 0.038），待查不阻塞。

## 1. XDNA2 tiling 编程模型（保留，复用价值高）

### 1.1 硬件背景 (Estévez 2026-05 博客 + llvm-aie)
- XDNA2 (Strix Halo 17f0:11) = `aie2p` (≈AIE-MLv2): 8 列 × (1 MemTile 行 + 4 计算行)。
- 计算核: VLIW SIMD, 1.8 GHz, 12×512b x 寄存器, **aie2p 只有 5 个累加器** (vs aie2ps 8)。
- MemTile 512KB; 计算核 64KB 本地内存 (地址 0x70000-0x80000; 西邻 0x50000, 北 0x60000, 南 0x40000)。
- 三种互连: AXI4-Stream 全阵交换网 (主数据通路, 静态电路交换), 邻居内存直访, 级联。
- shimNOC (行 0) 的 shimDMA 负责 host DDR ↔ 阵列, 经 AXI-S。
- DMA (shim/MemTile/核三级) 均支持 4D 步进寻址 —— tiling 的硬件基础。

### 1.2 BD (Buffer Descriptor) 解剖 (MemTile DMA BD, aie2p)
寄存器 8×32b (xaie2pgbl_params.h @ ~L20800, 块写命令 `[0x111][0x000a0105][0][addr][w0..w7]`):
- w0: LENGTH[16:0] (32b 字数)  w1: BASE_ADDR[18:0]
- w2/w3/w4: D0/D1/D2 各含 WRAP[26:17](10b) + STEPSIZE[16:0](17b)
- w5: D3 **只有 STEPSIZE** (无 wrap — 外层计数由 LENGTH 隐式耗尽)
- w6: ITERATION_WRAP[22:17](**6b**) + ITER_STEPSIZE[16:0] (第五槽!)
- w7: VALID_BD[31] + 4 个锁字段
- **语义** (aie-rt 打包规则): wrap 字段写原值(计数), step 字段写 (步长−1), 步长单位=32b字;
  ITER 的 wrap/step 也写 (值−1)。

### 1.3 工具链路径 (dims → BD 位)
```
design.py dims=[(size,stride),…]  (列表序 = 外层→内层, 最内 stride=1)
  → IRON objectfifo(dims_to_stream=…)
  → MLIR aie.objectfifo / aiex.dma_configure_task_for(aie.dma_bd dims=[…])
  → AIERT.cpp: NumDim=dims.size(); Dim[j={N-1-i}] = {stride×width/4, size}  (j>0)
     最内维: size×width/4 (字数), stride 原样
  → aie-rt XAie_DmaSetMultiDimAddr → _XAieMl_DmaSetMultiDim (校验上限) → BD 字
  → CDO 块写 / 运行时控制序列
```
- **NumAddrDim = 4**: objectfifo dims ≤ 4 维; ITER 槽只能经 XAie_DmaSetBdIteration 单独设置。
- shim DMA BD 不在静态 CDO 里 (本设计), 由 runtime sequence 动态编程。

### 1.4 本设计 (mha d=128) 的数据链与 BD 落点（实测 CDO 直方图）
- 静态 CDO 带 dims 的 BD 分布: col 3 (memK) ×1, col 4 (memV) ×1,
  col 6/7 (inQ split 4 条 + outO join 4 条) ×8 —— **共 10 条 len=4096w**
  `D0(w4,s0) D1(w8,s63) D2(w16,s3) D3s=511`; 另 cols 0-2 有 a/p_dims BD（全 wrap 8, 安全）。
- **只有 col 6/7 中属于 O-join 的 4+4 条真正失效**；K/V/Q 的 MM2S 同编码却正确。
  ⇒ 失效条件不是"D2 wrap=16"本身，而是 join 写路径（S2MM 散射写或该侧地址生成器）。
  具体落在哪一侧（MemTile S2MM vs 核侧 ELF 内 BD）未再细分——修复已绕开，不再需要。
- shim 读写 host: K/V 线性 (262144, stride 1); Q/O 对象 (256 行 × 128, stride 128/1)。
  shim DMA 的 (256-count) 维在 d=64 位精确 ⇒ shim 侧干净。

### 1.5 内核侧布局 (mm.cc 2x2 mmul, PV 例化 <64,64,128>, r=s=t=8)
- A/B/C 走读全部与 DMA dest 序一致（设计自洽，上游原版未改）。
- PV 的 B 块内是 kv 行主序 ⇒ e 只能 8 连续、c 步长=128。
- **C-tile（O）块内布局 = (z,j,k,e) 四轴**，本修复的反摆序即按此写成。

## 2. 决定性测量 (方法可复用)

- one-hot K + 行盲 V2=n ⇒ O2[n] = bf16_even((255/256)·m(n))，全 128 列解出映射 m。
- **方向判别**（O 侧 vs V 侧 σ 对此类探针不可分——两者预测同形）：
  /tmp/probe_oddgrp.py 把 Q/K 质量放奇数 d 组 ⇒ P 是否仍 one-hot 直接判定上行交付。
- **harness 陷阱**：`run_test`/`verify_buffer` 绕过 `forward()` 直接读写 buffer——
  任何放在 forward/_execute 里的输出后处理对 pytest 无效。放 `read_buffer` 覆写。
- bf16(255c/256) 求逆: c∈[1,2) 精确; 更大按 8-bit mantissa 网格最近邻（.5 边界伪差,
  如 n=124）。
- 探针 → npy → 离线 numpy 求逆/置换验证（不上 NPU）；用 arange 构造置换做数值反演。

## 3. 已排除项 (累计)

1. softmax.cc / of_depth / 陈旧归档 / 编译器 stride 换算 — 第 1 会话。
2. CDO BD 位编码错误 — 二次解码逐位一致。
3. mm.cc PV 例化 B/C 走读越界或错序 — 手工推导自洽。
4. mha.cc DIM_D 泛化 — 与 C 布局一致。
5. host 侧 shim 读法 — K/V 线性读。
6. "内核越界读杂散内存" — softmax 残差解释取代。
7. **"MemTile MM2S 读侧 D2(w16) 失效"（第 2 会话结论）— 被奇组探针推翻**：
   Q/K/V 上行完好；σ 在 O-join 写路径。

## 4. 根因 (定案)

**O-join 的 o_dims BD（4+4 条, MemTile col 6/7, D2=(wrap16,step4w)）被硬件执行为
(wrap8, step8w)** — stride×2、wrap÷2、LENGTH 守恒 ⇒ 偶数源列组交付两次、奇数组丢失。
相同编码的 K/V/Q MM2S BD 执行正确 ⇒ 触发条件与方向/路径相关（join 写侧），非单纯 wrap>8。
位流编码经解码器复核无误 ⇒ 属硬件执行层（或未文档化约束）。上游求证项照旧（不阻塞）。

## 5. 修复 (已实施并验证)

- `design.py` o_dims 处: `if vectorized and d > 64: o_dims = None`（O 线性下行）。
- `op.py`: `_unswizzle_c_tiles()` + `AIEMHA.read_buffer` 覆写（"O" 且 d>64 时）。
  q_dims/k_dims/v_dims/p_dims/a_dims、内核、mha.cc **全部不动**。
- 验证链: ① 线性流实测 = 正确 C-tile（反演 wrong-helper 后 identity）；
  ② one-hot 探针 m2 = identity（唯 n=124 求逆伪差）；③ 随机数据 max err 0.038
  （bf16 数值噪声级）；④ pytest 2048_128 **5/5 PASSED**。
- 修复代价: O 下行失去 dims 步进（shim BD 本来就是 (256,128) 行维, 无性能变化）;
  host 侧一次 (8,16,8,8) 转置拷贝（4.2M bf16 ≈ 8MB, μs 级）。

## 6. 下一步

1. ~~修复 d=128~~ ✅ 完成。
2. `configs/minicpm5_2b.json` 打开 use_aie_fused_mha，对比 CPU 基线
   (prefill 6.29s / decode 2.06 tok/s)。吞吐不足时考虑 int8/q8 tile 量化（用户备忘）。
3. 残留: 首 8 行 score 配对小异常（§0.6；也可能同样是缓存陈旧假象，待复测）。
4. 向上游求证 join 写路径 D2 约束 (mlir-aie issue) — 解释性收尾。
5. R3: 42 层 decode 正确性 vs GGUF@8080; Phase M (Rust) M0。

## 7. 方法论笔记

- CDO 解码器 /tmp/decode_bd.py; 按 addr 直方图定位 BD 属主（col = addr>>25 …见 §1.4）。
- 静态 CDO 只含 MemTile BD; shim BD 在 build/*.mlir.prj/input_with_addresses.mlir。
- 探针设计要点: ①点积对相同置换盲 ②one-hot 对列映射盲 ③harness 路径要核对。
- 参考材料: docs.amd.com UG1603 (JS-only); destevez.net 2026-05 (aie2p 架构/DMA);
  ironenv xaie2pgbl_params.h; aie-rt xaie_dma*.c。

---

# 第 4 会话：use_aie_fused_mha E2E 乱码调查（已定案 + 已修复）

日期: 2026-09-22 晚。现象: 配置打开 `use_aie_fused_mha` 后 42 层 prefill/decode 全乱码；
关闭则正常。探针（单 AIEMHA op，同维度同输入）却全对。**同输入跨进程结果不同。**

## 8. 根因（定案）：IRON 构建缓存不感知编译 flags → 陈旧 QK matmul 进 xclbin

- `compilation.py` 的可用性判定只比较**源文件 mtime**；`KernelObjectArtifact` 的
  `extra_flags`（`-DDIM_*`/`-DB_COL_MAJ` 例化）与 `rename_symbols` 不参与。
  op.py flags 变更后，build 目录里的 `mha_mm.o`（源 mm.cc 未动）被判定"有效"照抄进
  archive → 链进 xclbin → **QK matmul 用旧例化跑新布局** → 分数平坦/错误 → softmax
  均匀 → O = V 的因果均匀均值（每个输出 ≈ 前 s+1 个 V 的平均）。
- **build 目录跟随进程 CWD**（`AieOperatorBase.get_default_context()` 用相对路径
  `build/`）。三棵树：`xnpu/build`、`IRON/build`、`app/build` 各自缓存——
  "app 跑坏、探针跑好"的"间歇性"假象就是两棵树状态不同（探针 cwd=xnpu 是新建树，
  app cwd=app 目录里有移植中期的陈旧 .o）。**同一输入跨进程结果不同 → 先查 artifact
  缓存树，再怀疑硬件。**
- 证据链（自顶向下 256 字节逐级定位）: 坏 xclbin 127999B vs 好 128255B → PDI −256B
  → `cdo_elfs` −256B → 8× row-2 核 ELF 各 −32B（QK matmul worker）→ app/build
  `mha_mm.o` 陈旧（`.text` 592 vs 624 字节，mtime 12:14 vs 最终 flags 15:25）→
  aiecc 对同一输入确定性（3 次重跑 byte 一致，排除编译器随机）→ 时间线吻合
  （12:14 = 移植中期 flags；15:25 后 flags 定稿但 mm.cc 未再改）。

## 9. 修复：KernelObjectArtifact build-key 边车

`iron/common/compilation.py`：
- 新增 `KernelObjectArtifact._build_key()` = sha256(sorted(extra_flags) +
  sorted(rename_symbols))[:16]；`is_available()` 额外要求边车 `*.o.buildkey` 存在且
  与当前 key 一致（老缓存无边车 → 必然失效）。
- `PeanoCompilationRule.compile()` 成功后写边车。
- 共享名陷阱：`mha_mm.o` 等 kernel object **不带维度后缀**，d=64/d=128 同名仅靠
  flags 区分——正是本 bug 温床；修复后 flags 变更自动重编（d=64↔d=128 切换安全）。

## 10. 验证（app 目录真实场景，从零重建）

- `sudo rm` app/build xclbin 后从 app 目录跑 E2E：`mha_mm.o` 17:43 重编（2976B =
  好指纹；坏件 2928B）+ 边车落盘 + xclbin 17:45 重建 128255B（好指纹）。
- 层误差（capture 层 0/1/21/41 vs torch 参考）: 0.0002 / 0.0012 / 0.0042 / 0.0078
  （|ref| 0.016/0.037/0.271/0.434，bf16 噪声级）。
- 24 token 生成：李尔王原文精确续写，prefill 6.63s / decode 2.15 tok/s。
- **d=64 上游重测**（stash mha 四件套 + 擦 IRON/build/mha_* + 边车保护）:
  25/25 PASSED（307s）。§0.5 的"上游也坏"作废。

## 11. 方法论教训

1. **跨进程不一致 = 先查缓存树**。本会话在"内核数学/流水线/多实例/交错 op"上排除了
   一大片（探针 probe_repeat/two_instances/mixed_ops 全对），因为探针和受害路径用的
   根本是不同的 build 目录。应该第一步就 `ls -la` 三棵 build 树比对 mtime/尺寸。
2. **指纹优先**：xclbin/PDI/ELF/`.o` 的字节数是快速判别——差 256B/32B 直接指向哪个核、
   哪个对象变了。
3. **探针 Δ 是 log2 单位**：softmax 内核用 exp2，INV_SCALE = bf16(1/ln2)/√D。
   K = Δ/INV_SCALE → scaled = Δ → P = 2^Δ（非 e^Δ）。C{31:1} 单热真值 = 2/33 =
   0.0606（不是 e 底算的 0.0610）。参考值先对齐底数再谈"异常"。
4. **归一化盲区**：chunk 权重投影归一化后，"l 也坏"与"只有 O-rescale 坏"都给精确
   1/32——需非归一化信息才能区分。幸好没用它下结论。
5. harness 双路径：`run_test` 绕过 `forward()`（第 3 会话已知），read_buffer 覆写
   对两者都生效。

## 12. 状态

- R2b 完整闭环：σ 置换（会话 3）+ 缓存加固（本会话）→ E2E 数值与生成全部正常。
- 上游可提事项：①compilation.py flags 不入 key 的 bug（已本地修，可提 PR）；
  ②O-join 写路径 D2(w16→w8) 硬件约束求证（§4，解释性）。
- 下一步: R3 42 层 decode 正确性 vs GGUF@8080；Phase M (Rust) M0；吞吐不足时
  考虑 int8/q8 tile 量化（用户备忘）。

---

# 第 5 节：R3 — 42 层 decode 正确性对拍（2026-09-22 晚，通过）

方法：同 app 双配置对拍——`minicpm5_2b.json`(NPU 算子) vs `minicpm5_2b_cpu.json`
(全部 use_aie_*=false, 纯 torch CPU)。同 tokenizer、同 prompt(2048 tok 截断)、
greedy(temperature=0 → argmax; inference.py 新增 --temperature/--top_k 透传)。
hook `Llama3ModelWithJSONConfig.forward` 抓每步 last-position logits
(/tmp/r3_npu_logits.py, R3_TAG/R3_PROMPT_FILE 参数化)。

## 结果（两 prompt：李尔王背诵文 / 中文硬件笔记）

- **token 一致性**：NPU vs CPU 两 prompt 均**前 6 token 完全一致**(top1 6/6)，
  第 6 步分叉均落在近平局：prompt1 是 CPU 侧**精确平局**(oath/breach 同 9.8125，
  argmax 靠索引序破平)；prompt2 top-gap=0.125(12-13 处 1-2 个 bf16 ULP)。
- **logits 偏差**：rms Δ≈0.26-0.52（|logit| rms 3-4），max 2-3；**漂移平坦不增长**
  （step0 prefill 即 0.33-0.52，之后恒定 ~0.3）——每 forward 常量级噪声，非累积。
  与每算子 ~1-2% bf16 相对误差 × 42 层传播一致（pytest rel_tol 4% 同源）。
- **量化参照**：GGUF(8080, 量化版) vs CPU bf16 第 **2** token 即分叉——
  NPU 栈(第 6)比量化参考更贴 CPU。CPU 配置自身也复读退化（两 prompt 均同款
  循环），复读非 NPU 问题。
- 分叉后两配置续写质量相当（prompt2 NPU 甚至更连贯）。

## 结论

**R3 PASS**：bf16 NPU 栈 decode 与 CPU 参考 top-1 保持一致直至近平局
（gap ≤0.125 = 栈噪声 0.3 之内），logits 偏差恒定 ~0.3 rms。token 级精确匹配
不是 bf16 栈的正确性标准（CPU 自身在平局处靠索引序）。

## 性能附记（本次顺带测得）

| 配置 | prefill | decode |
|---|---|---|
| NPU 算子全开 | 6.5s | **2.26 tok/s** |
| 全 CPU | 8.0s | **4.55 tok/s** |

decode NPU 反而慢 2×：42 层 × ~8 算子/层 的逐算子同步开销主导（非算力）。
R2 基线(CPU attention+其余 NPU) decode 2.06 同理。后续要么层间融合/批量调度，
要么 Phase M 引擎接管；吞吐不足时上用户提的 int8/q8 tile 量化。

工具：/tmp/r3_npu_logits.py（logits 捕获 harness）、/tmp/r3_cmp.py（GGUF 对拍）、
configs/minicpm5_2b_cpu.json（A/B 参考配置，已入库）。
