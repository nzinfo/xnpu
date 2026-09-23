# NPU 性能实验笔记（perf-lab）

纪律：设计思路、尝试步骤、成败都留痕，编号 P1/P2/…。目标是**通用的 NPU
性能分析模型/机制/框架**（用户指令：这比任何单个推理模型的优化更重要）。
机制背景见 rust-drm-port-log.md（§12 度量方法学、§17 时间语义）。

---

## P1（2026-09-23）xnpu-perf crate + run-w4ulayer 改造

### 设计思路

**为什么设备无关核**：M0–M4a 的所有性能知识落在一小组常数上（CU 切换
~650µs、submit 往返 ~55µs、slot 流带宽上限）。把它们隔离进
`MachineModel`，核心只做「时间线记录 + 字节/FLOP 计数 + roofline 算术」，
换一台 NPU 只需重标定常数——框架就能覆盖通用机制而不是绑死 npu2。

**三条设计原则**（lib.rs 文档注释为准）：
1. 每条测量自带分母：报 GB/s / GFLOP/s 必须同时报占天花板比例
   （TileLang Analyzer 约定；数字没有分母不可比较）。
2. 三模式方法学：solo（submit+wait 墙钟上界）/ burst（同 op 连发，drain/n
   逼近纯设备时间）/ 链式 gap（墙钟 − 串行估计 = 调度开销或流水收益）。
3. 时间语义按 notes §17：t_complete = syncobj timeline wait 返回时刻；
   state-poll 不作为完成依据（pipelined 模式沿用 M3b 的 state drain +
   verify，是已知妥协，见下）。

**结构**（零外部依赖，手写 JSON）：
- `OpMeta`：每算子静态形状——有用字节（roofline 分子）+ `bytes_stream`
  替代口径（w4 padded slot 流）+ FLOP。
- `Recorder`：只记原始事实（solo/burst 事件 + 块 drain 标记），分析全在
  report 层。
- `MachineModel`：bw_stream_gbps / peak_gflops(None=未知) /
  cu_switch_reload_us / submit_overhead_us + provenance 字符串（数字不许
  裸奔）。`overlay_json` 支持校准文件覆盖（perf-calibrate 的接口预留）。
- `render_markdown`：per-op 表（双口径带宽、%bw、GF/s、verdict）+ 链式块
  表（serial est × coverage、Δ）+ 判定规则脚注。
- verdict 规则：overhead-dominated（solo≥2×burst）> compute-bound（≥50%
  peak）> memory-bound（≥50% bw）> latency-bound（<10% bw 且有 burst）。

### 尝试步骤

1. 写 Cargo.toml + lib.rs + model.rs + report.rs，4 个单测（overlay 往返、
   solo/burst 聚合、链块归属）全绿。
2. 接入 workspace + xnpu-cli 依赖，改造 run-w4ulayer：per-op 模式录 solo
   事件；per-layer/pipelined/grouped/chunk 全部录 burst 事件 + 每 iter 一
   个链式块；metas 按 4 形状建（跨层共享）；报告打印 + JSON/MD 落盘到
   build/perf/。
3. 上设备跑 42L × 5 iters——**第一次报告就抓到一个设计 bug**：Δ(iter−Σsolo)
   按每 op 只计一次算串行估计，但链每 iter 覆盖 42 层，Δ 虚高 +73ms。
   修复：ChainStats 携带块窗口内 distinct op 名集合，serial est =
   Σ solo_med × coverage（coverage = 每 iter 每 op 提交次数 = 168/4 = 42）。
   （中途否决过一个「求和 hook」的绕路设计，直接存名字集合更干净。）
4. 同一轮暴露**陈旧锚点**：per-op %bw 出现 122–175%，因为 M3b 的
   bw_stream=24.4 GB/s（pipelined state-poll 口径）已过时——今天 syncobj
   口径的 per-layer 链级 slot 流实测 33.3 GB/s。更新默认值 + provenance。

### 结果（42 层 × 4 形状，5 iters/模式）

per-op（solo 口径，n=210/形状）：

| op | solo µs | useful GB/s | slot GB/s | %bw | GF/s | verdict |
|---|---|---|---|---|---|---|
| qkv (2560×2048, F=5) | 274 | 10.80 | 32.33 | 97% | 38.3 | memory-bound |
| o (2048×2048, F=4) | 235 | 10.07 | 30.15 | 91% | 35.7 | memory-bound |
| gateup (12288×2048, F=24) | 990 | 14.33 | 42.95 | 129% | 50.8 | memory-bound |
| down (2048×6144, F=4) | 508 | 13.97 | 13.95 | 42% | 49.5 | mixed |

链式块（Δ = µs/iter − serial est，serial est = 84.29 ms）：

| 模式 | ms/iter | Δ ms |
|---|---|---|
| per-op（实测含逐层 verify） | 87.1 | — |
| per-layer | 78.0 | −6.3 |
| chunk4 | 77.9 | −6.4 |
| chunk12 | 75.8 | −8.5 |
| pipelined | 74.8 | −9.5 |
| grouped | 74.8 | −9.5 |
| chunk32/64/168 | 75.0–75.3 | −9.2~−9.3 |

trace 落盘：build/perf/w4ulayer_42L_*.json（10920 events/60 blocks，JSON
已用 python 校验）+ 同名 .md。

### 发现

1. **%bw>100% 是特性不是 bug**：单 op solo 超过链级平均带宽天花板 → 天花
   板常数低估，需要 perf-calibrate 用单 op burst 连发重标。框架第一次运
   行就抓到了一个过时锚点。
2. **同链带宽差 3×**：gateup（F=24）43 GB/s slot vs down（F=4）14 GB/s。
   F 大 → 每 slot 固定开销摊薄；down 同样 7.09MB slot 却比 o 慢 2.16×，
   因为 K=6144 每 weight 字节配 3× MAC（GF/s 49.5 全场最高）——down 最接
   近算力受限，但 peak 未知无法定论。
3. **有用字节口径聚在 10–14 GB/s**：四个形状的有用带宽挤在一起，说明链
   大体按有用字节走、slot padding 是形状相关的第二效应——单一带带宽分母
   会误判，双口径是必要的。
4. **流水收益 ~9.5 ms/token**：深队列（grouped/pipelined/chunk≥32）比串行
   solo 估计（84.3ms）净赚 ~11%，浅队列（chunk4/per-layer）只赚 ~7%——
   排队深度换 syncobj 等待次数的边际收益在 chunk12→32 之间饱和。
5. chunk6 的 verify 出现 1/5 flake（L39 读到 L40 值）——M3b 已知的写可见
   性滞后现象，与本次改造无关，待 §17 的 state 可见性问题一并处理。

### P1 续：run-fkprobe 改造（同晚）

solo 窗口只含 submit+wait（原 steady 计时把逐 iter 校验也计入了）；新增
burst 相位（24 连发 + 末尾 syncobj drain）拿纯设备时间：

| op | solo µs | burst/op µs | ovh µs | useful B | GB/s | %bw | GF/s | verdict |
|---|---|---|---|---|---|---|---|---|
| flowkv (16h/2kv d128 S=1024) | 2576.5 (4) | 2470.3 (24) | 106.2 | 1057344 | 0.43 | 1% | 3.4 | **latency-bound** |

- M4a 的"flowkv ~2.6ms、延迟受限"从手工推断升级为带分母判定：1% 带宽 +
  3.4 GF/s，双维度都贴地。优化方向只能是并行度（多 pos / 向量化 exp2），
  不是带宽。
- solo−burst = 106µs > 55µs 锚点：长 op 的 syncobj 往返更贵（或含长尾），
  submit_overhead 常数应按 op 时长分层——记入 perf-calibrate 需求。
- 过程中修了两个 verdict 边界：bytes_stream=None 时退回 useful 口径
  （无 padding 的 op 两者同义）；「有字节无计时」新增 no-timing（链式块
  里的 op 只有链统计时）。第二个边界是单测先炸出来的（unreachable 真的
  被到达）——报告层判定矩阵比想象的大，单测守住每格。

### P1 续②：run-decode 改造（E2E 链，M5a 验证标准）

checked step（含逐层 golden 磁盘读）不记录；timed step 逐 op solo + 每
iter 一个 decode-step 链块。两路径各 5 iters：

CPU-attention 路径：steady 101.8ms/token；per-op 表与 run-w4ulayer 手动
计时一致（qkv 272/o 235/gateup 990/down 508）——**M5a 验证标准达成**。
Δ(iter−serial) = +16.9ms = 标量 attention+norms/rope/swiglu 胶水，与
M4a 手估 ~17ms 吻合。

NPU-attention 路径：steady 240.5ms/token（4.2 tok/s，E2E PASS 4.3%）：

| op | solo µs | 解读 |
|---|---|---|
| qkv | 283.5 | 同 CPU 路径（前 op 同 CU，无切换） |
| flowkv (S=101) | 2821 | exec + 前置 CU0→CU1 切换 ~650µs |
| o | 874（CPU 路径 251） | **CU1→CU0 切换代价落在后继 op 的 solo 窗口** |
| gateup / down | 1003 / 517.5 | 正常（同 CU 相连） |

- **机制代价被自动归属到受影响 op**：每层 2 次 CU 切换 ≈1.3ms×42 =
  55ms/token，M4a 的手算分解（flowkv 109ms + 切换 55ms + 投影 75ms）在
  带分母的表格里重现。flowkv @S=101 与 @S=1024 几乎同速 → 内核时间与
  运行时 S 基本无关（满容量 DMA + 固定开销主导）。
- **发现运行间方差**：CPU 路径两轮 101.8 vs 121.8ms（per-submit 606→725
  µs），非框架开销——链 wall 目前是均值，需加 min/med/max 分布（待办）。
- 修了块归属第三 bug：链窗口只收 burst 事件名 → solo-only 链（decode
  本身就是逐 op 等待）names 空、cov 错算 168、空集求和 Some(0)。改为收
  任意模式事件 + 空 names 守卫，回归测试 solo_only_chain_block 锁住。

### 下一步（M5a 未完项）

- perf-calibrate 子命令：单 op burst 连发测 bw/peak 真天花板（gateup burst
  可能给出 >43 GB/s 的新锚点；peak_gflops 用纯 MAC 形状测）。
- 链 wall 由均值改为 min/med/max 分布（运行间方差 101.8↔121.8ms 需要它）。
- pipelined 模式换 syncobj-wait drain（去 state-poll 妥协）。
- ~~retrofit run-fkprobe / run-decode~~（已完成，见上）。
- M5b 笔记已写：docs/perf/01-tilelang-borrowings.md。

---

## P2（2026-09-24）M5c 前置：Q4NX 权重格式逆向（hy-mt2 1.8B）

### 设计思路

M5c 要逐算子对标 FLM（44.3 tok/s vs 我们 4.1–4.3），前提是**同一份权重**。
本地只有 FLM 的 model.q4nx（1.5GB，闭源格式）没有 HF safetensors——把
q4nx 解码成普通矩阵，既是我们的权重源（省 3.5GB 下载），又保证对拍
apples-to-apples（我们跑的就是 FLM 跑的那个 int4）。公开文献只有 Gemma3-
NPU 论文（arXiv 2602.06063）给过 Q4NX 语义（Q4_1 式 d·q+m，g=32），但
论文说的是 5120B/块带 zero-point 的变体——我们文件是 4608B/块，不同。

### 最终格式（hy-mt2 文件，100% pin）

```
文件   = u64 LE manifest_len(39856) + safetensors 式 JSON manifest + blob
张量   = 130 BF16（norms/q_norm/k_norm/embed）+ 225 I8（7 投影×32层+lm_head）
I8 张量 = ceil(M/32)*ceil(K/256) 个 tile，每 tile 4608B，序号 t = tr*ntc + tc
tile   = 32 行 × 256 列：
  [0,512)    256 个 bf16 scale，flat idx = group*32 + 行号（[g][lr] 组主序）
  [512,4608) int4 nibble：byte = 512 + col*16 + lr//2
             （偶行=低半 nibble，奇行=高半），有符号补码 [-8,7]
量化   = w = q*d，d = max(group)/-8（llama.cpp Q4_0 式对称，无 zero point）
```

与 1bit-MONSTER 逆向的 5120B 变体差异：无 [512,1024) zp 区、无 6B 头、
列距 16B vs 8B。与 IRON 自带 dequant/reference.py 的布局（nibble 在前
scale 尾置、uint4）也不同——三个来源三种布局，只能逐字节自证。

### 尝试步骤（含失败，按时间）

1. **文件结构 + 形状审计**：8B 长度前缀 + JSON 立即可读；全部 225 个 I8
   张量的 tile 数与 ceil(M/32)×ceil(K/256) 吻合（q 512 / k,v 128 /
   o 512 / gate,up,down 1536 / lm_head 30208）。
2. **oracle 根因 bug（本次最大的坑）**：早期所有布局假设检验全错，因为
   Python `struct.unpack("<e")` 是 **fp16 不是 bf16**——整个 session 的
   embed 参考值都是垃圾（±1.2 量级，真实 ±0.03）。正确解码 =
   `(u16.astype(np.uint32) << 16).view(np.float32)`。教训：**自造 oracle
   先用已知值标定**（比如 embed 第一行均值应 ~0.02）。
3. 失败假设清单（fp16 bug 修掉后全部立刻收敛）：连续 scale 区不存在→
   其实存在（[0,512)）；8B 列距→实际 16B；6B 头+回绕→无头；scale
   [lr][g] 序→实际 [g][lr]（tile1 曾因此误判失败 rms 1.45）。
4. **tile 序确认**：tile0=(tr0,tc0) 全 tile 解码 rms 验证后，tile1 用
   全 embed 表（120818×2048）量化整数域模糊搜索 → 98% nibble 精确
   命中 (row0, colwin1)，即 **tc-优先**（t=tr*8+tc）。残差 2% 是我
   round 与 FLM 量化器的平局舍入差。
5. **工具落地**：tools/q4nx_import.py——mmap 读 q4nx → 向量化解码
   （scale 侧 np.repeat 重排、nibble 侧偶奇切片）→ 拼接 qkv 3072×2048
   （16Q/4KV！）/gateup 12288×2048 → 走 w4_import 同一 packer（v2
   universal slots，M=3072 满足 32 整除，IRON 零改动）→ 输出契约与
   w4_import 完全一致（.bin/golden/meta.json + bf16.safetensors 的
   norms）。三次幼稚 bug：lr 轴分配 16（应 32）、nibble 只取前 8 字节
   （16 字节全用）、bf16() 忘 reshape——都是广播/形状错误当场炸出。
6. **全量验证（免费强校验）**：hy-mt2 tie_word_embeddings=true → 解码
   lm_head（全部 30208 tiles，含 14 行 padding 边界）vs 文件内 BF16
   embed 表：**rel_rms=0.0892**，与单 tile 0.096 同量级（int4 量化
   噪声），解码器无系统误差。

### 结果

- build/w4u_hy/：32 层 × 4 形状 v2 slots（67.3MB/层，共 ~2.15GB）+
  golden spot checks + bf16.safetensors（norms/q_norm/k_norm/final）。
- q4nx 成为本地自足权重源；M5c 对拍与 FLM 同权重同量化。
- 权重维度事实：qkv 3072（q2048+k512+v512，**16Q/4KV GQA**）→ flowkv
  fixture 需编 16h_4kv；q/k 各带 rms_norm（hy 是 **rope 后 norm**，
  hunyuan_npu.hpp 注释）；rope theta 1e4 + dynamic scaling。

### 留痕的网络工程细节

GitHub git 协议过代理（127.0.0.1:7890）持续 GnuTLS 复位：blobless clone
+ sparse-checkout / promisor fetch 全部不可用。可行模式：`gh api
repos/<o>/<r>/tarball` + curl --retry 断点续传（首跑 29.6MB 截断且 gh
exit 0 谎报成功，zcat --ignore-failed-read 部分提取救回）+ 单文件走
`gh api .../contents/<path>` base64。FastFlowLM 完整源码落在
refs/flm-ref2/ROCm-FastFlowLM-39ff855/（MIT 编排层 + 闭源 DLL 内核，
built on IRON）。

### 下一步（M5c 本体）

- decode_export.py 参数化（heads 16/kv 4/32 层/qk-norm 后置/rope 动态
  scaling）+ Rust 侧 W4U_SHAPES qkv 2560→3072。
- flowkv 16h_4kv fixture 编译 + qkv E2E 对拍。
- FLM per-op trace（hook.cpp C++ 扩展 vs LD_PRELOAD——待用户裁决，
  涉及禁 C++ 红线的例外申请）→ 同 schema 对比表。
