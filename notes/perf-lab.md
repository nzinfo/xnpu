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

## P3（2026-09-24）M5c 主体：hy-mt2 1.8B 全引擎 E2E PASS + per-op 表

### 设计

run-decode 双 arch 化：`DecArch` 结构（layers/heads/kv/qk_norm/rope_base/
qkv_m/qkv_f/decdir/w4dir），DEC_MINICPM 与 DEC_HY 两个 const profile。
CLI `run-decode [dirs] [iters] hy cpu`——"hy" 换 arch，"cpu" 保 Rust 标量
attention。hy 与 minicpm 的差异全部收敛进 profile：

- 32 层、16Q/4KV（GQA group 4）、qkv 融合 M=3072/F=6（3072/8 满足 packer
  32 整除，**PDI 与其它形状字节一致已验证**——通用内核前提不动摇）；
- qk-norm 在 rope **之后**（qk_rms_bf16 逐头 128 维 rms，权重
  qknorms.bin L×256=[q|k]）；
- rope base 11158840.0（f32 舍入 rel 7e-9，远低于角度噪声）；
- norms 索引 (2L+1)、KV cache L×4×1024×128、最终 norm 2L 偏移全部参数化。

decode_export.py 同步双 arch（--arch hy 出 build/dec_hy），minicpm 回归
bit-identical（仅 meta.json 加字段）。

### 踩坑（都留痕）

1. `const` 里不能调 `powf`（E0015）→ Python 算好硬编码 + 注释推导。
2. qk_rms 不能 alias 输入输出 → 借 attn 当 scratch 弹一次；
   `kr.copy_from_slice(attn)` 长度 512 vs 2048 panic → `&attn[..kdim]`。
3. `qr.copy_from_slice(&attn[..qr.len()])` E0502（qr 可变借用里再不可变
   借用 qr.len()）→ 先取 `let qlen = qr.len()`。
4. 权重加载循环里残留 `W4U_SHAPES`（minicpm 形状）→ hy qkv 尺寸校验必炸；
   gemv 闭包同样两处 → 全换 `shapes[si]`。教训：**参数化改造要 grep 到
   底，常量名相似（W4U_SHAPES vs shapes）编译器不救你**。

### 结果（2026-09-24，build/perf/decode_cpu_5it_1790184457）

- **E2E PASS**：final hidden rms 0.0265 = **1.0%** golden rms（门槛 5%）。
  最大绝对差 0.50 出现在 golden 85.0（0.6%）；L24 起零星 22/2048 超逐层
  容差（worst rel 3.1 是近零 golden 放大，最终收敛）——与 minicpm 同
  fingerprint，非系统误差。
- **12.8 tok/s**（78.4 ms/token），同 harness minicpm 4.1–4.3 → 3×；
  FLM hy2 基线 44.3 tok/s → 差 3.5×。
- per-op（solo=submit+wait 墙钟，5 iters×32 层各 160 样本）：

| op | 形状 | solo µs | stream GB/s | %bw | 判定 |
|---|---|---|---|---|---|
| qkv | 3072×2048 | 305 | 34.9 | 105% | memory-bound |
| o | 2048×2048 | 248 | 28.6 | 86% | memory-bound |
| gateup | 12288×2048 | 984 | 43.2 | **130%** | memory-bound |
| down | 2048×6144 | 503 | 14.1 | 42% | mixed |

- 链式：Σ solo = 65.3 ms vs 墙钟 78.4 ms → **Δ=13.1 ms/token =
  CPU glue**（标量 attention + rope + qk-norm + swiglu + x 复制 + 每 op
  submit/wait 往返），≈102 µs/op。
- **机器模型锚点失效发现**：bw_stream=33.3 GB/s（minicpm 42L 校准）被
  gateup 单 op 打到 43 GB/s（130%）——锚点依赖负载形态（Bo 大小/页
  驻留模式），%bw>100% 应触发重校准而非判不可能。TileSight 式模型要
  把"有效带宽按访问粒度分档"做进 MachineModel（后续）。

### 判读（vs FLM 44.3 tok/s 的 3.5× 差距构成）

1. 投影流本身 65.3 ms（若 burst 流水化可再压，单 op 级 submit+wait 是
   上界口径）；
2. 13.1 ms CPU glue——attention 上 NPU（flowkv 16h_4kv fixture）+
   图执行器消 per-op 往返是下一刀；
3. FLM 还有 layer 间双 buffer/prefill 融合图——先测准我们自己的，
   再对齐它的调度。

### 下一步

- flowkv 16h_4kv fixture 编译（IRON）→ hy NPU attention E2E。
- FLM per-op trace 同 schema 对比（hook.cpp C++ 例外待用户裁决）。
- MachineModel 带宽分档（按 op 流粒度重校准 bw_stream）。

## P4（2026-09-24）hy NPU attention 12.1% 判读：AIE2P exp2 硬件初等函数 3–6% 系统误差（实锤）

### 背景与判别设计

M4a 已修 double-rope（内核对 Q 施 RoPE；宿主改喂 identity angles
cos=1/sin=0（bf16 精确），q 预 rope+qk-norm 后送入——identity 下内核
rope 位精确，宿主/内核职责干净分离）。此后 hy npu E2E 仍 12.1%（门槛
5%，cpu 同链路 1.0%）→ 本条把 per-op flowkv 残差解剖到底。

工具链（tools/，一次编译多 run 复用 runlist）：
- `fk_hy_check.py`：E2E 精确 layer-0 数据（导出 norms/qknorms/cache）XRT
  单独跑 flowkv，多档逐位仿真对拍 + 逐头 rms/lstsq scale/maxscore 表；
- `fk_exp2_probe.py`：四模式 bisect（mono/shuf/dense×幅值）；
- `fk_exp2_fine.py`：细步长 arg 扫描，V one-hot 逐位置暴露权重；
- `fk_exp2_pairs.py`：成对同值 k（位置性 vs 取值性判别）+ 逆序（rescale 路径）；
- `fk_exp2_iso.py`：**隔离测量**——单点非零 k，其余 100 位置 arg≡0
  （f 恒 1），`x = 100p/(1−p)` 精确反解硬件 exp2 输出，201 个 arg。

### 死路（全部留痕）

1. Rust 打包：XRT 同打包复现 3.6%（O rms 0.080 上 0.00285）→ 非打包 bug。
2. 尾部/runtime-S：−1e30 sentinel 代数中性，tail-pad 模型拟合更差（0.00832，
   无 shrinkage）且逐头 lstsq scale≈1 早期即排除。
3. 文档化舍入全链仿真（bf16 score 存储、1.4453125 伪 log2e、bf16 f/C_c、
   l bf16 交叉、f32 Y）：hy 数据上仅 0.00046——**内核实测 0.00285 与文档
   算术不符，偏差与模型正交**。trueLog2e 变体无差别。
4. plain bf16 仿真 0.00051；exp2arg 模型 0.00280（更差）。
5. online-rescale 路径：shuf-deep（中流 max 更新）贴合 0.0007 → C_c 路径
   本身无辜（但见下：它放大真凶 3×）。
6. 分数幅值律：dense×9 复现 hy 量级 0.0030 —— 误判为 score-ULP 放大；
   mono 扫描 score 至 −16 却贴合 → score 侧额外 ULP 噪声被排除。
7. 值核 bf16 乘积假设（f·V 逐项舍入）：real 数据上无改善（0.00293）。
8. 位置性假设：成对同值 k 逐位相同（max 对内差 0.000000，跨头 0.000000）
   → 误差是 score **取值**的确定函数，与位置/头/FIFO 无关。

### 实锤（fk_exp2_iso.py，2026-09-24）

`aie::exp2<bfloat16>` = `::exp2(accum<accfloat>)` AIE2P 硬件初等指令
（elementary.hpp 仅此一路，无 f32 输出变体）的按值相对误差：

- **mean +3.25%，max +5.67%，min −0.6%**（n=201，arg∈[−24,0]）；
- arg=0 与整数 arg 处 ≈0；**峰值在 frac(arg)≈0.5–0.6，周期 1.0** ——
  尾数多项式中段系统性偏高（粗系数快速路径）；
- 确定性、逐位可复现（同 arg 同误差）。

### 由此解释全部观测

- l 虚高 +3.2%（f 均值 +3.25% → Σf 膨胀）→ 逐头 O scale 0.988–1.057；
- 逐头误差与 maxscore 反相关：尖 softmax 主权重 arg≈0（精确区），平坦
  softmax 的 arg 铺满 [−1,0]（误差区）——hy h(max 8.7)=0.4%、h(3.4)=6.5%；
- 逆序扫描（每 chunk max 更新）误差 ×3 至 14.9%——C_c 同 intrinsic，
  rescale 级联复利；
- pytest 全容量均匀数据 f≡1（arg≡0 精确区）→ fixture 免疫，与旧
  l-recursion bug 同款盲区；
- 向量 rms 指标的教训：单维 4% 误差被 128 维稀释成 0.0005——**逐位置
  相对误差才是 softmax 类算子的合格指标**（2f 反解法即为此设计）。

### 系统性外延

IRON `softmax.cc:61/146`、`mha.cc:256` 用同一 `aie::exp2<bfloat16>` 模式
→ prefill/MHA 同病。aie_api aie2p 无精确 exp2（16-bit 输出快速族仅
Fix2Float/Float2Fix/Inv/InvSqrt/Tanh/Exp2）。

### 修法选项（待用户裁决 C++ 红线）

1. flowkv.cc 换手写精确 2^x（f32 Horner 尾数多项式 + 指数位操作，~15 行，
   f/C_c 三处共用 helper；顺手可把逐位置 broadcast-exp2 改整 chunk 向量化，
   精度+性能双收）；
2. 不动 C++：hy attention 留 CPU（S=101 时本就更快更准：12.8 vs 6.4 tok/s），
   flowkv 数值噪声底记录在案；
3. 两者并行：先 2 后 1。

### 附带性能事实

flowkv 4col 1819 µs/层 ×32 = 58 ms/token，按容量 1024 流 KV（1.15 GB/s）
而非 runtime S——attention 上 NPU 前必须先修容量流（S-感知 DMA），否则
数值修好也慢于 CPU。

### P4 收尾（2026-09-24 凌晨）：修复落地 + 第二只 bug（E2E 独立）

**修法 = 选项 1**（用户授权 C++ 例外，vectorization 拆后项）：flowkv.cc 增
`flowkv_exp2_accurate`（[0,1) 上 5 阶 f32 Taylor，系数 = ln2 级数；r∈[1,2)
经 union 指数位 +n<<23 精确乘 2^n；x≤−126 clamp 0；x≤0 truncate+fix 求 floor），
f/C_c 三处调用替换，文档化的 bf16-arg 量化链保持不变。

三层验证（全绿）：
- pytest 10/10；
- 隔离曲线（fk_exp2_iso 重跑）：mean +3.25%→**+0.03%**、max +5.67%→**+0.5%**
  （= p 的 bf16 粒度）；
- 真实数据（fk_hy_check）：npu-vs-ref 0.00285→**0.00046**（文档化舍入噪声底）、
  EXACT-sim 拟合 0.00009、逐头 scale 0.988–1.057→0.989–1.011、最坏头
  rms 0.00088。

**但 E2E hy npu 仍 12.3% FAIL（12.1%→12.3%，exp2 修复只挪了 0.2pp）** →
独立 op 干净 + 集成仍坏 = E2E 路径自己的 bug，此前被"32 层复利 3.6% op 误差"
的归因掩盖。排查 main.rs 逐 token 路径（Q refresh/KV append/O 读）：

- **根因 = main.rs:4076 V 行源偏移硬编码 2304**（= 2048+2·128，MiniCPM kv=2
  常量）。hy kv=4 的 V 在 2048+kdim+kvh·128=2560+…。每层最新 token 的 V 行
  实际写入的是 K-head-2/3 与 V-head-0/1 的数据；CPU 路径（:4058）用对
  2048+kdim —— 这就是 cpu 1.0% / npu 12.3% 分野的全部。单行修复
  （`qkv[2048 + kdim + kvh*128 + j]`）。
- 判别信息复盘：standalone 真数据干净（python 打包）+ E2E 坏（Rust 打包）
  → bug 必在 Rust 打包/搬运层；it=0 即 layer-0 发散（S=101 新鲜、无跨步
  状态）→ 排除 stale header/跨 iter 累积；同一 qkv 里 K 行源（kr）对而 V
  行源错 —— 唯一 arch 相关常量就是 2304。

**E2E 终态（hy npu attention，IRON 43dd37e + xnpu b14b3b4）**：final hidden
rms **0.0226 = 0.9% golden**（门槛 5%，与 CPU 路径 1.0% 同噪声底）；
first divergence layer 0→20（19/2048，worst rel 落在近零 golden 上，仅信息
性）。**双 bug 故事定案：kernel exp2 偏差 + Rust V 偏移，各自把对方藏在
12.1% 的复合误差里** —— 修第一个不动第二个只挪 0.2pp 的教训：复合归因
（"32 层复利"）在两个独立 bug 叠加时会给出貌似自洽的错误解释。

**性能现状（M5a 口径）**：steady **5.5 tok/s**（181ms/token；CPU attention
12.8 tok/s / 78.4ms）——flowkv solo 1986.5µs ×32 ≈ 64ms/token 容量流 +
CU 切换。flowkv 表现：0.11 GB/s use / 1.06 GB/s stream / 3% bw → 判定
mixed，即 S 感知 DMA 之前的预期状态。数值与性能两条线就此分离：
数值线关闭，性能线（S-aware DMA / exp2 chunk 向量化 / CU 融合）转入
待办。
