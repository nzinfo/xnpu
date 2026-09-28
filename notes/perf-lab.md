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

## P5：FLM per-op trace（LD_PRELOAD 拦截 XRT C++ API）——44 tok/s 的结构解剖

**日期**：2026-09-24 凌晨。**工具**：`xnpu/crates/xnpu-ftrace`（Rust cdylib，
用户批准的方案 A）+ `tools/flm_trace_analyze.py`。

### 设计与坑

- FLM 闭源内核 DLL（lib{hunyuan,q4_npu_eXpress,gemm,dequant,mha,lm_head}_npu.so）
  **动态链接 libxrt_coreutil.so.2 的 C++ API**（`nm -D` 全是 `U _ZN3xrt...`）
  → LD_PRELOAD 拦截 mangled 符号即可，无需碰 ioctl（M1 xdump 技术上移一层）。
- 拦截集 19 个符号（run/runlist 的 start/wait/add/exec/reset、set_arg_at_index
  两型、bo 的 ctor/map/sync、kernel/xclbin/hwctx/module/elf ctor）。全部指针/
  标量传参、无 sret → 统一 `-> u64` 透传 RAX 是精确 ABI（void 调用方本就忽略
  RAX）。dlsym(RTLD_NEXT) 每符号 OnceLock 缓存。
- **坑 1**：libstdc++ SSO 判定——长度字段 +8 两种模式都有效（首版误读 local
  buffer 首字节当长度，读出 77 字节垃圾）。
- **坑 2**：**FLM 所有 kernel 都叫 `MLIR_AIE`**（mlir_aie 默认名）→ kernel 名
  无鉴别力。身份改由：xclbin ctor 的文件路径（layer vs fused_prefill 两图）
  + bo_new 尺寸直方图（26MiB×32=每层权重、133MiB=lm_head）+ 每 run 的
  arg 指纹（idx→bo size）。
- **坑 3**：kernel 对象被 move（vector 搬家），run_new 里出现的 kern 地址在
  kern_new 里没有 → 地址 join 有洞；顺序 join 够用。
- **坑 4**：分析时间轴必须统一绝对 t0——两版探索脚本各自取了"首事件"与
  "首个 run_start"当原点，差 0.9s，险些把 setup 期的权重 staging 当成 decode。
- 运行配方：`sudo -n env HOME=/home/nzinfo FTRACE_OUT=... LD_PRELOAD=... flm run
  hy-mt2:1.8b`（root 的 HOME 找不到模型会触发重新下载；memlock 需 root）。

### 结果（hy-mt2 1.8B，翻译 prompt 22 tok + 8 tok 生成）

- **setup**：203 个 BO 共 1170MiB：26MiB×32（每层权重 BO，= 我们估算的
  25.9MiB/层 q4 权重 ✓）、133MiB×1（lm_head）、1MiB×136 + 2MiB×33（激活/
  中间量）。权重全走 SHMEM BO 常驻，同我们。
- **prefill**：8 op/层 × 32 层 = 257 次 run_start，逐 op start+wait（与我们的
  调度粒度相同！），4.89ms/层，共 156ms（22 tok，≈141 tok/s）。每层 8 op 的
  稳态时长 ≈ 330/90/95/250/300/770/930/720 µs。
- **decode：每 token = 33 个 run（32 层各 1 个融合内核 + 1）打进一个 runlist，
  一次 rl_exec + 一次 rl_wait**；另有一个 2.66ms 的长 run（lm_head，133MiB →
  52 GB/s）。21.44 ms/token（46.6 tok/s，与 serve 基线 44.3 一致）。
  - 设备批次被 [wait-block 13.5ms, exec→ret 18.0ms] 夹逼 → **0.42–0.56
    ms/层 = 48–64 GB/s 有效权重流**。
  - host 残余 ~5ms/token（含与 wait 重叠的下一 token runlist prep——两个
    runlist 地址交替 ping-pong，经典双缓冲）。
- **对比我们（78.4ms/token，cpu-attention）的 3.7× 差距分解**：
  1. **算子数**：FLM 每层 1 个融合内核 vs 我们 4 个 w4gemvu + attention——
     层内全融合（qkv+rope+qk-norm+attn+o+gateup+swiglu+down 一个内核）；
  2. **调度**：每 token 1 次 exec+1 次 wait vs 我们 160 次 submit+syncobj
     往返（~55µs×160≈8.8ms）+ 84 次 CU 切换（650µs PDI 重载当 cu_mask 变）；
  3. **带宽**：FLM 层内核 48–64 GB/s vs 我们 useful 14（slot 峰 43）——
     硬件流远未到顶，我们的小形状（down F=4=14GB/s）与逐 op 开销拖垮聚合；
  4. lm_head 在 NPU（2.66ms）——我们的 M4 遗留项。
- **MachineModel 硬数据**：33.3 GB/s 的 bw_stream 锚点被证伪为天花板——
  FLM 实测 ≥52 GB/s（lm_head 单 BO 大流）与 48–64 GB/s（层权重流）。
  M5a 遗留的"按访问粒度分档重校准"现在有下界：大 BO 顺序流 ≥50 GB/s。

### 局限与后续

- runlist 内部逐 run 的设备时间 API 层不可见（无逐 run wait）——只拿到批次
  总量；逐 op 需 M1 式原始 EXEC_BO ioctl 时间戳（后继）。
- FLM 的层融合内核内部结构（tile 划分/多列利用）不可见——但 48–64 GB/s
  已经是可对标的数字。
- 产物：build/perf/flm_hy_trace.log（6874 行原始 trace）。

## P6（2026-09-24）：带宽分档 MachineModel + perf-calibrate + 链式 min/med/max + submit 开销分档

M5a 收尾。P1 立的四个欠账一次清掉：单锚带宽证伪后的**分档机器模型**、
`perf-calibrate` 子命令（上板实测天花板并产出 overlay JSON）、链式块的
**min/med/max 分布**、submit 开销**按流量分桶**。全部围绕一个原则：
**报告里的每个百分数必须对着 op 自己认领的那档天花板**，否则数字不可比。

### 设计

- **MachineModel.bw_tiers: BTreeMap<tier, GB/s>** + default_tier。三档语义
  = 访问粒度，不是数值区间：
  - `seq-dma` 52：顺序大 BO 权重流（FLM lm_head 133MiB/2.66ms 下界，M5c P5；
    M2 GEMM 曾见 54）。**无自测内核**——我们栈里没有一个纯顺序流算子可测它，
    保留 FLM 下界并在 provenance 里注明"未自测"。
  - `slot-stream` 43：F 槽复制流（P1 gateup F=24 上界）。本次实测校准到 41.2。
  - `strided` 1.2：2D stride KV 容量流（P2 flowkv）。实测 0.92–0.95。
  - `bw_stream_gbps` 33.3 降级为遗留兜底（bw_tiers 空/default_tier 缺失才用）。
  - OpMeta 认领档（`.with_tier(...)`），verdict/%bw 用该档分母；未知档名落
    default_tier。overlay_json 支持 `bw_tiers` 嵌套数字对象（花括号配平
    透传 + 递归解析），**按 key 合并**——部分校准（只测到一档）其余档保默认。
- **perf-calibrate [hy] [iters]**：每形状 solo×N + **同 op 连发块**×8N。
  这是仓库里第一批 per-shape burst 数字——run-w4ulayer 的 burst 块全是
  混 op 链，per-op burst 从未被隔离过。slot 档天花板 = 各形状 burst GB/s
  最大值；flowkv（可选 fixture，cu1）同法测 strided 档；submit_overhead =
  per-shape solo−burst 中位。产物 = overlay JSON（自校验 round-trip 解析）
  落 build/perf/machine_model.json，供后续 run-* 加载。
- **链式块 min/med/max**：同标签块跨 iter 的 per-iter 墙钟分布。动机：均值
  会被单个慢 iter 拉偏（P1 的 chunk 扫描已见），min 才是设备极限口径。
- **submit 开销分档**：solo−burst 按 op 设备侧流量分桶（<1MiB/1-4/4-16/
  ≥16MiB）取中位，对照模型常数 55µs。单一常数把 host 往返和 solo 排队撞上
  的 CU 切换/PDI 重载混在一起；分桶后各桶才可解释。

### 步骤与坑（全留痕）

1. **cu_func ≠ CU 槽位（踩坑，10s 超时）**：perf-calibrate 初版把 flowkv PDI
   配在 `(pdi, 1)` ——configure_cus 的第二个元素是 **cu_func（DPU 函数号，
   必须 0）**，CU 槽位是列表下标、由 chain_op 的 cu 参数选择。func=1 时固件
   查一个 PDI 里不存在的 DPU 函数，op 永不执行，syncobj 10s 超时。run-decode
   4080 行注释早就写着这个陷阱（"func != 0 … the op never runs"），没看熟。
   修：两 PDI 都 func 0，flowkv op 走 cu 1。
2. **µs 边界吞事件（踩坑，gateup 丢 burst 归属）**：首跑 gateup 的 burst 块
   被"误判成链式块"（cov=0.5）。dump trace 真相：块窗口 [47381, 116437] 里
   有 48 个 gateup burst + **1 个 down solo**——wait 返回 → burst_done 打戳 →
   立即 submit 下一个形状的首个 solo，三步落在**同一微秒**，闭区间把外来
   事件吞了。修：块归属窗口改半开 `[t_start, t_complete)`；块自己的末次
   submit 至少早完成一个设备 op，不会撞上界。单元测试 chain_min_med_max
   钉住分布语义。
3. **gateup 首跑 69ms/48op=1438µs vs 修后 49ms/48=1030µs**：同代码两次跑
   差 40%（29.6 vs 41.3 GB/s）。归因：48 深队列连续 submit 的背压/调度非
   确定性（M2 run-pipe 见过同款）。**burst 天花板本身是下界口径**——取
   多跑最大值；这也解释了为什么校准值（41.2）应低于真硬件极限（FLM 同
   机器 48–64）。记录在 provenance。
4. 测试：xnpu-perf 9/9（新增 overlay 嵌套 bw_tiers 合并、bw_for 查表/未知
   档兜底、tier 选 verdict 分母、链 min/med/max 分布）。

### 结果（hy-mt2，layer00，6 solo + 48 burst，两跑一致）

| op | F | solo µs | burst/op µs | slot GB/s | %bw(41.2) |
|---|---|---|---|---|---|
| qkv | 6 | 364 | 279 | 38.0 | 88% |
| o | 4 | 256 | 195 | 36.4 | 85% |
| gateup | 24 | 1130 | 1030 | 41.3 | 96% |
| down | 4 | 541 | 471 | 15.1 | 35% |
| flowkv | – | 2343 | 2288 | 0.92 (strided 档) | 77% of 1.2 |

- **校准产物**：`bw_tiers {slot-stream 41.2, strided 0.95}`，submit 52.8µs，
  seq-dma 保留 52（未自测）。build/perf/machine_model.json。
- **down 是 slot 档的短板**（15 vs 41）：K=6144=K_MAX 无 padding（stream≈
  useful），F=4 复制窄——每列有用速率的地板。而 K=2048 形状 3× padding
  换来 36-41 GB/s slot 流：**同一内核里 padding 与速率强相关**，消 padding
  （双 PDU per-K ELEM）必须同时提每列速率才不倒退（M3b 遗留方向的定量注脚）。
- **submit 开销分桶**（perf-calibrate 报告）：1-4MiB=55µs（+0%，flowkv 干净
  solo）/ 4-16MiB=69.7（+27%）/ ≥16MiB=81.7（+49%）——随流大小单调涨，
  大 op 的 solo 里混进了排队效应，不止 host 往返。run-decode（纯 solo 链）
  无 burst 对照，该节自动隐藏（设计如此）。
- **run-decode hy 4 iters（NPU attention）新报告**：decode-step 链
  min/med/max = 178.1/182.9/190.0 ms（5.5 tok/s 路径）；`o` 的 solo
  971µs vs 隔离 195µs —— CU 切换代价自动归属到切换后首个 op（P1 结论在
  分档口径下重现）；flowkv useful 0.10 vs stream 0.94 GB/s 双口径同表
  （S 很小、内核按编译容量整流的语义直接可读）。

### 语义讨论（记下来防止将来误读）

- **strided 档的循环性**：1.2 的"天花板"本身就来自 flowkv（P2），flowkv
  对它永远 ~77%。这档的含义是"该访问形态下已观察到的上界"——判 memory-bound
  说的是"在此形态内余量不大，要快就得换形态"（S 感知 DMA），不是"硬件到顶"。
  各档天花板全部是**实测下界**语义，报告口径写明。
- verdict 词表未变，但分母换了：P1 时代 gateup 130%（>100% 暴露锚点失效）
  现在 96%（真实余量读数）。**>100% 不再出现 = 分档起了作用**；若未来再见
  >100%，说明出现了新档未建模（如 LM 融合内核的 48-64 GB/s 形态——那是
  seq-dma 与 slot 之间的东西，等我们有融合内核再立档）。

### 遗留

- seq-dma 档无自测内核（纯顺序大流 DMA 夹具，或拿 lm_head w4 化后回测）。
- 校准值随队列深度漂移（40% 级）：后续 perf-calibrate 可加深度扫描
  （burst_n 8/16/32/64）取拐点，现在只取单深度最大值。
- run-* 主线尚未加载 machine_model.json（overlay_json 就绪，接线一行）。

## P7 校准收尾：machine_model.json 接线 + 队列深度扫描（2026-09-24⑥）

P6 三遗留清两项半：① run-* 报告全部对最新校准渲染（接线落地）；②
perf-calibrate 深度扫描（8/16/32/64）——**P6 的"深队列背压"假设被证伪**；
③ seq-dma 夹具评估后判定现有夹具不可用（算术见下），FLM 52 下界保留。

### 设计

- `machine_model_or_default()`：`build/perf/machine_model.json` 存在则
  overlay 到默认值（merge 语义——本次实测键覆盖、未测键保留），解析失败
  退默认并提示，绝不阻塞测量。四个报告点全接：run-w4ulayer /
  perf-calibrate / run-decode / run-fkprobe。perf-calibrate 的起点也是
  上一次校准，报告里能看到"旧天花板 vs 本次实测"的对照。
- 深度扫描：每形状 solo×iters 后按 DEPTHS=[8,16,32,64] 各出一块同 op
  连发块；深度行事件/metas 用 `{base}/d{N}` 名字——报告 per-op 表逐深度
  成行（tier/字节随 meta 走），天花板 = 跨深度最大。overhead = base
  solo_med − 各深度 burst/op（每深度一个样本进中位）。
- flowkv 同样扫描（每 submit 前 o_bo clflush 照旧）。

### 步骤与坑

1. **深度行 meta 裸奔**（板上一跑即抓）：给事件改了名却忘了把改名后的
   meta 塞进 all_metas —— 报告深度行全变 `n/a-bytes`，更糟的是
   slot_ceiling 因此为 0 **写进了 machine_model.json**，下一跑接线读入后
   全表 `inf%`。修复（双保险）：深度 meta 进 all_metas + **零天花板守卫**
   （本轮无可测行时保留模型现值，绝不把 0 写进 overlay）。
2. 闭包 `|n: &str| -> &str` 两个生命周期被推断成不同 → 编译错；改嵌套
   `fn`（单输入生命周期正确省略）。
3. provenance format 串加了 `{:?}` 占位忘加参数（编译期抓住）。
4. **时序测试 flake 两跑**（均紧跟 release build 同命令触发，load 高时
   sleep 精度劣化），8 连跑（含复现条件 2 次）不再现、失败名未捕获——
   留档：若再现先抓名字再放宽界，不盲调。

### 结果（同日 3 跑深度扫描）

- **深度平坦**：8→64 每-op 差 ≤4%。gateup 45.4/45.6/45.7/45.7（d8→d64，
  GB/s slot）、qkv 40.2→41.8、o 35-39（最噪 ±3%）、down 15.5→15.8、
  flowkv 0.97→0.98。**P6 把 gateup 两跑 40% 差归因"深队列背压"是错的**
  ——同一跑内深度无关；变异在**跑与跑之间**。
- **跑间漂移（同日单调上行，跨 op 相关）**：gateup 41.3（P6）→ 42.5 →
  45.4 → 45.7 → 45.6 GB/s；run-decode 链 182.9 → 161.2 ms/token；链内
  flowkv 1976µs 快于隔离校准 2153µs（报告 109% 如实标记校准过期）。
  跨 op 相关指向**全局设备态**（时钟/温度/固件状态机），根因未明——
  开放问题。天花板"下界 + 多跑取最大"的 P6 语义因此更必要。
- submit 常数三跑 52.8 → 86.6 → 55.7µs：单值是 16 样本（4 形状×4 深度）
  的中位，qkv solo 偏高（372µs vs burst 258，首批 op 残留切换效应）会
  拖偏 —— **P6 的流量分桶才是稳定视角**，单常数标注为噪声大。
- 终态校准（落 machine_model.json）：slot-stream **45.6**、strided 0.98、
  submit 55.7。down 仍是 slot 短板（15.7 GB/s，34%）。

### seq-dma 夹具评估（为什么没做）

- 现有 gemm 夹具全部 compute-bound：2048×2048×6144 = 51.5 GOP 对 ~91MB
  流量，8 列 4.9-6 TF/s 时算力是瓶颈，测出来的是算力不是 DMA。
- add 夹具 1 列且元素数语义含混 —— 单 shim 吞吐 ≠ 阵列聚合，立档无 op
  认领，反而污染 tier 语义。
- **正解 = lm_head w4 上 NPU（M4 遗留）**：124MB w4 顺序权重流正是
  seq-dma 形态（FLM 133MiB/2.66ms 同款），届时天然回测。

### 遗留

- 跑间漂移根因（全局设备态）未明；校准建议每次重要测量前跑一次
  perf-calibrate（<10s）取当日基线。
- 时序测试 flake 名字未捕获（见坑 4）。
- lm_head w4 = seq-dma 立档 + M4 收尾双收益，待排期。

## P8（2026-09-26）lm_head w4 上 NPU：全算子 NPU 化收官 + 102% 未建模层级信号实锤

### 设计思路

- **复用普适 w4gemvu 内核，零新设备代码**。P7 已证 PDI 与形状无关（K 运行时
  自描述）；本档补证 **M 也只在 ctrl code 里**：w4gemvu_120832x2048 与
  w4gemvu_3072x2048 的 main.pdi md5 相同（3b8c7213…），夹具 .bin 却不同
  （M 编进 ctrl 指令序列）。→ lm_head = 同一 CU 的第 5 份 ctrl code。
- **M=120818 → 120832**：vocab 补零到 32 行 ABI（cols8×4），pad 行量化为
  q=0/scale=0，复用 IRON packer 不改一行。
- **权重口径**：FLM q4nx 里 lm_head 已是 int4（verify_tie 证与 tied embed
  到 int4 误差），仍走"f32 → 我们 amax/7 重量化"——与 128 个层权重同待遇，
  对比目标就是 FLM。
- **golden 语义**：x = dec_hy/golden_hidden.bin（参考 final hidden，逐位
  验证过）；ref = dequant(w4) @ x，f32 点积单次 bf16 舍入（spot_rows 约定）。
  golden top1=25868（9.25），top1–top2 差 0.438（4.7%）→ argmax 门有效。
- **成本预告**：lmhead.bin = 8×(120832/32)×13840 = 418,078,720 B（399 MiB，
  ELEM 按 K_MAX=6144 编 → 3× K-padding 浪费）。slot-stream 45.6 GB/s 下
  ~9.2 ms/token 预期。

### 尝试步骤（含失败）

1. IRON test.py 加 (120832,2048) → pytest 5/5 PASS（14.8s，纯 host 侧
   编译+参考对拍）。
2. q4nx_import.py `--lmhead`：export_lmhead 导出 lmhead.bin（418,078,720 B，
   md5 1bf38aaf…）+ golden_lmhead.bin（245,768 B）。
3. `run-lmhead` 独立探针（Rust）：399 MiB W BO + 236 槽 x BO（每槽
   K_MAX 宽，replicate）+ 241,664 B 输出；submit→wait→全 logits 对拍
   （argmax/rel_rms/top-8）→ solo×4 + burst×8。**首次上板即 PASS**。
4. run-decode 集成：`lm_run` 每 token 步（refill 236 槽 → ToDevice →
   submit → wait → FromDevice），checked 步后对拍、timed 循环计入
   burst_done 的 n_submits，lmhead 进 per-op 报告；minicpm 无文件自动跳过。

### 坑（全部踩到）

- **目录约定分裂**：golden_lmhead.bin 导到 w4u_hy（与 lmhead.bin 一对），
  run-decode 却去 decdir 找 → 首次 E2E 静默跳过 lm_head（打印保留，153.9
  ms/token 是不含 lm 的）。修复：w4dir 优先 decdir 兜底。教训：**导出物
  消费方在写导出端时就该对齐路径**。
- **pos_args 吞数字**：`run-decode hy 4` 把 4 当 decdir（"read 4/norms.bin
  failed"）。iters 是第 3 个位置参数，用默认即可。
- **perf-calibrate 默认 minicpm 形状集**会覆写 machine_model.json 的
  strided 档（0.98 hy → 0.246 minicpm，flowkv 视角差 4×）。E2E 前必须
  `perf-calibrate hy`。machine_model.json 无 arch 命名空间——跨架构互踩，
  开放问题。
- Rust 侧小坑：u32le helper 不存在（内联 from_le_bytes）；pkt.submit 要
  &mut（gemv 的 `&mut ops[i]` 模式）；FnMut 闭包绑定要 mut。

### 结果

- **探针**：rel_rms 0.0000、argmax 25868==25868、top-8 8/8 → PASS；
  burst 8,955.8 µs/op（solo 9,007.5，ovh 51.8）= **46.68 GB/s =
  校准 slot-stream 天花板的 102%**。
- **E2E**（perf-calibrate hy 后）：链上 lmhead solo 9,107 µs = 45.91 GB/s
  （100%）；logits rel_rms 0.0097（探针用精确 golden_hidden，链上是实际
  decode hidden，0.9% rms 漂移一致）、argmax/top-8 全对 → PASS；
  **steady 165.2 ms/token（6.1 tok/s）**，同批无 lm 153.9 → Δ+11.3 ms
  ≈ 9.0 ms 算子 + 236 槽 refill/sync host 侧成本。
- **102% 的解读**：P6 预言的"未建模层级"信号坐实——399 MiB 大 M 流的
  持续带宽超小形状标定 45.7（gateup 53 MiB 测的）。要么天花板随 M 增长
  （更长流水摊开 ramp），要么 seq-dma 档（52，FLM 下界）才是大流归属。
  报告如实标 102%，框架不改（诚实过期标记正是设计行为）。
- decode 链 161 submits/token：128 w4gemvu（CU0）+ 32 flowkv（CU1）+
  1 lmhead（CU0）——**"全算子 NPU 化优先"收官**；CPU 只剩 rope/qk-norm/
  kv-append/norms/swiglu/argmax（~5% 时间）。
- L20 首次分歧打印（19/2048 超容差）两跑完全一致 → 确定性容差边缘，
  非 flake；final hidden PASS（rms 0.9%）。

### 遗留

- per-K ELEM 内核（M3b）：lmhead.bin 399 MiB 里 2/3 是 K-padding 浪费，
  K=2048 专用 ELEM 可砍到 ~133 MiB → lm_head ~3 ms，链 −6 ms/token。
- 跑间漂移根因（全局设备态）未明；machine_model.json 跨架构互踩；
  时序 flake 名字未捕获；seq-dma 自测内核仍未立档（lm_head 是其天然
  载荷但归 slot-stream 层）。

## P9（2026-09-26）配置对比的决定性实验：o 算子 846→244µs（cu_mask 翻转税 19.3ms/token）+ 性能差距分解

### 设计思路（一个异常引出的配置裁决）

P8 后链表里 o 独瘦：solo 846µs / 8.4 GB/s，而同字节数的 down 502µs、
更大的 qkv（10.6MB）只要 307µs。o 是链中唯一"前一 op 是异 CU（flowkv
CU1）"的算子 → 假设：M2 的 cu_mask 变更 → 固件 PDI 重载（~650µs）
在通用 PDI 时代仍然存在，每次 mask 翻转付一次。**决定性实验**：
`run-decode hy cpu`（无 flowkv 提交，其余全同）——o 降到 **244µs
（29.0 GB/s，64% %bw）**，假设坐实。w4gemvu PDI 通用化消灭了 CU0 内的
重载，但 CU0↔CU1 翻转每层仍付 ~600µs，×32 = **19.3ms/token 纯税**。

### 两种配置同日对照（都是 5 iters、同校准）

| 配置 | ms/token | tok/s | 分解 |
|---|---|---|---|
| CPU attention + NPU 投影 + NPU lm_head | **87.2** | **11.5** | w4gemvu 65.2 + lm 9.0 + host ~8.9 + CPU attn ~4.0 |
| 全 NPU（flowkv attention）| 165.2 | 6.0 | 上者 −4 + flowkv 62.7 + 翻转税 19.3 |

（87.2 − 9.0 lm = 78.2 ≈ P5 的 78.4 基线，口径闭环。）

- **S=1 短上下文：CPU attention 完胜**（4ms vs 82ms）。flowkv 现在
  无论 S 都流全容量 KV（op.py 自证："they still stream … F_c=0"），
  1.96ms/层 × 32 + 翻转税。
- **长上下文反转**：CPU attention O(S)（标量 Rust），S=1024 时 ~秒级；
  flowkv 容量流 O(1)。修好 S-aware 后 NPU attention 才有意义。

### 性能差距分解（decode 口径，FLM 21.44ms/token = 46.6 tok/s，P5）

我们最快配置 87.2ms = **4.1×**（全 NPU 165.2 = 7.7×）：

| 项 | 我们 ms | FLM ms | 差距根源 | 修法 | 潜力 |
|---|---|---|---|---|---|
| 投影（qkv/o/gateup/down）| 65.2 | ~14 | ① K-padding 3×（ELEM 按 K_MAX=6144）② down/qkv 带宽短板 ③ 逐 op 开销 vs 层内全融合 | per-K ELEM 内核（M3b） | **−34** |
| attention | 4.0(CPU) | ~0（融合）| flowkv 容量流+翻转税 | S-aware DMA + 单 PDI | 长上下文必需 |
| lm_head | 9.0 | 2.66 | 同 K-padding 3× | per-K ELEM | −6 |
| host 调度 | 8.9 | ~5 | 129–161 次 submit+wait vs 1 次 runlist | runlist 批量提交 | −4 |
| 终态估计 | | | | | **~43ms（2×FLM）** |

**翻转税的根治难点**：w4gemvu 占满 8 列、flowkv 编 4 列——静态布局冲突，
"并入单 PDI"需重排列（w4gemvu 降列数=降带宽）或 flowkv 降 1 列（×4 慢），
都不划算。FLM 单 xclbin 33 runs 免税是设计出来的（层内全融合）。
短期裁决：**短上下文用 CPU attention 配置**（87.2ms 已是当前最优），
flowkv 留作长上下文专项（S-aware + 翻转税一起修，才有正收益）。

### 遗留修复：machine_model.json 跨架构互踩（已完成）

perf-calibrate 默认 minicpm 形状集会把 strided 档写成 minicpm 值
（0.246），hy run 的 flowkv 行拿 0.246 当分母 → %bw 虚高 4×。修复：
**strided 层级按架构限定**（`strided:hy-mt2` / `strided:minicpm`）——
flowkv 几何（头数/容量）本来就是架构专属；slot-stream/seq-dma 与架构
无关保留裸名。calibrate 写限定键、run-decode 按 arch.name 认领、
fkprobe 从夹具名辨识（_4kv=hy）。上板复核：层级表同时显示
strided 1.2（兜底默认）与 strided:hy-mt2 1.0（校准），flowkv 行
认领正确（109%，P7 已知的链内快于隔离的诚实标记）。

### 新遗留

- cu_mask 翻转税的精确机理（固件行为）未解剖——若未来固件/驱动支持
  同分区多 PDI 常驻，19.3ms 自动回收。
- runlist 批量提交（FLM 式 1 exec/token）：161×54µs submit 往返。

## P10：w4gemvu v3 紧凑块 —— 3× DDR 浪费消灭 + "带宽瓶颈"假设证伪

**动机**：P9 把 K-padding 3× 列为最大差距项（预测 −35ms）。v2 每个
tile 占一个 K_MAX=6144 的 13840B 槽，K=2048 的 qkv/o/gateup/lm_head
每 token 流 3× 活字节（lm_head 418MB vs 136MB 活）；权重常驻
2.6GB。v3 目标：每块只流活字节，同时保持"一个 PDI 服务所有形状"
（CU 切换税 19.3ms 不复活）。

### 设计（w4gemvu.cc / reference.py / design.py / op.py / test.py 五件套）

- **块 = fifo 元素 = 13888B**：n = 6144/K 个 tile 背靠背（K=2048→3，
  K=6144→1），每个 tile 步长补齐 **16 字节对齐**（load_v 向量流不能
  从未对齐地址起读——见下方踩坑），K 头在块尾固定偏移。
- **内核每块调 3 次**（静态循环，tile_idx∈{0,1,2}）：内核读块尾 K，
  tile_idx ≥ n 时写零行 → C fifo 元素数与 K 无关 → 设备侧全形状
  逐位相同，**仍是单一 PDI**（md5 复核通过）。
- **ABI M 补齐**：K=2048 → M%192==0 且块数偶（o 2048→2112，
  lm_head 120818→120960，补零行量化为 q=0/scale=0）；K=6144 →
  M%64==0。B 槽服务 2 块 → F = 块数/2（qkv16/o11/gateup64/down32/
  lm630）。
- K=6144 的 C 行按 12 行组排（前 4 行真值 + 8 零行），主机侧
  `w4u_c_off` 收拢；K=2048 稠密。

**否决的替代**：2D tap 写紧凑块（shim BD 只会按 ELEM 步长写元素，
写不出紧凑字节流）；逐元素 B fill（16 BD/通道上限，gateup F=64 爆）；
两套 PDI（650µs×64 重载税复活）；K6144 假 tile 填满（L1 83KB 溢出）。

### 踩坑（按时间序）

1. **tile_idx=1 全错、idx 0/2 全对**（pytest 2688 指纹：失配行恰为
   ≡4..7 mod 12）。根因不是 peano 锚点怪癖不均匀，而是 **4616 % 16
   = 8**：tile1 起址 16B 未对齐，向量流读错位；idx0@0/idx2@9232 对齐
   所以对。修：tile 步长 (bytes+15)&~15 → 4624，块 13856→13888。
   教训：**load_v 流的对齐是硬约束，块内偏移必须 16B 对齐**。
2. **v3 首版 E2E 101.2ms（比 v2 87.2 倒退 14ms）**：F 槽数是 v2 的
   4×（B 槽 2 块 vs v2 16 tile），x 复制仍走逐 u16 标量循环 → 每
   token 24MB 慢速填充。修：x 一次转连续小端字节 + 逐槽
   `copy_from_slice`（memcpy 速率）+ 按形状 F 限量 clflush → 87.64。
3. pytest mmap EAGAIN（MAP_LOCKED 撞 memlock 上限）：须
   `sudo -n env PATH=… prlimit --memlock=unlimited:unlimited -- pytest`，
   与驱动无关（rmmod/modprobe 无效，P10 复现确认）。

### 结果（正确性 + 性能）

- pytest 6/6（30 项含 metrics）全过；E2E 门全同 v2：final hidden
  PASS（rms 0.0265 = 1.0%）、argmax 25868 对、top-8 8/8、lm
  rel_rms 0.0118。层 24 spot 检查 22/2048 出 1% 容差（worst rel
  3.09 在 bf16 舍入级小值上）——v2 同在，门未受影响，留观。
- **per-op（solo µs，流 GB/s）**：qkv 317/11.2 · o 254/9.6 ·
  gateup 1014/14.0 · down 539/13.2 · lm 9118/15.4（活字节口径）。
- **E2E：87.64 ms/token（11.4 tok/s）**，与 v2 87.2 持平。
  权重常驻 2.6GB→1.0GB，投影流 87MB→29MB/token。

### 关键发现：w4gemvu 是发射瓶颈，不是带宽瓶颈（P9 预测证伪）

lm_head A/B：v2 流 408MB/8.98ms=46GB/s（恰在 slot 天花板，P8 据此
判带宽瓶颈）；v3 流 136MB/9.15ms=15GB/s —— **流量降 3× 时间不变**。
每块耗时 ~7.3µs 与形状无关：768 组 × ~10 条向量指令 @1.8GHz ≈ 实测。
"省 35ms"的预测建立在时间∝字节上，被直接 A/B 证伪。v3 的真实收益：
**3× 带宽余量**（供未来重叠/更多列）+ 1.6GB 权重容量 + lm_head
同速下腾出带宽。

### 对 FLM 对标路线的修正

FLM 21.44ms 的构成是 ~448µs/层全融合 + lm 2.66ms。我们仅 4 个投影
solo 就 2.12ms/层——**即使 host/attention 全免费也 77ms 设备下限**。
要到 FLM 水平，唯一杠杆是内核循环本身：现在每 32 MAC ≈ 10 条向量
指令（unpack×2 + to_float×2 + mul×2 + mac×2 + 载入×2），理论
压缩到 ~4 条（int8 MAC 路径 / 双行共享 x 载入 / 常量折叠）→ 5× 潜力，
正好把投影 68→14ms、lm 9→2.7ms。**P11 = w4gemvu 内层循环重写**，
这是通往"不弱于 FLM"的主线；runlist 批量提交（−4ms）与 S-aware
flowkv 退居其后。

## P11：w4gemvu v4 矩阵单元内核 —— 投影设备时间 2.2×，E2E 87.6→58.4ms

**动机**（P10 结论）：v3 内层每 32 MAC ≈ 10 条向量指令，发射瓶颈。
aie2p 有矩阵单元：`mmul<4,16,16,int8,int4>` = `mac_4x16_16x16_conf`，
一条指令 1024 MAC。探针实测 131072 MAC/call（16× v3 块的 8192）约
1.3µs ⇒ ~100 GMAC/s/core，**每 MAC ~29–32× 于 v3 向量路径** → v4 把
GEMV 整个搬到矩阵单元。

### 设计（五件套 + 导入器 + Rust）

- **数值 ABI 变更**：激活改 int8 per-group-32 对称量化（d=amax/127
  bf16，q∈[−127,127]），内积变**精确 int32**（|qw·qx|·32 ≤ 2^15），
  sf·d 在 f32 域后乘累加（partial 可达 2^15，bf16 域会溢出/失精度；
  aie_api 无 bf16→f32 向量重载，用 `mul(sfb,db).to_vector<float>()`
  ——两 bf16 之积在 f32 精确）。golden 对量化后的 x 计算；Rust
  `w4u_quantize_x` 必须与 torch `quantize_vector` 逐位一致（RNE bf16
  + round_ties_even）。
- **块 = 18560B = 一个 16 行 × 2048k tile**：[0,16384) nibble 按 B 操作
  数序（组主序，元素 g*256+k*16+n，字节打包行 n,n+1）；[16384,18432)
  sf_t 转置 bf16[64][16]（行 n 的组 g scale 在 g*16+n，一次 32B 载入）；
  [18552,18556) K 头（恒 2048，guard）。内核每块一次调用：2×（载 x16
  int8 复制 4 份成 A[4×16] + 载 B 256×int4 + 2 条 mm.mac），64 组循环。
- **K=6144**：每 tile 流 3 个 chunk-块（chunk 主序，一个 B 槽的 2 块
  共享同一 x chunk），C 全行有效，主机 f32 求和 3 个 chunk partial 后
  舍入 bf16。B 槽 6528B = x int8(6144，活 chunk 在 0..2048) + d bf16
  [192]（活 64 在 6144）。ABI 统一 **M%256==0**（o 2048 免补、lm
  120818→121088）。
- 导入器 v4（`gemv_ref` 镜像设备分块求和）；Rust 常量/打包/C 收拢
  helper 化（`w4u_quantize_x`/`w4u_pack_slots`/`w4u_c_at`）。

### 踩坑（按时间序，都是硬约束）

1. **`concat` 两半必须等宽**：`concat(x0, concat(x0,x0))` 非法 →
   `concat(concat(x0,x0), concat(x0,x0))`。
2. **ELEM=18448（%32=16）**：奇数元素缓冲的 scale 载入读到垃圾
   （1e18 级值）——诊断内核把 K 头/nibble 和/sf 和回显到 c_out 证明
   **输入全健康**，垃圾产生于计算内。
3. **ELEM=18464（%32==0 但 ≡16 mod 64）**：scale 对了、nibble 流仍
   垃圾（有界错值）。aie2p（arch 21）`ld_st.hpp` vector_ldst_align：
   128b→16B、256b→32B、其余 **64B**；`load_v<256>` int4 = 1024 位
   ⇒ 需 64B 对齐，而 depth-2 fifo 两元素缓冲在 base 与 base+ELEM。
   **ELEM%64==0 是硬约束**，18560 收敛（18448/18464/18560 三指纹）。
4. **K=6144 golden 失配**：我按全 M chunk 截面排 C，实际 drain 按列
   生产序（col*3*rpc + c*rpc + w）——reference 的 shuffle/unshuffle
   与 Rust `w4u_c_at` 都按**每列 chunk 主序**重写后通过。
5. 诊断 pytest 必须放 IRON 树内（conftest 的 aie_context fixture），
   用完删除（test_debug_tmp.py）。

### 结果

- pytest 30/30 全过（6 形状 × 含 metrics）。
- **per-op solo（含 ~54µs submit，链上）**：qkv 3072 180µs/19.8GB/s ·
  o 157.5/15.0 · gateup 447/31.7 · down 276/25.6 · lm 3614/38.9。
  对比 v3（P10）：qkv 317 · o 254 · gateup 1014 · down 539 ·
  lm 9118 —— 投影层设备时间 **2.2×**，lm 2.5×。
- **E2E（CPU attention）：58.41 ms/token（17.1 tok/s）**，v3 87.64 →
  −33%。门全过：argmax 25868 ✓ top-8 8/8 ✓ lm rel_rms 0.0258 ✓
  final hidden rms 1.8%。层 16 起 19/2048 出 1% 容差（int8 激活量化
  漂移，worst rel 9.6 在小值上）——终局门不受影响，留观。
- **NPU attention 路径 130.7ms 倒退**：flowkv strided 档 1.07GB/s
  （2.1MB/1.97ms）×32 + w4gemvu(CU0)↔flowkv(CU1) 列冲突把 o solo
  157→766µs（≈650µs CU 切换税 × 64 次/层序）。CPU attention 反而快
  2.2×——attention 回 NPU 必须走**同 PDI 列内融合**，不能双 CU 交替。

### 差距分解（58.41 vs FLM 21.44）

设备 Σsolo ≈ 37.6ms（扣 submit 129×54µs≈7ms → 设备 ~30.6ms），
CPU glue（rope/attn/norms/swiglu/量化打包/clflush）≈ 20.8ms。
per-column A 流实测 2.9–4.9 GB/s（lm_head 4.9 封顶）：**单 shim
DMA 通道 ~5GB/s 是当前流上限**，8 列 → ~40GB/s 封顶；FLM 21.44ms
× 1.015GB 流量 ⇒ ≥47GB/s —— 他们必然用了**每 shim 双通道**。
**P12 主线：A 流双通道（每列 2 fifo 交错元素）→ 期望投影/lm 设备
~2×**；其次 runlist 批量提交（−5-7ms）、attention 同 PDI 融合。

## P12（2026-09-27）：A 流通道数假设证伪 + B 流才是真凶
- **假设**（P11 遗留）：per-column A 流 ~4.9 GB/s = 单 shim DMA 通道上限；
  16 通道（每列 2 fifo）→ ~2×。
- **探针设计**（design_2ch.py/test_2ch.py，PERF-ONLY）：B 流整体删除，
  权重按列切成 N_channel 段线性 fill（channels=1 → A 占 shims0-3 共 8 通道；
  channels=2 → A0 shims0-3 + A1 shims4-7 = 16 通道，npu_insts 实证）。
  fifo 元素=9280B（半块，%64==0）——内核 K 头守卫（w4gemvu.cc:80，读 +18552
  落在相邻 L1 buffer，垃圾≠2048）走零行路径：**线上字节与生产完全一致、
  核侧每元素几条指令**。2×18560×2 fifo 的 L1 放不下 → 半元素是探针成立的
  前提（4×18560=74KB > 64KB tile 内存）。
- **三处编译墙全记录**：①多元素 fifo L1 超限（元素必须 ≤~15.5KB）；
  ②多非平凡维 tap 降成**单个 BD**（每维 size 字段 10 位 ≤1023，rescale 老墙）
  且 **BD 维数上限 4**（2 tensor 维 + 至多 2 非平凡维）→ 交错 tap 不可能，
  改**按列线性分段**（自动链式 BD，生产同款，17MB 实证）；
  ③**C 数量必须=每 fifo 元素数**——channels=1 时核产 2×blocks 个 C 而 drain
  只要 blocks → C fifo 满死锁（fill 不完成→task_group 永等）。修=统一
  c_rows=cols×(seg/9280)×16。
- **结果（pytest 同 harness，5 iter 中位）**：
  | shape | 生产 v4 | 1ch 无B | 2ch 无B |
  |---|---|---|---|
  | o 2048×2048 | 207µs | 135 | 140 |
  | gateup 12288 | ~390* | 362 | 398 |
  | down 2048×6144 | ~210* | 221 | 258 |
  | lm 121088 | 3724 (36.7GB/s) | **2620 (53.6)** | 2674 |
  | 边际速率(o→lm 拟合) | 39.3 GB/s | **55.6** | 55.2 |
  (*E2E solo 扣 submit 估计，非同 harness)
- **结论一（假设证伪）**：1ch ≡ 2ch（噪声内）→ 8 通道已到墙；
  v4 的 36.7 不是通道数限制，**B 流本身**（313KB 槽 + 逐槽锁握手 + A 流
  2D 跨槽 tap）拖掉 ~30%。
- **结论二（墙的坐标）**：~55 GB/s ≈ FLM 层融合实测 48-64、seq-dma 校准
  52 → 这是器件级权重流天花板（DDR/NOC），不是通道数。FLM 没有通道魔法，
  他们的优势=每层 1 个融合内核（33 run/token vs 我们 129）。
- **v5 设计裁决**：单 fifo/列（ELEM 18560 保持，无 L1 困难）+ **B 流删除**：
  x 作为 A fifo 的**先导元素**（fill#1 从独立 X 张量读同一 18560B 区域×8 列）
  → 核 prologue `w4gemvu_xload` 拷入 .bss（6528B：x8+d），主循环读 .bss。
  副产收益：F 槽复制 memcpy（P10 曾 24MB/token）整体消失；权重 BO 纯块
  连续 → 单线性 fill/列（最快形态）。预期 E2E −7~8ms（权重流 25.8→18.3ms）。
- 探针文件保留（可复用 harness），随 v5 提交。

## P13（2026-09-27）：v5 落地 —— 计算墙现形、v5.2 预构建 A 操作数、E2E 58.4→56.9ms
- **v5 机器全绿但慢于预期**：B 流删除版（单 A fifo/列 ELEM 18560 depth2，
  x 先导元素 K=0 + 块自描述 K=2048/chunk 字），30/30 pytest 金标 PASS
  （x_stage .bss 24960B 放置核验 _ZL7x_stage@0x79580）。但全量 v5 每形状
  比 P12 1ch 探针高 ~30%（lm 3658 vs 2620µs）。
- **v5s 探针定位（test_v5s.py）**：把每个权重块 K 头改成垃圾值 0xBEEF——
  内核守卫走零行路径（几条指令），X 元素仍活（真实 staging+1 零 C）。
  线上字节/fill/drain/内核镜像全同。结果 lm 2645 ≈ P12 探针 2620 →
  **v5 填充机器（X fill 插入、2D C tap、18560B BD）成本≈0，差距=计算暴露**。
- **计算墙的坐标**：稳态节拍 = max(fill, compute)。fill 2.77µs/块
  （53.6GB/s 器件墙）；v5.0 每块计算 ≈3.8µs → **计算受限**，每块暴露
  ~1.07µs。P11 的 ~100 GMAC/s/核过于乐观：有效 8.4 GMAC/s。
  mac_4x16_16x16 在 M=4（aie2p int8xint4 唯一形状）烧 4× 冗余 A 行——
  GEMV rank-1 浪费是内在的。v4 的 3.93µs 节拍其实是 B 流+计算同时受限。
- **v5.1 反例（代码生成教训）**：外层 g4/内层 j 运行时 j → extract<16>(j)
  每组 vshuffle + d[g] 栈溢出（`p4=sp-0xc0`、动态标量载入）。o 反而从
  180 涨到 199、down 341。**规则：只有扁平 g 循环 + g 的地址算术可靠**。
- **v5.3 系列反例**：lambda+sfb4 extract 批处理 → 金标碎（16+ 行错）；
  带向量/累加引用的 helper 函数编译过但**核挂死**（栈 0x400/ABI 破坏，
  挂 ~7min 到超时）。全部回退。
- **v5.2 终版**：staging 时一次性预构建复制 A 操作数 [x0x0x0x0]/[x1x1x1x1]
  进 .bss（x_stage 24960B），热循环 A 侧 = 2 个裸 load_v<64> + 融合
  aie::mac(acc, rf, sfd)。暴露 1.07 → ~0.5-0.6µs/块；o 到地板（~155µs），
  down 299→264。
- **pytest 中位（干净窗口）**：qkv ~222 / o ~155-212 / gateup ~457-474 /
  down ~264-295 / lm ~3530（宿主负载 11.5 时 lm 噪声 3479-3990——E2E 设备
  时间才是决定性指标）。地板（v5s=P12）：o 135-155 / gateup 362-389 /
  down 221-237 / lm 2620-2645。
- **Rust E2E v5 改装**（main.rs）：w4u_pack_slots→w4u_build_x_elem（单
  ELEM：q 全 chunk @0..K、d@K_MAX、K=0@ELEM-8）；w4u_c_rows=8×(blocks+2)×16
  （标量行，BD 4B 对齐 → 每列截面偶数元素、+1 pad 元素）；w4u_c_at 新索引
  （跳 16 零行，K=6144 f32 求和）；x BO 两槽→两个 ELEM；f/qkv_f/W4U_B_SLOT
  全删。**夹具坑**：build/ 的 .prj 是 00:08 旧版（C 截面修复+v5.2 内核之前），
  PDI/ctrl 都要重拷 IRON/build —— 陈旧 ctrl 不会报错只会算错。
- **E2E（run-decode hy cpu，5 iter）**：稳态 **56.93ms/token（17.6 tok/s）**，
  med 55.17 / min 54.45。全门 PASS：hidden rms 0.0483（1.8%）、lm argmax
  25868==golden、top-8 8/8、rel_rms 0.0258。solo 中位 qkv 172.5 / o 156.0 /
  gateup 428.0 / down 270.0 / lmhead 3439µs。vs v4 58.41 → **−1.5ms**，
  小于 solo 差值预测（~4ms）——链上本已深流水（Δ=−58ms），单 op 收益被
  非 solo 段稀释；且 E2E 中下一 op 的 fill 可盖住 op 尾部计算暴露。
- **下一步**（按预期收益排序）：runlist 批量提交（129 submit×54µs ≈7ms
  host 开销）；attention 同 PDI 融合（FLM 33 run/token vs 我们 129）；
  CPU 侧胶水（rope/norm/swiglu 每层 ~0.2ms×32）。

## P14（2026-09-27）：v5.4 双累加器 —— 串行 FMA 链假设证实，E2E 56.9→49.6ms
- **假设**（P13 遗留 ~0.5µs/块计算暴露的嫌犯）：热循环 `acc = aie::mac(acc, rf, sfd)`
  是 64 深循环携带依赖（fpmac ~4-5 拍延迟 → 每块 ~130-160 拍纯等待）。
- **修法**：偶/奇组各一条独立累加链（flat 循环步长 2，全部索引仍是 g 的地址
  算术——P13 铁律不破），尾部和 acc0+acc1。
- **第一版撞墙**：`acc0 + acc1`（accum 相加）lowering 成 `G_FADD <16 x s32>`
  peano Legalizer 无法合法化（clang backend fatal）。**修 = 终和也走 fpmac**：
  `mac(acc0, broadcast(1.0f), acc1.to_vector<float>())`——本循环全程依赖的
  合法形态，accum 裸加法以后永远不要再写。
- **pytest（30/30 PASS）**：lm 3530→**2719-2722µs（50.1GB/s）**、gateup
  457→~400、qkv 222→~175、down 264→247、o 已在地板（146-180 噪声带）。
- **E2E（run-decode hy cpu）**：**49.56ms/token（20.2 tok/s）**，门全过
  （argmax 25868/top-8 8/8/rel_rms 0.0278/hidden 1.9%）。med 49.11。
  链上 solo：qkv 152 / o 141 / gateup 348（41GB/s，90% 档）/ down 229.5 /
  lmhead 2655（52.9GB/s ≈ 器件墙）。Σ 设备 ≈30.5ms，CPU 间隙 18.6ms
  （attention 13.4 + 胶水 5.2，trace 逐 op 分解：op 前 gap o=417.9µs×32 /
  down=105.9 / qkv=32.6 / gateup=22.6 / lm=33）。
- **性能台账**：v4 58.41 → v5.2 56.93 → v5.4 **49.56** ms/token。FLM 21.44
  → 差距 2.31×。设备侧剩余：小形状每 op 固定成本（o 141µs vs 裸流 44µs ≈
  ~90µs/op × 129 ≈ 4-8ms，融合奖池）+ gateup/down 距墙的 10-20%。
- **下一杠杆排序**（更新）：①attention 同 PDI 融合或 S 感知 flowkv（13.4ms
  CPU + 终态全 NPU 要求）②层内 4 GEMV 融合/批量（消每 op 固定成本）③胶水
  NPU 化。纯 runlist 批量在 CPU-interlocked 链上无收益（每 op x 依赖前 op C）。

## P15（2026-09-27）：CPU 侧两刀（位保真）—— E2E 49.6→41.0ms
- **trace 分解**（P14 后）：设备 30.6ms + gap 18.6ms，其中 attention 前 gap
  417.9µs×32=13.4ms、down 前 105.9µs、qkv/gateup 前 ~30/23µs。
- **attention_bf16 重写（位保真）**：K/V 行每 kv head 转 f32 一次（组内 4 个
  q head 共享——原来每次乘法都 bf16→f32，~8× 转换量）；q 每 head 转一次；
  PV 改 [t 外 j 内] 顺序累加（每输出维的求和序不变 → 逐位同结果，hidden
  rms 0.0492 / lm rel_rms 0.0278 与改前完全一致）。gap 417.9→192.9µs。
- **w4u_read_c**：整 op C 读改成按列截面顺序遍历（w4u_c_at 每行 div/mod
  +字节拼装是 C 读真成本），值与序不变。
- **E2E**：**41.01ms/token（24.4 tok/s）** med 41.09，门全过；设备 solo 不变
  （qkv 153/o 143/gateup 346/down 231/lm 2674µs）。gap 剩 10.3ms：attention
  6.2（K/V staging+dot）+ down 前 2.7（swiglu 6144 次 expf 为主）+ 其余 1.4。
- **设备侧结构定论**：Σ设备 30.6ms vs 纯流地板 18.9ms（1.015GB@53.6GB/s）
  = **每 op 固定 ~90-100µs × 129 op**（marginal 拟合 0.335µs/块=55GB/s 在墙
  上；intercept 拟合 o 143−128×0.335≈100，gateup 89，down 102）。固定成本
  嫌犯 = ctrl/DPU 逐 fill-drain-wait 的发行时延（ctrl .bin 所有形状同 3248B
  → 发行步数与 M 无关）。
- **M6 架构判据（下会话主线）**：v5 自描述元素 + drain-wait 定序的 ctrl 使
  **层内融合可行**——胶水算子成为元素口味（rms 需全向量：8 列冗余算+经
  DDR 往返；swiglu 列内本地但 gate/up 配对要导出侧重排列；attention=flowkv
  元素化）。每层 1-2 op 替代 4+胶水，消 ~95µs/op×~96 + 全部 CPU gap。
  今晚不做；CPU 侧再抠（swiglu expf 向量化 ~2.7ms、f32 KV 缓存 ~2ms）皆为
  脚手架，与终态（全 NPU）冲突，搁置。

## P16（2026-09-27）：M6 融合 rms-pair 机制点亮 —— 层内融合首证 + STACK LAW 实锤
- **目标**：op1(gemv)→add-residual+rms+逐组32 int8量化→op2(gemv) 合一 exec，
  拿掉 per-op 固定 ~95µs 与 CPU gap（P15 判据：v5 自描述元素 + drain-wait
  定序使融合可行）。
- **机制（landed）**：K=1 元素（tg2 首 fill）= C 窗口 [res | headers]——tg1 的
  drain 已落到 C，DPU 程序序即跨 op 屏障；stage1 把分块 partial 紧凑化 +
  residual 暂存到死区。K=3 元素（op2 权重 fill 的头）= rms 权重；fifo 序即
  屏障，胶水先于一切 op2 块运行；op2 完全没有 X fill（内核直接预构建 A
  操作数）。8 核冗余跑全 2048 行胶水。
- **ctrl 律**：两 task group 都必须是 4 fill + 2 drain 的 v5 形状——S2MM
  drain 的 queue 值钉在通道 slot 4/5，前置 fill 超 4 个即 value/描述符地址
  失配挂死（首次尝试整跑挂死的根因）。
- **STACK LAW（0x400）**：peano linker 每核只给 0x400 栈窗，A-fifo buffer
  直接叠在栈顶上方；帧 = 序言一条 `paddxm [sp], #N`。单体 M6 内核 0x440 >
  0x400 → 溢出写坏 fifo 流 → 首次 exec 即挂。修法：0x20 dispatcher 读 K 头
  tail-call 各 noinline 口味（热路径保 0x340）。
- **标量胶水结果**：正确（golden 镜像 f32 语义）但 **1673µs**——probe（关
  胶水口味）机制地板 **497µs**，胶水占 **1176µs**：peano 每 f32 标量 op 都是
  soft call，~15/行×2048 行/核。→ 融合的胜负手在胶水向量化（P17）。

## P17（2026-09-27）：胶水向量化 + P17b 帧成本定律 —— 双参数全过，o+gateup 485µs
- **向量化（只用实证可降形的 op）**：mul(bf16,bf16)→accfloat（bf16→f32 的
  精确 trick）、mac(accfloat, f32vec, f32vec)（裸 acc+acc 非法，经 mac 过
  桥）、to_vector、load_v/store_v、concat、broadcast、set_rounding(conv_even)。
  首版单体向量化 **483µs @ 34.74GB/s**（超地板 497 之下、破 split 506）但
  op2 区全错：帧 **0x480 > 0x400**，STACK LAW 二次咬人——溢出写坏胶水期
  间暂存在栈上方 A-fifo 的 op2 权重元素。
- **P17b 帧成本解剖（离线 clang+objdump 秒级迭代，~22 个消融变体）**：
  to_fixed<int32>(f32vec) ≈ **0x600 禁用**；abs/reduce_max +0x140；同函数内
  标量读向量 store +0x140（跨函数免费）；int min/max +0x80；纯向量函数
  **0x0 帧**；帧大小非单调（小扰动会随机变胖 0x480→0x5c0）。soft 除/rsqrt
  本身不是驱动者。
- **两个语义精确的替换**（golden 不动）：(1) int8 取整 = invd 缩放后加魔数
  1.5·2²³（该指数下 ulp=1 → conv_even 加法即 RNE 到尾数里的整数），q = bits
  − 0x4B400000 再标量 clip（先取整后 clip，宿主同序）；(2) amax = 对
  (bits & 0x7fffffff) 取**整数 max**——非负 IEEE 浮点位序=值序，精确等于
  f32 max|x|，零浮点 op。
- **编译坑（>10min 卡死 clang）**：u32 循环与 soft-float 调用混在同一函数
  （v22 stage2c）；load_v/mac 环里 broadcast 运行时载入的标量。对策：int 与
  float 各自成 stage；逐组 invd **复制展开**（f32[64][32]）使缩放 pass 纯
  load_v/mac/store_v。
- **终态结构**：stage2a(h2+sumsq, 0x0) → 2b(xn 向量, **0x200**) → 2c(整数
  amax, 0x40) → 2d(d+invd, 0x40) → 2e(缩放+魔数, 0x0) → 2f(q 提取+预建,
  0x0)，0x40 sequencer；最坏链 dispatcher+0x40+0x200 ≈ 0x280 < 0x400。
- **结果（两参数 5/5 全过）**：o+gateup **485.6µs @ 34.55GB/s**（med ~513，
  split 506 之下/持平）；down+qkv **med 394µs**（best 372.7，split 409 之下）
  。正确性 = golden 镜像（tol rel 0.07 / abs 0.7）。旧标量融合 1673µs →
  3.4×。
- **结论**：层内融合机制+向量化胶水成立；下一层收益在把两个融合对塞进
  decode 链（E2E 每 op 固定成本 ~95µs×可消 2 个/层 + gap）与 attention/
  swiglu 元素化（M5 计划遗留）。

## P18（2026-09-27）：融合对进 decode 链（run-decode fused）—— E2E 41.7→37.5ms
- **接线设计**：`run-decode hy fused`（hy-only，qkv_m 3072 形状契约用
  `assert!` 非 debug_assert——release 会编译掉）。PDI 分槽：plain w4gemvu
  CU0 slot0 + w4gemvuf CU1 slot1；npu_attn 时 flowkv 再挪 CU2 slot2（三
  PDI 三 CU 共存首证）。链拓扑：L0 qkv plain → 每层 pair A(o→ln2_n→
  gateup) → n<L−1 时 pair B(down→ln1_{n+1}→qkv(n+1)，留在 Scratch.qkv
  供下层) → 尾层 down plain。63 对（32A+31B）替掉 126 个 split exec：
  w4gemvu 族 128→65 exec/token（cpu 模式总 op 129→66）。
- **宿主侧每对契约**：act 量化进共享 x 元素 BO（与 plain 同路）；residual
  4KB sync 进 C 行 [c_rows1..+2048)；K=1/blocks1 头部 setup 期一次性写好
  （chain_tensor 全 BO sync 捎带）；handles=[ctrl,p1,p2,x,c]（rt.sequence
  (A1,A2,X,C)）；读回：op1 区 = plain v5 布局同 `w4u_read_c`，op2 区 =
  9280 行窗口后每列 section2、跳 2 个 16 行哑元。
- **数值口径**：宿主残差链 bit-exact 于 split（同 w4u_read_c 值 + 同
  add_bf16）——残差流零漂移；唯一差别是 op2 量化输入携带 NPU 胶水 h2 的
  f32 少一次舍入（≤1 ulp）。实测 fused final rms **1.3%**（baseline 1.9%）
  ，lm argmax 同 token、top-8 8/8——两门全过且反而略好。
- **闭包借用架构（编译约束留痕）**：`live` 改为 gemv/gemv_pair/
  decode_step 的参数（无闭包捕获）；`live_fused`+`fst` 仅被 gemv_pair 捕获
  ；gemv_pair 内先 `fst.as_ref()` 读常量（NLL 结束借用）再 `as_mut` 取
  ops/handles 字段不相交借用。
- **踩坑**：(1) build/ 顶层 root 属主，fixture `sudo cp`（无密码 sudo 实证
  ）。(2) op 计数公式首写 2L−3=61 错——真实 63=2L−1（每对净消 1 exec），
  burst_done/E2E 打印两处修正。(3) NPU-attention 基线 121ms/token 与 P15
  的 41ms 不可比——41ms 是 **cpu attention** 口径（129 op）；npu 口径被
  flowkv strided ~2ms×32 支配。A/B 必须同口径。
- **结果（四配置）**：cpu 41.72→**37.55 ms/token（24.0→26.7 tok/s，
  −10.2%）**，双门 PASS；npu 121.30→118.38（同省 ~3ms，flowkv 主导）。
  链上 per-op solo：fused_o_gateup **472µs**（独立 485）、fused_down_qkv
  **369µs**（394）——独立对性能在链上复现。尾层 down solo 741µs（n=5，
  CU1→CU0 切换后首 exec，待查）。
- **收支分析**：省 4.2ms vs 理论 63×~95µs≈6ms——差额 = 尾层 down 变慢
  ~0.5ms + 每对新增 host sync（4KB residual ToDevice + 全 C BO FromDevice
  ioctl ~44KB）。
- **下一步**：attention/swiglu 元素化（M5 遗留）——cpu 口径剩余 gap =
  CPU attention 6.2ms + swiglu 2.7ms；npu 口径瓶颈 = flowkv strided 1GB/s
  需重排布。
- **补记（尾层 down 741µs 结案）**：CU1→CU0 翻转后首个 exec，按 P9 机制
  = ~500µs fw PDI 重载计入该 op solo（232 基线 + 税）。fused 链每 token
  2 次翻转（首 plain qkv CU0→CU1、尾 down CU1→CU0）是结构性的（两 PDI
  异 CU），非 bug。

## P19（2026-09-27）：四联 exec（quad）——每层一个 exec，设计先行
### 设计思路
**拓扑**：把 pair A+B 合成每层一个 exec：
`[o →rms1→ gateup →swiglu→ down →rms2→ qkv(n+1)]`，31 个 quad（n=0..30）
+ L0 plain qkv + L31 pair A + 尾层 down plain（沿用 P18 夹具）。exec 数
65→**34**（−31×~95µs 固定成本 ≈ −2.9ms）+ 消宿主 gateup 读回/swiglu/
量化/sync 路径（~140-150µs/层 ≈ −4.4ms，其中 swiglu 标量本体 2.7ms）。
目标 37.55 → **~31-33 ms/token**。swiglu 必须上设备是本设计的根本前提
（exec 内 host 往返不可能）。

**C 布局（行）**：o sections 2304（16 blk/列）| win1 9280（residual1 @
+2304，hdr K=1/blocks1=16 @9276-9279）| gate cols0-3 6272 | padA 3008
（宿主写 K=4 hdr @窗口行 9276-9279）| up cols4-7 6272 | padB 3008（K=5
hdr）| down sections 6400（48 blk/列）| win2 9280（residual2 @+6400，
hdr blocks1=48）| qkv sections 3328（24 blk/列）。合计 49152 行 = 96KB。
padA/padB 存在的原因：C-sourced fill 必须产整元素（9280 行），而
K=4/K=5 头部字（元素尾 ELEM-8/ELEM-4 = 窗口行 9276-9279）若落在
live C 数据上会破坏 gate/up 输出——垫 3008 行宿主区，头字写进垫里。
op2 的 drain 逐列偏移带 gap（col≥4 偏 +3008）。

**tg/BD（≤4 fills/shim 律全过）**：tg1=[X, A1(o 块)]→C1(o) wait；
tg2=[win1(K=1), A2(w1 K=3 元素+gateup 96 块)]→C2 gateup sections wait；
tg3=[swA(K=4 gate 窗), swB(K=5 up 窗), A3(down 48 块，无 X)]→C3 down
wait；tg4=[win2(K=1), A4(w2 K=3+qkv 24 块)]→C4 qkv。宿主每层写
residual1+residual2（8KB）+ 一次性 padA/padB 头字；sync 策略=整 C BO
一次 ToDevice（96KB clflush，1 ioctl/层，比 P18 每对 2 次更省）。

**内核新口味**（K=1/K=3 原样复用——quad 的 win1 blocks1=16 同 pair A、
win2 blocks1=48 同 pair B）：
- **K=4（fused_stage3a）**：gate 窗（cols0-3）16 宽向量拷贝到
  x_stage[0..12288) bf16——纯向量，stage1 同形。
- **K=5（fused_stage3b）**：up 从 a_in（j→u16 idx (j/1536)*1568+32+
  j%1536），gate 从暂存；逐 32 组：sw = g·sigmoid(g)·u，f32 向量；
  sigmoid = 1/(1+2^{−g·log2e})（**硬件 exp2<bfloat16>，相位 1**，P4
  偏差 mean+3.25%/max+5.67% 已知，E2E 双门裁决）；向量 f32 除法帧成本
  未探（回退备选 tanh<bfloat16> 初等指令：sig=0.5+0.5·tanh(g/2) 免除法
  ，同属硬件初等带偏差）。组内 amax=整数位序 max，d=bf16(amax/127)，
  q=魔数加取整（P17 全套 trick 复用），**A 操作数直接按复制形态写目标
  [128·gi)**（免中间 q 数组）；d[gi] @ kDStageOff+2gi；r 暂存 128B
  temp @24960（x_stage 扩 128B，L1 预算 63.3KB<64KB）。

**scratch 冲突消解（关键推导，写错就静默错数）**：K=5 单次调用内
gate 暂存 [0..12288) 与 A-ops 目标 [0..24576) 同域。**块序倒序**
（c=2,1,0）：A-ops 块 c 写 [8192c..8192(c+1))，未读 gate=[0..4096(c+1))
恒在写区间下方 ✓（c=1 写 [8192..16384) vs 未读 gate [0..8192)；c=0
时 gate 已全消费）。组序亦降序（gi 191→0：A-ops 写 [128gi,+128)，
未读 gate ≤64gi+64 ≤ 128gi ✓）。sw f32 现算现用不落盘（寄存器），
r/q 经 128B temp 中转。per-group broadcast(invd) 是已知编译 hazard
（load_v/mac 环内 broadcast 运行时标量）→ **逐组 noinline 函数**
（broadcast 在函数入口、无循环），sequencer 192 次调用；帧超 0x400 再
对半拆 v/s 两函数。

**风险清单**：(1) 向量 f32 除法/exp2/tanh 的可降形与帧成本——离线
clang+objdump 先探（v24.cc）；(2) exp2 偏差经 31 层复利超 5% 门——
E2E 裁决，回退=换 tanh 或（最坏）swiglu 留宿主+双 exec 妥协形态；
(3) C-sourced fill×4（win1/swA/swB/win2）的 tap 维数/stride——照抄
design_fused 的 C1_taps 形；(4) op2 drain 逐列 gap 偏移是新 ctrl 形，
首次跑挂死优先查 BD slot 律；(5) per-group 函数代码量 192 调用点。

### 尝试步骤（进行中）
1. **离线探针 v24 系（P17b 方法，全部留档 /tmp/v24*.cc）**：
   - API 事实：`aie::exp2<bfloat16>(vector<float>)`/`aie::tanh<bfloat16>
     (vector<float>)` 都是 XDNA_2 门控、**float 进 bf16 出**；
     `aie::div(v,v)` = mul(a, inv(b))，inv 是 ElementaryOp，返回直接是
     vector（非 accum）。
   - v24 单体（向量 sigmoid + 标量 amax/quant/提取同函数）：**>5min 挂死
     clang**——P17b「u32 循环混 soft-float」定律再犯确认。
   - bisect（v24a/b/c）：exp2+div 向量形 2.8s/0xc0（入口含栈上 temp 数组
     的探针伪影，内层向量函数 0x0）；tanh 形同；**挂死全在标量尾**。
   - v24d（每 chunk 相位化）：64 迭代 sigmoid 环编译出 **0x400 帧**（~9 个
     活 f32 向量寄存器压力溢出）——链溢出。
   - **v24e 终态（设计随之简化）**：sw chunk 暂存整个取消，改**全 per-group
     流水**：sig(0x0)→amax(0x0)→quant(0x280)→x(0x40)，stage3b 序 0x40，
     链 0x20+0x40+0x280=**0x2e0 ✓**，4.7s 编译。scratch 只剩 gate 暂存
     [0..12288)（降序 gi 被 A-ops 覆盖：写 [128gi,+128) 恒在未读 gate
     [64gi,+64) 上方）+ 专属 128B temp @24960（x_stage 扩至 25088，
     L1 63.3KB）+ d[192] @kDStageOff。**降序 gi 是第 14 次推出同一结论：
     任何升序都会写穿未读 gate。**
2. **设计复核揪出两个错误（写 design 前闭卷重推 C 布局时发现，都改在设计里）**：
   - **残差 2 宿主不可算**：residual2 = x'_n = x_n + o_out，而 o_out 是本
     exec 内 tg1 才算出来的——P18 能算是因为 pair A 与 pair B 之间有读回。
     原设计「宿主每层写 residual1+residual2」对 residual2 不成立。**解法 =
     tg4 加第 3 个 fill：win1 窗重读**（o sections 在 C 里 tg1 drain 后一直
     有效）：tg4 = [win2(K=2), win1-re(K=1'), A4(K=3+24块)]。新口味：
     - **K=2（win2 变体）**：down sections 有 **2 个哑元组**（K=4/K=5 各产
       1 个零 C），stage1 的 +16 跳过不够 → +32 变体 fused_stage1b；win2
       hdr 是专属 pad 行，宿主直接写 K=2 字，无需 flag。
     - **K=1'（win1 重读，flag 派发）**：重读窗=[0,9280) 行，其 hdr 字节
       就是 tg2 用过的 K=1（数据即字节，无法改字）→ 用 **quad flag**：
       stage3b（K=5）尾部写 flag=1 @temp+124（24960+124，K=5 循环后）；
       K=1 派发见 flag → fused_stage1r 并清零。pair 流程 flag 恒 0（唯一
       set 者是 quad-only 的 K=5，唯一 consume 者是其后第一个 K=1），向后
       兼容。stage1r：o partials 紧凑化到 [4096,8192)（chunks=1，4096B），
       与元素内 residual1（窗行 2304..4352，连续）向量求和 → h2' bf16 存
       **kResStageOff**——正好是 stage2a 读 res 的位置，**K=3 零改动**；
       stage1r 先跑（把 stage1b 暂存的 res2 覆盖成 h2'，x_n 不再需要，
       宿主不写 residual2）。顺序闭环：stage2a 读 res=h2' + partials=
       down（stage1b 暂存 [8192,20480)）→ h = x_{n+1} = down+x_n+o_out ✓。
       曾考虑把重读放 tg3（K=5 后同组）——**死路**：K=5 的 A-ops 降序写
       [0,24576) 会踩 h2'；重读必须在 tg4。
   - **gate/up/down 窗口全是 2 哑元，内核 +16 全错**：gateup 段 C 元素 =
     K=1 零 C + K=3 零 C + 96 块 = 98 组，真值在 [32,1568)——已写的
     stage3a/stage3b 用了 +16（照抄 stage1 的 1 哑元形），要改 **+32**。
     同理 qkv 段 3 哑元（win2 零 C + win1' 零 C + K=3 零 C）：27 组写入、
     stride 28 组（448 行）、宿主读 section+48。
   - **C 总行数勘误**：原记 49152 行是把 down sections 与 win2 重复计数；
     win1/win2 都内含各自 op1 sections。实际 **40704 行 = 81408B**：
     [0,9280) win1(o 2304+res1 2048+pad+hdr@9276) | [9280,15552) gate
     cols0-3 | [15552,18560) padA(K=4 hdr@18556) | [18560,24832) up cols4-7
     | [24832,27840) padB(K=5 hdr@27836) | [27840,37120) win2(down 6400+
     res2 2048@+6400+pad+hdr@37116,K=2/48) | [37120,40704) qkv sections
     （8×448，27 组写入）。op2 drain 逐列 gap：col≥4 偏 +3008 行。
   - tg4 变 3 fill（win2/win1-re/A4）后 ≤4 fill/shim 律仍全过（2/2/3/3）；
     P16 律按「每组 drain 前填充链 ≤4」读，quad 全部满足（若板上挂死，
     回退=把 tg3/tg4 各拆两组，全回到已证 2F+1D 形）。

## P19b（2026-09-28）：quad 点亮全程 —— PMEM 16KB 律、5-BO 上限、缺零 C 红鲱鱼、hw sigmoid 落地
### 尝试步骤与成败
1. **PMEM=16KB 是 CDO loader 的硬律（第三次实证）**：peano ld.script 声称
   program LENGTH=0x20000，但 `_XAie_LoadProgMemSection` 实际只装 16KB。
   链接胶水（crt0+MLIR core wrapper）恒 **3052B**（手工 aiecc 复现法：
   mlir+w4gemvu.o 拷进同一 cwd，`--aie-generate-npu` 后看 .prj 里 ELF）。
   三轮超标 18036→14868→13748→**11476**（.o text）：
   - 轮1：sw_amax/sw_x/sumsq-reduce 加 `#pragma clang loop unroll(disable)`
     （clang 全展开小标量循环，32 次比较链 ~0x1d0、8 存储体 ×16 展开
     ~0x410——展开是 PMEM 之敌）；
   - 轮2：stage1/stage1b 合并共享 `fused_stage1s(skip)`；
   - 轮3：共享 `fused_h2make`（stage1r/stage2a 同一 h2 循环）+ 共享
     `fused_zero_c`（5 个口味各自的 16 存零尾 ~100B×5）+ sumsq 标量归约
     roll。热循环 w4gemvu_compute 2368B 一字未动。
2. **NPU ctrl kernel 签名上限 5 个 BO（新律）**：mlir_aie
   emit_design_kernel_json 硬编码 `[f"bo{i}" for i in range(5)]`
   （aiecc/main.py:180-190）。第 6 个张量 → 宿主 prepare_runtime 在
   `xrt::run::set_arg_at_index` SEGFAULT。修法=P12 老招：X 激活元素骑
   packed1 头部（6→5 BO：packed1..4+output），rt.sequence 序 A1,A2,A3,A4,C。
3. **npu_insts.mlir 可读（关键调试杠杆，新发现）**：aiecc 保留
   `.prj/npu_insts.mlir` = 事务层 MLIR（write32/blockwrite/maskwrite/sync
   全可读）。由此实测寄存器语义：每 shim 2×MM2S+2×S2MM 通道（8 核 2 列
   ×4 行，4 shim 各带 2 核）；BD 槽号=**每组从 0 重数**（fill+drain 共用
   计数）；MM2S 起始寄存器逐个 enqueue 槽号；S2MM 队列值=槽号；
   `dma_free_task` **不产生任何指令**（纯编译期槽回收标记）；组末
   npu.sync 只等 drain 方向（fill 完成由 drain 完成+核消费传递隐含）。
   tg3/tg4（无 drain 组）fill 后无任何屏障——本次不是问题源头，但此
   「无 sync 组的槽重用」仍是未证形，后续设计慎用。
4. **红鲱鱼：『tg2 gateup 块重投递』两天的误诊**：首板现象=down 段
   dummy0 全 8 列垃圾，值精确等于 gate col0 倒数第二块的已存 C（行
   10816 配对命中）→ 以为是 BD 槽/队列碰撞（P16 律重演）。拆 tg3/tg4
   （3F+1D→2F|1F+1D）**无效**→ 排除组结构。真相：**fused_stage3a
   （K=4）漏写 fused_zero_c(c_out)**——C 元素契约（每个 A 元素恰产
   一个 16 行 C）被破坏，tg3+tg3b 只产 49 C 对 drain 50 组，饿死的
   S2MM 首组读了**陈旧 L1 fifo 缓冲**（恰是最后一块 gateup 的 C）再
   自行对齐。一行修复（补 zero_c）后 dummy 全零、结构全对。
   教训：值配对定位到「重投递」时，先数**元素收支**再怪 BD。
5. **hw sigmoid 落地量化（相位 1 裁决数据）**：golden 用精确 exp，
   内核用 exp2<bfloat16>。down 段 partial 是 ±8000 量级的抵消和，
   组间 2-3% 微分偏差按**项规模**传播：max_abs=192（≈2.4% 项规模）、
   max_rel=41（近零结果无界相对误差）、47/6144 行超 rel1.0/abs10。
   但 rms2 重归一吸收：qkv 段（吃这些 partial）偏差 ≤**0.75 绝对**。
   测试带：down 段 rel0.08/abs200（其余段 rel0.08/abs0.8 严格），
   E2E 双门仍是数值最终裁判。若 E2E 失败，回退=div 换
   tanh<bfloat16>（sig=0.5+0.5·tanh(g/2)，免除法，同族偏差）。
6. **板上偶发漂移复现**：3 次 pytest 中 1 次 o/gate 段报错（不同行），
   另 2 次+复跑 15 次全过——即积压的「run 间漂移」在 quad 长事务上
   更易现。root cause 仍欠（backlog），E2E A/B 取多 run 口径。
### 结果
**test_quad 全过（15/15 稳定）**：quad 单发 **~1063-1118µs（均值
~1087µs，25.5 GB/s 权重流）**。对照：P18 pair A+B 独立和=879µs+宿主
swiglu/量化/往返（E2E 每层 ~1173µs）——quad 单发看似更贵，但省掉一次
exec 提交+宿主回读往返；净赚多少由 E2E A/B 裁决（Rust `quad` flag 下
一步）。

## P19c (2026-09-28): Intel NPU 开源调研(跨厂商借鉴)

用户命题:不限 AMD,调研 Intel NPU(MTL 37xx/LNL 40xx)开放资料里可移植的优化思路。
全文见 `notes/intel-npu-survey.md`(来源分级:driver 头文件=硬契约 / OpenVINO=官方 / 媒体 / 逆向)。
最有价值的不是文档而是 **linux-npu-driver 的 firmware/include 头文件**:WLM(workload
management)、DMA 描述符、NCE 寄存器、CMX 布局全是硬事实。

对我们最有用的五条(按与 65-exec 问题的相关度排序):
1. **invariant/variant 两级描述符**(260B 层静态 + 44B workload 动态,LUT 关联,从 DDR
   预取进 CMX 固定槽双缓冲)→ 直接对应我们的 16-BD 槽 + 每 exec 重发:描述符能分层复用,
   层间只换 variant。
2. **同步前移数据化**:编译器把整图编成 work items + barrier 重编程表,DMA 任务自己往
   引擎 FIFO 喂描述符,barrier 编程也可 DMA 化(ALL_BARRIER_DMAS_SCHEDULED 运行时零参与)
   → 固件/host 彻底退出热路径,是 65-exec 问题的正面答案(XDNA2 上等价物 = 把 task-group
   链编进一个 ctrl 序列,host 每 token 只提交 1-2 次)。
3. **写回侧 swizzle**:ODU 带 swizzle_key/permutation、IDU nthw_ntk 布局,硬件做 transpose
   → 对应 flowkv 的显式转置开销(strided 主导 118ms 的 npu 口径)。
4. **PPE/ODU 融后处理**:scale/bias/prelu/LUT/dtype 融在矩阵消费侧 → 与我们 rms/swiglu
   融进 GEMV 消费端同构,佐证 P16-P19 路线。
5. **latency/throughput 双拓扑编译期选择**:LATENCY 用满 tile、THROUGHPUT 少 tile + 8
   outstanding requests → 我们的 8 核 decode 是 LATENCY 型,可借鉴其"多 tile vs 深 FIFO"权衡。

另:barrier 每组 32/16、64 位 prod/cons 掩码;CMX 描述符槽 256 DMA/32 inv/256 var(37xx);
DMA 描述符 80B/64B 对齐/链表式;I4/U4/FP8 + pallet[8] 权重调色板;npunlock 证实 blob=ELF+
MMIO preactions(可在 MTL 上写 C 编 ACT-SHAVE)。不开放别追:固件二进制、barrier FIFO
深度、STT/SIF 互连、SHAVE 工具链官方分发、NPU5 细节。

## P19d（2026-09-28）：quad E2E 点亮 —— missing-down host bug、双门 PASS、间歇 hang 未解

1. 现象：E2E 雪球 L0 64/2048 → L1 196 → L2 705 → 垃圾（final rms 2.639 = golden 能量
   的 100.1%、max err 105 —— "同能量、不相关"），且 ~50% run 在随机层 hang。
2. 定位手段（可复用）：
   - in-process 对拍：quad(0) 的 o sections vs plain op = **0/4608 bytes differ** → o 路径 bit-exact；
   - QUAD_DUMP 逐层原始 qkv(n+1) 读回，quad vs fused 对拍：v-slice max_abs **2e-4**（L1）
     → 5e-3（L10），L11 才跳 0.18 → qkv 读回与数值全程正确，排除读回 bug；
   - 决定性一步：golden_L00 − (x0+o) 就是 down(0)：max 0.035、mean 0.0073 ——
     L0 的 64 行"超差"正是**缺失的 down 本身**。
3. 根因：gemv_quad 的 host 残差更新只做 x+o，**漏了 down**。device 内部 h2''=(x+o)+down
   是对的（所以 qkv 全对），host 轨迹每层丢一个 down → Σdown 逐层累积 → L11 起
   attention 去相关 → 雪球。早先"L0 64 行 = hw sigmoid 数值类"的判断是**错的**：
   hw sigmoid 的微分误差实测只有 ~2e-4 量级（v-diff），phase-1 预留的 tanh fallback
   **不需要**。
4. 修复：读 quad C 的 win2 down sections（base **27840**、8×800 行、2 dummy groups、
   chunk-major 3×256），x_{n+1} = bf16(f32(bf16(x+o)) + Σ_c f32(p_c))，与
   golden/device 的舍入顺序一致。（第一次把 base 写成 37116 —— 那是 K=2 header
   words 的位置，越界 panic 暴露。）
5. 结果：**双门 PASS** —— final hidden rms 0.0454（golden 2.635 的 1.7%）、
   lm_head rel_rms 0.0221、argmax 25868 一致、top-8 8/8；逐层 ≤16 bad rows
   （无 debug 行触发）；跨 run 数值完全确定。
6. 性能（run-decode hy cpu，5 iters）：quad **43.76 ms/token** vs fused 38.46 ——
   回归 +5.3ms（用户裁定：向 NPU 最大化路线的短暂回归可接受）。分解：quad exec
   solo 1066µs × 31 = 33.0ms；serial est 39.7ms vs fused 59.0ms（**串行口径省
   19.4ms**），但 quad 链 Δ=+4100µs（零流水 + host 开销）vs fused Δ=−20582µs。
   已知可改：(a) X 重写现在整 BO clflush 2.5MB/exec，应只 sync 8 个 head 区
   148KB；(b) K=5 glue 计算成本；(c) 层间 submit 流水化。
7. 遗留：**间歇 hang**（~1%/exec，Timer expired，dmesg 无驱动错误）——与数值无关
   （gates pass 后 timed iters 也 hang），quad 专属（fused 全天稳定），命中层随机
   （L12/L15/L29，seq 13/100/191）。疑点：5 task-group 链 / win1 re-read /
   BD slot 交互。→ P20。

## P20（2026-09-28，IRON design_quad.py 未提交 + probe）：parked-Cs 设计修复

1. 设计缺陷（tg3/tg4 跨 task-group 停留零 C）：fill-only 组（K=4 gate 窗口、
   K=5 up 窗口、K=2 win2、K=1' re-read）各自在 depth-2 C fifo 里留下零 C，
   等下一组的 drain 当 section dummy 消费 —— 这个跨组交接在 ~1-3%/exec 上
   竞态失败（P19 指纹：整列 down/qkv chunk 缺失，首缺行总是 section 首数据行）。
2. 修复 = 回到 pair 已证形态：**每组自带 fills + 自己的 drain（2F+1D）**。
   tg3 drain glue3→padA 死区（GLUE_C3_OFF=15600）、tg4 drain glue4→padB
   （24880）；C3 只盖 48 down blocks（+2 dummy 行保持 c_init 零）；C4 重排为
   [K=3 头零 C | 24 blocks] = 25 组 @+2 dummy（block 0 落 row 48）——A4 头元素
   的组内消费定律（与 C2 同型）。逐列元素收支 192/192 平衡。
3. 自坑两枚：C4 第一次写成 blocks_q+3 dummies → qkv 移位一组（ZZZZ X×23 Z
   指纹）；pytest "Mismatch in output[17]" 的索引列表是 qkv 幅度不是 o 行
   （debug_quad 证 o bit-identical，勿信错误列表索引当 C 行）。
4. 验证：pytest 5/5；quad_probe2（pyxrt 500-iter loop）clean。
5. **E2E 仍 ~1.7%/exec hang** → 引擎路径专属，设计修复没碰到 → P20b。

## P20b（2026-09-28②，xnpu main.rs/ert.rs）：引擎路径 hang/corruption 三周目
   ——真凶=夹具陈旧（P13 复犯），XRT 线格式全档补录

1. `run-quadloop`（引擎侧 standalone：单 PDI cu0、同 BO 反复 submit+wait、
   与 iter0 快照逐字节比对）：plain 模式 ~10%/iter **静默腐蚀** + 偶发 hang
   （wait 超时、C 全零 = exec 从头卡死）——零 host 写、同包同址，纯提交机制
   复现 P19 症状。
2. 站不住的假设逐一排除（都有实验）：`sleep` 2ms（hang it62，更糟）、`syncs`
   （pyxrt 式 6 个整 BO To_DEVICE 屏障，div it70 —— 腐蚀指纹与 plain 同型
   不同位）、`fresh`（每 iter 重建 ctrl BO+ERT 包，div it37）→ **非 host 时序、
   非 posted-write、非包头/ctrl 复用**。
3. 重建 M1 ioctl 拦截器（/tmp/xdump2，Rust cdylib：拦 ioctl+mmap，
   CREATE_BO→GET_BO_INFO→mmap 链解析 handle→VA，EXEC_CMD 逐字节 dump +
   链式子命令展开）抓 pyxrt 真包，**XRT 线格式全档**（比 M1 多一层）：
   - ioctl：ty=0、cmd_count=1、cmd_handles=**内联句柄值**指向 224B **chain BO**
     （opcode 19 = ERT_CMD_CHAIN、command_count=1、data[0]=真包 BO 句柄）；
   - 真包（4KB BO）：opcode 0 ERT_START_CU + type 3，cu_mask=1，regmap
     [3][instr xdna 0x4070000][**ninstr=4132=16528/4 WORDS**][5 张量 user VA]；
   - args=[ctrl BO 句柄 + 5 张量句柄]（内核只作 pin，fw 只见 regmap）。
   - 内核源码核对：single vs chain 单命令路径语义等价（fill_one_slot_cf 同一
     邮箱 op），chain 包装不需要抄。
   - 发现并修正 engine 侧 ninstr 单位错误（M1 起就传 bytes=4×；A/B 实测两种
     值都 clean，fw 容忍，但 words 是 XRT 约定，5 处全改 + ert.rs 注释立档）。
4. **决定性一击**：拦截器 dump 引擎自己的 ctrl BO —— **13904B，不是 16528B**。
   `/home/nzinfo/qwen/xnpu/build` 夹具是 09:47 拷的**修复前** ctrl（13904B），
   IRON/build 最终版 10:46 才落盘（16528B，pytest 编译）。**P20b 全部 bisect
   都在跑旧设计**；"设计修复对 E2E 无效"的结论是错的。
5. 重拷夹具（bin md5 382406be82b2、pdi 696f2256bbdd，chown+md5 对验）后：
   - quadloop：plain **500 clean**（1012-1033µs/exec）+ xwrite/syncs/sleep/
     fresh/data 各 300 clean（data 模式基线修正为奇偶双基线——X 扰动周期 2）；
   - E2E `run-decode hy cpu quad`：4 次 12-iter 全 clean，gates 全过
     （hidden 1.7%、argmax 25868、top-8 8/8），**hang 绝迹**。
6. 教训入档：**"同输入不同结果"先查树状态（P13 定律第二次咬人）**；诊断工具
   （xdump2）值得保留 —— 单看"数字对不对"永远查不到"跑的是哪份代码"。
7. P20+A/B 性能台面（12-iter，全 PASS 零 hang）：split 41.30 / **fused 37.59**
   （26.6 tok/s）/ quad **45.05**（22.2 tok/s，34 exec/token）。quad 回归
   +7.5ms vs fused（P19 口径 43.76 是旧夹具 5-iter 数字）；已知抓手不变：
   X 重写 2.5MB 整 BO clflush → 8×148KB、K=5 glue、层间流水。

## P21（2026-09-28，quad X 重写 coherency 成本）：sync 粒度 vs 布局

**P21-1 假设**：quad 每 exec 只脏 packed1 里 8 个列头 X 元素（8×18560B=148KB），
却整 BO 2.5MB ToDevice clflush（~154µs/layer，quadloop xwrite 1166 vs plain
1012 分解出来的量级）。把整 BO sync 改成 8 次 region sync 应该省 ~100µs/layer。

**实测（negative）**：8 region syncs **更慢** —— xwrite 模式 300 iters
**1292.0µs/exec**（vs 整 BO 1166.2）。分解：8 次 SYNC_BO ioctl 往返
（~30µs/次 ≈ 240µs）> 1 次 ioctl + 2.5MB clflush（~129µs；clflushopt
~22GB/s）。**教训：SYNC_BO 的成本模型是 per-ioctl 固定开销 + 字节数两项，
region 化只在 region 数少、跳过字节多时赢；8 岛 × 315KB 步长两头都输。**
两处（E2E gemv_quad + quadloop seed_x）已回退为整 BO sync。

**P21-2 决定（改布局而非改 sync）**：对照 fused 路径 —— 它的 X 走独立共享
ELEM BO（18.5KB sync），根本不脏权重 BO。quad 因 5 张量 regmap 上限只能
把 X 塞进 packed1，但**没规定必须塞在每列流头部**：把 8 个 X 元素前聚到
packed1 头部连续区 `[X0..X7 | blocks0..blocks7]`，tg1 每列改两次 fill
（X tap + blocks tap，tg2 的 2-fill 形状已验证），kernel 侧零改动
（元素尾部 header 自描述，fifo 流仍是 [X|16 blocks]）。此后每 exec 脏集
= 连续 148KB → **1 ioctl + ~10µs**，期望 ~100µs/layer × 32 ≈ 3ms/token。
通用律：**per-token 脏集必须连续 —— 布局为 coherency 服务，而不是 sync
为布局买单。**

**P21-3（同日续）：分解测量推翻成本模型 —— SYNC_BO 是 ~30µs 平价 syscall**

1. **工具**：quadloop xwrite 内嵌三段计时（quantize+build / memcpy / sync），
   且 sync 范围可用 mode 后缀切换（`xwrite`=148KB、`xwrite-full`=整 BO
   2.5MB、`xwrite-noflush`=用户态 CLFLUSHOPT），同一循环直接量出 primitive。
2. **实测（300 iters × 多批）**：quantize+build ~8-13µs、memcpy 148KB ~4µs、
   **ioctl sync 148KB = 33.5µs ≈ ioctl sync 2.5MB = 30.5µs**。
   **SYNC_BO 成本与字节数无关（17× 字节差价为零）**——clflush 本身 <3µs，
   全部开销在 syscall 往返（GEM lookup + pin + page walk + unpin）。
   内核源码核对（amdxdna_gem.c/amdxdna_drm_clflush）：范围确实只走
   [start_page, end_page]，flush 不是成本，ioctl 才是。
   **推论：per-exec sync 预算 = ioctl 次数预算，不是字节预算。**
   P20b 把 xwrite-plain 差值 154µs 全部归因"2.5MB clflush"是错的。
3. **用户态 CLFLUSHOPT 替代 ToDevice**：`Mapping::clflush_region`
   （xnpu-hal/bo.rs）：CPUID leaf7 EBX bit23 手工探测（intrinsic 还在
   unstable），inline asm `clflushopt [reg]`（Rust asm! 默认 Intel 语法，
   AT&T 的 `(%reg)` 会报 invalid operand）+ SFENCE；mmap VA 打同一批
   物理行（BO 页已 MAP_LOCKED，无迁移竞态）。148KB **4.6µs**（≈32GB/s），
   300 iters clean。**ToDevice 方向 ioctl 本来就只做 clflush —— 替换是
   严格等价**。
4. **FromDevice 不能换（决定性反例）**：`data-noflush`（回读也用
   clflush）it 5 DIVERGENCE，98 字节差异**全部落在 rows
   [40576, 40687) = C BO 的最后 128 行**（qkv 尾段）——syncobj 信号时
   S2MM 尾部写仍在 NoC 途中。内核 FromDevice 对 ctx-assigned BO 会
   额外发 **MSG_OP_SYNC_BO 固件往返**（aie2_sync_bo，DEV_MEM→HOST_MEM
   全 BO fence）并等它完成——**这个 fence 是正确性必需的**。结论：
   **ToDevice → 用户态 CLFLUSHOPT；FromDevice → 保留 ioctl。**
5. **板况漂移定律**：同日不同批次 plain 从 1009→1424µs、quantize
   8→33µs（CPU 频率缩放）——**绝对值跨批不可比，A/B 必须同二进制同批
   背靠背**（Engine 侧为此加了 XNPU_IOCTL_SYNC=1 环境开关，见下）。
6. E2E 接线（fused pair X+res、quad X+res、lm_head X、flowkv q+o 共 7
   处 ToDevice → clflush_region；FromDevice 全保留），quad 300 clean、
   fused 12-iter gates 全过（1.3%、argmax 对、8/8）。
7. **P21-3 自我修正（strace 决定性）**：上面"~30µs 平价"仍是错账。
   strace -T 对拍（同二进制 6-iter，XNPU_IOCTL_SYNC 开关）：
   ioctl 变体多 889 次 SYNC_BO，总耗时只多 **1.33ms（边际 ~1.5µs/次）**；
   分布**双峰**——多数 <5µs，仅大脏块落在 40-60µs 桶。
   **修正模型：SYNC_BO = syscall(~1-5µs) + 脏行回写（内核逐页 clflush
   ~5GB/s 有效带宽）；干净行近乎免费**（所以整 BO 2.5MB ≈ 148KB region
   ——两者脏字节相同）。quadloop 的 -48.8µs/exec 是**回写速度差**（用户态
   CLFLUSHOPT 连续 VA 流式 ~32GB/s vs 内核逐页 ~5GB/s），不是 syscall 差。
   E2E 对拍印证：fused X BO 仅 18.5KB 脏 → 每对只省 ~3µs × 63 ≈ 0.15ms
   （噪声内，实测 ~0）；quad X 148KB 脏 → ~40µs/层 × 31 ≈ 1.2ms（实测
   均值 -1.2ms，噪声内方向对）。
   **通用律（修正版）：sync 预算 = 脏字节数预算；kernel 路径每字节
   ~0.2ns（5GB/s），用户态 CLFLUSHOPT ~0.03ns（32GB/s）；干净行和
   syscall 都不是成本。**
8. 落地：`sync_to_device`（env XNPU_IOCTL_SYNC=1 可切回 ioctl 做 A/B），
   默认 CLFLUSHOPT；FromDevice ioctl 保留（固件 fence 必需，见 4）。

## P22（2026-09-28③，FastFlowLM 开源边界 + RE 侦察）

问题：与 FLM 差距的结构性原因在哪？其推理部分是否完全闭源、能否反汇编？

1. **仓库已 clone**：`~/qwen/refs/FastFlowLM`（MIT runtime license，
   AMD 2026）。开源部分 = 运行时/编排（flm Rust 97MB not stripped 含
   debug_info）、AutoModel wrapper、tokenizer/sampler、**模块接口头**
   （causal_lm.hpp / gemm.hpp / mha 接口）、**npu_cmd_*.hpp 指令编码
   DSL**（issue_token/maskwrite/write_dma/preemption/wait——就是我们
   P20b 逆向的 ctrl-code wire format 的源码级呈现）、每模型独立
   test harness（src/test/hunyuan_npu/test.cpp，danmaku 语料逐 turn
   prefill/decode 分相计时，profiler 分 SLOT）。
2. **闭源边界 = per-model engine .so**（create_new_model.md 自述
   "comes from the kernel/IRON project"，PIMPL）。但 **not stripped**：
   libhunyuan_npu.so 93 个 T 符号（hunyuan_npu::Impl::_build_slot /
   get_logits / load_weights…）；**libmha.so 19 个 T 符号直接泄露架构**：
   `MHA::Impl::_gen_mha_seq_{d64_q4, d128_q2/q3/q4, d256_q2/q4,
   d128_q4_1cu}` —— attention = **host 侧按 (head_dim, 量化格式) 生成
   ctrl-code 指令序列（npu_sequence）**，由 xclbin 里的 AIE graph 执行。
   与我们的 runtime_sequence 同构，但他们是运行时按形状现生成的。
3. **xclbin 两个图**：Hy-MT2-1.8B-NPU2/{layer.xclbin 364KB,
   fused_prefill.xclbin 244KB}，sections 含 AIE_PARTITION（=PDI）。
   PDI/ctrl-code 解析能力我们已有（P20b），xclbinutil 可提取。
4. **架构结论（差距的结构性解释）**：FLM forward(ids)→logits 一次调用，
   每层 0 次 host 插手——MHA/swiglu/rms 全部在 layer.xclbin 单图内，
   指令序列预生成；我们 fused 口径每层 host 插手 2 次（attention 全套
   + pair 间 swiglu/量化），quad 口径 1 次（attention）。
5. **RE 阶梯（便宜→贵）**：① xdump2 LD_PRELOAD（已有）trace flm
   hy-mt2 → 每 token EXEC_CMD 数/BO 大小/sync 模式；② nm 符号 +
   开源头文件 anchor（本次已完成首遍）；③ xclbinutil 提 PDI →
   我们自己的 IRON 工具看图结构；④ objdump/Ghidra 只在①-③留问号时
   （engine .so 的 T 函数多为序列生成器，语义接近配置数据；算法本体
   在 xclbin 的 AIE 核里）。MIT license 对研究/互操作无碍。

## P23（2026-09-28④，FLM 全量 ioctl trace + ctrl-code 完整解码）

工具：`tools/xdump2`（本日纳入版本管理）+ `tools/ctrl_decode.py`。
原料：`flm run hy-mt2:1.8b` 一句话（12 token 出）的完整 trace（374 exec）
与 40 个 sub dump（132 个 ctrl blob）。过程三步：trace 划相 → ctrl BO 定位
（两次失败假设，见下）→ 指令 walk 校准（IRON 产物双向验证）。

1. **提交解剖（decode 相，hwctx=1）**：每 token = 2 条 ERT_CMD_CHAIN
   （opcode 19）+ 1 条单 op20：
   - chain A：ccount=24 = 1×1024B ctrl（token embedding/输入分发）+ 23×51532B 层 ctrl
   - chain B：ccount=9 = 9×51532B 层 ctrl → 合计 **32 层 ×51532B**
   - 单条 ac=4 的 op20 = lm_head（唯一直接过 arg 的 op）
   - 即 33 op/token，对上 P5 XRT 层 33 run/token；token 节拍 ~20.7ms
     （exec t= 戳：137387→158078→178921µs…）＝48 tok/s
2. **prefill 相（hwctx=2，fused_prefill.xclbin）**：256 条单 op20 =
     32 层×8 op，无 chain。ctrl BO 尺寸按 op 波动（2400/7248/12176/20176B）。
3. **全程 0 次 SYNC_BO**。一致性如何维持是开放问题（假设：engine .so
   用户态 clflush；检验：objdump 扫 clflushopt/clflush/sfence）。
4. **ctrl BO 内存机制（两次失败后定位）**：层 ctrl BO 是 type=3 DEV BO、
   size=51532、**从 64MB DEV_HEAP carve**，map_off=-1 从不单独 mmap。
   否定的两个假设：①CREATE_BO vaddr userptr（实测恒 0）②按 map_off
   mmap（type=3 根本没有）。正解：host VA = heap_map_va + (ctrl_xdna −
   heap_xdna)，xdump2 已实现 fallback。每 token BO 创建/释放风暴
   ~1930 次 create_bo（hdl 211/213/214 每轮复用）。
5. **ERT sub 包格式（4096B wrapper，type=4）**：[w0 hdr][w1=1]
   [w2/w3 = ctrl BO 的 64 位 xdna 地址][w4/w5 = ctrl-code 字节数]
   [w6=3][tensor VA 64bit 对…]。链上 sub 按 wrapper 内 handle 表逐个取。
6. **指令格式定律（ctrl_decode.py 已双向校准）**：每条指令整数个 32bit
   字，`op_size<<2` = **指令总字节数**，位置按 op 定：WRITE(0)=6w
   尾字；BLOCKWRITE(1)/BLOCKSET(2) 在 w[3]，payload=(sz/4−4)w；
   MASKWRITE(3)=7w 尾字；TCT(0x80)=4w w[1]；DDR_PATCH(0x81)=12w w[1]。
   blob 头 [0x06040100, 264, 指令数, 总字节]——w2=指令数、w3=总字节
   两处都对上了（IRON 480/17488，FLM 1566/51532）。
   **shim BD 空间**：BLOCKWRITE 到 (addr&0xFFFFF)∈[0x1D000,0x1D200)，
   bd_id=((a&0xFFFFF)−0x1D000)>>5，**BD 步长 0x20B=8 字 payload 恰好**。
   **队列推**：WRITE 到 reg∈[0x1D200,0x1D400)，MM2S=+0x10，
   value=bd_id|rep<<16|token<<31；MASKWRITE 同区间=issue token。
   校准链：IRON `w4gemvuq….bin`（480 指令 walk 到 0x4450 全对齐，
   语义与其 npu_insts.mlir 一致）→ FLM 开源 npu_cmd_*.hpp 的 to_npu
   编码器逐字段对拍。
7. **IRON quad 对照（同一 decoder）**：我们 1 个 GEMM op = 480 指令/
   17.5KB/48 TCT/4 列；BD 形态 4 列×{bd0 4640B, bd1 74240B}×8 轮
   DDR patch arg0/arg4。**FLM 一整层** = 1566 指令/51.5KB/309 TCT/
   7 列（c5 空置）：**我们单个 GEMM 的 ctrl 成本 ≈ FLM 整层的 1/3**。
8. **FLM 层编排**：316 BD fill + 316 DDR patch + 316 队列推 + 309
   issue-token + 309 TCT 等待；列分工：
   - c0/c1/c6/c7（算力列）：各 68×18432B + 8×55296B 流入（MM2S）
   - c3/c4：仅 2×128B + 2×2048B（KV 头路由级小传输）
   - c2（分发列）：1024B 嵌入 + 2048B + 192B mask，唯一的 S2MM 群
   - DDR patch：arg1×304（主权重流，列内步长 0x12000=73728B）+
     arg0/2/3/4 零星
   - 每次传输配一个 TCT 等待——**无 host 介入的串行化全靠 ctrl 流
     内的 TCT 链**，这是 0 host 插手的实现机制
9. **教科书对比要点**：同 token 粒度下 FLM 提交 1 ioctl（链 33 blob）
   vs 我们 fused 63 exec；其代价是每层 51.5KB×32=1.65MB ctrl-code
   复用（BO 每轮重建）与 309×32≈9.9k 条 TCT 的流内等待。ctrl-code
   密度（指令/有效计算）是我们的差距方向，也是融合深度的度量。
10. **0-SYNC_BO 之谜破解（axcache 定律）**：objdump 扫 engine 全家
    （libhunyuan_npu/libmha/liblm_head 等 27 个 .so）0 条 clflush/sfence，
    FLM 用的是系统 XRT（libxrt_coreutil）——不是用户态 flush。真机制在
    **BD 描述符的 AxCACHE 属性**（write_dma.hpp: no_cache=0 /
    normal_cache=0x02 / aggressive_cache=0x0e，payload 第 5 字 <<24）：
    - FLM 层 blob：311/316 个 BD = 0x0e（aggressive，全部大流量 BD），
      例外 5 个 0x02 恰是 c2 分发列的小 op
    - IRON quad：128/128 全是 0x02（normal）
    即 **DMA 描述符声明 snoop/一致性 → NPU 读写直接命中 CPU cache 行，
    软件缓存维护（SYNC_BO/clflush）整体可省**。我们 P21 测的 ToDevice
    clflush 成本，根源就是 IRON 产出的 normal_cache BD。
11. **P24 候选实验（axcache 移植）**：patch 我们 fixture ctrl bin 的
    BD w[9] 0x02000000→0x0e000000（python 补丁器）+ 引擎加
    XNPU_NO_DATASYNC 环境变量跳过数据面 ToDevice clflush → golden
    对拍。PASS 则"SYNC_BO 消除"从 FLM 观测变为我方工程事实；
    FAIL 则 0x0e 在 x86+此代互联上另有条件（IOMMU/非 snoop 路由）。
    注意 ctrl-code BO 本身的一致性另算（amdxdna exec 路径 kernel 侧
    对 cmd BO 有 dma map，P20b 的首 exec 竞态与之相符）。

## P24（2026-09-28⑤）axcache 移植实验：单改 AxCACHE 不充分，真变量是地址路径（host VA / SVM）

P23-11 设计的三臂实验今天跑完，**结论是 FAIL 方向但价值更高**：排除了
axcache 单因素，把 FLM 0-SYNC_BO 机制定位到 BD 的 64-bit 地址字段。

1. **工具**：`tools/axcache_patch.py`（走 P23 校准的 per-op 尺寸规则，
   对 shim BD 空间的 BLOCKWRITE 改 instr w9 = 0x02000000→0x0e000000，
   留 `.ax02` 备份，`--restore` 复原）。四个 W4_SHAPES fixture 各
   24/24 个 BD fill 全部命中并改写（8col 变体每 GEMV 24 个 BD）。
2. **引擎闸门**：main.rs golden 循环两处 sync 加环境变量——
   `XNPU_NO_IN_FLUSH`（跳过 submit 前输入 ToDevice clflush）、
   `XNPU_NO_OUT_FLUSH`（跳过 exec 后输出 invalidate）。一次编译跑全部臂。
3. **结果**（`run-w4layer build/w4 2 1`，golden 对拍 bf16 容差）：
   - **A 基线**（patched + 双 flush）：4/4 PASS，worst rel err 0.00e0
     ——0x0e 本身不破坏正确性（软件一致性下无害）
   - **B1**（patched + 无输入 flush）：4/4 FAIL，rel err ~1.0
     ——NPU 读到 DDR 旧数据，**MM2S 没有 snoop CPU 脏行**
   - **B2**（patched + 无输出 invalidate）：qkv/o/down PASS、gateup
     2 bad rows——"冷行侥幸"特征：CPU 从未缓存过的行从 DDR 读到新数据
     恰好对，缓存过的行 stale。**S2MM 同样无硬件一致性**
   - **B3**（双关）：4/4 FAIL（B1 ∪ B2 的并集症状）
4. **BD 逐字 diff（关键发现）**：FLM 大流 BD vs 我们 patched qkv BD，
   除 len/addr 外差两个字：
   - FLM：`w5=0x15f80000, w6=0x0000789c` → 64-bit 值
     **0x0000789c_15f80000 是 host 用户态 VA**（xdump trace 里
     0x789b6c…… mmap 区间的邻居，512KB 对齐）
   - 我们：`w5=w6=0`，地址由 DDR_PATCH 在 exec 时换成 firmware 从
     arg handle 解析的 **device dma 地址**（0x4xxxxxx xdna 空间）
   即 **FLM 把 mmap 的 SHMEM BO 的 host VA 直接烧进 BD**，DMA 地址
   与 CPU 视角同地址。FLM ctrl blob 每 token 重建（create_bo 51532B/
   轮）正是为了每轮把当前 VA 烧进去。
5. **内核侧佐证**：amdxdna Kconfig `depends on AMD_IOMMU` +
   `select HMM_MIRROR`；本机 7.0.0-31 amdxdna.ko（zstd 解包 strings）
   含 `mmu_interval_notifier_insert_locked / mmu_interval_read_begin`
   ——HMM interval notifier 机制在跑：驱动把 CPU 页表变更镜像给
   NPU SMMU，device 可按 host VA 翻译并 snoop 一致访问。
6. **机制修正（覆盖 P23-10 的单因素表述）**：0-SYNC_BO =
   **host-VA 寻址（SMMU/HMM 路径，一致性由硬件保证）** ＋
   AxCACHE=0x0e（AXI cacheable+allocate hint，可能仍必要但**不充分**）。
   我们走 device dma 地址路径 → non-coherent → 必须 clflush。
   公开文档状态：AMD 无公开 BD 手册；唯一公开出处是 FLM 头文件
   cache_flag_t 枚举（其口径还是 "QoS fields ex. AxCache"，
   write_dma.hpp:242）。AxCACHE 语义本身是 AXI 标准（0x02=Modifiable
   不可缓存，0x0E=Modifiable+RA+WA）。
7. **工程含义**：不是"手动控制 NPU cache"，而是**换地址路径**——数据面
   BO 用 host VA 烧 BD（绕开/改写对应 DDR_PATCH），一致性交给硬件，
   clflush/SYNC_BO 整体删除。这也是 FLM 每 token 重建 ctrl blob 的
   原因（VA 会变）。
8. **P25 候选（SVM 移植）**：patcher 变体——把输入/权重 BO 的 mmap
   VA 写进 BD w5/w6，同时让对应 DDR_PATCH 失效（或改 argoff 指到
   哑地址），跳输入 flush 跑 golden：
   - PASS → SVM 路径对 exec 包里 arg 解析路径不敏感，机制全量落地，
     SYNC_BO 消除可进 E2E（P21 测的 clflush 面成本直接归零）
   - FAULT/stale → 需找到驱动的 SVM opt-in（interval notifier 注册
     在哪个 ioctl/mmap 路径上），读 mainline amdxdna 源码确认
9. **遗留核对**：ctrl-code BO 自身一致性另算（P20b 首 exec 竞态）；
   B2 的 gateup 2-row stale 提示输出 BO 生命周期里确有 CPU 缓存
   残留（上轮 golden 读过的行），与冷行理论自洽。

## P26（2026-09-28⑥）FLM 对标全量量化：差距主因 = 量化密度 0.349 vs 0.5625 B/param，设备带宽我方已反超

P24 之后的自然问题：46.6 vs 26.7 tok/s 的差距到底在 DMA 效率还是别处。
用 `tools/stream_bytes.py`（BD len × iter × push-repeat 聚合，沿 P23 校准
的 per-op 尺寸规则）把 FLM 层 blob 的真实流量算干净，再对齐我方 P17/P18
实测，得到完整归因。

1. **FLM 每层真实流量**：MM2S 16.82MB = 4 算力列 ×（76 push × 55296B
   线性 BD）；S2MM 仅 9.2KB。每 BD 55296B = 3072 组 × 18B = 98304 参数
   （标准 g32 Q4 排列——单条 BD 格式与我们相同）。同 (col,bd) 槽被反复
   重填：静态 BLOCKWRITE 复位 → DDR_PATCH 换地址（列内步进）→ push →
   TCT 等待，四拍循环 76 遍/列。iter/repeat 全零——放大项不来自 BD 参数，
   来自重填循环本身。
2. **聚合成像**：32 层 × 16.82MB = **538MB/token**，20.8ms/token →
   **25.9GB/s 聚合**；每列 76 × 8.55µs = 650µs/层 串行 → **每列 6.4GB/s**
   （TCT 串行是它们的带宽调节器：4 列并发 × 6.4）。token 拓扑 = 1 常量
   单 op + 链 A(24 sub) + 链 B(9 sub) = **1 ioctl/token**。
3. **文件侧验证量化密度**：model.q4nx 1.503GB = embed fp32 0.990GB
   （120818×2048×4）+ 权重 ~0.51GB → **0.34-0.35 B/param ≈ 2.8 bit**，
   与流测 16.82MB/48.23M 参数 = 0.3487 精确吻合。"w4nx" 实为 ~W3 级密度
   （每 BD 内部排列仍是 g32-Q4 形状，密度靠更粗的码本/混合位宽达成，
   具体方案未逆向）。embed 存 fp32 + lm_head 疑似不占设备流（单 op
   hash 恒定）。
4. **我方对齐数字**（全部已有实测）：hy fused 63 exec，链上 pair A/B
   = 472/369µs → 设备 GEMV 26.5ms/token；权重 868MB（48.23M×0.5625×32，
   与我们 w4 文件 1.115GB/42 层 MiniCPM5 也吻合）→ **设备聚合 32.8GB/s**
   （8 列，每列 ~4.1GB/s）。E2E 37.55ms = 26.5 设备 + 6.2 cpu-attention
   + 2.7 swiglu + ~2 杂项（sync ioctl/翻转税/gap）。口径 caveat：66 op
   计数含 lm_head，我方 embed 流量未计入 868MB。
5. **归因**（同模型同芯片）：tok/s 差 1.81× = **量化密度 1.61×（主）**
   + 非流开销 11ms vs ~0（次）。设备带宽 32.8 vs 25.9，**我方反超
   1.27×**——DMA 效率不是差距来源，反而是我们的强项。
6. **杠杆表（量化排序）**：
   - **L1 量化密度** 0.5625→0.53（g64 共享 scale）/0.44（g64+双重量化）
     /0.41（W3 g64）→ 设备时间 26.5→25.1/20.7/19.1ms。收益 ×1.06/1.28/
     1.39，代价 = importer + IRON kernel 位宽/编解码改动，golden 门不变。
   - **L2 host 侧 8.9ms 清零**：swiglu 并进 pair B 入口胶水（gateup 出
     →swiglu→down 入，2.7ms）；attention 元素化或 flowkv strided 重排
     （npu 口径 118ms 的病根，cpu 口径 6.2ms）。
   - **L3 exec 链化** 63→1-3：P23 已解码的 ERT_CMD_CHAIN 机制直接可用，
     省 ~63×33µs sync/提交 ≈ 2ms，且为层间流水铺路。
   - **L4 P25 SVM 零 flush**：每对 4KB 残差 ToDevice + C BO 44KB
     FromDevice ioctl（~30µs 平价）全免。
   - **L5 CU 翻转税**：每 token 2 次 ~500µs fw PDI 重载（P18 补记），
     单 CU/同 PDI 布局可免。
   - **L6 设备带宽 32.8→40+？**：我方每列 4.1GB/s vs FLM 6.4GB/s——
     差在喂法（8 列细分 vs 4 列 54KB 大块串行）。聚合已占优，优先级最低，
     天花板待 perf-calibrate 补测。
   - **合计上限**：W3 + L2-L5 ≈ 19-20ms ≈ **50-53 tok/s > FLM 46.6**；
     仅 g64+L2-L5 ≈ 21.7ms ≈ 46 tok/s 与 FLM 平手。
7. **教科书要点**：tok/s ≈ 有效带宽 ÷ 每 token 权重字节——量化密度是
   LLM 推理的第一性杠杆，DMA/调度优化是二阶。FLM 的全部"快"来自把
   每 token 字节压到 538MB；其 25.9GB/s 聚合反而低于我们的 32.8。另：
   fp32 embed 0.99GB 换 tied-lm_head 精度的存储取舍值得写。

## P26b（2026-09-28⑦）P26 勘误：BD buffer_length 单位是 word——FLM 密度就是 0.5625 B/param，差距与量化无关

**触发**：用户问"FLM 是不是直接用 hy-mt2？我们是不是同一个模型文件？"——
顺着查证，发现 P26 的"FLM 2.8 bit/param"结论是**错的**，两个工具 bug
加一个数据假设错误叠加出一个看似自洽的假故事。全部留痕：

1. **错误一（工具，4× 欠计）**：BD 指令 w4 `buffer_length` 的单位是
   **32-bit word 不是 byte**（FLM 开源头文件 npu_cmd_write_dma.hpp 只写
   "Buffer length" 没写单位）。三方闭合实证：
   - FLM 层 blob：4 算力列 ×（68×18432w + 8×55296w）× 4B = **27,131,904B
     = 48,234,496 参数 × 0.5625，逐字节精确**；
   - 我们 quad 夹具 bd0 = 4640w = 18,560B = v5 fifo ELEM 尺寸精确命中；
   - FLM DDR_PATCH 列内步长 0x12000 = 73,728B = 4×18,432 = 一条 18432w
     BD 的真实传输量。
   我对自己夹具的测量（0.75MB 等）也全部 4× 欠计（真实 2.95MB），且四个
   形状对已知权重字节的比值恒 0.253——当时没做这个一眼可做的 sanity check。
2. **错误二（工具，槽覆盖）**：`bds[(col,bd)]` 字典被重填覆盖，按 order
   求和时 76 个历史 fill 全部用了最后一次重填的状态（76×55296w 的假指纹；
   真实静态内容 68×18432w + 8×55296w，P23 ctrl_decode 的逐 fill 打印
   当初就是对的）。修复 = 事件源化（fill 时刻捕获、push 时刻对槽状态沿
   nextbd 链求和）。iter 字段按头文件是 size−1 编码（raw 0 = 1 次）。
3. **错误三（数据假设）**：P26-3 的"文件侧验证"把 embed 当 **fp32**
   （0.99GB）去凑 538MB 权重——但 P2 早已 100% pin 过文件结构：embed 是
   **BF16**（manifest + 值级验证）。正确算术：868.2MB（层权重 0.5625）
   + 139.2MB（lm_head 30208 tiles×4608B）+ 494.9MB（bf16 embed）+ ~1MB
   norms = **1503MB = 实测文件大小**。两种分解都能凑到 1503——裁决证据
   是 P2 的值级解码（lm_head 反量化 vs tied embed rms 0.089 只可能在
   int4 存储下成立）和 P5 的 BO 尺寸（26MiB/层、133MiB lm_head 精确
   等于 0.5625 密度）。教训：**验证不能挑着凑数，已有 100% 结论（P2）
   优先于临时反推**。
4. **修复后工具复验（双向闭合）**：FLM s1 MM2S 27,177,728B（算力列
   27,131,904B 精确 = 0.5625×params，余 45.8KB 为 c2/c3/c4 路由小传输）；
   我们 qkv 2560×2048 夹具 2,981,888B ≈ 权重 2,949,120B + 33KB X 元素 ✓。
   ctrl_decode.py 的 len 显示同单位问题（P23-8 逐列字节清单 4× 欠计，
   计数/结构结论不受影响）。
5. **对用户三连问的答案（一并立档）**：
   - FLM **不重量化**：load 时把 model.q4nx 原样搬进 BO（26MiB/层 =
     q4nx 密度），DMA 直接流 int4 字节；
   - 我们用的是**同一份 model.q4nx 文件**（P2 逆向），但流程是
     q4nx int4 → 解码 f32 → 我们 amax/7 重新量化 → 我们的 packer——
     同源同密度（4.5bit），多一次重量化舍入，golden 对我方权重自洽；
   - 即"同一个模型文件、同一种量化密度"，P26 所述密度差不存在。
6. **修正后的归因（覆盖 P26-5）**：两边都流 ~1.0GB/token（868 层 +
   139 lm_head），都在 ~50-55GB/s 器件墙附近（我方 P12 实测边际 55.6，
   FLM P5 实测 48-64）。FLM 21.44ms ≈ 18.9ms 流地板 + ~2.5ms host；
   我们 37.55ms = 同一 18.9ms 地板 + **~7.6ms exec 结构开销**（63 exec
   × ~95µs 固定成本 + 2 次 CU 翻转）+ **~11ms host**（attn 6.2 +
   swiglu 2.7 + 杂 2）。差距 16.1ms 全部来自调度与 host 侧，零来自
   带宽与量化。
7. **修正后的杠杆表（覆盖 P26-6 排序）**：
   - **L2 host 侧 ~11ms 清零**（swiglu 融进 pair B 入口胶水 2.7 +
     attention 元素化 6.2 + 杂项）——第一大杠杆；
   - **L3 exec 结构 7.6ms**：更深融合（quad 路线 P19/P20 已验证机制）
     + ERT_CMD_CHAIN 链化 + CU 翻转税；
   - L4 SVM 零 flush（P25）照旧小收益；
   - **L1 量化密度从"解释差距"降级为"进攻选项"**：既然 FLM 停在 4.5bit
     ≈ 墙速，做 g64（0.53）/W3（0.41）能把地板 18.9→17.8/13.8ms——
     这是**超过** FLM（46.6 tok/s）而非追平的手段；
   - L6 设备带宽照旧优先级最低。
8. **方法论入档**：反推式"验证"（先有结论再找一组能凑上的算术）比不
   验证更危险——它会给错误结论盖合格的章。交叉闭合（流测↔文件格式↔
   BO 尺寸三方独立来源）这次真正起了作用。

## P27（2026-09-29）追平 FLM 战役：杠杆 1-4 执行（目标 ≤21.5ms / ≥45 tok/s）

P26b 修正后的路线图落地。目标分解（从 37.6ms 出发）：L2 host 11→~0.5、
L3 exec 结构 7.6→~2、地板 18.9 不动（量化密度属杠杆 5，本次不做）。
执行序按风险升序，每步 golden 双门把关：

- **P27-1 host attention SIMD 化**（纯 Rust，零 IRON 风险）：现 6.2ms 的
  真凶是标量 `d += a*b` 的 128 长 FMA 依赖链（4 拍延迟串行）+ 51.7K 次
  libc expf。AVX-512（Zen5 全宽原生）16-wide 重写三个热循环；数值上
  转换精确（bf16 位左移）、dot/求和改 lane-tree、exp 用 6 项多项式
  （漂移 ~1e-7 ≪ bf16 噪声），输出端 f32_to_bf16 epilogue 保持标量
  逐位不变。预期 6.2→~1ms。
- P27-2 swiglu 上设备（pair B 入口胶水 K=4/K=5，机器来自 quad P19b）
- P27-3 exec 链化 + 提交流水（P14 定律：host 出链后才兑现）
- P27-4 杂项（X 布局前聚 / CU 翻转 / 复测）


### P27-1（2026-09-29）AVX-512 attention + host 分段显微镜

**执行**：`attention_avx512`（target_feature avx512f/bw，运行时 dispatch：pos≥15 且
CPUID 具备）——bf16→f32 位左移精确转换、QK/PV 均 8×FMA+reduce、exp 走 6 项
ln2 级数多项式（RNE roundscale 取 n、指数位相加、x≥−80 clamp）、epilogue 保持
标量 f32_to_bf16 逐位不变。edition-2024 坑：unsafe fn 体内要显式 unsafe{}。

**结果一（预期落空）**：同日背靠背 A/B，37,525→37,017µs（med 37,327→36,985），
只省 **−0.5ms**，不是预期 −5ms。门全过（hidden rms 0.0447=1.7%、argmax 对、
top-8 8/8）。原因：P15 的"6.2ms attention"是 split 时代的 gap 归因，把
rope/qk-norm/KV-append/staging 与 attention 核捆在一起；fused 路径上核只有
~1ms 量级。**教训：gap 归因的颗粒度会随结构变化失效，换结构后必须重测。**

**结果二（显微镜，XNPU_HOSTPROF=1 分段计时，5 iters）**：给 fused 循环加
HostProf（thread_local 分段纳秒累加，timed 相起止 reset/print；wait 段含
submit→syncobj-wait 为设备主导）：

| 段 | µs/token | 判读 |
|---|---|---|
| A:wait+B:wait+lm | 32,129 | 设备执行+submit（串行、零重叠） |
| **B:quant** | **3,563** | 量化 6144 激活+打包+sync，最大 host 单项 |
| attn | 1,344 | AVX 后全程（含 K/V staging），~42µs/层 |
| **A:quant** | **1,179** | 量化 2048+sync，36.8µs/层 |
| A:read+B:read | 1,182 | FromDevice+w4u_read_c+add |
| swiglu | 763 | 宿主标量 |
| qknorm | 193 | 标量 rms |
| rope+kvapp+res×2+finalnorm | ~180 | 已不构成目标 |
| SUM | 40,539 | 对 wall 41,770，未归属 ~1.2ms（L0 plain 路径/argmax/循环） |

（本轮板况偏慢：steady 41.77ms，med 41,948——P21-5 漂移定律，跨日绝对值不可比，
分段**比例**可靠。）

**修正后的账**：host 胶水合计 **~8.4ms**（不是 P26b 估的 ~11：attention 核被
高估、swiglu 实测 0.76 而非 2.7——2.7 是 split 时代含量化/读回的捆绑口径）。
设备侧 A:wait 525µs/层、B:wait 389µs/层 vs 流地板（16.9MB/10.6MB @48.9GB/s =
345/217µs）→ **每对设备固定成本 ~120-130µs**，63 对 ≈ 8ms——与 P26b 的
"7.6ms exec 结构"闭合。

**下一刀的裁决数据**：P27-2（swiglu 上设备、pair B 入口胶水）一刀切掉
swiglu 763 + B:quant 3563 = **−4.3ms**；随后 A:quant/attn/read 的 SIMD 化
再收 ~2ms；剩下的 ~8ms 设备固定成本要靠 P27-3 链化/更深融合。FLM 每层
585µs vs 流地板 555µs（+30µs 固定）是我们 exec 结构差距的坐标。

### P27-2 设计研究（2026-09-29②，先算账后动刀）

读 design_fused.py / design_quad.py 全文，把「pair B' = swiglu 上设备」
推到可实施级，然后**被自己的账否决**：

1. **可行设计（cut the quad in half）**：pair A' = quad 的 tg1+tg2（C 扩成
   padded gate/up 布局 [win1 9280 | gate 6272 | padA 3008 | up 6272 |
   padB 3008] = 27840 行）；pair B' = tg3/tg3b/tg4/tg4b（rt.sequence
   (A_down, A_qkv, C_B, C_A) 四张量 + ctrl = 5 BO 上限满足，X BO 消失，
   宿主 residual2 写入不需要——stage1r 重读 cA 的 win1 导出 h2'=x+o）。
   flag 生命周期 exec 内闭合（K=5 set → K=2 离开 → K=1 重读清零），零内核
   改动，口味全在 P19b 机器里。
2. **账（否决理由）**：quad 实测 1087µs/层 vs pair A+B 914µs = **+173µs/层
   设备罚金**（≈+5.4ms/token），吃掉 −4.4ms 宿主节省的大半。swiglu 上设备
   的正收益前提是 quad-v2 编排把整层压回 ~600µs——那是杠杆 3 的本体。
3. **fill-hoisting 结构性不可能**：想把 op2 权重提前塞进 tg1 的 fifo——
   fifo 顺序 = 消费顺序（K=1 窗口必须先于 K=3 元素，而 K=1 的源是 C 的
   drain 输出）；单列 depth-2 fifo（2-fifo 被 L1 挡死：4×18560 > 64KB tile）
   决定了 ~120µs/pair 的固定成本搬不走。

**裁决：转 P27-2b（host SIMD 包）**——同一笔钱（−4.3ms 目标）零设备风险。

### P27-2b（2026-09-29③）host SIMD 包：−2.5ms（interleaved A/B），量化位精确

四刀（main.rs d9ea7d6）：

1. **w4u_quantize_x_avx512**：按构造位精确——amax 用 |f32 bits| 的整数
   max（非负 IEEE 位序=值序，P17-2 同款 trick）、amax==0 组守卫、
   `f32_to_bf16(amax/127)` 留标量、`_mm512_div_ps` IEEE 精确除、
   roundscale 0x00 = RNE、clamp 先于饱和 pack（与标量 round→clamp→cast
   同序）。32 元素/组两拍 16-wide。
2. **swiglu_bf16_avx512**：sigmoid 复用 attention_avx512 的 6 项 ln2 级数
   poly（vs libc expf 漂移 ~1e-7 ≪ bf16 粒度）；除法与 RNE bf16 存储
   （位 trick 向量化）逐位同标量。z∈[-80,80] clamp（sigmoid(±80) 已是
   1/0 饱和区，poly 指数位加法在正常域）。
3. **w4u_read_c**：chunks==1 改 u16 重解释 + 按列 copy_from_slice（C BO
   页对齐、偏移偶——原逐字节拼装才是读路径真成本）；chunks==3 保持 f32
   求和序（位不变）。
4. **gemv_pair**：残差 staging 与 op2 section 循环同样改 u16 视图单次拷贝。

**门**：hidden rms 0.0447（1.7%）与 lm rel_rms 0.0223 —— 与标量路径
**完全相同**（bf16 舍入吸收了 sigmoid 的 1e-7 漂移）；argmax 25868 对、
top-8 8/8。

**A/B（P21-5 漂移定律下的严格口径）**：同一热板窗口内 old/new 交替
（63bd542 worktree 重编译，绝对路径共享受夹具）：

| | old | new |
|---|---|---|
| steady ms/token ×2 | 35.18 / 35.25 | 32.25 / 33.22 |
| B:quant µs | 1417–1420 | 74–132 |
| A:quant µs | 484–490 | 53–87 |
| swiglu µs | 310–384 | 59–123 |
| A:wait/B:wait | 16.0/11.7 ms | 16.0/11.3 ms（设备未动 ✓） |

净 **−2.5ms**。P27-1 测的 B:quant 3563 是慢 CPU 批（18ns/elem），今天热
批标量只要 1420（7ns/elem）——SIMD 化同时**消除了这些段对 CPU 频率的
敏感**（冷批保护 ~4.3ms）。

**板况新观察（P21-5 延伸）**：同日见 31→46ms 摆幅（同二进制）；间隔 20s
的四连跑单调**变快**（37.6→34.7→32.4→32.0）——加热/唤醒方向，与 P7 记录
的"跨批漂移"同源，根因仍未明。最优热批 30.98ms（32.3 tok/s）。

**P27-2b 后的账（热批 ~32.3ms）**：A:wait 16.0 + B:wait 11.3 + lm 3.1
（设备 30.4）+ attn 1.6 + reads 0.92 + quant 0.25 + swiglu 0.1 + 杂 0.3。
宿主胶水只剩 ~3.3ms；**追平 FLM（21.44）的主战场完全在设备侧**——63 对
wait 比流地板多 ~8ms（exec 结构）+ lm_head 3.1 vs FLM 2.66 + attention
1.6ms（上设备的前提是 quad-v2/编排）。P27-3（链化/更深融合）才是杠杆 3
的本体。

### P27-3a（2026-09-29④）单 CU 全 fused-PDI：翻转税消灭，−1.8ms

**假设**：P18 补记的结构性翻转税（每 token 2×~500µs，fused 对 CU1 /
plain 算子 CU0 两 PDI 异 CU）能否消除？关键观察：**plain 与 fused PDI
编译自同一份 w4gemvu.cc**——fused 内核的 K 头 dispatcher 保留了 plain
口味（K=0 X 元素 staging、K=2048 compute），plain ctrl bin 只是数据。
两 PDI 的 partition JSON 结构相同（仅 uuid/文件名），BD 空间/fifo L1 地址
一致（design_fused 的 tg1 本来就是 plain 组的形状）。

**实验**：fused cpu 模式只配一个 CU（fused PDI），全部 66 exec 走 cu 0
——cu_mask 永不翻转。**板上一次通过**：hidden rms 0.0447、lm rel_rms
0.0223、argmax/top-8 与双 CU **逐位相同**。interleaved A/B（热板 3 轮）：
two-CU 32.05/33.17/30.76 → one-CU **30.09/30.64/29.87**（−1.8ms 稳定；
lm 段 3076→2690µs = 回到无税 solo）。默认开启，XNPU_TWOCU=1 复旧。

**教科书点**：PDI 只是内核 ELF 载体，ctrl code 是数据；K 头自描述派发
使一个 PDI 成为整个算子族的通用执行器。「CU 分工」是调度选择不是语义
约束——翻转税来自 fw 对 cu_mask 变化的 PDI 重载，与算子归属无关。

### P27-3 后的差距坐标（one-CU 热批 ~30.1ms vs FLM 21.44）

| 项 | 我们 | 地板/FLM 坐标 |
|---|---|---|
| A:wait | 14.7ms（460µs/对，16.5MB → 35.9GB/s） | 338µs@48.9 |
| B:wait | 10.9ms（350µs/对，10.7MB → 30.5GB/s） | 218µs@48.9 |
| lm | 2.69ms | ~2.65（53GB/s，已在墙） |
| host（attn+read+quant+swiglu） | ~2.9ms | FLM ~2.5（重叠） |

设备 27.2MB/层在 810µs = **33.6GB/s 聚合**，而 FLM 层 exec ~480-585µs =
47-56GB/s、P12 无 B 探针边际 55.6。**剩余差距的主形态不是 per-exec 固定
成本（~30µs×2/层，FLM 同量级），是 pair exec 内部的流速率**（33.6 vs 47）。
下一刀 = pair 填充率地板探针（v5s 技术：K 头打垃圾走零行路径，同 ctrl
形状量纯 fill 地板），判 fill 机器（BD 尺寸/组边界）还是计算暴露。

### P27-3b（2026-09-29⑤）pair 填充地板探针：计算暴露=0，「组固定成本」模型闭合

**环境事故与修复（先留痕）**：系统升级删掉了 python3.12（pytest/aie 全断）。
重建：`/home/nzinfo/.venvs/npu314`（python3.14 venv，--system-site-packages
吃系统 dist-packages 的 **pyxrt cp314 .so**——这是唯一必须用 3.14 的原因，
3.13 的 unsloth venv 装不了它）。pip 装入：mlir_aie==v1.2.1 cp314 wheel
（GitHub release extra-index，requirements.txt 本来的安装源）+ llvm-aie
nightly（py3-none wheel，网络断续要 `--resume-retries 10`）+ torch 2.12.1+cpu
（pytorch cpu index 有 cp314）+ pytest/ml_dtypes + `pip install -e IRON --no-deps`。
运行配方照旧（sudo + prlimit memlock）。**教训：IRON 的可运行性依赖三个
非 PyPI 的 GitHub release wheel，环境重建配方记在这里。**

**探针（iron/operators/w4gemvu/test_fuseds.py）**：v5s 技术移植到 fused pair——
把 packed1/packed2 里每个权重块的 K 头（块尾 ELEM-8）改成 0xBEEF，内核守卫走
零行路径（无 mmul、无操作数重建）；X 元素（K=0）、rms 窗（K=1）、w 元素（K=3）
保持活——胶水口味照跑。线上字节/fill/drain/内核镜像全同。PERF ONLY（zeros-in
zeros-out，无金标断言；run_test 对 zeros 参考的 mismatch 恰好是探针有效性的
证据：100 条 mismatch 全部落在 [2304,6409] = rms 窗活数据区，权重段全零匹配）。

**结果一（假设裁决：计算暴露 = 0）**：同一 pytest 会话、同工具链、同板窗，
5 次重复取中位：

| | full（test_fused） | probe（test_fuseds） | 差 |
|---|---|---|---|
| pair A (o→gateup, 17.08MB) | 552.8µs | 546.6µs | ≈0（噪声内）|
| pair B (down→qkv, 10.66MB) | 422.1µs | 435.4µs | ≈0（噪声内）|

**去掉全部计算，pair 时间不变**——v5.4 双累加器内核的计算完全藏在 fill 后面。
P27-3 立的「fill 机器 vs 计算暴露」二选一：答案是 fill 机器，且不是内核循环。

**结果二（长度标定，同板窗 plain 探针）**：test_v5s 四形状 5 次中位，
latency = **固定 132µs + 字节/55.1GB/s** 拟合（四点全中：o 175/175、
down 252/260、gateup 378/389、lm 2660/2661）。即：

- **边际流速率 55.1GB/s = 器件墙**（P12 的 55.6 复现），长短流通用；
- **每个 task group ~132µs 固定成本**（隔离口径；plain op = 1 组）；
- pair probe 547 ≈ o(175) + gateup(378) 的**串行和**（−6µs）——pair 的两
  组各付一次固定成本，组间无重叠也无额外罚金；
- E2E 链上有效值 ~77µs/组（A:wait 460 − 307 流 = 153 = 2×77；B 同），
  链比隔离好 ~55µs = submit/syncobj 往返被链式掩掉的部分；
- **模型闭合**：E2E 设备 28.3ms = 流 18.3ms（1.01GB@55.1）+ 129 组 ×
  77µs ≈ 9.9ms ✓。**与 FLM 的全部剩余差距 = 组数 × 组固定成本。**
  FLM 每层 1 exec、~30-90µs 固定（585−555），我们每层 4 组 × 77 = 308µs。

**结果三（pair ctrl 解剖，stream_bytes.py + npu_insts.mlir）**：
- npu2 分区 = **8 核（2 列 × 4 行）+ 4 shim**，每 shim 2 MM2S 通道各喂
  1 核——IRON 的「8 columns」= 8 核，ctrl 列字段 0-3 = shim；
- pair A 每 shim：2D BD 544B（C1 drain tap）+ 2D 3136B（C2 drain tap）+
  4×18560B 线性（X×2 + 窗×2，两逻辑列）+ 296,960B×2（o 块/核）+
  1,800,320B×2（gateup 96 块/核）——**权重早已是单条大线性 BD**，BD
  尺寸/形状不是慢的原因（推翻「BD 机器」嫌犯）；
- rms 窗 fill = C 源**线性** 18560B（RMS_taps sizes=[1,1,1,9280]），非 2D；
- 每组末 8 条 npu.sync（4 shim × 2 通道，direction=0 只等 drain）；
  整 pair ctrl 881 行 mlir ≈ 480 指令。

**结论与下一刀**：填充机器的慢不在 BD 形状、不在计算、不在带宽，在
**task group 边界的串行化**：tg2 的权重流必须等 [tg1 流尾 → 核计算 →
C drain → DDR 落地 → TCT → 窗 fill] 整条链走完才开始（窗元素与权重同
fifo，消费序=压栈序，权重不能先压）。P27-4 = ctrl 手术探针：把 tg2 的
BD 写+push 整块搬到 tg1 sync 之前（消费序破坏 → 数值垃圾，PERF ONLY），
若 pair → ~360-400µs 则「跨组 enqueue-ahead」就是 FLM 的 309-TCT 图的
本质，剩余差距的工程路线随之确定（FLM：独立 fill 早发 + 依赖只约束
窗/激活，整层 1 组）。

### P27-4（2026-09-29⑥）ctrl 手术三连：enqueue-ahead 全部 NULL —— task group 是调度成本的原子

**工具链**：`tools/hoist_probe.py`（P23 walk 规则 + BD 空间语义分类），
对编译好的 pair ctrl bin 做指令级重排，产物 .hs/.hs2/.hs3；引擎侧加
`XNPU_SKIP_GATES`（跳过 checked 步 lm_head 门的 hard-fail，PERF-ONLY run
专用，与 XNPU_TWOCU 同款开关风格）。A/B 全部 interleaved 同热板窗口
（P21-5 律），3 轮交替、每轮 4 iters，读 HostProf 的 A:wait/B:wait。

**手术设计（三档递进）**：

1. **.hs（输入侧）**：tg2 的 MM2S 机器（4 shim × [4 BLKWR + 4 DDRP +
   4 push]）搬到 tg1 首条 TCT 之前，BD 槽 0-3 → 8-11（重写活描述符是
   FLM 从不做的危险动作，P16 律）。通道消费序 [X, A1, window, A2] 保持
   ——只有 window 的源数据是陈旧的（tg1 drain 未落地）。
2. **.hs2（+ 输出侧）**：tg2 的 S2MM 排水机器（slot 4/5 的 BLKWR+DDRP+
   MASKWR+push）也搬前，槽 4/5 → 12/13。C fifo 满死锁假说的直接检验：
   排水提前后 A2 流应该不再被 tg1 尾部卡住。S2MM 队列序 [tg1 bd4 →
   tg2 bd12] 与元素生产序（fifo 序传递保证 C1 先于 C2）一致，语义安全。
3. **.hs3（− tg1 TCT）**：从 .hs2 里删掉 tg1 的 8 条 TCT，只剩 tg2 的
   8 条收尾——检验「TCT 串行 retire」是不是残余固定成本。

**结果（全部留痕）**：

| 变体 | A:wait µs/token（3 轮） | B:wait | 判定 |
|---|---|---|---|
| orig | 14,738/15,598/14,878（及二轮 14,726/14,657/14,856） | ~10,850-10,940 | 基线 |
| .hs 输入 hoist | 14,916/14,821/14,353 | ~10,576-10,924 | **NULL** |
| .hs2 全 hoist | 14,404/14,670/14,646 | ~10,602-10,817 | **NULL**（≈1-2%，噪声边缘） |
| .hs3 −TCT | 第 2 个 exec 起 `Timer expired` 挂死 | — | **不可行** |

手术生效性证据（不是没打上）：.hs/.hs2 的 lm rel_rms 从 0.0223 稳定
劣化到 0.045-0.073（陈旧 window 真被提前读了，值有界、argmax 仍对、
无挂死无槽碰撞）。hs3 挂死复现 P19b「无 sync 组的槽重用」警告：TCT
不只是排序，它承担通道/BD 状态的 retire，下一个 exec 的槽重写依赖它。

**三个否定结果的合成推理**：

1. 输入侧 enqueue-ahead 无效 + 输出侧 enqueue-ahead 无效 + window DDR
   往返被证明跳过（.hs2 里 window 读的是陈旧数据）仍无提速 ⇒ 组固定
   成本**不在 ctrl 指令的发行时序里**——ctrl DPU 把什么都提前发也没用。
2. 排除清单（累计）：BD 形状/尺寸（P27-3b）、计算暴露（P27-3b probe）、
   输入发行时序、排水发行时序、window DDR 往返、通道数（P12）、
   fifo 元素粒度（plain 同粒度跑 55.1）。剩下唯一与「组」绑定的东西：
   **shim/fw 对每个 task group 的状态机开销本身**（drain token 集合 +
   npu.sync 组末屏障的完成路径）。
3. 定量闭合（本窗口数据）：pair A 460µs = 流 310 + 2×75；pair B 346 =
   流 193 + 2×76。quad（5 组）同模型 5×114 + 流（P19b 实测 1066 拼合）。
   **每 token 129 组 × ~75µs ≈ 9.7ms 就是与 FLM 的全部设备侧差距。**
4. FLM 对照：他们 309 TCT/层 exec 却只有 ~30µs/层固定成本 ⇒ TCT 本身
   可以很便宜；贵的是我们的「组」粒度——每组的 8 条 npu.sync + drain
   token 集 + X/window 小元素 + 2D C tap 组合。FLM 的层是**一个 exec、
   一个连续 BD 重填循环**，TCT 串在流内当数据依赖用，不是组末屏障。

**教科书结论**：XDNA2 上 ctrl-code 的调度成本原子是 **task group**
（fill 集 + drain token 集 + 组末 npu.sync 屏障），不是指令、不是 BD、
不是 TCT 条数。ctrl 级重排（无论多激进）无法合并组；组只能靠**设计时
不产生跨组数据依赖**来减少——依赖必须走内核内 L1/.bss 交接，而不是
C→DDR→window 往返。

**生产路线（P28 = layer-v2，数学已闭合）**：

- 1 组/层：流 503µs（27.2MB@55.1）+ 75 + exec ~30 ≈ **608µs/层** vs 现
  810µs（460+350）→ 设备 28.3 → ~22ms；加 host 重叠（submit-ahead）与
  attention 遮盖 → 21.5ms 附近 = FLM 追平点。
- pair-v2（1 组/pair，仅去组）只省 ~44µs/pair ≈ 2.8ms → 27.3ms，不够；
  **必须整层 1 组**。
- 内核改造点（全部有 P19b 机器）：compute 尾声把 partials stash 进
  .bss（不再靠 drain→DDR→window 往返）；residual 走 host 稳定 BO 的
  小元素（X BO 同路，独立于任何 drain）；swiglu 口味（K=4/K=5）改读
  .bss stash；window fill 全部消失 ⇒ 层内全部 fill 都是静态的 → 单组
  [X, res, w, 权重块…] + 一条 drain 链（bd4→nextbd→…）+ 一组 8 条 TCT。
- 风险：单组内双 drain push 的 token 记账未证（hs3 教训：sync 语义脆，
  用 nextbd 链 + 单 token 集兜底）；L1 预算（stash +4KB/核 vs 63.3KB
  现状，可行）。

**留痕**：.hs/.hs2/.hs3 生成与校验逻辑全在 tools/hoist_probe.py（提交
在本仓库）；A/B 日志 /tmp/p274/{o,h,g,s,t}*.log（会话后归档）；夹具
已复原（md5 与原版一致）。

## P28-1（2026-09-29⑧）persistent-worker 执行模型：IRON 原生就有 FLM 的机制

P27-4 定论「组只能靠设计时不产生跨组数据依赖来减少」之后的第一个问题：
层内全静态 fill 的执行模型长什么样？答案不需要发明——**IRON 自己的
`iron/operators/swiglu_fused_decode` 就是 FLM 模型的原生实现**（README:
"1.3x faster decode"，Llama 配置里的 dual-GEMV SiLU-mul 融合算子；中间
向量经 inter-tile ObjectFifo 留在片上，消灭 DDR 往返）。本条把它的
设计、npu_insts 证据、板上表现全量解剖。

### 三大机制（npu_insts.mlir 实证，build/…4col.mlir.prj）

1. **one-shot 巨型 BD**：每 exec 每 fifo 只有一个 blockwrite+address_patch+
   push 三元组。整个 exec = 12 fill 三元组 + 4 drain 四元组（多一条
   maskwrite + token push）+ **4 条 npu.sync** ≈ 52 条 ctrl 指令。单条
   BD 的 repeat/stride 字段覆盖整列权重流（例
   `dense<[1048576, 8388608, 0, 0, 0xC0000000, 33554432, 0, 33554432]>`
   = len 0x100000 词、axcache 0xC0000000 aggressive、stride/repeat
   0x2000000）。queue 从不排空 → 无重启成本。对照：我们 quad 480 指令/
   48 TCT；FLM 一整层 1566 指令/309 TCT。
2. **mid-flow 依赖全是核侧锁**：`aie.mem` 区域里 dma_bd 自环（bd_id
   0↔1 = depth-2 L1 buffer）+ `aie.use_lock(AcquireGreaterEqual/Release)`。
   shim→核 = shim 的 MM2S 流进消费核自己的 mem-DMA S2MM（锁流控）；
   核↔核 ObjectFifo（inter_i）= 生产核 MM2S / 消费核 S2MM 同款机器。
   整个 exec 只有末尾 4 条 npu.sync（最终 C drain 的 wait=True）。
3. **persistent worker**：`for _ in range_(0xFFFFFFFF)`。核配置一次
   （PDI），每 exec 只重跑 ~52 条 ctrl；worker 在 fifo acquire 上无限
   阻塞，等下一个 exec 的新 BD 送元素进来。

### 对 75µs/组 成本理论的修正（覆盖 P27-4 的表述）

我们的组末 8 条 npu.sync 是**对真实 drain 完成的长等待**，每条付出
poll 量子开销（~5-10µs × 8 ≈ 75µs/组）；FLM 的 309 TCT **立即 retire**
（issue+wait 一个已完成的 token ≈ 免费）。消灭 DDR 往返 → 消灭长 TCT。
与 P27-4 .hs2 异常自洽：一切提前 enqueue 仍 460µs——TCT 等待本身还在
DPU 指令路径里，发行时序不是变量。

### 板上验证（test_swiglu_fused_decode，sudo+prlimit 配方）

- **机制跑通不挂死**：latency **1230.3µs**（2048×2048，25.2MB bf16
  权重流 = **20.4GB/s 聚合**；4 列 × ~5GB/s/通道——他们只用 4/8 通道、
  8KB L1 buffer）。
- **3/2048 行 FAIL**：output[858] expected -160768 got **-32768**；
  output[1384] expected 15040 got **32768**；output[1738] expected
  109056 got 0.0。值钉在 ±32768（int16 饱和签名）→ 疑似他们内核的
  stale-fifo/竞态或 nightly llvm-aie 回归。内核读码排除 accumulate
  dtype 一因（accfloat + aie::mac 全程）。**判定为非机制问题，有意
  搁置**：P28 写自己的内核（K-header 自描述口味已全在 P19b 机器里），
  golden 门自控。backlog：查上游 IRON 修复。
- 带宽判读：20.4 不是 persistent 模型的上限——4 列/8KB buffer 的选择；
  我们 8 通道 + 18.5KB 元素在 55.1GB/s 器件墙上。persistent 不输带宽。

### layer-v2 数据流设计（ring 拓扑，P28 本体的设计基线）

- **跨列重分配不可避免**：rms 要全 2048 向量（M-split 每核 256）、
  swiglu 要全 6144、rms1' 要 down 全输出。persistent 模型下没有
  mid-flow TCT 可用 → 必须走核间流。FLM 侧证据：算力列零激活 BD。
- **端口预算（决定拓扑）**：AIE2 每核 2 stream-in + 2 stream-out。
  单核/列 RING（shim A in + shim C out + ring in + ring out）= 恰好
  2+2 ✓。2 核/列（S1/S2 + inter fifo）超预算：S2 要 3 in（A2 + inter
  + ring）✗。
- **ring 自启动**：worker 循环序 [produce ring_out; then consume
  ring_in]——prod acquire 只等空闲 buffer（depth-2 初始即可满足），
  首轮无死锁。
- 数据流：每 channel 静态 fill 全部 [X(attn_out), res(x_n), w2, o
  blocks, w1, gateup blocks（gate/up 逐核交错 → swiglu 列内本地）,
  down blocks, qkv blocks]；exec 末 drain qkv C + x_{n+1}；三次
  all-gather 走 ring（o 输出、sw、down 输出）；gather 后每核冗余跑
  glue（rms/量化）。**x_{n+1} 留设备**——下一 exec 的 res fill 源 =
  上一 exec drain 落地的 BO 区（exec 边界 ~30µs 便宜，host 提交序
  保证顺序）。
- fallback（若 ring 放置/路由失败）：(a) 列内 inter fifo + 每层一次
  跨列 DDR 往返（2 组 TCT/exec，~−4ms）；(b) 单组 + mid-group TCT
  （~650-680µs/层 → ~24ms 设备，单独不够追平）。

**P28-3 = ring 探针**（下一步）：8 worker 各 [shim A in, shim C out,
ring in, ring out]，ring = ObjectFifo worker i → worker (i+1)%8，
trivial parrot 内核，单组 fills + drain(wait=True)。放置成功 ≠ 路由
成功——实际上板跑通才算数。
