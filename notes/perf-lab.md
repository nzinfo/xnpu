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
