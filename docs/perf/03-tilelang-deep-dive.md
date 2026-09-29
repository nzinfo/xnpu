# perf 笔记 03：TileLang 成本模型与度量体系深挖

2026-09-29。原料：tile-ai/tilelang 稀疏 clone `~/qwen/refs/tl-ref`（本篇全部
源码引用的行号都来自它）+ TileSight 论文（arXiv:2607.22432，摘要页已复核）+
我们自己的实证台账 `notes/perf-lab.md`（P1/P6/P7/P27-3b/P27-4）。

与 01 笔记（`01-tilelang-borrowings.md`）的分工：01 是"可借鉴项清单"（五条
结论 + 不迁移清单），本篇是把其中四条——资源向量模型、max(bw,issue) 计分、
autotune 结构、do_bench 纪律——**下钻到源码行级**，补上 01 没做的：搜索空间
表示与剪枝时机、profiling 闭环的缓存/并行工程、tilelang-ascend 教训的 tl-ref
侧证据核对，以及逐条 XDNA2 映射。结论口径：**[S]** = 源码实锤（附 文件:行号），
**[P]** = TileSight/TileLang 论文口径（代码未开源，见 §2.1），**[L]** = 我们
台账实证，**[I]** = 本篇推断（推断链写明）。

一个总纲先立住：TileLang 的性能体系是**三层正交**的——

1. **编译期符号代价模型**（`layout_cost_model.cc`）：给编译器搜索用的，纯算术，
   不上设备，错率高没关系，**序保持**就行；
2. **分析型包络模型**（TileSight 论文，未进主仓库）：给人/上层调度器用的，
   直接预测 latency，有 MAPE 验证方法论；
3. **测量机器**（autotuner + do_bench）：最终裁决，工程重心在**口径纪律与
   缓存正确性**，不在统计技巧。

我们的 xnpu-perf 已有第 3 层，正在建第 2 层（数据流模型），第 1 层还没有——
本篇的映射建议主要服务第 2 层、并为第 1 层立结构参考。

---

## 1. layout_cost_model.cc 逐段解剖

文件：`src/transform/layout_inference/layout_cost_model.cc`（1201 行）+ 同目录
`layout_cost_model.h`（90 行）。01 笔记只引了它的 "never profit from opacity"
一句话；本节展开完整机制。

### 1.1 记账单位 = 语句，三态协议是核心契约 [S]

成本模型的记账单位不是 kernel、不是 tile，而是**一条碰 global memory 的语句**
（layout_cost_model.cc:57-59）：`fragment<->global tl.copy` 或带直接全局访问的
parallel loop。每条语句先被构造成一个 `StatementProbe`，其三态协议
（layout_cost_model.cc:106-135）是全文件最重要的设计：

| 状态 | 判据 | 计费 |
|---|---|---|
| 无全局访问 | `accesses.empty()` | 记 0 |
| 可测 | `measurable == true` 且评分没抛异常 | `max(bw, issue)` |
| 模型外 | `measurable == false` 或评分失败/抛异常 | `WorstCaseBytes(probe)` |

消费者必须遵守协议（`ChargeStatement`，layout_cost_model.cc:1123-1152）。
关键在第三行的语义：**"模型外"不是记 0 也不是记 ∞，而是按保守最坏情形计费**
——`WorstCaseBytes = worst_elements × repeat × segment_bytes`，即"每个元素
独占一个完整 128B coalescing segment"（layout_cost_model.cc:178-187）。原则
原文："an attempt must never profit from opacity — evaluability depends on the
layout under test"（layout_cost_model.cc:180-181，及 84-88 的总注释）。

这句话的后半段（"evaluability depends on the layout under test"）比 01 笔记
转述的更锋利：**可测性本身是被试布局的函数**。同一个语句在布局 A 下地址
仿射、可精确计分，在布局 B 下非仿射、按最坏计费——于是"把地址搞乱"在搜索
里自动变成劣势，而不需要模型懂非仿射地址有多糟。这是用计费结构而非建模
精度来引导搜索。

**发现一个协议豁口**[S]：parallel loop 的 domain 本身无法定尺寸（symbolic
parallel extent）时，`BuildLoopProbe` 返回 `nullopt`，`ChargeStatement` 对
`!probe.has_value()` 记 **0**（layout_cost_model.cc:786-788 注释 "truly
unsizeable: nothing sensible to charge"；1125-1128 "probe unavailable;
contribution=0"）。也就是说"连最坏情形都算不出大小"的语句**逃过了计费**。
这与 never-profit-from-opacity 精神相悖（symbolic shape 反而免费），推断其
取舍是：与其对 sizeable 一无所知的循环瞎编一个字节数，宁可整个跳过——但
对我们的启示是反向的：**我们的数据流模型必须要求每个动作的 footprint 可
静态定尺寸，宁可强制用户提供形状，也不要留"不可定尺寸=免费"的后门**。

### 1.2 评分的代数机器：probe-then-prove + RightInverse [S]

可测语句的打分不模拟执行，而是在 CuTe layout 代数上做符号推导
（layout_cost_model.cc:60-92 总注释）：

1. 把 fragment 的 forward maps（点坐标→(thread, slot)）打包成一个 plain
   multi-output layout，其 row-major 输出序列化恰好就是物理 cell 索引
   `thread × slots + slot`（layout_cost_model.cc:62-66）；
2. `LayoutFromTileLang` 恢复其 (shape, stride) 正规形——注释明说这是
   **"probe-then-prove, so the conversion is self-certifying"**
   （layout_cost_model.cc:66-67）：先试探性恢复、再证明等价，**错误恢复
   不可能溜过去**（`ProbeExprsToCute` 注释，layout_cost_model.cc:227-232）；
3. `RightInverse`（size 检查验证双射性，layout_cost_model.cc:353-361）+
   `Composition` 逐访问派生 `cell → element address` 布局
   （layout_cost_model.cc:367-412）；
4. 之后所有问题都变成 mode 算术：`FlatModes` 把布局压成 (extent, stride)
   对列表，用 `EvalModes` 纯 int64 求值（layout_cost_model.cc:242-255），
   "每点纳秒级"。

外部变量处理有个干净的观察：`ZeroForeignVars` 把不属于本 probe 坐标系的
整型变量置零——块索引在 region 偏移里会等量平移每条地址，从 contiguity
和 segment 几何中**抵消**；非整型外来变量保留并让 affine 恢复失败，语句
落入模型外（layout_cost_model.cc:197-209）。这是一个可以整体迁移的技巧：
**凡是"对所有访问等量平移"的量，代价模型都可以直接消去**。

**一次性 oracle 审计**的方法论[S]（layout_cost_model.cc:86-89）：mode 算术
曾经对着一个全枚举 oracle 在整个测试语料上审计过（零分歧），然后 oracle
被删除，Python 镜像留在 `maint/layout_inference` 分支。即：**快速近似 +
一次性的穷举对拍 + 保留对拍工具**，而不是长期维护两套模型。这直接可抄：
我们的数据流模型做简化（如组固定成本 75µs 单常数化）时，配一个"全量
ctrl-walk 精确计数"的离线对拍器，对拍归零后归档。

### 1.3 bw 与 issue 的具体算法 [S]

`StatementTraffic { bw, issue; Time() = max(bw,issue) }`（layout_cost_model.cc:
137-141）。两项的算法完全不对称，值得分开看：

**向量宽度先于一切**（layout_cost_model.cc:422-460）：从 32 往下试 2 的幂，
四个必要条件（414-421 注释）：(a) 宽度上限 = vectorizer 的共享策略
`MaxVectorLoadBits`（global-only 访问 256bit，否则 128bit，`loop_vectorize.cc:
1250-1257`）；(b) `slots % cand == 0`；(c) 内层 slot mode 必须是 stride-1
连续段且长度被宽度整除；(d) **基址对齐**——所有其它非零 mode stride（更高
slot modes 与全部 thread modes）都被宽度整除。这就是"vectorizer 的问题在
正规形上回答"：代价模型和实际做向量化的是**同一套判据**，不会出现模型认为
可向量化而 lower 不给的分歧。

**issue 项 = 排队论式的保守发射**（layout_cost_model.cc:477-479）：
`issue = steps × repeat × threads × vector_lane_bytes`——每线程指令深度 ×
"满忙 block 在该步数下能流的字节数"。注释（77-81）点明：空闲 lane 不缩短
深度，所以 **thread-collapse 病理在这里现形**（引用 #1729）。

**bw 项 = 精确段计数**（layout_cost_model.cc:466-516）：对每个 (vector step,
warp) 组合，逐 lane 求值派生出的地址布局，把 `[first/seg_elems, last/seg_elems]`
覆盖的 segment 去重计数（segments 向量 per warp 清空，498-504），乘
segment_bytes。两个精细处：
- **store 的 replica 守卫**：replica ≠ 0 的 lane 在 store 时休眠（读回 inverse
  的 replica 索引判断，492-495），不产生段；
- 计数复杂度被机器形状 bound 住（"steps × warps × warp_size points, bounded
  by the machine shape"，468-470）——不是按逻辑点数走。

**max 而非 sum** 是显式的瓶颈语义：一条语句要么被带宽卡要么被发射卡。
01 笔记把它对照我们的"每次改动先算理论字节/MAC"——本篇再深一层：max 的
两个分母必须**同单位**（都是字节），这是后续 §1.5 spill 计费能进同一通道
的前提。单位统一是让异质成本可比的最省力手段。

### 1.4 硬件几何参数化：只暴露两个旋钮 [S]

`BindMemoryGeometry`（layout_cost_model.cc:559-564）：warp_size 取 target 的
`thread_warp_size`（32 CUDA / 64 ROCm），`segment_bytes` 硬编码 128B——注释
明说是 NVIDIA L1 line 凝聚粒度、"a serviceable approximation elsewhere，
**until calibration demands a per-target dispatch here**"。即承认 128 是拍的，
预留了校准替换点。向量宽度上限复用 vectorizer 的策略而不是自己定
（layout_cost_model.cc:90-92, 679-680）。整个模型对硬件的依赖就这两个数 +
dtype 宽度。**启示：代价模型的硬件面越小、校准责任越集中**——我们的
MachineModel 常数（bw、cu_switch、submit_overhead）已经是这个形态，是对的。

### 1.5 spill 进同一字节通道；寄存器做字典序 tiebreak [S]

`CountSpilledBytes`（layout_cost_model.cc:892-972）：fragment 的 slot 索引若
依赖 thread 变量，整个 per-thread 寄存器数组降级为 local memory，计费
`max(array_bytes, 2 × threads × per_thread_iters × elem_bytes)`——注释
（896-902）给了理由：**让它与 io-aware 的全局字节估计"竞争而不是否决"**
（competing rather than vetoing），"8 字节数组的 spill 不能一票否决 KB 级的
并行全局带宽"。

`AttemptCost`（layout_cost_model.h:47-56）：字典序比较，`mem` 先、`regs` 后。
这样 RegisterCountCostModel（默认，regs 为主、spill 字节为 mem，
layout_cost_model.cc:974-994）和 IOAwareCostModel（io-aware，
layout_cost_model.cc:996-1118）共用一个比较语义，且"无 spill 时 io-aware
模型的排序不被寄存器数扰动"。**同构映射**：我们报告里的 %bw（分母=档位
天花板）与 GF/s 双列，实质也是"主通道 + 次通道"；P7 台账的教训"单常数
标注为噪声大、流量分桶才是稳定视角"（perf-lab.md:640-642）说明我们的
主通道该是字节、算力列做 tiebreak——与 TileLang 的选择一致。

### 1.6 语句级 memoize：收敛的尝试免费重评 [S]

`CachedStatementMem`（layout_cost_model.cc:1154-1177）：语句计费按
(op index, layout 结构相等) memoize。注释：不同 attempt root 经常收敛到同一
布局——那些 attempt 对该语句**免费**。这是对搜索空间形状的经验观察直接
变现：布局推断的解空间有大量重叠加权，缓存键用结构相等（`IsEqual`）而不是
序列化字符串。对我们：数据流模型若用于候选剪枝，同样应按"(动作, 布局)
→ 代价"缓存，代价是纯函数。

### 1.7 双模型 A/B 的落地形态 [S]

`LayoutCostModel::Create`（layout_cost_model.cc:1186-1198）：按
`tl.layout_cost_model` pass config 实例化，"io-aware"（RFC design B2）与
"register-count"（默认、历史行为）二选一，未知名字硬错误并列出合法值。
env var `TILELANG_LAYOUT_COST_MODEL` 可设默认（`tilelang/env.py:413-415`，
`tilelang/transform/pass_config.py:119`）。**新旧模型以可切换、可 A/B 的
方式共存，默认保守**——新的更精细的模型不直接替换默认，先 opt-in。
这个发布纪律本身比 io-aware 模型更值得抄。

---

## 2. TileSight 包络模型：论文口径与仓库现实

### 2.1 先划清实锤边界 [S+P]

**TileSight 的代码不在 tl-ref 里**（全仓库 grep "TileSight/tilesight" 零命中），
arXiv 摘要页明示 "TileSight will be open-sourced upon publication"。所以 01
笔记里 `T = T_pro + (N−d)·T_steady + T_epi`、12.35% MAPE、95% 剪枝保 99.66%
这些数字全部是**论文口径 [P]**，不能标注为 tilelang 仓库行为。tl-ref 里
真实存在的两件东西是：`layout_cost_model.cc`（逐语句、无时间轴、纯字节）
和 `tilelang/tools/Analyzer.py` 的 roofline（`estimated_time = max(mem_time,
compute_time)`，Analyzer.py:203-204；peak_tflops 来自 compute capability 查
表、不支持则为 None 退化为纯带宽，Analyzer.py:179-204）。TileSight 是这两
者的学术化上层，三层结构 [P]（摘要）：

- **intra-tile**：动作 = 资源向量，"spanning network, memory, and compute
  pipelines"，建模 compute-memory pipeline overlap；
- **inter-tile**：调度依赖与有序动作以**暴露合法重叠**（expose legal
  overlap），从 tile reuse distance 推多级 cache 命中率；
- **cross-device**：远端张量访问映射到 placement，走 alpha-beta stage cost。

验证口径 [P]：单 GPU **pooled MAPE 12.35%**（A100/H200/B200/B6000 四架构），
强调跨架构迁移优于基线；L2 命中率预测每卡误差 ~1 个百分点；分布式 16.18%
wMAPE（fused kernels）/ 13.52% wMAPE（e2e vLLM，≤32 GPU）；优化用途上
"pruning 95% of candidate schedules and retaining the predicted top 5% reaches
99.66% of exhaustive-search"。**MAPE 方法学上用 pooled（合并全部样本算），
剪枝指标用的是"保留 top-5% 后的穷举最优达成率"**——这两条是我们做模型
验证时该抄的指标形状：不追求单点准，追求"排序 + 剪掉 95% 后不丢最优"。

### 2.2 包络公式与我们台账的惊人同构 [P+L]

```
T = T_pro + (N − d) · T_steady + T_epi
T_steady = max_r Σ_o u_r(o)     # 瓶颈资源取包络，跨资源不求和
d = stages × 并发 tile − 1      # 短环修正（prologue/epilogue 吃进稳态）
```

对照 P27-3b 的长度标定（perf-lab.md:2013-2024）：

```
latency = 固定 132µs + bytes / 55.1 GB/s        # 隔离口径，plain op = 1 组
E2E: 设备 28.3ms = 流 18.3ms + 129 组 × 77µs/组
```

**逐项同构**：`bytes/55.1GB/s` ↔ `T_steady`（瓶颈资源 = 器件墙带宽维，
perf-lab.md:2015 "边际流速率 55.1GB/s = 器件墙"）；`132µs 固定` ↔
`T_pro + T_epi`（单组时 prologue=epilogue=组边界成本）；`129 组 × 77µs` ↔
`N × 组间不重叠的固定成本`。P27-3b 台账还给出了包络模型的**短环修正对应物**：
pair probe 547 ≈ o(175)+gateup(378) 串行和（−6µs），即"两组各付一次固定成本、
组间无重叠无额外罚金"（perf-lab.md:2019-2021）——这恰好是 d 修正失效
（d=0，pipeline 深度 1）的特例。而 P1 的 chunk12→32 收益饱和、32→168 无增益
（perf-lab.md:95-97，01 笔记已引）就是 (N−d) 形状在 chunk 维度上的体现。

**推断 [I]**：TileSight 的资源向量维度选择原则（从其三层描述读出）是
"按**独立管线**各立一维"——GPU 上是 network/memory/compute；在 AIE2P 上
天然收敛为更少维度：{shim DMA(8 通道), 核内向量 MAC, 核间 NoC/fifo, ctrl/
组固定成本}。01 笔记已给前三项，本篇补第四项的实证地位：P27-4 的三个
否定结果（输入 hoist NULL/输出 hoist NULL/−TCT 挂死）证明组固定成本**不在
指令发行时序里**（perf-lab.md:2082-2091），所以它必须是独立的一维资源，
而不能塞进"发射带宽"里——这正是"资源向量按独立管线分维"的实操含义：
**分维的依据是"能否被独立地同时阻塞"**。

### 2.3 资源争用处理 [P+I]

TileSight 的 inter-tile 层用"调度依赖与有序动作暴露合法重叠"来处理争用
（摘要），即：不做排队论，而是**枚举合法调度序后在每个时间槽上取资源
包络**。这与 layout_cost_model 的 `max(bw, issue)` 是同一哲学的两个尺度
（语句内两维取 max / 稳态跨资源取 max）。对我们的映射：npu2 上"合法重叠"
的集合是显式已知的（哪些 BD 可以 enqueue-ahead、哪些 TCT 必须串行——
P27-4 的手术就是手工枚举合法重叠的实验），所以我们的 inter-tile 层可以
比 TileSight 更简单：**重叠合法性直接从 ctrl 依赖图读出，不需要推断**。

---

## 3. 调度器/搜索空间：legalization by construction，cost model 只在终点介入

文件：`src/transform/layout_inference/layout_inference.cc`（1664 行）+
`src/transform/pipeline_planning.cc`（1374 行）。

### 3.1 布局推断的搜索空间表示 [S]

`InferInFreeMode`（layout_inference.cc:1396-1548）：

1. **连通分量切分**：union-find 按"共享 Buffer 对象"合并算子
   （layout_inference.cc:1405-1419），再按"共享底层 storage Var"跨 Buffer
   合并（处理 reshape 等别名，1420-1440）。每个连通分量独立搜索——
   **搜索空间按数据流连通性分解，不是按程序顺序**；
2. **尝试 = 选一个根**：对分量内每个算子轮换当 attempt root
   （1483-1516），root 先解自己的布局再 BFS 传播到全分量
   （`RunOneAttempt`，1340-1394）。复杂度 O(members)（register-count 模型
   在 eligible reducer root 上再加一档 scalar 尝试，1490-1492——注释明确
   "Adds one attempt per eligible root, not a Cartesian width search"，
   layout_cost_model.h:74-76）；
3. **legalization 不存在于独立的合法性 pass 里**：传播过程中任何
   `LayoutConflictException / NormalizeIterException /
   LoopLayoutInjectiveException` 直接杀死该尝试（1373-1382），infer_list
   恢复入场快照（1345, 1392）——**失败的尝试零残留**（"A failed attempt
   can leave pending propagation work. Keep both the queue and its membership
   flags local"，1347-1350）；
4. **cost model 只给完整合法解计分**（1385：`cost_model.Score(...)` 在 ok
   之后才调用），比较用 `AttemptCost.BetterThan`，**平局保留最早根**
   （1510-1512：`attempt_infer_root < best_infer_root`）——确定性 tiebreak；
5. **兜底救援**：全尝试失败时用 ReducerDstSteering 的 wide fallback seeds
   重试一次（1517-1538）——宁可接受"能编译的宽布局"也不硬失败。

**结构启示**：剪枝不是"对候选打分砍分数"，而是"约束传播自然死亡 + 少数
幸存者比代价"。cost model 的职责被压缩为**在合法解之间排序**，永远不见
非法解。这大幅降低了对模型精度的要求（不需要对非法/病态候选外推）。
对我们的图编译剪枝器原型（01 笔记"下一步落点"）这是最直接的形状参考：
候选 = 切法/布局的传播解，非法切法在传播中死掉，代价模型只排序幸存者。

另一个值得单独记录的机制：`ReducerDstSteering`（layout_inference.cc:162-235）
的**所有权语义**——未被 annotate 的 finalize dst 的第一份布局只能来自它的
finalize（"a consumer completing the buffer first is exactly how that verdict
used to get bypassed (then billed after inference as a thread-indexed publish
copy)"，layout_inference.cc:154-158）。注释里那个 "billed after inference"
的历史 bug 描述值得背下来：**让消费者先定布局，事后就会以一条本可避免的
publish copy 的形式付账**。这正是 tilelang-ascend 教训"显式地址规划 day-one"
在 GPU 语境下的同款问题（见 §6）。

### 3.2 pipeline order/stage 注记：一等契约 + 有界不动点 [S]

`pipeline_planning.cc`：`PipelineStageInfo`（406-445）携带 reads/writes 的
BufferRegion、`original_stmt_index`、`order`、`stage`、`copy_stage`、
`tma_copy`、`conditional_execution`、`producer_for_copy`、
`last_use_stmt_index`。行为：

- 用户注记 `tl_pipeline_order` / `tl_pipeline_stage` 存在则直接采纳
  （1084-1088，过滤到实际被调度的语句 1112-1117）；否则隐式规划：copy
  stages 进 stage 0、程序序为 order（1216-1246）；
- 跨 stage 依赖用 region 相交判定（`MayConflict`，72-84，IntSet 精确相交
  而非保守 may-alias）；
- **所有不动点传播都有迭代上限并对超限 FATAL**（copy-stage 依赖传播
  `max_iterations = size*4+16`，712-713、823-825）——"cyclic or pathological
  dependency graph" 直接报错而不是死循环；
- 契约用断言执行：`ValidateScalarDependencies` ICHECK 标量 def/use 不跨
  stage 且同 stage 内保序（831-852 附近）——**"调度器保证的性质"全部写成
  可失败的检查**，不是文档承诺。

对我们：ctrl code 的 fill/drain 机器（P28 设计：单组 [X, res, w, 权重块…] +
一条 drain 链）本质上就是 pipeline order 的显式注记——P28-1 的证据是 IRON
侧 one-shot 巨型 BD + 核侧锁流（perf-lab.md:2137-2152）。TileLang 这边补充
的增量是工程性的：**传播算法必须有迭代上限 + 超限即错**，以及**保序承诺
用断言执行**。我们 hoist_probe.py 的手术重排（P27-4）恰好是反面教材的
对照组：手工重排破坏了隐式契约（hs3 挂死 = 无 sync 组的槽重用，perf-lab.md:
2079-2081）——如果有 ValidateXxx 断言层，这类破坏在提交前就能被抓。

---

## 4. profiling 反馈闭环：CompileArgs/ProfileArgs、缓存、并行测量

文件：`tilelang/autotuner/tuner.py`（1419 行）、`tilelang/autotuner/param.py`
（623 行）、`tilelang/autotuner/grouped_compile.py`（199 行）。

### 4.1 两类参数的分离与缓存键 [S]

`CompileArgs`（param.py:47-95）与 `ProfileArgs`（param.py:98-151）都是 frozen
dataclass，都实现 `__hash__` 且**都进 autotune 缓存键**（tuner.py:348-359）。
三个细节：

- **CompileArgs.__hash__ 对 pass_configs 先过 `normalize_pass_configs`**
  （param.py:83-92），注释："Resolve env-var-derived pass-config defaults so
  a changed environment does not silently reuse tuning results produced under
  another one"——**环境变量派生的默认值必须先解析再入键**。这是我们 §13
  buildkey 教训（缓存键必须含 flags）的更完整表述：键里的每个成分都要是
  **解析后的终值**，不是"配置入口的地址"；
- **ProfileArgs 也入键**（warmup/rep/timeout/backend/tolerance/skip_check/
  cache_input_tensors，param.py:136-151）——**测量口径是结果语义的一部分**：
  用 rep=10 测出的 best_config 不能被 rep=1000 的查询复用；
- **回调无恒等则禁缓存**（tuner.py:328-336）：ref_prog/supply_prog/
  manual_check_prog 任一非空 → `generate_cache_key` 返回 None → 完全不缓存。
  注释："Arbitrary callbacks do not have a reliable persistent identity …
  Avoid cache reuse rather than risk serving results produced with different
  input or validation behavior"。**不可散列的影响源一律退出缓存**，而不是
  用函数名近似。

另外缓存键还包含**闭包自由变量的值**（tuner.py:846-864：把 fn 的 cell
contents 里可序列化的 int/float/str/bool 提出来），解决"M/N/K 在闭包里、
只哈希源码会把不同形状混进一个键"的问题。

### 4.2 磁盘缓存的原子性与完整性 [S]

`AutotuneResult.save_to_disk`（param.py:406-495）：全部文件（best_config.json、
function.pkl、out_idx.json、latency.json、kernel 源/库/params）写入共享
`.staging` 下的临时目录，**逐文件 fsync 后整体 `os.rename` 发布**
（478-492）；完整性 = 必需文件集齐全（`_get_complete_result_files`，
585-600），残缺目录在发布前清除（612-618），rename 碰撞视为"别的进程赢了
竞态"并放弃（487-492, 620-623）。**读侧同样校验**：load 时必需文件不齐 →
返回 None 当 miss（353-355）。这是多进程共享缓存的标准姿势，xnpu-perf 的
calibration JSON（machine_model.json 的 overlay 语义，P7）目前是单文件直写，
若将来多进程并行校准，照抄 staging+rename。

### 4.3 并行结构：编译池 × 测量 worker × 流水 [S]

tuner.py 的 `run()`（809-1167）：

- **编译并行**：ThreadPoolExecutor，worker 数 = 可用 CPU × 0.9（env
  `TILELANG_AUTO_TUNING_CPU_UTILITIES` 等，tuner.py:428-447）；
- **分组编译**（grouped compile）：CUDA+tvm_ffi 下按 per-config pass_configs
  分桶、每桶 `group_compile_size` 个 config 一组，**合并设备 IR 一次编译、
  共享 device module**（tuner.py:506-522；grouped_compile.py:29-43 的五步
  flow）。同 flag 的 config 合并编译 = 摊薄编译固定成本；
- **测量 worker**：每设备一个线程，队列驱动（`_benchmark_worker_loop`，
  533-638）；`use_pipeline=True` 时测量不等编译结束，主线程
  `FIRST_COMPLETED` 事件循环边收编译结果边喂测量队列、非阻塞排水
  （1084-1116）——**编译与测量的生产者-消费者流水**；
- **超时 = 放弃不击杀**：每次测量调用放进新 daemon 线程 `join(timeout)`
  （582-633），超时记 "timeout" 结果继续跑下一个 config——承认无法安全
  中断 GPU 调用，只做隔离。挂死 config 不拖死整个 tuning 会话，这个语义
  对我们（板上挂死常态，P27-4 hs3）特别相关；
- **early_stop 跨 worker 共享**：`shared_best_latency` 是跨线程共享的单元素
  list（897, 989-990），传给 do_bench 的 `early_stop_baseline = best × factor`
  （tuner.py:717-724）——5 次预估超阈值就跳过完整测量（bench.py:222-230）。
  **测量中的自适应剪枝用"当前最优的倍数"做阈值**，且 factor 必须 ≥1.0
  （841-842）；
- **多 GPU**：worker 按设备划分、任务按 index 确定性映射（1020：
  `idx × n_queues // n_configs`）——同 config 在同设备复测可复现；
- ref 延迟只测一次并缓存（`ref_latency_cache`，727-736, 740）。

### 4.4 输入供给的坑位 [S]

`cache_input_tensors` 默认 True（param.py:134），跨 config 复用输入张量，
但带 dtype/shape 兼容检查，不兼容就重新生成并告警（tuner.py:679-706）——
**复用必须自证兼容**。另一处：`set_autotune_inputs` 上下文捕获的真实张量
优先于任何 supply_type（tuner.py:249-269），且线程安全（"freeze captured
tensors now so benchmark worker threads do not lose them via thread-local
storage lookups"）。

---

## 5. do_bench 的统计纪律：口径细节与 01 笔记的修正

文件：`tilelang/profiler/bench.py`（249 行）、`torch_bench.py`（195 行）、
`wall.py`（51 行）、`profiler/__init__.py`。

**预算制而非次数制** [S]：`warmup=25, rep=100` 的单位是**毫秒预算**不是
次数（bench.py:60-61）；先用 5 次估时（209-219），再
`n_warmup = warmup/estimate`、`n_repeat = rep/estimate`（233-234），最少 1 次。
短 kernel 自动获得大样本、长 kernel 自动小样本——总时长受控。
我们 xnpu-perf 的 iters 是固定次数制（5 次中位），对 µs 级探针样本偏少；
预算制值得抄。

**L2 flush 的三种实现** [S]：event 后端每次 rep 前 `cache.zero_()`——
256MB（默认 `cache_size=256`，MB）缓冲清零，fast_flush 用 int32 少写 4×
字节（bench.py:65, 202-207；torch_bench.py:77-81）；cudagraph 后端 flush
在 graph 之外、replay 之前（torch_bench.py:158-180 注释："Cache flushing is
done before graph replay, not within the graph"——图内不能有 flush）；cupti
后端把 flush 包进 `record_function(_CACHE_FLUSH_ID)` 标注、事后从总时长里
**减掉 flush 自身的 kernel 时间**（torch_bench.py:118-141），并处理了
"`cache.zero_()` 与用户 `torch.zeros` 共享 kernel 名" 的排除歧义问题。
第三种（按时长排除 flush）对我们的 burst 口径有直接参考价值：我们 burst
连发前的队列状态固定动作若自身有设备成本，应从设备时间里显式扣除。

**聚合方式——修正 01 笔记** [S]：01 笔记写"min/median 双聚合"。实况：
`return_mode` 支持 min/max/mean/median + 任意 quantiles，**默认 "mean"**
（bench.py:40；Profiler.do_bench 同样默认 mean，profiler/__init__.py:230）。
min/median 是可选项不是默认。wall 后端的 median 用 `median_low` 并注释
"matching the GPU benchmark's aggregation conventions"（wall.py:48-49）——
两种计时路径的聚合语义刻意对齐。**对我们的修正含义**：默认 mean 的理由是
autotune 要估计"期望成本"（含波动），而设备极限口径该用 min——我们 P5a
的结论"min 才是设备极限口径"（perf-lab.md:524）与 TileLang 的默认选择
**不冲突而是场景不同**：选优用 mean、定天花板用 min/多跑取最大（P7 的
"下界+多跑取最大"语义）。文档里应写明每个数字的聚合口径，这正是 P1 立的
规矩（链 wall 从均值改 min/med/max 分布）。

**early_stop_baseline 语义** [S]：阈值在 estimate 阶段就判（bench.py:222-230），
返回 estimate 本身并注明 skipped——**被剪枝的样本不进入统计**，也不冒充
完整测量。报告可区分"测了/估了"。

**方差处理** [S]：没有比 quantiles 更花哨的东西——不剔除离群点、不做
bootstrap；方差通过 quantiles（如 [0.5, 0.95]）暴露给调用者自行判断。
统计纪律的重心全在**口径控制**（flush、同步、warmup）而不是**事后统计**。
这与我们 P7 的发现（跑间漂移 40% 级、跨 op 相关，perf-lab.md:635-639）的
应对一致：漂移靠"每次重要测量前跑一次 perf-calibrate 取当日基线"
（perf-lab.md:657-658）解决，不靠统计技巧。

**wall 后端的同步纪律** [S]：每次采样**前后都 synchronize**（wall.py:26-33，
"exclude preceding device work" + 计入被计时调用的完成）——solo 口径的
标准姿势，与我们 solo 窗口 = submit+wait 同构。

---

## 6. tilelang-ascend 的教训：tl-ref 侧能对上多少

01 笔记从 tilelang-ascend 总结了两条：显式地址规划 day-one、跨核 flag 同步
显式。核对 tl-ref（sparse clone 只有根文件/`tilelang/`/`src/transform/`）：

- **ascend 后端代码不在这个 clone 里**：全库 grep 只命中
  `tilelang/tools/lower_trace/core.py:87-88` 的两个 codegen FFI 名字
  （`target.build.tilelang_ascend` / `tilelang_ascend_pto`）。仅有的信息：
  它们被列在 `_SOURCE_ONLY_CODEGEN_FFIS`——"返回的 module 按源码而非已编译
  二进制消费，用户编辑过的工作副本可以安全地作为新 CSourceModule 返回"
  （core.py:73-76 注释）[S]。即 ascend 后端在 lower_trace 的世界观里是
  **源码级 codegen 边界**，与 `tilelang_c`/`webgpu` 同类。这弱证实了
  ascend 适配是"把布局/地址决策留在源码层可见"的路线 [I]；
- **主仓库侧的同型机制**（可作为教训的 GPU 侧对应物）：
  1. `ReducerDstSteering` 的所有权语义（layout_inference.cc:144-161）——
     消费者抢先定布局 → 事后以 thread-indexed publish copy 计费。这就是
     "地址/布局规划不 day-one 显式，就会在后面的 pass 里付搬运账"的编译器
     版 [S]；
  2. `plan_update_buffer_allocation_location.cc`、
     `hoist_global_buffer_allocations.cc` 等 pass 名单（`src/transform/`
     目录）显示主仓库把**分配位置规划**独立成 pass [S]——布局与放置是
     显式规划对象，不是 lowering 的副产品；
  3. `verify_buffer_init.cc` / `verify_parallel_loop.cc` / 等一族 verify pass
     [S]：跨核可见性类契约（初始状态、并行合法性）用**编译期验证 pass**
     兜住——对应"跨核 flag 同步要显式"的教训：同步语义必须是被验证的
     契约，我们的对应物是 P19b 的槽重用警告与 P27-4 的 hs3 挂死教训
     （TCT 承担通道/BD 状态 retire，perf-lab.md:2079-2081）。

结论：tilelang-ascend 的两条教训**不能从 tl-ref 直接溯源**（代码在别的
仓库），但主仓库有三个同型机制佐证其一般性。标注：01 笔记的 ascend 部分
维持"外部 repo 口径"，本篇不动它的结论、只补主仓库侧证据。

---

## 7. XDNA2 / NPU2 映射表

逐条：TileLang 设计思想 → 我们的等价物 / 适配点 / 不适用点。硬件事实以
台账为准：npu2 分区 = 8 核（2 列×4 行）+ 4 shim × 2 MM2S 通道 = 8 通道
（perf-lab.md:2028-2030）；器件墙 55.1GB/s（P27-3b 复现 P12 的 55.6）；
task-group 固定成本 75-132µs（隔离 132 / E2E 有效 77，perf-lab.md:2015-2024）；
PMEM 16KB、L1 64KB/tile（63.3KB 现状，P28 预算）；软浮点禁用；load_v 16B
对齐硬约束（perf-lab.md:831-833）、depth-2 fifo 64B 对齐（:913）。
（任务描述写"8 shim×2 通道"，按台账口径是 4 shim×2 通道=8 通道，本文
统一用台账口径。）

| # | TileLang 设计思想（源码证据） | XDNA2 等价物 / 适配点 | 不适用或反向 |
|---|---|---|---|
| 1 | 三态计费协议：0 / 保守最坏 / 实测模型（cost_model.cc:106-135, 1123-1152） | 数据流模型的动作分类：无全局流（L1 内交接）记 0；OpMeta 手填可信；**填不出 = 按最坏计费并显式标注**，不许"字段缺失=免费"。xnpu-perf 报告的 bytes_stream 替代口径（01 表格）升级为三态 | — |
| 2 | never profit from opacity；evaluability 依赖被试对象（cost_model.cc:180-181） | 图编译器选布局时，符号分析不出的地址形态按 WorstCaseBytes 类逻辑计费（如无法证明 16B 对齐就按 strided 档 0.95 计，而不是按 slot 档 41.2 计）——**对齐证明就是我们的"affine 恢复"** | 豁口要堵：symbolic extent 免费逃逸（§1.1）提醒我们 footprint 必须强制可定尺寸 |
| 3 | 逐语句 max(bw, issue)，两项同单位字节（cost_model.cc:137-141, 477-479） | 逐动作 max(流时间 = bytes/55.1, 发射/ctrl 时间)。我们的"issue 项"实证上是**组固定成本**（75-132µs），不是指令发射——P27-4 证明指令发行时序不是变量（perf-lab.md:2082-2091） | "满忙 block 的发射带宽"形式不适用（VLIW+DMA 无 warp）；但"第二维独立资源取 max"结构完全适用 |
| 4 | 向量宽度四条件：上限/整除/连续段/基址对齐（cost_model.cc:414-460） | 直接映射 AIE 对齐律：load_v 16B（perf-lab.md:831-833）、fifo 64B（:913）。数据流模型里 BD 打包合法性检查 = 同款判据；pack 律"列 w 装 pos(w) 位"是我们的 mode 算术 | 256bit 宽度档（MaxVectorLoadBits）不适用，AIE 向量宽由内核口味决定 |
| 5 | spill 进同一字节通道 max(array, 2×threads×iters×elem)（cost_model.cc:892-972） | L1 stash 预算的计费方式：P28 把 partials stash 进 .bss（+4KB/核 vs 63.3KB 现状）应记为"L1 占用↔DDR 往返节省"的同通道比较，而不是硬约束一票否决 | 寄存器计数 tiebreak 无对应（无寄存器分配问题）；字典序 tiebreak 可用"组数"替代 regs |
| 6 | 语句级 memoize by (op, layout 结构相等)（cost_model.cc:1154-1177） | 候选剪枝器的 (切法, 布局) → 代价缓存；纯函数无副作用才可缓存 | — |
| 7 | 连通分量 + 逐根尝试 + 传播失败即剪枝 + 完整解才计分 + 平局取最早（layout_inference.cc:1396-1548） | 图编译切法搜索：按数据依赖连通分量分解；非法切法在依赖传播中死掉；代价模型只排序幸存者。**搜索粒度必须等于成本原子粒度**——我们的成本原子是 task group（P27-4），不是 BD/指令 | 尝试次数 O(members) 的轮换根策略对我们偏大（层内算子少，可全枚举） |
| 8 | TileSight 包络 T = T_pro + (N−d)T_steady + T_epi；T_steady = max_r Σu_r [P] | **已在台账闭合**：latency = 132µs + bytes/55.1（P27-3b 拟合）；E2E = 流 + 129 组×77µs。资源维 u = {DMA 流, MAC, 核间 NoC/fifo, 组固定成本}。d 修正对应 chunk 深度饱和（P1） | 概率 cache 命中率层不需要（AIE local mem 显式管理，01 已定）；GPU 的 network 维在单器件 npu2 退化为核间 fifo 维 |
| 9 | 分维依据 = 能否被独立地同时阻塞 [I from P27-4] | 组固定成本独立成维的实证：输入/输出 enqueue-ahead 双 NULL ⇒ 与 DMA 发行独立 | — |
| 10 | pipeline order/stage 注记 + 有界不动点 + 断言执行契约（pipeline_planning.cc:406-445, 712-713, 831-852） | ctrl code 的 fill/drain 集就是 pipeline 注记的物化（P28-1: 52 条 ctrl 指令/exec）；hoist_probe 类重排工具必须先过"契约断言"（槽不重用、TCT retire 完整——hs3 挂死即违约实例） | — |
| 11 | CompileArgs/ProfileArgs 分离 + 解析后终值入哈希 + 测量口径入键 + 回调禁缓存（param.py:47-151; tuner.py:328-359） | xnpu-perf 校准缓存键应含：MachineModel 版本、口径（solo/burst）、深度档、当日基线引用。我们 §13 buildkey 教训的完备化 | — |
| 12 | staging+fsync+rename 原子发布 + 完整性=文件集齐 + 读侧校验（param.py:406-495, 585-618） | machine_model.json 的多进程安全化（目前单文件直写 + overlay；零天花板守卫已有，P7 坑 1） | 单进程阶段可缓做 |
| 13 | 编译池×测量 worker×FIRST_COMPLETED 流水 + daemon 线程超时放弃 + early_stop 共享最优（tuner.py:428-447, 533-638, 1084-1116） | 上板扫描框架：编译（Rust 构建 ctrl bin）与测量流水；**挂死 exec 放弃不击杀**（我们挂死常态）；剪枝阈值 = 当日最优×factor≥1 | 多 GPU 部分不适用（单板）；线程模型改进程池（pyxrt 需独占） |
| 14 | grouped compile：同 flags config 合并一次设备码编译共享 module（tuner.py:506-522; grouped_compile.py:29-43） | 同 CU 多 op 的 ctrl code 合并编译（我们已天然有：PDI 与形状无关、M 在 ctrl 里，P8）；反向借鉴：**不同形状同内核的 ctrl bin 生成可共享模板** | — |
| 15 | do_bench 预算制（25/100ms）+ L2 flush 三实现 + mean 默认/口径明示 + flush 成本显式扣除（bench.py; torch_bench.py:118-141） | 预算制改 iters；burst 前的队列固定动作若占设备时间应从 burst 时间扣除（cupti 模式同型）；报告每数标注聚合口径（P1 已立） | L2 flush 本身无对应物（无 L2），对应物=队列深度/PDI 驻留状态固定（01 已定） |
| 16 | 一次性 oracle 审计后删除 oracle（cost_model.cc:86-89） | 数据流模型 v1 与全量 ctrl-walk 精确计数对拍（tools/hoist_probe.py 的 walk 规则可扩展成对拍器），零分歧后归档对拍器 | — |
| 17 | tilelang-ascend：显式地址规划 day-one、跨核同步显式（外部 repo，01 笔记） | 主仓库同型证据：ReducerDstSteering 所有权（消费者抢定布局→事后 publish copy 计费）；allocation-location 规划独立 pass；verify_* 断言族。XDNA2 版：fill 源/窗口/槽重用必须 day-one 进图（P28 层内全静态 fill 正是此路线） | — |

**给数据流模型的结构性建议**（浓缩为六条，供另一条线直接取用）：

1. **模型形态**：`T_exec = Σ_group (C_group + Bytes_group / B_eff)`，其中
   C_group ≈ 75-132µs（隔离/E2E 两档校准）、B_eff = 55.1GB/s 器件墙（长短
   流通用，P27-3b 已闭合）。这就是 TileSight 包络在"无流水重叠（d=0）"下
   的特例；将来 layer-v2（1 组/层）自动退化为 `T = C_group + T_steady`，
   与 P28 生产路线的 608µs/层预算（perf-lab.md:2106-2110）同式。
2. **动作三态**：每个动作给 (footprint_bytes, channel_mask, provenance)，
   provenance ∈ {measured, symbolic-worst, assumed}——TileLang 三态协议
   的直接移植；报告层按 provenance 分列，反推式验证的教训
   （perf-lab.md:1815-1817）靠 provenance 标注守住。
3. **分维**：资源向量固定四维 {DMA 通道聚合带宽, MAC, 核间 fifo, 组边界}，
   分维判据 = "能否独立同时阻塞"（P27-4 实证支持第四维独立）。
4. **重叠合法性从 ctrl 依赖图读出**，不做概率重叠推断——我们的依赖是
   显式的（TCT/fifo 序），比 TileSight inter-tile 层的"expose legal overlap"
   问题更容易。
5. **校准闭环**：machine_model 键含口径与解析后 flags；每次重要测量前
   perf-calibrate 取当日基线（P7 结论）；多跑漂移用"下界+多跑取最大"。
6. **验证指标形状**：pooled MAPE + "剪 top-5% 后穷举最优达成率"（TileSight
   口径）；对拍器 = 全量 ctrl-walk 精确计数（#16）。

---

## 8. 01 笔记被修正/深化的点

1. **do_bench 聚合口径修正**：01 §5 "min/median 双聚合" → 实际默认
   `return_mode="mean"`（bench.py:40; profiler/__init__.py:230），min/median/
   quantiles 是可选项。两个场景分工：选优 mean、设备极限 min——与我们
   P5a/P7 的结论互补而非印证原表述。
2. **TileSight 归属澄清**：01 §1 的包络公式与 12.35%/99.66% 数字是**论文
   口径**，代码"upon publication"才开源、tl-ref 里没有（全库 grep 零命中）。
   引用时应标论文不标仓库。tl-ref 里实际存在的预测器是 layout_cost_model
   （逐语句、纯字节、无时间轴）与 Analyzer.py roofline
   （`max(mem, compute)`，Analyzer.py:203-204）两个。
3. **"max(bw, issue) 逐语句计分"的完整语义**：01 §2 只说了 worst-case 原则；
   本篇补齐 issue 项的排队论式定义（steps×threads×lane_bytes，空闲 lane 不
   缩短深度）、bw 项的段去重计数、store replica 守卫，以及**两项同单位
   （字节）是 spill 可进同一通道的前提**（§1.3/§1.5）。
4. **autotune 结构的缓存键完备化**：01 §4 说"缓存键必须含 flags"；本篇
   补上三条更强的：env 派生默认值先解析再入哈希（param.py:83-92）、
   **测量口径（ProfileArgs）也入键**（param.py:136-151）、回调无恒等直接
   禁缓存（tuner.py:328-336）、闭包自由变量入键（tuner.py:846-864）。
5. **01 未涉及的搜索空间表示**（本篇 §3）：连通分量分解 + 传播失败即剪枝
   + 完整解才计分 + 平局取最早的完整流程；cost model 永不见非法解——
   这是对"图编译候选剪枝器原型"最直接的形状输入。
6. **tilelang-ascend 教训的证据等级**：01 引的两条教训**无法从 tl-ref 溯源**
   （ascend 代码不在本 clone），本篇补了主仓库三个同型机制
   （ReducerDstSteering 所有权语义、allocation-location pass 族、verify_*
   断言族）作为一般性佐证，但教训本身维持外部口径。
7. **包络模型与台账的定量同构**（本篇 §2.2）：01 只说"退化版也够剪枝用"；
   本篇给出逐项映射——132µs ↔ T_pro+T_epi、55.1GB/s ↔ T_steady、
   129 组×77µs ↔ N×组固定成本、chunk 饱和 ↔ (N−d) 形状——并据此立了
   "第四资源维（组边界）独立性"的实证依据（P27-4 双 NULL）。

## 留痕

- 本篇未改动任何现有文件；新建本文件。
- tl-ref 为 sparse clone（根文件 / `tilelang/` / `src/transform/`），
  `src/layout`、`src/op` 未 checkout：CuTe 代数实现（`LayoutFromTileLang` /
  `RightInverse` / `Composition`）只从 cost_model 的注释与调用面引用，未读
  实现——如需行级引用需补 checkout。
- TileSight 论文摘要页 2026-09-29 复核（arxiv.org/abs/2607.22432）；正文
  公式（d 修正、u(o) 维度细节）以 01 笔记转述为准，未逐式核对 PDF。
