# perf 笔记 01：TileLang 前后端的可借鉴项（M5b）

2026-09-23。来源：tile-ai/tilelang @ e4e1415（本地稀疏 clone
`~/qwen/refs/tl-ref`，blobless + sparse-checkout：根文件 / `tilelang/` /
`src/transform/`，6.9MB）+ TileSight 论文（arXiv:2607.22432，TileLang 团队
的资源向量代价模型）。目的不是移植 TileLang，而是把它的**优化手段和度量
纪律**里可迁移的部分搬进我们的 Rust 引擎与 xnpu-perf 框架。

## 结论先行（按优先级）

### 1. TileSight 资源向量包络模型 → 图编译器的代价函数

每动作 o 的资源向量 `u(o) = ⟨各独立管线上的时间⟩`，稳态包络：

```
T = T_pro + (N − d)·T_steady + T_epi
T_steady = max_r Σ_o u_r(o)      # 瓶颈资源取包络，不求和
d = stages × 并发 tile − 1        # 短环（prologue/epilogue 混稳态）修正
```

- GPU 上 12.35% GEMM MAPE；用作 autotune 剪枝可剪 95% 空间保 99.66% 最优。
- **AIE2P 特化机会（我们的简化红利）**：AIE 的 local memory 是显式管理
  （taplib sizes/strides、手动 double buffer），没有 GPU 那种概率 cache 命
  中模型——footprint 是**确定性算术**。TileSight 的管线维度在 AIE2P 上
  ≈ {向量计算核, shim DMA, stream/NoC, 级联}。
- 对我们的意义：M3 定论的「图编译合并单 PDI」选择候选布局/切法时，需要
  一个不打到板上的预筛代价函数。xnpu-perf 的 MachineModel 常数（bw、
  cu_switch、submit_overhead）+ 每动作 footprint 算术就是这个模型的退化
  版本；TileSight 告诉我们退化版也够剪枝用。

### 2. max(bw, issue) 逐语句计分 → 内核优化时的"不许侥幸"

`src/transform/layout_inference/layout_cost_model.cc`：每个语句的成本 =
max(带宽项, 发射项)，符号分析算不出的按 WorstCaseBytes 计费——
**"never profit from opacity"**（算不出就当最坏，不给模糊性送分）。
- 对应我们：w4gemvu 手工优化时（§11-§14 的多累加器/scale 折叠）每次改动
  应先算理论字节/MAC 数再上板验证，板测显著好于算术 → 先怀疑计时口径
  （M3b 的 state-poll 假完成正是反例），而不是先庆祝。

### 3. pipeline order/stage 注记为一等契约

TileLang 让开发者显式标 pipeline 阶段/顺序，编译器保序。对应 IRON 调度：
DMA 与计算的交错在图上显式表达（而不是靠 submit 顺序隐式涌现）。
短环修正 d 公式直接适用我们的 chunk 深度选择——P1 实测 chunk12→32 收益
饱和、32→168 无增益，与 (N−d) 包络形状一致。

### 4. autotune 工程结构

CompileArgs/ProfileArgs 分离、`(best_config.json, latency.json)` 磁盘缓存、
并行编译+测量 worker（tilelang/autotuner/tuner.py）。Phase M 图编译器直接
抄形状：配置编译与测量解耦才能并行；缓存键必须含 flags（我们已在 IRON 侧
用 `*.o.buildkey` 边车解决过一次同构 bug，§13）。

### 5. do_bench 度量纪律

`tilelang/profiler/bench.py`：rep 间 flush L2、min/median 双聚合。AIE 无
L2 但 DDR 流同样有残留状态（队列深度、PDI 驻留），我们的对应物 = 每次
测量前固定队列状态 + solo/burst 双口径（xnpu-perf 已内建）。

### 6. BackendModule manifest 模式

多 target 后端各带 manifest 描述能力/约束——xnpu-hal 对 npu2/npu4/npu5
的机型差异表应该长这样（M0 的 ctx 预算 16 也是 manifest 字段）。

## 明确不迁移（与失败路径）

- **SIMT/warp/TMA/mbarrier 全套 GPU 机制词汇**：AIE 是 VLIW+DMA+显式锁，
  概念不对应；迁移只会带来错误类比。
- **概率 cache 模型**：AIE local mem 显式管理 → 用确定性 footprint 算术
  （见上）。曾考虑照搬 TVM 的 cache hit 概率建模，判定为负收益。
- **TVM 本身 / tvmtir 中间表示**：我们已有 IRON 图 + 自研 HAL，再引一层
  TVM IR 只增加翻译边界。tilelang-ascend（华为昇腾适配）的经验反而是
  反面教材兼警示：显式地址规划要 day-one（先手动 annotate 再自动化
  planning pass）、跨核 flag 同步要显式——与我们 M3「图编译合并单 PDI」
  的方向互相印证。

## 与 xnpu-perf 的对接现状

| TileLang 侧 | xnpu-perf 已有 | 缺口 |
|---|---|---|
| roofline estimated_time = max(mem, compute) | MachineModel 常数 + %bw/GF 列 | peak_gflops 未知 → compute 维度缺分母 |
| do_bench L2 flush | solo/burst 双口径 | pipelined 模式仍是 state-poll drain |
| autotune 剪枝 | —（图编译器未起） | TileSight 包络模型候补 |
| WorstCaseBytes | bytes_stream 替代口径 | 符号分析不做，手填 OpMeta |

下一步落点：perf-calibrate 补 peak_gflops 分母后，TileSight 包络模型的
退化版（MachineModel + OpMeta 算术）即可作为图编译候选剪枝器原型。
