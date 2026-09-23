# 06 · 新引擎架构草案（xnpu）

## 设计原则（源自调研结论）

1. **静态优先**：NPU 是数据流阵列，调度尽量编译期固化（HRX .xdna 哲学、FLM 指令流
   同构），宿主只做"提交命令包 + 等 retirement"。
2. **薄 HAL、多传输轨可插拔**：XRT/ERT（复用/对照）、HSA/AQL（实验）、
   libamdf+DRM（主轨）——同一抽象下三种实现，防 API 变动（hrx 尚 early-access）。
3. **层参数编译期绑定**：layer 图必须按确切形状+层数编译（28↔42 惨案 + NPU4/NPU5
   不互换的官方背书）。引擎加载工件时校验"编译档案指纹"（形状/层数/设备 ID/固件）。
4. **混合执行是目标形态**：iGPU HIP 快 prefill（HRX2 lane 实测 1200+ tok/s）+
   NPU decode（能效），GPU 路径留接口。

## Crate 划分（Rust workspace）

```
xnpu/
├── crates/
│   ├── xnpu-hal/        # unsafe 圈禁区：DRM ioctl(amdxdna)、mmap、BO 生命周期、
│   │                    #   命令提交；trait DeviceBackend {probe/load/submit/wait}
│   │                    #   实现：XdnaDrm(主) / XrtCompat(对照,extern "C") / Hsa(实验)
│   ├── xnpu-artifact/   # 工件解析：.xclbin(AXLF 7 段) 与 .xdna(镜像) 双格式；
│   │                    #   编译档案指纹校验；绑定=地址 patch（safe Rust，核心雷区）
│   ├── xnpu-compiler/   # loomc C API 绑定（AOT+JIT 特化：形状/层数参数）；
│   │                    #   备选：调 mlir-aie Python 流程的离线编译器（构建期）
│   ├── xnpu-engine/     # 静态调度器：per-layer 命令序列预编排；KV cache 分配器
│   │                    #   （42 层环形布局，层敏感编址）；Q4NX 权重映射；执行循环
│   ├── xnpu-model/      # MiniCPM5/qwen3-arch 模型定义；GQA 16:8→2:1 已在权重侧
│   │                    #   展开(作者方案可复用)；tokenizers crate；minijinja 模板
│   ├── xnpu-server/     # axum：OpenAI 兼容 /v1/chat/completions（流式+工具）
│   └── xnpu-cli/        # 探测/加载/基准/诊断（hrx-info 的对应物）
└── kernels/             # 设备侧（不经 Rust）：Loom .loom 源或 mlir-aie Python 图
```

依赖方向：server → model → engine → (artifact, compiler) → hal。hal 之外零 unsafe。

## 双轨设备策略

| 轨 | 编译 | 加载/提交 | 用途 |
|---|---|---|---|
| **B（先跑通）** | mlir-aie/IRON Python 流程离线出 xclbin | amdxdna DRM ioctl 直加载 HW context（绕过 XRT，M0 验证） | 尽早点亮单算子/单层 |
| **A（主目标）** | Loom loomc 出 .xdna | libamdf range submission | 长期形态，开放度最高 |

两轨共用 engine/artifact/model 上层——只有 hal 与 compiler 不同。

## 里程碑（2026-09-22 重排：复用优先，推理最早化）

### 阶段 R：Python/IRON 复用路线（通往推理的主线，全部现成代码起步）

- **R0 点亮（0-2 天，复用率 100%）**：按 mlir-aie 指令装 XRT 用户态
  （`source /opt/xilinx/xrt/setup.sh`——IRON 宿主走 XRT，此阶段 XRT 是工具不是敌人）；
  `ironenv` venv + `pip install -r requirements.txt`（wheel 支持 Python 3.10/3.12/3.13）；
  跑通 `iron/operators/axpy/test.py` = 公开栈在本机 NPU 的第一次计算。
- **R1 首个 LLM 推理（1-3 天，复用率 ~100%）**：跑
  `iron/applications/llama_3_2_1b/inference.py`——Llama-3.2-1B 用 ungated 镜像
  （unsloth/ 或 NousResearch/，meta 原仓 gated），`torch_to_npy.py` 转权重。
  **自此本机有非 FLM 的 NPU LLM 推理**（golden-model 速度，先正确后快）。
- **R2 MiniCPM5 进 IRON（1-2 周，三份现成代码拼接）**：
  1. KV 头展开 2→8：作者 `expand_kv_heads.py`（现成）
  2. Q4NX 权重：直接用作者 HF 包里的 `model.q4nx`，或 FLM_Q4NX_Converter 重转（现成）
  3. 图代码：IRON llama 应用 `transformer.py`/`gqa.py`/`feed_forward.py` 改造——
     42 层、GQA 16:8、d_ffn 6144、d_head 128；MiniCPM5 **无 qk_norm，比 llama 还简单**；
     权重经 IRON **Dequant(Q4NX→bf16) 算子**（公开，`iron/operators/dequant/`，
     AWQ→Q4NX→bf16）流入 GEMM——2B 权重仅 ~1.2GB，DMA 逐层喂
- **R3 正确性**：42 层 decode 输出 vs GGUF(8080) 参照对比——踩在 FLM 倒下的确切位置。

### 阶段 M：Rust/性能（R3 之后才动手；即原 M 系列）

- **M0 DRM 点亮**：Rust 复刻 `irene-xdna-run`（mul_i32 夹具）；验证绕 XRT 直载。
  蓝本升级：`~/qwen/xnpu/iree-amd-aie` 的 `driver/amdxdna/shim/linux/kmq/`
  （生产级开源直连 DRM 用户层，device/bo/hwctx/hwq/ert 全套 ~2.5k 行，
  见 docs/04 §J）——xnpu-hal 按 crate 拆模块对照移植。
- **M1 单算子基准** → **M2 单层融合**（FusedDQP 思路）→ **M3 KV/checkpoint** →
  **M4 模型+server** → **M5 端到端提速** → **M6 混合执行**（iGPU HIP prefill +
  NPU decode，HRX2 lane 实测 1200+ tok/s prefill 的路子）。

R 阶段产物（Python 图 + 验证脚本）保留为 M 阶段的行为参照实现（golden model），
Rust 实现逐算子对拍。

## 硬件实测约束（2026-09-22，R1 期间发现）

**并发 HW context 上限随 NPU 型号不同，且无法查询，只能实测**。
官方规范（内核文档 [accel/amdxdna/amdnpu](https://docs.kernel.org/accel/amdxdna/amdnpu.html)）：

| NPU | 并发 workload context | 阵列拓扑 | 共享 L2 | 每 context 指令缓冲 |
|---|---|---|---|---|
| Phoenix / Hawk Point | **6** | 4×5 | 2560 KB | 64 MB（host 驻留，PASID 保护） |
| Strix Point | **16** | 4×8 | 4096 KB | 64 MB |
| **Strix Halo（本机 17f0:11）** | **16**（实测；官方表未列） | （xrt-smi 报 6×8） | — | 64 MB |

规范要点（原文）：
- 每 workload context 由**独立 ERT 固件实例**服务；管理走单一特权 MERT ——
  context 上限本质是固件实例表上限，这就是 16 是硬上限的原因。
- **Mixed Spatial and Temporal Scheduling**：spatial 分区可独占绑定 1 个 context；
  另一分区可被**多个 context 时间复用**（微控制器在切换时改写 PASID）。
  所以 16 个 8 列 context 能塞进 8 列阵列——靠时间片，不是列够用。
- 驱动内 **Resource Solver** 按 workload 元数据（所需列数）+ 启发式决定
  （重）分区策略，固件强制执行 context↔列绑定。
- mlir_aie `NPU_CONTEXT_CACHE_SIZE = {npu1: 6, npu2: 32}`：npu1=6 与官方一致，
  **npu2=32 是错的**（官方 16）——我们踩坑的直接原因。
- 16 ctx × 64 MB = 1 GB host 锁页 —— memlock unlimited 要求的真正来源。

推论（自研引擎设计约束）：
1. `xnpu-hal` 初始化必须 `probe_context_budget()`：开哑 context 递增直到失败，缓存结果；
   不信 profile 名（同代 iGPU 系列、不同 NPU，预算不同）。
2. 图规划以 context 预算为第一约束：42 层模型必须**大融合图复用 context**
   （FLM 的 layer.xclbin 思路：一层一图，42 层共用 1 个 context），或逐层串行复用。
3. IRON 现状：per-op xclbin × LRU 缓存（`XRT_CONTEXT_CACHE_SIZE` 可调），但
   runlist 在 prepare 时冻结绑定 context，逐出即死——R2 若图数超预算需改造为惰性重载。

## 风险登记

| 风险 | 应对 |
|---|---|
| libamdf/loomc early-access，API 会动 | 薄 HAL + 轨 B 兜底；pin commit |
| .xdna/镜像格式文档滞后 | M0 起就写解析器 + 指纹校验，格式漂移早暴露 |
| NPU5 档案严格性（NPU4≠NPU5） | 工件带编译档案指纹，加载即校验 |
| Loom XDNA 目标成熟度（README 自述以 HSACO 最成熟） | 轨 B 先行，A 轨并进 |
| 固件 1.1.2.65 与驱动 0.7 版本耦合 | 环境快照记录在 docs/03，复现脚本化 |
| 自研图性能初期远低于 FLM | 接受；M6 才谈优化，期间 GGUF@8080 是日常主力 |
| 工程量（周-月级） | M0-M2 是可放弃点：任一里程碑受阻即评估止损 |
| context 预算耗尽（上文实测 16） | 图规划器按预算分箱；融合优先；引擎侧惰性重载 |
