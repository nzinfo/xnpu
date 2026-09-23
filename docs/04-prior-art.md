# 04 · 外部路线调研（prior art）

> 调研日期 2026-09-22。来源：mlir-aie `docs/UsedIn.md` 总索引 + 逐仓库核实。

## A. Gemma3-on-NPU 论文 ⭐ 最重要

**"Mapping Gemma3 onto an Edge Dataflow Architecture"** —
[arXiv:2602.06063](https://arxiv.org/abs/2602.06063)（2026，非 AMD 第三方）

- **第三方用公开 MLIR-AIE + IRON 端到端部署 Gemma3 1B/4B 全家到 Ryzen AI NPU**
  ——"自建引擎不靠 FLM"的成功先例与配方论文
- 发表 **Q4NX 4-bit 格式**（与 FLM 权重格式同名同源）——此前该布局只存在于闭源引擎
- 三个图技术，全部对准 decode 痛点：
  - **FlowQKV**（prefill）：分块流水注意力
  - **FusedDQP**（decode）：反量化+投影融成单内核（正对 FLM 卡死的 layer 图）
  - **FlowKV**（decode）：重构注意力使 decode 期访存高利用
- 成绩：vs iGPU prefill **5.2×** / decode **4.8×**；vs CPU 33.5×/2.2×；能效 67×/223×
- ⚠️ 无代码（CC BY-NC-SA）——是蓝图不是现成引擎；本项目 06 架构的核心理念来源

## B. IRON Llama-3.2-1B 参考应用

[amd/IRON `iron/applications/llama_3.2_1b/`](https://github.com/amd/IRON)：完整推理
（`src/block/transformer.py`、`gqa.py`、`feed_forward.py` + `inference.py` + safetensors
转换脚本）。定位 golden-model（数值验证），bf16，性能远低于生产引擎。**是宿主循环
与图组织的最好公开骨架。**

## C. Triton-XDNA（AMD 官方实验）

[amd/Triton-XDNA](https://github.com/amd/Triton-XDNA)（MIT）：`@triton.jit` →
vendored TTIR → triton-shared(Linalg) → MLIR-AIR + Transform dialect → aircc →
xclbin/elf/pdi。XRT 与 HSA 双派发；宣称编译器 matmul（I8/I16/BF16）达手写内核
≥90% 吞吐（>90% 配置 ≥90% 基线）。生产率最高的内核编写路径。

## D. HRX2 llama.cpp lane（1bit-MONSTER 研究）

llama.cpp fork + HRX（HIP 兼容层/HSA）在 **gfx1151（本机 iGPU）** 上跑 **Q4NX 权重**：
- HIP 快 prefill 1227–1313 tok/s + HRX 暖 decode ~80–87 tok/s 混合设计
- round 25i：真零 DMA 拷贝，decode +10–49%，但 pp32 prefill 121.5→27.8 回退
  （归因 GTT 缓存一致性税，仍在争论）
- 意义：**吃 FLM 权重格式的非 FLM 引擎已存在**；iGPU-prefill + NPU/GPU-decode
  混合架构可行性的实测证据
- 研究 log：`1bit-MONSTER/1bit-MONSTER` 仓库 `research/ws12-hrx-loom/`

## E. open-xdna（XDNA1，方法论移植）

[Scottcjn/open-xdna](https://github.com/Scottcjn/open-xdna)：Phoenix NPU 全开源
bringup。对本项目的价值：
- 驱动/固件/工具链对坑清单（见 docs/03）
- 手写 AIE 向量内核方法论（AltiVec→AIE 词汇映射）
- **诚实性能观**：XDNA1 上 NPU 密集 GEMM 比 iGPU 慢 6×，NPU 价值在 6.6W 功耗地板
  与剪枝/选择算子——XDNA2 强得多，但"NPU 不是暴力 matmul 机器"的直觉要记住

## F. 论文清单

| 论文 | 贡献 |
|---|---|
| IRON: "Efficiency, Expressivity…Close-to-Metal NPU Programming Interface" [arXiv:2504.18430](https://arxiv.org/abs/2504.18430), FCCM'25 | IRON 设计论文（如何构建此类编程层） |
| "Mapping Gemma3…" [arXiv:2602.06063](https://arxiv.org/abs/2602.06063) | 全模型 on NPU 配方（见 A） |
| GPT-2 训练 on NPU [arXiv:2504.03083](https://arxiv.org/abs/2504.03083), FCCM'25 | bare-metal 工具流做前向+微调 |
| GEMM 跨代优化 [arXiv:2512.13282](https://arxiv.org/abs/2512.13282) | IRON 显式数据流多级 tiling 方法论 |
| MLIR-AIR "From Loop Nests to Silicon" [arXiv:2510.14871](https://arxiv.org/abs/2510.14871) | AIR 编译器 + LLaMA-2 MHA 案例 |
| Dato 任务级编程 [arXiv:2509.06794](https://arxiv.org/abs/2509.06794) | Python 任务模型→AIE，XDNA GEMM+融合注意力 |
| Stream (KU Leuven) IEEE TC'25 | DSE 框架，Strix 上 GEMM+SwiGLU |

## G. 生态趋势判断（amd-oss-knowledge 结论，本项目认同）

- AMD 研究类 OSS 正向 NPU 栈（AIR/IRON/Triton-XDNA）与 CPU 推理（PACE）集中
- 硬件门槛是常态：有意义的工作都需真 NPU + XRT（本机已具备）
- 编译器优先战略：AIR(共享)+AIE(per-tile)+Peano 是 DSL 路（Triton）与结构路
  （IRON）的共同后端——**一个底座两种编程模型**，自建引擎两头都能借力

## H. llama.cpp 对 NPU 的支持现状（2026-09 查证）

**结论：llama.cpp 没有 AMD NPU 后端，也没有任何 PR 接近合并。**

- 上游仅有的 NPU 相关算子实验（synthetic Llama-2 式 layer projections）
  明确自述 "not a llama.cpp backend and does not run a real model yet"
  （[支持请求 issue](https://github.com/ggml-org/llama.cpp/issues)，2025-06）。
- 社区实测一致：llama.cpp / ollama 在 NPU 机器上只跑 CPU
  （Framework/_radxa 论坛多方确认，2025）。
- AMD 官方博客《Accelerating Llama.cpp Performance in Consumer LLM》
  （2024-10）讲的是 **Zen CPU + Radeon iGPU/dGPU** 路线，NPU 不参与。
- 原因（与本项目 R1 发现互证）：GGML 的动态图/动态调度模型与 NPU 的
  **预序列化命令包 + 固件 ERT 实例模型**（docs/08 §1）根本不匹配——
  每层每形状都要独立 xclbin + context，16 上限立刻爆。LLM 上 NPU 的
  正确姿势是编译期把整层/整图固化（FLM/IRON/Gemma3 论文皆如此）。

## I. 其他人的方案盘点（2026-09 检索，按与本机相关性排序）

| 方案 | 形态 | 与本机关系 | 备注 |
|---|---|---|---|
| **FastFlowLM**（AMD 官方，闭源内核层） | 轻量 runtime，256k ctx | ✅ 在用（qwen3:4b 19.8tok/s） | MiniCPM5 42 层不可用（docs/02）；正被接入 ONNX Runtime GenAI 的 inference providers（2026-03 提案） |
| **Ryzen AI Software**（AMD 官方） | ONNX Runtime + Vitis AI EP；1.3+ 起 Linux 支持，官方文档有 "LLM on Linux via NPU" 教程页 | 可试 | ONNX 生态（Transformers→ONNX→AWQ），非 GGUF；路线与本项目正交 |
| **iree-amd-aie**（nod-ai/AMD，开源） | IREE 编译器+运行时 AIE 插件（XRT + llvm-aie/Peano） | ⭐ 值得评估 | torch/stablehlo → AIE 全自动编译；推荐配 Vitis AIE Essentials；FOSDEM 2025 有 AMD 讲座；是 Phase M 之外另一条可能的自动编译轨 |
| **GAIA**（AMD 开源） | 本地 LLM 代理框架（llama.cpp/CPU + 部分 NPU） | 间接 | 不是推理引擎，是应用层 |
| **Riallto**（AMD 开源教学框架） | Python NPU 教学，npu.lib 预置内核 | 教程素材 | 教学价值高（vision 例程讲透 NPU 架构），无 LLM |
| **destevereuz《Getting peak TOPS on Ryzen AI 350》**（2026-05 博客） | 微基准方法论 | 教程素材 | 逐块测出 50 TOPS 数字来源；与 docs/08 互补 |
| **MLPerf Client v1.0（AMD 提交）** | 基准 | 参照 | Ryzen AI Max+ 395（本机同款 APU）Phi-3.5 TTFT <0.7s —— 官方栈的能力上限参照 |
| Gemma3-on-NPU（arXiv 2602.06063） | 论文（Q4NX + FlowQKV/FusedDQP/FlowKV） | ⭐ R2 蓝图 | 见 §A |
| Triton-XDNA / HRX2 lane | 实验 | 观察 | 见 §C/§D |

**对 R2/Phase M 的启示**：iree-amd-aie 是"自动编译整模型"的开源尝试，
若其 context 管理适配 16 上限，可作为 MiniCPM5 的备选轨；但主动权仍在
自己手里的 IRON 改造（大融合图）路线，与 docs/06 既定架构一致。

## J. iree-amd-aie 源码级探查（2026-09-22，clone 未编译）

**形态**：仅源码（PyPI 无 iree-amd-aie/iree-compiler/iree-runtime wheel）；
构建 = IREE 全量 + 插件 cmake，子模块 GB 级（iree、nod-ai/XRT fork、mlir-air、
aie-rt、bootgen、openssl），且 pin 特定 xdna-driver commit 20e1f74（与我们 apt
XRT 2.25 有版本偏差风险）。本机网络下 clone+build ≈ 数小时。

**Context 模型（决定性问题）**：
- XRT 路径 `native_executable.cc:308`：**每个 entry point 一个 exclusive
  hw_context**，无去重无缓存 → LLM（多 dispatch 入口）必撞 16 上限。
  其 e2e 测试都是小程序（matmul/b2b2 等少数入口），从未触及该墙。
- 原生 amdxdna/KMQ 路径：executable 粒度惰建 context，直连 DRM。
- 摊薄手段：`reconf_data_runlist`（同一 xclbin 打补丁服务多入口）——
  有价值的技术，但非通用解。
- Strix Halo 支持面：仓库无 npu5 专属代码；芯片枚举来自 mlir-aie，而
  mlir_aie 的映射（`hostruntime.py:77`）把 npu4/npu5/npu6 全归 "npu2" 桶
  且缓存给 32（**即我们 R1 撞墙的精确出处**）。

**R2 结论：不是捷径**——无 Strix Halo 端到端 LLM 实证、入口粒度 context、
构建重。IRON 改造路线不变（docs/06）。

**但 Phase M 的真正金矿（已 clone 在 ~/qwen/xnpu/iree-amd-aie）**：
`runtime/src/iree-amd-aie/driver/amdxdna/shim/linux/kmq/`（约 2.5k 行：
device/bo/hwctx/hwq/fence/kernel/ert.h）= 生产级开源直连 DRM 的 amdxdna
用户层——xnpu-hal 的移植蓝本（比 irene-xdna-run 更完整：BO 生命周期、
context 创建销毁、hwq 提交、ERT 头全有）。另有编译侧
`AMDAIEControlCodeLowering.cpp`（控制码/指令缓冲生成 = 工件格式参照）。
