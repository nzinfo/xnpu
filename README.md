# xnpu — 自研 AMD XDNA2 NPU 推理引擎

**目标**：绕开 FastFlowLM（FLM）闭源内核层，在 Strix Halo XDNA2 NPU 上直接构造
自己的 LLM 推理机制，首个目标模型 MiniCPM5-2B（42 层 qwen3-arch，FLM 官方无法支持）。

**宿主语言**：Rust（评估见 [docs/05-language-choice.md](docs/05-language-choice.md)，明确禁用 C++）。

## 为什么存在这个项目

FLM 三轮逆向（2026-09 终局结论）：HF 模型包 `julianmb/MiniCPM5-2B-NPU2` 的
`layer.xclbin` 借用自 Qwen3-1.7B 的 28 层图，MiniCPM5 有 42 层；decode 首个 layer
指令（0xd38c 字节）被 NPU 固件永久拒绝派发（ERT state 卡 NEW）。FLM 的图编译器
（私有 `FastFlowLM_IRON` 树）不开源，官方 issue #712 无回应 → **唯一出路是自己
成为"上游"**。全部公开工具链已验证可用（见 docs/01、04）。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01-landscape.md](docs/01-landscape.md) | 生态全景：软件栈、三条 NPU 传输轨、关键仓库 |
| [02-flm-autopsy.md](docs/02-flm-autopsy.md) | FLM 解剖：两层架构、xclbin 结构、三轮逆向根因链与 v8 证据 |
| [03-hardware.md](docs/03-hardware.md) | 本机硬件/驱动/固件实测数据，ERT 观测记录 |
| [04-prior-art.md](docs/04-prior-art.md) | 外部路线调研：Gemma3-on-NPU 论文、IRON、Triton-XDNA、HRX2 等 |
| [05-language-choice.md](docs/05-language-choice.md) | C vs Rust 评估（结论：Rust） |
| [06-architecture.md](docs/06-architecture.md) | 新引擎架构草案：crate 划分、双轨设备策略、里程碑 |
| [07-references.md](docs/07-references.md) | 全部参考链接 |
| [08-npu-programming-notes.md](docs/08-npu-programming-notes.md) | NPU 编程第一手笔记（教程素材）：context 模型/编译流水线/运行时对象/环境配方 |

## 当前状态（2026-09-22）

- [x] 生态调研完成，公开流水线存在性已证实
- [x] FLM 根因定案（换任何借用 xclbin 均不可行）
- [x] 宿主语言选型（Rust，仅阶段 M；阶段 R 用 Python 复用 IRON）
- [x] 里程碑重排为复用优先（docs/06 阶段 R/M）
- [x] **R0 完成（2026-09-22）**：`pytest iron/operators/axpy/test.py` → **160 passed in 75.45s**
      （坑：test.py 必须 pytest 跑；python3-xrt 提供 pyxrt；venv 开 system-site-packages；
      MAP_LOCKED 需 `sudo prlimit --memlock=unlimited`；NPU=xrt-smi 名 RyzenAI-npu4 aie2p 6×8）
- [x] **R1 完成（2026-09-22）**：IRON llama_3.2_1b 跑通首个 NPU LLM 推理
      （2048-tok prefill 3.29s，decode 4.53 tok/s）。**发现固件并发 HW context
      上限=16**（全 AIE 配置要 22 个；关 rope/attn_projection_gemm/final_gemm
      降到 15 过）→ 自研引擎第一设计约束：大融合图复用 context（42 层 × 1 context）
      或惰性重载。补丁：llvm-18 apt 装（objcopy 硬编码）、assign_weights fused
      分支 bug、编译失败必须清 build/（脏缓存假完成）、prompt 须填满 prompt_len
- [x] **R2 完成（2026-09-22）**：MiniCPM5-2B（2.52B 参数，42 层）跑通 IRON 全公开栈
      NPU 推理——CPU attention 基线：2048 prefill 6.29s、decode 2.06 tok/s、输出连贯
      （`configs/minicpm5_2b.json` + tokenizer_hf.py shim + LLAMA_APP_CONFIG 环境变量）
- [ ] **R2b**：d=128 fused MHA 内核构造（上游 op.py 只许 d=64；根因 = B_kv==d
      方形巧合掩盖三处维度错位：PV matmul 编译单元、v/o/P 的 dims_to_stream、
      mha.cc rescale 循环。改动已落：mha.cc/op.py/design.py/gqa.py/test.py，
      测试用例 (2048,128,16,8,0) 编译验证中）
- [ ] **R3**：42 层 decode 正确性对拍（FLM 倒下的位置）
- [ ] M 阶段（Rust）：见 docs/06

## 现役可用方案（本项目期间的实际服务）

- MiniCPM5-2B GGUF @ llama-server 端口 8080，CUDA ~77 tok/s（日常用）
- NPU 上跑 LLM 暂用 FLM 自带 qwen3:4b（36L, 19.8 tok/s）/ qwen3:1.7b（28L, 43 tok/s）
