# 07 · 参考链接

## 工具链与运行时

- MLIR-AIE（编译器）：https://github.com/Xilinx/mlir-aie · docs: https://xilinx.github.io/mlir-aie
  - UsedIn.md（生态总索引）：https://github.com/Xilinx/mlir-aie/blob/2e07278fd48dc6916f4a97754c785e1e172ed012/docs/UsedIn.md
- IRON（Python 图编程 + 算子库 + Llama-3.2-1B 应用）：https://github.com/amd/iron
- Peano / llvm-aie（AIE LLVM 后端）：https://github.com/Xilinx/llvm-aie
- MLIR-AIR（空间编译器）：https://github.com/Xilinx/mlir-air · https://xilinx.github.io/mlir-air/dev/
- Triton-XDNA（Triton→NPU）：https://github.com/amd/Triton-XDNA
- HRX System（另一种 HIP；含 Loom、libamdf、experimental/xdna）：
  https://github.com/ROCm/hrx-system
  - NPU 执行模型：`experimental/xdna/README.md`
  - Loom 编译器：`loom/README.md`（loomc C API）
- aie-rt：https://github.com/Xilinx/aie-rt
- XRT（旧轨，对照用）：https://github.com/Xilinx/XRT

## FLM 相关

- FastFlowLM：https://github.com/ROCm/FastFlowLM （内核闭源；`src/create_new_model.md`
  讲清两层架构）
- Issue #712（MiniCPM5 支持请求，julianmb 本人开，无回应）：
  https://github.com/ROCm/FastFlowLM/issues/712
- MiniCPM5-2B-NPU2 模型包：https://huggingface.co/julianmb/MiniCPM5-2B-NPU2
- 作者移植仓库（根因自述 + KV 头展开脚本）：https://github.com/julianmb/minicpm5-xdna2
- Lemonade（OpenAI 兼容本地服务，FLM 为其 NPU 后端）：https://github.com/lemonade-sdk/lemonade

## 社区知识库

- open-xdna（XDNA1 全开源 bringup 配方）：https://github.com/Scottcjn/open-xdna
- amd-oss-knowledge（14 仓库考证 + 栈图 + HRX2 lane 摘要）：
  https://github.com/1bit-MONSTER/amd-oss-knowledge
- 1bit-MONSTER 主仓库（HRX2 llama.cpp 研究 log：research/ws12-hrx-loom/）：
  https://github.com/1bit-MONSTER/1bit-MONSTER

## 论文

- IRON 设计（FCCM'25）：arXiv:2504.18430 — https://arxiv.org/abs/2504.18430
- **Gemma3 on NPU 配方（本项目蓝图）**：arXiv:2602.06063 — https://arxiv.org/abs/2602.06063
- GPT-2 训练 on NPU（FCCM'25）：arXiv:2504.03083 — https://arxiv.org/abs/2504.03083
- GEMM 跨 Ryzen AI 代优化：arXiv:2512.13282 — https://arxiv.org/abs/2512.13282
- MLIR-AIR + LLaMA-2 MHA：arXiv:2510.14871 — https://arxiv.org/abs/2510.14871
- Dato 任务级编程：arXiv:2509.06794 — https://arxiv.org/abs/2509.06794
- Stream DSE（KU Leuven）：https://kuleuven-micas.github.io/stream/

## 本机相关日志与工具（项目证据）

- 逆向 hook：`~/qwen/flmfix/hook.cpp`（v8）+ `libflmfix.so`
- v8 诊断日志：`~/qwen/flm-swap-test.log`、`flm-qwen3-v8.log`、`flm-qwen3-4b-v8.log`
- FLM 版本备份：0.9.36 @ `~/qwen/flm0936`；现行 1.0.6 @ /opt/fastflowlm
- GGUF 日常服务：llama-server 端口 8080（CUDA ~77 tok/s）
