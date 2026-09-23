# 01 · 生态全景

## 软件栈（自上而下）

```
Triton 内核 / IRON Python API        ← 两种用户编程模型（写内核/图）
        │
   MLIR-AIR                           ← 空间调度（tiling/placement/缓冲/同步，AIR 方言）
        │  air-to-aie + aircc
   MLIR-AIE  ◄── IRON 算子库建在这     ← 共同底座（IRON 完全绕过 AIR，直连 AIE）
        │
   LLVM-AIE (Peano)                   ← AIE ISA 代码生成（LLVM fork）
```

关键事实（amd-oss-knowledge 14 仓库考证）：
- **MLIR-AIE 是共享底座，MLIR-AIR 不是**；IRON 零 `air.*` 导入。
- FLM 的 xclbin 里 ip_layout 内核名 `MLIR_AIE:MLIRAIE` —— FLM 图由同一编译器产出。

## 三条 NPU 传输轨（Linux）

| 轨 | 派发机制 | 状态 | 谁在用 |
|---|---|---|---|
| **XRT/ERT**（旧） | xclbin 中心；命令经 NPU 固件里的 ERT 调度器派发 | 生产 | FLM 现行路径；本项目逆向主战场（卡死处） |
| **HSA/AQL**（实验） | 用户态 AQL 包队列 + doorbell；HSACO 代码对象 | 实验 | mlir-aie 系 Triton-XDNA 的 "HSA dispatch path" |
| **libamdf / 原生 DRM**（新） | 直连 `/dev/accel` DRM ioctl；提交 Loom 预编 `.xdna` 静态命令包；**无 XRT、无 HSA、无 ROCr** | experimental/ | ROCm/hrx-system；FLM pin 的 `jtuyls/hrx` amdxdna fork（1.0.6 的 `lib/hrx/` 双路径） |

HRX"通吃 GPU/NPU/CPU"的真相：**上层统一（libhrx C ABI + HIP 兼容 libamdhip64.so +
IREE HAL + Loom 工件体系），下层分叉**——GPU 走 HSA/AQL 动态调度，NPU 走 libamdf/
DRM 提交静态命令包（调度在编译期固化），CPU 走本地码。JIT（Loom loomc，AOT+JIT 双模，
1-15ms 特化）只管"一份源码按目标+形状出工件"，不管调度。

## 关键仓库

| 仓库 | 许可 | 角色 |
|---|---|---|
| [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) | Apache-2.0 (LLVM-ex) | 编译器本体；1.2 加 IRON host runtime 抽象层 + Strix BF16 matmul；1.3 加 aiecc C++ 驱动 |
| [amd/IRON](https://github.com/amd/iron) | Apache-2.0 | Python 结构化图编程；28 算子 + aie2p 内核 + `applications/llama_3.2_1b` 完整示例；CI 覆盖 Phoenix/Krackan |
| [amd/Triton-XDNA](https://github.com/amd/Triton-XDNA) | MIT | `@triton.jit` → TTIR → triton-shared → AIR → aircc → xclbin/elf/pdi；XRT+HSA 双派发；matmul 达手写 ≥90% |
| [Xilinx/mlir-air](https://github.com/Xilinx/mlir-air) | MIT | 空间编译器；LLaMA-2 MHA 案例 (arXiv:2510.14871) |
| [Xilinx/llvm-aie (Peano)](https://github.com/Xilinx/llvm-aie) | Apache-2.0 (LLVM-ex) | AIE 目标的 LLVM 后端 |
| [ROCm/hrx-system](https://github.com/ROCm/hrx-system) | early-access | 另一种 HIP：libhrx C ABI + HIP 兼容层 + IREE 内嵌（目标含 gfx1151=本机 iGPU）+ Loom + libamdf + `experimental/xdna` |
| [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM) | MIT(编排) | 生产级 NPU LLM 运行时；**内核层闭源**（官方 UsedIn.md 自述 "kernels are distributed as proprietary binaries"） |
| [Scottcjn/open-xdna](https://github.com/Scottcjn/open-xdna) | AGPLv3 | XDNA1 全开源 bringup 配方（驱动/固件/工具链四大坑 + FAQ）；XDNA1 专属但方法论通用 |
| [1bit-MONSTER/amd-oss-knowledge](https://github.com/1bit-MONSTER/amd-oss-knowledge) | — | AMD 开源 NPU 链知识库（14 仓库考证 + 栈图） |
| [Xilinx/aie-rt](https://github.com/Xilinx/aie-rt) | — | AIE 运行时驱动 + FAL |

## 命名警告（易踩坑）

NPU 世代命名有两套并存：
- IRON 旧称：NPU1=Phoenix(AIE2)、NPU2=Strix Halo(AIE2P/XDNA2)
- HRX 新称：NPU4=Strix(`17f0:10`)、**NPU5=Strix Halo(`17f0:11` ← 本机)**

且 HRX 明文：**NPU4/NPU5 镜像不可互换**（"shared NPU2 array architecture does not
make the images interchangeable"，ELF 须匹配设备精确编译档案）。XDNA 世界没有"差不多
兼容"——这与 FLM layer.xclbin 28↔42 层不匹配的教训同源。
