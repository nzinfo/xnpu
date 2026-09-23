# 02 · FLM 解剖与三轮逆向终局

## 两层架构（官方文档 `src/create_new_model.md` 自述）

1. **引擎** `class <model>_npu : public causal_lm` —— NPU 内核图，PIMPL 隐藏，
   以**预编译** .so 交付（`lib/xrt/lib<model>_npu.so`；HRX 路径为 `lib/hrx/`）。
   "You normally do not write this; it comes from the **kernel/IRON project**."
2. **AutoModel 包装层**（开源部分）—— tokenizer/sampler/chat template/prompt cache，
   经 `prefill()/forward()/checkpoint()/restore()` 驱动引擎。

私有内核树（仓库脚本 `update_qwen35.sh`、`test/*/activate.sh` 泄露）：

```
/scratch/<dev>/FastFlowLM_IRON/            ← 不在 GitHub
├── FLM_Xclbin/                            ← 每模型一个 MLIR 图工程
│   ├── Qwen3_5/qwen3_5_decoding/build/$MODEL_TYPE/xclbins/layer.xclbin
│   │      （CMake MODEL_TYPE 参数化：QWEN3_5_08B/2B/4B/9B…）
│   ├── Qwen3_5/lm_head_npu_bin/、dequant_mm_512x512x512/
│   └── Gemma4_12B_QAT/{decoding,attention_DH_512_prefill,fused_prefill}/
└── FLM_DLL/build/lib/                     ← 引擎 .so（libqwen3_5vl_npu.so、libmha.so…）
```

CI 只做打包，从不编内核。官方 mlir-aie UsedIn.md 确认内核为专有二进制。

## xclbin 解剖（实测）

AXLF 容器 + XCLBIN_MIRROR_DATA JSON，7 个 section：

| Section | Kind | 内容（实测） |
|---|---|---|
| mem_topology | 6 | HOST 64MB + SRAM 48MB @ 0x4000000（每 HW context） |
| aie_partition | 32 | PDI blob（编译后的 AIE 图/DPU 可执行体） |
| embedded metadata | 2 | 0x547 字节 |
| ip_layout | 8 | 1 个 IP：IP_PS_KERNEL/DPU，kernel_id **0x901**，名 **`MLIR_AIE:MLIRAIE`** |
| connectivity | 7 | — |
| group_connectivity | 27 / group_topology | 26 |

FLM 的 4 类 xclbin：
- **attn / mm / dequant**：形状泛化——同形状模型间逐字节相同（md5 证实）
- **embedding / layer**：每架构专属，**层绑定**（编译期含层数信息）

## MiniCPM5-2B 根因链（三轮，2026-09-22 定案）

模型包（`julianmb/MiniCPM5-2B-NPU2`）的 layer.xclbin 借自 Qwen3-1.7B（28 层图）；
MiniCPM5 有 42 层。decode 的 run[1]（首个 layer，instr 0xd38c）ERT state 永卡
**1=NEW**——固件根本不派发。对照 qwen3:0.6b/4b 全程 1→4 正常。

**关键机理推断**：指令流按引擎的 MAX_L=42 生成了逐层内容（0xd38c ≫ 1.7B 的
0x818c，4B 36 层为 0x13a0c——尺寸随形状+层数缩放），而 28 层图的解析表无法解析
42 层指令 → 固件拒绝。**layer 图是层敏感的，编译期常量=层数。**

已排除因素（全部实验证伪）：

| 因素 | 证伪方式 |
|---|---|
| runlist 长度（作者"42 连发超缓冲深度"论） | v7 fire-all 与单 run 顺序重放**都挂** |
| 引擎版本 | 0.9.36 / 1.0.4 / 1.0.6 同挂 |
| xclbin 构建版本 | 换 1.0.6 原生 1.7B layer.xclbin 也挂 |
| 指令大小 | 4b 的 0x13a0c 更大却正常 |
| AIE 几何 | 头部相同 |

结论：**图↔引擎指令 ABI 不匹配，唯上游（或我们自己）重编 42 层图可解。**
作者 julianmb 本人开的 issue #712（2026-09-08）至今零官方回应。

## 模型目录佐证

FLM 全目录 42 个模型：qwen3 系上限 **36 层**（8B）；42 层的只有 Gemma4-E4B
（gemma4e 引擎，另一套 attn 布局）；Qwen3.5 是 Q4_K+GateDeltaNet 混合架构。
→ **不存在可借用的 42 层 qwen3-arch 图**。

## 逆向工具与证据位置

- `~/qwen/flmfix/hook.cpp`（v8）+ `libflmfix.so`：LD_PRELOAD 拦截 XRT C++ 符号；
  mode 0=v6 单 start/wait，1=legacy 分块，2=v7 fire-all，3=v7c fire-分块，
  4=v8 poll-diag（顺序 start + state() 轮询，dump 16×u32 ERT 字，20s 卡死超时）
- 关键日志：`flm-server-fix.log`、`flm-qwen3-v8.log`、`flm-qwen3-4b-v8.log`、
  `flm-swap-test.log`、`flm-0936-hook.log`（均在 ~/qwen）
- ERT 包格式：ert_start_kernel_cmd 头 + cu_mask + ert_npu_data{instruction_buffer,
  size, prop_count} + args
- 指令尺寸实测：embedding 0x268；MiniCPM5 layer 0xd38c；qwen3-0.6b layer 0x818c；
  qwen3-4b layer 0x13a0c；prefill mm 0x2390/0x11d0/0x3550
- 作者原 layer.xclbin 备份：`~/qwen/flmfix/layer.xclbin.author.bak`
  （md5 d69ed4dd78e10cedfde4bd5b750b9cb1）；flm 0.9.36 备份 `~/qwen/flm0936`
