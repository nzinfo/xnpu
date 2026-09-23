# 03 · 本机硬件与驱动实测

## SoC：AMD Ryzen AI Max+ 395（Strix Halo）

| 部件 | 规格 |
|---|---|
| CPU | 16× Zen5（32 线程） |
| iGPU | Radeon 8060S，40 CU `gfx1151`（HRX 构建目标列表里点名） |
| **NPU** | XDNA2 / AIE2P，48 tiles，8 列，50 TOPS；PCI `1022:17f0`，HRX 命名 **NPU5 = Strix Halo (`17f0:11`)** |
| 内存 | 128GB LPDDR5X-8000 统一内存 ~273GB/s |

## 驱动/固件栈

- NPU 固件：`amdnpu/17f0_11/npu.sbin` v**1.1.2.65**
- 内核：Linux 7.0.0-31-generic，in-tree `amdxdna` **0.7.0**（/dev/accel/accel0）
- 启动参数：`iommu=pt iommu.passthrough=0`
- 注意：FLM 需 `prlimit --memlock=unlimited`（xclbin/BO pin 内存）
- open-xdna 记录的 XDNA1 通用坑（对本机有参照价值）：需要 `libxrt_driver_xdna.so`
  而非仅 `libvxdna.so`；staging `amdxdna.ko` 与固件版本必须匹配，否则
  `ERT_CMD_STATE_ABORT`；`llvm-objcopy` 需在 PATH（GNU objcopy 解不了 AIE2 ELF）

## 每 HW context 内存（xclbin mem_topology 实测）

- HOST 桥：64MB
- SRAM（AIE 阵列本地）：48MB @ 0x4000000

## ERT 观测数据（v8 hook，~/qwen 日志）

- ERT 命令状态字：**1=NEW → 4=COMPLETE**（正常路径）；卡死即永驻 1
- 模型加载期：43 runs ×3 轮全正常；prefill 5-run（mm 0x2390/0x11d0/0x3550）全正常
- decode 42 层 runlist：run[0] embedding（0x268）1→4 正常；
  **run[1] layer（0xd38c，buffer @0x41e0000）永卡 NEW，20s 超时**
- 对照组：qwen3:0.6b / 4b 全部 run 1→4 正常
- 破坏性退出：`xrt::run destructed while command is still in progress`

## 驱动接口备查（Track B 依赖，待 M0 验证）

in-tree amdxdna 是 DRM 驱动，ioctl 含创建 HW context 并**直接传入 xclbin blob**
（XRT 之下即走此路）→ 理论上不经 XRT 也能加载图。M0 spike 中用 Rust 复刻
`irene-xdna-run`（hrx-system `experimental/xdna/`，纯 C 参考实现）验证。

## .xdna 执行模型（HRX/libamdf 路线，来自 experimental/xdna/README）

镜像 = tile 程序 + DMA 目录 + 完整 "establishing command" + 绑定记录；
加载器只拷显式 range、patch 声明的地址字段（"native command bytes 是可执行代码，
对加载器不透明"）；每次提交 = reset/配置 → 加载 tile 程序 → IO DMA → 等 retirement；
上下文时间片制，不保证 tile 状态跨提交存活。libamdf 只管 device admission /
scoped memory / range submission，走原生 DRM（Linux）或 MCDM（Windows）。
