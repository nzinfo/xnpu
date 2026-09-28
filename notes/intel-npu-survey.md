# Intel NPU(MTL/LNL/ARL/PTL)开源资料调研 —— 可移植到 XDNA2 引擎的优化技术

日期:2026-09-28。方法:直接读 intel/linux-npu-driver 固件头文件(最权威的开源 HW 契约)、OpenVINO intel_npu 插件文档、公开架构分析(Chips and Cheese)、逆向项目(npunlock)。每条标注来源与可信度:【driver】= 驱动/固件头文件代码(硬事实);【OV】= OpenVINO 官方文档(官方但营销层);【媒体】= 第三方分析;【RE】= 逆向工程。

---

## 1. 各代硬件概览(带来源)

### NPU3 = Meteor Lake "NPU 3720"(PCI 0x7D1D)
- **2 个 tile**(每 tile = 1 DPU + CMX slice):`VPU_MAX_TILES = 2`、`VPU_DPU_PER_TILE = 1`【driver, firmware/include/api/vpu_nce_hw_37xx.h】
- **每 tile CMX(SRAM)= 2MB**,基址 0x2E000000 / 0x2E200000,软件全管理(无 tag/无虚实转换):`VPU_CMX_SLICE_LENGTH = 2*1024*1024`【driver, vpu_nce_hw_37xx.h】;Chips and Cheese 证实 "2 MB software-managed SRAM per NCE, not caches"【媒体, https://chipsandcheese.com/2023/12/22/intel-meteor-lakes-npu/】
- **DPU 吞吐**:2 tile 合计 4096 INT8 MAC/cycle,1.16GHz 下 ≈9.5 TOPS(FP16 半速);每 NCE 512 个 MPE(每 MPE 4 INT8 MAC/cycle)【媒体, 同上】
- **可编程核**:
  - ActShave(blob 可编程):`VPU_AS_PER_TILE = 2` → 4 个;SHAVE kernel 有独立 stack(CMX 内 14KB/tile 平分)【driver, vpu_nce_hw_37xx.h + vpu_cmx_info_37xx.h】
  - SNN SHAVE(SHAVE-NN)1/tile,仅 37xx 存在,40xx 删除(`No SHAVE-NN for VPU40XX`)【driver, vpu_cmx_info_40xx.h 注释】
  - 另有 firmware 私用的 utility SHAVE + 2 个 LEON MCU(LeonRT 收 host 命令、LeonNN 做低级硬件任务调度,各自带 cache)【媒体, Chips and Cheese】(数量官方未公开)
- **DMA**:2 个引擎(`VPU_MAX_DMA_ENGINES = 2`);DMA 引擎分 from-DDR / from-CMX 两个方向接口(sub_unit 0/1)【driver, vpu_nnrt_wlm.h 的 VpuWorkItem 表】。实测 NPU 共享 LPDDR5 带宽 <10 GB/s(iGPU >19GB/s),大矩阵(7168²)严重掉速(疑 TLB/DMA 延迟)【媒体, Chips and Cheese】
- **DPU 数据通路寄存器级结构**(IDU→MPE→PPE→ODU):IDU 输入分布(sparsity/se 表)、MPE 4x4/16x1 grid、PPE 做 scale/bias/prelu/clamp/LUT 化后处理、ODU 带 `swizzle_key`/`permutation`/dtype 转换写回;输入 dtype 含 I4/U4/FP8,输出 FP16/I8/I32 等【driver, vpu_nce_hw_37xx.h】
- **DPU workload 描述 = invariant(260B)+ variant(44B)两级**【driver, 同上】

### NPU4 = Lunar Lake "NPU 4000"(0x643E,48 TOPS【OV/营销, hothardware.com/review/intel-lunar-lake-deep-dive】)/ Arrow Lake(0xAD1D,驱动按 40xx 处理,插件表格仍标 3720)【OV, openvino/src/plugins/intel_npu/README.md】
- **最多 6 tile**:`VPU_MAX_TILES = 6`(LNL 实为 5/6 tile,由 OpenVINO 表格佐证:4000 "2 out of 5/6")【driver vpu_nnrt_common.h + OV README】
- **每 tile CMX 仍 2MB**(由 40xx 布局反推:workspace 1440KB + reserved 512KB + metadata 81KB + actshv 14KB+1KB ≈ 2MB)【driver, vpu_cmx_info_40xx.h】
- **DMA 变为 1 个物理引擎、逻辑拆 2 接口**(注释原文:"On NPU4, there is only one physical DMA engine, but it is logically split into two interfaces")【driver, vpu_nnrt_common.h】
- **variant 寄存器扩到 192B,内含 128 位 producer/consumer barrier 掩码**(cbarrier_lo/hi, pbarrier_lo/hi)和 6 个 halo region(多 tile 切分边界交换)、`ppe_lut_ptr`(PPE 内 LUT)、`invar_ptr`+`var_tag`(variant 经 LUT 指向 invariant)、`next_sram_job_addr`(SRAM 描述符链)【driver, vpu_nce_hw_40xx.h】
- 新增 HF8 输入、`pallet[8]`(权重调色板)、`small_hw_opt_en` 等【driver, 同上】
- **ActShave 2/tile → 6 tile 共 12 个**;SNN 删除【driver】

### NPU5 = Panther Lake "NPU 5010"(0xB03E)/ Wildcat Lake "NPU 5020"(0xFD3E)
- OpenVINO 表:**5010 = 3 tile(LATENCY 用满 3 个),5020 = 1 tile**;THROUGHPUT 最优 request 数 = 8,LATENCY = 1【OV, intel_npu/README.md】
- 驱动已支持(linux-npu-driver release notes 列 MTL/ARL/LNL,PTL 在新 release)【driver, docs/overview.md】;内部微架构公开资料极少(TOPS 等营销数字本调研未验证)

### 与 XDNA2 的硬件对照
| | Intel NPU3/4 | AMD XDNA2(我们) |
|---|---|---|
| 可编程核 | 2 ActShave/tile(6 tile) | 8 AIE2P,VLIW+matrix |
| 专用 MAC 阵列 | 每 tile 2048 INT8 MAC(40xx 更强) | AIE2P matrix unit |
| 片上 SRAM | 2MB/tile 全软件管理 | 512KB L1 总量 + 16KB PM/core |
| DMA | 2 接口(37xx 2 物理引擎;40xx 1 物理 2 逻辑) | shim DMA,16 BD 描述符 |
| 同步 | HW barrier:FIFO 寄存器,prod/cons 计数,16~32/组,64 位掩码 | HW 同步核/事件 |
| 描述符槽位(CMX 内) | DMA task 256(37xx)/80(40xx),invariant 32/64,variant 256/128,kernel range 32/64,invocation 64/64 | BD 16/FIFO depth 2 |

---

## 2. 他们的调度/同步模型(图 → 设备工作)

全链路(【driver docs/overview.md】):OpenVINO 插件 → Level-Zero graph API → UMD(libze_intel_npu.so)+ Compiler-in-Driver(libnpu_driver_compiler.so;2026.1 起默认 Compiler-in-Plugin,【OV, intel_npu/README.md】)→ KMD(intel_vpu.ko,/dev/accel/accel0)→ 固件(闭源二进制,但头文件开源)。

**编译产物 = ELF "inference blob"**(vpux_elf 子模块解析;编译器即 openvinotoolkit/npu_compiler,MLIR 实现,目录名 vpux_compiler/vpux_driver_compiler —— Movidius VPUX 血统;注意 npu_compiler 仓库是公开的 Apache-2.0,含 `sw_runtime_kernels`(SHAVE C 内核)与 `artifacts/precomputed_strategy_cache`)【https://github.com/openvinotoolkit/npu_compiler】。逆向项目 npunlock 确认 blob 内含 ACT-SHAVE ELF 段、MMIO preactions(推理前 MMIO 写序列)等【RE, https://github.com/hsfzxjy/npunlock】。

### 2.1 运行时数据结构(VpuManagedMappedInference,【driver, vpu_nnrt_wlm.h】)
文件头注释直接说明设计:"任务(DPU/Shave/DMA)向 FIFO 的入队由 **management task(往 FIFO 写数据的 DMA 任务)** 完成,这些任务是编译器产出的 Managed Mapped Inference DAG 的一部分;把任务描述符从 DDR 喂到 CMX 的 DMA 称为 **workload propagation task**"。即:**DMA 引擎自己给各引擎的硬件 FIFO 喂描述符 —— 设备侧自驱动的图播放器**。分两档:partial WLM(固件仍负责入队,只有描述符搬运 DMA 化)和 full WLM(完全 DMA 自调度)。

- **VpuWorkItem(64B)**:`{wi_desc_ptr, type(DPU/DMA/SHV/MEDIA/DPU_AUTO), unit(tile 或 DMA 引擎号), sub_unit(DMA: 0=from DDR,1=from CMX;SHV: tile 内 shave id)}`。即编译器显式选择"哪个引擎的哪个 FIFO"。
- **VpuTaskBarrierMap**:`{producer_count, consumer_count, real_id, work_item_idx, enqueue_count}` —— barrier 解除后应入队哪些 work item 的静态表。
- **bootstrap_workitems_count**:初始 work item(通常是往 CMX 喂描述符的 DMA),之后靠 barrier 级联推进。

### 2.2 Barrier 模型(【driver, vpu_dma_hw_37xx.h + vpu_nnrt_wlm.h】)
- 硬件 barrier = 有 producer/consumer 计数的同步单元;每个 DMA 描述符/DPU variant 带 **64 位 prod_mask/cons_mask**(37xx DMA 描述符内两组 mask;40xx DPU variant 内 cbarrier/pbarrier 128 位)。
- **虚拟 barrier → 物理 barrier 映射**:编译器用无限虚拟 barrier,映射到少量物理 barrier(37xx:32/组;40xx:16/组),每个物理 barrier 有重编程次数 `num_of_barrier_reprogrammings` —— **barrier 当寄存器分配**。
- **barrier 编程本身可以 DMA 化**:`VpuBarrierProgrammingMode` 枚举 LEGACY(运行时编程)→ NO_BARRIER_DMAS_SCHEDULED → INITIAL_BARRIER_DMAS_SCHEDULED(编译器只排初始)→ ALL_BARRIER_DMAS_SCHEDULED(_4K)(**编译器把所有 barrier FIFO 的 top-up 都排进 DMA 时间表,运行时零参与**)。`barriers_configuration` 数组的内存布局就是"可直接 DMA 进 barrier FIFO 寄存器"的格式。

### 2.3 DMA 描述符(【driver, vpu_dma_hw_37xx.h】)
80 字节、64B 对齐(= L2 cache line),字段:
- `link_address`(40b):**链表**指向下一描述符(非环形!);
- `cfg_bits`:type(1D/2D)、**burst_length**(8b)、**critical**(优先)、interrupt_en/trigger、**skip_nr**(运行时可跳过)、**order_forced**(强制按序派发)、watermark_en、dec_en、barrier_en;
- 2D:src/dst width/stride + plane stride(多平面);
- 1D 时复用为 barrier prod/cons mask。
- 链表约束(头文件原文):**链上只有第一个 DMA 可消费 barrier,只有最后一个可生产 barrier**。

### 2.4 DPU 任务的 invariant/variant 两级描述(【driver, vpu_nce_hw_37xx.h/40xx.h】)
- `VpuDPUInvariantRegisters`(37xx 260B / 40xx 288B):tensor 布局、kernel 形状、dtype、PPE/ODU 配置 —— 每层一次;
- `VpuDPUVariantRegisters`(37xx 44B / 40xx 192B):workload 尺寸/起点、weight_size/num、barrier mask —— 每 workload 一个,经 `invar_ptr+var_tag` LUT 关联 invariant;
- **CMX 里的槽位法(硬件定律)**(【driver, vpu_cmx_info_37xx.h/40xx.h】):每 tile CMX 中固定划出 metadata 区存放运行期描述符:37xx = DMA task 256 + invariant 32 + variant 256 + act kernel range 32 + invocation 64;40xx = DMA 80 + invariant 64 + variant 128 + range 64 + invocation 64。**hardware "feeder" FIFO**:4 个 component feeder(variant/invariant/range/invocation)+ 2 DMA = 6 个 metadata feeder 从 CMX 槽位向引擎供描述符;
- 37xx 的 `VpuMetadataMapDual0/Dual1`:tile0 存 dma0+invariant+range+desc、tile1 存 dma1+variant+invocation —— **元数据跨 tile 交错放置,让两个 DMA 引擎各自从本地 CMX 取描述符**;
- 布局对齐律(注释原文):workspace 起始必须 16K(37xx)/32K(40xx)对齐,"否则某些 DPU 操作(swizzling)不工作"。

### 2.5 Host 侧执行路径(【driver, umd/vpu_driver/source/command/inference_execute.cpp + docs/overview.md】)
- L0 command list 里放 `VPU_CMD_INFERENCE_EXECUTE`,指向 host-mapped inference(驱动解析 ELF 为 HostParsedInference,`applyInputOutputs` 把用户 IO 指针 patch 进 blob);
- **zeMutableCommandList**:免重录改参(v1.6.0+,overview.md changelog)——图级"免重发"机制;
- shared scratch buffer 每图一个,按需替换 handle;
- 支持 in-order 执行、优先级、driver 侧 blob 缓存(~/.cache/ze_intel_npu_cache,LRU 1GB)。

### 2.6 与 AMD XDNA2 BD/FIFO 模型对比
| 维度 | Intel NPU | XDNA2(现状) |
|---|---|---|
| 图→设备工作 | 编译期完全静态:DAG → work items + barrier 重编程表 + DMA 自喂 | host 提交 ctrl kernel 指令流,~65 exec/token,正在做大融合 |
| 描述符来源 | DDR → DMA 预取到 CMX 槽位 → feeder FIFO | host/shim 直接写 16 BD 链 |
| 描述符形态 | 80B 链表 + invariant/variant 分级 | BD 阵列 + object FIFO(depth 2) |
| 同步 | 64 位掩码 barrier + 计数 + 编译期回收复用 | 同步核/事件,每 exec 边界同步 |
| 运行时动态性 | skip_nr(跳描述符)、mutable cmdlist 改参 | 重发指令流 |
| 多核数据流 | tile 间 halo 区 + barrier DAG | 正在做多核元素化 |

核心差异:Intel 把"执行粒度"做进 **编译期静态 DMA 时间表 + 设备侧自播放**,host 只发一次 inference_execute;我们目前 host/命令处理器深度参与每次 exec。

---

## 3. 具体可移植优化技术清单

格式:**技术 | 为什么有效 | 代价 | 来源**。

1. **DMA 自调度(management DMA 往引擎 FIFO 写描述符)** —— 把"下一个 exec 的启动"变成数据流的一部分,消除 host/固件逐 exec 参与;NPU 用它把固件从推理热路径剔除(WLM 演进)。代价:需要静态可分析的图 + 额外 DMA 描述符。【driver, vpu_nnrt_wlm.h 文件头注释】
   - XDNA2 对应:用 BD 链预先写好"下一层 BD + 同步事件",让 shim 序列器/同步核级联推进;至少把 65 exec/token 变成"每 N 层一次 host 干预"。
2. **描述符分级:invariant(层静态)/ variant(workload 动态)+ LUT 关联** —— 每 tile 每层只搬 44~192B 的 variant,而非整包配置;描述符带宽降 5-10 倍,且同一 invariant 可服务整个 tile 循环。代价:两级管理复杂度 + 32/64 个 invariant 槽上限。【driver, vpu_nce_hw_37xx.h/40xx.h + vpu_cmx_info_*.h】
   - XDNA2 对应:16KB PM 里的指令流拆"层静态骨架(常驻)+ 每 tile 小参数块(DMA 进 L1)",W4 GEMV 的 K-loop 控制流常驻,变的是地址/长度。
3. **描述符预取环(workload propagation)+ 槽位双缓冲律** —— DDR 描述符提前 DMA 进 CMX 固定槽(37xx:256 DMA 槽/32 invariant/256 variant),硬件 feeder 从 CMX 消费;槽位数=可并行的描述符深度=流水线深度上限。代价:CMX 预算(37xx ~63KB/tile metadata)。【driver, vpu_cmx_info_37xx.h/40xx.h】
   - XDNA2 对应:512KB L1 里给"BD 影子区"做双缓冲:当前链在跑,下一层 BD 链已在写;16 BD 上限用"链尾跳到预取好的下一链"绕开。
4. **barrier 当寄存器分配(虚拟→物理映射 + 重编程计数)** —— 任意复杂 DAG 在 16~32 个物理 barrier 上跑;编译期算好每个物理 barrier 的复用次数与 top-up 时机。代价:编译器复杂度。【driver, vpu_nnrt_wlm.h(BarrierReferenceMap/num_of_barrier_reprogrammings)】
5. **barrier 编程 DMA 化(barriers_configuration 直接 DMA 进 FIFO 寄存器)** —— 同步本身的成本也进数据流;ALL_BARRIER_DMAS_SCHEDULED 模式下运行时零 barrier 编程。代价:静态调度约束(需知全部 producer/consumer 计数)。【driver, vpu_nnrt_wlm.h(VpuBarrierProgrammingMode)】
6. **DMA 描述符链表化 + 链端 barrier 语义**(链中只有首消费/尾生产 barrier)+ **order_forced/critical/skip_nr/watermark 控制位** —— 长 DMA 序列共享一次同步;skip_nr 是"静态调度里挖一个动态逃生口"(运行时跳过描述符而不重排);watermark 提供进度观测。代价:链表指针追迹延迟(需预取)。【driver, vpu_dma_hw_37xx.h】
7. **多 tile 切分 + halo 区(40xx 每 workload 6 个 halo region)+ z 方向 split(num_ses_in_z_dir)** —— 跨核张量切分的边界交换被形式化为描述符字段,编译器可自由选切法。代价:边界重复计算/搬运。【driver, vpu_nce_hw_40xx.h(halo_region_t)】
   - XDNA2 对应:8 核 GEMV 的 K-split 归约树/列切分,把边界协议固化进 object FIFO 拓扑而不是每层手写。
8. **写回路径做布局变换(ODU swizzle_key/permutation/ nthw_ntk 8_8/4_16/16_4)+ 2D/多平面 strided DMA** —— transpose/swizzle 零 kernel 成本,在写回或取数时完成。代价:消费端必须匹配 layout 约定。【driver, vpu_nce_hw_37xx.h/40xx.h + vpu_dma_hw_37xx.h】
   - XDNA2 对应:flowkv strided 主导(118ms 口径)——在 shim 写回侧定死 swizzle,让 attention 读侧变成顺序。
9. **MAC 后处理融进 ODU/PPE(scale/bias/prelu/clamp/LUT/dtype 转换;40xx 加 ppe_lut_ptr)** —— 逐元素 epilogue 不产生独立任务。代价:PPE 功能固定,复杂激活仍要 SHAVE。【driver, vpu_nce_hw_37xx.h/40xx.h】
   - XDNA2 对应:rms-norm/swiglu 元素化进 GEMV 消费核(与我们融合对方向一致,Intel 做到了写回级)。
10. **tile 数按性能模式静态选择:LATENCY 用更多 tile(4000:4/6),THROUGHPUT 用少 tile(2)+ 8 个 outstanding request** —— 同一硬件两种拓扑映射,编译期决定。代价:两套调度。【OV, intel_npu/README.md 表格】
    - XDNA2 对应:decode(单 token)= 8 核全并在一层(latency 映射);多流/预填=核分组流水(throughput 映射)。
11. **元数据跨 tile 交错放置(Dual0/Dual1)让多 DMA 各取本地 CMX** —— 消除描述符取数的跨片互连。代价:布局约束。【driver, vpu_cmx_info_37xx.h】
    - XDNA2 对应:BD/参数块放离 shim/消费核最近的 L1 侧。
12. **权重调色板 + 稀疏表:pallet[8]、SE/SP 表、wt_swizzle、I4/U4 原生** —— W4 之外再用每 group 调色板/结构化稀疏省带宽;dec_en(DMA 内建解码)。代价:编译期统计 + 精度风险。【driver, vpu_nce_hw_40xx.h】
13. **mutable command list / 免重录改参 + defer_weights_load + blob 缓存(FEIL/FIL 指标)** —— 参数换绑不重编译不重录;权重延迟到首次推理才载入。代价:驱动复杂度。【driver docs/overview.md changelog + OV NPU device 文档】
14. **每 workload 硬件 profiling:DMA HW profiling log(logaddr_dma_hwp)、HWP stat mode(dense/sparse act/wt、IDU/ODU time stat)、unique_id** —— 帧内每 workload 的算子利用率和时间统计,直接定位 bubble。代价:CMX/带宽占用。【driver, vpu_nce_hw_37xx.h(VpuHWPStatMode)/40xx hwp_ctrl】
    - XDNA2 对应:给 peano 帧成本定律加"每 BD/workload watermark"计数(第 6 条的 watermark 语义)。
15. **SHAVE 内核=普通 C 代码 + MoviTools 编译(npunlock 证实可脱离 OpenVINO 写自定义内核)** —— 证明"可编程核上写任意 C kernel 并挂进图"是可行路线;carrier op(寄生于现有 Abs 等算子)注入法。代价:无官方支持、需挖 toolchain。【RE, github.com/hsfzxjy/npunlock + reddit.com/r/ReverseEngineering/comments/1wn4ps7】
16. **单物理 DMA 拆双逻辑接口(from-DDR / from-CMX)**(40xx)—— 长权重搬运与片上小拷贝分通道,避免队头阻塞。代价:仲裁。【driver, vpu_nnrt_common.h】

---

## 4. 什么不开放(别追幽灵)

- **固件本体**(intel-fw-npu 二进制):LeonRT/LeonNN 的调度实现、任务播放循环 —— 只有头文件契约。【driver, docs/overview.md(固件以二进制包分发)】
- **SHAVE 工具链不分发**:MoviTools/MVC_DEPEND 需从 OEM 驱动包里抠(npunlock 的做法,合法性灰色);mlibm.a 只在驱动 payload 里。【RE, npunlock README】
- **DPU 微架构细节**:MPE 阵列内部、STT/SIF 互连、barrier FIFO 深度(重编程间隔的下限!)未公开 —— 我们只知每组 barrier 数(16/32)和掩码宽度(64b),**不知道 FIFO 能缓存几次重编程**。
- **NPU compiler 的代价模型/调度器内部**:npu_compiler 仓库公开(Apache-2.0, MLIR),但本次未能深挖其调度 pass 文档(GitHub 抓取故障;zread 未索引该仓库)——**值得后续单独 mine**,尤其 `artifacts/precomputed_strategy_cache`(预计算策略缓存)与 `sw_runtime_kernels`。
- **旧 KeemBay vpux 插件**(OpenVINO ≤2022.1 时代,barrier/DMAType/Workloads IR 文档比现在开放)已从 master 删除,需翻旧 tag(本调研未验证)。
- **NPU5/Panther Lake 架构细节**:只有 tile 数(3)来自插件表格;其余在 NDA 后。
- blob 格式无官方规范;只有 vpux_elf 子模块源码 + npunlock 逆向可作为事实来源。

---

## 5. 对我们 W4 GEMV decode 引擎的"可采纳想法"排序

按(收益 × 可行性)排,结合现状(E2E 37.55ms / 26.7 tok/s,63 融合对替 126 exec,rms 双门过,attention/swiglu 元素化待做):

1. **invariant/variant 分级 + 描述符双缓冲环(技术 2+3)** —— 直接打击 exec 粒度开销与 16 BD 上限:层静态控制流常驻 PM,每 tile 只 DMA 小参数块进 L1 槽位(槽位数做成显式"定律",学 VPU_DMA_TASK_COUNT=256 那种预算表)。最像我们已验证的融合对路线的自然延伸。
2. **同步前移/数据化(技术 1+5)** —— 目标"host 每 token 只发 1-2 次":预写整层链的同步事件序列,让 shim/同步核级联。Intel 的演进路径(LEGACY→ALL_BARRIER_DMAS_SCHEDULED)就是证据:每一步把运行时工作搬进静态调度都有收益。
3. **attention 写回侧 swizzle/strided 定形(技术 8)** —— 我们 npu 口径 flowkv strided 主导 118ms;Intel 在 ODU/IDU 硬做 layout 换法,我们至少在 BD 侧固定 2D stride 模式消掉显式转置 exec。
4. **swiglu/rms 融到 GEMV 消费侧(技术 9)** —— 与 Intel PPE/ODU 同构;对我们是把"融合对"再往矩阵单元消费者里推一层。
5. **8 核 latency/throughput 双映射(技术 10)** —— 编译期固定两种核拓扑:decode=全核并一层(K-split+归约树),batch/预填=分组流水;OpenVINO 的 tile 表证明这是业界标准做法而非 hack。
6. **每-BD watermark 计数(技术 6+14)** —— 给帧成本定律加细粒度证据,先于优化找到 bubble 位置;DMA watermark 是一行描述符位的事(在他们 HW 上)。
7. **描述符链尾跳链 + skip 位(技术 6)** —— 16 BD 不够时"链表尾接预取好的下一段";静态调度里留动态逃生口(如提前退出解码)。
8. **多 shim/多核元数据就近放置(技术 11)** —— BD 与参数块放在消费核本地 L1 侧,避免跨核取描述符。
9. (远期)**权重调色板/结构稀疏(技术 12)** —— W4 之后再榨带宽的手段;需精度实验。

---

## 附:本次未读但值得跟进
- `firmware/include/api/vpu_jsm_api.h`、`vpu_jsm_job_cmd_api.h`(host↔fw IPC、job 命令含 barrier/copy/timestamp 命令编码);
- `umd/vpu_driver/source/device/vpu_hw_40xx.cpp`(设备能力上报:tile/DMA 数、频率);
- npu_compiler 仓库 `src/vpux_compiler` 的调度/barrier pass 与 `docs/`(ELBRUS 血统的 tiling/调度文档);
- Chips and Cheese 的 Lunar Lake NPU 篇(本次 URL 404,标题应为 Intel Lunar Lake NPU 深度分析)。

### 主要来源 URL 汇总
- 驱动固件头(核心):https://github.com/intel/linux-npu-driver/tree/master/firmware/include/api
  - vpu_nnrt_wlm.h / vpu_nnrt_common.h / vpu_dma_hw_37xx.h / vpu_nce_hw_37xx.h / vpu_nce_hw_40xx.h / vpu_cmx_info_37xx.h / vpu_cmx_info_40xx.h
- 驱动文档:https://github.com/intel/linux-npu-driver/blob/master/docs/overview.md
- 推理执行命令:https://github.com/intel/linux-npu-driver/blob/master/umd/vpu_driver/source/command/inference_execute.cpp
- OpenVINO NPU 插件 README(tile/request 表):https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md
- OpenVINO NPU device 文档:https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
- Meteor Lake NPU 微架构实测:https://chipsandcheese.com/2023/12/22/intel-meteor-lakes-npu/
- NPU 编译器(公开 MLIR 仓库):https://github.com/openvinotoolkit/npu_compiler
- 逆向自定义 SHAVE 内核:https://github.com/hsfzxjy/npunlock ;讨论:https://www.reddit.com/r/ReverseEngineering/comments/1wn4ps7/ ;https://news.ycombinator.com/item?id=49800513
- Lunar Lake 48 TOPS(营销口径):https://www.hothardware.com/review/intel-lunar-lake-deep-dive
