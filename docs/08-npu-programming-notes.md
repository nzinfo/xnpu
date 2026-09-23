# 08 · XDNA2 NPU 编程第一手笔记（教程素材）

> 来源：2026-09-22 R1 攻坚（IRON llama_3.2_1b 全公开栈首次在本机跑通 LLM 推理）
> 全部为实测证据，标注【实测】【官方】区分。教程可直接引用。
> 环境：Ryzen AI Max+ 395（Strix Halo，17f0:11），内核 7.0.0-31，amdxdna 0.7，
> 固件 1.1.2.65，XRT 2.25（apt python3-xrt + libxrt-utils-npu），mlir_aie 1.2.1，
> llvm-aie 22.0.0.2026091701（Peano），IRON decode-fusion-llama 分支（95ceeff）。

## 1. Context 模型：NPU 编程的第一约束

### 1.1 官方规范【官方】

来源：内核文档 <https://docs.kernel.org/accel/amdxdna/amdnpu.html>
（amdxdna 驱动补丁系列 2024-07 首发，2024-11 V9 定稿合入）。

- 每个 **workload context** 由固件中一个**专用 ERT 实例**（isolated
  non-privileged context）服务；管理命令走单一特权 **MERT**。
  → context 数上限 = 固件 ERT 实例表容量，是**硬上限**，与列数无关。
- **Mixed Spatial and Temporal Scheduling**：
  - spatial 分区可 *exclusively* 绑定 1 个 context；
  - 一个分区也可被**多个 context 时间复用**（微控制器逐切换改写分区 PASID）。
  → 16 个 8 列图塞进 8 列阵列是合法的：靠时间片，不是列够。
- 驱动内 **Resource Solver**：读 workload 元数据声明的列数 + 启发式决定
  （重）分区；固件强制执行 context↔列绑定。
- 每 context 一个 **64MB host 驻留指令缓冲**（ctrlcode 拷入，PASID 保护，
  同时映射进用户态）→ 16 ctx 满载 = 1GB 锁页。
  → 这就是一切 NPU 程序都要 `memlock unlimited` 的系统性原因。
- 每 context 的指令缓冲内容 = 预序列化命令包（insts.bin），
  NPU 执行的是**静态命令包**，非动态 JIT 调度（与 FLM 的 .xdna 观察一致）。

| NPU | 并发 contexts【官方】 | 拓扑 | 共享 L2 |
|---|---|---|---|
| Phoenix / Hawk Point | 6 | 4×5 | 2560 KB |
| Strix Point | 16 | 4×8 | 4096 KB |
| Strix Halo（本机） | **16**【实测，官方表未列】 | xrt-smi 报 6×8 | — |

### 1.2 超限的精确签名【实测】

第 17 个 context 起：

```
用户态: RuntimeError: DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22)
驱动态: aie2_send_mgmt_msg_wait: command opcode 0x2 failed, status 0x2000003
         aie2_hwctx_init: Alloc hw resource failed, ret -22
```

4.61GB host-only BO 池 + 16 context 并存无碍（排除内存压力假设）；
上限对 1 列小图同样生效（排除列数假设）。

### 1.3 预算探测法（教程级代码）

规范没提供查询接口，mlir_aie 的表还写错了（`npu2: 32`，官方实为 16）。
唯一可靠办法 = 初始化时实测：

```python
# 探测本机并发 HW context 预算：开哑 context 递增直到 EINVAL
import pyxrt
device = pyxrt.device(0)
xclbin = pyxrt.xclbin("any.xclbin")      # 任一可用图
device.register_xclbin(xclbin)
uuid = xclbin.get_uuid()
n, ctxs = 0, []
while True:
    try:
        ctxs.append(pyxrt.hw_context(device, uuid)); n += 1
    except RuntimeError:
        break                              # err=-22 → 预算 = n
```

陷阱：pyxrt 正确序列是 `xclbin→get_uuid→register_xclbin→hw_context`，
漏掉 register 会得到另一种假错（`No xclbin with uuid ...`）。

### 1.4 对图规划的推论

- FLM 的 layer.xclbin（一层一图，42 层复用 1 context + DMA 逐层喂数）与
  规范模型完全对齐：1 个 ERT 实例 + 1 个 64MB 指令缓冲跑整个模型。
- IRON 的 per-op × per-shape 图（llama-1B 全开 = 22 context）在此固件上
  必然撞墙；自研引擎必须把 context 预算当第一约束
  （大融合图复用 / 逐层串行 / 惰性重载三选一或组合）。

## 2. 编译流水线解剖（IRON artifact DAG）

```
SourceArtifact (aie_kernels/*.cc)
  └─KernelObjectArtifact      clang++ -O2 -std=c++20 --target=aie2p-none-unknown-elf
      │                       -D 宏参数化同一内核源码（mm.cc 行主/列主两个变体）
      ├─(可选) rename_symbols  llvm-objcopy-18 --redefine-sym 旧=新   ← 硬编码 18
      └─KernelArchiveArtifact  打包 kernels.a（解决一图多内核符号冲突）
PythonGeneratedMLIRArtifact   design.py 回调生成 mlir（参数=形状/头数/流水级数）
  └─XclbinArtifact             aiecc.py --aie-generate-xclbin --aie-generate-npu
      └─InstsBinArtifact       同一 mlir 生成 .bin（预序列化命令包）
```

关键坑【实测】：

1. **objcopy 版本硬编码**：`iron/common/compilation.py` 里 `"llvm-objcopy-18"`。
   Ubuntu 26.04 只有 21 → `apt install llvm-18`（universe 源有 1:18.1.8）。
2. **子进程用裸命令 `python`**：必须 venv bin 放 PATH 最前：
   `env PATH="/path/ironenv/bin:$PATH" python inference.py ...`
3. **脏缓存假完成（最阴的坑）**：clang 编译成功落盘 .o 后，若 rename 步骤
   （objcopy 缺失）失败，下次重跑**复用未改名 .o 并跳过 rename** →
   archive 内两成员同名符号冲突 → 链接期 `undefined symbol`。
   规则：任何编译失败后 `sudo rm -rf build/`（root 属主）再重跑。
4. **形状按 prompt_len 编译**：swiglu/mha/gemm 的 prefill 图在构造时把
   `prompt_length`（默认 2048）烤进形状，运行时 assert 精确匹配。
   短提示词必须填满（上游默认 prompt.txt=3908 token 截断到 2048 正好）。

## 3. 运行时对象模型（pyxrt / xrtruntime）

```python
device   = pyxrt.device(0)                 # "RyzenAI-npu4"（xrt-smi 命名）
xclbin   = pyxrt.xclbin(path)
uuid     = xclbin.get_uuid()
device.register_xclbin(xclbin)
context  = pyxrt.hw_context(device, uuid)  # ← 消耗 1 个固件 ERT 实例

insts_bo = bo(host_only); write(open(insts.bin))   # 命令包进 device buffer
kernel   = pyxrt.kernel(context, xclbin, "MLIR_AIE")  # kernel_id 0x901

runlist  = pyxrt.runlist(context)          # 批提交：一次 sync 跑多内核
run      = pyxrt.run(kernel)
run.set_arg(0, 3)                          # opcode=3：执行 insts 缓冲
run.set_arg(1, insts_bo); run.set_arg(2, len)
run.set_arg(3.., 数据 BO)
runlist.add(run); runlist.submit(); runlist.wait()
```

- **BO 池**：`bo.host_only` 三类池——静态权重池（llama-1B bf16 = 4.61GB
  全驻 device buffer）+ 动态 IO 池按冲突分析分配；
  同形状 op 跨层共享 xclbin（mlir_aie 按 path+mtime 做 context LRU 缓存）。
- **mlir_aie 缓存逃生门**：`XRT_CONTEXT_CACHE_SIZE` 环境变量覆盖默认表
  （`{npu1:6, npu2:32}`，后者是错的）。但 IRON 在 prepare 阶段把 runlist
  冻结绑定 context，逐出后 handle 失效不会自动重载 —— 对 IRON 降缓存
  只会把崩溃推迟到运行中；正解是图数 ≤ 预算。
- **runlist 语义**：同一 runlist 内所有 run 必须属于同一 context
  （IRON 里有 `assert this_context == context`）。

## 4. 权重注入与融合开关（llama 应用层）

- config json 逐算子 `use_aie_*` 开关 = 现成的 bisect 矩阵，也是
  **context 预算调节器**：R1 关 rope(-4)/attn_projection_gemm(-2)/
  final_gemm(-1) 把 22 → 15 context 塞进 16 预算，decode 主路径
  （MHA/GEMV/FusedSwiGLU/Norm）全留 NPU。
- 融合算子是这套栈的常态而非特例：`swiglu_fused_decode`（gate+up+SiLU+mul
  四合一，Gemma3 论文技术族）、`mha`（QKᵀ+softmax+·V 整图）、runlist 化
  SwiGLU prefill。
- assign_weights 有上游 bug：fused-swiglu 配置下 `__init__` 跳建 gemv 算子
  但 assign 不跳（本仓已打补丁，`src/block/feed_forward.py:233`）。

## 4b. Tile 内存预算：d=128 MHA 移植的第一手教训（2026-09-22）

把 fused MHA 从 d=64 移植到 d=128（MiniCPM5 head_dim）撞上的第二堵墙
（第一堵是 context 16 上限）——**每 tile 数据内存硬上限**：

- 计算核（row 1-3）：64KB 数据内存，分 4 个 16KB bank；栈、RTP、idx buffer
  都从里面出。d=128 时 QK worker 需 memQ(2×16K)+memK(2×16K)+memA(2×8K)
  ≈ 83KB → `aie.tile op allocated buffers exceeded available memory`
  （先试 bank-aware 分配，再退化为顺序分配，都失败才报错——dump 里能看全
  每个 buffer 的地址区间，是调 SRAM 的第一工具）。
- MemTile（row 0 下）：256KB。Q/O 分发侧（inQ split + memO join 同列）
  d=128 时需 ~384KB → 同样爆。
- 解法：ObjectFifo `depth` 2→1（单缓冲）。代价 = 无 ping-pong 预取，
  prefill 吞吐降；正确性无损。**depth 默认值是 2**（objectfifo.py:43），
  不写 depth 的 OF（inQ/memO）也要显式压。
- 经验法则：tile 内存 ≈ Σ(OF depth × obj 大小 × 该 tile 上的端点数)。
  改 head_dim/批块前先算这笔账。

## 5. 环境配方速查（全部实测）

```bash
# 依赖
sudo apt install python3-xrt libxrt-utils-npu llvm-18

# 一切 NPU 命令的统一前缀（64MB/ctx 指令缓冲 + BO 池 → 锁页）
sudo prlimit --memlock=unlimited:unlimited -- \
     env HOME=/home/nzinfo PATH="/home/nzinfo/qwen/xnpu/ironenv/bin:$PATH" \
     python inference.py <weights> <tokenizer> --num_tokens 8 -vv

# 注意
# - 日志级别只用 -vv 及以上（-v 撞上游 KeyError bug）
# - IRON test.py 必须 pytest 跑；python 直跑只导入
# - venv pyvenv.cfg 开 include-system-site-packages（拿系统 pyxrt）
# - 分支/版本配对：decode-fusion-llama ↔ mlir_aie 1.2.1 wheel
# - hf-mirror 下载被单连接限速 → 6 路并行 -r 分段可达 16MB/s
```

## 6. 命名对照表（教程防混淆专用）

| 语境 | 名字 | 实指 |
|---|---|---|
| xrt-smi | RyzenAI-npu4 | 本机 Strix Halo NPU（aie2p） |
| HRX | npu5 | 同一设备（Strix=HRX npu4） |
| mlir_aie | npu1 / npu2 | Phoenix / Strix(+Halo 被并入 npu2 → 踩坑) |
| IRON config | `"device": "npu2"` | 编译档案名，与 context 预算无关 |
| xclbin ip_layout | kernel_id 0x901 "MLIR_AIE:MLIRAIE" | mlir-aie 生成的内核 |

## 7. 参考链接

- 官方架构文档：https://docs.kernel.org/accel/amdxdna/amdnpu.html
- 驱动仓库：https://github.com/amd/xdna-driver
- V9 补丁系列（含 "multiple concurrent fully isolated contexts" 原句）：
  https://lore.kernel.org/lkml/20241111181711.662686-1-lizhi.hou@amd.com/
- 硬件时间片调度（新驱动）：Phoronix "AMDXDNA Driver Preps Hardware Scheduler Time"
- 本项目 R1 复盘：docs/02（FLM 尸检）、README 状态区
