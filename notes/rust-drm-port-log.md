# Phase M0: Rust 直连 DRM (amdxdna) 移植日志

日期: 2026-09-22（第 6 会话）
项目: ~/qwen/xnpu/xnpu — Rust workspace `xnpu-hal` + `xnpu-cli`,复刻 irene-xdna-run/KMQ shim
蓝本: iree-amd-aie `runtime/src/iree-amd-aie/driver/amdxdna/shim/linux/kmq/` (本地 checkout)
      torvalds/linux `drivers/accel/amdxdna/` (7.0.0-31 in-tree, 与 master 字符串一致)
硬约束: host 禁 C++,Python→Rust;无 XRT/pyxrt 依赖,无 sudo

## 0. 结果 (TL;DR)

`xnpu-cli info` + `xnpu-cli ctx-probe` 全部打通:
- info: 8 列×6 行 aie2p (1,1), core 4 行 (start 2), mem 1 (start 1), shim 1 (start 0),
  locks/tile=16, fw 1.1.2.64, MP-NPU 1267MHz。
- **ctx-probe: 并发 HW context 预算实测 = 16**(1 列 ctx,第 17 个 EINVAL),与 docs/06
  设计约束吻合;全部正常销毁(DESTROY + SYNCOBJ_DESTROY)。

## 1. 打通过程的"四关"(每关的内核判定与修法)

| # | 现象 (errno/dmesg) | 内核判定点 | 修法 |
|---|---|---|---|
| 1 | EFAULT | create ioctl 无条件 copy_from_user(qos_p) | qos_p 指向真实零化 [u32;6] |
| 2 | ENOENT "dev heap object not exist" | client->dev_heap 未建 | Device::open 先建 64MB DEV_HEAP BO |
| 3 | EINVAL "Invalid dev heap userptr" | amdxdna_gem_heap_alloc 检查 heap uva;uva 只在 mmap 路径 (HMM register) 记录 | heap 必须 mmap 并保活 |
| 4 | EINVAL "Invalid num_col 10" | num_col = num_tiles/core.row_count,须 1..=total_col | num_tiles = cols×core_rows(=32),mem 行不计入;探测用 1 列=4 |
| 5 | fw status 0x4000003 (INVALID_PARAM) "Map host buffer failed" | **heap 的 uva 必须 64MB 对齐** | reserve+MAP_FIXED 对齐 mmap(见 §2) |

## 2. 关键认知:KMQ/SVM 模式下的地址模型(本次最重要发现)

- 这套内核把 host 内存以 **SVM/PASID 模式**交给固件:hwctx_init 的
  `aie2_map_host_buf(fw_ctx_id, heap_dma/uva, size)` 中固件实际消费的是 **heap 的
  用户虚拟地址**,且要求 **64MB 对齐**,否则 RTOS 返回 0x4000003 INVALID_PARAM。
  这正是 XRT shim bo.cpp 里 `mmap_bo(align=64MB)`(reserve 2×64MB-1 父区间 + 对齐
  地址 MAP_SHARED|MAP_FIXED)存在的原因——不是可选优化,是硬性要求。
- 排除法过程:num_tiles/mem_size/max_opc/qos 全扫无效;umq_bo/log_buf_bo=0 与 shim
  的 AMDXDNA_INVALID_BO_HANDLE=0 一致(UAPI 定义 INVALID=0,不是 0xffffffff);
  sudo+memlock 无关。最终 64MB 对齐 mmap 一击命中。
- GET_BO_INFO 返回的 DEV_HEAP `xdna_addr` = dev_mem_base (0x4000000),是设备侧堆基址;
  与 fw 的 host buffer 映射(uva)是两回事。

## 3. 排障方法论教训(本次绕的最大弯路)

- **python ctypes harness 初版 ioctl 全错**:DRM 驱动私有 ioctl 的 nr 必须加
  **DRM_COMMAND_BASE (0x40)**:cmd = _IOC(3, 0x64, 0x40+driver_nr, size)。漏掉 0x40
  会打到 DRM 核心 ioctl 上(表现为 EACCES=SET_VERSION 之类),看似"权限问题"实则
  cmd 号错。strace 解码名(如 DRM_IOCTL_SET_VERSION)是识别此类错的捷径——若名字
  不是 AMDXDNA_*,就是 nr 基址错了。Rust ioctl.rs 从一开始就是对的(0x40+id)。
- strace 对未知/共享号 DRM ioctl 只显示指针不显示字节;gdb catch syscall 在该环境
  抓不到 ioctl 停点(原因未明)。最终用 ctypes harness(修正 nr 后)复现+扫参,
  30 秒一轮,比改 Rust 重编译快得多。
- 内核源码(zread torvalds/linux)+本地 UAPI header(/usr/src/linux-headers-7.0.0-31)
  +XRT shim 三方对照定语义;dmesg 的 XDNA_ERR 字符串是内核版本定位锚点。

## 4. 当前 workspace 结构

```
xnpu/
├── Cargo.toml            # workspace: crates/xnpu-hal, crates/xnpu-cli
└── crates/
    ├── xnpu-hal/src/
    │   ├── ioctl.rs      # AmdxdnaCmd enum, amdxdna_ioctl(), raw_ioctl()
    │   ├── accel.rs      # Device{fd, dev_heap(BO+aligned Mapping)}, GET_INFO 查询
    │   ├── bo.rs         # BufferObject(CREATE_BO/GET_BO_INFO/SYNC_BO/GEM_CLOSE),
    │   │                 #   Mapping{new, new_aligned(reserve+MAP_FIXED), Drop=munmap}
    │   └── hwctx.rs      # HwContext::create(qos 零化指针), configure_cu(PDI->SHMEM BO
    │                     #   + CONFIG_HWCTX), Drop=DESTROY+SYNCOBJ_DESTROY
    └── xnpu-cli/src/main.rs  # info / ctx-probe [max] [cols]
```

clippy 0 warning;ctx-probe 32×1col 稳定输出 budget=16。

## 5. 下一步 (M0→M1)

1. xclbin PDI 段提取(section 解析)→ configure_cu(cu_func=0)→ 固件加载 CU。
2. ERT exec 包构造(EXEC_CMD=6):CMD_BO(AMDXDNA_BO_CMD) + arg BO handles,
   等待 = syncobj timeline wait(SYNCOBJ_WAIT DRM ioctl)。
3. 夹具:add_1c_2ch xclbin(/home/nzinfo/qwen/xnpu/build/)做第一个真实算子点灯。
4. UAPI 尺寸备忘:create_hwctx=64B? 实为 56B(3×u64+8×u32);create_bo=32;
   get_bo_info=48;sync_bo=24;exec_cmd=48;get_info=16;config_hwctx=24。

## 6. 陷阱清单(供后续阶段)

- BufferObject 持 raw fd(非引用):Drop 里 GEM_CLOSE 若在 Device fd 关闭后执行会
  EBADF——目前靠栈序保证,后续若引入多线程需显式生命周期管理。
- DEV BO(AMDXDNA_BO_DEV)的 uva = heap uva + offset;EXEC 包里给 AIE 的地址在
  SVM 模式下是 uva 系(待 M1 实证),非 GET_BO_INFO 的 xdna_addr。
- MAP_LOCKED 需 memlock rlimit;Mapping 有无锁回退(驱动提交路径自会 pin)。

---

# M1: 第一个真实算子点亮 (2026-09-22 完成)

**结果:`xnpu-cli run-add` VERIFY: PASS (2048 elements, bf16 exact),连续 4 次稳定。**
链路 = PDI 加载(CONFIG_HWCTX)+ ERT 包提交(EXEC_CMD)+ syncobj 等待 + 数据对拍,
全程 Rust 直连 DRM,零 XRT 依赖。

夹具:IRON add_1c_2ch_2048_2048t(1 core, 2048×bf16, ctrl-code 420 B)。
输入 in1[i]=(i%7)-3, in2[i]=(i/7)%5 → 和为小整数,bf16 精确可对拍。

## 1. 排障过程中钉死的四个事实

### ① BO 类型决定地址语义(本阶段核心认知)

| BO | 类型 | regmap/instruction 里填什么 | fw 如何访问 |
|---|---|---|---|
| ctrl-code (insts) | **DEV (ty=3)**, heap 里 0x4028000 | `xdna_addr`(heap 设备地址) | fw 直接读 heap |
| 张量 (in1/in2/out) | **SHMEM (ty=1)**, xdna=INVALID | **用户 VA**(mmap 地址) | fw 走 SVM/PASID 翻译主机页表 |

- IRON runtime 实证(`aie_context.py`):insts = `bo.cacheable + group_id(1)`
  → XRT shim 落 heap(DEV);张量 = `bo.host_only + 0x10000` → SHMEM。
- **张量地址给 heap 地址 = fw 静默忽略**:命令 COMPLETED(~200µs)但 AIE 不写任何
  数据(out 保持初值)。第一版全 DEV 布局即死于此。
- 反之 instruction_buffer 给用户 VA → fw ABORT(state=6)。
- 之前"fw 不收用户 VA"的结论只对 instruction 字段成立;张量字段恰恰**必须**是
  用户 VA(SVM/PASID 模式),且 BO 须出现在 EXEC_CMD 的 arg_bos 里让内核 pin。

### ② ERT 包真实线格式(用 LD_PRELOAD ioctl 拦截器抓 XRT 原包解码)

XRT 在 Strix Halo (npu2) 上发的 exec 包 = `ert_start_kernel_cmd`:
- header: state=1(NEW), **opcode=0(ERT_START_CU)**, type=3(ERT_CU),
  count=[22:12] 含 cu_mask 在内的 payload 字数
- `cu_mask` @0x04 = 1
- **payload @0x08 起就是 regmap 本身,没有 ert_npu_data 前导**:
  `[opcode u64=3][instr u64=heap地址][ninstr u32=420][in1 VA u64][in2 VA][out VA]`
- 内核 `aie2_cmdlist_fill_npu_cf`(npu_exec_message_ops,ERT_START_CU 分支)把整段
  payload 原样 memcpy 进 `cmd_chain_slot_npu.type=EXEC_NPU_TYPE_NON_ELF` 的 args。

**致命陷阱**:若按 ert.h 文档发 opcode=20(ERT_START_NPU)+ 16B `ert_npu_data`
前导,内核走 `fill_npu_dpu` → `EXEC_NPU_TYPE_PARTIAL_ELF`,fw 按 ELF 语义解析
裸 txn bin → **静默不执行**(依旧 COMPLETED)。这就是"包看似合法但无输出"的根因。

### ③ EXEC_CMD UAPI (Linux 7.0)
`amdxdna_drm_exec_cmd`(56B):ext, ext_flags, hwctx, type, **cmd_handles(u64,
cmd_count==1 时内联句柄值,否则指针)**, args(u64, 指向 u32 句柄数组),
cmd_count, arg_count, seq(out)。XRT 传 arg_bos=[ctrl, in1, in2, out]。

### ④ 其它钉死的坑
- CONFIG_HWCTX cu_config 打包:{u16 num_cus; u16 rsvd[3]} + {u32 cu_bo; u8 cu_func;
  u8 pad[3]} → 句柄在 offset 8、func 在 12(写错位 → EINVAL)。
- SYNCOBJ_DESTROY = _IOWR('d', 0xc0) 非 0xb6。
- SYNC_BO direction=1(FROM_DEVICE)无 debug BO 时 EINVAL;刷 cache 用 direction=0
  即可(clflush 与方向无关)。
- bf16 RNE:`bias=0x7fff+((b>>16)&1); (b+bias)>>16`(进位加在 32 位原值上)。
- SHMEM BO 的 Mapping 必须存活到 run 结束(临时映射语句结束即 unmap,VA 被复用
  → fw 读到错误数据)。

## 2. 调试方法论记录

1. 最小 pyxrt 复刻(/tmp/pyxrt_add*.py)与 IRON pytest 同环境同 XRT → 排除驱动/
   硬件问题,锁定"调用方式差异"。
2. strace 只见 ioctl 名不见参数 → 写 **LD_PRELOAD ioctl 拦截器(Rust cdylib,
   /tmp/xdump.rs)**:拦 ioctl+mmap,维护 handle→(vaddr,xdna,map) 表,EXEC_CMD
   时 dump 命令 BO 全量 hex + arg BO 前 256B。IRON pytest 与复刻脚本各跑一遍,
   逐字节对拍 → 一锤定音。
3. 拦截器两处翻车记录:① Mutex 非重入,锁内调 log_line 再取锁 = 死锁;
   ② cmd_count==1 时 cmd_handles 是内联值非指针,盲解引用 = 段错误。
3'. XRT exec 包由 C++ 构造,Python 层看不到 → LD_PRELOAD 是唯一无侵入观测点。
4. IRON 实际走 `pyxrt.runlist`(xrt::runlist 批执行,aie_base.py run_runlist),
   但单 run(kern(3, insts, len, *bos))与 runlist 的包格式一致,复刻单调用即可。

## 3. M1 代码落点

- `xnpu-hal/src/ert.rs`:StartNpuCmd 改为 XRT 线格式(opcode=0、无 npu_data、
  regmap 从 0x08 起);set_ctrl 保留为 no-op(API 对称)。
- `xnpu-cli/src/main.rs` run-add:ctrl=DEV BO(heap)+ 张量=SHMEM+常驻 Mapping+
  用户 VA 进 regmap;arg_handles=[ctrl,in1,in2,out];读回=direction 0 刷 cache
  后直接读 mapping。
- `xnpu-hal/src/hwctx.rs`:cu_config 偏移修复;clippy 清零。

## 4. 遗留/下一步

- seq=0 疑点:exec 返回 seq 恒 0,timeline point 0 可能预 signaled,wait 或为假过
  → 用包 state 轮询兜底(现已有 state==COMPLETED 检查),M2 前查内核 seq 赋值。
- M2 方向:把 R2b 的 Python IRON 调用层换成本 HAL(先 matmul 夹具,再逐层迁移
  MiniCPM5 算子);decode 吞吐目标 > CPU 4.55 tok/s,不足则层间融合/int8(用户备忘)。
- 上游求证(外发,需用户确认):①O-join 写路径 D2(w16→w8);②compilation.py 缓存 bug。

# M1b:量化计算模型(int8/q8/fp4)调研 + int8 算子点亮(2026-09-22 深夜)

用户指令:"2048 元素 bf16 加法,但是需要考虑 q8 int8 fp4 之类的计算模型"。
结论:**int8 elementwise 已在 Rust HAL 上 VERIFY PASS(2048 元素精确,4 次稳定)**,
IRON 侧 1/2/4/8 列配置 20/20 通过。fp4 全栈无支持;q8 有明确两条路线(下述)。

## 1. 本机 AIE2P 量化计算能力盘点(源码级)

- **int8 原生 MAC**:`aie2p/mm.cc` combos 表含 `i8×i8→{i8,i16,i32}`(8×8×8 MAC
  形状)及 i16 组合,`*_ONLY` 编译开关按需实例化 → int8 GEMM 的积木已在。
- **int4 打包权重 + bf16 块 scale**:`aie2p/fused_dequant_gemv.cc`
  (`fused_dequant_matvec<block_size>`)是 R2 吃 model.q4nx 的真实路径:
  权重 uint4 打包(2 值/字节)+ 每组 bf16 scale,在线反量化
  `uint4→aie::unpack→uint8→uint16→bf16→×scale→bf16 MAC`。
  即:**现有量化推理 = 反量化到 bf16 再算,MAC 停在 bf16**。
- **fp4/e2m1/mxint/mxfp**:IRON 全树 grep 零命中;AIE2P 向量 ISA 无 fp4 数据类型。
  → fp4 只能走自定义内核:位操作解包 + LUT(或乘 scale)升 bf16,再进 bf16 MAC;
  `lut_based_ops.cpp` 的 LUT 手法可复用。存储收益与 int4 相同,计算无原生加速。
- elementwise_add 的 dtype 硬编码三处 + reference/test 的参数化缺口,本次已全部
  打通(见 §2)。

## 2. int8 elementwise 全链路改动(IRON 侧 5 文件)

- `aie_kernels/generic/add.cc`:extern "C" 增 `eltwise_add_i8_vector`
  (i8×i8→i8,补码回绕;模板 `eltwise_vadd<T_in,T_out>` 天然支持)。
  **教训:`aie::vector` 没有 `to_vector`(那是 accumulator 的方法)**,窄入宽出
  (i8→i16)不能靠它转,首版混合内核编译失败后砍掉——宽累加场景由 mm.cc 的
  i8_i32 combos 承担。
- `elementwise_add/design.py`:dtype 参数(bf16 默认/i8),选 np_dtype 与内核符号;
  CLI 增 `--dtype`。
- `elementwise_add/op.py`:dtype 贯通 add_buffer(dtype=np.int8,字节数=count×
  itemsize)/test_pattern/read_buffer;**artifact 名加 `_i8` 后缀,bf16 名字节不变**
  →旧缓存全复用,零回归。
- `elementwise_add/reference.py`:整数 dtype 走 `torch.randint(-60,61)`(torch.rand
  不支持整数;值域保证和∈[-120,120] 不回绕,精确比对有意义)。
- `elementwise_add/test.py`:i8 参数化用例(id 加 `_i8`);**精确匹配的坑:
  `nearly_equal` 断言 rel_tol≥float32 eps,拒绝 0** → 用
  `rel_tol=eps, abs_tol=0` 实现(整数错配差≥1,阈值<1e-6,不可能漏判)。
- `common/test_utils.py` **verify_buffer 硬编码 `// 2`**(bf16 itemsize)且
  read_buffer 默认 bfloat16 → 改为随参考值 dtype 推导 itemsize;bf16 行为不变。

## 3. Rust 侧改动(xnpu)

- `xnpu-cli run-add [prj] [bf16|i8]`:DType 枚举(itemsize/pack/unpack);
  int8 张量 2048×1B SHMEM,期望值 = 回绕 i8 加法。HAL 零改动——**SHMEM/用户 VA
  契约与 ERT 线格式对 dtype 完全无关**(本来就只是内存字节),这正是 M1 地址
  语义定论的直接推论。
- 夹具:`add_1c_2ch_2048_2048t_i8.{mlir,bin,xclbin}` 在 ~/qwen/xnpu/build/
  (pytest 从 ~/qwen/xnpu 跑,与 M1 同一棵 build 树)。

## 4. q8 计算模型设计备忘(块缩放 int8,待 M2 实现)

Q8_0 式(32 元素块 + 每块 scale)两条内核路线:
- **A. 原生 int8 MAC + 尾部重标定**:激活/权重均 int8,mm.cc `i8_i32_ONLY`
  combos 出 int32 累加,逐块 `acc×(sa×sw)` 收敛 bf16。峰值吞吐最高
  (int8 MAC 2×bf16 密度),需写 scale epilogue + 两次重量化;
- **B. fused_dequant 模式(已验证的现成范式)**:int4 路径直接改造——权重 int8
  + bf16 块 scale,内核内 `int8→bf16→×scale→bf16 MAC`。改动小、精度与 R2
  Q4NX 同级,但 MAC 停在 bf16,吞吐无增益(仅省权重带宽)。
- 决策依据:R3 实测 decode 瓶颈是逐算子同步开销(2.26 vs CPU 4.55 tok/s),
  **先做层间融合/批量调度,量化是第二优先级**;届时按 A 路线做 GEMM、elementwise
  已就位。

## 5. 遗留

- i8→i16 混合 elementwise(需要显式加宽手法:aie::unpack 链或逐元素转)——暂缓,
  GEMM 侧 i8_i32 已覆盖宽累加需求。
- fp4 自定义内核(LUT 解包)未动工,确有需求再做。

## 6. pytest 全量跑撞 16-context 预算墙(与 int8 无关,顺手钉死)

- 现象:i8 单独跑 20/20 过;与 bf16 混跑(100 用例)时排在后面的全挂
  `CREATE_HWCTX EINVAL`。根因:elementwise_add 全量 = 20 个唯一 xclbin
  (16 bf16 + 4 i8)> 16 预算,而 `AIEDeviceManager` 是单例、
  `DefaultNPURuntime`(CachedXRTRuntime)是进程级全局,context 跨测试累积。
- mlir_aie hostruntime 两个上游缺陷:①load() 逐出重试只认 ENOENT
  ("No such file or directory"),amdxdna 预算耗尽返回的 EINVAL 直接 raise;
  ②`XRT_CONTEXT_CACHE_SIZE` env 读进来是 str,比较 `len>=cache_size` 报
  TypeError(env 变量形同虚设)。
- 修法(IRON conftest.py,pytest 作用域,版本化):
  `DefaultNPURuntime._cache_size = 4`(主动逐出在 create 之前发生)+ monkey-patch
  `CachedXRTRuntime._evict` 逐出后 `gc.collect()`——**逐出的 pyxrt context 在
  引用环里,不 GC 就不释放 DRM context**(ENOENT 重试路径有 gc.collect,主动
  逐出路径没有;这就是第一次只修 cache_size 仍剩 6 个 EINVAL 的原因)。
- 作用域安全性:每测试 ≈1 个唯一 xclbin,逐出不会打中活 handle;推理路径
  (~12 活 kernel)不走 pytest,不受影响。100/100 + axpy 180/180 验证通过。
- 对自研引擎的意义:这与 R1 结论同源——16 预算 + 缓存不释放 = 长会话必炸;
  xnpu 引擎侧由我们自己管理 ctx 生命周期(HwContext Drop 即 destroy),无此坑。

# M2(进行中):GEMM 重算子上板 — Rust 直连 DRM 跑通矩阵乘(2026-09-22 深夜②)

## 1. 结果

- `xnpu-cli run-gemm [prj] [M K N]`:
  - 小夹具 gemm_192x384x64_48x96x16_0_0(f32 累加, gemm/test.py 的
    prio_accuracy=True 编法):**VERIFY PASS, 12288 元素逐位精确**。
  - gemm_2048x2048x2048_64x64x64_0_0(1 列, f32 累加):**4194304 元素逐位
    精确 PASS**, 58ms。
  - gemm_2048x2048x6144_64x64x64_0_0(R2 swiglu 遗留, 8 列, **bf16 累加**):
    max 3 ULP(相对 f32 精确参考)→ 判 PASS, 10-12ms。
- **吞吐(bf16, 含提交开销)**:
  | 夹具 | 列 | 累加 | 校验 | 吞吐 |
  |---|---|---|---|---|
  | 192×384×64 (48×96×16) | 4 | f32 | 逐位 | 61-104 GF/s |
  | 2048³ (64³ tile) | 1 | f32 | 逐位 | 294 GF/s |
  | 2048³ (32×32×128) | 8 | f32 | **逐位** | **2545 GF/s** |
  | 2048×2048×6144 (64³) | 8 | bf16 | 3 ULP | 4897 GF/s |
  列数是第一吞吐杠杆(1→8 列 ≈ 8.7×);同列数下 bf16 累加 + 64³ tile 比
  f32 累加 + 32×32×128 tile 快 ~1.9×(累加器宽度 + tile 形状)。

## 2. 新钉死事实

- **IRON 恒保留全阵列**:main_aie_partition.json 的 column_width=8 与设计
  实际列数无关(add 1 列夹具也是 8)→ hwctx num_tiles 按 8 列建即可,Rust 侧
  已改为从该文件解析(load_fixture 返回 (pdi, instr, cols))。
- **regmap 参数序 = runlist 参数序**:op.py add_to_runlist("gemm","A","B_0",
  "C_0") → [3][instr][ninstr][A VA][B VA][C VA];main_kernels.json 的
  bo0..boN 偏移(0x14 起, 每个 8B 对齐到 u32 槽)与 M1 完全一致,**对所有
  IRON 算子普适**——这就是引擎层的算子 ABI。
- **夹具累加模式决定可否逐位对拍**:bf16_f32_ONLY(f32 累加)可逐位精确
  (输入取小整数 → 任意求和序的 f32 部分和均为 <2^24 精确整数, 终值一次
  RNE);bf16_bf16_ONLY(R2 swiglu 默认)有累加噪声(实测 max 3 ULP)。
  验证器双层:bit-exact 优先, 否则 ≤64 ULP 判 PASS(bf16-accum 夹具),
  超限才 FAIL。
- B/C 布局由夹具名后缀 `_{b_col_maj}_{c_col_maj}` 决定;0_0 = 全行主序,
  Rust 端按行主序打包/校验即可。

## 3. 代码落点

- xnpu-cli/src/main.rs:load_fixture(共享 PDI/ctrl/分区解析)、cmd_run_gemm
  (A=((i+j)%7)-3, B=((5j+3l)%7) 整数模式;warmup 一次再计时;GFLOP/s 打印;
  双层校验)。
- 夹具由 pytest 编译(CWD=~/qwen/xnpu, 与 M1/M1b 同一棵 build 树)。

## 4. M2 后续(引擎方向)

- MiniCPM5 decode 逐层 GEMM 形状是 M=1 的 GEMV 侧(gemv/dual_gemv_silu_mul
  算子)与 prefill 的整行 GEMM——后者形状正是 2048×2048×6144(实测 4.9
  TFLOP/s)。下一步:①多 BO 常驻 + 单 ctx 多算子切换(ctx 复用省 PDI 重载);
  ②i8 GEMM 夹具(mm.cc i8 combos);③把 R2b Python 层的 forward 换成本 HAL。

## 5. 多算子单 ctx:驱动侧语义(源码级,待实测)

- **CONFIG_HWCTX 一次可挂 ≤32 个 CU**(aie2_msg_priv.h `MAX_NUM_CUS 32`),
  每个 cu_config = {cu_bo(PDI 的 DEV BO 句柄), cu_func};fw 侧
  MSG_OP_CONFIG_CU 收 cfgs[] = {PDI heap 地址 | cu_func 位域}。
- **禁止二次配置**:aie2_hwctx_cu_config 对已有 cus 直接 EINVAL
  ("Not support re-config CU")→ 引擎必须**静态图一次挂全所有算子 PDI**。
- exec 路径:包的 cu_mask 取 bit → cu_idx → fw 选 cfgs[cu_idx]。
  即 CU 选择是**每包的**,ctrl-code 仍是 regmap 逐次传——单 ctx 理论上
  可轮转 N 个算子,0 次 PDI 重载,16-ctx 预算对引擎不再构成约束。
- 未知(待实测):两个 PDI 都是全阵列预留(IRON 恒 column_width=8),fw 是否
  允许共存/切换时是否重放 CDO(性能)→ 用 add+gemm 双 CU 实验。

## 6. 多算子单 ctx:双 CU 实测 PASS(2026-09-22)

> **结论修正(见 §8)**:无 clobber 成立;但"零切换代价"是当时的测量
> 盲区 —— 没有单 CU 连续执行基线对照。run-pipe 基线补上后,交替执行
> 的 gemm 1.2ms/op 里其实有 ~1ms 是 CU 切换的 PDI 重载。

**结论:单 ctx 多 CU 静态图路线成立。** `xnpu-cli run-multi`:一个 8 列 ctx,
一次 CONFIG_HWCTX 挂 2 个 CU(CU0=add bf16 2048,CU1=gemm 192×384×64 f32acc),
per-packet cu_mask 选择,交替执行 **8 轮 × 16 次** 全部校验通过:

- round 1 即 PASS(两 PDI 共存被 fw 接受,无 EINVAL);
- **round 2+ 的 add 依然 bit-exact → 后续 CU 的加载/执行不 clobber 前一 CU**;
- 延迟零退化:add 166-203µs、gemm 1.20-1.39ms,8 轮平稳无趋势 →
  **CU 间切换代价低于测量噪声**(fw 要么把多 PDI 同时驻留,要么切换
  开销极小;从外部无法区分,工程上等价于"无代价")。
- 语义上 16-ctx 预算对引擎**不再构成约束**:一个 ctx 挂满 32 CU,
  MiniCPM5 全部唯一算子(~12 个 xclbin)远低于上限,0 次 PDI 重载。

代码:`HwContext::configure_cus(&[(pdi, cu_func)])`(param = num_cus@0 u16 +
每 CU 8 字节 {u32 cu_bo, u8 cu_func},PDI BO 全部保活于 `_pdi_bos: Vec`);
`configure_cu` 保留为单 CU 包装。CLI 侧 `OpState`(ctrl BO + 张量 BOs +
packet 打包)+ `run-multi [add-prj] [gemm-prj] [M K N]`;校验抽出
`check_add`/`check_gemm` 供 run-add/run-gemm 复用。

## 7. q8 路线 A 实测:int8 GEMM 上板(2026-09-22)

**int8 原生 MAC(i8×i8→i32 累加)全链路点亮**:IRON gemm dtype 化 +
Rust `run-gemm [prj] [M K N] i8` 直连 DRM。**2048³ 逐位精确 PASS
(4194304 元素,宿主 i64 参考 = 核内 i32 累加,bit-exact)**,吞吐:

| 配置(8 列) | dtype | 吞吐 | 对拍 |
|---|---|---|---|
| 32×32×128 tile | bf16 + f32 acc | 2546-2560 GF/s | bit-exact(小整数) |
| 32×32×128 tile | **i8 + i32 acc** | **6060(pytest)/6314(Rust)GOP/s** | **bit-exact(纯整数)** |

→ **同 tile int8 是 bf16 的 2.4×**,q8 路线 A(原生 int8 MAC + 块 scale
尾部重标定)吞吐价值实证。小问题 192×384×64:i8 109.5 vs bf16 90.7 GF/s。
**int8 的对拍基础比 bf16 更强**:整数累加无舍入序问题,任意求和序恒等,
不依赖"部分和 <2^24"构造。

IRON 侧改动(gemm 5 文件,复用 M1b elementwise 模式):
- design.py:`--dtype_in` 增 i8、`--dtype_out` 增 i32;mac_dim_map
  npu2["i8"]=(8,8,8)(原生 int8 mmul 恒 8×8×8,与 bf16 emulate 无关)。
- op.py:i8 分支 flags 只加 `-Di8_i32_ONLY`(禁 EMULATE/ROUND_CONV_EVEN/
  bf16 combo);prio_accuracy 强制 False(i32 累加本就精确);artifact 名
  加 `_{dtype_in}_{dtype_out}` 后缀(缓存不互踩);add_buffer/read_buffer
  按 np dtype;顺手修 `_execute_aie_operation` 重加 B 的上游 bug
  (原 `self.M * self.N` 应为 `self.K * self.N`)。
- reference.py:整数走 torch.randint(-4..4)+ int32 加宽 matmul
  (int8@int8 会按提升规则回绕 int8);test.py 加 2 个 i8 用例
  (192 小 + 2048³ 8col),容差 rel_tol=eps/abs_tol=0(整数错配差≥1)。
- pytest 10/10 通过;mm.cc(aie2p)零改动(上游本就有 i8_i32_ONLY 组合,
  导出 matmul_i8_i32/zero_i32)。

Rust 侧:`pack_i8`/`check_gemm_i8`(i64 累加参考);`run-gemm` 尾参
`bf16|i8` 选路;bf16/multi 回归全绿。**q8 后续**:块 scale (de)quant 算子
(int32→bf16 重标定 elementwise,mm.cc 需新内核)+ GEMV 形状 M=1 的 i8
路线 + 权重 int8 量化 importer。

## 8. run-pipe:命令队列流水线 + CU 切换代价定量(2026-09-22,M2 真正收官)

**`xnpu-cli run-pipe [prj] [M K N] [iters]`**:单 ctx 单 CU,N 个 cmd BO
(同 regmap)背靠背 submit 后统一 poll 完成,对比逐 op submit+wait。

### 结果一:seq 疑点关闭
exec 返回的 timeline seq **逐次递增**(sequential 32 次 = 1..32,pipelined
= 33..64)。M1 的"seq 恒 0"只是单提交进程的首点 —— `wait(seq)` 是真等待,
timeline 语义正常,不是假过。

### 结果二:小算子流水 2.4×(decode 场景的直接答案)
gemm 192×384×64(8 列):
- sequential(逐 op wait):**94.8µs/op**(256 iters 复测 89.8µs)
- pipelined(全 submit 后 drain):**41.2µs/op**,其中 submit 仅
  5.3µs/包,**纯设备时间 36µs/op**(drain 1.15ms/32)
- → **submit+wait 往返 ≈ 53-59µs/op**,这就是 R3 Python decode
  (2.26 tok/s < CPU 4.55)的元凶;队列深度足够消化 256 个小包。

### 结果三:大算子不要流水(队列背压)
gemm 2048³(6.7ms/op 计算主导):pipelined 反而 15.2ms/op —— submit
loop 143ms 才入队 16 包(队列满后 submit 阻塞等 fw 消化)。**流水线收益
在 overhead 主导的小算子串上**,大算子 sequential 即可。

### 结果四:CU 切换代价 = PDI 重载(run-multi burst vs interleaved)
| 调度 | add(2.4KB PDI) | gemm 192(57KB PDI) |
|---|---|---|
| burst(同 CU 连续 ×8) | 74.8µs/op | 226.0µs/op |
| interleaved(交替 ×8) | 162.5µs/op | 1235.6µs/op |
| **切换代价/次** | **+87.7µs** | **+1009.6µs** |

fw 在 cu_mask 变化时**重载目标 CU 的 PDI**(~18-37µs/KB + 固定开销);
正确性不受影响(重载是完整的,校验全过),但性能上跨 CU 交替很贵。

### 引擎执行模型定论(M3 输入)
1. 单 ctx 多 CU 静态挂载(§6)+ **层内 DAG 拓扑序连续 submit,只在
   host 需要数据处(层末/logits)同步** —— per-op 开销 59µs→5µs;
2. 调度按 CU 分组:能连续同 CU 的算子排一起,跨 CU 切换按 PDI 大小
   计价(gemm 级算子一次 ~1ms,值得重排省);
3. 根本解仍是 M3 图编译:多算子合并进单 PDI,算子间 object FIFO
   片上直连,0 次切换 0 次 host 往返。

## 9. M3 第一实验 run-chain:跨 CU 数据依赖链(2026-09-22,git 20696b3)

**`xnpu-cli run-chain [add-prj] [gemm-prj] [M K N] [reps]`**:单 ctx 双 CU
(gemm=cu1 → add=cu0),E[0..2048] = (A@B)[0..2048] + D,其中 **gemm 输出 C
与 add 输入 1 是同一个 SHMEM BO** —— 同一 VA 出现在两个包的 regmap 里,
in-order 队列保序,算子间零 host 往返零拷贝。这就是一层 decode 的形状
(GEMM → 偏置/残差加)。

### 结果一:共享 BO 直连依赖成立

sequential(逐 op wait)与 chained(submit gemm+add 背靠背、每链一次
drain)的 E 输出**完全相同**(同 318/2048 diff、同 first)→ chained
submit 不破坏跨 CU 数据依赖。分阶段 check_e 的设计就是为此:sequential
也错 ⇒ 错在参考;只有 chained 错才是依赖被破坏。

### 结果二:318 个 diff 的根因 = add 内核 tie 舍入非 RNE

@[16]:C = RNE(-773) = -772,D = 2,精确和 -770 恰为 -772/-768 的中点;
host RNE 选偶(-768 = 0xc440),内核给 -772(0xc441)。全部 318 个 diff
都是 max 1 ULP 的同类。**根因:AIE 的 f32→bf16 转换默认舍入不是
half-to-even** —— mm.cc 要显式 `-DROUND_CONV_EVEN` 才是 RNE(该宏存在
的原因);add 夹具早于此认知,没开。M1 的 2048 元素 bit-exact PASS 没有
暴露它,因为小整数和全部精确可表示、无 tie。
**规范(自建内核默认):凡有浮点收窄转换的内核一律开 ROUND_CONV_EVEN,
否则引擎校验层必须按 per-kernel 舍入语义给容差(≤1 ULP tie tier)。**

### 结果三:chained 提速 1.25×,残余被 PDI 重载主导

sequential 1.39ms/chain → chained 1.11ms/chain(192³ gemm + 2048 add,
8 reps)。链上只有一次 cu_mask 变化(gemm→add),但那次变化就要重载
57KB gemm PDI(~1ms,§8 表)—— 即 **链式 submit 拿回的是同步开销,
拿不回切换开销**。再次确认 M3 调度规则:同 CU 分组 + 图编译合并单 PDI。

## 10. rescale 内核:q8 路线 A 尾部算子上板(2026-09-23,IRON bd26393)

**目标**:`C = bf16(f32(A_i32) × sa[M] × sw[N])` —— i8 GEMM 出的精确 int32
累加,乘 per-row(激活)× per-col(权重)scale 收窄回 bf16。这是 q8 路线 A
三件套的最后一块:i8 GEMM(§7,bit-exact)→ **rescale** → bf16。与参考同乘
法序(`(acc×sa)×sw`),pytest 两用例(192×64、256×128)10/10 全过。

### aie_api 转换语义五坑(全踩过,自建内核直接绕行)

1. `vector<T>::unpack<T2>` 要求 type_bits(To) > type_bits(From)——int32→float
   同宽被拒,bf16→float/bf16 pack 也拒;宽化走别的路。
2. `accum<accfloat>::from_vector(int32_vec)` 是**位重解释**不是数值转换:
   负数符号位变指数 → NaN。症状是输出大片 NaN。
3. `aie::to_float(vec)` 才是数值 Fix2Float(Gen2 支持 int32→float),q8
   量级下 int32→f32 宽化精确。
4. `aie::mul(float_vec, float_vec)` 的 AccumTag 不可推断,须显式
   `aie::mul<accfloat>(...)`。
5. 32-lane bf16 `from_vector` emulation 实测返回错段(sw 指针读到 sa 的
   数据)—— 规避:scales 以 f32 流经 fifo(host 侧 bf16→f32 无损加宽),
   内核全走原生 float 向量 load。

依赖名模板调用要 `.template to_vector<float>()`。舍入规范落地:内核内
`::aie::set_rounding(aie::rounding_mode::conv_even)`,bf16 收窄 tie 与
host RNE 一致(§9 结果二的规范第一次实践)。

### taplib / NPU shim 两条硬限制

- **BD 尺寸字段 10 位,范围 0..1023**:`aie.dma_bd` 每维 size ≤ 1023,连
  1024 都非法。chunk > 1023 时堆叠到迭代维(选 512 最省心:
  `[1,1,total//512,512] / [0,0,512,1]`)。
- **taplib sizes/strides 按 itertools.product 全迭代**:`[1,1,1,chunk]`
  只覆盖首块(elementwise 夹具每 worker 单 chunk 没暴露);多块必须把
  块数放进迭代维。症状:行块 0 全对、块 1+ 全 0。

### 结构取舍

- 单 tile DMA 通道预算:3 输入 fifo 报 "number of input DMA channel
  exceeded" → sa+sw 合并单 scales fifo,每行块拼 `[tile_m 行 scale | N 列
  scale]` 连续 f32 块,降到 2 进 1 出。
- Kernel 参数序惯例:指针在前标量在后,dump-probe(把 m/n/s[0]/s[m] 写进
  c_out 从 pytest mismatch 打印读回)实证 MLIR 传参正确。

### 手动编译命令(内核改动的快速验证环)

```
P=ironenv/lib/python3.14/site-packages
$P/llvm-aie/bin/clang++ -O2 -std=c++20 --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw \
  -I$P/mlir_aie/include -c IRON/aie_kernels/generic/rescale.cc -o /tmp/rescale.o
```

小尺寸(≤256×128)延迟 125~245µs,固定开销主导;真实带宽等 Rust 侧与
i8 GEMM 链成闭环后量。下一步:`run-rescale` 子命令 + q8 闭环对拍。

### q8 路线 A 闭环:`run-q8` bit-exact(2026-09-23,xnpu 30fba74)

**`xnpu-cli run-q8 [gemm-prj] [rescale-prj] [M K N tile_m] [reps]`**:单 ctx
双 CU,i8 GEMM(cu1)写精确 i32 累加 C,rescale(cu0)乘 sa×sw 收窄 bf16,C 是
同一个 SHMEM BO(VA 进两个包)。**结果:sequential 与 chained 两阶段均
0/12288 diffs、max 0 ULP —— 整条流水线逐位一致**(host 参考同乘法序 +
RNE 收窄;conv_even 规范端到端成立,这是本项目最强验证层级)。

- **±0 符号位**:sa=0 的行乘出 -0,AIE f32 乘零操作数给 +0(仅零操作数触发,
  其余含全部负值 bit-exact)→ 校验器归一化 ±0。推理语义等价,但写引擎校验
  层时记得这一条。
- 时序:sequential 1.45ms/loop → chained 1.18ms/loop,与 run-chain 定量吻合
  —— 残余仍是链上唯一 cu 切换的 57KB gemm PDI 重载(~1ms)。小尺寸下算力
  完全被切换开销淹没,再次指向图编译合并单 PDI。
- q8 三件套至此全通:**i8 GEMM(§7 bit-exact)→ rescale(本节)→ bf16**,
  M3 剩图执行器 / M=1 GEMV i8 化 / int8 权重 importer。

## 11. w4 权重路线首测:上游 fused_dequant_gemv 的三个定性(2026-09-23,xnpu 09abe15)

**背景**:上游有完整 INT4 路线 —— uint4 权重 + 每 32 元素组 bf16 scale 打包进
DDR,寄存器内解量化,单遍 GEMV(`fused_dequant_gemv` operator,2048×2048
pytest 全过)。这是 decode 的理想权重形态(权重流量 = bf16 的 1/4)。Rust 侧
`run-w4gemv` 直驱该 fixture,**首次即 bit-exact**(0/2048×3 阶段):dyadic
scale((g%4)+1)/16 + 小整数 x → w_dequant 精确 bf16、全 f32 部分和精确,
唯一舍入 = 尾部 conv_even 收窄。M1 的"精确测试数据"配方平移到量化权重路线。

### 定性一:瓶颈在内核内循环,不在 DMA 也不在调用开销

| 配置 | 延迟 | 权重流 |
|---|---|---|
| 4col × tsi=1/8/16 | ~1190µs(平) | 1.9 GB/s |
| 8col × tsi=8/16 | ~660µs | 3.5 GB/s |
| tsi=64 | 编译失败:tile 缓冲超 tile 内存 | — |

- **tsi(每 kernel 调用行数)1→16 完全无效** → 不是调用开销;
- **cols 4→8 近线性(1.8×)** → 不是总带宽,是每列各自 ~2.4µs/行;
- 每行 ~2.4µs = 2400 周期跑 ~600 条向量指令,MAC 利用率仅列峰值的 ~2.7%
  → **64 组深度的 mac 累加器循环携带依赖链 + 每组 scalar scale
  load→broadcast 串行链** 才是上限。
- tsi=64 失败根因:tile = 64×1152B = 72KB,fifo 深度 2 → 超 compute tile
  数据内存。tsi 上限 ≈16。

### 定性二:该内核撑不动 decode

投影 MiniCPM5(2.66G 权重 → w4 0.70GB/token):8col 3.5 GB/s → ~340
ms/token ≈ 2.9 tok/s,**低于 CPU 基线 4.55 tok/s**。上游内核形态正确但
内循环不行。

### 定性三:自建内核的两个抓手(下一步)

1. **多累加器展开**打破 mac 依赖链(64 深 → 4~8 条独立链);
2. **scale 折叠到组尾**:acc_group = Σ nibble×x(f32,或 i8×i8→i32 走
   route A),再乘组 scale 汇总 —— 省掉每组 bf16 mul + to_vector 收窄,
   依赖链直接砍半。
   预期 4~6×:12~20 GB/s → 35~58ms/token ≈ 17~28 tok/s,决定性超过 CPU。
   (i8 mac 路线即 q8 route A 在 M=1 的自然延伸,组级 scale。)

### 附:persistent-kernel 设计

worker `range_(0xFFFFFFFF)` 死循环等 fifo,launch 间无核心重启 —— 引擎
侧值得继承(省的是 core 消息传递往返)。DDR 打包布局 m_input=1 时恒等于
行主序(flat_tile = col·rpc + t = i),与 cols 无关。

## 12. w4gemv2 闭环:栈预算踩坑与 2 累加器定型(2026-09-23,IRON ac36445 / xnpu fd51b46)

§11 尾声:自建 w4gemv2 内核(4 累加器交织)性能 3.9× 但 row 0 损坏,
tsi=1 全行错。本节 = 根因 + 修复 + 最终数据。

### 根因:peano 栈预算 0x400,placer 把邻居缓冲贴在栈顶上方

链条(全部实验证实):

1. **ld script**:`_sp_start_value_DM_stack = <image 末尾>; . += 0x400;` ——
   peano 链接器只给内核镜像上方留 **1024 字节栈**;
2. **placer**:mlir_aie 把下一个 tile 缓冲放在 sp+0x400(tsi=1 实测
   sp=0x70000,B 缓冲恰在 0x70400);
3. **peano 栈向 sp 正方向增长**(帧 = [sp, sp+frame)),不是常见向下;
4. 上游内核恰好**零栈**(全寄存器),所以上游从未暴露此约束;
5. 我的 4 累加器版帧 0x4C0 > 0x400 → 累加器 spill 直接写进 B 缓冲前
   192B = **x[0..95] 被 spill 值覆盖**。tsi=1 时 b fifo depth=1、512 个
   kernel 调用共用一次 acquire → 一个调用 spill,全体后续调用读同一份
   损坏的 x → 全行错。got=1832 等大值 = exact 部分 + Σ nib·s·(spill
   浮点),对任何干净子集/错位/他行假设都不匹配 —— 因为垃圾输入根本
   不在我的数据里。

### 诊断方法(dump-probe 两连,值得复用)

- **probe1**(纯标量内核,把 m/k/row_off/group + 首段 W/scales/x 原始
  u16 + a/b/c 指针写进 c_out):证明**输入全部完好**(weights、scales、
  x、参数逐位正确),且暴露 b_in=0x70400、sp=0x70000 的相邻关系;
- **probe2**(快照 x[0..95] → 强制 1216B volatile 栈流量 → 再快照):
  后快照出现 (0x0000,0x439C)(0xC000,0x439C)… = f32 对 (312, 313.5,
  315…=q×1.5) —— **probe 自己的 volatile 数组写坏了 x**,正向栈帧
  实锤。
- 坑:iron build 产物落 cwd/build;相对路径 + shell cwd 漂移 + root
  属主(sudo rm)三连坑让两次"重跑"实际跑了旧 fixture。排障时先核对
  `main.pdi` mtime 再下结论。

### 修复:2 累加器(帧 0x1C0 ≤ 0x400)

组对交织(g%2 → acc0/acc1),链深 64→32。**bit-exact 全过**(pytest
15/15:1tsi/4col、16tsi/4col、16tsi/8col;CLI 三阶段对拍上游配方):

| 配置 | sequential | pipelined | vs 上游 |
|---|---|---|---|
| 1tsi/4col | 402µs (6 GB/s) | 347µs (7 GB/s) | 上游 1190µs → **3.0×** |
| 16tsi/4col | 379µs (6 GB/s) | 318µs (7 GB/s) | ~3.7× |
| 16tsi/8col | 226µs (10 GB/s) | **174µs (14 GB/s)** | 上游 660µs → **3.8×** |

投影:**52~67 ms/token 权重流 ≈ 15~19 tok/s**,决定性超过 CPU 4.55。
tsi=1 与 16tsi 差距小 → 主要收益来自循环重构,不是 tile 打包。

### 遗留约束(写进内核头注释)

- 任何加宽交织(4acc/8acc)前必须重查 ELF `paddxm [sp], #imm` ≤ 0x400;
  4acc 帧 0x4C0 是踩过的雷。
- 想再提速:①scale 折叠到组尾(§11 抓手 2,曾触发 peano RegBankSelect
  -O2 崩溃,需换形状重试);②多行交织(行维独立链,寄存器压力同样
  存在);③这本身是 mlir_aie/peano 上游缺陷(栈预留 0x400 与 placer
  不协调),可报 issue。

### 本次其他修正

- run-w4gemv 打包泛化到 tsi 行 tile([tsi·K/2 nibble | tsi·(K/32)·2
  scale]),上游 16tsi fixture 对拍 bit-exact 后才用于判别"打包 vs
  内核"(上游内核+我的打包 = 全对 → 打包无罪)。
- 判别实验链:x/scale 移位扫描、W 流任意字节偏移扫描、窗口 stride 8/32B
  —— 全空,正是"垃圾不来自我发的数据"的信号,把假设空间逼到输入侧
  损坏,才有了 probe1。

## §13 有符号 int4 ABI 定型 + 三个引擎形状上板 (M3a, 2026-09-23)

### 动机:上游 uint4 方案只能表示测试数据

复核上游 fused_dequant_gemv 的 reference:quantize_and_pack = 无符号 uint4
[0,15] + 正 scale + zero_point=0 → **只能编码非负权重**。真 LLM 权重必
须有符号。上游内核/参考这对组合对 rand 正权重自洽,搬真权重即废。

### 修复:零开销换 int4

- aie_api 本就有 `int4` 类型(types.hpp),`unpack` 的
  `get_next_integer_type_t` 做**符号扩展** → uint4→int4 仅两处类型改动
  (`const int4*` 指针 + `load_v<int4,32>`),unpack 链 int8→int16→
  to_float 不变,nibble 变 two's-complement [-8,7]。
- 陷阱:`to_float(v, shift)` 第二参是**定点移位**不是加数,不能用它补
  -8 偏移;零点项直接不要(对称量化,scale=amax/7)。
- reference.py 全部自建:quantize_and_pack(amax/7→bf16 scale、
  clamp [-8,7]、低 nibble 在前、tile 列主序布局)+ golden
  (W_dequant @ x)。零组 NaN 边界:`where(amax==0, 0, round(W/s))`,
  先 where 后除,不能先除再置零 scale。
- Rust run-w4gemv 配方:nibble = ((13i+7k)%16)−8,打包 `&0xF | hi<<4`,
  宿主参考同值 → dyadic 精确性保持。

### 验证(全过)

pytest 6 配置 × 5 iter:三个旧形状 + **三个 MiniCPM decode 引擎形状**
(qkv 2560×2048、gate_up 12288×2048、down 2048×6144,全部 8col)。
CLI 三阶段 bit-exact(warmup/seq/pipelined 各 0 diff),性能与无符号版
**完全持平**(226/174µs @2048²)——符号扩展零开销实证。

### 实测引擎形状(CLI,signed ABI,bit-exact)

| 形状 | tsi | sequential | pipelined | GB/s |
|---|---|---|---|---|
| qkv 2560×2048 | 16 | 261µs | 211µs | 11-14 |
| gate_up 12288×2048 | 16 | 959µs | 907µs | 15-16 |
| down 2048×6144 | 4 | 505µs | 450µs | 14-16 |

每层三投影合计 1.73ms seq / 1.57ms pip → **42 层 = 72.5/65.9 ms/token
≈ 13.8/15.2 tok/s(纯投影)**,未含 MHA/swiglu/add/CU 切换,但已决定性
超 CPU 4.55。真权重/token 流量 = 42×43M 权重×0.5625B = **1.02 GB**
(CLI 旧投影 0.70GB 系低估,已修)。

### 新钉死:tile 数据内存预算 = 64 KB/计算核

down 形状 K=6144 放置失败的放置图直接给出预算:A fifo(2×27648)+
B(12288)+C(2×512)=0x10FFF≈68KB 溢出;tsi=16(tile 55296)与 tsi=8
(27648)都放不下,**tsi=4(tile 13824,总 41.6KB)过且仍 14-16GB/s**
——双缓冲 depth 2 下 tsi=4 流水未被打破。设计规则:
`2×tile_bytes + K×2 (B fifo) + 2×tso×2 (C) ≤ 64KB`。
(§9 的 tsi=64 失败同因,当时只记了"超 tile 内存"。)

### 环境坑补遗

- 后台任务命令里带 `| tail` 会截断输出文件,排障信息丢失——编译失败
  类任务不要在后台命令里加 tail,grep 过滤也要保 exit code 可见。

### run-w4layer:真权重 42 层投影链首次闭环(§13 续)

`tools/w4_import.py`(离线,复用 IRON 打包器;safetensors→每层 4 形状
packed .bin,42 层共 1114.8 MB)+ `run-w4layer`(4 CU 单 ctx:qkv cu0 /
o cu1 / gateup cu2 / down cu3;每层权重独立 SHMEM BO 常驻,计时环零
host 拷贝;层 0 golden 对拍)。

**golden 4/4 PASS,worst rel err = 0.00**——真权重 f32 累加下宿主顺序
求和与内核 lane 交织求和在 bf16 收窄后逐位一致(求和序差 ~1e-7 远小于
bf16 半 ULP 0.4%,翻转概率极低,抽样 20 行未遇)。

调度模式(42 层 × 4 GEMV = 168 op,含 CU 切换):

| 模式 | ms/token | tok/s | GB/s |
|---|---|---|---|
| per-op submit+wait(R3) | 184.7 | 5.4 | 6.0 |
| per-layer(4 submit+drain) | 117.6 | 8.5 | 9.5 |
| pipelined(168 submit+1 drain) | **234.0(反常最慢)** | 4.3 | 4.8 |
| grouped(同 CU 分组) | **75.2** | **13.3** | **14.8** |

- **每 op CU 切换代价 ≈ 650µs**(per-op 1.10ms/op − grouped 448µs/op;
  30KB PDI ≈ 22µs/KB,与 §M3 run-multi 标定一致)。层序调度每 op 换
  CU,168 次 ≈ 109ms/token 纯切换。
- pipelined 比 per-op 还慢:深队列里每次 cu_mask 变化仍触发 fw PDI
  重载,且 168 个 in-flight cmd BO 加剧队列背压——**流水线救不了切换**。
- grouped 证明切换消失后即达单 CU 满速(14.8 GB/s = 单形状微基准水
  平);但 decode 层间/层内全链串行依赖(qkv→mha→o→…→down→下层),
  **分组序对真实 decode 不可行**——图编译把 4 形状合并进单 PDI(或单
  一参数化 persistent kernel)是唯一根治,M3b/M4 的核心工作。
- 现实可用数:**真实层序最好 per-layer 8.5 tok/s**(CPU 4.55 的
  1.87×),未含 MHA/swiglu/add;合并 PDI 后上限 13.3 tok/s。

环境/实现备忘:importer 按**文件路径**加载 reference.py(import iron
包会连 NPU 设备初始化一起拖入);safetensors 键带 `.weight` 后缀;
torch bf16 张量 `.numpy()` 不支持须先 `.to(float32)`;numpy 标量
`.tobytes()` 行为怪异用 struct.pack;4 PDI 单次 configure_cus 正常。

## §14 w4gemvu 通用内核:单 PDI 四形状闭环 (M3b, 2026-09-23)

### 设计(为何可行)

四投影形状 (qkv 2560×2048 / o 2048×2048 / gateup 12288×2048 / down
2048×6144) 的 device 侧完全一致——worker 循环、fifo 几何、放置全同,
只有 ctrl-code (BD 长度/tap) 随 (M,K) 变 → **一个 PDI,零 CU 切换**。
K 从 tile 槽尾运行时读取 (自描述 tile),m=4 编译期内建。

固定几何:ELEM=13840B = padding 后 max-K tile,acquire(1) 单指针契约
(acquire(n>1) 两重死路:放置器不保证元素连续 0x44000/0x48000/…,
dynamic-objFifo lowering 链接失败 `undefined symbol: A_..._cons_buff_1`)。
TILES_PER_B=16 = gcd{80,64,384,64} 的最大公因子,每 run F =
tiles/16 ∈ {5,4,24,4} 个 B 元素。B fifo depth 必须 ≥2:depth-1 自环
BD (bd N→N) 的重挂与流内已缓冲数据竞态。

### 排错一:peano -O2 锚点怪癖(int4 流指针丢 +8)

全输出确定性错误 (qkv 2444/2560,down 1963/2048),跨 depth-1/2、跨
脚本/pytest 稳定。**单位向量指纹法**定位:x=e_j + 图样权重 (全 +1,
scale 1),扫描 j 得每行有效权重向量——row0 = [0,0,-8,0,…,1,1,1],
恰是槽头 K 字节 `[00 08 00 00]` 当 nibbles 解包 (0x08 低 nibble = 有符
号 -8,位置 2);每行首 8B 打 0x33 标记复测:**四行权重窗口全部 -8B
偏移** (row r 有效窗口 = 槽 [r·k/2, r·k/2+k/2)),x 对齐与 scale 锚点
(+8) 和 K 读取锚点 (+0) 均正确。即 peano 把
`weights_packed=(int4*)(a_in+8)` 编成从 a_in+0 读 (movs/padda 流式路
径),而索引式 scale 载入 (lda.s16 [p2,dj0]) 与标量 K 载入都保留 +8。
字节级复现:当前源码 clang++ -O2 重编 .o 与部署 cmp 相同——编译器行
为确定,不是陈旧产物。判别式:常数误差 -4.959 = (-8-1)·x[2] −
Σ_{j≠2,j<16} x[j] (bf16 收窄后 -39.25) 精确吻合。

**修法 = 布局 v2 顺应有效锚点**(不改源码锚点,防怪癖翻转):
`[行主序 nibbles (row r @ r·k/2)][8B 洞][scales][pad][K u32 @13832]`。
内核只改一处:k 从 `a_in+ELEM-8` 读。指纹复验:每行 j=0..15 命中各自
标记 → mirror 三场景 bad=0。

### 排错二:shim BD 上限 16 → 单 BD 多元素 B fill

12288×2048 (F=24) 编译失败 `Allocator exhausted available buffer
descriptor IDs`——每元素一个无 tap fill 消耗 F 个 shim BD,通道上限
16。**修法**:vector buffer = F×K_MAX 元素,host 复制 x F 份
(零填充每槽尾),单大 BD 正 stride tap——与 A 路径同构的已验证流形
状。引擎侧代价 ~1.9ms/token (18.6MB/token memcpy),M3b 可接受。
陷阱:`run_test` 原样写入 input_buffers,裸 (K,) x 只覆盖槽 1,槽
2..F 陈旧 → group0 对、其余全错的指纹;op 暴露 `replicate_vector()`
供 forward 与 test 共用。

### 验证与数据

- pytest 4 形状 × 5 iter = **20/20 PASS** (rel 0.07 / abs 0.7)。
- **4 个 main.pdi sha256 逐字节相同**
  (20e56620884828a0bb8f5a200c365a376726398bfde4a3de8b3fc07144ebd05b)
  ——通用性硬前提成立,差异全在 insts.bin ctrl-code。
- 延迟 (µs):qkv ≈ 320,o ≈ 325,gateup ≈ 1080,down ≈ 600
  → 层合计 ≈ 2.3ms → 42 层 ≈ **97ms/token ≈ 10.3 tok/s**
  (对照:四形状各自 PDI + 层序切换 ≈ 205ms/token;grouped 上限
  13.3 tok/s)。gateup 有效权重带宽 12.9 GB/s 已近本机 DDR 顶。
- K=2048 的 A 流有 2/3 padding 带宽浪费 (1.1MB 流 vs 370KB 有效),
  是后续优化空间 (per-K ELEM 会破坏 PDI 一致性,需权衡)。

### 下一步

Rust run-w4layer 切单 CU × 4 ctrl code (PDI 只 load 一次);importer
换 v2 打包器 (槽尾 K + 13840 槽 + 洞);重测四模式期望全 ≈97ms/token;
然后 MHA/swiglu/add 链全 42 层 decode 对拍。

上游求证清单新增:⑨peano -O2 对 int4 流式指针丢字节偏移 (movs/padda
路径) 而索引式载入保留;⑩shim DMA 通道 BD 分配上限 16 无诊断信息
(gateup F=24 直接耗尽)。

## §14b M3b 收官:Rust 单 CU 全链验证 + 调度方法论纠偏 (2026-09-23)

`run-w4ulayer`(main.rs)+ importer `--layout v2`(tools/w4_import.py):
42 层 × 4 形状全部挂在 **1 个 CU** 上(PDI 断言逐字节相同后只 configure
一次),importer 输出 build/w4u(2.75GB,42 层 × 65.5MB,含每层 golden
`golden_L{n:02}_{shape}.bin`)。x BO 共 2 个:xu2048 = 24×6144 槽
(F∈{5,4,24} 读前缀)、xu6144 = 4×6144;golden 前把 x 按槽复制写入
(bare (K,) 只覆盖槽 1 的老坑在 Rust 侧同样存在)。

### 结果 (5 iters, 42 层)

- **GOLDEN 4/4 PASS**(真权重,worst rel err = 0)。
- per-op 84ms;**其余一切调度 ≈ 75ms/token ≈ 13.3-13.4 tok/s**:
  chunk 4/6/8/12/16/24/32/64/168 全落 74.7-78.7ms,逐迭代方差 <1%。
- 全部逐层校验通过(per-layer 210/210;chunk 扫描逐批验末层)。
- v1 的 pipelined 反常 (234ms) 消失 → 确认那是 168 次 cu_mask 变化
  PDI 重载的代价,单 CU 下顺序无关。

### 调度方法论三课 (这次纠偏的核心产出)

1. **包状态轮询 (state-poll drain) 不是完成信号**。两层失效:
   ①`submit` 从不重置包头 state 字段——只有驱动完成时写 COMPLETED,
   所以任何包的**第二次及以后执行**,轮询会读到上一次的陈旧 COMPLETED
   秒回;②即使首跑,状态置 COMPLETED 也早于最终 DMA 写抵达主机内存
   (可见性滞后,睡 2ms 重验全转好 = "transient" 判别)。
   **唯一可信信号 = syncobj 时间线等待**(per-op 模式 126/126 零失败
   为证)。曾据此信过的 50/24/15ms"加速"全是计时假象(迭代 ≥1 从未
   真等设备;末迭代尾巴流出计时窗)。修正后所有模式平坦收敛。
2. **校验目标必须与批边界对齐**:chunk=6 (非 4 倍数) 批界切进层中间,
   验"末层"读到的是下一半层的输出,酷似错序——诊断器(输出对 42 层
   golden 全匹配,报告"layer 10 holds L11")一眼定位是覆盖不是错序。
   逐层诊断器本身值得保留:reorder/stale 给 "L{n}",真损坏给
   "garbage(worst x.x)"。
3. **同 CU 排队执行按序且正确**:168 op 同 CU 深队列,零错序零丢失
   (所有可验层全对)。ERT/fw 对单 CU 命令是串行保序的,引擎可以直接
   依赖。

### 真实瓶颈定位

75ms/token ≈ 65.5MB/层 ÷ **39GB/s 槽流**(= 每列 ~5GB/s,与单 op
gateup 39GB/s 一致)。这不是硬墙:M2 GEMM 夹具曾以不同 DMA 模式跑到
**54GB/s 汇聚**。两条提速路线(优先级排序):
1. **消 K padding**:K=2048 形状槽流 3 倍于有效字节。per-K ELEM
   (4616/13840) 需 2 PDI 2 CU(fifo 元素几何编译进 PDI,单 PDI 动态
   ELEM 死路已验证推理)→ 槽流 2.75GB→1.11GB,若仍 39GB/s 则
   **~28ms/token ≈ 35 tok/s**。
2. DMA 模式调优向 54GB/s 逼近(fifo depth、BD 尺寸、多 tile 迭代维)。

### 本次新增钉死事实

- submit 不重置 state(见上);syncobj timeline 点按序完成,可当批屏障。
- SHMEM BO 的 FromDevice sync 不能等在途 DMA(只做 cache 失效),
  校验前必须已有 syncobj 屏障。
- 2.75GB 常驻 168 BO 无压力(91GB RAM);configure_cus 单 PDI 单 CU
  与 4 PDI 4 CU 同稳。

上游求证清单不变(⑨⑩待发)。M3b 剩:MHA d=128 fixture + swiglu + add
链全 42 层 decode 对拍 CPU 4.55 tok/s 基线。

## §15 M3b 收官：全 42 层 decode 链 E2E 对拍 PASS（2026-09-23）

`run-decode`：w4gemvu 单 CU 投影（168 op）+ Rust CPU 胶水，一步真实
decode（pos=100, cache_seq=1024），对拍 torch 参考（导出器
tools/decode_export.py，f32 计算 + 每 op 边界一次 bf16 舍入 + w4 反量化
权重，与 R3 同哲学）。

**结果：PASS —— final hidden rms 误差 0.0234 = golden 自身 rms(4.562)
的 0.5%**；最大绝对偏差 0.5 落在 golden=109 的元素上（0.5%），top-3
偏差全部 ~0.4-0.6% 相对 = 纯 bf16 栈噪声（R3 全 IRON 栈也是这个量级）。
逐层看：L0-L10 全部在容差内，L11 起 >26/2048 元素出 1-2% 带（渐变
漂移，非 bug 特征——真 bug（rope 错/GQA 映射错）会在 L0 就大规模爆）。

**性能：steady 102.9ms/token ≈ 9.7 tok/s**（首步 114ms 带校验），
= CPU 全软栈 4.55 tok/s 的 2.1×。分解（对照 §14b 数据）：
- 75ms：NPU 投影地板（39GB/s 槽流，§14b 已定位）；
- ~9ms：168×55µs submit+syncobj 串行往返（per-op 模式代价，§14b per-op
  84ms 同源）；
- ~17ms：Rust 标量 attention（42 层×16 头×2×101×128 MAC ≈ 35 MFLOP，
  标量 ~2GFLOP/s）+ 复制写 x 槽（~13MB）+ 胶水。
→ M4 把 attention 换 NPU 通用 decode-MHA + 消 K padding（§14b 路线①
28ms 地板）后，35 tok/s 路线畅通。

### 踩坑

1. **导出器 x0.bin bug**：x 在层循环里被消费后才写 x0.bin → 文件内容
   是"final hidden 输入"而非"decode 步输入"。goldens 在循环内先算完
   （正确），只有 x0 错。修复 = 循环前 clone；只重生成 x0.bin 即可，
   无需重导（42 层量化分钟级）。
2. **FromDevice sync 的 EINVAL 是常态**：gemv 读输出处 `.ok()?` 直接
   把整个 step 判失败。SHMEM 本身 cache 一致，方向 1 ioctl 无 debug BO
   必 EINVAL（M1 老坑在新代码里复发）——读路径一律 best-effort。
3. 借鉴既有纪律：per-layer 校验用 0.01+0.02|g| 容差（裸 rel 在近零
   golden 上必然爆炸，max rel 6835 全是这类）；最终判定用
   **rms/golden_rms 相对值**。

### 认知

- CPU 胶水（rms/rope/GQA/swiglu/add）逐位镜像参考数学是可行且足够的
  对拍策略：只要 op 边界舍入一致，误差就是纯 NPU gemv 求和序噪声，
  42 层后仍稳定在 0.5% 量级、不放大（残差流的自愈性，R3 logits 同观）。
- 引擎骨架就此闭合：加载→常驻 BO→逐层 (norm→gemv→glue)→final norm，
  全在 Rust，无 XRT/无 Python。M4 = 换掉 CPU attention + 提速两路线。

## §16 M4a：flowkv_decode 扩展到 MiniCPM5 形状——两个连环根因与 dump-probe 方法论（2026-09-23，IRON 62475f6）

**目标**：上游 flowkv_decode（llama-1B 4Q×64d GQA）扩到 MiniCPM5
（16Q/2KV/d128，group8，cache≤1024）+ 运行时 S（守护进程单 PDI 全
 decode 步复用）。板上结果：**35 passed / 5 skipped**（skip = 8col
 512s 形状放置失败，HEAD 同败，纯几何：8 列要 8 个 shim DMA tile）。

### 根因 A：头 S==0 语义缺省 → 全死行 → NaN

不传 seq_len_cur 的旧调用（上游主测试）头全零 → 内核 S=0 → 所有行
判死 → l=0 → `O = 0 · inv(0) = NaN`；且哨兵 bf16(-1.0039e30) 与
m_old f32(-1e30) 差非零 → F=0 而非 1，死行并不代数中性。
**修复**：头 S==0 语义为"满编译容量"——rope_q 新增第 4 参
seq_len_cap（design 编译期常量传入）。

### 根因 B（真根因，A 修完后露出）：peano 流式指针锚点丢常量偏移

给 Q 元素**前面**加 16 元素头后，peano -O2 把 rope 循环 angles 流式
指针基址的 +16 常量偏移丢掉（x1/x2/q_head 基址保留）→ 内核实际
angles 基址 = q_in + nqh·hd = 上游无头公式。证据 = 反解内核实际用的
每对 (c,s)：j≥8 对恰为 golden 对 j−8，j<8 对 |c|>1 垃圾（读到 Q 尾部
数据）。与 §13 w4gemvu "流式权重指针丢 +8 字节"同类——**peano 的流式
指针锚点若不再与它认识的基址形式逐字一致，常量偏移会被静默丢弃**。
**修法（§13 同款"顺应锚点"）**：头移元素尾部，布局
`[Q_heads | angles | hdr(16)]`，rope 的 angles/q_head 基址恢复与上游
逐字相同；头只走标量尾读。探针复验 rotq vs golden = 0.0137（bf16 噪声）。

### dump-probe 方法论（两连升级，判别力极强）

- **v1**：value_accum 暂存 inter 包 F/C/l + V 行0，normalize 写入
  output 替代 O → 一次上板拿到 score→value 边界全状态。
- **v2**：design 把 inter 元素 +2hd+16+hd，score_chunk 在包尾写
  `[k_row0 | rotq_h0 | qhdr | raw_q_h0]` → score tile 内部状态跨 tile
  可见。反解出 rotq 用的角度 ⟹ 定位锚点差 16。
- **O drain 归属语义**（避免误判）：每批 drain 写各自 kv 区域，
  `output[c*256:(c+1)*256]` 是**对应批次**的 dump（batch0 的 dump 在
  batch1 处理前已落盘）；value tile 的 cur_chunk_base 恒 0 → 暂存条件
  恒真 → 暂存的是最后一个 chunk。曾因误归属指控"第二批 DMA 卡旧地址"，
  MLIR 验证 fill 偏移全对——**先核对 dump 归属再立假设**。

### 排除清单（省未来重查）

statics 尺寸/放置（MAX 缩回 4/64 仍败）、dot 动态循环 spill（展开仍
败→但展开式+标量哨兵作为最终形态保留）、DMA 第二 task group 用旧地址
（fill 偏移全对）、no_rope/dot32/k_stuck/no_scale/KV swap/sin flip/
partial-S/sharpness/l_miss/角度变体——全部离线指纹不匹配。

### 遗产（最终形态保留）

- 死行 -1e30 哨兵 + exp2 下溢 → 在线 softmax 更新代数中性（C=1,F=0,
  l 不变），死 chunk 无需特判下游；标量选择、不进向量路径。
- dot 的 mac 偏移保持编译期常量全展开（动态偏移循环形态会 spill 过
  0x400 栈进 K fifo）。
- NOCPP 下 math.h 不可用 → inv_sqrt(d) 用三元常量（精确 f32 位）。

### 下一步

M4a 集成：Rust run-decode 把 CPU attention（~17ms/token 标量）换成
本 op（导出 d=128 fixture；Q 头/angles 打包进 Rust——头在尾部布局；
KV cache 交错布局 BO，k/v 写 cache 暂留宿主侧）→ 预期 attention 归零、
token 时间 102.9→~86ms。

## 17. M4a 收官：flowkv E2E 集成 + 首 exec O 读竞态（2026-09-23）

### E2E 结果（run-decode，w4gemvu CU0 + flowkv CU1）

- **速度回退**：NPU attention 全链 241–256ms/token（~4 tok/s），反而
  慢于 CPU attention 路径的 102.9ms——§16 预估的 86ms 没有兑现。原因
  二：flowkv 单 exec ~2.6ms × 42 层 ≈ 109ms（kernel 逐 pos 标量 exp2，
  ~3 GF/s 延迟受限）+ CU0(w4gemvu)↔CU1(flowkv) 逐 op 交替触发 ~650µs
  PDI 重载 × ~84 次/token ≈ 55ms。这是 M5 性能阶段的第一靶子。
- final hidden vs golden rms 4.3–4.4%（golden rms 的百分比），A/B
  （NPU vs CPU attention 同链）rms 差 8–10% —— 落在 bf16 score +
  exp2-arg 量化噪声底（§16 的 fk_head6.py 离线模型同量级），非实现错误。
- 提交：IRON ce3c674（flowkv 头尾布局 + 哨兵 + 测试），xnpu 19d1704。

### 首 exec O 读竞态（本节核心遗产）

**症状**：PDI 加载后的第一次 exec（fkprobe iter-0 / decode layer-0），
O 输出部分为零/错值（head-0 lstsq scale 0.75 一类）；第二次起干净。
decode E2E 表现为 layer 0 起就发散。

**机制**：O drain（设备 DMA 写回宿主 SHMEM）可滞后于完成 syncobj——
syncobj 只承诺固件侧命令完成，不承诺宿主可见。首次 exec 后续迭代
之所以干净，是前面的读路径顺带做了 clflush。

**修法（照抄 XRT）**：每次 submit 前对输出 BO 做
`sync(SyncDirection::ToDevice, 0, size)`（方向无关的 clflush 语义，
§1 就发现 direction-1 需要 debug BO，方向 0 只 flush）。fkprobe 与
run-decode flowkv 路径各加一行，3/3 复验干净，E2E 对拍恢复。

**方法论**：syncobj timeline 是唯一可信的"完成"信号；state-poll
（包状态字段）会更早返回（w4ulayer §14b 已见）；而"完成"≠"宿主读
得到"——首次 DMA 写回需要显式 flush 屏障。M5 度量框架的 Event 语义
按此定义：t_complete 取 syncobj wait 返回，读数据前另记 flush 成本。

### 遗留（进 M5）

- 每 op submit+syncobj 往返 ~55µs ×168 op ≈ 9ms/token 纯调度开销；
  CU 切换 PDI 重载 ~650µs（§14b）。这是 M5a 度量框架的第一批锚点。
- flowkv 2.6ms/op @ S=101：per-position 标量 exp2 循环 → ~3 GF/s，
  latency-bound，是 M5 优化的头号候选。
