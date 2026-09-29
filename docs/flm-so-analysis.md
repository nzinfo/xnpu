# FLM 闭源引擎 .so 静态逆向分析（NPU 教科书素材）

编号 **F1, F2, …**（本文件事实编号，供 perf-lab 笔记与教科书互引）。
方法与纪律对齐 perf-lab.md：设计思路/证据/推断全部留痕；**实锤**（符号名、反汇编
地址、strings、开源头文件行号）与**推断**（算术闭合、语义解读）严格分开标注。

- 分析对象：`~/qwen/refs/FastFlowLM/src/lib/hrx/*.so`（27 个，2026-09 快照）
- 工具：`nm -C` / `objdump -d/-s -C` / `readelf -d` / `strings`（纯离线，未上板、
  未跑 flm/pytest——板子由主会话占用，本任务书明令禁止）
- 开源锚点：`~/qwen/refs/FastFlowLM/src/include/`（hrx_cpp.hpp、npu_utils/、modules/）
  与已知的 P22/P23/P26b trace 事实
- 地址引用约定：均为各 .so 内的 vaddr（`objdump -d` 第一列）

---

## 1. 总体图景：27 个 .so 的三层分工（F1–F5）

### F1【实锤】库分层由 DT_NEEDED 一刀切开

`readelf -d` 全 27 个 .so 的 NEEDED 普查结果分三类：

| 类 | 成员 | NEEDED 特征 |
|---|---|---|
| **模型引擎**（22 个） | libhunyuan_npu / libllama_npu / libqwen2_npu / libqwen3_npu / libqwen3vl_npu(+_flash) / libqwen3_5vl/5_omni/6_moe / libgemma_npu / libgemma4_12b / libgemma4e / libgemma_embedding / libgemma_text / libphi4_npu / libgpt_oss_npu / liblfm2_npu / libnanbeige_npu / libwhisper_npu | **libhrx.so.0** + libstdc++（部分 + libmvec/libgomp） |
| **算子生成器库**（4 个） | **libmha.so、libgemm.so、libdequant.so、libq4_npu_eXpress.so** | **只有 libstdc++/libgcc/libc，不链 libhrx** |
| 新旧双轨 | libdequant_new.so（链 libhrx+libgomp）vs libdequant.so | 同名算子两代共存 |

结论（教科书要点）：**闭源边界不是"一个黑盒"，而是"指令生成器（纯 host 计算，
产出 TXN 字节流）+ 运行时（libhrx，设备 I/O）"两层**。libmha/libgemm/libdequant
连设备句柄都拿不到——它们只消费/产出 `npu_sequence`。引擎 .so 再把生成器
**静态链进自己**（F5）。

### F2【实锤】引擎 .so 内嵌了开源运行时代码（W 弱符号群）

所有链 libhrx 的 .so 同时以 **W（weak）符号导出整份 hrx_cpp/npu_utils 实现**：
`npu_sequence::cmds2seq/dump_patch_table/clear_cmds`、`npu_app::_setup_kernel`、
`npu_cmd_*::dump_cmd/print_cmd/get_op_lines`、`hrx::run npu_app::create_run<...>`、
`hrx::Runtime::ensure` 等（liblm_head.so 中 `nm -C | grep -c "W npu_|W hrx"` = 47）。
即 `src/include/hrx_cpp/hrx_cpp.hpp`（header-only，571 行）与 npu_utils 头被**编译进
每个闭源 .so**。开源侧 hrx_cpp.hpp:~120 的 `build_or_get_executable`/`BufCoh` 就是
这些 W 符号的源码。

### F3【实锤】构建出处泄露（strings）

liblm_head.so 内嵌 RPATH 字符串：

```
$ORIGIN:$ORIGIN/../lib:/scratch/zani/github-runners/xsjstrnuc104-so-flm/_work/
FastFlowLM_IRON/FastFlowLM_IRON/hrx-integration/.hrx-release/
hrx-amdxdna-2026.09.02-amdxdna-hal-native-rel-6867cba-linux-x86_64/lib
```

即引擎与 **hrx-amdxdna-2026.09.02（amdxdna-hal-native-rel-6867cba）** 同仓联编
（CI runner 路径），印证 create_new_model.md 所述 "comes from the kernel/IRON
project"。教科书可引用为"闭源内核与运行时同步发版"的证据。

### F4【实锤】引擎自带 TXN 落盘开关 FLM_DUMP_TXN

libhunyuan_npu.so 与 liblm_head.so 的 strings：

```
FLM_DUMP_TXN
/tmp/flm_dump
/txn_%02d.bin
/patch_%02d.bin
[FLM_DUMP_TXN] kernel %d '%s': txn=%zu words patch=%zu triples
```

设 `FLM_DUMP_TXN` 环境变量即可让引擎把每个 kernel 的 TXN（ctrl-code）与 host
patch 表直接写成文件。**这修正了 P23 的方法论前提**（当时只能靠 ioctl 拦截
xdump2 抓 ctrl BO）；也证明 TXN 与 patch 表是引擎的一等公民数据结构
（`npu_app::update_ctrl_seq()` 符号另证 ctrl 序列可原位更新）。

### F5【实锤】libhunyuan_npu.so 静态链入了 gemm/lm_head/mha 生成器

nm 显示 libhunyuan_npu.so 内同时存在
`Gemm::Impl::generate_seq`、`Gemm::generate_seq`（两个重载）、
`hunyuan_npu_sequence::gen_lm_head_seq`、`gen_mha_engine_seq`、
`gen_dequant_mm_512`。即模型引擎把公共算子库**复制链接**进自身
（符号 T 存在于本 .so，而非从 libgemm.so 动态解析——后者也在发行目录里给
其它模型当共享库用）。

---

## 2. 公共指令层：npu_sequence / npu_cmd DSL（开源锚点）

这一层完全开源（`src/include/npu_utils/`），闭源 .so 内的 W 符号与之逐一对齐，
因此**指令 wire format 不需要逆向**——逆向的价值在"生成器怎么用这套 DSL"。

- `npu_instr_utils.hpp`：`npu_sequence` 类。TXN 头 4 字（:258-265 解析、:324-326
  生成）：w0=(major)|(minor<<8)|(dev_gen<<16)|(rows<<24)，w1=(cols)|(mem_tile_rows<<8)，
  w2=指令数，w3=总字节数。NPU2 profile（:241-243）major=0/minor=1/dev_gen=4
  → w0=0x06040100 由 rows=6<<24 合成；`instruction_lines = seq[3]/4`（:265）。
  **与 P23 抓到的层 blob 头 [0x06040100, 264, 1566, 51532] 逐字段闭合**
  （264 = 8 cols | 32 mem_tile_rows<<8）。【实锤：头文件 + P23 trace 对拍】
- `npu_tiles` 枚举：IT0-7=0-7（shim 行），MT=0x10-，CT00-37=0x20-0x57
  （compute tile，编码 0x20 + row*0x10 + col，col∈0-7/row∈0-3 的解读见 F13）。
- `npu_cmd_write_dma.hpp`：BLOCKWRITE BD 与队列推。`bd_id_shift=5`（:15）→
  shim BD 空间 0x1D000+bd_id*0x20；`next_bd_id_shift=27`（:26）BD 链；
  AxCACHE 在 BD 第 5 字 <<24（no_cache=0 / normal=0x02 / aggressive=0x0e，
  P23-10 已对拍）。
- `npu_cmd_ddr.hpp`：DDR_PATCH（op 0x81，12 字），运行时把 arg 对应的设备地址
  写进 BD 的 addr 字段；`dump_patch_table()`（npu_instr_utils.hpp:662-690）产出
  host patch 三元组 `(BD addr_low 所在字偏移, arg_idx, arg_offset)`——**取代
  aiebu relocation，host 侧补丁表由引擎自己生成**（F4 的 patch_%02d.bin）。
- `npu_utils_hrx.hpp`：`npu_app::_setup_kernel` 用 `ctrl_seq->dump()` 的 u32 数组
  直接建 HRX "direct executable"（entry point 恒为 `MLIR_AIE`，F8），**没有独立
  的 assembler 阶段**——序列生成器输出的就是可执行事务流。
- `hrx_cpp.hpp`：`BufCoh` 脏位追踪（host_dirty/dev_dirty）门控
  `hrx_buffer_flush_range`/`invalidate_range`；`run/runlist` 封装
  `hrx_stream_dispatch/flush/wait`。

---

## 3. libhunyuan_npu.so —— hy-mt2 1.8B 的 decode/prefill 引擎（F6–F16）

### 3.1 角色与调用关系（F6）

【实锤：nm + 交叉引用（objdump call 目标统计）】两条独立的 ctrl 生成路径：

- **decode**：`hunyuan_npu::Impl::forward`@0x17ad0 → `_select_slot(i)` →
  bytes 拷贝 → `sync_to_device`（BufCoh 门控 `hrx_buffer_flush_range`）→
  `hrx_stream_dispatch/flush/wait`；slot 惰性构建：
  `Impl::_build_slot(i,j)`@0x16ce0 = `hunyuan_npu_sequence::gen_layer_rtp_seq`
  + `gen_layer_seq`@0x30c70 + `npu_app::_setup_kernel` ×2 +
  `_set_rope_rms_weights`。slot 机器：`_select_slot` / `decode_slot_t` /
  `_invalidate_slots` / `_sync_slot_rope_weights` / `set_context_length` /
  `get_current_context_length`（全部 nm 实锤）。
- **prefill**：`hunyuan_npu::Impl::prefill(std::vector<int>&, void*)`@0x19250 →
  `hunyuan_prefill_context` / `hunyuan_attn_block_prefill_context::forward`@0x28fa0
  （唯一调用 `gen_mha_engine_seq`@0x31f20 的函数）+ `gen_dequant_mm_512`@0x30f80
  ×7（Impl::prefill 调 3 次 + attn_block forward 调 4 次，call-site 统计实锤）。
  **7 个 dequant-GEMM + 1 个 MHA = 8 op/层，与 P5 实测 prefill 8 op/层精确吻合**
  （qkv/o/gate/up/down 五个投影 + 2 个 q/k-norm 相关 + 1 attention 的组合）。

即：**decode 用预生成的整层融合序列（slot），prefill 用算子级生成器现场拼装**。
这与 P23 观测的 decode 每 token 33 个 51532B 层 blob（slot 复用）vs prefill
256 条单 op（尺寸 2400/7248/12176/20176B 波动）完全对上。

### 3.2 导出符号清单与命名解读（F7）

T 符号约 93 个（P22 已数过），关键簇：

| 簇 | 符号（nm -C 摘录） | 解读 |
|---|---|---|
| 公开 API | `hunyuan_npu::forward/prefill/load_weights/set_context_length/clear_context/get_current_context_length` | causal_lm.hpp 的 PIMPL 后端 |
| 序列生成器 | `hunyuan_npu_sequence::gen_layer_seq / gen_layer_rtp_seq / gen_dequant_mm_512 / gen_mha_engine_seq / gen_lm_head_seq / set_max_length` | 本 .so 的核心：按形状生成 TXN |
| 生成器原语 | `_send_x`@0x2f050、`_send_rms_weights`、`_send_rope_rms_weights`、`_move_weights(npu_sequence*, u64, u64, weight_desc_t const&)`@0x2f4a0、`_receive_kv_cache(seq, int)`@0x30470、`_move_kv_cache(seq, u64)`@0x30a00 | gen_layer_seq 的子例程（私有 but 未 strip） |
| tile 表（数据符号） | `hunyuan_npu_sequence::proj_tiles / attn_qk_tiles / attn_kv_tiles / mvm_tiles` | 列分工静态表（F13） |
| 权重装载 | `hunyuan_desc::load_layer_weights(int, Q4NX&, buffer<u8>&, buffer<bf16>&, buffer<bf16>&)`（含完整 assert 字符串，见 F9） | 从 q4nx 拆 12 类层权重 |
| 内嵌库 | `Gemm::*`、`hrx::*`、`npu_sequence::*`（W） | F2/F5 |

### 3.3 反汇编解剖（代表性函数）

#### F8【实锤】`gen_layer_seq`@0x30c70 —— decode 整层序列的拓扑

call 序列（objdump 目标统计）：

```
clear_cmds
→ _send_x                    (激活 X 进分发列)
→ _send_rms_weights          (RMSNorm 权重)
→ _send_rope_rms_weights     (rope + qk-norm 权重)
→ [内联] npu_dma_memcpy_nd(tile=0xa, bd_1, packet=-1)   (小分发传输)
→ _receive_kv_cache(seq, i)  @0x30470：内部调 _move_weights + 3× npu_dma_wait
→ _move_kv_cache(seq, u64)   @0x30a00
→ _move_weights ×3           (weight_desc_t @+0x210 / +0x318 / +0x3c8；#2 dst=field*2)
→ npu_dma_wait
→ cmds2seq                   (DSL → TXN 字节流)
```

要点：
- **KV cache 的装载复用权重搬运器**——`_receive_kv_cache` 的实现就是
  `_move_weights` + 3 个 dma_wait（0x308f4/0x30903/0x30912/0x30921 call 实锤）。
  KV cache 在 ctrl 流里只是"又一种权重"。
- 三次 `_move_weights` 对应三组 weight_desc_t；desc#2 的目的 tile 字段 ×2
  （两套目的 FIFO）。
- 序列以 `cmds2seq()` 收尾 = npu_instr_utils.hpp:317-327 的 TXN 组包
  （`instruction_lines = 4 + Σ get_op_lines()`）。

#### F9【实锤】`_move_weights`@0x2f4a0 —— 权重流 BD 生成器（trace 中 76×… 的来源）

- 入口 `movzbl 0x20(%r8),%ecx` + `cmp $8` 分支：**weight_desc_t+0x20 是量化
  格式字节**（0-8；`flm_q40` 枚举名出现在 assert 字符串
  `get_quantization_byte_size((size_t)desc.DK * desc.D, flm_q40)`）。
- 格式相关位运算收敛到 per-tile 字节数常量：**{0x10, 0x11, 0x12} << 8 =
  {4096, 4352, 4608} 字节**（另有 format 8 → 0x940 路径）。
  解读：0x10/0x11/0x12 = **每 32 参数组的字节数 {16,17,18}**（q4 = 16B nibble +
  2B scale = 18B = 0x12，0.5625 B/param，与 P26b 精确一致）；<<8 = ×256 组 =
  一个 q4nx tile（32 行 × 256 列 = 256 组）的字节数。**4608 B/tile 与 P2 逆向的
  q4nx 文件 tile 尺寸逐字节一致**【实锤：常量 + P2/P26b 对拍；q2/q3→16/17 的
  具体映射为推断】。
- 函数体两个 `npu_dma_memcpy_nd` 调用点（0x2f712 / 0x2f85a）与多重回边循环
  （0x2f8c9/0x2f925/0x977 等）= 对每组目的 tile 循环发 BD + DDR patch。
- **BD 长度选择与 P23 trace 的算术闭合**【推断（强），三方对拍】：
  每列观测 68×18432w + 8×55296w。换算：18432w=73,728B=**16 tiles**；
  55296w=221,184B=**48 tiles**。每层权重按矩阵切：
  qkv(3072×2048)=48 BD、o(2048×2048)=32、gateup(12288×2048)=192、
  down(2048×6144)=96（全部按 16-tile BD 计），四矩阵总 368 个 16-tile 等效块，
  4 列均分 92/列 = 68+24 → **down 的 24 块合并成 8 个 48-tile 大 BD**。
  即三组 desc = (qkv+o 合并 20×18432w/列)、(gateup 48×18432w/列，双目的)、
  (down 8×55296w/列)；列合计 68×18432w + 8×55296w ✓，DDR_PATCH 步长
  0x12000=73,728B=一个 16-tile BD ✓，arg1 patch 76×4=304 条 ✓ 全部与
  P23-8/P26b 逐项闭合。

#### F10【实锤】`gen_dequant_mm_512`@0x30f80 —— prefill 的 512 块 dequant-GEMM 生成器

- 签名（demangle）：`(npu_sequence*, uint M, uint K, uint N, u64, u64, int,
  uint, uint, uint)`。
- prologue 参数守卫（0x30fe5-0x31092）：`M & 0x1ff`、`N & 0x1ff`（**M、N 必须是
  512 的倍数**）、`K & 0x7f`（**K 是 128 的倍数**）、`cmp $0x3fff`（**M ≤ 16384**）、
  除法整除性检查——不满足走 .cold 抛错。
- 随即计算 `M>>9`、`M>>8`、`(K>>7)` 等分块数（0x31046-0x3106d）。
- **RTP 广播循环**（0x31123-0x31205）：以静态表 `::IT@0x4b160 = {2,3,4,5}`
  （.rodata 实测 02 03 04 05）为 tile 列表，对 row=0x20/0x40/0x60（CT2x-CT6x）
  三行 × 4 列与 row=0x30/0x50/0x70 两组，向 **0xb200 / 0xe080 / 0x3080** 三个
  RTP 地址各 `rtp_write` 一轮——即 prefill GEMM 的形状参数走 RTP（运行时参数
  寄存器）而非重编指令。assert `elems % chunk_elems == 0`、
  `chunk_size / m / cores > 0`、`Heads % num_cu == 0`（strings）同源。
- 内含 6 个 `npu_dma_memcpy_nd` 调用点 + 多层循环（权重/X/Y 三个方向的分块流）。

### 3.4 与开源 npu_cmd_*.hpp 编码器的对拍

- TXN 头：`cmds2seq()`（W 符号，源码 npu_instr_utils.hpp:317-327）→ P23 blob
  头逐字段闭合（见 §2）。
- 每条 `npu_dma_memcpy_nd` = BLOCKWRITE BD fill + DDR_PATCH +（S2MM 时）
  issue_token MASKWRITE + 队列推 WRITE 的四拍组合（npu_cmd_write_dma.hpp/
  issue_token.hpp 源码 + P23 的 316+316+316+309+309 计数对拍）。
- `dump_patch_table()`（npu_instr_utils.hpp:662-690）↔ F4 的 patch_%02d.bin
  落盘 ↔ amdxdna host-patch ioctl 路径（P23-5 的 ERT wrapper）。
- AxCACHE：libmha.so `memcpy_nd` clone 的第 5 参（cache_flag_t）在 BD w[9]<<24
  ——P23-10 的 0x0e/0x02 观测即此参数的取值。

### 3.5 与 xclbin 图的关系（F11–F13）

**F11【实锤】strings 直接给出图绑定**：libhunyuan_npu.so 内有
`layer.xclbin`、`fused_prefill.xclbin`、`lm_head.xclbin` 三个文件名常量 +
`xclbins`/`xclbin_valid`/`Max number of xclbins reached`（npu_xclbin_manager，
开源侧 max 16）。结合 F6 调用关系：

- **layer.xclbin = decode 整层融合图**（slot 的 gen_layer_seq 序列跑在它上面；
  P23 hwctx=1 上 32×51532B blob）；
- **fused_prefill.xclbin = prefill 8-op 图**（gen_dequant_mm_512 + MHA 的宿主；
  P23 hwctx=2 上 256 条单 op）；
- **lm_head.xclbin 由 liblm_head.so 使用**（见 §5）。

另：`model.layers.%d.self_attn.{q,k,v,o}_proj.weight`、
`...mlp.{gate,up,down}_proj.weight`、`...{input_layernorm,post_attention_
layernorm,k_norm,q_norm}.weight`、`num_hidden_layers` —— `load_layer_weights`
按 HF 命名从 Q4NX manifest 取张量（Q4NX::get_tensor 的 key 约定与 safetensors
一致）。

**F12【实锤】decode 的 assert 链暴露 layer_desc 布局**：

```
desc.layer_desc.attn_v.offset == desc.layer_desc.attn_k.offset + get_quantization_byte_size(DK*D, flm_q40)
desc.layer_desc.attn_q.offset == desc.layer_desc.attn_v.offset + get_quantization_byte_size(DK*D, flm_q40)
```

即权重 BO 内 K→V→Q 顺序毗邻（KV 在前，Q 随后）——KV cache 复用权重流布局的
又一证据（与 F8 的 `_receive_kv_cache`→`_move_weights` 呼应）。

**F13【实锤】列分工静态表（.rodata）与 P23 trace 列分工互证**：

| 表 | 地址 | 值（npu_tiles） | 含义 |
|---|---|---|---|
| `proj_tiles` | 0x4b240 | {CT00,CT10,CT20,CT30; CT01,CT11,CT21,CT31; CT06,CT16,CT26,CT36; CT07,CT17,CT27,CT37} | **16 个算力 tile = 列 {0,1,6,7} × 行 {0,1,2,3}**（0x20+row*0x10+col 编码） |
| `attn_kv_tiles` | 0x4b280 | {CT13,CT33,CT14,CT34} | KV 路由：列 {3,4} × 行 {1,3} |
| `attn_qk_tiles` | 0x4b290 | {CT03,CT23,CT04,CT24} | QK：列 {3,4} × 行 {0,2} |
| 列号表 | 0x4b2a0 | {0,1,6,7} | proj 用的 4 个 shim 列 |

与 P23-8 的列分工观测（c0/c1/c6/c7 大流量算力列、c3/c4 小传输 KV 路由列、
c2 分发列）**逐列一致**。NPU2 的 8 列里 FLM 用 0/1/6/7 跑投影、3/4 跑 attention
路由、2 分发——一张表同时回答"哪些列干什么"。

### 3.6 教科书要点（libhunyuan_npu.so）

1. **"层引擎"的本体是一台 host 侧序列生成器**：forward() 只是查 slot + flush +
   dispatch；全部智能在 gen_layer_seq 及其子例程里（F6/F8）。
2. **权重、KV cache、激活在 ctrl 流里是同一种东西**（BD + DDR patch 的四拍
   重填），区别只在 weight_desc_t 的格式字节与长度（F8/F9/F12）。
3. **BD 粒度按矩阵形状选择**：K=2048 的矩阵 16-tile/BD，K=6144 的 down 用
   48-tile/BD——BD 长度是生成器的自由参数，不是硬件定值（F9）。
4. **decode 与 prefill 是两套生成器两套图**：slot 预生成（0 host 插手）vs
   算子级现场拼装（8 op/层）（F6 + P5/P23 对拍）。
5. **形状参数走 RTP**（0xb200/0xe080/0x3080 广播到 CT2x-CT6x），使一张
   fused_prefill 图服务任意 M%512==0 的批次（F10）。

---

## 4. libmha.so —— 多模式 attention 序列生成器（F14–F16）

### 4.1 角色与调用关系

不链 libhrx（F1）：**纯生成器**，输入 (npu_sequence*, j1, j2, pos, bool, int)，
输出指令；被模型引擎（hy 里是 gen_mha_engine_seq 的静态副本）调用。仅 6 个
`_gen_mha_seq_*` T 符号 + `MHA::get_chunk_size`（**与 P22 所记 19 个 T 符号
含 d128_q4_1cu 不同——P22 数的可能是 .dll 版本；此处按当前 .so 实测 6 个
模式变体，记为差异**）。

### 4.2 导出符号清单

```
MHA::Impl::_gen_mha_seq_d64_q4 / d128_q2 / d128_q3 / d128_q4 / d256_q2 / d256_q4
MHA::Impl::_parameter_check(uint, uint, bool, int)
MHA::Impl::IT = {0..7}          (数据符号 @0x128c0，IT0-IT7 全 shim 行)
MHA::get_chunk_size / MHA::MHA(...) / ~MHA
```

六模式 = (head_dim ∈ {64,128,256}) × (量化 ∈ {q2,q3,q4} 的合法组合)。
assert 字符串：`L_begin should be multiple of chunk size!` /
`L_end should be multiple of chunk size!`。

### 4.3 反汇编解剖

**F14【实锤】dispatcher@0xbfe0 + 跳转表@0x12280**：mode 0-5 →
{d64_q4, d128_q2, d128_q3, d128_q4, d256_q2, d256_q4}。Impl 对象 40 字节布局：
+0 mode、+4/+8/+0xc 三个 RTP 地址
（d128: 0x3c00/0x6200/0x9080；d64: 0xe800/0x3080/0xb080；d256: 0xf080/0x2e80/0xf0a0）、
+0x10 head_dim（CSWTCH.565@0x12300 = {64,128,128,128,256,256}）、
+0x14 L（序列长度上限）、+0x18 L×量化字节数{4,2,3,4,2,4}、
+0x20 {16 或 32}（heads 相关）、+0x24 chunk_size {128 或 256}。

**F15【实锤】`_gen_mha_seq_d128_q4`@0x9220 的三层循环**：

1. `_parameter_check(j1, j2, causal, pos)`@0x4d70（j1<j2 且 j1/j2 ≡0 mod
   chunk 128，即 assert 字符串的运行时形态）；
2. `clear_cmds` 后 **32-tile RTP 广播**（0x9300-0x935b）：tile = 0x20|row 起步
   ×8 列 × {0x20..0x50} 共 32 个 compute tile，每 tile 3 条 `rtp_write`
   （src = Impl+4/+8/+0xc 的三个 RTP 地址，dst = 形状值）——attention 的
   (j1, j2, pos) 每次生成都广播到全阵列；
3. **外层 chunk 循环** `ceil((j2-j1)/1024)`、内层 2 次子循环：每轮 5 处
   `npu_dma_memcpy_nd`（0x9666/0x97f5/0x9959/0x9afb/0x9c2e：Q 块、K/V 块、
   score/out 等）+ 2 处 `npu_dma_wait`（0x9ce2-0x9d02 循环 = 对 32-tile 表
   逐个 wait）；尾跳 `cmds2seq`（0x9d50）。

MHA 的 chunk=128/1024 分块与 P5 prefill 中 MHA op 的 ~300-900µs 量级一致；
prefill 语境下 j1/j2 以 chunk 为单位滑动。

### 4.4 与开源编码器对拍 / 4.5 与 xclbin 关系

同 §3.4（共用 npu_sequence DSL；`cache_flag_t` 第 5 参实传）。RTP 地址
（0x3c00/0x6200/0x9080…）是 fused_prefill.xclbin 图内 buffer 的偏移——
即 MHA 图把 (j1, j2, pos, L) 做成 RTP 端口，同一图服务所有 chunk。

### 4.6 教科书要点（libmha.so）

1. **一个 .so = 一个算子族 ×（head_dim × 量化格式）矩阵**，dispatcher 查表
   分发——"模式"只是参数组合，不是不同内核（F14）。
2. **滑动窗口 attention = 三层循环的指令展开**：tile 广播 → chunk 循环 →
   5×memcpy+2×wait 的循环体；序列长度只影响循环次数与 RTP 值（F15）。
3. 生成器与设备完全解耦（无 hrx 依赖）——**可在无 NPU 的机器上离线生成
   attention TXN 做单元测试**（教科书工程实践点）。

---

## 5. liblm_head.so —— 独立的分类头引擎（F16–F18）

### 5.1 角色与调用关系

链 libhrx（自带设备 I/O）：`LMHead(LM_Config, npu_xclbin_manager*)` →
`load_weights(Q4NX&)`（lm_head.weight）→ 每 token `execute()` → `wait()` →
`x_exposed()`。开源接口 modules/lm_head.hpp:29-44（PIMPL）。
kernel 参数签名实锤：`npu_app::create_run<buffer<bf16>&, buffer<u8>&,
buffer<bf16>&>`（W 符号）= **(x, int4 权重, logits) 三参 kernel**——对上 P23
的"lm_head 唯一直接过 arg 的单 op（ac=4）"。

### 5.2 导出符号清单（T，节选）

```
LMHead::Impl::Impl/execute@0xde40/_generate_seq@0xc390/load_weights@0xb6d0/wait@0xba00
LMHead::{execute,load_weights,wait,x_exposed,~LMHead}
LM_Config::~LM_Config (W)
npu_app::update_ctrl_seq (W)          ← ctrl 序列原位更新
```

### 5.3 反汇编解剖

**F16【实锤】`Impl::execute`@0xde40 的运行时序**：

```
pthread_mutex_lock → BufCoh map lookup (0xdec0, hrx_buffer_s*→BufCoh 哈希)
→ 检查 host_dirty (0xdf11/0xdf1b/0xdf58 三个标志位)
→（惰性）npu_app::_setup_kernel (0xdf2d)
→ hrx_buffer_flush_range@plt (0xe191)      ← 只在 host_dirty 时
→ hrx::Runtime::ensure → hrx_stream_dispatch (0xe24e)
→ hrx_stream_flush (0xe277) → hrx_stream_wait (0xe29c)
```

教科书点：**"0 次 SYNC_BO"的软件侧机制在此可见**——引擎不调 XRT 的 sync，
而是用 BufCoh 脏位决定是否 `hrx_buffer_flush_range`（hrx 内部再走 BD 的
host-VA/aggressive-cache 路径，P24 已证）。FOM 的每个 BO 都有脏位记账。

**F17【实锤】`_generate_seq`@0xc390**：`operator new`+vtable+`emplace_back`
手工构造 `npu_wait_cmd` 对象 ×4（0xcafa-0xccac），2 处
`npu_dma_memcpy_nd`（0xc57e/0xc719），回边循环 3 组（0xcd66/0xd728/0xd7f8…）
——静态序列 + 循环展开的权重流（30208 tiles / 471×64 行分块），assert
`chunk_size % (m * cores) == 0`、`chunk_size / m / cores > 0` 给出分块律：
**vocab 按 (m × cores) 对齐切块**。与 P5 的 133MiB lm_head BO、2.66ms 单 op
对上（P26b：30208 tiles × 4608B = 139.2MB）。

### 5.4–5.5 对拍与 xclbin

`lm_head.xclbin`（strings 实锤）+ `MLIR_AIE` entry + `FLM_DUMP_TXN` 落盘
（F4）。lm_head 是独立 hwctx 上的单 kernel 图；权重常驻 SHMEM BO
（P5：133MiB×1）。

### 5.6 教科书要点（liblm_head.so）

1. **分类头独立成引擎**的原因：vocab 维度（30208 行）远超层内其它矩阵，
   单独一张图 + 独立 BO + 独立 flush 策略最省事（F16/F17）。
2. 三参 kernel 签名 = (激活, int4 权重, logits)——**dequant 在 kernel 内做**，
   host 不经手 fp 权重（F16 create_run 模板实参）。
3. BufCoh 脏位 + 条件 flush 是"每 token 0 ioctl sync"的 host 侧一半
   （P23-3 的开放问题在软件侧的答案）。

---

## 6. libgemm.so —— 通用 bf16/fake-quant GEMM 生成器（F18）

- 不链 hrx（F1）。开源接口 modules/gemm.hpp:39-40：
  `generate_seq(seq, M, K, N, weight_offset, ADD_BIAS, OUTPUT_MODE, bias_offset[, x_offset])`，
  Activation_Type_t 枚举（gemm.hpp:20）——**输出模式/偏置在生成期烧进指令**。
- `Gemm::Impl::generate_seq`@0x59d0：大量 `sar/shr/imul`（含 `shr $0x9`=M/512
  分块、`and $0xf` 16 路对齐）+ operator new/delete 构造 cmd 对象；无 memcpy_nd
  直调（经 .constprop 克隆内联）。
- `Gemm::Gemm(LM_Config&)`、`get_m/get_n/get_k` —— 被 prefill/其他模型引擎
  静态链入（F5）。
- 教科书点：**通用 GEMM 生成器与模型特定生成器共用同一 DSL**；M%512 约束与
  gen_dequant_mm_512 的 0x1ff 守卫一致——512 是 prefill 图的 tile 分块原子。

---

## 7. libdequant.so —— 反量化/激活打包生成器（F19）

- 不链 hrx。T 符号 6 个：
  `Dequant::Impl::generate_dequant_q4_1_seq`@0x86b0（wrapper 0x9a90）、
  `generate_dequant_q80_packed_in_q4nx_seq`@0x71d0（wrapper 0x86a0）、
  `reorder_cpy(u8*, buffer<u8>&, quant_block_t, int,int,int,int)`@0x4dc0。
- 150+ 个 `npu_dma_memcpy_nd` 调用点（静态展开的克隆）。
- `dequant_output_mode_t` 枚举（输出 bf16/f32/再打包）；assert
  `D_in % k_tile_q4 != 0`、错误串 `generate_dequant_q80_packed_in_q4nx_seq,
  D_in: <val>`。
- 语义：**q80（int8 激活）按 q4nx 的 tile 排布打包**（"packed_in_q4nx"）——
   与 IRON P11 的 int8 激活量化同思想（激活与权重用同一种 tile 流格式）。
- `reorder_cpy` 是 host 侧拷贝重排（非序列生成）——q4nx 文件序 → 设备流序。
- 教科书点：**量化/反量化/激活打包也是"生成器 + 图"的普通一员**；新代号
  libdequant_new.so（链 hrx+gomp）显示该算子正被引擎化改造（F1 第三类）。

---

## 8. libq4_npu_eXpress.so —— 模型格式库，不是设备引擎（F20）

25 个 T 符号全部属于两个类（nm 实锤）：

```
Q4NX: ctor, convert_model, _convert_to_q4nx, _convert_llama, _q4nx_reorder,
      _process_json_header, _grap_metadata(原文如此,源码拼写), get_block_size,
      get_weight_per_chunk, _to_bf16
SafeTensors: ctor, load_weights, switch_model, has_tensor, get_tensor_metadata,
      get_metadata, write_safetensors, _open_file, _load_tensors, _get_data_size
匿名命名空间全局: row_chunk_size / col_chunk_size / weight_per_chunk
```

- **不链 hrx、无 npu_sequence T 符号**：这是 safetensors ↔ q4nx 的转换器与
  读取器（含 _convert_llama 的模型族特化），与设备无关。2719 个符号大多是
  W 模板的实例化噪音。
- `row/col_chunk_size`、`weight_per_chunk` 与 P2 逆向的
  "tile = 32 行 × 256 列 × 4608B" 对应（256 列 = col_chunk，32 行 = row_chunk，
  18B/组 × 256 = weight_per_chunk 4608）。
- 教科书点：**"权重格式"与"引擎"分离**——格式库被所有引擎共用（F5 的
  load_layer_weights(Q4NX&) 依赖注入）。

---

## 9. 与 P22/P23 已知结论的差异与修正（F21）

| # | P22/P23 原表述 | 本次静态逆向的修正/细化 | 证据 |
|---|---|---|---|
| 1 | P22：libmha.so 19 个 T 符号，含 `d128_q4_1cu` | 当前 .so 仅 6 个模式生成器 + 辅助函数，**无 _1cu 变体**；19 个符号疑似 .dll 版本或旧快照 | nm -C libmha.so（F14） |
| 2 | P23：ctrl blob 只能 ioctl 拦截获取 | 引擎自带 `FLM_DUMP_TXN` 落盘开关（txn/patch 分文件），**逆向/调试有官方后门** | strings libhunyuan_npu/liblm_head（F4） |
| 3 | P23-3："0 次 SYNC_BO，一致性如何维持是开放问题（假设用户态 clflush）" | 实为两层：BD host-VA + AxCACHE（P24 已定案）之上，**host 侧还有 BufCoh 脏位门控的条件 flush**（execute@0xde40 可见 flush_range 只在 host_dirty 时调用）——不是无条件零 flush | F16 反汇编 |
| 4 | P23-8：列分工 "c0/c1/c6/c7 算力列、c3/c4 KV" | 静态表 proj_tiles/attn_kv_tiles/attn_qk_tiles 把同一分工**编进 .rodata**（0x4b240-0x4b2a0），且 attn 的 Q/K 与 KV 用不同行（QK 行 0/2，KV 行 1/3）——比 trace 观测多出一层行级细分 | F13 |
| 5 | P23-8：每列 68×18432w + 8×55296w 的组成未解释 | **68 = qkv+o(20) + gateup(48) 的 16-tile BD；8 = down 的 48-tile 大 BD**；0x12000 步长 = 16-tile BD 字节数；304 条 arg1 patch = 76 BD × 4 列，三方算术闭合 | F9 推导 |
| 6 | P5 时代："闭源 .so 动态链接 libxrt_coreutil 的 C++ API" | **当前发行版 .so 链 libhrx.so.0（C API），不再直接链 XRT**——xrt 拦截法对现版本无效，LD_PRELOAD 应改为拦 hrx_*（hrx 再下沉到安装版 XRT/驱动） | readelf -d 全家（F1） |
| 7 | P22："MHA 生成器由 xclbin 里的 AIE graph 执行"（含糊） | 细化：MHA 序列的 RTP 地址（0x3c00/0x6200/0x9080 等）是 **fused_prefill.xclbin 图的 buffer 偏移**；decode 路径（layer.xclbin）不用 MHA 生成器 | F6/F15 |

---

## 10. 教科书要点汇总（跨 .so，10 条）

1. **闭源栈 = 生成器 + 运行时两层**（F1）：生成器（mha/gemm/dequant/q4nx）
   无设备依赖可离线测试；引擎 .so 静态链生成器 + hrx 运行时。
2. **"编译"发生在 host 的序列生成器里**：npu_sequence→cmds2seq→TXN 字节流
   直接成为 HRX executable（entry=MLIR_AIE），无 assembler 步（§2/F8）。
3. **一切数据流（权重/KV/激活）统一为 BD 四拍重填**：BLOCKWRITE BD fill →
   DDR_PATCH 换址 → （S2MM）issue_token → 队列推，每拍配 TCT 等待（F9）。
4. **BD 长度是生成器自由参数**：16-tile（73,728B）为主，宽 K 矩阵用 48-tile
   （221,184B）；量化格式只改 per-tile 字节（{16,17,18}B/32 参数组，F9/F10）。
5. **decode=slot 复用（整层 1 blob），prefill=算子级生成（8 op/层）**，
   两张 xclbin 各司其职（F6/F11）。
6. **列分工静态编表**：proj_tiles{0,1,6,7 列}×4 行 / attn_qk{3,4 列,行 0,2} /
   attn_kv{3,4 列,行 1,3}（F13）——trace 列观测的源码对应物。
7. **形状参数走 RTP 广播**（512 块 GEMM、MHA chunk 参数），一张图服务任意
   合法形状；约束全部前置在生成器 assert（M%512、K%128、chunk 整除）（F10/F15）。
8. **一致性 = BD host-VA/AxCACHE（设备侧）+ BufCoh 条件 flush（host 侧）**，
   双层机制取代 SYNC_BO（F16 + P24）。
9. **权重格式库与引擎解耦**（Q4NX/SafeTensors 转换器独立成 .so），q4nx tile
   （32×256×4608B）从文件到 BD 不重排（F8/F9/F20）。
10. **调试后门是工业实践**：FLM_DUMP_TXN 落盘 txn+patch、每 .so 内嵌
    print_cmd/dump_cmd（W 符号）——"可观测性内建"值得写进工程章节（F4）。

---

## 附录 A：复现命令（全部离线）

```bash
SO=~/qwen/refs/FastFlowLM/src/lib/hrx
readelf -d $SO/*.so | grep NEEDED                 # F1 分层
nm -C $SO/libhunyuan_npu.so | grep ' T \| W '     # 符号清单（F5/F7）
objdump -d --start-address=0x30c70 --stop-address=0x31f20 -C $SO/libhunyuan_npu.so
objdump -s -j .rodata --start-address=0x4b240 --stop-address=0x4b2b0 $SO/libhunyuan_npu.so  # F13
objdump -d -C $SO/libmha.so | sed -n '/_gen_mha_seq_d128_q4/,/^$/p'   # F15
objdump -d --start-address=0xde40 -C $SO/liblm_head.so               # F16 execute
strings -a $SO/liblm_head.so | grep -E 'FLM_DUMP|xclbin|chunk_size'   # F4/F17
```

## 附录 B：地址速查（vaddr）

| .so | 符号/表 | 地址 |
|---|---|---|
| libhunyuan_npu.so | Impl::forward / _build_slot / Impl::prefill / attn_block forward | 0x17ad0 / 0x16ce0 / 0x19250 / 0x28fa0 |
| libhunyuan_npu.so | gen_layer_seq / gen_dequant_mm_512 / gen_mha_engine_seq / gen_lm_head_seq | 0x30c70 / 0x30f80 / 0x31f20 / 0x331b0 |
| libhunyuan_npu.so | _send_x / _move_weights / _receive_kv_cache / _move_kv_cache | 0x2f050 / 0x2f4a0 / 0x30470 / 0x30a00 |
| libhunyuan_npu.so | proj_tiles / attn_kv_tiles / attn_qk_tiles / 列号表 / dq512::IT | 0x4b240 / 0x4b280 / 0x4b290 / 0x4b2a0 / 0x4b160 |
| libmha.so | dispatcher / 跳转表 / head_dim 表 / Impl::IT / _gen_mha_seq_d128_q4 / _parameter_check | 0xbfe0 / 0x12280 / 0x12300 / 0x128c0 / 0x9220 / 0x4d70 |
| liblm_head.so | Impl::execute / _generate_seq / load_weights / wait | 0xde40 / 0xc390 / 0xb6d0 / 0xba00 |
| libgemm.so | Gemm::Impl::generate_seq | 0x59d0 |
| libdequant.so | q4_1(impl/wrapper) / q80_packed(impl/wrapper) / reorder_cpy | 0x86b0/0x9a90 / 0x71d0/0x86a0 / 0x4dc0 |

（F1–F21 编号事实以上；分析完成于 2026-09-29，纯静态、未占用 NPU。）
