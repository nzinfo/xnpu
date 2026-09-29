# 交接简报（agent handoff）— 2026-09-29

给接手的 agent/人：本文件是**操作性**交接（工具在哪、规矩是什么、进行到哪）。
机制与实验的全部细节以 `notes/perf-lab.md`（P1–P28-6f-9 编号台账）为唯一真源，
本文不重复其内容，只给指针。

---

## 1. 工具链：llvm-aie / mlir-aie 在哪

- **venv**：`/home/nzinfo/.venvs/npu314`（python3.14，`--system-site-packages`
  吃系统 dist-packages 的 **pyxrt cp314 .so** —— 这是必须 3.14 的唯一原因）。
- **llvm-aie 工具链本体**（pip 装 `llvm-aie` nightly，py3-none wheel）：
  ```
  /home/nzinfo/.venvs/npu314/lib/python3.14/site-packages/llvm-aie/bin/
  ```
  当前版本 `llvm_aie 22.0.0.2026092801+b0d37423`。
  **AIE 反汇编只有这里的 `llvm-objdump` 可用**——系统 objdump 报
  "architecture UNKNOWN"，llvm-objdump-18/21 报 "can't find target"
  （P28-4c 定律，第三次踩过）。
- venv `bin/` 下另有：`aiecc.py`、`aie-opt`、`aie-translate`、`aie-lsp-server`。
- **重建配方**（环境被系统升级打掉时，P27-3b 留档）：
  `mlir_aie==v1.2.1` cp314 wheel 来自 **GitHub release extra-index**（非 PyPI）；
  llvm-aie nightly 同源，网络断续要 `pip install --resume-retries 10`；
  torch `2.12.1+cpu` 走 pytorch cpu index；再 `pip install -e IRON --no-deps`。
  三个非 PyPI wheel 是可运行性的全部外部依赖。

## 2. 仓库拓扑与推送纪律（三条铁律）

| 仓库 | 路径 | 分支/上游 | 推送 |
|---|---|---|---|
| **引擎（Rust，禁 C++）** | `~/qwen/xnpu/xnpu` | master → origin/master | 允许（github.com/nzinfo/xnpu.git） |
| **笔记/文档/tools** | `~/qwen/xnpu`（外层，嵌套上面那个仓） | master → origin/**tutorial**（push 要显式 `git push origin master:tutorial`，push.default=simple 会拒） | 允许 |
| **IRON（内核 C++，豁免区）** | `~/qwen/xnpu/IRON` | decode-fusion-llama → fork/decode-fusion-llama | **只推 fork**（`fork` = git@github.com:nzinfo/IRON.git，2026-09-29 起可用；origin = amd/IRON 永不推） |

- commit 仅在用户明说时做；消息末尾 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。
- `~/qwen` 本身不是 git 仓。

## 3. 上板纪律

- **板上永远一次只有一个 pytest/板测进程**。TaskStop 后 `pgrep -af pytest|sudo`
  确认子进程真死再跑下一个（并发 = 互相污染，6f-1 付费教训）。
- 板测配方（root + memlock 都必需）：
  ```
  sudo -n env PATH="/home/nzinfo/.venvs/npu314/bin:$PATH" HOME=/home/nzinfo \
    prlimit --memlock=unlimited:unlimited -- \
    /home/nzinfo/qwen/xnpu/xnpu/target/release/xnpu-cli run-lv2loop <iters> <n>
  ```
  pytest 同理（cwd=IRON 对应 operator 目录）。`LV2LOOP_PROBE=<rings>` 选
  AIELv2Probe 夹具；`LV2LOOP_WAIT_S=<s>` per-exec 超时。
- **板况漂移定律**（P7/P21-5）：跨批绝对值 ±40% 不可比；A/B 必须同二进制
  同热板窗口背靠背交替。
- **夹具陈旧定律**（P13/P20b/P28-5 三犯）：IRON/build 编译产物拷到
  `~/qwen/xnpu/build/` 后必须核对 md5/时间戳——陈旧 ctrl bin 不报错只算错/挂死。
- sync 纪律（P21-3）：ToDevice → 用户态 CLFLUSHOPT（`clflush_region`）；
  FromDevice → 保留 ioctl（fw fence 正确性必需）。
- 完成语义（6f-7）：syncobj 信号 ≠ 成功；必须读 exec BO 状态字 bits[3:0]==4。

## 4. 当前进度快照（2026-09-29 深夜，P28-12 终局）

- **E2E 现状**：lv2 路径 33 exec/token，**26.3ms/token（38.0 tok/s）**，双门 PASS；
  对标 FLM 21.44ms（差距 ~4.9ms）。P28-7 quant 向量化（762→707µs）+
  P28-8 流水化（哨兵轮询+提前 submit，28.5→26.2）+ P28-10 幽灵完成守卫
  （状态字全模式检查，weed A 含 exec32 补查）全部落地。
- **P28-9 ask 模式**：引擎可回答真实问题（prefill 参考链 + NPU 生成环，
  EOS 停止律 {120020, 120001}——120020 解码为空串的"隐形 EOS"曾致续写
  漂移；FLM 对等复现已做）。
- **P28-12（本日终局）：device-side attention 数值链板上 10/10 PASS**
  （design_lv2attn/test_lv2attn + w4gemvu_attn_* 微入口）：rope + qk-norm +
  KV 流式 + GQA 16/4 online-softmax + finalize，4e-2 容差。**layer-v3 全部
  内核数值件就绪，剩余为纯集成**（施工图 perf-lab P28-11 §3：KV tensor
  常驻 + X/XN 合并腾 regmap 槽 + 整层流 + host 改造；预期 E2E ≈23.6→22.5ms）。
- 调试战役五定律（P28-12 系列）：栈律复发 0x580（每加向量子程序必查
  paddxm）/ 探针极性（默认必须=禁用路径）/ exp2 输入域 clamp −40 /
  **仪器必须放输出区外**（写进输出区的仪器读回全是 finalize 覆盖后的
  假数据，误导两轮）/ **同秒 mtime 构建缓存陷阱**（快改快跑跑旧内核，
  touch 或 rm build）。另有：lv_ctr 字节数组存 u32 会截断（逐字段宽度
  核对）、sin 表偏移错位读到表间零填充。
- **瓶颈已定案**（perf-lab 6f-9 §7 终模型，两级自我否证后）：
  `T_exec = W_bytes/55.3GB/s + Σ胶水相位停摆(~250µs)`；36.6GB/s = 55.6×占空比 2/3；
  FLM 同墙 duty≈0.9+。差距本质 = 占空比，不是带宽/通道/计算/ring。
- **已处决的方向**（别再花板时）：通道加宽（N=16 零收益）、元素做大
  （L1 放不下且只省 ~13µs）、同深度 mem-DMA W（零收益）、ring 优化（≤20µs）。
- **下一刀（layer-v3 集成，数值件全部就绪）**：按 perf-lab P28-11 §3
  施工——X/XN 合并腾第 5 tensor 槽 → KV 常驻 BO（67MB，v1 = S_max=1024
  静态 fill + runtime-S 消费，+1.2ms/token；v2 = bucket xclbin 对标 FLM
  slot 机器 ~0.6ms）→ design_layerv3（整层流 [X0/xn|w1|qkv|attn|o|w2|
  gate/up|down|drain]）→ host 改造。**预期 26.3 → ~23.6（v1）→ ~22.5ms
  （v2）= FLM 追平点**。attention 内核口味（K=210/211/212）已板上验证，
  集成时照搬 + 把 qkv 输出从 C drain 改为内核内直通。
- 关键契约：runtime-N 内核从 X 元素 [6404,6408) u32 读 worker 数
  （`design_layerv2.py:35`）；K-header 口味表（2048/101/102/103/104/105）；
  IRON 内核符号必须 `extern "C"`。
- **地板判别已上板并闭合**（P28-7）：`tools/lv2_floor_pack.py` + 内核
  K=2049 口味 + main.rs `LV2LOOP_W` 覆盖。floor 632µs 直接实证 6f-9。
  用法：`LV2LOOP_W=/tmp/lv2_floor_exec05.bin xnpu-cli run-lv2loop 12 8`
  （pack 先由工具生成）。

## 5. 进行中 / 悬而未决

- ~~在飞 subagent~~ **已完成并提交**：`docs/perf/02-dataflow-perf-model.md`
  §7 合并修订落地（元素节拍 β → 占空比终模型，§4.2 三维包络+第五维停摆段，
  验证表 16 检查点）。后续排队论抽象见 `docs/perf/04-queueing-abstraction.md`。
- 未解（低优先，都留档在 perf-lab）：60.8s 看门狗 fw 侧本体（moot，路径不可达）；
  C FromDevice EINVAL（clflush_region+2ms 绕过中）；pyxrt 爬行机理（引擎路径免疫）。
- `notes/.diag` / `notes/.shelltest`：磁盘配额勘查残片，垃圾，勿提交。
- allium 待办：规约 B/C/D 同步（ReassembleAndAttend 双路径析取、
  sentinel_poll 字段声明、开放问题与契约对齐）+ E（ask 模式入规约——
  dynamic pos/embedding 胶水/EOS 律/关闭两条已答开放问题）。代码侧
  weed A 已修（最终 exec 补查前驱状态字）。

## 6. 文档地图

| 文件 | 内容 |
|---|---|
| `notes/perf-lab.md` | **编号实验台账 P1..P28-6f-9**（一切机制的 provenance） |
| `docs/perf/02-dataflow-perf-model.md` | 数据流定量性能模型（占空比终模型）+ xnpu-dfsim 工具设计 |
| `docs/perf/04-queueing-abstraction.md` | 排队论再形式化：休假/窗口流控/汇结/闭网络映射 + 可证伪预言 |
| `docs/flm-so-analysis.md` | FLM 27 个 .so 静态逆向（F1-F21）；`FLM_DUMP_TXN` 官方后门 |
| `docs/perf/03-tilelang-deep-dive.md` | TileLang/TileSight 借鉴 |
| `tools/` | 全部研究脚本（xdump2、ctrl_decode、hoist_probe、layerv2_pack/golden、p28*/ 存档） |
| `~/qwen/refs/FastFlowLM` | FLM 源码 + 27 .so（MIT 编排层，金标准参照） |
| 权重/夹具 | `build/w4u_hy`（v4 包）、`build/lv2_hy`(N=8)、`build/lv2_hy_w16`、模型 `~/.config/flm/models/Hy-MT2-1.8B-NPU2`（model.q4nx，与 FLM 同一份文件） |
