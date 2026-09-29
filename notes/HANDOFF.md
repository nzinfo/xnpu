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
| **IRON（内核 C++，豁免区）** | `~/qwen/xnpu/IRON` | decode-fusion-llama | **永不 push**（origin = amd/IRON） |

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

## 4. 当前进度快照（截至本文件时刻）

- **E2E 现状**：lv2 路径 33 exec/token，**30.6ms/token（32.7 tok/s）**，双门 PASS；
  对标 FLM 21.44ms。
- **瓶颈已定案**（perf-lab 6f-9 §7 终模型，两级自我否证后）：
  `T_exec = W_bytes/55.3GB/s + Σ胶水相位停摆(~250µs)`；36.6GB/s = 55.6×占空比 2/3；
  FLM 同墙 duty≈0.9+。差距本质 = 占空比，不是带宽/通道/计算/ring。
- **已处决的方向**（别再花板时）：通道加宽（N=16 零收益）、元素做大
  （L1 放不下且只省 ~13µs）、同深度 mem-DMA W（零收益）、ring 优化（≤20µs）。
- **下一刀（已批准方向）**：内核侧胶水分块穿插 W 块消费 ——
  `IRON/iron/operators/w4gemvu/w4gemvu_layer.cc`（K 口味 dispatcher 结构）、
  `design_layerv2.py`（fill 拓扑）。目标 exec 505-520µs → E2E ~22.5ms ≈ FLM。
  配套增量分解：o 块消费完立即累 sumsq 分量；quant 与 gather 分段流水。
- 关键契约：runtime-N 内核从 X 元素 [6404,6408) u32 读 worker 数
  （`design_layerv2.py:35`）；K-header 口味表（2048/101/102/103/104/105）；
  IRON 内核符号必须 `extern "C"`。
- **T_stall 直接测量工具已就绪未上板**：`tools/lv2_floor_pack.py`
  （P28-7）——把全部 W 元素 K 头改 2049（dummy-compute 口味），跑通
  A 流+计算+C drain 但零胶水相位；T(full)−T(floor) = 胶水暴露量，
  即 6f-9 §7 闭合反推 ~250µs 的直接裁决。与 r0 探针配对可再剥掉
  gather。

## 5. 进行中 / 悬而未决

- **在飞 subagent**：数据流模型文档 `docs/perf/02-dataflow-perf-model.md`
  正在做 §7 合并修订（元素节拍 β → 占空比）。它只改这个文件，别动它。
- 未解（低优先，都留档在 perf-lab）：60.8s 看门狗 fw 侧本体（moot，路径不可达）；
  C FromDevice EINVAL（clflush_region+2ms 绕过中）；pyxrt 爬行机理（引擎路径免疫）。
- `notes/.diag` / `notes/.shelltest`：磁盘配额勘查残片，垃圾，勿提交。

## 6. 文档地图

| 文件 | 内容 |
|---|---|
| `notes/perf-lab.md` | **编号实验台账 P1..P28-6f-9**（一切机制的 provenance） |
| `docs/perf/02-dataflow-perf-model.md` | 五维资源包络模型 + xnpu-dfsim 工具设计（修订中） |
| `docs/flm-so-analysis.md` | FLM 27 个 .so 静态逆向（F1-F21）；`FLM_DUMP_TXN` 官方后门 |
| `docs/perf/03-tilelang-deep-dive.md` | TileLang/TileSight 借鉴 |
| `tools/` | 全部研究脚本（xdump2、ctrl_decode、hoist_probe、layerv2_pack/golden、p28*/ 存档） |
| `~/qwen/refs/FastFlowLM` | FLM 源码 + 27 .so（MIT 编排层，金标准参照） |
| 权重/夹具 | `build/w4u_hy`（v4 包）、`build/lv2_hy`(N=8)、`build/lv2_hy_w16`、模型 `~/.config/flm/models/Hy-MT2-1.8B-NPU2`（model.q4nx，与 FLM 同一份文件） |
