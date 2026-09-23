# 05 · 宿主语言评估：C vs Rust（禁用 C++）

> 结论先行：**Rust**。C 保留为"阅读参考语言"（hrx-system 的 `experimental/xdna/`
> 全套 .c 就是我们 DRM 用法的活文档），不作为实现语言。

## 评估前提

1. 语言选择只影响**宿主侧**。设备侧（AIE 内核/图）由 Loom 或 mlir-aie 编译，
   与宿主语言无关——宿主只提交命令包和 buffer。
2. 明确禁用 C++ → 自动排除：XRT C++ API（本来就要逃）、llama.cpp 内核复用
   （其 minja 聊天模板、GGUF 内核均为 C++）、tokenizers-cpp。
3. 引擎宿主的核心危险区恰好是**内存安全的经典雷区**：解析二进制工件、patch DMA
   地址/重定位、BO 生命周期、mmap、多线程共享 KV cache。

## 决定性事实：AMD 底层全线是 C ABI

| 接口 | ABI | Rust 接入方式 |
|---|---|---|
| amdxdna DRM ioctl | C 结构体 | 手写 bindgen/nix，几十行 |
| libamdf | C 源码库 | extern "C" 直接链接 |
| loomc（Loom 编译器） | **公开 C API**（AOT/JIT/缓存/调优嵌入面） | extern "C" |
| libhrx | **公开 C ABI** | extern "C" |
| HSA / HIP C API | C | extern "C"（如走 HSA 轨） |

**C++ 只存在于我们要逃离的东西里**（XRT、FLM 引擎）。这个约束和生态现状天然对齐。

## 逐项对比

| 维度 | C | Rust | 权重 |
|---|---|---|---|
| C ABI 互操作 | 原生 | `extern "C"`/bindgen，一等公民（unsafe 限定在 hal 层） | 平 |
| DMA/重定位/工件解析的内存安全 | 全靠自律，错一个偏移=内核 oops 或静默算错 | 编译期保证 + offset/切片显式 | **高** |
| 并发（server 线程 + decode 循环 + prefill 分块） | 手工锁纪律 | Send/Sync 编译期约束 | **高** |
| Tokenizer | 无好选择 | HF `tokenizers` crate——**官方实现本体就是 Rust**（tokenizers-cpp 包的就是它） | **高** |
| 聊天模板（Jinja） | 无可用方案（minja 是 C++，被禁） | `minijinja` crate（minja 的上游） | **高** |
| OpenAI 兼容 server + JSON | jansson/cJSON + 手卷 HTTP，痛苦 | axum + serde，一晚上 | 高 |
| GGUF 后备路线（混合架构） | 需绑 llama.cpp（C++，禁） | candle/llm crate 原生读 GGUF | 中 |
| 工程性（测试/包管/重构） | 手搓 | cargo + 内建 test + 类型驱动重构 | 中 |
| 极端可控性/零运行时 | 原生 | std-only 或 no_std 可接近；静态链接 musl 单文件 | 平 |
| unsafe 仍需的场景 | — | mmap/ioctl/volatile MMIO（收敛进 `xnpu-hal` 一个 crate） | 平 |
| 团队上手 | 已熟练（hook 工具链即 C） | 需爬坡，但以本项目的模块规模（见 06）可控 | 中 |
| 先例 | hrx-system xdna runner（纯 C 测试器） | candle/burn/mistral.rs（Rust 推理引擎已成熟品类） | 平 |

## 判断

- C 的唯一实质优势是"已经很熟"。但本项目 80% 的代码量恰好处在 C 最容易出事的
  区域（二进制 patch、DMA、并发服务），且 tokenizer/Jinja/HTTP 三个必备件在
  C（且禁 C++）下没有体面的答案——这三件在 Rust 里都是现成 crate。
- Rust 的 unsafe 不会被消灭，但会被**圈禁**：约定所有 unsafe 只出现在
  `xnpu-hal`（ioctl/mmap/volatile），其上全部 safe Rust。
- 参考实现策略：hrx-system `experimental/xdna/*.c`（executable.c、irene-xdna-run.c、
  amdf_status.c）当"可执行的文档"逐行翻译成 Rust——M0 spike 就是干这个。

## 若将来必须退回 C 的触发条件

记录在案，避免反复：只有当 (a) libamdf/loomc 之外还需链接某个仅 C++ 的关键库，
且 (b) 该库无 C 封装、且 (c) 无法用 bindgen/cxx 隔离时——才重开本议题。
