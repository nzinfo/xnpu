//! xnpu-perf: NPU 性能度量核心（M5a）。
//!
//! 设计原则（notes/perf-lab.md P1）：
//! - **设备无关核**：本 crate 只懂「提交/完成时间线 + 字节/FLOP 计数 +
//!   roofline 算术」。一切设备特性（CU 切换代价、submit 开销、带宽天花板）
//!   都隔离在 [`MachineModel`] 的常数里，可被校准数据替换。
//! - **每条测量自带分母**：报告任何 GB/s / GFLOP/s 时同时报告它占机器
//!   天花板的比例（TileLang Analyzer 约定），否则数字不可比较。
//! - **三模式方法学**（M2/M3 实证）：
//!   - solo：submit+wait 墙钟 = 含 host/调度开销的上界；
//!   - burst：同 op 连发不等待，drain/n = 纯设备时间；
//!   - gap：链式墙钟 − Σ设备时间 = 调度/host 开销（CU 切换、syncobj 等待）。
//! - 时间语义按 notes §17：t_complete 取 syncobj timeline wait 返回时刻
//!   （唯一可信完成信号），state-poll 不作为完成依据。

pub mod model;
pub mod report;

pub use model::{tier, MachineModel};
pub use report::{render_markdown, OpStats, ReportSummary};

use std::time::Instant;

/// 一个被度量算子的静态形状描述。
///
/// `bytes_in`/`bytes_out` 是**有用**流量（roofline 分子）；当设备侧实际
/// 流量不同（如 w4gemvu 的 padded slot 流），另记 `bytes_stream`，报告
/// 同时给出两个口径。
#[derive(Clone, Debug)]
pub struct OpMeta {
    pub name: String,
    /// 形状族（"qkv" / "flowkv"），跨层聚合用。
    pub family: String,
    pub cu: u32,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub flops: u64,
    /// 设备侧实际流量的替代口径（None = 与有用流量同）。
    pub bytes_stream: Option<u64>,
    /// 带宽分档标签（[`crate::model::tier`]；None = 机器模型 default_tier）。
    /// 同一台机器上不同访问粒度的天花板差 40×（seq-dma 52 vs strided
    /// 1.2 GB/s），op 必须认领自己的档，%bw 才有意义（P6）。
    pub tier: Option<String>,
}

impl OpMeta {
    pub fn new(
        name: impl Into<String>,
        family: impl Into<String>,
        cu: u32,
        bytes_in: u64,
        bytes_out: u64,
        flops: u64,
    ) -> Self {
        OpMeta {
            name: name.into(),
            family: family.into(),
            cu,
            bytes_in,
            bytes_out,
            flops,
            bytes_stream: None,
            tier: None,
        }
    }

    /// 认领带宽分档（builder：`OpMeta::new(..).with_tier(tier::STRIDED)`）。
    pub fn with_tier(mut self, tier: impl Into<String>) -> Self {
        self.tier = Some(tier.into());
        self
    }

    pub fn useful_bytes(&self) -> u64 {
        self.bytes_in + self.bytes_out
    }
}

/// 一条时间线记录属于哪种测量模式。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    /// submit+wait 单发：墙钟上界。
    Solo,
    /// 同 op 连发块中的一员：块 drain/n 逼近纯设备时间。
    Burst,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Solo => "solo",
            Mode::Burst => "burst",
        }
    }
}

/// 原始时间线事件（Recorder 的落盘单位）。
#[derive(Clone, Debug)]
pub struct Event {
    pub name: String,
    pub family: String,
    pub cu: u32,
    pub mode: Mode,
    /// Recorder epoch 起的微秒数。
    pub t_submit_us: u64,
    /// solo 事件 = wait 返回时刻；burst 事件 = None（块尾另有 drain 标记）。
    pub t_complete_us: Option<u64>,
    pub seq: u64,
    pub iter: u32,
}

/// burst 块的 drain 标记：块内 n 次提交共用一个完成时刻。
#[derive(Clone, Debug)]
pub struct BlockDone {
    /// 块标签：单 op 块用 op 名；链式块任意（如 "chain:pipelined"）。
    /// 报告层按块窗口内的 burst 事件名自动判单 op / 链式，标签仅供显示。
    pub label: String,
    pub mode: Mode,
    pub iter: u32,
    pub t_start_us: u64,
    pub t_complete_us: u64,
    pub n_submits: u32,
    /// 块覆盖的逻辑迭代数（链式块 = token/iter 数；单 op 块 = n_submits）。
    pub iters: u32,
}

/// 时间线记录器：只记原始事实，分析全部留给 report 层。
pub struct Recorder {
    epoch: Instant,
    pub events: Vec<Event>,
    pub blocks: Vec<BlockDone>,
}

impl Recorder {
    pub fn new() -> Self {
        Recorder {
            epoch: Instant::now(),
            events: Vec::new(),
            blocks: Vec::new(),
        }
    }

    fn us(&self) -> u64 {
        self.epoch.elapsed().as_micros() as u64
    }

    /// solo：submit 前取 t0，wait 返回后调用。
    pub fn solo(&mut self, meta: &OpMeta, iter: u32, t0: Instant, seq: u64) {
        let t_submit_us = t0.duration_since(self.epoch).as_micros() as u64;
        self.events.push(Event {
            name: meta.name.clone(),
            family: meta.family.clone(),
            cu: meta.cu,
            mode: Mode::Solo,
            t_submit_us,
            t_complete_us: Some(self.us()),
            seq,
            iter,
        });
    }

    /// burst：每次 submit 后调用（完成时刻未知，块尾补 BlockDone）。
    pub fn burst_submit(&mut self, meta: &OpMeta, iter: u32, seq: u64) {
        self.events.push(Event {
            name: meta.name.clone(),
            family: meta.family.clone(),
            cu: meta.cu,
            mode: Mode::Burst,
            t_submit_us: self.us(),
            t_complete_us: None,
            seq,
            iter,
        });
    }

    /// burst 块收尾：块起点 t0（第一次 submit 前）到 drain 完成的墙钟。
    pub fn burst_done(
        &mut self,
        label: &str,
        mode: Mode,
        iter: u32,
        t0: Instant,
        n_submits: u32,
        iters: u32,
    ) {
        let t_start_us = t0.duration_since(self.epoch).as_micros() as u64;
        self.blocks.push(BlockDone {
            label: label.to_string(),
            mode,
            iter,
            t_start_us,
            t_complete_us: self.us(),
            n_submits,
            iters,
        });
    }
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

// ---- 手写 JSON（保持 workspace 零外部依赖） ----

pub fn jstr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// 把 (key, json-value) 对列表编成扁平 JSON 对象文本。
pub fn json_obj(pairs: &[(&str, String)]) -> String {
    let body: Vec<String> = pairs
        .iter()
        .map(|(k, v)| format!("{}: {}", jstr(k), v))
        .collect();
    format!("{{{}}}", body.join(", "))
}

pub fn jnum(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.4}")
    } else {
        "null".to_string()
    }
}

pub fn ju64(v: u64) -> String {
    v.to_string()
}

/// 序列化整次记录（事件 + 块 + 机器模型 + 摘要）为多行 JSON 文本。
pub fn trace_json(
    rec: &Recorder,
    model: &MachineModel,
    title: &str,
    summary: &ReportSummary,
) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"title\": {},\n", jstr(title)));
    out.push_str(&format!("  \"model\": {},\n", model.to_json_value()));
    out.push_str("  \"events\": [\n");
    for (i, e) in rec.events.iter().enumerate() {
        let line = json_obj(&[
            ("name", jstr(&e.name)),
            ("family", jstr(&e.family)),
            ("cu", e.cu.to_string()),
            ("mode", jstr(e.mode.as_str())),
            ("t_submit_us", e.t_submit_us.to_string()),
            (
                "t_complete_us",
                e.t_complete_us.map(|t| t.to_string()).unwrap_or("null".into()),
            ),
            ("seq", e.seq.to_string()),
            ("iter", e.iter.to_string()),
        ]);
        out.push_str(&format!("    {line}{}\n", if i + 1 < rec.events.len() { "," } else { "" }));
    }
    out.push_str("  ],\n");
    out.push_str("  \"blocks\": [\n");
    for (i, b) in rec.blocks.iter().enumerate() {
        let line = json_obj(&[
            ("label", jstr(&b.label)),
            ("mode", jstr(b.mode.as_str())),
            ("iter", b.iter.to_string()),
            ("t_start_us", b.t_start_us.to_string()),
            ("t_complete_us", b.t_complete_us.to_string()),
            ("n_submits", b.n_submits.to_string()),
            ("iters", b.iters.to_string()),
        ]);
        out.push_str(&format!("    {line}{}\n", if i + 1 < rec.blocks.len() { "," } else { "" }));
    }
    out.push_str("  ],\n");
    out.push_str(&format!(
        "  \"summary\": {}\n",
        summary.to_json_value(model)
    ));
    out.push_str("}\n");
    out
}
