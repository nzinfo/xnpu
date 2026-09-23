//! 报告层：把 Recorder 的原始时间线与 OpMeta 的字节/FLOP 计数合成
//! per-op 对比表 + roofline 判定。
//!
//! 判定规则（P1 阈值，校准后可调）：
//! - `overhead-dominated`：solo ≥ 2×burst —— 一半以上墙钟花在 submit/wait 往返
//! - `compute-bound`：GFLOP/s ≥ 50% peak（peak 未知则跳过该维度）
//! - `memory-bound`：stream GB/s ≥ 50% bw_stream
//! - `latency-bound`：stream GB/s < 10% bw_stream 且 burst 已剥离 overhead
//! - `n/a-bytes`：没给 OpMeta，只有时间没有分母
//! - 其余 `mixed`
//!
//! 块归属规则：块窗口 `[t_start, t_complete]` 内的 burst 事件若只有唯一
//! op 名 → 单 op 块（wall/n_submits 计入该 op 的 burst 时间）；多于一个
//! 名字 → 链式块（wall/iters = 每 iter 链时间）。

use crate::{json_obj, Event, MachineModel, Mode, OpMeta, Recorder};
use std::collections::{BTreeMap, BTreeSet};

/// 单个算子的聚合统计（solo 事件 + burst 块）。
#[derive(Clone, Debug)]
pub struct OpStats {
    pub name: String,
    pub family: String,
    pub cu: u32,
    pub n_solo: u32,
    pub solo_min_us: Option<f64>,
    pub solo_med_us: Option<f64>,
    pub n_burst: u32,
    pub burst_us_per_op: Option<f64>,
    pub useful_bytes: u64,
    pub stream_bytes: Option<u64>,
    pub flops: u64,
}

impl OpStats {
    /// 采用的设备时间：优先 burst（纯设备），退回 solo（含开销上界）。
    pub fn device_time_us(&self) -> Option<f64> {
        self.burst_us_per_op.or(self.solo_med_us)
    }

    /// 有用流量口径的带宽（bytes/µs → GB/s 除以 1000）。
    pub fn achieved_useful_gbps(&self) -> Option<f64> {
        Some(self.useful_bytes as f64 / (1000.0 * self.device_time_us()?))
    }

    /// 设备实际流量的口径（如 w4 padded slot 流）。
    pub fn achieved_stream_gbps(&self) -> Option<f64> {
        Some(self.stream_bytes? as f64 / (1000.0 * self.device_time_us()?))
    }

    pub fn gflops(&self) -> Option<f64> {
        Some(self.flops as f64 / (1000.0 * self.device_time_us()?))
    }

    /// solo − burst = 每 op 的 host/调度开销估计（无 burst 时为 None）。
    pub fn overhead_us(&self) -> Option<f64> {
        Some(self.solo_med_us? - self.burst_us_per_op?)
    }

    pub fn verdict(&self, m: &MachineModel) -> String {
        if self.n_solo == 0 && self.n_burst == 0 {
            return "no-data".into();
        }
        if let (Some(s), Some(b)) = (self.solo_med_us, self.burst_us_per_op) {
            if b > 0.0 && s >= 2.0 * b {
                return "overhead-dominated".into();
            }
        }
        if let Some(peak) = m.peak_gflops {
            if self.gflops().is_some_and(|f| f >= 0.5 * peak) {
                return "compute-bound".into();
            }
        }
        if let Some(g) = self.achieved_stream_gbps() {
            if g >= 0.5 * m.bw_stream_gbps {
                return "memory-bound".into();
            }
            if g < 0.1 * m.bw_stream_gbps && self.burst_us_per_op.is_some() {
                return "latency-bound".into();
            }
            return "mixed".into();
        }
        "n/a-bytes".into()
    }

    pub fn to_json_value(&self, m: &MachineModel) -> String {
        let opt = |v: Option<f64>| v.map(crate::jnum).unwrap_or_else(|| "null".into());
        json_obj(&[
            ("name", crate::jstr(&self.name)),
            ("family", crate::jstr(&self.family)),
            ("cu", self.cu.to_string()),
            ("n_solo", self.n_solo.to_string()),
            ("solo_min_us", opt(self.solo_min_us)),
            ("solo_med_us", opt(self.solo_med_us)),
            ("n_burst", self.n_burst.to_string()),
            ("burst_us_per_op", opt(self.burst_us_per_op)),
            ("useful_bytes", crate::ju64(self.useful_bytes)),
            (
                "stream_bytes",
                self.stream_bytes
                    .map(crate::ju64)
                    .unwrap_or_else(|| "null".into()),
            ),
            ("flops", crate::ju64(self.flops)),
            ("achieved_useful_gbps", opt(self.achieved_useful_gbps())),
            ("achieved_stream_gbps", opt(self.achieved_stream_gbps())),
            ("gflops", opt(self.gflops())),
            ("verdict", crate::jstr(&self.verdict(m))),
        ])
    }
}

/// 链式块（一块内混多算子）的统计。
#[derive(Clone, Debug)]
pub struct ChainStats {
    pub label: String,
    pub mode: Mode,
    pub n_submits: u32,
    pub iters: u32,
    pub wall_us: f64,
    /// 块窗口内出现过的不同 op 名集合（Δ 串行估计与覆盖倍数用）。
    pub names: Vec<String>,
}

impl ChainStats {
    pub fn per_iter_us(&self) -> f64 {
        self.wall_us / self.iters.max(1) as f64
    }
    pub fn per_submit_us(&self) -> f64 {
        self.wall_us / self.n_submits.max(1) as f64
    }
    /// 每 iter 每个 distinct op 被提交的次数（假设各 op 均匀）。
    pub fn coverage(&self) -> f64 {
        let per_iter = self.n_submits as f64 / self.iters.max(1) as f64;
        per_iter / self.names.len().max(1) as f64
    }
}

/// 整次测量的汇总（trace JSON 的 summary 字段）。
#[derive(Clone, Debug)]
pub struct ReportSummary {
    pub n_events: usize,
    pub n_blocks: usize,
    pub t_first_us: Option<u64>,
    pub t_last_us: Option<u64>,
    pub ops: Vec<OpStats>,
    pub chains: Vec<ChainStats>,
}

impl ReportSummary {
    /// 假设每 iter 覆盖每个 op 一次时的串行 solo 估计。
    pub fn sum_solo_med_us(&self) -> f64 {
        self.ops.iter().filter_map(|o| o.solo_med_us).sum()
    }

    pub fn to_json_value(&self, m: &MachineModel) -> String {
        let ops: Vec<String> = self.ops.iter().map(|o| o.to_json_value(m)).collect();
        let chains: Vec<String> = self
            .chains
            .iter()
            .map(|c| {
                json_obj(&[
                    ("label", crate::jstr(&c.label)),
                    ("mode", crate::jstr(c.mode.as_str())),
                    ("n_submits", c.n_submits.to_string()),
                    ("iters", c.iters.to_string()),
                    ("wall_us", crate::jnum(c.wall_us)),
                    ("per_iter_us", crate::jnum(c.per_iter_us())),
                    ("per_submit_us", crate::jnum(c.per_submit_us())),
                    ("coverage", crate::jnum(c.coverage())),
                ])
            })
            .collect();
        json_obj(&[
            ("n_events", crate::ju64(self.n_events as u64)),
            ("n_blocks", crate::ju64(self.n_blocks as u64)),
            (
                "t_first_us",
                self.t_first_us
                    .map(crate::ju64)
                    .unwrap_or_else(|| "null".into()),
            ),
            (
                "t_last_us",
                self.t_last_us
                    .map(crate::ju64)
                    .unwrap_or_else(|| "null".into()),
            ),
            ("sum_solo_med_us", crate::jnum(self.sum_solo_med_us())),
            ("ops", format!("[{}]", ops.join(", "))),
            ("chains", format!("[{}]", chains.join(", "))),
        ])
    }
}

/// 聚合 + 渲染 markdown 报告。
pub fn render_markdown(
    rec: &Recorder,
    metas: &[OpMeta],
    model: &MachineModel,
    title: &str,
) -> (String, ReportSummary) {
    let summary = summarize(rec, metas);

    let mut md = String::new();
    md.push_str(&format!("# {title}\n\n"));
    md.push_str(&format!(
        "机器模型 **{}**：bw_stream {:.1} GB/s，cu_switch {:.0} µs，submit_overhead {:.0} µs，peak_gflops {}\n\n",
        model.name,
        model.bw_stream_gbps,
        model.cu_switch_reload_us,
        model.submit_overhead_us,
        model
            .peak_gflops
            .map(|p| format!("{p:.0}"))
            .unwrap_or_else(|| "未知".into())
    ));
    md.push_str(&format!("常数来源：{}\n\n", model.provenance));

    // ---- per-op 表 ----
    md.push_str("## Per-op\n\n");
    md.push_str("| op | cu | solo µs (n) | burst/op µs (n) | ovh µs | useful B | stream B | GB/s use | GB/s str | %bw | GF/s | verdict |\n");
    md.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for o in &summary.ops {
        let dash = || "–".to_string();
        let solo = o.solo_med_us.map(|v| format!("{v:.1}")).unwrap_or_else(dash);
        let burst = o
            .burst_us_per_op
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(dash);
        let ovh = o.overhead_us().map(|v| format!("{v:.1}")).unwrap_or_else(dash);
        let useful = if o.useful_bytes > 0 {
            crate::ju64(o.useful_bytes)
        } else {
            dash()
        };
        let stream = o
            .stream_bytes
            .map(crate::ju64)
            .unwrap_or_else(dash);
        let guse = o
            .achieved_useful_gbps()
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(dash);
        let gstr = o
            .achieved_stream_gbps()
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(dash);
        let bw_pct = o
            .achieved_stream_gbps()
            .map(|g| format!("{:.0}", 100.0 * g / model.bw_stream_gbps))
            .unwrap_or_else(dash);
        let gfs = o.gflops().map(|v| format!("{v:.1}")).unwrap_or_else(dash);
        md.push_str(&format!(
            "| {} | {} | {} ({}) | {} ({}) | {} | {} | {} | {} | {} | {}% | {} | {} |\n",
            o.name,
            o.cu,
            solo,
            o.n_solo,
            burst,
            o.n_burst,
            ovh,
            useful,
            stream,
            guse,
            gstr,
            bw_pct,
            gfs,
            o.verdict(model)
        ));
    }

    // ---- 链式块 ----
    if !summary.chains.is_empty() {
        // 名字 → solo_med 查表（串行估计用；Δ 需要链内每个 op 都有 solo 数据）
        let solo_of: BTreeMap<&str, f64> = summary
            .ops
            .iter()
            .filter_map(|o| o.solo_med_us.map(|v| (o.name.as_str(), v)))
            .collect();
        md.push_str("\n## 链式块\n\n");
        md.push_str("| label | mode | subm/iter | iters | wall µs | µs/iter | µs/submit | cov | serial est µs | Δ(iter−serial) |\n");
        md.push_str("|---|---|---|---|---|---|---|---|---|---|\n");
        for c in &summary.chains {
            let per_iter = c.per_iter_us();
            let cov = c.coverage();
            let serial: Option<f64> = c
                .names
                .iter()
                .map(|n| solo_of.get(n.as_str()).copied())
                .sum::<Option<f64>>()
                .map(|s| s * cov);
            let serial_s = serial
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "–".into());
            let delta_s = serial
                .map(|s| format!("{:+.1}", per_iter - s))
                .unwrap_or_else(|| "–".into());
            md.push_str(&format!(
                "| {} | {} | {:.0} | {} | {:.1} | {:.1} | {:.2} | {:.1} | {} | {} |\n",
                c.label,
                c.mode.as_str(),
                c.n_submits as f64 / c.iters.max(1) as f64,
                c.iters,
                c.wall_us,
                per_iter,
                c.per_submit_us(),
                cov,
                serial_s,
                delta_s
            ));
        }
        md.push_str("\nserial est = Σ solo_med(链内各 op) × cov（cov = 每 iter 每 op 覆盖次数）。\n");
        md.push_str("Δ < 0 = 流水重叠收益；Δ > 0 = 每 iter 多出的调度/host 开销（含 CU 切换 PDI 重载、syncobj 等待）。\n");
    }

    md.push_str("\n## 判定规则\n\n");
    md.push_str("overhead-dominated: solo ≥ 2×burst；compute-bound: GF/s ≥ 50% peak；memory-bound: stream GB/s ≥ 50% bw_stream；latency-bound: stream GB/s < 10% bw_stream（burst 口径）；无 OpMeta → n/a-bytes。\n\n");
    md.push_str("时间语义：t_complete = syncobj timeline wait 返回时刻（notes §17）；solo = submit+wait 墙钟上界，burst = 同 op 连发块 drain/n。\n");

    (md, summary)
}

fn summarize(rec: &Recorder, metas: &[OpMeta]) -> ReportSummary {
    // 事件按 op 名分组（BTreeMap 保证报告顺序稳定）
    let mut evs: BTreeMap<&str, Vec<&Event>> = BTreeMap::new();
    for e in &rec.events {
        evs.entry(e.name.as_str()).or_default().push(e);
    }

    // 块归属：窗口内 burst 事件的 op 名集合
    let mut per_op_burst: BTreeMap<&str, (f64, u32)> = BTreeMap::new(); // (wall_sum, submits)
    let mut chain_map: BTreeMap<(String, Mode), ChainStats> = BTreeMap::new();
    for b in &rec.blocks {
        let names: BTreeSet<&str> = rec
            .events
            .iter()
            .filter(|e| {
                e.mode == Mode::Burst
                    && e.t_submit_us >= b.t_start_us
                    && e.t_submit_us <= b.t_complete_us
            })
            .map(|e| e.name.as_str())
            .collect();
        let wall = (b.t_complete_us - b.t_start_us) as f64;
        if let Some(name) = names.iter().next().copied().filter(|_| names.len() == 1) {
            let acc = per_op_burst.entry(name).or_insert((0.0, 0));
            acc.0 += wall;
            acc.1 += b.n_submits;
        } else {
            // 同标签链式块跨 iter 聚合：Σwall / Σiters = 每 iter 链时间
            let e = chain_map
                .entry((b.label.clone(), b.mode))
                .or_insert(ChainStats {
                    label: b.label.clone(),
                    mode: b.mode,
                    n_submits: 0,
                    iters: 0,
                    wall_us: 0.0,
                    names: Vec::new(),
                });
            e.n_submits += b.n_submits;
            e.iters += b.iters;
            e.wall_us += wall;
            for n in &names {
                if !e.names.iter().any(|x| x == n) {
                    e.names.push(n.to_string());
                }
            }
        }
    }
    let chains: Vec<ChainStats> = chain_map.into_values().collect();

    let mut ops: Vec<OpStats> = Vec::new();
    for (name, ev_list) in &evs {
        let mut solo_d: Vec<f64> = ev_list
            .iter()
            .filter(|e| e.mode == Mode::Solo)
            .filter_map(|e| e.t_complete_us.map(|t| (t.saturating_sub(e.t_submit_us)) as f64))
            .collect();
        solo_d.sort_by(|a, b| a.total_cmp(b));
        let n_solo = solo_d.len() as u32;
        let n_burst = ev_list.iter().filter(|e| e.mode == Mode::Burst).count() as u32;
        let burst_us_per_op = per_op_burst.get(name).map(|(w, n)| w / (*n as f64));
        let meta = metas.iter().find(|m| m.name == *name);
        let (family, cu, useful, stream, flops) = match meta {
            Some(m) => (m.family.clone(), m.cu, m.useful_bytes(), m.bytes_stream, m.flops),
            None => (ev_list[0].family.clone(), ev_list[0].cu, 0, None, 0),
        };
        ops.push(OpStats {
            name: name.to_string(),
            family,
            cu,
            n_solo,
            solo_min_us: solo_d.first().copied(),
            solo_med_us: median(&solo_d),
            n_burst,
            burst_us_per_op,
            useful_bytes: useful,
            stream_bytes: stream,
            flops,
        });
    }

    let t_first_us = rec.events.iter().map(|e| e.t_submit_us).min();
    let t_last_us = rec
        .events
        .iter()
        .filter_map(|e| e.t_complete_us)
        .chain(rec.blocks.iter().map(|b| b.t_complete_us))
        .max();

    ReportSummary {
        n_events: rec.events.len(),
        n_blocks: rec.blocks.len(),
        t_first_us,
        t_last_us,
        ops,
        chains,
    }
}

fn median(sorted: &[f64]) -> Option<f64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    Some(if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::{Duration, Instant};

    #[test]
    fn solo_and_burst_stats() {
        let mut rec = Recorder::new();
        let meta = OpMeta::new("gemm", "test", 0, 1_000_000, 1_000, 2_000_000_000);
        for i in 0..5u32 {
            let t0 = Instant::now();
            sleep(Duration::from_micros(300));
            rec.solo(&meta, i, t0, i as u64);
        }
        let t0 = Instant::now();
        for i in 0..4u32 {
            rec.burst_submit(&meta, i, i as u64);
            sleep(Duration::from_micros(80));
        }
        sleep(Duration::from_micros(80));
        rec.burst_done("gemm", Mode::Burst, 0, t0, 4, 4);

        let (md, summary) = render_markdown(&rec, &[meta.clone()], &MachineModel::default(), "test");
        assert!(md.contains("| gemm |"), "table row missing:\n{md}");
        let st = summary
            .ops
            .iter()
            .find(|o| o.name == "gemm")
            .expect("gemm stats");
        assert_eq!(st.n_solo, 5);
        assert_eq!(st.n_burst, 4);
        assert!(st.solo_med_us.unwrap() >= 250.0);
        assert!(st.burst_us_per_op.unwrap() >= 60.0);
        assert!(st.useful_bytes == 1_001_000);
        // 时间来自 sleep，不判具体 verdict，只判非 no-data
        assert_ne!(st.verdict(&MachineModel::default()), "no-data");
    }

    #[test]
    fn chain_block_attribution() {
        let mut rec = Recorder::new();
        let a = OpMeta::new("opA", "f", 0, 100, 10, 0);
        let b_ = OpMeta::new("opB", "f", 1, 100, 10, 0);
        let t0 = Instant::now();
        rec.burst_submit(&a, 0, 0);
        sleep(Duration::from_micros(50));
        rec.burst_submit(&b_, 0, 1);
        sleep(Duration::from_micros(50));
        rec.burst_done("chain:pipelined", Mode::Burst, 0, t0, 2, 1);

        let (_, summary) = render_markdown(&rec, &[a, b_], &MachineModel::default(), "t");
        assert_eq!(summary.chains.len(), 1, "mixed block must be chain");
        assert_eq!(summary.chains[0].iters, 1);
        assert!(summary.chains[0].wall_us >= 80.0);
        // 两个 op 都没有 burst 计入（块是链式的）
        for o in &summary.ops {
            assert!(o.burst_us_per_op.is_none());
        }
    }
}
