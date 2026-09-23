//! 机器模型：把设备相关常数隔离在一处，可被校准数据替换。
//!
//! 默认值全部来自 M2–M5c 在本机（XDNA2 / npu2）的实测锚点，出处写在
//! 每个字段注释里；`perf-calibrate`（M5a）重新测量后可从 JSON 覆盖。
//!
//! 带宽按**访问粒度分档**（P6）：M5c 的 FLM trace 证伪了单锚点——
//! 33.3 GB/s 不是这台机器的天花板，只是 slot 复制流的一种形态；FLM
//! 融合层内核 48–64、lm_head ≥52、M2 GEMM 54 都是不同形态下的更高
//! 实测。单一 `bw_stream_gbps` 会把 seq-dma 形态的 op 判成 >100%。
//! 现在每档一个天花板，op 用 [`crate::OpMeta::tier`] 认领自己的档。

use crate::json_obj;
use std::collections::BTreeMap;

/// 带宽分档名（访问粒度语义，不是数值区间）。
pub mod tier {
    /// 顺序大 BO 权重流（FLM lm_head 133MiB/2.66ms ≥52 GB/s 下界）。
    pub const SEQ_DMA: &str = "seq-dma";
    /// F 槽复制流（w4gemvu：x 复制进 F 个 K_MAX 槽；实测 14–43 GB/s，
    /// 上界 = 最宽形状 gateup F=24 的 43.2）。
    pub const SLOT_STREAM: &str = "slot-stream";
    /// 2D stride KV 容量流（flowkv：S 运行时但按编译容量整流，~1.2 GB/s）。
    pub const STRIDED: &str = "strided";
}

/// 一台 NPU 的性能天花板与机制代价（单位见字段名）。
#[derive(Clone, Debug)]
pub struct MachineModel {
    pub name: String,
    /// 同 op burst 连发下的权重流带宽上限（GB/s）。
    /// 遗留单锚点：2026-09-23 run-w4ulayer 42L per-layer 模式（全程
    /// syncobj wait）slot 流 33.3 GB/s。M5c 已证伪其为天花板（FLM ≥52）；
    /// 现仅当 `bw_tiers` 为空 / `default_tier` 缺失时兜底。
    pub bw_stream_gbps: f64,
    /// 带宽分档：tier 名 → 天花板 GB/s。键集开放——perf-calibrate 可以
    /// 只覆盖它测得到的档（未知档继承这里的默认）。
    pub bw_tiers: BTreeMap<String, f64>,
    /// 无 tier 标注的 op 采用的档名。
    pub default_tier: String,
    /// 峰值算力（GFLOP/s）。尚无可靠实测（None = 未知，判 bound 时跳过
    /// compute 维度）。
    pub peak_gflops: Option<f64>,
    /// CU 切换（cu_mask 变化 → 固件 PDI 重载）代价（µs）。
    /// 实测：run-multi burst vs interleaved 差值 ~650µs/次（M2）。
    pub cu_switch_reload_us: f64,
    /// 一次 submit+syncobj-wait 的 host 往返开销（µs）。
    /// 实测：solo − burst 差 ≈55µs/op（M3 run-pipe/w4layer）。
    pub submit_overhead_us: f64,
    /// 常数来源说明（进报告，防止数字裸奔）。
    pub provenance: String,
}

impl Default for MachineModel {
    fn default() -> Self {
        MachineModel {
            name: "xdna2-npu2-defaults".to_string(),
            bw_stream_gbps: 33.3,
            bw_tiers: BTreeMap::from([
                // FLM trace（M5c P5）下界：lm_head 133MiB/2.66ms ≈ 52；
                // M2 GEMM 2048³ 8col 曾见 54 —— seq-dma 无自测内核前取 52。
                (tier::SEQ_DMA.to_string(), 52.0),
                // 我们自己的 burst 实测上界：gateup F=24 43.2 GB/s（P1）。
                (tier::SLOT_STREAM.to_string(), 43.0),
                // flowkv 容量流（P2 fkprobe：1MiB/2470µs ≈ 1.15；M4a E2E
                // 1986µs/32 层折算同量级）。
                (tier::STRIDED.to_string(), 1.2),
            ]),
            default_tier: tier::SLOT_STREAM.to_string(),
            peak_gflops: None,
            cu_switch_reload_us: 650.0,
            submit_overhead_us: 55.0,
            provenance: "实测锚点（未校准默认值）：bw_tiers={seq-dma 52=FLM lm_head 下界(M5c P5), slot-stream 43=gateup F=24 burst 上界(P1), strided 1.2=flowkv 容量流(P2)}；单锚 33.3=P1 per-layer 口径(已证伪为天花板,仅兜底)；cu_switch=run-multi burst/interleaved 差；submit=run-pipe solo−burst 差".to_string(),
        }
    }
}

impl MachineModel {
    /// 某档的天花板。未知档名 / None 都落到 default_tier；连 default_tier
    /// 都不在表里（校准数据只写了一部分）时退回遗留单锚。
    pub fn bw_for(&self, tier: Option<&str>) -> f64 {
        match tier {
            Some(t) => self
                .bw_tiers
                .get(t)
                .or_else(|| self.bw_tiers.get(&self.default_tier))
                .copied()
                .unwrap_or(self.bw_stream_gbps),
            None => self
                .bw_tiers
                .get(&self.default_tier)
                .copied()
                .unwrap_or(self.bw_stream_gbps),
        }
    }

    pub fn to_json_value(&self) -> String {
        let tiers: Vec<String> = self
            .bw_tiers
            .iter()
            .map(|(k, v)| format!("{}: {}", crate::jstr(k), crate::jnum(*v)))
            .collect();
        json_obj(&[
            ("name", crate::jstr(&self.name)),
            ("bw_stream_gbps", crate::jnum(self.bw_stream_gbps)),
            ("bw_tiers", format!("{{{}}}", tiers.join(", "))),
            ("default_tier", crate::jstr(&self.default_tier)),
            (
                "peak_gflops",
                self.peak_gflops.map(crate::jnum).unwrap_or("null".into()),
            ),
            ("cu_switch_reload_us", crate::jnum(self.cu_switch_reload_us)),
            ("submit_overhead_us", crate::jnum(self.submit_overhead_us)),
            ("provenance", crate::jstr(&self.provenance)),
        ])
    }

    /// 从 JSON 文本覆盖非默认字段（perf-calibrate 的产物）。
    /// 扁平对象 + 数字/字符串值，唯一允许的嵌套是 `bw_tiers` 的数字
    /// 对象；未知键忽略，缺失键保默认。
    pub fn overlay_json(mut self, text: &str) -> Result<Self, String> {
        for (k, v) in parse_flat_json(text)? {
            match k.as_str() {
                "name" => self.name = unquote(&v),
                "bw_stream_gbps" => self.bw_stream_gbps = num(&v)?,
                "bw_tiers" => {
                    let inner = v.trim();
                    if !inner.starts_with('{') {
                        return Err("bw_tiers must be an object".into());
                    }
                    // 按 key 合并：部分校准（只测得到一档）时其余档保默认。
                    for (tk, tv) in parse_flat_json(inner)? {
                        if let Ok(n) = num(&tv) {
                            self.bw_tiers.insert(tk, n);
                        }
                    }
                }
                "default_tier" => self.default_tier = unquote(&v),
                "peak_gflops" if v.trim() != "null" => self.peak_gflops = Some(num(&v)?),
                "peak_gflops" => self.peak_gflops = None,
                "cu_switch_reload_us" => self.cu_switch_reload_us = num(&v)?,
                "submit_overhead_us" => self.submit_overhead_us = num(&v)?,
                "provenance" => self.provenance = unquote(&v),
                _ => {}
            }
        }
        Ok(self)
    }
}

fn num(v: &str) -> Result<f64, String> {
    v.trim()
        .parse::<f64>()
        .map_err(|e| format!("bad number {v:?}: {e}"))
}

fn unquote(v: &str) -> String {
    let t = v.trim();
    t.trim_matches('"').to_string()
}

/// 极简扁平 JSON 对象解析：{"k": v, ...}，v 为数字或字符串；值里唯一
/// 允许的容器是 bw_tiers 那层嵌套数字对象（花括号配平后整段透传）。
/// 覆盖 machine_model.json 这一种用途，不追求通用。
fn parse_flat_json(text: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    let b: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    let skip_ws = |i: &mut usize, b: &[char]| {
        while *i < b.len() && b[*i].is_whitespace() {
            *i += 1;
        }
    };
    skip_ws(&mut i, &b);
    if i >= b.len() || b[i] != '{' {
        return Err("expected '{'".into());
    }
    i += 1;
    loop {
        skip_ws(&mut i, &b);
        if i < b.len() && b[i] == '}' {
            return Ok(out);
        }
        if i >= b.len() || b[i] != '"' {
            return Err(format!("expected key at {i}"));
        }
        let mut key = String::new();
        i += 1;
        while i < b.len() && b[i] != '"' {
            key.push(b[i]);
            i += 1;
        }
        i += 1; // closing quote
        skip_ws(&mut i, &b);
        if i >= b.len() || b[i] != ':' {
            return Err(format!("expected ':' at {i}"));
        }
        i += 1;
        skip_ws(&mut i, &b);
        if i < b.len() && b[i] == '"' {
            let mut val = String::new();
            i += 1;
            while i < b.len() && b[i] != '"' {
                val.push(b[i]);
                i += 1;
            }
            i += 1;
            out.push((key, format!("\"{val}\"")));
        } else {
            let start = i;
            if i < b.len() && b[i] == '{' {
                // 嵌套对象：花括号配平后整段作为值（bw_tiers 用）。
                let mut depth = 0usize;
                while i < b.len() {
                    match b[i] {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                i += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
                if depth != 0 {
                    return Err("unbalanced braces in nested object".into());
                }
            } else {
                while i < b.len() && b[i] != ',' && b[i] != '}' {
                    i += 1;
                }
            }
            out.push((key, b[start..i].iter().collect()));
        }
        skip_ws(&mut i, &b);
        if i < b.len() && b[i] == ',' {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_roundtrip() {
        let m = MachineModel::default()
            .overlay_json(r#"{"bw_stream_gbps": 31.5, "peak_gflops": null}"#)
            .unwrap();
        assert!((m.bw_stream_gbps - 31.5).abs() < 1e-9);
        assert!(m.peak_gflops.is_none());
        // 缺失键保默认
        assert!((m.cu_switch_reload_us - 650.0).abs() < 1e-9);
    }

    #[test]
    fn overlay_bad_input_errors() {
        assert!(MachineModel::default().overlay_json("not json").is_err());
    }

    #[test]
    fn overlay_bw_tiers_nested_object() {
        let m = MachineModel::default()
            .overlay_json(
                r#"{"bw_tiers": {"slot-stream": 39.8, "strided": 1.3}, "default_tier": "slot-stream", "submit_overhead_us": 48.0}"#,
            )
            .unwrap();
        assert!((m.bw_tiers["slot-stream"] - 39.8).abs() < 1e-9);
        assert!((m.bw_tiers["strided"] - 1.3).abs() < 1e-9);
        // 未覆盖的档保默认
        assert!((m.bw_tiers["seq-dma"] - 52.0).abs() < 1e-9);
        assert_eq!(m.default_tier, "slot-stream");
        assert!((m.submit_overhead_us - 48.0).abs() < 1e-9);
        // 覆盖后可再序列化 / 再解析（calibrate → 磁盘 → 下次 overlay）
        let rt = MachineModel::default()
            .overlay_json(&m.to_json_value())
            .unwrap();
        assert!((rt.bw_tiers["slot-stream"] - 39.8).abs() < 1e-9);
    }

    #[test]
    fn bw_for_tier_lookup() {
        let m = MachineModel::default();
        assert!((m.bw_for(Some(tier::SEQ_DMA)) - 52.0).abs() < 1e-9);
        assert!((m.bw_for(Some(tier::STRIDED)) - 1.2).abs() < 1e-9);
        // 未知档 → default_tier；None → default_tier
        assert!((m.bw_for(Some("nope")) - m.bw_for(None)).abs() < 1e-9);
        assert!((m.bw_for(None) - 43.0).abs() < 1e-9);
        // 空 tiers + 缺 default → 遗留单锚兜底
        let mut bare = m.clone();
        bare.bw_tiers.clear();
        bare.default_tier = "gone".into();
        assert!((bare.bw_for(None) - 33.3).abs() < 1e-9);
    }
}
