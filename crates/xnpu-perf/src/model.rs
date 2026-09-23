//! 机器模型：把设备相关常数隔离在一处，可被校准数据替换。
//!
//! 默认值全部来自 M2–M4a 在本机（XDNA2 / npu2）的实测锚点，出处写在
//! 每个字段注释里；`perf-calibrate`（M5a 后续）重新测量后可从 JSON 覆盖。

use crate::json_obj;

/// 一台 NPU 的性能天花板与机制代价（单位见字段名）。
#[derive(Clone, Debug)]
pub struct MachineModel {
    pub name: String,
    /// 同 op burst 连发下的权重流带宽上限（GB/s）。
    /// 实测：2026-09-23 run-w4ulayer 42L per-layer 模式（全程 syncobj wait）
    /// slot 流 33.3 GB/s；旧锚 24.4 为 M3b pipelined state-poll 口径，已过时。
    pub bw_stream_gbps: f64,
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
            peak_gflops: None,
            cu_switch_reload_us: 650.0,
            submit_overhead_us: 55.0,
            provenance: "实测锚点（未校准默认值）：bw=2026-09-23 run-w4ulayer 42L per-layer syncobj 口径 slot 流 33.3 GB/s；cu_switch=run-multi burst/interleaved 差；submit=run-pipe solo−burst 差".to_string(),
        }
    }
}

impl MachineModel {
    pub fn to_json_value(&self) -> String {
        json_obj(&[
            ("name", crate::jstr(&self.name)),
            ("bw_stream_gbps", crate::jnum(self.bw_stream_gbps)),
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
    /// 只认扁平对象 + 数字/字符串值；未知键忽略，缺失键保默认。
    pub fn overlay_json(mut self, text: &str) -> Result<Self, String> {
        for (k, v) in parse_flat_json(text)? {
            match k.as_str() {
                "name" => self.name = unquote(&v),
                "bw_stream_gbps" => self.bw_stream_gbps = num(&v)?,
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

/// 极简扁平 JSON 对象解析：{"k": v, ...}，v 为数字或字符串。
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
            while i < b.len() && b[i] != ',' && b[i] != '}' {
                i += 1;
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
}
