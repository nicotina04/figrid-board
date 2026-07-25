//! Deterministic CB-QAT1 P0 quantization-headroom statistics.
//!
//! The caller owns corpus replay and evaluator correctness. This module accepts
//! only the resulting six-candidate utilities and applies the preregistered
//! CE, ranking, drift, component-bootstrap, and open/stop rules.

use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const K6: usize = 6;
pub(crate) const EXPECTED_SLATES: usize = 1_336;
pub(crate) const EXPECTED_COMPONENTS: usize = 388;
pub(crate) const EXPECTED_ROWS_PER_COLOR: usize = 668;
pub(crate) const BOOTSTRAP_REPLICATES: usize = 100_000;
pub(crate) const BOOTSTRAP_SEED: u64 = 2_026_726_001;

pub(crate) const GO_LABEL: &str = "GO_PAIRED_QAT_TRAIN";
pub(crate) const NO_GO_LABEL: &str = "NO_GO_PRECONDITION";
pub(crate) const INVALID_LABEL: &str = "INVALID_QAT1_P0";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum RootColor {
    Black,
    White,
}

impl RootColor {
    fn index(self) -> usize {
        match self {
            Self::Black => 0,
            Self::White => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct P0Slate {
    pub(crate) row_uid: String,
    pub(crate) component_uid: String,
    pub(crate) root_color: RootColor,
    pub(crate) ordinal: u8,
    pub(crate) q_teacher: [f64; K6],
    pub(crate) teacher_top: [bool; K6],
    pub(crate) fp32_utilities: [f64; K6],
    pub(crate) ptq_utilities: [f64; K6],
}

#[derive(Clone, Debug)]
pub(crate) struct AnalysisOutcome {
    pub(crate) final_label: &'static str,
    pub(crate) report: Value,
}

#[derive(Clone, Copy, Debug, Default)]
struct Neumaier {
    sum: f64,
    correction: f64,
}

impl Neumaier {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum + self.correction
    }
}

#[derive(Clone, Copy, Debug)]
struct RowMetrics {
    fp32_ce: f64,
    ptq_ce: f64,
    delta_ce: f64,
    fp32_top: usize,
    ptq_top: usize,
    q_top: usize,
    teacher_top: usize,
    pair_order_disagreements: u64,
    q_transition: QTransition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QTransition {
    None,
    Fp32Superior,
    PtqSuperior,
    Equal,
}

#[derive(Clone, Copy, Debug, Default)]
struct Aggregate {
    slates: u64,
    fp32_ce: Neumaier,
    ptq_ce: Neumaier,
    delta_ce: Neumaier,
    top1_disagreements: u64,
    pair_order_disagreements: u64,
    fp32_q_superior: u64,
    ptq_q_superior: u64,
    q_equal: u64,
    fp32_q_argmax_correct: u64,
    ptq_q_argmax_correct: u64,
    fp32_teacher_top_correct: u64,
    ptq_teacher_top_correct: u64,
}

impl Aggregate {
    fn observe(&mut self, row: RowMetrics) -> Result<(), String> {
        self.slates = self
            .slates
            .checked_add(1)
            .ok_or_else(|| "aggregate slate count overflow".to_string())?;
        self.fp32_ce.add(row.fp32_ce);
        self.ptq_ce.add(row.ptq_ce);
        self.delta_ce.add(row.delta_ce);
        self.pair_order_disagreements = self
            .pair_order_disagreements
            .checked_add(row.pair_order_disagreements)
            .ok_or_else(|| "pair-order disagreement count overflow".to_string())?;
        if row.fp32_top != row.ptq_top {
            self.top1_disagreements = self
                .top1_disagreements
                .checked_add(1)
                .ok_or_else(|| "top-1 disagreement count overflow".to_string())?;
        }
        match row.q_transition {
            QTransition::None => {}
            QTransition::Fp32Superior => self.fp32_q_superior += 1,
            QTransition::PtqSuperior => self.ptq_q_superior += 1,
            QTransition::Equal => self.q_equal += 1,
        }
        self.fp32_q_argmax_correct += u64::from(row.fp32_top == row.q_top);
        self.ptq_q_argmax_correct += u64::from(row.ptq_top == row.q_top);
        self.fp32_teacher_top_correct += u64::from(row.fp32_top == row.teacher_top);
        self.ptq_teacher_top_correct += u64::from(row.ptq_top == row.teacher_top);
        Ok(())
    }

    fn mean_delta(self) -> Result<f64, String> {
        mean(self.delta_ce, self.slates, "aggregate delta CE")
    }

    fn q_net(self) -> i64 {
        self.fp32_q_superior as i64 - self.ptq_q_superior as i64
    }

    fn json(self) -> Result<Value, String> {
        if self.slates == 0 {
            return Err("cannot report an empty aggregate".to_string());
        }
        Ok(json!({
            "slates": self.slates,
            "fp32_ce": mean(self.fp32_ce, self.slates, "FP32 CE")?,
            "ptq_ce": mean(self.ptq_ce, self.slates, "PTQ CE")?,
            "delta_ce_ptq_minus_fp32": self.mean_delta()?,
            "top1_disagreements": self.top1_disagreements,
            "pair_order_disagreements": self.pair_order_disagreements,
            "q_on_top1_disagreement": {
                "fp32_q_superior": self.fp32_q_superior,
                "ptq_q_superior": self.ptq_q_superior,
                "q_equal": self.q_equal,
                "fp32_minus_ptq_net": self.q_net(),
            },
            "q_argmax_accuracy_lowest_index_tie_break": {
                "fp32_correct": self.fp32_q_argmax_correct,
                "ptq_correct": self.ptq_q_argmax_correct,
                "fp32_fraction": self.fp32_q_argmax_correct as f64 / self.slates as f64,
                "ptq_fraction": self.ptq_q_argmax_correct as f64 / self.slates as f64,
            },
            "stored_teacher_top_accuracy": {
                "fp32_correct": self.fp32_teacher_top_correct,
                "ptq_correct": self.ptq_teacher_top_correct,
                "fp32_fraction": self.fp32_teacher_top_correct as f64 / self.slates as f64,
                "ptq_fraction": self.ptq_teacher_top_correct as f64 / self.slates as f64,
            }
        }))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ComponentDelta {
    sum: Neumaier,
    slates: u64,
}

impl ComponentDelta {
    fn observe(&mut self, value: f64) -> Result<(), String> {
        require_finite("component row delta", value)?;
        self.sum.add(value);
        self.slates = self
            .slates
            .checked_add(1)
            .ok_or_else(|| "component slate count overflow".to_string())?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct DriftSummary {
    count: usize,
    p50: f64,
    p90: f64,
    p95: f64,
    p99: f64,
    max: f64,
}

impl DriftSummary {
    fn json(self) -> Value {
        json!({
            "count": self.count,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "max": self.max,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct GateInputs {
    prerequisite_mismatches: u64,
    combined_delta: f64,
    bootstrap_p10: f64,
    black_delta: f64,
    white_delta: f64,
    top1_disagreements: u64,
    combined_q_net: i64,
    black_q_net: i64,
    white_q_net: i64,
}

fn registered_gate_values(inputs: GateInputs) -> [(&'static str, bool); 9] {
    [
        (
            "prerequisite_mismatches_eq_0",
            inputs.prerequisite_mismatches == 0,
        ),
        ("combined_point_delta_ce_gt_0", inputs.combined_delta > 0.0),
        (
            "component_bootstrap_p10_delta_ce_gt_0",
            inputs.bootstrap_p10 > 0.0,
        ),
        ("black_point_delta_ce_ge_0", inputs.black_delta >= 0.0),
        ("white_point_delta_ce_ge_0", inputs.white_delta >= 0.0),
        ("top1_disagreements_ge_7", inputs.top1_disagreements >= 7),
        ("combined_fp32_q_net_ge_2", inputs.combined_q_net >= 2),
        ("black_fp32_q_net_ge_0", inputs.black_q_net >= 0),
        ("white_fp32_q_net_ge_0", inputs.white_q_net >= 0),
    ]
}

pub(crate) fn analyze(
    slates: &[P0Slate],
    prerequisite_mismatches: u64,
) -> Result<AnalysisOutcome, String> {
    if prerequisite_mismatches != 0 {
        return Ok(AnalysisOutcome {
            final_label: INVALID_LABEL,
            report: json!({
                "stage": "CB_QAT1_P0",
                "final_label": INVALID_LABEL,
                "prerequisite_mismatches": prerequisite_mismatches,
                "statistics_evaluated": false,
                "all_gates_pass": false,
                "next_stage": "STOP_CB_QAT1",
            }),
        });
    }
    validate_shape(slates)?;

    let mut combined = Aggregate::default();
    let mut colors = [Aggregate::default(); 2];
    let mut ordinals = BTreeMap::<u8, Aggregate>::new();
    let mut components = BTreeMap::<String, Aggregate>::new();
    let mut component_deltas = BTreeMap::<String, ComponentDelta>::new();
    let mut logit_drifts = Vec::with_capacity(slates.len() * K6);
    let mut probability_drifts = Vec::with_capacity(slates.len() * K6);

    for slate in slates {
        let (row, row_logit_drifts, row_probability_drifts) = row_metrics(slate)?;
        combined.observe(row)?;
        colors[slate.root_color.index()].observe(row)?;
        ordinals.entry(slate.ordinal).or_default().observe(row)?;
        components
            .entry(slate.component_uid.clone())
            .or_default()
            .observe(row)?;
        component_deltas
            .entry(slate.component_uid.clone())
            .or_default()
            .observe(row.delta_ce)?;
        logit_drifts.extend(row_logit_drifts);
        probability_drifts.extend(row_probability_drifts);
    }

    let bootstrap =
        bootstrap_component_delta(&component_deltas, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)?;
    let gate_inputs = GateInputs {
        prerequisite_mismatches: 0,
        combined_delta: combined.mean_delta()?,
        bootstrap_p10: bootstrap.p10,
        black_delta: colors[RootColor::Black.index()].mean_delta()?,
        white_delta: colors[RootColor::White.index()].mean_delta()?,
        top1_disagreements: combined.top1_disagreements,
        combined_q_net: combined.q_net(),
        black_q_net: colors[RootColor::Black.index()].q_net(),
        white_q_net: colors[RootColor::White.index()].q_net(),
    };
    let gate_values = registered_gate_values(gate_inputs);
    let all_gates_pass = gate_values.iter().all(|(_, pass)| *pass);
    let final_label = if all_gates_pass {
        GO_LABEL
    } else {
        NO_GO_LABEL
    };

    let ordinal_json = ordinals
        .into_iter()
        .map(|(ordinal, aggregate)| Ok((ordinal.to_string(), aggregate.json()?)))
        .collect::<Result<Map<String, Value>, String>>()?;
    let component_json = components
        .into_iter()
        .map(|(uid, aggregate)| Ok((uid, aggregate.json()?)))
        .collect::<Result<Map<String, Value>, String>>()?;
    let gates = gate_values
        .into_iter()
        .map(|(name, pass)| {
            let observed = match name {
                "prerequisite_mismatches_eq_0" => {
                    json!(gate_inputs.prerequisite_mismatches)
                }
                "combined_point_delta_ce_gt_0" => json!(gate_inputs.combined_delta),
                "component_bootstrap_p10_delta_ce_gt_0" => {
                    json!(gate_inputs.bootstrap_p10)
                }
                "black_point_delta_ce_ge_0" => json!(gate_inputs.black_delta),
                "white_point_delta_ce_ge_0" => json!(gate_inputs.white_delta),
                "top1_disagreements_ge_7" => json!(gate_inputs.top1_disagreements),
                "combined_fp32_q_net_ge_2" => json!(gate_inputs.combined_q_net),
                "black_fp32_q_net_ge_0" => json!(gate_inputs.black_q_net),
                "white_fp32_q_net_ge_0" => json!(gate_inputs.white_q_net),
                _ => Value::Null,
            };
            let (operator, threshold) = match name {
                "prerequisite_mismatches_eq_0" => ("==", json!(0)),
                "combined_point_delta_ce_gt_0" | "component_bootstrap_p10_delta_ce_gt_0" => {
                    (">", json!(0.0))
                }
                "black_point_delta_ce_ge_0" | "white_point_delta_ce_ge_0" => (">=", json!(0.0)),
                "top1_disagreements_ge_7" => (">=", json!(7)),
                "combined_fp32_q_net_ge_2" => (">=", json!(2)),
                "black_fp32_q_net_ge_0" | "white_fp32_q_net_ge_0" => (">=", json!(0)),
                _ => ("invalid", Value::Null),
            };
            json!({
                "name": name,
                "observed": observed,
                "operator": operator,
                "threshold": threshold,
                "pass": pass
            })
        })
        .collect::<Vec<_>>();

    Ok(AnalysisOutcome {
        final_label,
        report: json!({
            "stage": "CB_QAT1_P0",
            "final_label": final_label,
            "prerequisite_mismatches": 0,
            "inventory": {
                "slates": slates.len(),
                "candidates": slates.len() * K6,
                "components": component_deltas.len(),
                "black_slates": colors[RootColor::Black.index()].slates,
                "white_slates": colors[RootColor::White.index()].slates,
                "ordinals": [1, 2, 4, 6, 8],
            },
            "aggregates": {
                "combined": combined.json()?,
                "black": colors[RootColor::Black.index()].json()?,
                "white": colors[RootColor::White.index()].json()?,
                "ordinals": ordinal_json,
                "components": component_json,
            },
            "drift": {
                "absolute_logit": describe_drift(logit_drifts)?.json(),
                "absolute_probability": describe_drift(probability_drifts)?.json(),
                "percentile_rule": "nearest-rank ceil(p*N)-1 after f64::total_cmp",
            },
            "component_bootstrap": bootstrap.json(),
            "gates": gates,
            "all_gates_pass": all_gates_pass,
            "next_stage": if all_gates_pass {
                "RUN_ONE_REGISTERED_PAIRED_PTQ_QAT_FIT"
            } else {
                "STOP_CB_QAT1_AND_ADVANCE_TO_CB_AL1"
            },
        }),
    })
}

fn validate_shape(slates: &[P0Slate]) -> Result<(), String> {
    if slates.len() != EXPECTED_SLATES {
        return Err(format!(
            "P0 slate count {}, expected {EXPECTED_SLATES}",
            slates.len()
        ));
    }
    let mut rows = BTreeSet::<&str>::new();
    let mut components = BTreeSet::<&str>::new();
    let mut component_colors = BTreeMap::<&str, [u64; 2]>::new();
    let mut colors = [0usize; 2];
    let mut ordinals = BTreeSet::<u8>::new();
    for slate in slates {
        validate_uppercase_uid("row_uid", &slate.row_uid)?;
        validate_uppercase_uid("component_uid", &slate.component_uid)?;
        if !rows.insert(&slate.row_uid) {
            return Err(format!("duplicate row_uid {}", slate.row_uid));
        }
        components.insert(&slate.component_uid);
        component_colors.entry(&slate.component_uid).or_default()[slate.root_color.index()] += 1;
        colors[slate.root_color.index()] += 1;
        ordinals.insert(slate.ordinal);
        validate_q(&slate.row_uid, &slate.q_teacher)?;
        if slate
            .teacher_top
            .iter()
            .filter(|&&selected| selected)
            .count()
            != 1
        {
            return Err(format!(
                "{} must contain exactly one teacher_top",
                slate.row_uid
            ));
        }
        for (arm, utilities) in [
            ("FP32", &slate.fp32_utilities),
            ("PTQ", &slate.ptq_utilities),
        ] {
            for (candidate, &utility) in utilities.iter().enumerate() {
                require_finite(
                    &format!("{} {arm} utility[{candidate}]", slate.row_uid),
                    utility,
                )?;
            }
        }
    }
    if components.len() != EXPECTED_COMPONENTS {
        return Err(format!(
            "component count {}, expected {EXPECTED_COMPONENTS}",
            components.len()
        ));
    }
    if colors != [EXPECTED_ROWS_PER_COLOR; 2] {
        return Err(format!(
            "Black/White counts {}/{}, expected {EXPECTED_ROWS_PER_COLOR} each",
            colors[0], colors[1]
        ));
    }
    if let Some((uid, counts)) = component_colors
        .iter()
        .find(|(_, counts)| counts[0] == 0 || counts[1] == 0)
    {
        return Err(format!(
            "component {uid} lacks a paired color: Black/White={}/{}",
            counts[0], counts[1]
        ));
    }
    let expected_ordinals = BTreeSet::from([1u8, 2, 4, 6, 8]);
    if ordinals != expected_ordinals {
        return Err(format!(
            "ordinal inventory {ordinals:?}, expected {expected_ordinals:?}"
        ));
    }
    Ok(())
}

fn validate_uppercase_uid(field: &str, uid: &str) -> Result<(), String> {
    if uid.is_empty()
        || !uid.is_ascii()
        || uid
            .bytes()
            .any(|byte| byte.is_ascii_lowercase() || byte.is_ascii_whitespace())
    {
        return Err(format!(
            "{field} must be non-empty uppercase raw ASCII, got {uid:?}"
        ));
    }
    Ok(())
}

fn validate_q(row_uid: &str, q: &[f64; K6]) -> Result<(), String> {
    let mut sum = Neumaier::default();
    for (candidate, &value) in q.iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "{row_uid} q_teacher[{candidate}] is not a finite probability"
            ));
        }
        sum.add(value);
    }
    let total = sum.total();
    if (total - 1.0).abs() > 1.0e-12 {
        return Err(format!(
            "{row_uid} q_teacher sum {total}, expected one without renormalization"
        ));
    }
    Ok(())
}

fn row_metrics(slate: &P0Slate) -> Result<(RowMetrics, [f64; K6], [f64; K6]), String> {
    let fp32_ce = cross_entropy(&slate.q_teacher, &slate.fp32_utilities)?;
    let ptq_ce = cross_entropy(&slate.q_teacher, &slate.ptq_utilities)?;
    let delta_ce = ptq_ce - fp32_ce;
    require_finite("row delta CE", delta_ce)?;
    let fp32_probabilities = softmax(&slate.fp32_utilities)?;
    let ptq_probabilities = softmax(&slate.ptq_utilities)?;
    let fp32_top = top1_lowest_index(&slate.fp32_utilities)?;
    let ptq_top = top1_lowest_index(&slate.ptq_utilities)?;
    let q_top = top1_lowest_index(&slate.q_teacher)?;
    let teacher_top = slate
        .teacher_top
        .iter()
        .position(|&selected| selected)
        .ok_or_else(|| "teacher_top is empty".to_string())?;
    let q_transition = if fp32_top == ptq_top {
        QTransition::None
    } else {
        match slate.q_teacher[fp32_top]
            .partial_cmp(&slate.q_teacher[ptq_top])
            .ok_or_else(|| "q transition comparison was unordered".to_string())?
        {
            std::cmp::Ordering::Greater => QTransition::Fp32Superior,
            std::cmp::Ordering::Less => QTransition::PtqSuperior,
            std::cmp::Ordering::Equal => QTransition::Equal,
        }
    };
    let mut pair_order_disagreements = 0u64;
    for left in 0..K6 {
        for right in (left + 1)..K6 {
            let fp32_order = slate.fp32_utilities[left]
                .partial_cmp(&slate.fp32_utilities[right])
                .ok_or_else(|| "FP32 pair ordering was unordered".to_string())?;
            let ptq_order = slate.ptq_utilities[left]
                .partial_cmp(&slate.ptq_utilities[right])
                .ok_or_else(|| "PTQ pair ordering was unordered".to_string())?;
            pair_order_disagreements += u64::from(fp32_order != ptq_order);
        }
    }
    let logit_drift = std::array::from_fn(|index| {
        (slate.ptq_utilities[index] - slate.fp32_utilities[index]).abs()
    });
    let probability_drift =
        std::array::from_fn(|index| (ptq_probabilities[index] - fp32_probabilities[index]).abs());
    if logit_drift
        .iter()
        .chain(&probability_drift)
        .any(|value| !value.is_finite())
    {
        return Err("non-finite row drift".to_string());
    }
    Ok((
        RowMetrics {
            fp32_ce,
            ptq_ce,
            delta_ce,
            fp32_top,
            ptq_top,
            q_top,
            teacher_top,
            pair_order_disagreements,
            q_transition,
        },
        logit_drift,
        probability_drift,
    ))
}

fn cross_entropy(q: &[f64; K6], utilities: &[f64; K6]) -> Result<f64, String> {
    let max = finite_max(utilities)?;
    let mut exponential_sum = Neumaier::default();
    for &utility in utilities {
        exponential_sum.add((utility - max).exp());
    }
    let exponential_sum = exponential_sum.total();
    if !exponential_sum.is_finite() || exponential_sum <= 0.0 {
        return Err(format!("invalid softmax exponential sum {exponential_sum}"));
    }
    let log_sum_exp = max + exponential_sum.ln();
    let mut ce = Neumaier::default();
    for candidate in 0..K6 {
        ce.add(q[candidate] * (log_sum_exp - utilities[candidate]));
    }
    let ce = ce.total();
    require_finite("cross-entropy", ce)?;
    if ce < 0.0 {
        return Err(format!("negative cross-entropy {ce}"));
    }
    Ok(ce)
}

fn softmax(utilities: &[f64; K6]) -> Result<[f64; K6], String> {
    let max = finite_max(utilities)?;
    let values = utilities.map(|utility| (utility - max).exp());
    let mut sum = Neumaier::default();
    for value in values {
        sum.add(value);
    }
    let denominator = sum.total();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(format!("invalid softmax denominator {denominator}"));
    }
    let result = values.map(|value| value / denominator);
    if result.iter().any(|value| !value.is_finite()) {
        return Err("non-finite softmax result".to_string());
    }
    Ok(result)
}

fn finite_max(values: &[f64; K6]) -> Result<f64, String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("non-finite six-way value".to_string());
    }
    Ok(values.iter().copied().fold(f64::NEG_INFINITY, f64::max))
}

fn top1_lowest_index(values: &[f64; K6]) -> Result<usize, String> {
    finite_max(values)?;
    let mut best = 0usize;
    for index in 1..K6 {
        if values[index] > values[best] {
            best = index;
        }
    }
    Ok(best)
}

fn mean(sum: Neumaier, count: u64, name: &str) -> Result<f64, String> {
    if count == 0 {
        return Err(format!("{name} has zero observations"));
    }
    let value = sum.total() / count as f64;
    require_finite(name, value)?;
    Ok(value)
}

fn require_finite(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} is non-finite: {value}"))
    }
}

fn describe_drift(mut values: Vec<f64>) -> Result<DriftSummary, String> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("drift stream is empty, negative, or non-finite".to_string());
    }
    values.sort_by(f64::total_cmp);
    Ok(DriftSummary {
        count: values.len(),
        p50: nearest_rank(&values, 50, 100),
        p90: nearest_rank(&values, 90, 100),
        p95: nearest_rank(&values, 95, 100),
        p99: nearest_rank(&values, 99, 100),
        max: values[values.len() - 1],
    })
}

fn nearest_rank(values: &[f64], numerator: usize, denominator: usize) -> f64 {
    let rank = numerator
        .checked_mul(values.len())
        .expect("registered percentile multiplication fits usize")
        .div_ceil(denominator);
    values[rank.saturating_sub(1).min(values.len() - 1)]
}

#[derive(Clone, Copy, Debug)]
struct BootstrapSummary {
    p10: f64,
    p50: f64,
    min: f64,
    max: f64,
}

impl BootstrapSummary {
    fn json(self) -> Value {
        json!({
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "components_sampled_per_replicate": EXPECTED_COMPONENTS,
            "sampling": "next_u64()%388 over uppercase-UID lexicographic components",
            "point_statistic": "equal-slate mean delta_ce",
            "p10_index_zero_based": 9_999,
            "min": self.min,
            "p10": self.p10,
            "p50": self.p50,
            "max": self.max,
        })
    }
}

fn bootstrap_component_delta(
    components: &BTreeMap<String, ComponentDelta>,
    replicates: usize,
    seed: u64,
) -> Result<BootstrapSummary, String> {
    if components.len() != EXPECTED_COMPONENTS {
        return Err(format!(
            "bootstrap component count {}, expected {EXPECTED_COMPONENTS}",
            components.len()
        ));
    }
    if replicates == 0 {
        return Err("bootstrap replicate count is zero".to_string());
    }
    let ordered = components.values().copied().collect::<Vec<_>>();
    if ordered.iter().any(|component| component.slates == 0) {
        return Err("bootstrap received an empty component".to_string());
    }
    let mut rng = SplitMix64::new(seed);
    let mut distribution = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut sum = Neumaier::default();
        let mut rows = 0u64;
        for _ in 0..EXPECTED_COMPONENTS {
            let index = (rng.next_u64() % EXPECTED_COMPONENTS as u64) as usize;
            let component = ordered[index];
            sum.add(component.sum.total());
            rows = rows
                .checked_add(component.slates)
                .ok_or_else(|| "bootstrap row count overflow".to_string())?;
        }
        distribution.push(mean(sum, rows, "bootstrap delta CE")?);
    }
    distribution.sort_by(f64::total_cmp);
    let p10_index = 10usize
        .checked_mul(replicates)
        .expect("bootstrap percentile multiplication fits usize")
        .div_ceil(100)
        .saturating_sub(1);
    Ok(BootstrapSummary {
        min: distribution[0],
        p10: distribution[p10_index],
        p50: nearest_rank(&distribution, 50, 100),
        max: distribution[distribution.len() - 1],
    })
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_q() -> [f64; K6] {
        [1.0 / 6.0; K6]
    }

    fn slate(q: [f64; K6], fp32: [f64; K6], ptq: [f64; K6], teacher: usize) -> P0Slate {
        P0Slate {
            row_uid: "ROW".to_string(),
            component_uid: "COMP".to_string(),
            root_color: RootColor::Black,
            ordinal: 1,
            q_teacher: q,
            teacher_top: std::array::from_fn(|index| index == teacher),
            fp32_utilities: fp32,
            ptq_utilities: ptq,
        }
    }

    #[test]
    fn uniform_zero_logits_have_ln_six_cross_entropy() {
        let ce = cross_entropy(&uniform_q(), &[0.0; K6]).unwrap();
        assert_eq!(ce.to_bits(), (6.0f64).ln().to_bits());
    }

    #[test]
    fn delta_sign_is_ptq_minus_fp32() {
        let q = [0.70, 0.06, 0.06, 0.06, 0.06, 0.06];
        let row = slate(q, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0; K6], 0);
        let metrics = row_metrics(&row).unwrap().0;
        assert!(metrics.delta_ce > 0.0);
    }

    #[test]
    fn top1_ties_choose_lowest_index() {
        assert_eq!(
            top1_lowest_index(&[3.0, 3.0, 2.0, 1.0, 0.0, -1.0]).unwrap(),
            0
        );
    }

    #[test]
    fn q_superiority_categories_are_oriented_correctly() {
        let q = [0.6, 0.2, 0.05, 0.05, 0.05, 0.05];
        let fp_better = slate(
            q,
            [2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            0,
        );
        assert_eq!(
            row_metrics(&fp_better).unwrap().0.q_transition,
            QTransition::Fp32Superior
        );
        let ptq_better = slate(
            q,
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            0,
        );
        assert_eq!(
            row_metrics(&ptq_better).unwrap().0.q_transition,
            QTransition::PtqSuperior
        );
        let equal_q = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1];
        let equal = slate(
            equal_q,
            [2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            0,
        );
        assert_eq!(
            row_metrics(&equal).unwrap().0.q_transition,
            QTransition::Equal
        );
    }

    #[test]
    fn all_fifteen_pair_orders_are_compared() {
        let row = slate(
            uniform_q(),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            0,
        );
        assert_eq!(row_metrics(&row).unwrap().0.pair_order_disagreements, 15);
    }

    #[test]
    fn nearest_rank_boundaries_use_ceil_minus_one() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(nearest_rank(&values, 50, 100), 3.0);
        assert_eq!(nearest_rank(&values, 90, 100), 5.0);
        assert_eq!(nearest_rank(&values, 99, 100), 5.0);
    }

    #[test]
    fn splitmix_and_small_bootstrap_are_deterministic() {
        let mut known = SplitMix64::new(0);
        assert_eq!(known.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(known.next_u64(), 0x6E78_9E6A_A1B9_65F4);

        let mut components = BTreeMap::new();
        for index in 0..EXPECTED_COMPONENTS {
            let mut component = ComponentDelta::default();
            component.observe(index as f64 / 10_000.0).unwrap();
            components.insert(format!("C{index:03}"), component);
        }
        let a = bootstrap_component_delta(&components, 32, BOOTSTRAP_SEED).unwrap();
        let b = bootstrap_component_delta(&components, 32, BOOTSTRAP_SEED).unwrap();
        assert_eq!(a.p10.to_bits(), b.p10.to_bits());
        assert_eq!(a.p50.to_bits(), b.p50.to_bits());
    }

    #[test]
    fn gate_boundaries_are_strict_where_registered() {
        let passing = GateInputs {
            prerequisite_mismatches: 0,
            combined_delta: f64::EPSILON,
            bootstrap_p10: f64::EPSILON,
            black_delta: 0.0,
            white_delta: 0.0,
            top1_disagreements: 7,
            combined_q_net: 2,
            black_q_net: 0,
            white_q_net: 0,
        };
        assert!(
            registered_gate_values(passing)
                .iter()
                .all(|(_, pass)| *pass)
        );

        let mut failing = passing;
        failing.prerequisite_mismatches = 1;
        assert!(!registered_gate_values(failing)[0].1);
        failing = passing;
        failing.combined_delta = 0.0;
        assert!(!registered_gate_values(failing)[1].1);
        failing = passing;
        failing.bootstrap_p10 = 0.0;
        assert!(!registered_gate_values(failing)[2].1);
        failing = passing;
        failing.top1_disagreements = 6;
        assert!(!registered_gate_values(failing)[5].1);
        failing = passing;
        failing.combined_q_net = 1;
        assert!(!registered_gate_values(failing)[6].1);
    }

    #[test]
    fn nonfinite_values_fail_closed() {
        assert!(cross_entropy(&uniform_q(), &[f64::NAN; K6]).is_err());
        assert!(describe_drift(vec![0.0, f64::INFINITY]).is_err());
    }

    #[test]
    fn prerequisite_mismatch_has_invalid_label_without_statistics() {
        let outcome = analyze(&[], 1).unwrap();
        assert_eq!(outcome.final_label, INVALID_LABEL);
        assert_eq!(
            outcome.report["statistics_evaluated"].as_bool(),
            Some(false)
        );
    }
}
