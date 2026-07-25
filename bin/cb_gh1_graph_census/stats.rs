//! CB-GH1 P0 train-only residual statistics.
//!
//! This module deliberately accepts already-interned *exact* graph identities.
//! Graph construction, exact-byte collision checks, and A0 equivariance belong
//! to the caller.  The important boundary here is that `a1_census` never reads
//! `q_teacher` or `product_root_utility`.

use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const K6: usize = 6;
pub(crate) const EXPECTED_SLATES: usize = 1_336;
pub(crate) const EXPECTED_COMPONENTS: usize = 388;
pub(crate) const EXPECTED_ROWS_PER_COLOR: usize = 668;
pub(crate) const BOOTSTRAP_REPLICATES: usize = 100_000;
pub(crate) const BOOTSTRAP_SEED: u64 = 0xCB01_2026_0726_0011;

const RECURRENT_COMPONENTS: usize = 3;
const A1_COMBINED_MIN: f64 = 0.25;
const A1_COLOR_MIN: f64 = 0.15;
const A1_ORDINAL_MIN: f64 = 0.10;
const A2_RELATIVE_MIN: f64 = 0.03;
const A2_QTOP_PP_MIN: f64 = 1.00;
const KL_NEGATIVE_INVALID: f64 = -1.0e-12;

pub(crate) const STATE_EXPLOSION_LABEL: &str = "NO_GO_STATE_EXPLOSION";
pub(crate) const NO_SIGNAL_LABEL: &str = "STOP_NO_GRAPH_SIGNAL";
pub(crate) const OPEN_INCREMENTAL_LABEL: &str = "OPEN_GH1_INCREMENTAL_GATE";

/// Root color is only a reporting stratum. Graph codes must already be
/// role-relative before entering this module.
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

/// Statistics input for one physical RQ615C train row.
///
/// `code_ids` must be dense or sparse identifiers interned by exact canonical
/// graph bytes, never by the prospective u64 digest alone.
#[derive(Clone, Debug)]
pub(crate) struct StatsSlate {
    pub(crate) row_uid: String,
    pub(crate) component_uid: String,
    pub(crate) root_color: RootColor,
    pub(crate) ordinal: u8,
    pub(crate) q_teacher: [f64; K6],
    pub(crate) product_root_utility: [f64; K6],
    pub(crate) code_ids: [u32; K6],
    /// Stored corpus role, kept separate from the probability-target truth.
    /// A1 never reads this field.
    pub(crate) teacher_top: [bool; K6],
    /// True when this exact graph code intentionally aliases more than one
    /// exact role-relative rooted transition. A1 never reads this field.
    pub(crate) code_is_abstraction: [bool; K6],
}

#[derive(Clone, Debug)]
pub(crate) struct AnalysisOutcome {
    pub(crate) final_label: &'static str,
    pub(crate) report: Value,
}

/// Run the registered sequential A1 -> A2 -> A3 decision.
///
/// Structural and numeric invalidity is returned as `Err`; the harness should
/// wrap that error in its A0 report and label the card `INVALID_CB_GH1_P0`.
/// Registered signal failures are normal `Ok` outcomes with their final label.
pub(crate) fn analyze(slates: &[StatsSlate]) -> Result<AnalysisOutcome, String> {
    validate_label_blind_shape(slates)?;

    // A1 is intentionally completed before q_teacher or product utility is
    // touched. Keep this call and early return above `prepare_rows`.
    let a1 = a1_census(slates)?;
    if !a1.passed {
        return Ok(AnalysisOutcome {
            final_label: STATE_EXPLOSION_LABEL,
            report: json!({
                "stage": "CB_GH1_P0",
                "final_label": STATE_EXPLOSION_LABEL,
                "stages_evaluated": ["A1_LABEL_BLIND_REUSE"],
                "a1": a1.report,
                "a2": Value::Null,
                "a3": Value::Null,
                "q_or_product_fields_read": false,
                "next_stage": "STOP_CB_GH1"
            }),
        });
    }

    let prepared = prepare_rows(slates, &a1.recurrent_codes, &a1.component_index)?;
    let full_corrections = fit_observation_weighted_recurrent(&prepared, &a1.recurrent_codes)?;
    let residual_dispersion =
        recurrent_residual_dispersion(&prepared, &a1.recurrent_codes, &full_corrections)?;
    let full = evaluate_projection(&prepared, |row, candidate| {
        full_corrections
            .get(&row.code_ids[candidate])
            .copied()
            .unwrap_or(0.0)
    })?;
    let a2 = a2_decision(&full, residual_dispersion)?;
    if !a2.passed {
        return Ok(AnalysisOutcome {
            final_label: NO_SIGNAL_LABEL,
            report: json!({
                "stage": "CB_GH1_P0",
                "final_label": NO_SIGNAL_LABEL,
                "stages_evaluated": [
                    "A1_LABEL_BLIND_REUSE",
                    "A2_RECURRENT_FULL_FIT"
                ],
                "a1": a1.report,
                "a2": a2.report,
                "a3": Value::Null,
                "all_code_projection": {
                    "computed": false,
                    "gating": false
                },
                "next_stage": "STOP_CB_GH1"
            }),
        });
    }

    let loo = build_loo_corrections(&prepared)?;
    let loo_projection =
        evaluate_projection(&prepared, |row, candidate| loo[row.row_index][candidate])?;
    let bootstrap = bootstrap_components(
        &prepared,
        &loo_projection.row_scores,
        EXPECTED_COMPONENTS,
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
    )?;
    let a3 = a3_decision(&loo_projection, &bootstrap)?;
    let final_label = if a3.passed {
        OPEN_INCREMENTAL_LABEL
    } else {
        NO_SIGNAL_LABEL
    };

    Ok(AnalysisOutcome {
        final_label,
        report: json!({
            "stage": "CB_GH1_P0",
            "final_label": final_label,
            "stages_evaluated": [
                "A1_LABEL_BLIND_REUSE",
                "A2_RECURRENT_FULL_FIT",
                "A3_COMPONENT_LOO"
            ],
            "a1": a1.report,
            "a2": a2.report,
            "a3": a3.report,
            "all_code_projection": {
                "computed": false,
                "gating": false
            },
            "next_stage": if a3.passed {
                "PREREGISTER_GH1_INCREMENTAL_CORRECTNESS_AND_COST"
            } else {
                "STOP_CB_GH1"
            }
        }),
    })
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

#[derive(Debug)]
struct A1Outcome {
    recurrent_codes: BTreeSet<u32>,
    component_index: BTreeMap<String, usize>,
    report: Value,
    passed: bool,
}

fn validate_label_blind_shape(slates: &[StatsSlate]) -> Result<(), String> {
    if slates.len() != EXPECTED_SLATES {
        return Err(format!(
            "train slate count {}, expected {EXPECTED_SLATES}",
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
        checked_increment(
            &mut component_colors.entry(&slate.component_uid).or_default()
                [slate.root_color.index()],
            1,
            "component color row count",
        )?;
        colors[slate.root_color.index()] = colors[slate.root_color.index()]
            .checked_add(1)
            .ok_or_else(|| "color row counter overflow".to_string())?;
        ordinals.insert(slate.ordinal);
    }
    if components.len() != EXPECTED_COMPONENTS {
        return Err(format!(
            "component count {}, expected {EXPECTED_COMPONENTS}",
            components.len()
        ));
    }
    if colors != [EXPECTED_ROWS_PER_COLOR; 2] {
        return Err(format!(
            "color counts Black/White={}/{}, expected {EXPECTED_ROWS_PER_COLOR} each",
            colors[0], colors[1]
        ));
    }
    if let Some((component, counts)) = component_colors
        .iter()
        .find(|(_, counts)| counts[0] == 0 || counts[1] == 0)
    {
        return Err(format!(
            "component {component} is not paired across root colors: Black/White={}/{}",
            counts[0], counts[1]
        ));
    }
    let expected_ordinals = BTreeSet::from([1u8, 2, 4, 6, 8]);
    if ordinals != expected_ordinals {
        return Err(format!(
            "ordinal set {:?}, expected {:?}",
            ordinals, expected_ordinals
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

fn a1_census(slates: &[StatsSlate]) -> Result<A1Outcome, String> {
    if slates.is_empty() {
        return Err("A1 cannot run on an empty corpus".to_string());
    }

    let component_names = slates
        .iter()
        .map(|slate| slate.component_uid.clone())
        .collect::<BTreeSet<_>>();
    let component_index = component_names
        .into_iter()
        .enumerate()
        .map(|(index, uid)| (uid, index))
        .collect::<BTreeMap<_, _>>();

    let mut component_support = BTreeMap::<u32, BTreeSet<usize>>::new();
    let mut observation_counts = BTreeMap::<u32, u64>::new();
    for slate in slates {
        let component = *component_index
            .get(&slate.component_uid)
            .ok_or_else(|| format!("missing component index for {}", slate.component_uid))?;
        for &code in &slate.code_ids {
            component_support.entry(code).or_default().insert(component);
            checked_increment(
                observation_counts.entry(code).or_default(),
                1,
                "code occurrence",
            )?;
        }
    }
    let recurrent_codes = component_support
        .iter()
        .filter_map(|(&code, support)| (support.len() >= RECURRENT_COMPONENTS).then_some(code))
        .collect::<BTreeSet<_>>();

    let mut combined = CoverageCounter::default();
    let mut colors = [CoverageCounter::default(); 2];
    let mut ordinals = BTreeMap::<u8, CoverageCounter>::new();
    for slate in slates {
        for &code in &slate.code_ids {
            let recurrent = recurrent_codes.contains(&code);
            combined.observe(recurrent)?;
            colors[slate.root_color.index()].observe(recurrent)?;
            ordinals
                .entry(slate.ordinal)
                .or_default()
                .observe(recurrent)?;
        }
    }

    let combined_fraction = combined.fraction()?;
    let color_fractions = [colors[0].fraction()?, colors[1].fraction()?];
    let ordinal_fractions = ordinals
        .iter()
        .map(|(&ordinal, counter)| Ok((ordinal, counter.fraction()?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;

    let mut occurrence_histogram = BTreeMap::<u64, u64>::new();
    let mut component_histogram = BTreeMap::<usize, u64>::new();
    for (&code, support) in &component_support {
        let observations = *observation_counts
            .get(&code)
            .ok_or_else(|| format!("missing observation count for code {code}"))?;
        checked_increment(
            occurrence_histogram.entry(observations).or_default(),
            1,
            "occurrence histogram",
        )?;
        checked_increment(
            component_histogram.entry(support.len()).or_default(),
            1,
            "component histogram",
        )?;
    }

    let gates = vec![
        fraction_gate(
            "combined_recurrent_fraction_ge_25pct",
            combined_fraction,
            A1_COMBINED_MIN,
        ),
        fraction_gate(
            "black_recurrent_fraction_ge_15pct",
            color_fractions[0],
            A1_COLOR_MIN,
        ),
        fraction_gate(
            "white_recurrent_fraction_ge_15pct",
            color_fractions[1],
            A1_COLOR_MIN,
        ),
        json!({
            "name": "every_ordinal_recurrent_fraction_ge_10pct",
            "threshold": A1_ORDINAL_MIN,
            "observed": ordinal_fractions
                .iter()
                .map(|(ordinal, fraction)| (ordinal.to_string(), json!(fraction)))
                .collect::<serde_json::Map<String, Value>>(),
            "pass": ordinal_fractions
                .values()
                .all(|&fraction| fraction >= A1_ORDINAL_MIN)
        }),
    ];
    let passed = gates_pass(&gates);

    let report = json!({
        "stage": "A1_LABEL_BLIND_REUSE",
        "status": if passed {
            "A1_PASS_A2_REQUIRED"
        } else {
            STATE_EXPLOSION_LABEL
        },
        "label_blind": true,
        "q_or_product_fields_read": false,
        "slates": slates.len(),
        "rooted_candidates": combined.total,
        "components": component_index.len(),
        "distinct_codes": component_support.len(),
        "recurrent_codes": recurrent_codes.len(),
        "recurrent_definition": {
            "minimum_distinct_components": RECURRENT_COMPONENTS,
            "observation_weighted": false
        },
        "code_multiplicity_distribution": {
            "observation_count_to_distinct_codes": integer_histogram_json(&occurrence_histogram),
            "component_count_to_distinct_codes": integer_histogram_json(&component_histogram)
        },
        "coverage": {
            "combined": coverage_json(combined, combined_fraction),
            "black": coverage_json(colors[0], color_fractions[0]),
            "white": coverage_json(colors[1], color_fractions[1]),
            "ordinals": ordinals
                .iter()
                .map(|(ordinal, counter)| {
                    (
                        ordinal.to_string(),
                        coverage_json(
                            *counter,
                            *ordinal_fractions
                                .get(ordinal)
                                .expect("ordinal fraction was built from the same map")
                        )
                    )
                })
                .collect::<serde_json::Map<String, Value>>()
        },
        "gates": gates,
        "all_a1_gates_pass": passed,
        "next_stage": if passed {
            "A2_RECURRENT_FULL_FIT"
        } else {
            "STOP_CB_GH1"
        }
    });

    Ok(A1Outcome {
        recurrent_codes,
        component_index,
        report,
        passed,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct CoverageCounter {
    recurrent: u64,
    total: u64,
}

impl CoverageCounter {
    fn observe(&mut self, recurrent: bool) -> Result<(), String> {
        checked_increment(&mut self.total, 1, "coverage total")?;
        if recurrent {
            checked_increment(&mut self.recurrent, 1, "coverage recurrent")?;
        }
        Ok(())
    }

    fn fraction(self) -> Result<f64, String> {
        if self.total == 0 {
            return Err("coverage denominator is zero".to_string());
        }
        let value = self.recurrent as f64 / self.total as f64;
        require_finite("coverage fraction", value)?;
        Ok(value)
    }
}

fn coverage_json(counter: CoverageCounter, fraction: f64) -> Value {
    json!({
        "recurrent_candidates": counter.recurrent,
        "total_candidates": counter.total,
        "fraction": fraction
    })
}

fn integer_histogram_json<K>(histogram: &BTreeMap<K, u64>) -> Value
where
    K: ToString + Ord,
{
    Value::Object(
        histogram
            .iter()
            .map(|(key, &count)| (key.to_string(), json!(count)))
            .collect(),
    )
}

fn checked_increment(slot: &mut u64, amount: u64, name: &str) -> Result<(), String> {
    *slot = slot
        .checked_add(amount)
        .ok_or_else(|| format!("{name} overflow"))?;
    Ok(())
}

fn fraction_gate(name: &str, observed: f64, threshold: f64) -> Value {
    json!({
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "pass": observed >= threshold
    })
}

fn gates_pass(gates: &[Value]) -> bool {
    gates
        .iter()
        .all(|gate| gate.get("pass").and_then(Value::as_bool) == Some(true))
}

#[derive(Clone, Debug)]
struct PreparedRow {
    row_index: usize,
    component: usize,
    root_color: RootColor,
    ordinal: u8,
    q: [f64; K6],
    utilities: [f64; K6],
    residuals: [f64; K6],
    code_ids: [u32; K6],
    recurrent: [bool; K6],
    code_is_abstraction: [bool; K6],
    target_tie_cardinality: u8,
    stored_teacher_top_in_target_set: bool,
    entropy: f64,
    base: BasicRowMetrics,
}

#[derive(Clone, Copy, Debug)]
struct BasicRowMetrics {
    ce: f64,
    kl: f64,
    sse: f64,
    credit: f64,
}

fn prepare_rows(
    slates: &[StatsSlate],
    recurrent_codes: &BTreeSet<u32>,
    component_index: &BTreeMap<String, usize>,
) -> Result<Vec<PreparedRow>, String> {
    let mut prepared = Vec::with_capacity(slates.len());
    for (row_index, slate) in slates.iter().enumerate() {
        validate_q(&slate.row_uid, &slate.q_teacher)?;
        for (candidate, &utility) in slate.product_root_utility.iter().enumerate() {
            require_finite(
                &format!("{} product_root_utility[{candidate}]", slate.row_uid),
                utility,
            )?;
        }

        let mut mean_log_q = Neumaier::default();
        let mut mean_utility = Neumaier::default();
        for candidate in 0..K6 {
            mean_log_q.add(slate.q_teacher[candidate].ln());
            mean_utility.add(slate.product_root_utility[candidate]);
        }
        let mean_log_q = mean_log_q.total() / K6 as f64;
        let mean_utility = mean_utility.total() / K6 as f64;
        require_finite("mean log q", mean_log_q)?;
        require_finite("mean product utility", mean_utility)?;

        let mut residuals = [0.0f64; K6];
        let mut sse = Neumaier::default();
        for candidate in 0..K6 {
            let target = slate.q_teacher[candidate].ln() - mean_log_q;
            let base = slate.product_root_utility[candidate] - mean_utility;
            residuals[candidate] = target - base;
            require_finite(
                &format!("{} residual[{candidate}]", slate.row_uid),
                residuals[candidate],
            )?;
            sse.add(residuals[candidate] * residuals[candidate]);
        }
        let entropy = entropy(&slate.q_teacher)?;
        let (ce, kl) = ce_and_kl(&slate.q_teacher, &slate.product_root_utility)?;
        let credit = q_top_credit(&slate.q_teacher, &slate.product_root_utility)?;
        let teacher_top_indices = slate
            .teacher_top
            .iter()
            .enumerate()
            .filter_map(|(index, &selected)| selected.then_some(index))
            .collect::<Vec<_>>();
        if teacher_top_indices.len() != 1 {
            return Err(format!(
                "{} has {} stored teacher_top roles, expected exactly one",
                slate.row_uid,
                teacher_top_indices.len()
            ));
        }
        let q_max = slate
            .q_teacher
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let target_tie_cardinality = slate
            .q_teacher
            .iter()
            .filter(|&&value| value == q_max)
            .count();
        if target_tie_cardinality == 0 || target_tie_cardinality > K6 {
            return Err(format!(
                "{} has invalid exact q-max tie cardinality {target_tie_cardinality}",
                slate.row_uid
            ));
        }
        let stored_teacher_top_in_target_set = slate.q_teacher[teacher_top_indices[0]] == q_max;
        let sse = sse.total();
        require_nonnegative_finite("base row SSE", sse)?;

        prepared.push(PreparedRow {
            row_index,
            component: *component_index
                .get(&slate.component_uid)
                .ok_or_else(|| format!("missing component index for {}", slate.component_uid))?,
            root_color: slate.root_color,
            ordinal: slate.ordinal,
            q: slate.q_teacher,
            utilities: slate.product_root_utility,
            residuals,
            code_ids: slate.code_ids,
            recurrent: slate.code_ids.map(|code| recurrent_codes.contains(&code)),
            code_is_abstraction: slate.code_is_abstraction,
            target_tie_cardinality: target_tie_cardinality as u8,
            stored_teacher_top_in_target_set,
            entropy,
            base: BasicRowMetrics {
                ce,
                kl,
                sse,
                credit,
            },
        });
    }
    Ok(prepared)
}

fn validate_q(row_uid: &str, q: &[f64; K6]) -> Result<(), String> {
    let mut sum = Neumaier::default();
    for (candidate, &value) in q.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "{row_uid} q_teacher[{candidate}] must be finite and positive, got {value}"
            ));
        }
        sum.add(value);
    }
    let sum = sum.total();
    if !sum.is_finite() || (sum - 1.0).abs() > 1.0e-12 {
        return Err(format!("{row_uid} q_teacher sum is {sum}, expected one"));
    }
    Ok(())
}

fn entropy(q: &[f64; K6]) -> Result<f64, String> {
    let mut entropy = Neumaier::default();
    for &value in q {
        entropy.add(-value * value.ln());
    }
    let entropy = entropy.total();
    require_nonnegative_finite("target entropy", entropy)?;
    Ok(entropy)
}

fn ce_and_kl(q: &[f64; K6], logits: &[f64; K6]) -> Result<(f64, f64), String> {
    let log_probs = log_softmax(logits)?;
    let mut ce = Neumaier::default();
    let mut kl = Neumaier::default();
    for candidate in 0..K6 {
        ce.add(-q[candidate] * log_probs[candidate]);
        kl.add(q[candidate] * (q[candidate].ln() - log_probs[candidate]));
    }
    let ce = ce.total();
    let kl = kl.total();
    require_finite("row cross-entropy", ce)?;
    require_finite("row KL", kl)?;
    if ce < 0.0 {
        return Err(format!("row cross-entropy is negative: {ce}"));
    }
    if kl < KL_NEGATIVE_INVALID {
        return Err(format!(
            "row KL {kl} is below registered invalid boundary {KL_NEGATIVE_INVALID}"
        ));
    }
    Ok((ce, kl))
}

fn log_softmax(logits: &[f64; K6]) -> Result<[f64; K6], String> {
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("log-softmax received a non-finite logit".to_string());
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exponential_sum = Neumaier::default();
    for &logit in logits {
        exponential_sum.add((logit - max).exp());
    }
    let exponential_sum = exponential_sum.total();
    if !exponential_sum.is_finite() || exponential_sum <= 0.0 {
        return Err(format!(
            "invalid log-softmax exponential denominator {exponential_sum}"
        ));
    }
    let log_denominator = max + exponential_sum.ln();
    require_finite("log-softmax log denominator", log_denominator)?;
    let result = logits.map(|logit| logit - log_denominator);
    if result.iter().any(|value| !value.is_finite()) {
        return Err("non-finite log-softmax result".to_string());
    }
    Ok(result)
}

fn q_top_credit(q: &[f64; K6], logits: &[f64; K6]) -> Result<f64, String> {
    if q.iter().any(|value| !value.is_finite()) || logits.iter().any(|value| !value.is_finite()) {
        return Err("q-top credit received non-finite input".to_string());
    }
    let q_max = q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let predicted_max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut predicted = 0u64;
    let mut accepted = 0u64;
    for candidate in 0..K6 {
        if logits[candidate] == predicted_max {
            predicted += 1;
            if q[candidate] == q_max {
                accepted += 1;
            }
        }
    }
    if predicted == 0 {
        return Err("q-top prediction set is empty".to_string());
    }
    let credit = accepted as f64 / predicted as f64;
    require_finite("q-top credit", credit)?;
    Ok(credit)
}

#[derive(Clone, Copy, Debug, Default)]
struct SumCount {
    sum: Neumaier,
    count: u64,
}

impl SumCount {
    fn add(&mut self, value: f64) -> Result<(), String> {
        require_finite("residual observation", value)?;
        self.sum.add(value);
        checked_increment(&mut self.count, 1, "residual observation count")
    }

    fn mean(self, name: &str) -> Result<f64, String> {
        if self.count == 0 {
            return Err(format!("{name} has zero observations"));
        }
        let value = self.sum.total() / self.count as f64;
        require_finite(name, value)?;
        Ok(value)
    }
}

fn fit_observation_weighted_recurrent(
    rows: &[PreparedRow],
    recurrent_codes: &BTreeSet<u32>,
) -> Result<BTreeMap<u32, f64>, String> {
    let mut sums = BTreeMap::<u32, SumCount>::new();
    for row in rows {
        for candidate in 0..K6 {
            let code = row.code_ids[candidate];
            if recurrent_codes.contains(&code) {
                sums.entry(code)
                    .or_default()
                    .add(row.residuals[candidate])?;
            }
        }
    }
    let mut corrections = BTreeMap::new();
    for (&code, &sum) in &sums {
        corrections.insert(code, sum.mean(&format!("full-fit code {code} mean"))?);
    }
    if corrections.len() != recurrent_codes.len() {
        return Err(format!(
            "full-fit recurrent code count {}, expected {}",
            corrections.len(),
            recurrent_codes.len()
        ));
    }
    Ok(corrections)
}

#[derive(Clone, Copy, Debug, Default)]
struct DispersionAccumulator {
    observations: u64,
    residual_sum: Neumaier,
    within_code_sse: Neumaier,
}

impl DispersionAccumulator {
    fn observe(&mut self, residual: f64, code_mean: f64) -> Result<(), String> {
        require_finite("dispersion residual", residual)?;
        require_finite("dispersion code mean", code_mean)?;
        checked_increment(&mut self.observations, 1, "dispersion observation count")?;
        self.residual_sum.add(residual);
        let centered = residual - code_mean;
        self.within_code_sse.add(centered * centered);
        Ok(())
    }

    fn report(self, distinct_codes: usize, name: &str) -> Result<Value, String> {
        if self.observations == 0 {
            return Ok(json!({
                "distinct_codes": distinct_codes,
                "observations": 0,
                "raw_residual_mean": Value::Null,
                "within_code_sse": 0.0,
                "within_code_population_variance": Value::Null,
                "within_code_rms": Value::Null
            }));
        }
        let denominator = self.observations as f64;
        let residual_mean = self.residual_sum.total() / denominator;
        let sse = self.within_code_sse.total();
        let variance = sse / denominator;
        let rms = variance.sqrt();
        require_finite(&format!("{name} raw residual mean"), residual_mean)?;
        require_nonnegative_finite(&format!("{name} within-code SSE"), sse)?;
        require_nonnegative_finite(&format!("{name} within-code variance"), variance)?;
        require_nonnegative_finite(&format!("{name} within-code RMS"), rms)?;
        Ok(json!({
            "distinct_codes": distinct_codes,
            "observations": self.observations,
            "raw_residual_mean": residual_mean,
            "within_code_sse": sse,
            "within_code_population_variance": variance,
            "within_code_rms": rms
        }))
    }
}

fn recurrent_residual_dispersion(
    rows: &[PreparedRow],
    recurrent_codes: &BTreeSet<u32>,
    corrections: &BTreeMap<u32, f64>,
) -> Result<Value, String> {
    let mut class_by_code = BTreeMap::<u32, bool>::new();
    let mut classes = [DispersionAccumulator::default(); 2];
    let mut class_codes = [BTreeSet::<u32>::new(), BTreeSet::<u32>::new()];

    for row in rows {
        for candidate in 0..K6 {
            let code = row.code_ids[candidate];
            if !recurrent_codes.contains(&code) {
                continue;
            }
            let abstraction = row.code_is_abstraction[candidate];
            if let Some(&prior) = class_by_code.get(&code) {
                if prior != abstraction {
                    return Err(format!(
                        "graph code {code} has inconsistent abstraction classification"
                    ));
                }
            } else {
                class_by_code.insert(code, abstraction);
            }
            let class = usize::from(abstraction);
            class_codes[class].insert(code);
            let code_mean = *corrections
                .get(&code)
                .ok_or_else(|| format!("missing recurrent correction for code {code}"))?;
            classes[class].observe(row.residuals[candidate], code_mean)?;
        }
    }
    if class_by_code.len() != recurrent_codes.len() {
        return Err(format!(
            "dispersion classified {} recurrent codes, expected {}",
            class_by_code.len(),
            recurrent_codes.len()
        ));
    }
    Ok(json!({
        "scope": "A1-recurrent observations only",
        "dispersion": "population residual dispersion around each code's observation-weighted full-fit mean",
        "single_exact_transition_or_duplicate_code": classes[0].report(
            class_codes[0].len(),
            "single-exact/duplicate"
        )?,
        "abstraction_collision_code": classes[1].report(
            class_codes[1].len(),
            "abstraction collision"
        )?
    }))
}

#[derive(Clone, Copy, Debug)]
struct RowProjectionScore {
    component: usize,
    root_color: RootColor,
    ordinal: u8,
    entropy: f64,
    base: BasicRowMetrics,
    corrected: BasicRowMetrics,
    recurrent_teacher_mass: f64,
    target_tie_cardinality: u8,
    stored_teacher_top_in_target_set: bool,
}

#[derive(Debug)]
struct ProjectionReport {
    combined: StratumMetrics,
    colors: [StratumMetrics; 2],
    ordinals: BTreeMap<u8, StratumMetrics>,
    row_scores: Vec<RowProjectionScore>,
}

fn evaluate_projection<F>(
    rows: &[PreparedRow],
    mut correction: F,
) -> Result<ProjectionReport, String>
where
    F: FnMut(&PreparedRow, usize) -> f64,
{
    let mut combined = StratumAccumulator::default();
    let mut colors = [StratumAccumulator::default(); 2];
    let mut ordinals = BTreeMap::<u8, StratumAccumulator>::new();
    let mut row_scores = Vec::with_capacity(rows.len());

    for row in rows {
        let mut corrected_logits = row.utilities;
        let mut corrected_sse = Neumaier::default();
        let mut recurrent_teacher_mass = Neumaier::default();
        for candidate in 0..K6 {
            let value = correction(row, candidate);
            require_finite("projection correction", value)?;
            corrected_logits[candidate] += value;
            require_finite("corrected product utility", corrected_logits[candidate])?;
            let residual = row.residuals[candidate] - value;
            corrected_sse.add(residual * residual);
            if row.recurrent[candidate] {
                recurrent_teacher_mass.add(row.q[candidate]);
            }
        }
        let corrected_sse = corrected_sse.total();
        require_nonnegative_finite("corrected row SSE", corrected_sse)?;
        let (corrected_ce, corrected_kl) = ce_and_kl(&row.q, &corrected_logits)?;
        let corrected_credit = q_top_credit(&row.q, &corrected_logits)?;
        let recurrent_teacher_mass = recurrent_teacher_mass.total();
        if !recurrent_teacher_mass.is_finite()
            || recurrent_teacher_mass < 0.0
            || recurrent_teacher_mass > 1.0 + 1.0e-12
        {
            return Err(format!(
                "invalid recurrent teacher mass {recurrent_teacher_mass}"
            ));
        }

        let score = RowProjectionScore {
            component: row.component,
            root_color: row.root_color,
            ordinal: row.ordinal,
            entropy: row.entropy,
            base: row.base,
            corrected: BasicRowMetrics {
                ce: corrected_ce,
                kl: corrected_kl,
                sse: corrected_sse,
                credit: corrected_credit,
            },
            recurrent_teacher_mass,
            target_tie_cardinality: row.target_tie_cardinality,
            stored_teacher_top_in_target_set: row.stored_teacher_top_in_target_set,
        };
        combined.observe(score)?;
        colors[row.root_color.index()].observe(score)?;
        ordinals.entry(row.ordinal).or_default().observe(score)?;
        row_scores.push(score);
    }

    Ok(ProjectionReport {
        combined: combined.finish("combined")?,
        colors: [colors[0].finish("Black")?, colors[1].finish("White")?],
        ordinals: ordinals
            .into_iter()
            .map(|(ordinal, accumulator)| {
                Ok((ordinal, accumulator.finish(&format!("ordinal {ordinal}"))?))
            })
            .collect::<Result<_, String>>()?,
        row_scores,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct StratumAccumulator {
    rows: u64,
    entropy: Neumaier,
    base_ce: Neumaier,
    corrected_ce: Neumaier,
    base_kl: Neumaier,
    corrected_kl: Neumaier,
    base_sse: Neumaier,
    corrected_sse: Neumaier,
    base_credit: Neumaier,
    corrected_credit: Neumaier,
    recurrent_teacher_mass: Neumaier,
}

impl StratumAccumulator {
    fn observe(&mut self, row: RowProjectionScore) -> Result<(), String> {
        checked_increment(&mut self.rows, 1, "stratum row count")?;
        self.entropy.add(row.entropy);
        self.base_ce.add(row.base.ce);
        self.corrected_ce.add(row.corrected.ce);
        self.base_kl.add(row.base.kl);
        self.corrected_kl.add(row.corrected.kl);
        self.base_sse.add(row.base.sse);
        self.corrected_sse.add(row.corrected.sse);
        self.base_credit.add(row.base.credit);
        self.corrected_credit.add(row.corrected.credit);
        self.recurrent_teacher_mass.add(row.recurrent_teacher_mass);
        Ok(())
    }

    fn finish(self, name: &str) -> Result<StratumMetrics, String> {
        if self.rows == 0 {
            return Err(format!("{name} has zero rows"));
        }
        let metrics = StratumMetrics {
            rows: self.rows,
            entropy: self.entropy.total(),
            base_ce: self.base_ce.total(),
            corrected_ce: self.corrected_ce.total(),
            base_kl: self.base_kl.total(),
            corrected_kl: self.corrected_kl.total(),
            base_sse: self.base_sse.total(),
            corrected_sse: self.corrected_sse.total(),
            base_credit: self.base_credit.total() / self.rows as f64,
            corrected_credit: self.corrected_credit.total() / self.rows as f64,
            recurrent_teacher_mass: self.recurrent_teacher_mass.total() / self.rows as f64,
        };
        metrics.validate(name)?;
        Ok(metrics)
    }
}

#[derive(Clone, Copy, Debug)]
struct StratumMetrics {
    rows: u64,
    entropy: f64,
    base_ce: f64,
    corrected_ce: f64,
    base_kl: f64,
    corrected_kl: f64,
    base_sse: f64,
    corrected_sse: f64,
    base_credit: f64,
    corrected_credit: f64,
    recurrent_teacher_mass: f64,
}

impl StratumMetrics {
    fn validate(self, name: &str) -> Result<(), String> {
        for (metric, value) in [
            ("entropy", self.entropy),
            ("base CE", self.base_ce),
            ("corrected CE", self.corrected_ce),
            ("base KL", self.base_kl),
            ("corrected KL", self.corrected_kl),
            ("base SSE", self.base_sse),
            ("corrected SSE", self.corrected_sse),
            ("base credit", self.base_credit),
            ("corrected credit", self.corrected_credit),
            ("recurrent teacher mass", self.recurrent_teacher_mass),
        ] {
            require_finite(&format!("{name} {metric}"), value)?;
        }
        if self.base_ce < 0.0
            || self.corrected_ce < 0.0
            || self.base_sse < 0.0
            || self.corrected_sse < 0.0
            || !(0.0..=1.0).contains(&self.base_credit)
            || !(0.0..=1.0).contains(&self.corrected_credit)
            || !(0.0..=1.0 + 1.0e-12).contains(&self.recurrent_teacher_mass)
        {
            return Err(format!("{name} has an out-of-domain aggregate metric"));
        }
        Ok(())
    }

    fn relative_kl(self, name: &str) -> Result<f64, String> {
        relative_gain(&format!("{name} base KL"), self.base_kl, self.corrected_kl)
    }

    fn relative_sse(self, name: &str) -> Result<f64, String> {
        relative_gain(
            &format!("{name} base SSE"),
            self.base_sse,
            self.corrected_sse,
        )
    }

    fn q_top_delta_pp(self) -> Result<f64, String> {
        let value = 100.0 * (self.corrected_credit - self.base_credit);
        require_finite("q-top credit delta", value)?;
        Ok(value)
    }
}

fn relative_gain(name: &str, base: f64, corrected: f64) -> Result<f64, String> {
    if !base.is_finite() || base <= 0.0 {
        return Err(format!(
            "{name} denominator must be finite and positive, got {base}"
        ));
    }
    require_finite(&format!("{name} corrected value"), corrected)?;
    let value = (base - corrected) / base;
    require_finite(&format!("{name} relative gain"), value)?;
    Ok(value)
}

struct StageDecision {
    report: Value,
    passed: bool,
}

fn a2_decision(
    full: &ProjectionReport,
    residual_dispersion: Value,
) -> Result<StageDecision, String> {
    let combined_kl = full.combined.relative_kl("A2 combined")?;
    let combined_sse = full.combined.relative_sse("A2 combined")?;
    let combined_qtop = full.combined.q_top_delta_pp()?;

    // All displayed stratum ratios are true stratum-specific denominators.
    let colors = [
        stratum_json(full.colors[0], "A2 Black")?,
        stratum_json(full.colors[1], "A2 White")?,
    ];
    let ordinals = full
        .ordinals
        .iter()
        .map(|(&ordinal, &metrics)| {
            Ok((
                ordinal.to_string(),
                stratum_json(metrics, &format!("A2 ordinal {ordinal}"))?,
            ))
        })
        .collect::<Result<serde_json::Map<_, _>, String>>()?;
    let teacher_top = teacher_top_diagnostics(&full.row_scores)?;

    let gates = vec![
        numeric_gate(
            "aggregate_R_KL_full_ge_0_03",
            combined_kl,
            A2_RELATIVE_MIN,
            combined_kl >= A2_RELATIVE_MIN,
        ),
        numeric_gate(
            "aggregate_R_SSE_full_ge_0_03",
            combined_sse,
            A2_RELATIVE_MIN,
            combined_sse >= A2_RELATIVE_MIN,
        ),
        numeric_gate(
            "aggregate_DeltaQTop_full_pp_ge_1",
            combined_qtop,
            A2_QTOP_PP_MIN,
            combined_qtop >= A2_QTOP_PP_MIN,
        ),
        numeric_gate(
            "combined_recurrent_teacher_mass_ge_25pct",
            full.combined.recurrent_teacher_mass,
            0.25,
            full.combined.recurrent_teacher_mass >= 0.25,
        ),
        numeric_gate(
            "black_recurrent_teacher_mass_ge_15pct",
            full.colors[0].recurrent_teacher_mass,
            0.15,
            full.colors[0].recurrent_teacher_mass >= 0.15,
        ),
        numeric_gate(
            "white_recurrent_teacher_mass_ge_15pct",
            full.colors[1].recurrent_teacher_mass,
            0.15,
            full.colors[1].recurrent_teacher_mass >= 0.15,
        ),
    ];
    let passed = gates_pass(&gates);
    Ok(StageDecision {
        report: json!({
            "stage": "A2_RECURRENT_FULL_FIT",
            "status": if passed {
                "A2_PASS_A3_REQUIRED"
            } else {
                NO_SIGNAL_LABEL
            },
            "estimator": {
                "codes": "A1 recurrent exact graph codes only",
                "fit": "observation-weighted mean centered-log residual",
                "unsupported_correction": 0.0,
                "all_code_projection_gating": false
            },
            "combined": stratum_json(full.combined, "A2 combined")?,
            "black": colors[0],
            "white": colors[1],
            "ordinals": ordinals,
            "probability_target_truth": teacher_top,
            "within_code_residual_dispersion": residual_dispersion,
            "primary": {
                "R_KL_full": combined_kl,
                "R_SSE_full": combined_sse,
                "DeltaQTop_full_pp": combined_qtop
            },
            "gates": gates,
            "all_a2_gates_pass": passed,
            "next_stage": if passed {
                "A3_COMPONENT_LOO"
            } else {
                "STOP_CB_GH1"
            }
        }),
        passed,
    })
}

fn teacher_top_diagnostics(rows: &[RowProjectionScore]) -> Result<Value, String> {
    if rows.is_empty() {
        return Err("teacher-top diagnostics received zero rows".to_string());
    }
    let mut membership = 0u64;
    let mut nonmembership = 0u64;
    let mut tie_cardinalities = BTreeMap::<u8, u64>::new();
    for row in rows {
        if row.target_tie_cardinality == 0 || row.target_tie_cardinality as usize > K6 {
            return Err(format!(
                "invalid target tie cardinality {}",
                row.target_tie_cardinality
            ));
        }
        checked_increment(
            tie_cardinalities
                .entry(row.target_tie_cardinality)
                .or_default(),
            1,
            "target tie cardinality count",
        )?;
        if row.stored_teacher_top_in_target_set {
            checked_increment(&mut membership, 1, "teacher_top membership count")?;
        } else {
            checked_increment(&mut nonmembership, 1, "teacher_top nonmembership count")?;
        }
    }
    if membership + nonmembership != rows.len() as u64 {
        return Err("teacher_top membership accounting mismatch".to_string());
    }
    Ok(json!({
        "acceptable_target_set": "all candidates exactly equal to max(q_teacher)",
        "stored_teacher_top_roles_per_slate": 1,
        "stored_teacher_top_membership_rows": membership,
        "stored_teacher_top_nonmembership_rows": nonmembership,
        "target_tie_cardinality_rows": integer_histogram_json(&tie_cardinalities),
        "stored_role_is_gating_truth": false
    }))
}

fn stratum_json(metrics: StratumMetrics, name: &str) -> Result<Value, String> {
    Ok(json!({
        "rows": metrics.rows,
        "target_entropy_sum": metrics.entropy,
        "base_cross_entropy_sum": metrics.base_ce,
        "corrected_cross_entropy_sum": metrics.corrected_ce,
        "base_excess_loss_kl_sum": metrics.base_kl,
        "corrected_excess_loss_kl_sum": metrics.corrected_kl,
        "R_KL": metrics.relative_kl(name)?,
        "base_residual_sse": metrics.base_sse,
        "corrected_residual_sse": metrics.corrected_sse,
        "R_SSE": metrics.relative_sse(name)?,
        "base_q_top_credit": metrics.base_credit,
        "corrected_q_top_credit": metrics.corrected_credit,
        "DeltaQTop_pp": metrics.q_top_delta_pp()?,
        "recurrent_teacher_mass": metrics.recurrent_teacher_mass
    }))
}

fn numeric_gate(name: &str, observed: f64, threshold: f64, pass: bool) -> Value {
    json!({
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "pass": pass
    })
}

fn build_loo_corrections(rows: &[PreparedRow]) -> Result<Vec<[f64; K6]>, String> {
    let mut by_code_component = BTreeMap::<u32, BTreeMap<usize, SumCount>>::new();
    for row in rows {
        for candidate in 0..K6 {
            by_code_component
                .entry(row.code_ids[candidate])
                .or_default()
                .entry(row.component)
                .or_default()
                .add(row.residuals[candidate])?;
        }
    }

    let mut component_means = BTreeMap::<u32, BTreeMap<usize, f64>>::new();
    for (&code, components) in &by_code_component {
        let means = components
            .iter()
            .map(|(&component, &sum)| {
                Ok((
                    component,
                    sum.mean(&format!(
                        "LOO code {code} component {component} residual mean"
                    ))?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        component_means.insert(code, means);
    }

    let mut corrections = vec![[0.0f64; K6]; rows.len()];
    for row in rows {
        for candidate in 0..K6 {
            let code = row.code_ids[candidate];
            let means = component_means
                .get(&code)
                .ok_or_else(|| format!("missing LOO means for code {code}"))?;
            corrections[row.row_index][candidate] =
                leave_one_component_out_mean(means, row.component)?;
        }
    }
    Ok(corrections)
}

fn leave_one_component_out_mean(
    component_means: &BTreeMap<usize, f64>,
    held_out: usize,
) -> Result<f64, String> {
    let mut sum = Neumaier::default();
    let mut donors = 0u64;
    for (&component, &mean) in component_means {
        require_finite("LOO component mean", mean)?;
        if component != held_out {
            sum.add(mean);
            checked_increment(&mut donors, 1, "LOO donor count")?;
        }
    }
    if donors < 2 {
        return Ok(0.0);
    }
    let correction = sum.total() / donors as f64;
    require_finite("LOO correction", correction)?;
    Ok(correction)
}

#[derive(Clone, Copy, Debug, Default)]
struct BootstrapCell {
    rows: u64,
    base_kl: Neumaier,
    corrected_kl: Neumaier,
    base_sse: Neumaier,
    corrected_sse: Neumaier,
    base_credit: Neumaier,
    corrected_credit: Neumaier,
}

impl BootstrapCell {
    fn observe(&mut self, score: RowProjectionScore) -> Result<(), String> {
        checked_increment(&mut self.rows, 1, "bootstrap component row count")?;
        self.base_kl.add(score.base.kl);
        self.corrected_kl.add(score.corrected.kl);
        self.base_sse.add(score.base.sse);
        self.corrected_sse.add(score.corrected.sse);
        self.base_credit.add(score.base.credit);
        self.corrected_credit.add(score.corrected.credit);
        Ok(())
    }

    fn finish(self) -> BootstrapContribution {
        BootstrapContribution {
            rows: self.rows,
            base_kl: self.base_kl.total(),
            corrected_kl: self.corrected_kl.total(),
            base_sse: self.base_sse.total(),
            corrected_sse: self.corrected_sse.total(),
            base_credit: self.base_credit.total(),
            corrected_credit: self.corrected_credit.total(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BootstrapContribution {
    rows: u64,
    base_kl: f64,
    corrected_kl: f64,
    base_sse: f64,
    corrected_sse: f64,
    base_credit: f64,
    corrected_credit: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ComponentContribution {
    combined: BootstrapContribution,
    colors: [BootstrapContribution; 2],
}

#[derive(Debug)]
struct BootstrapSummary {
    replicates: usize,
    draws_per_replicate: usize,
    seed: u64,
    combined_r_kl: Quantiles,
    combined_r_sse: Quantiles,
    combined_qtop_delta_pp: Quantiles,
    color_r_kl: [Quantiles; 2],
    color_qtop_delta_pp: [Quantiles; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Quantiles {
    p05: f64,
    p95: f64,
}

fn bootstrap_components(
    rows: &[PreparedRow],
    scores: &[RowProjectionScore],
    component_count: usize,
    replicates: usize,
    seed: u64,
) -> Result<BootstrapSummary, String> {
    if component_count == 0 || replicates == 0 {
        return Err("bootstrap component count and replicates must be positive".to_string());
    }
    if rows.len() != scores.len() {
        return Err(format!(
            "bootstrap rows/scores length mismatch: {} != {}",
            rows.len(),
            scores.len()
        ));
    }

    let mut combined_cells = vec![BootstrapCell::default(); component_count];
    let mut color_cells = vec![[BootstrapCell::default(); 2]; component_count];
    for (row, &score) in rows.iter().zip(scores) {
        if row.component >= component_count || score.component != row.component {
            return Err(format!(
                "bootstrap invalid component index {}/{}",
                row.component, component_count
            ));
        }
        if score.root_color != row.root_color || score.ordinal != row.ordinal {
            return Err("bootstrap row score stratum mismatch".to_string());
        }
        combined_cells[row.component].observe(score)?;
        color_cells[row.component][row.root_color.index()].observe(score)?;
    }
    if combined_cells.iter().any(|cell| cell.rows == 0) {
        return Err("bootstrap encountered an empty component".to_string());
    }
    let contributions = (0..component_count)
        .map(|component| ComponentContribution {
            combined: combined_cells[component].finish(),
            colors: [
                color_cells[component][0].finish(),
                color_cells[component][1].finish(),
            ],
        })
        .collect::<Vec<_>>();

    let mut rng = SplitMix64::new(seed);
    let mut combined_r_kl = Vec::with_capacity(replicates);
    let mut combined_r_sse = Vec::with_capacity(replicates);
    let mut combined_qtop = Vec::with_capacity(replicates);
    let mut color_r_kl: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::with_capacity(replicates));
    let mut color_qtop: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::with_capacity(replicates));

    for replicate in 0..replicates {
        let mut combined = ResampleAccumulator::default();
        let mut colors = [ResampleAccumulator::default(); 2];
        for _ in 0..component_count {
            let drawn = (rng.next_u64() % component_count as u64) as usize;
            combined.add(
                contributions[drawn].combined,
                &format!("bootstrap replicate {replicate} combined"),
            )?;
            for color in 0..2 {
                colors[color].add(
                    contributions[drawn].colors[color],
                    &format!("bootstrap replicate {replicate} color {color}"),
                )?;
            }
        }
        let combined_metrics =
            combined.finish(&format!("bootstrap replicate {replicate} combined"))?;
        combined_r_kl.push(combined_metrics.r_kl);
        combined_r_sse.push(combined_metrics.r_sse);
        combined_qtop.push(combined_metrics.qtop_delta_pp);
        for color in 0..2 {
            let metrics =
                colors[color].finish(&format!("bootstrap replicate {replicate} color {color}"))?;
            color_r_kl[color].push(metrics.r_kl);
            color_qtop[color].push(metrics.qtop_delta_pp);
        }
    }

    Ok(BootstrapSummary {
        replicates,
        draws_per_replicate: component_count,
        seed,
        combined_r_kl: quantiles(&mut combined_r_kl)?,
        combined_r_sse: quantiles(&mut combined_r_sse)?,
        combined_qtop_delta_pp: quantiles(&mut combined_qtop)?,
        color_r_kl: [
            quantiles(&mut color_r_kl[0])?,
            quantiles(&mut color_r_kl[1])?,
        ],
        color_qtop_delta_pp: [
            quantiles(&mut color_qtop[0])?,
            quantiles(&mut color_qtop[1])?,
        ],
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct ResampleAccumulator {
    rows: u64,
    base_kl: Neumaier,
    corrected_kl: Neumaier,
    base_sse: Neumaier,
    corrected_sse: Neumaier,
    base_credit: Neumaier,
    corrected_credit: Neumaier,
}

impl ResampleAccumulator {
    fn add(&mut self, value: BootstrapContribution, name: &str) -> Result<(), String> {
        self.rows = self
            .rows
            .checked_add(value.rows)
            .ok_or_else(|| format!("{name} row count overflow"))?;
        for (metric, number) in [
            ("base KL", value.base_kl),
            ("corrected KL", value.corrected_kl),
            ("base SSE", value.base_sse),
            ("corrected SSE", value.corrected_sse),
            ("base credit", value.base_credit),
            ("corrected credit", value.corrected_credit),
        ] {
            require_finite(&format!("{name} {metric} contribution"), number)?;
        }
        self.base_kl.add(value.base_kl);
        self.corrected_kl.add(value.corrected_kl);
        self.base_sse.add(value.base_sse);
        self.corrected_sse.add(value.corrected_sse);
        self.base_credit.add(value.base_credit);
        self.corrected_credit.add(value.corrected_credit);
        Ok(())
    }

    fn finish(self, name: &str) -> Result<ResampleMetrics, String> {
        if self.rows == 0 {
            return Err(format!("{name} has zero resampled rows"));
        }
        let base_kl = self.base_kl.total();
        let corrected_kl = self.corrected_kl.total();
        let base_sse = self.base_sse.total();
        let corrected_sse = self.corrected_sse.total();
        let base_credit = self.base_credit.total() / self.rows as f64;
        let corrected_credit = self.corrected_credit.total() / self.rows as f64;
        let r_kl = relative_gain(&format!("{name} KL"), base_kl, corrected_kl)?;
        let r_sse = relative_gain(&format!("{name} SSE"), base_sse, corrected_sse)?;
        let qtop_delta_pp = 100.0 * (corrected_credit - base_credit);
        require_finite(&format!("{name} q-top delta"), qtop_delta_pp)?;
        Ok(ResampleMetrics {
            r_kl,
            r_sse,
            qtop_delta_pp,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct ResampleMetrics {
    r_kl: f64,
    r_sse: f64,
    qtop_delta_pp: f64,
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

fn quantiles(values: &mut [f64]) -> Result<Quantiles, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("bootstrap quantiles require non-empty finite values".to_string());
    }
    values.sort_by(f64::total_cmp);
    Ok(Quantiles {
        p05: nearest_rank(values, 0.05)?,
        p95: nearest_rank(values, 0.95)?,
    })
}

fn nearest_rank(sorted: &[f64], q: f64) -> Result<f64, String> {
    if sorted.is_empty()
        || sorted.iter().any(|value| !value.is_finite())
        || !q.is_finite()
        || q <= 0.0
        || q > 1.0
    {
        return Err("nearest-rank received an invalid input".to_string());
    }
    if sorted.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("nearest-rank values are not sorted".to_string());
    }
    let index = ((q * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Ok(sorted[index])
}

fn a3_decision(
    loo: &ProjectionReport,
    bootstrap: &BootstrapSummary,
) -> Result<StageDecision, String> {
    let combined_kl = loo.combined.relative_kl("A3 combined")?;
    let combined_sse = loo.combined.relative_sse("A3 combined")?;
    let combined_qtop = loo.combined.q_top_delta_pp()?;
    let color_kl = [
        loo.colors[0].relative_kl("A3 Black")?,
        loo.colors[1].relative_kl("A3 White")?,
    ];
    let color_qtop = [
        loo.colors[0].q_top_delta_pp()?,
        loo.colors[1].q_top_delta_pp()?,
    ];
    let ordinal_points = loo
        .ordinals
        .iter()
        .map(|(&ordinal, &metrics)| {
            Ok((
                ordinal,
                (
                    metrics.relative_kl(&format!("A3 ordinal {ordinal}"))?,
                    metrics.q_top_delta_pp()?,
                ),
            ))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;

    let gates = vec![
        numeric_gate(
            "combined_R_KL_loo_ge_0_03",
            combined_kl,
            0.03,
            combined_kl >= 0.03,
        ),
        numeric_gate(
            "combined_R_KL_loo_bootstrap_p05_gt_0",
            bootstrap.combined_r_kl.p05,
            0.0,
            bootstrap.combined_r_kl.p05 > 0.0,
        ),
        numeric_gate(
            "combined_R_SSE_loo_ge_0",
            combined_sse,
            0.0,
            combined_sse >= 0.0,
        ),
        numeric_gate(
            "combined_DeltaQTop_loo_pp_ge_1",
            combined_qtop,
            1.0,
            combined_qtop >= 1.0,
        ),
        numeric_gate(
            "combined_DeltaQTop_bootstrap_p05_ge_0",
            bootstrap.combined_qtop_delta_pp.p05,
            0.0,
            bootstrap.combined_qtop_delta_pp.p05 >= 0.0,
        ),
        json!({
            "name": "black_and_white_R_KL_loo_points_ge_0",
            "threshold": 0.0,
            "observed": {
                "Black": color_kl[0],
                "White": color_kl[1]
            },
            "pass": color_kl.iter().all(|&value| value >= 0.0)
        }),
        json!({
            "name": "black_and_white_R_KL_bootstrap_p05_ge_0",
            "threshold": 0.0,
            "observed": {
                "Black": bootstrap.color_r_kl[0].p05,
                "White": bootstrap.color_r_kl[1].p05
            },
            "pass": bootstrap.color_r_kl.iter().all(|value| value.p05 >= 0.0)
        }),
        json!({
            "name": "black_and_white_DeltaQTop_points_ge_0",
            "threshold": 0.0,
            "observed": {
                "Black": color_qtop[0],
                "White": color_qtop[1]
            },
            "pass": color_qtop.iter().all(|&value| value >= 0.0)
        }),
        json!({
            "name": "every_ordinal_R_KL_and_DeltaQTop_point_ge_0",
            "threshold": 0.0,
            "observed": ordinal_points
                .iter()
                .map(|(ordinal, (r_kl, qtop))| {
                    (
                        ordinal.to_string(),
                        json!({
                            "R_KL": r_kl,
                            "DeltaQTop_pp": qtop
                        })
                    )
                })
                .collect::<serde_json::Map<String, Value>>(),
            "pass": ordinal_points
                .values()
                .all(|&(r_kl, qtop)| r_kl >= 0.0 && qtop >= 0.0)
        }),
    ];
    let passed = gates_pass(&gates);
    let ordinal_report = loo
        .ordinals
        .iter()
        .map(|(&ordinal, &metrics)| {
            Ok((
                ordinal.to_string(),
                stratum_json(metrics, &format!("A3 ordinal {ordinal}"))?,
            ))
        })
        .collect::<Result<serde_json::Map<_, _>, String>>()?;

    Ok(StageDecision {
        report: json!({
            "stage": "A3_COMPONENT_LOO",
            "status": if passed {
                OPEN_INCREMENTAL_LABEL
            } else {
                NO_SIGNAL_LABEL
            },
            "estimator": {
                "held_out_unit": "whole component_uid",
                "donor_weighting": "equal component mean",
                "minimum_donor_components": 2,
                "smoothing": false,
                "global_mean": false,
                "backoff": false
            },
            "combined": stratum_json(loo.combined, "A3 combined")?,
            "black": stratum_json(loo.colors[0], "A3 Black")?,
            "white": stratum_json(loo.colors[1], "A3 White")?,
            "ordinals": ordinal_report,
            "primary": {
                "R_KL_loo": combined_kl,
                "R_SSE_loo": combined_sse,
                "DeltaQTop_loo_pp": combined_qtop
            },
            "bootstrap": {
                "method": "whole-component cluster bootstrap",
                "components_per_replicate": bootstrap.draws_per_replicate,
                "replicates": bootstrap.replicates,
                "seed_hex": format!("{:016X}", bootstrap.seed),
                "rng": "SplitMix64; next_u64 % 388; continuous stream; no rejection",
                "quantiles": "nearest-rank, zero-based ceil(q*N)-1",
                "combined": {
                    "R_KL_loo": quantiles_json(bootstrap.combined_r_kl),
                    "R_SSE_loo": quantiles_json(bootstrap.combined_r_sse),
                    "DeltaQTop_loo_pp": quantiles_json(bootstrap.combined_qtop_delta_pp)
                },
                "black": {
                    "R_KL_loo": quantiles_json(bootstrap.color_r_kl[0]),
                    "DeltaQTop_loo_pp": quantiles_json(bootstrap.color_qtop_delta_pp[0])
                },
                "white": {
                    "R_KL_loo": quantiles_json(bootstrap.color_r_kl[1]),
                    "DeltaQTop_loo_pp": quantiles_json(bootstrap.color_qtop_delta_pp[1])
                }
            },
            "gates": gates,
            "all_a3_gates_pass": passed,
            "next_stage": if passed {
                "PREREGISTER_GH1_INCREMENTAL_CORRECTNESS_AND_COST"
            } else {
                "STOP_CB_GH1"
            }
        }),
        passed,
    })
}

fn quantiles_json(quantiles: Quantiles) -> Value {
    json!({
        "p05": quantiles.p05,
        "p95": quantiles.p95
    })
}

fn require_finite(name: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("{name} must be finite, got {value}"));
    }
    Ok(())
}

fn require_nonnegative_finite(name: &str, value: f64) -> Result<(), String> {
    require_finite(name, value)?;
    if value < 0.0 {
        return Err(format!("{name} must be nonnegative, got {value}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn structural_slate(
        row: &str,
        component: &str,
        color: RootColor,
        ordinal: u8,
        code_ids: [u32; K6],
    ) -> StatsSlate {
        StatsSlate {
            row_uid: row.to_string(),
            component_uid: component.to_string(),
            root_color: color,
            ordinal,
            q_teacher: [f64::NAN; K6],
            product_root_utility: [f64::NAN; K6],
            code_ids,
            teacher_top: [false; K6],
            code_is_abstraction: [false; K6],
        }
    }

    fn prepared_row(row_index: usize, component: usize, residual: f64, code: u32) -> PreparedRow {
        let q = [1.0 / K6 as f64; K6];
        let utilities = [0.0; K6];
        PreparedRow {
            row_index,
            component,
            root_color: if component % 2 == 0 {
                RootColor::Black
            } else {
                RootColor::White
            },
            ordinal: 1,
            q,
            utilities,
            residuals: [residual; K6],
            code_ids: [code; K6],
            recurrent: [true; K6],
            code_is_abstraction: [false; K6],
            target_tie_cardinality: K6 as u8,
            stored_teacher_top_in_target_set: true,
            entropy: entropy(&q).unwrap(),
            base: BasicRowMetrics {
                ce: (K6 as f64).ln(),
                kl: 0.1,
                sse: 6.0 * residual * residual,
                credit: 1.0,
            },
        }
    }

    #[test]
    fn recurrence_requires_three_distinct_components_and_is_label_blind() {
        let slates = vec![
            structural_slate(
                "ROW1",
                "COMP1",
                RootColor::Black,
                1,
                [7, 10, 11, 12, 13, 14],
            ),
            structural_slate(
                "ROW2",
                "COMP2",
                RootColor::White,
                1,
                [7, 10, 21, 22, 23, 24],
            ),
            structural_slate(
                "ROW3",
                "COMP3",
                RootColor::Black,
                1,
                [7, 31, 32, 33, 34, 35],
            ),
        ];
        // q and utilities are NaN on purpose. A1 must not inspect them.
        let a1 = a1_census(&slates).unwrap();
        assert!(a1.recurrent_codes.contains(&7));
        assert!(!a1.recurrent_codes.contains(&10));
    }

    #[test]
    fn analyze_returns_after_a1_without_reading_nan_label_fields() {
        let ordinals = [1u8, 2, 4, 6, 8];
        let mut slates = Vec::with_capacity(EXPECTED_SLATES);
        let mut row_index = 0usize;
        for component in 0..EXPECTED_COMPONENTS {
            let pairs = 1 + usize::from(component < 280);
            for _ in 0..pairs {
                for color in [RootColor::Black, RootColor::White] {
                    let codes = std::array::from_fn(|candidate| {
                        u32::try_from(row_index * K6 + candidate).unwrap()
                    });
                    slates.push(structural_slate(
                        &format!("ROW{row_index:04}"),
                        &format!("COMP{component:03}"),
                        color,
                        ordinals[row_index % ordinals.len()],
                        codes,
                    ));
                    row_index += 1;
                }
            }
        }
        assert_eq!(slates.len(), EXPECTED_SLATES);
        let outcome = analyze(&slates).unwrap();
        assert_eq!(outcome.final_label, STATE_EXPLOSION_LABEL);
        assert_eq!(outcome.report["q_or_product_fields_read"], false);
    }

    #[test]
    fn full_fit_does_not_memorize_nonrecurrent_codes() {
        let rows = vec![
            prepared_row(0, 0, 2.0, 9),
            prepared_row(1, 1, 4.0, 9),
            prepared_row(2, 2, 6.0, 9),
            prepared_row(3, 3, 100.0, 99),
        ];
        let recurrent = BTreeSet::from([9u32]);
        let correction = fit_observation_weighted_recurrent(&rows, &recurrent).unwrap();
        assert_eq!(correction.len(), 1);
        assert_eq!(correction.get(&9), Some(&4.0));
        assert!(!correction.contains_key(&99));
    }

    #[test]
    fn recurrent_dispersion_splits_consistent_abstraction_codes() {
        let mut rows = vec![
            prepared_row(0, 0, 2.0, 9),
            prepared_row(1, 1, 4.0, 9),
            prepared_row(2, 2, 6.0, 9),
        ];
        let recurrent = BTreeSet::from([9u32]);
        let corrections = fit_observation_weighted_recurrent(&rows, &recurrent).unwrap();
        let report = recurrent_residual_dispersion(&rows, &recurrent, &corrections).unwrap();
        assert_eq!(
            report["single_exact_transition_or_duplicate_code"]["distinct_codes"],
            1
        );
        assert_eq!(report["abstraction_collision_code"]["observations"], 0);

        rows[0].code_is_abstraction[0] = true;
        assert!(
            recurrent_residual_dispersion(&rows, &recurrent, &corrections)
                .unwrap_err()
                .contains("inconsistent abstraction")
        );
    }

    #[test]
    fn q_top_credit_is_exact_prediction_set_precision() {
        let q = [0.40, 0.40, 0.05, 0.05, 0.05, 0.05];
        assert_eq!(
            q_top_credit(&q, &[3.0, 3.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
            1.0
        );
        assert_eq!(
            q_top_credit(&q, &[3.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap(),
            0.5
        );
        assert_eq!(
            q_top_credit(&q, &[2.0, 1.0, 3.0, 1.0, 1.0, 1.0]).unwrap(),
            0.0
        );
    }

    #[test]
    fn loo_excludes_held_out_component_and_requires_two_donors() {
        let means = BTreeMap::from([(0usize, 10.0), (1, 2.0), (2, 4.0)]);
        assert_eq!(leave_one_component_out_mean(&means, 0).unwrap(), 3.0);
        let only_two = BTreeMap::from([(0usize, 10.0), (1, 2.0)]);
        assert_eq!(leave_one_component_out_mean(&only_two, 0).unwrap(), 0.0);
    }

    #[test]
    fn splitmix_and_nearest_rank_are_deterministic() {
        let mut first = SplitMix64::new(BOOTSTRAP_SEED);
        let mut second = SplitMix64::new(BOOTSTRAP_SEED);
        for _ in 0..1_000 {
            assert_eq!(first.next_u64(), second.next_u64());
        }
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(nearest_rank(&sorted, 0.05).unwrap(), 1.0);
        assert_eq!(nearest_rank(&sorted, 0.95).unwrap(), 5.0);
    }

    #[test]
    fn bootstrap_is_deterministic_and_rejects_invalid_denominator() {
        // Every frozen component contains both paired root colors, so every
        // whole-component bootstrap replicate has a positive color denominator.
        let rows = (0..3)
            .flat_map(|component| {
                [RootColor::Black, RootColor::White]
                    .into_iter()
                    .enumerate()
                    .map(move |(color_index, color)| {
                        let row_index = component * 2 + color_index;
                        let mut row = prepared_row(row_index, component, 1.0, 1);
                        row.root_color = color;
                        row
                    })
            })
            .collect::<Vec<_>>();
        let scores = rows
            .iter()
            .map(|row| RowProjectionScore {
                component: row.component,
                root_color: row.root_color,
                ordinal: row.ordinal,
                entropy: row.entropy,
                base: row.base,
                corrected: BasicRowMetrics {
                    ce: row.base.ce,
                    kl: 0.05,
                    sse: 3.0,
                    credit: 1.0,
                },
                recurrent_teacher_mass: 1.0,
                target_tie_cardinality: row.target_tie_cardinality,
                stored_teacher_top_in_target_set: row.stored_teacher_top_in_target_set,
            })
            .collect::<Vec<_>>();
        let first = bootstrap_components(&rows, &scores, 3, 100, 123).unwrap();
        let second = bootstrap_components(&rows, &scores, 3, 100, 123).unwrap();
        assert_eq!(first.combined_r_kl, second.combined_r_kl);
        assert_eq!(first.combined_r_sse, second.combined_r_sse);

        let invalid_scores = scores
            .iter()
            .copied()
            .map(|mut score| {
                score.base.kl = 0.0;
                score
            })
            .collect::<Vec<_>>();
        let error = bootstrap_components(&rows, &invalid_scores, 3, 1, 123).unwrap_err();
        assert!(error.contains("denominator"));
    }

    #[test]
    fn kl_below_registered_boundary_is_invalid() {
        let q = [1.0 / K6 as f64; K6];
        let logits = [0.0; K6];
        let (_, kl) = ce_and_kl(&q, &logits).unwrap();
        assert!(kl >= KL_NEGATIVE_INVALID);
        assert!(require_finite("invalid", f64::NAN).is_err());
        assert!(quantiles(&mut [0.0, f64::NAN]).is_err());
    }
}
