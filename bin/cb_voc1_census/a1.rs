//! CB-VOC1 Stage A1 train-only decision-gradient census.

use super::gradient::{self, Neumaier, DIM, K6};
use super::orbit::{
    color_swap_sigma, released_geometry, select_exact_capacity, OrbitGeometry, SelectionMembership,
    SelectionMetricAccumulator, SelectionMetrics, INCUMBENT_ROWS,
};
use super::{independent_forward, sha256_hex, stone_index, Slate};
use figrid_board::board::{Stone, NUM_CELLS};
use figrid_board::codebook_eval::{evaluate_full_quantized, QuantizedCodebookWeights};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

const SLOTS_PER_SLATE: u64 = (K6 * NUM_CELLS * 4) as u64;
const SUPPORT_PROTECTION: u64 = 128;

pub(crate) struct A1Outcome {
    pub(crate) report: Value,
    pub(crate) passed: bool,
}

#[derive(Default)]
struct RawCensus {
    combined: BTreeMap<u32, u64>,
    colors: [BTreeMap<u32, u64>; 2],
    ordinals: BTreeMap<usize, BTreeMap<u32, u64>>,
}

impl RawCensus {
    fn observe_slate(
        &mut self,
        side: Stone,
        ordinal: usize,
        slots: &BTreeMap<u32, u32>,
    ) -> Result<(), String> {
        let color = stone_index(side);
        let ordinal_counts = self.ordinals.entry(ordinal).or_default();
        for (&raw, &count) in slots {
            let count = u64::from(count);
            checked_add_count(&mut self.combined, raw, count)?;
            checked_add_count(&mut self.colors[color], raw, count)?;
            checked_add_count(ordinal_counts, raw, count)?;
        }
        Ok(())
    }
}

pub(crate) fn run_a1(
    slates: &[Slate],
    product: &QuantizedCodebookWeights,
    topk_bytes: &[u8],
) -> Result<A1Outcome, String> {
    if slates.is_empty() {
        return Err("A1 cannot run on an empty train corpus".to_string());
    }

    let geometry = released_geometry(topk_bytes)?;
    let mut gradient_sums = BTreeMap::<u32, [Neumaier; DIM]>::new();
    let mut residual_sums = BTreeMap::<u32, Neumaier>::new();
    let mut raw_census = RawCensus::default();
    let mut ce_sum = Neumaier::default();
    let mut ce_rows = Vec::with_capacity(slates.len());
    let mut zero_preactivations = 0u64;
    let mut max_alpha_sum_abs = 0.0f64;
    let mut max_policy_sum_error = 0.0f64;

    for slate in slates {
        let mut children = Vec::with_capacity(K6);
        for candidate in &slate.candidates {
            let mut child = slate.parent.clone();
            child.make_move(candidate.mv);
            let analysis = gradient::analyze_product_child(&child, product)?;
            let a0_independent = independent_forward(&child, product)?;
            let released = evaluate_full_quantized(&child, product);
            if analysis.ell.to_bits() != released.to_bits()
                || analysis.ell.to_bits() != a0_independent.value.to_bits()
            {
                return Err(format!(
                    "{} candidate {} gradient/A0/public forward mismatch: {:08X}/{:08X}/{:08X}",
                    slate.row_uid,
                    candidate.mv,
                    analysis.ell.to_bits(),
                    a0_independent.value.to_bits(),
                    released.to_bits()
                ));
            }
            if analysis.precast.to_bits() != a0_independent.precast.to_bits() {
                return Err(format!(
                    "{} candidate {} independent pre-cast mismatch: {:016X} != {:016X}",
                    slate.row_uid,
                    candidate.mv,
                    analysis.precast.to_bits(),
                    a0_independent.precast.to_bits()
                ));
            }
            zero_preactivations = zero_preactivations
                .checked_add(analysis.zero_preactivations)
                .ok_or_else(|| "zero-preactivation counter overflow".to_string())?;
            children.push(analysis);
        }
        let children: [gradient::ChildAnalysis; K6] = children
            .try_into()
            .map_err(|_| "A1 candidate array conversion failed".to_string())?;
        let row = gradient::combine_k6_gradient(&children, &slate.q_teacher)?;

        ce_sum.add(row.ce);
        ce_rows.push(row.ce);
        let alpha_sum = row.alpha.iter().copied().sum::<f64>().abs();
        let policy_sum_error = (row.policy.iter().copied().sum::<f64>() - 1.0).abs();
        max_alpha_sum_abs = max_alpha_sum_abs.max(alpha_sum);
        max_policy_sum_error = max_policy_sum_error.max(policy_sum_error);

        for (&raw, dimensions) in &row.raw_gradient {
            if !geometry.contains_universe_row(raw) {
                return Err(format!(
                    "gradient raw token is outside released universe: {raw:08X}"
                ));
            }
            let target = gradient_sums.entry(raw).or_default();
            for dimension in 0..DIM {
                target[dimension].add(dimensions[dimension]);
            }
        }
        for (&raw, &residual) in &row.raw_occurrence_residual {
            residual_sums.entry(raw).or_default().add(residual);
        }
        raw_census.observe_slate(slate.root_side, slate.ordinal, &row.raw_slots)?;
    }

    let expected_total = (slates.len() as u64)
        .checked_mul(SLOTS_PER_SLATE)
        .ok_or_else(|| "expected slot total overflow".to_string())?;
    require_count_total("combined", &raw_census.combined, expected_total)?;
    let expected_color = expected_total / 2;
    require_count_total("Black", &raw_census.colors[0], expected_color)?;
    require_count_total("White", &raw_census.colors[1], expected_color)?;
    let ordinal_total = raw_census.ordinals.values().try_fold(0u64, |sum, counts| {
        sum.checked_add(counts.values().copied().sum::<u64>())
            .ok_or_else(|| "ordinal slot total overflow".to_string())
    })?;
    if ordinal_total != expected_total {
        return Err(format!(
            "ordinal slot total {ordinal_total}, expected {expected_total}"
        ));
    }

    let denominator = slates.len() as f64;
    let mut row_values = BTreeMap::<u32, f64>::new();
    for (&raw, sums) in &gradient_sums {
        let mut norm = Neumaier::default();
        for sum in sums {
            let mean = sum.total() / denominator;
            norm.add(mean * mean);
        }
        let value = norm.total();
        if !value.is_finite() || value < 0.0 {
            return Err(format!("invalid row value for {raw:08X}: {value}"));
        }
        row_values.insert(raw, value);
    }

    let selection = select_exact_capacity(&geometry, INCUMBENT_ROWS, |raw| {
        row_values.get(&raw).copied().unwrap_or(0.0)
    })?;
    validate_selection(&geometry, &selection.rows)?;
    let r_phi = selection.r_phi()?;
    let membership = SelectionMembership::new(&geometry.incumbent, &selection.rows)?;

    let combined = selection_metrics(&membership, &raw_census.combined)?;
    let colors = [
        selection_metrics(&membership, &raw_census.colors[0])?,
        selection_metrics(&membership, &raw_census.colors[1])?,
    ];
    let mut ordinals = BTreeMap::<usize, SelectionMetrics>::new();
    for (&ordinal, counts) in &raw_census.ordinals {
        ordinals.insert(ordinal, selection_metrics(&membership, counts)?);
    }

    let selected_set: BTreeSet<u32> = selection.rows.iter().copied().collect();
    let gained_orbits = changed_orbits(&geometry, &selected_set, true);
    let lost_orbits = changed_orbits(&geometry, &selected_set, false);
    if gained_orbits.len() != selection.gained_orbits || lost_orbits.len() != selection.lost_orbits
    {
        return Err("changed-orbit detail/count mismatch".to_string());
    }
    let mut protected_removed = Vec::<Value>::new();
    for &orbit in &geometry.orbits {
        let incumbent = geometry.incumbent.contains(orbit.first);
        let selected = selected_set.contains(&orbit.first);
        if incumbent && !selected {
            let first_support = support_of(&raw_census.combined, orbit.first);
            let second_support = support_of(&raw_census.combined, orbit.second);
            if first_support.max(second_support) >= SUPPORT_PROTECTION {
                protected_removed.push(json!({
                    "first": hex8(orbit.first),
                    "second": hex8(orbit.second),
                    "first_support": first_support,
                    "second_support": second_support,
                    "max_member_support": first_support.max(second_support)
                }));
            }
        }
    }

    let ordinal_gain_pass = ordinals
        .values()
        .all(|metric| metric.gain_percentage_points >= 0.0);
    let ordinal_loss_pass = ordinals
        .values()
        .all(|metric| metric.gross_loss_percentage_points <= 0.50);
    let gates = vec![
        gate("r_phi_ge_0_03", r_phi, 0.03, r_phi >= 0.03),
        gate(
            "combined_addressability_gain_ge_1pp",
            combined.gain_percentage_points,
            1.00,
            combined.gain_percentage_points >= 1.00,
        ),
        gate(
            "black_addressability_gain_ge_0_75pp",
            colors[0].gain_percentage_points,
            0.75,
            colors[0].gain_percentage_points >= 0.75,
        ),
        gate(
            "white_addressability_gain_ge_0_75pp",
            colors[1].gain_percentage_points,
            0.75,
            colors[1].gain_percentage_points >= 0.75,
        ),
        json!({
            "name": "every_ordinal_addressability_gain_ge_0pp",
            "threshold": 0.0,
            "pass": ordinal_gain_pass
        }),
        gate(
            "combined_gross_loss_le_0_25pp",
            combined.gross_loss_percentage_points,
            0.25,
            combined.gross_loss_percentage_points <= 0.25,
        ),
        json!({
            "name": "every_color_gross_loss_le_0_50pp",
            "threshold": 0.50,
            "pass": colors
                .iter()
                .all(|metric| metric.gross_loss_percentage_points <= 0.50)
        }),
        json!({
            "name": "every_ordinal_gross_loss_le_0_50pp",
            "threshold": 0.50,
            "pass": ordinal_loss_pass
        }),
        json!({
            "name": "zero_removed_incumbent_orbit_with_member_support_ge_128",
            "threshold": 0,
            "observed": protected_removed.len(),
            "pass": protected_removed.is_empty()
        }),
    ];
    let passed = gates
        .iter()
        .all(|gate| gate.get("pass").and_then(Value::as_bool) == Some(true));

    ce_rows.sort_by(f64::total_cmp);
    let mut row_value_stream = Vec::with_capacity(geometry.universe.len() * 20);
    for &raw in &geometry.universe {
        row_value_stream.extend_from_slice(&raw.to_le_bytes());
        row_value_stream.extend_from_slice(
            &row_values
                .get(&raw)
                .copied()
                .unwrap_or(0.0)
                .to_bits()
                .to_le_bytes(),
        );
        row_value_stream.extend_from_slice(&support_of(&raw_census.combined, raw).to_le_bytes());
    }
    let mut selected_bytes = Vec::with_capacity(selection.rows.len() * 4);
    for &raw in &selection.rows {
        selected_bytes.extend_from_slice(&raw.to_le_bytes());
    }

    let report = json!({
        "stage": "A1_POINT_UPPER_BOUND",
        "status": if passed {
            "A1_PASS_A2_REQUIRED"
        } else {
            "NO_GO_PRECONDITION"
        },
        "geometry": {
            "universe_rows": geometry.universe.len(),
            "anchor_boundary_rows": geometry.anchor_boundary_rows,
            "universe_orbits": geometry.orbits.len(),
            "universe_fixed_orbits": geometry.fixed_orbits,
            "universe_pair_orbits": geometry.pair_orbits,
            "incumbent_rows": geometry.incumbent.rows_by_id().len(),
            "incumbent_fixed_orbits": geometry.incumbent.fixed_orbits,
            "incumbent_pair_orbits": geometry.incumbent.pair_orbits,
            "incumbent_frequency_rows": 4_096,
            "incumbent_closure_tail_rows": 169,
            "rare_selectable": false,
            "selection_realisable_unique_orbit_closed": true
        },
        "selector": {
            "capacity_rows": selection.capacity,
            "fixed_orbits": selection.fixed_orbits,
            "pair_orbits": selection.pair_orbits,
            "selector_prefix_objective": selection.selector_objective,
            "phi_selected": selection.phi_selected,
            "phi_incumbent": selection.phi_incumbent,
            "r_phi": r_phi,
            "incumbent_rows_retained": selection.incumbent_rows_retained,
            "symmetric_difference_rows": selection.symmetric_difference_rows,
            "gained_rows_count": selection.gained_rows.len(),
            "lost_rows_count": selection.lost_rows.len(),
            "gained_orbits": selection.gained_orbits,
            "lost_orbits": selection.lost_orbits,
            "gained_orbit_details": gained_orbits,
            "lost_orbit_details": lost_orbits,
            "selected_rows_le_u32_sha256": sha256_hex(&selected_bytes),
            "selected_rows_hex": hex_rows(&selection.rows),
            "gained_rows": row_details(
                &selection.gained_rows,
                &row_values,
                &raw_census.combined
            ),
            "lost_rows": row_details(
                &selection.lost_rows,
                &row_values,
                &raw_census.combined
            )
        },
        "decision_gradient": {
            "train_slates": slates.len(),
            "children": slates.len() * K6,
            "raw_slots": expected_total,
            "observed_raw_rows": raw_census.combined.len(),
            "nonzero_gradient_rows": row_values.values().filter(|&&value| value > 0.0).count(),
            "zero_cell_preactivations": zero_preactivations,
            "mean_cross_entropy": ce_sum.total() / denominator,
            "cross_entropy": describe_sorted(&ce_rows),
            "max_abs_alpha_sum": max_alpha_sum_abs,
            "max_policy_sum_error": max_policy_sum_error,
            "gradient_vs_a0_precast_bit_mismatches": 0,
            "gradient_vs_a0_f32_bit_mismatches": 0,
            "gradient_vs_public_f32_bit_mismatches": 0,
            "all_finite": true,
            "row_value_support_stream_sha256": sha256_hex(&row_value_stream),
            "raw_occurrence_residual": residual_diagnostics(&residual_sums, denominator)?
        },
        "addressability": {
            "combined": metrics_json(combined),
            "black": metrics_json(colors[0]),
            "white": metrics_json(colors[1]),
            "ordinals": ordinals
                .iter()
                .map(|(ordinal, metric)| (ordinal.to_string(), metrics_json(*metric)))
                .collect::<serde_json::Map<String, Value>>()
        },
        "support_buckets": {
            "definition": {
                "protected_orbit": "max(member raw occurrence)>=128; member counts are not pooled",
                "buckets": ["zero", "1", "2..7", "8..31", "32..127", "128+"]
            },
            "universe": support_summary(
                geometry.universe.iter().copied(),
                &raw_census.combined
            ),
            "incumbent": support_summary(
                geometry.incumbent.rows_by_id().iter().copied(),
                &raw_census.combined
            ),
            "selected": support_summary(
                selection.rows.iter().copied(),
                &raw_census.combined
            ),
            "gained": support_summary(
                selection.gained_rows.iter().copied(),
                &raw_census.combined
            ),
            "lost": support_summary(
                selection.lost_rows.iter().copied(),
                &raw_census.combined
            ),
            "protected_removed_orbits": protected_removed
        },
        "gates": gates,
        "all_a1_gates_pass": passed,
        "next_stage": if passed {
            "A2_COMPONENT_HELD_OUT_ROBUSTNESS"
        } else {
            "STOP_CB_VOC1"
        }
    });
    Ok(A1Outcome { report, passed })
}

fn validate_selection(geometry: &OrbitGeometry, rows: &[u32]) -> Result<(), String> {
    if rows.len() != INCUMBENT_ROWS {
        return Err(format!(
            "selected row count {}, expected {INCUMBENT_ROWS}",
            rows.len()
        ));
    }
    if rows.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("selected rows are not unique and strictly ascending".to_string());
    }
    let selected: BTreeSet<u32> = rows.iter().copied().collect();
    for &raw in rows {
        if !geometry.contains_universe_row(raw) {
            return Err(format!(
                "selected row is not released-realizable: {raw:08X}"
            ));
        }
        let partner = color_swap_sigma(raw);
        if !selected.contains(&partner) {
            return Err(format!(
                "selected vocabulary is not color-orbit closed: {raw:08X} -> {partner:08X}"
            ));
        }
    }
    Ok(())
}

fn changed_orbits(geometry: &OrbitGeometry, selected: &BTreeSet<u32>, gained: bool) -> Vec<Value> {
    geometry
        .orbits
        .iter()
        .filter_map(|orbit| {
            let in_incumbent = geometry.incumbent.contains(orbit.first);
            let in_candidate = selected.contains(&orbit.first);
            let changed = if gained {
                in_candidate && !in_incumbent
            } else {
                in_incumbent && !in_candidate
            };
            changed.then(|| {
                json!({
                    "first": hex8(orbit.first),
                    "second": hex8(orbit.second),
                    "cost": orbit.cost()
                })
            })
        })
        .collect()
}

fn checked_add_count(counts: &mut BTreeMap<u32, u64>, raw: u32, count: u64) -> Result<(), String> {
    let slot = counts.entry(raw).or_default();
    *slot = slot
        .checked_add(count)
        .ok_or_else(|| format!("raw count overflow for {raw:08X}"))?;
    Ok(())
}

fn require_count_total(
    name: &str,
    counts: &BTreeMap<u32, u64>,
    expected: u64,
) -> Result<(), String> {
    let total = counts.values().try_fold(0u64, |sum, &count| {
        sum.checked_add(count)
            .ok_or_else(|| format!("{name} count total overflow"))
    })?;
    if total != expected {
        return Err(format!("{name} raw slots {total}, expected {expected}"));
    }
    Ok(())
}

fn selection_metrics(
    membership: &SelectionMembership,
    counts: &BTreeMap<u32, u64>,
) -> Result<SelectionMetrics, String> {
    let mut accumulator = SelectionMetricAccumulator::default();
    for (&raw, &count) in counts {
        accumulator.observe_count(membership, raw, count)?;
    }
    accumulator.finish()
}

fn metrics_json(metric: SelectionMetrics) -> Value {
    json!({
        "total_slots": metric.counts.total_slots,
        "incumbent_addressed_slots": metric.counts.incumbent_addressed,
        "candidate_addressed_slots": metric.counts.candidate_addressed,
        "retained_addressed_slots": metric.counts.retained_addressed,
        "gained_addressed_slots": metric.counts.gained_addressed,
        "lost_addressed_slots": metric.counts.lost_addressed,
        "incumbent_percent": metric.incumbent_percent,
        "candidate_percent": metric.candidate_percent,
        "gain_percentage_points": metric.gain_percentage_points,
        "gross_loss_percentage_points": metric.gross_loss_percentage_points
    })
}

fn gate(name: &str, observed: f64, threshold: f64, pass: bool) -> Value {
    json!({
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "pass": pass
    })
}

fn support_of(counts: &BTreeMap<u32, u64>, raw: u32) -> u64 {
    counts.get(&raw).copied().unwrap_or(0)
}

fn support_bucket(support: u64) -> usize {
    match support {
        0 => 0,
        1 => 1,
        2..=7 => 2,
        8..=31 => 3,
        32..=127 => 4,
        _ => 5,
    }
}

fn support_summary(rows: impl IntoIterator<Item = u32>, counts: &BTreeMap<u32, u64>) -> Value {
    let mut row_counts = [0u64; 6];
    let mut slot_mass = [0u64; 6];
    for raw in rows {
        let support = support_of(counts, raw);
        let bucket = support_bucket(support);
        row_counts[bucket] += 1;
        slot_mass[bucket] += support;
    }
    json!({
        "rows": {
            "zero": row_counts[0],
            "1": row_counts[1],
            "2..7": row_counts[2],
            "8..31": row_counts[3],
            "32..127": row_counts[4],
            "128+": row_counts[5]
        },
        "slot_mass": {
            "zero": slot_mass[0],
            "1": slot_mass[1],
            "2..7": slot_mass[2],
            "8..31": slot_mass[3],
            "32..127": slot_mass[4],
            "128+": slot_mass[5]
        }
    })
}

fn row_details(
    rows: &[u32],
    values: &BTreeMap<u32, f64>,
    counts: &BTreeMap<u32, u64>,
) -> Vec<Value> {
    rows.iter()
        .map(|&raw| {
            json!({
                "raw": hex8(raw),
                "value": values.get(&raw).copied().unwrap_or(0.0),
                "support": support_of(counts, raw)
            })
        })
        .collect()
}

fn hex_rows(rows: &[u32]) -> Vec<String> {
    rows.iter().map(|&raw| hex8(raw)).collect()
}

fn hex8(raw: u32) -> String {
    format!("{raw:08X}")
}

fn residual_diagnostics(
    residuals: &BTreeMap<u32, Neumaier>,
    denominator: f64,
) -> Result<Value, String> {
    let mut l1 = Neumaier::default();
    let mut l2 = Neumaier::default();
    let mut max_abs = 0.0f64;
    let mut stream = Vec::with_capacity(residuals.len() * 12);
    for (&raw, sum) in residuals {
        let mean = sum.total() / denominator;
        if !mean.is_finite() {
            return Err(format!("non-finite occurrence residual for {raw:08X}"));
        }
        l1.add(mean.abs());
        l2.add(mean * mean);
        max_abs = max_abs.max(mean.abs());
        stream.extend_from_slice(&raw.to_le_bytes());
        stream.extend_from_slice(&mean.to_bits().to_le_bytes());
    }
    Ok(json!({
        "claim": "diagnostic only; never used as selector value",
        "rows": residuals.len(),
        "mean_l1": l1.total(),
        "mean_l2_squared": l2.total(),
        "max_abs": max_abs,
        "stream_sha256": sha256_hex(&stream)
    }))
}

fn describe_sorted(values: &[f64]) -> Value {
    json!({
        "count": values.len(),
        "min": values.first().copied().unwrap_or(0.0),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": values.last().copied().unwrap_or(0.0)
    })
}

fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let rank = ((q * values.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[rank]
}
