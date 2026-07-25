#![cfg(feature = "codebook-eval")]

//! Authoritative CB-GH1 P0 rooted threat-transition graph census.
//!
//! This binary accepts only the five sealed RQ615C-train/model inputs and a
//! create-new report path. It has no CLI surface for dev, safety, game
//! outcomes, the frozen 64-game holdout, or the frozen 1,022-root trace.

use figrid_board::board::{Board, Stone};
use figrid_board::d4_hash::{D4HashState, exact_canonical_state};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[path = "cb_gh1_graph_census/corpus.rs"]
mod corpus;
#[path = "cb_gh1_graph_census/graph.rs"]
mod graph;
#[path = "cb_gh1_graph_census/provenance.rs"]
mod provenance;
#[path = "cb_gh1_graph_census/stats.rs"]
mod stats;

const FORMAT: &str = "cb-gh1-p0-rooted-threat-graph-census-v1";
const PREREGISTER_COMMIT: &str = "3e52280";
const EXECUTABLE_STEM: &str = "cb-gh1-graph-census";
const CANONICAL_BUILD: &str =
    "cargo build --release --locked --features codebook-eval --bin cb-gh1-graph-census";
const CRITICAL_SOURCES: [(&str, &[u8]); 44] = [
    ("Cargo.toml", include_bytes!("../Cargo.toml")),
    ("Cargo.lock", include_bytes!("../Cargo.lock")),
    ("src/board.rs", include_bytes!("../src/board.rs")),
    ("src/book.rs", include_bytes!("../src/book.rs")),
    (
        "src/candidate_local_ensemble.rs",
        include_bytes!("../src/candidate_local_ensemble.rs"),
    ),
    (
        "src/candidate_ranker.rs",
        include_bytes!("../src/candidate_ranker.rs"),
    ),
    (
        "src/codebook_eval.rs",
        include_bytes!("../src/codebook_eval.rs"),
    ),
    (
        "src/codebook_sidecar.rs",
        include_bytes!("../src/codebook_sidecar.rs"),
    ),
    ("src/coord.rs", include_bytes!("../src/coord.rs")),
    ("src/d4_hash.rs", include_bytes!("../src/d4_hash.rs")),
    ("src/eval.rs", include_bytes!("../src/eval.rs")),
    (
        "src/factored_codebook.rs",
        include_bytes!("../src/factored_codebook.rs"),
    ),
    ("src/features.rs", include_bytes!("../src/features.rs")),
    ("src/heuristic.rs", include_bytes!("../src/heuristic.rs")),
    (
        "src/legacy/evaluator.rs",
        include_bytes!("../src/legacy/evaluator.rs"),
    ),
    ("src/legacy/mod.rs", include_bytes!("../src/legacy/mod.rs")),
    ("src/legacy/rec.rs", include_bytes!("../src/legacy/rec.rs")),
    (
        "src/legacy/rec_base.rs",
        include_bytes!("../src/legacy/rec_base.rs"),
    ),
    (
        "src/legacy/rec_checker.rs",
        include_bytes!("../src/legacy/rec_checker.rs"),
    ),
    ("src/legacy/row.rs", include_bytes!("../src/legacy/row.rs")),
    (
        "src/legacy/rule.rs",
        include_bytes!("../src/legacy/rule.rs"),
    ),
    (
        "src/legacy/tree.rs",
        include_bytes!("../src/legacy/tree.rs"),
    ),
    ("src/lib.rs", include_bytes!("../src/lib.rs")),
    (
        "src/pattern_dense.rs",
        include_bytes!("../src/pattern_dense.rs"),
    ),
    (
        "src/pattern_table.rs",
        include_bytes!("../src/pattern_table.rs"),
    ),
    (
        "src/relation_fusion_gate.rs",
        include_bytes!("../src/relation_fusion_gate.rs"),
    ),
    (
        "src/relation_lite.rs",
        include_bytes!("../src/relation_lite.rs"),
    ),
    (
        "src/rq423_root_accept.rs",
        include_bytes!("../src/rq423_root_accept.rs"),
    ),
    ("src/search.rs", include_bytes!("../src/search.rs")),
    (
        "src/threat_field.rs",
        include_bytes!("../src/threat_field.rs"),
    ),
    (
        "src/token_delta.rs",
        include_bytes!("../src/token_delta.rs"),
    ),
    (
        "src/transposition.rs",
        include_bytes!("../src/transposition.rs"),
    ),
    ("src/tss.rs", include_bytes!("../src/tss.rs")),
    ("src/vct.rs", include_bytes!("../src/vct.rs")),
    (
        "src/white_root_order.rs",
        include_bytes!("../src/white_root_order.rs"),
    ),
    ("data/topk.bin", include_bytes!("../data/topk.bin")),
    (
        "models/gomoku_codebook_v1_swapclosed.json",
        include_bytes!("../models/gomoku_codebook_v1_swapclosed.json"),
    ),
    (
        "models/gomoku_codebook_v1_swapclosed_factored.cbf",
        include_bytes!("../models/gomoku_codebook_v1_swapclosed_factored.cbf"),
    ),
    (
        "bin/cb_gh1_graph_census.rs",
        include_bytes!("cb_gh1_graph_census.rs"),
    ),
    (
        "bin/cb_gh1_graph_census/corpus.rs",
        include_bytes!("cb_gh1_graph_census/corpus.rs"),
    ),
    (
        "bin/cb_gh1_graph_census/graph.rs",
        include_bytes!("cb_gh1_graph_census/graph.rs"),
    ),
    (
        "bin/cb_gh1_graph_census/provenance.rs",
        include_bytes!("cb_gh1_graph_census/provenance.rs"),
    ),
    (
        "bin/cb_gh1_graph_census/stats.rs",
        include_bytes!("cb_gh1_graph_census/stats.rs"),
    ),
    (
        "experiments/2026-07-26/cb_gh1_threat_graph_preregister.md",
        include_bytes!("../experiments/2026-07-26/cb_gh1_threat_graph_preregister.md"),
    ),
];

#[derive(Debug)]
struct Args {
    inputs: corpus::InputPaths,
    out_report: PathBuf,
}

#[derive(Clone, Debug, Default)]
struct AuditAggregate {
    transformed_board_checks: u64,
    transformed_board_mismatches: u64,
    coordinate_graph_checks: u64,
    coordinate_graph_mismatches: u64,
    coordinate_role_checks: u64,
    coordinate_role_mismatches: u64,
    canonical_bytes_checks: u64,
    canonical_bytes_mismatches: u64,
    digest_checks: u64,
    digest_mismatches: u64,
    key64_checks: u64,
    key64_mismatches: u64,
    exact_role_checks: u64,
    exact_role_mismatches: u64,
    min_mask_checks: u64,
    min_mask_mismatches: u64,
    color_role_checks: u64,
    color_role_mismatches: u64,
    details: Vec<String>,
    details_truncated: u64,
}

impl AuditAggregate {
    fn add(&mut self, audit: graph::TransitionAudit, row_uid: &str, candidate: usize) {
        let missing_detail_for_mismatch = audit.mismatch_count() != 0 && audit.details.is_empty();
        self.transformed_board_checks += audit.transformed_board_checks;
        self.transformed_board_mismatches += audit.transformed_board_mismatches;
        self.coordinate_graph_checks += audit.coordinate_graph_checks;
        self.coordinate_graph_mismatches += audit.coordinate_graph_mismatches;
        self.coordinate_role_checks += audit.coordinate_role_checks;
        self.coordinate_role_mismatches += audit.coordinate_role_mismatches;
        self.canonical_bytes_checks += audit.canonical_bytes_checks;
        self.canonical_bytes_mismatches += audit.canonical_bytes_mismatches;
        self.digest_checks += audit.digest_checks;
        self.digest_mismatches += audit.digest_mismatches;
        self.key64_checks += audit.key64_checks;
        self.key64_mismatches += audit.key64_mismatches;
        self.exact_role_checks += audit.exact_role_checks;
        self.exact_role_mismatches += audit.exact_role_mismatches;
        self.min_mask_checks += audit.min_mask_checks;
        self.min_mask_mismatches += audit.min_mask_mismatches;
        self.color_role_checks += audit.color_role_checks;
        self.color_role_mismatches += audit.color_role_mismatches;
        for detail in audit.details {
            if self.details.len() < 64 {
                self.details
                    .push(format!("{row_uid}/candidate-{candidate}: {detail}"));
            } else {
                self.details_truncated += 1;
            }
        }
        if missing_detail_for_mismatch {
            if self.details.len() < 64 {
                self.details.push(format!(
                    "{row_uid}/candidate-{candidate}: mismatch counter had no detail"
                ));
            } else {
                self.details_truncated += 1;
            }
        }
        self.details_truncated += audit.details_truncated;
    }

    fn mismatch_count(&self) -> u64 {
        self.transformed_board_mismatches
            + self.coordinate_graph_mismatches
            + self.coordinate_role_mismatches
            + self.canonical_bytes_mismatches
            + self.digest_mismatches
            + self.key64_mismatches
            + self.exact_role_mismatches
            + self.min_mask_mismatches
            + self.color_role_mismatches
    }

    fn json(&self) -> Value {
        json!({
            "mismatch_count": self.mismatch_count(),
            "transformed_board": {
                "checks": self.transformed_board_checks,
                "mismatches": self.transformed_board_mismatches,
            },
            "coordinate_graph": {
                "checks": self.coordinate_graph_checks,
                "mismatches": self.coordinate_graph_mismatches,
            },
            "coordinate_role": {
                "checks": self.coordinate_role_checks,
                "mismatches": self.coordinate_role_mismatches,
            },
            "canonical_bytes": {
                "checks": self.canonical_bytes_checks,
                "mismatches": self.canonical_bytes_mismatches,
            },
            "digest": {"checks": self.digest_checks, "mismatches": self.digest_mismatches},
            "key64": {"checks": self.key64_checks, "mismatches": self.key64_mismatches},
            "exact_role": {
                "checks": self.exact_role_checks,
                "mismatches": self.exact_role_mismatches,
            },
            "min_mask": {"checks": self.min_mask_checks, "mismatches": self.min_mask_mismatches},
            "color_role": {
                "checks": self.color_role_checks,
                "mismatches": self.color_role_mismatches,
            },
            "details": self.details,
            "details_truncated": self.details_truncated,
        })
    }
}

#[derive(Clone, Debug, Default)]
struct RoleIdentityCensus {
    occurrences: u64,
    color_mask: u8,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("CB-GH1 INVALID_CB_GH1_P0: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    refuse_existing(&args.out_report)?;
    let started_unix_ms = provenance::unix_millis()?;

    let source_before = provenance::source_identity(PREREGISTER_COMMIT, &CRITICAL_SOURCES)?;
    let environment = provenance::environment_identity(CANONICAL_BUILD)?;
    let toolchain = provenance::toolchain_identity()?;
    let cpu = provenance::cpu_identity()?;
    let executable = provenance::executable_identity(EXECUTABLE_STEM)?;

    let coordinate_audit = graph::audit_d4_coordinate_formulas();
    if !coordinate_audit.is_clean() {
        return Err(format!(
            "D4 coordinate formula audit failed: mismatches={} details={:?}",
            coordinate_audit.mismatch_count(),
            coordinate_audit.details
        ));
    }

    let corpus::CorpusBundle {
        slates,
        product: _product,
        lineage: _lineage,
        diagnostics: corpus_diagnostics,
    } = corpus::load_validate_and_replay(&args.inputs)?;

    let (stat_slates, graph_report) = build_graph_census(&slates, &coordinate_audit)?;
    let analysis = stats::analyze(&stat_slates)?;

    corpus::recheck_inputs(&args.inputs)?;
    let source_after = provenance::source_identity(PREREGISTER_COMMIT, &CRITICAL_SOURCES)?;
    if source_after != source_before {
        return Err("critical source identity changed during census".to_string());
    }
    let finished_unix_ms = provenance::unix_millis()?;
    let elapsed_ms = finished_unix_ms
        .checked_sub(started_unix_ms)
        .ok_or_else(|| "system clock moved backwards during census".to_string())?;

    let report = json!({
        "format": FORMAT,
        "final_label": analysis.final_label,
        "claim_boundary": {
            "train_only_exploratory_precondition": true,
            "rooted_candidate_ranking_code_not_position_value": true,
            "graph_is_exact_tt_or_proof_key": false,
            "incremental_runtime_implemented": false,
            "model_or_dictionary_trained": false,
            "product_default_changed": false,
            "benchmark_or_arena_opened": false,
            "forbidden_row_inputs_opened": 0,
            "forbidden_inputs": [
                "RQ615C dev",
                "RQ615C safety_internal",
                "RQ508",
                "game outcomes",
                "frozen 64-game search holdout",
                "frozen 1022-root timing trace"
            ]
        },
        "preregistration": {
            "commit": PREREGISTER_COMMIT,
            "document": "experiments/2026-07-26/cb_gh1_threat_graph_preregister.md"
        },
        "provenance": {
            "started_unix_ms": started_unix_ms.to_string(),
            "finished_unix_ms": finished_unix_ms.to_string(),
            "elapsed_ms": elapsed_ms.to_string(),
            "source": source_before,
            "environment": environment,
            "toolchain": toolchain,
            "cpu": cpu,
            "executable": executable,
        },
        "inputs": corpus::input_artifacts_json(&args.inputs),
        "corpus_a0": corpus_diagnostics,
        "graph_a0_a1_structure": graph_report,
        "statistical_stages": analysis.report,
        "downstream": downstream_json(analysis.final_label),
    });

    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("failed to serialize report: {error}"))?;
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.out_report)
        .map_err(|error| {
            format!(
                "failed to create report {}: {error}",
                args.out_report.display()
            )
        })?;
    if let Err(error) = output
        .write_all(&bytes)
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.flush())
        .and_then(|_| output.sync_all())
    {
        drop(output);
        let cleanup = fs::remove_file(&args.out_report)
            .map(|_| "partial report removed".to_string())
            .unwrap_or_else(|cleanup_error| {
                format!("partial report cleanup also failed: {cleanup_error}")
            });
        return Err(format!(
            "failed to write and durably sync report {}: {error}; {cleanup}",
            args.out_report.display()
        ));
    }
    println!(
        "CB-GH1 {}: slates={} transitions={} report={}",
        analysis.final_label,
        slates.len(),
        stat_slates.len() * corpus::K6,
        args.out_report.display()
    );
    Ok(())
}

fn build_graph_census(
    slates: &[corpus::Slate],
    coordinate_audit: &graph::CoordinateFormulaAudit,
) -> Result<(Vec<stats::StatsSlate>, Value), String> {
    let mut state_keys = BTreeMap::<u64, [u8; 66]>::new();
    let mut state_checks = 0u64;
    let mut code_by_bytes = BTreeMap::<Vec<u8>, u32>::new();
    let mut digest_to_bytes = BTreeMap::<[u8; 32], Vec<u8>>::new();
    let mut key_to_bytes = BTreeMap::<u64, Vec<u8>>::new();
    let mut role_census = BTreeMap::<(u32, Vec<u8>), RoleIdentityCensus>::new();
    let mut role_sets = BTreeMap::<u32, BTreeSet<Vec<u8>>>::new();
    let mut audit = AuditAggregate::default();
    let mut shapes = Vec::<graph::GraphShape>::with_capacity(slates.len() * corpus::K6);
    let mut masks = Vec::<u64>::with_capacity(slates.len() * corpus::K6);
    let mut code_stream = Vec::<u8>::with_capacity(slates.len() * corpus::K6 * 36);
    let mut stat_slates = Vec::with_capacity(slates.len());

    for slate in slates {
        audit_state_hash(&slate.parent, &mut state_keys)?;
        state_checks += 1;
        let mut code_ids = [0u32; corpus::K6];
        for (candidate_index, candidate) in slate.candidates.iter().enumerate() {
            let mut child = slate.parent.clone();
            child.make_move(candidate.mv);
            audit_state_hash(&child, &mut state_keys)?;
            state_checks += 1;

            let transition = graph::canonical_transition(&slate.parent, candidate.mv)?;
            validate_shape(&transition)?;
            let transition_audit =
                graph::audit_transition_equivariance(&slate.parent, candidate.mv)?;
            audit.add(transition_audit, &slate.row_uid, candidate_index);

            if let Some(previous) = digest_to_bytes.get(&transition.digest) {
                if previous != &transition.bytes {
                    return Err(format!(
                        "graph SHA-256 collision at {}/candidate-{candidate_index}",
                        slate.row_uid
                    ));
                }
            } else {
                digest_to_bytes.insert(transition.digest, transition.bytes.clone());
            }
            if let Some(previous) = key_to_bytes.get(&transition.key64) {
                if previous != &transition.bytes {
                    return Err(format!(
                        "graph u64 collision at {}/candidate-{candidate_index}",
                        slate.row_uid
                    ));
                }
            } else {
                key_to_bytes.insert(transition.key64, transition.bytes.clone());
            }

            let next_code = u32::try_from(code_by_bytes.len())
                .map_err(|_| "graph dictionary length exceeds u32".to_string())?;
            let code_id = *code_by_bytes
                .entry(transition.bytes.clone())
                .or_insert(next_code);
            code_ids[candidate_index] = code_id;
            role_sets
                .entry(code_id)
                .or_default()
                .insert(transition.exact_role_bytes.clone());
            let role = role_census
                .entry((code_id, transition.exact_role_bytes))
                .or_default();
            role.occurrences += 1;
            role.color_mask |= match slate.root_side {
                Stone::Black => 1,
                Stone::White => 2,
            };

            shapes.push(transition.shape);
            masks.push(u64::from(transition.min_mask.count_ones()));
            code_stream.extend_from_slice(&code_id.to_le_bytes());
            code_stream.extend_from_slice(&transition.digest);
        }
        stat_slates.push(stats::StatsSlate {
            row_uid: slate.row_uid.clone(),
            component_uid: slate.component_uid.clone(),
            root_color: match slate.root_side {
                Stone::Black => stats::RootColor::Black,
                Stone::White => stats::RootColor::White,
            },
            ordinal: u8::try_from(slate.ordinal)
                .map_err(|_| format!("ordinal does not fit u8: {}", slate.ordinal))?,
            q_teacher: slate.q_teacher,
            product_root_utility: slate.product_root_utilities,
            code_ids,
            teacher_top: std::array::from_fn(|index| slate.candidates[index].teacher_top),
            code_is_abstraction: [false; corpus::K6],
        });
    }

    if audit.mismatch_count() != 0 {
        return Err(format!(
            "transition equivariance audit failed: mismatches={} details={:?}",
            audit.mismatch_count(),
            audit.details
        ));
    }

    let mut dictionary_stream = Vec::new();
    for bytes in code_by_bytes.keys() {
        dictionary_stream.extend_from_slice(
            &u32::try_from(bytes.len())
                .map_err(|_| "graph byte string length exceeds u32".to_string())?
                .to_le_bytes(),
        );
        dictionary_stream.extend_from_slice(bytes);
    }

    let abstraction_collision_groups = role_sets.values().filter(|set| set.len() > 1).count();
    let abstraction_codes = role_sets
        .iter()
        .filter_map(|(&code, roles)| (roles.len() > 1).then_some(code))
        .collect::<BTreeSet<_>>();
    for slate in &mut stat_slates {
        slate.code_is_abstraction = slate.code_ids.map(|code| abstraction_codes.contains(&code));
    }
    let abstraction_identity_excess: usize = role_sets
        .values()
        .map(|set| set.len().saturating_sub(1))
        .sum();
    let color_isomorphism_groups = role_census
        .values()
        .filter(|entry| entry.color_mask == 3)
        .count();
    let duplicate_occurrence_excess: u64 = role_census
        .values()
        .map(|entry| {
            entry
                .occurrences
                .saturating_sub(u64::from(entry.color_mask.count_ones()))
        })
        .sum();

    let graph_report = json!({
        "status": "A0_GRAPH_INTEGRITY_PASS",
        "transitions": shapes.len(),
        "expected_transitions": slates.len() * corpus::K6,
        "coordinate_formula_audit": {
            "in_board_checks": coordinate_audit.in_board_checks,
            "in_board_mismatches": coordinate_audit.in_board_mismatches,
            "virtual_checks": coordinate_audit.virtual_checks,
            "virtual_mismatches": coordinate_audit.virtual_mismatches,
            "details": coordinate_audit.details,
            "details_truncated": coordinate_audit.details_truncated,
        },
        "equivariance": audit.json(),
        "production_state_hash": {
            "states_checked": state_checks,
            "distinct_u64_keys": state_keys.len(),
            "true_collisions": 0,
        },
        "graph_hash": {
            "distinct_exact_codes": code_by_bytes.len(),
            "distinct_sha256": digest_to_bytes.len(),
            "distinct_u64": key_to_bytes.len(),
            "sha256_collisions": 0,
            "u64_collisions": 0,
            "code_identity_stream_sha256": provenance::sha256_hex(&code_stream),
            "dictionary_lexicographic_stream_sha256": provenance::sha256_hex(&dictionary_stream),
        },
        "abstraction": {
            "exact_graph_groups_with_multiple_exact_role_transitions": abstraction_collision_groups,
            "exact_role_identity_excess_within_graph_groups": abstraction_identity_excess,
            "color_role_isomorphism_groups": color_isomorphism_groups,
            "exact_duplicate_occurrence_excess_after_color_isomorphism": duplicate_occurrence_excess,
        },
        "shape": {
            "total_nodes": describe_u64(
                shapes
                    .iter()
                    .map(|shape| u64::from(shape.board_cells) + u64::from(shape.boundary_cells))
                    .collect()
            )?,
            "affected_sites": describe_u64(shapes.iter().map(|shape| u64::from(shape.affected_sites)).collect())?,
            "board_cells": describe_u64(shapes.iter().map(|shape| u64::from(shape.board_cells)).collect())?,
            "boundary_cells": describe_u64(shapes.iter().map(|shape| u64::from(shape.boundary_cells)).collect())?,
            "factors": describe_u64(shapes.iter().map(|shape| u64::from(shape.factors)).collect())?,
            "incidences": describe_u64(shapes.iter().map(|shape| u64::from(shape.incidences)).collect())?,
            "serialized_bytes": describe_u64(shapes.iter().map(|shape| u64::from(shape.bytes)).collect())?,
            "minimum_transform_count": describe_u64(masks)?,
        },
    });
    Ok((stat_slates, graph_report))
}

fn validate_shape(transition: &graph::CanonicalTransition) -> Result<(), String> {
    let shape = transition.shape;
    if shape.affected_sites > 44
        || shape.factors > 88
        || shape.incidences != shape.factors * 9
        || usize::try_from(shape.bytes).ok() != Some(transition.bytes.len())
        || transition.min_mask == 0
    {
        return Err(format!(
            "graph shape invariant failed: shape={shape:?} bytes={} mask={:#04x}",
            transition.bytes.len(),
            transition.min_mask
        ));
    }
    Ok(())
}

fn audit_state_hash(board: &Board, observed: &mut BTreeMap<u64, [u8; 66]>) -> Result<(), String> {
    let key = D4HashState::rebuild(board).canonical_context().key;
    let exact = exact_canonical_state(board).bytes;
    if let Some(previous) = observed.get(&key) {
        if previous != &exact {
            return Err(format!(
                "production D4 state-hash collision for key {key:016X}"
            ));
        }
    } else {
        observed.insert(key, exact);
    }
    Ok(())
}

fn describe_u64(mut values: Vec<u64>) -> Result<Value, String> {
    if values.is_empty() {
        return Err("cannot describe empty value stream".to_string());
    }
    values.sort_unstable();
    let sum = values.iter().try_fold(0u128, |acc, &value| {
        acc.checked_add(u128::from(value))
            .ok_or_else(|| "u64 summary sum overflow".to_string())
    })?;
    Ok(json!({
        "count": values.len(),
        "min": values[0],
        "p05": nearest_rank(&values, 5, 100),
        "p50": nearest_rank(&values, 50, 100),
        "p95": nearest_rank(&values, 95, 100),
        "max": values[values.len() - 1],
        "mean": (sum as f64) / (values.len() as f64),
    }))
}

fn nearest_rank(values: &[u64], numerator: usize, denominator: usize) -> u64 {
    let rank = (numerator * values.len()).div_ceil(denominator);
    values[rank.saturating_sub(1).min(values.len() - 1)]
}

fn downstream_json(label: &str) -> Value {
    let opened = label == "OPEN_GH1_INCREMENTAL_GATE";
    json!({
        "next_stage": if opened {
            "PREREGISTER_GH1_INCREMENTAL_CORRECTNESS_AND_COST"
        } else {
            "STOP_CURRENT_GH1_REPRESENTATION"
        },
        "incremental_gate_opened": opened,
        "graph_model_training_opened": false,
        "dictionary_or_artifact_build_opened": false,
        "runtime_or_search_consumer_opened": false,
        "benchmark_or_arena_opened": false,
        "product_promotion_opened": false,
    })
}

fn parse_args() -> Result<Args, String> {
    let mut values = BTreeMap::<String, String>::new();
    let mut iter = env::args().skip(1);
    while let Some(option) = iter.next() {
        if option == "-h" || option == "--help" {
            return Err(usage().to_string());
        }
        if !matches!(
            option.as_str(),
            "--product-model"
                | "--topk"
                | "--train"
                | "--manifest"
                | "--lineage-model"
                | "--out-report"
        ) {
            return Err(format!(
                "unknown or forbidden option {option:?}\n{}",
                usage()
            ));
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {option}"))?;
        if value.starts_with("--") {
            return Err(format!("missing value for {option}"));
        }
        if values.insert(option.clone(), value).is_some() {
            return Err(format!("duplicate option {option}"));
        }
    }
    let mut take = |name: &str| {
        values
            .remove(name)
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing required {name}\n{}", usage()))
    };
    let inputs = corpus::InputPaths {
        product_model: take("--product-model")?,
        topk: take("--topk")?,
        train: take("--train")?,
        manifest: take("--manifest")?,
        lineage_model: take("--lineage-model")?,
    };
    let out_report = take("--out-report")?;
    if !values.is_empty() {
        return Err(format!("unexpected arguments: {values:?}"));
    }
    Ok(Args { inputs, out_report })
}

fn usage() -> &'static str {
    "usage: cb-gh1-graph-census --product-model MODEL.json --topk topk.bin \
     --train rq615c_k6_train.jsonl --manifest rq615c_final_corpus_manifest.json \
     --lineage-model rq569_model.json --out-report NEW.json"
}

fn refuse_existing(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Err(format!("refusing to overwrite {}", path.display()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_uses_registered_index_rule() {
        let values = [1, 2, 3, 4, 5];
        assert_eq!(nearest_rank(&values, 5, 100), 1);
        assert_eq!(nearest_rank(&values, 50, 100), 3);
        assert_eq!(nearest_rank(&values, 95, 100), 5);
    }

    #[test]
    fn forbidden_cli_surface_is_absent_from_usage() {
        let text = usage().to_ascii_lowercase();
        for forbidden in ["dev", "safety", "holdout", "outcome", "trace"] {
            assert!(!text.contains(forbidden));
        }
    }

    #[test]
    fn compiled_critical_sources_match_the_build_worktree() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        for &(relative, compiled_bytes) in &CRITICAL_SOURCES {
            let disk = std::fs::read(root.join(relative)).unwrap();
            assert_eq!(
                disk, compiled_bytes,
                "compiled source bytes differ for {relative}"
            );
        }
    }
}
