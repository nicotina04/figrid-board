#![cfg(feature = "codebook-eval")]

//! Authoritative CB-QAT1 P0 train-only quantization-headroom census.
//!
//! The only row-bearing input accepted here is the sealed RQ615C train
//! projection. Dev, safety, professional-validation, trace, game, and arena
//! inputs are deliberately absent from the CLI.

use figrid_board::board::Stone;
use figrid_board::codebook_eval::{CodebookWeights, evaluate_full};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[path = "cb_gh1_graph_census/corpus.rs"]
mod corpus;
// The shared provenance module's SHA-256 implementation deliberately routes
// through this already-audited module.
#[allow(dead_code)]
#[path = "cb_gh1_graph_census/graph.rs"]
mod graph;
#[path = "cb_gh1_graph_census/provenance.rs"]
mod provenance;
#[path = "cb_qat1_headroom/stats.rs"]
mod stats;

const FORMAT: &str = "cb-qat1-p0-headroom-v1";
const PREREGISTER_COMMIT: &str = "c08aa68";
const PREREGISTER_DOCUMENT: &str = "experiments/2026-07-26/cb_qat1_integer_lattice_preregister.md";
const EXECUTABLE_STEM: &str = "cb-qat1-headroom";
const CANONICAL_BUILD: &str =
    "cargo build --release --locked --features codebook-eval --bin cb-qat1-headroom";
const REGISTERED_CARGO_LOCK_BYTES: usize = 11_841;
const REGISTERED_CARGO_LOCK_SHA256: &str =
    "3F90AA762C0D7B1F0172C22397588835C79B9C924BB5A931D162B2A5714A202C";
const REGISTERED_RUSTC_RELEASE: &str = "release: 1.88.0";
const REGISTERED_RUSTC_COMMIT: &str = "commit-hash: 6b00bc3880198600130e1cf62b8f8a93494488cc";
const REGISTERED_LLVM: &str = "LLVM version: 20.1.5";
const REGISTERED_CARGO: &str = "cargo 1.88.0 (873a06493 2025-05-10)";

const CRITICAL_SOURCES: &[(&str, &[u8])] = &[
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
        "bin/cb_qat1_headroom.rs",
        include_bytes!("cb_qat1_headroom.rs"),
    ),
    (
        "bin/cb_qat1_headroom/stats.rs",
        include_bytes!("cb_qat1_headroom/stats.rs"),
    ),
    (
        PREREGISTER_DOCUMENT,
        include_bytes!("../experiments/2026-07-26/cb_qat1_integer_lattice_preregister.md"),
    ),
];

#[derive(Clone, Debug)]
struct Args {
    inputs: corpus::InputPaths,
    out_report: PathBuf,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("CB-QAT1 INVALID_QAT1_P0: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args_from(env::args().skip(1))?;
    refuse_existing(&args.out_report)?;
    let output_target = output_target_identity(&args.out_report)?;
    let started_unix_ms = provenance::unix_millis()?;

    let source_before = provenance::source_identity(PREREGISTER_COMMIT, CRITICAL_SOURCES)?;
    let environment_before = provenance::environment_identity(CANONICAL_BUILD)?;
    let forbidden_environment_before = forbidden_environment_identity()?;
    let toolchain_before = provenance::toolchain_identity()?;
    let cargo_before = cargo_identity()?;
    validate_registered_build_contract(&toolchain_before, &cargo_before)?;
    let cpu = provenance::cpu_identity()?;
    let runtime_features_before = runtime_feature_identity()?;
    let executable_before = provenance::executable_identity(EXECUTABLE_STEM)?;

    let corpus::CorpusBundle {
        slates,
        product_float,
        product: _product,
        lineage: _lineage,
        diagnostics: corpus_diagnostics,
    } = corpus::load_validate_and_replay(&args.inputs)?;
    let p0_slates = build_p0_slates(&slates, &product_float)?;
    let analysis = stats::analyze(&p0_slates, 0)?;

    corpus::recheck_inputs(&args.inputs)?;
    let source_after = provenance::source_identity(PREREGISTER_COMMIT, CRITICAL_SOURCES)?;
    if source_after != source_before {
        return Err("critical source identity changed during P0".to_string());
    }
    let environment_after = provenance::environment_identity(CANONICAL_BUILD)?;
    let forbidden_environment_after = forbidden_environment_identity()?;
    if environment_after != environment_before
        || forbidden_environment_after != forbidden_environment_before
    {
        return Err("environment identity changed during P0".to_string());
    }
    let toolchain_after = provenance::toolchain_identity()?;
    let cargo_after = cargo_identity()?;
    if toolchain_after != toolchain_before || cargo_after != cargo_before {
        return Err("toolchain identity changed during P0".to_string());
    }
    let runtime_features_after = runtime_feature_identity()?;
    if runtime_features_after != runtime_features_before {
        return Err("runtime CPU feature identity changed during P0".to_string());
    }
    let executable_after = provenance::executable_identity(EXECUTABLE_STEM)?;
    if executable_after != executable_before {
        return Err("executable identity changed during P0".to_string());
    }

    let finished_unix_ms = provenance::unix_millis()?;
    let elapsed_ms = finished_unix_ms
        .checked_sub(started_unix_ms)
        .ok_or_else(|| "system clock moved backwards during P0".to_string())?;
    let final_label = analysis.final_label;
    let report = json!({
        "format": FORMAT,
        "final_label": final_label,
        "claim_boundary": claim_boundary_json(final_label),
        "preregistration": {
            "commit": PREREGISTER_COMMIT,
            "document": PREREGISTER_DOCUMENT,
        },
        "provenance": {
            "started_unix_ms": started_unix_ms.to_string(),
            "finished_unix_ms": finished_unix_ms.to_string(),
            "elapsed_ms": elapsed_ms.to_string(),
            "source_before": source_before,
            "source_after_equal": true,
            "environment_before": environment_before,
            "environment_after_equal": true,
            "forbidden_environment_before": forbidden_environment_before,
            "forbidden_environment_after_equal": true,
            "rustc_before": toolchain_before,
            "rustc_after_equal": true,
            "cargo_before": cargo_before,
            "cargo_after_equal": true,
            "cpu": cpu,
            "runtime_features_before": runtime_features_before,
            "runtime_features_after_equal": true,
            "executable_before": executable_before,
            "executable_after_equal": true,
            "inputs_rechecked_after_analysis": true,
        },
        "inputs": corpus::input_artifacts_json(&args.inputs),
        "output": {
            "target": output_target,
            "absent_before": true,
            "create_new": true,
            "flush_and_sync_all": true,
        },
        "corpus_a0": corpus_diagnostics,
        "headroom": analysis.report,
        "downstream": downstream_json(final_label),
    });
    write_create_new_synced(&args.out_report, &report)?;
    println!(
        "CB-QAT1 {final_label}: slates={} candidates={} report={}",
        p0_slates.len(),
        p0_slates.len() * corpus::K6,
        args.out_report.display()
    );
    Ok(())
}

fn build_p0_slates(
    slates: &[corpus::Slate],
    product_float: &CodebookWeights,
) -> Result<Vec<stats::P0Slate>, String> {
    let mut output = Vec::with_capacity(slates.len());
    for slate in slates {
        let mut fp32_utilities = [f64::NAN; corpus::K6];
        for (candidate_index, candidate) in slate.candidates.iter().enumerate() {
            if !slate.parent.is_legal_move(candidate.mv) {
                return Err(format!(
                    "{} candidate-{candidate_index} is illegal during FP32 replay",
                    slate.row_uid
                ));
            }
            let mut child = slate.parent.clone();
            child.make_move(candidate.mv);
            if child.side_to_move != slate.root_side.opponent() {
                return Err(format!(
                    "{} candidate-{candidate_index} child-side mismatch",
                    slate.row_uid
                ));
            }
            let natural_child_value = evaluate_full(&child, product_float);
            if !natural_child_value.is_finite() {
                return Err(format!(
                    "{} candidate-{candidate_index} non-finite FP32 value",
                    slate.row_uid
                ));
            }
            fp32_utilities[candidate_index] = -(natural_child_value as f64);
        }
        if slate
            .product_root_utilities
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(format!(
                "{} has a non-finite PTQ root utility",
                slate.row_uid
            ));
        }
        output.push(stats::P0Slate {
            row_uid: slate.row_uid.clone(),
            component_uid: slate.component_uid.clone(),
            root_color: root_color(slate.root_side),
            ordinal: u8::try_from(slate.ordinal)
                .map_err(|_| format!("ordinal does not fit u8: {}", slate.ordinal))?,
            q_teacher: slate.q_teacher,
            teacher_top: std::array::from_fn(|index| slate.candidates[index].teacher_top),
            fp32_utilities,
            // These values were already filled by the shared public-vs-
            // independent PTQ replay in root-mover coordinates. Do not negate.
            ptq_utilities: slate.product_root_utilities,
        });
    }
    Ok(output)
}

fn root_color(side: Stone) -> stats::RootColor {
    match side {
        Stone::Black => stats::RootColor::Black,
        Stone::White => stats::RootColor::White,
    }
}

fn claim_boundary_json(final_label: &str) -> Value {
    json!({
        "rq615c_train_only_consumed_diagnostic": true,
        "fresh_validation_claim": false,
        "quantization_headroom_only": true,
        "training_performed": false,
        "checkpoint_or_scale_selected": false,
        "artifact_built": false,
        "timing_trace_opened": false,
        "game_or_arena_input_opened": false,
        "product_default_changed": false,
        "later_stage_opened": final_label == "GO_PAIRED_QAT_TRAIN",
        "forbidden_row_inputs_opened": 0,
        "forbidden_inputs": [
            "RQ615C dev",
            "RQ615C safety_internal",
            "RQ508 professional validation",
            "frozen 1022-root trace",
            "64-game search logs",
            "game outcomes",
            "Pela artifacts",
            "arena artifacts"
        ],
    })
}

fn downstream_json(final_label: &str) -> Value {
    let opened = final_label == "GO_PAIRED_QAT_TRAIN";
    json!({
        "next_stage": if opened {
            "RUN_ONE_REGISTERED_PAIRED_PTQ_QAT_FIT"
        } else {
            "STOP_CB_QAT1_AND_ADVANCE_TO_CB_AL1"
        },
        "paired_training_opened": opened,
        "validation_opened": false,
        "artifact_gate_opened": false,
        "timing_or_correctness_opened": false,
        "games_opened": false,
        "product_promotion_opened": false,
    })
}

fn parse_args_from<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = BTreeMap::<String, String>::new();
    let mut iter = args.into_iter();
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
    "usage: cb-qat1-headroom --product-model MODEL.json --topk topk.bin \
     --train rq615c_k6_train.jsonl --manifest rq615c_final_corpus_manifest.json \
     --lineage-model rq569_model.json --out-report NEW.json"
}

fn refuse_existing(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Err(format!("refusing to overwrite {}", path.display()));
    }
    Ok(())
}

fn output_target_identity(path: &Path) -> Result<Value, String> {
    let leaf = path
        .file_name()
        .ok_or_else(|| format!("output path has no file name: {}", path.display()))?;
    let parent = path.parent().filter(|value| !value.as_os_str().is_empty());
    let parent = match parent {
        Some(value) => value.to_path_buf(),
        None => env::current_dir().map_err(|error| format!("current_dir failed: {error}"))?,
    };
    let canonical_parent = parent.canonicalize().map_err(|error| {
        format!(
            "failed to canonicalize output parent {}: {error}",
            parent.display()
        )
    })?;
    let resolved = canonical_parent.join(leaf);
    if resolved.exists() {
        return Err(format!(
            "resolved output target already exists: {}",
            resolved.display()
        ));
    }
    Ok(json!({
        "requested": path,
        "canonical_parent": canonical_parent,
        "resolved_nonexistent_target": resolved,
    }))
}

fn write_create_new_synced(path: &Path, report: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to serialize P0 report: {error}"))?;
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|error| format!("failed to create report {}: {error}", path.display()))?;
    if let Err(error) = output
        .write_all(&bytes)
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.flush())
        .and_then(|_| output.sync_all())
    {
        drop(output);
        let cleanup = fs::remove_file(path)
            .map(|_| "partial report removed".to_string())
            .unwrap_or_else(|cleanup_error| {
                format!("partial report cleanup also failed: {cleanup_error}")
            });
        return Err(format!(
            "failed to write and durably sync report {}: {error}; {cleanup}",
            path.display()
        ));
    }
    Ok(())
}

fn cargo_identity() -> Result<Value, String> {
    let output = std::process::Command::new("cargo")
        .arg("-V")
        .output()
        .map_err(|error| format!("failed to invoke cargo -V: {error}"))?;
    if !output.status.success() {
        return Err(format!("cargo -V failed with {}", output.status));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("cargo -V emitted non-UTF-8 output: {error}"))?;
    Ok(json!({"cargo_v": stdout}))
}

fn validate_registered_build_contract(rustc: &Value, cargo: &Value) -> Result<(), String> {
    const LOCK: &[u8] = include_bytes!("../Cargo.lock");
    if LOCK.len() != REGISTERED_CARGO_LOCK_BYTES {
        return Err(format!(
            "registered Cargo.lock byte mismatch: got {}, expected {REGISTERED_CARGO_LOCK_BYTES}",
            LOCK.len()
        ));
    }
    let lock_sha256 = provenance::sha256_hex(LOCK);
    if lock_sha256 != REGISTERED_CARGO_LOCK_SHA256 {
        return Err(format!(
            "registered Cargo.lock SHA-256 mismatch: got {lock_sha256}, expected \
             {REGISTERED_CARGO_LOCK_SHA256}"
        ));
    }
    let rustc_vv = rustc
        .get("rustc_vv")
        .and_then(Value::as_str)
        .ok_or_else(|| "rustc provenance lacks rustc_vv".to_string())?;
    for required in [
        REGISTERED_RUSTC_RELEASE,
        REGISTERED_RUSTC_COMMIT,
        REGISTERED_LLVM,
    ] {
        if !rustc_vv.lines().any(|line| line == required) {
            return Err(format!("registered rustc contract missing {required:?}"));
        }
    }
    let cargo_v = cargo
        .get("cargo_v")
        .and_then(Value::as_str)
        .ok_or_else(|| "cargo provenance lacks cargo_v".to_string())?
        .trim();
    if cargo_v != REGISTERED_CARGO {
        return Err(format!(
            "registered cargo contract mismatch: got {cargo_v:?}, expected {REGISTERED_CARGO:?}"
        ));
    }
    Ok(())
}

fn forbidden_environment_identity() -> Result<Value, String> {
    let mut forbidden = env::vars_os()
        .filter_map(|(name, _)| {
            let rendered = name.to_string_lossy().into_owned();
            let upper = rendered.to_ascii_uppercase();
            (upper.starts_with("FIGRID_")
                || upper.starts_with("RAYON_")
                || upper.starts_with("CARGO_PROFILE_")
                || matches!(
                    upper.as_str(),
                    "LLVM_PROFILE_FILE"
                        | "RUSTC_WRAPPER"
                        | "RUSTC_WORKSPACE_WRAPPER"
                        | "RUSTDOCFLAGS"
                        | "CARGO_ENCODED_RUSTFLAGS"
                ))
            .then_some(rendered)
        })
        .collect::<Vec<_>>();
    forbidden.sort();
    if !forbidden.is_empty() {
        return Err(format!(
            "FIGRID_*, RAYON_*, or Rust instrumentation overrides are forbidden: {forbidden:?}"
        ));
    }
    Ok(json!({"forbidden_environment_variables": forbidden}))
}

#[cfg(target_arch = "x86_64")]
fn runtime_feature_identity() -> Result<Value, String> {
    let avx2 = std::arch::is_x86_feature_detected!("avx2");
    let bmi2 = std::arch::is_x86_feature_detected!("bmi2");
    let fma = std::arch::is_x86_feature_detected!("fma");
    if !avx2 || !bmi2 {
        return Err(format!(
            "runtime CPU lacks registered required features: avx2={avx2} bmi2={bmi2}"
        ));
    }
    Ok(json!({"avx2": avx2, "bmi2": bmi2, "fma": fma}))
}

#[cfg(not(target_arch = "x86_64"))]
fn runtime_feature_identity() -> Result<Value, String> {
    Err(format!(
        "CB-QAT1 P0 requires x86_64, observed {}",
        env::consts::ARCH
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use figrid_board::board::{BOARD_SIZE, Board};

    fn valid_cli() -> Vec<String> {
        [
            "--product-model",
            "product.json",
            "--topk",
            "topk.bin",
            "--train",
            "train.jsonl",
            "--manifest",
            "manifest.json",
            "--lineage-model",
            "lineage.json",
            "--out-report",
            "new.json",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    }

    #[test]
    fn cli_accepts_exact_six_registered_paths() {
        let parsed = parse_args_from(valid_cli()).unwrap();
        assert_eq!(parsed.inputs.product_model, PathBuf::from("product.json"));
        assert_eq!(parsed.out_report, PathBuf::from("new.json"));
    }

    #[test]
    fn cli_rejects_forbidden_or_duplicate_surface() {
        let mut forbidden = valid_cli();
        forbidden.extend(["--dev".to_string(), "dev.jsonl".to_string()]);
        assert!(parse_args_from(forbidden).is_err());

        let mut duplicate = valid_cli();
        duplicate.extend(["--train".to_string(), "replacement.jsonl".to_string()]);
        assert!(parse_args_from(duplicate).is_err());
    }

    #[test]
    fn usage_has_no_later_stage_input_surface() {
        let text = usage().to_ascii_lowercase();
        for forbidden in [
            "dev",
            "safety",
            "professional",
            "holdout",
            "trace",
            "outcome",
            "arena",
            "pela",
        ] {
            assert!(!text.contains(forbidden));
        }
    }

    #[test]
    fn fp32_root_utility_is_negative_natural_child_value() {
        let parent = Board::new();
        let mv = 7 * BOARD_SIZE + 7;
        assert!(parent.is_legal_move(mv));
        let root_side = parent.side_to_move;
        let weights = CodebookWeights::deterministic(16, 8);
        let mut child = parent.clone();
        child.make_move(mv);
        assert_eq!(child.side_to_move, root_side.opponent());
        let natural = evaluate_full(&child, &weights);
        let root_utility = -(natural as f64);
        assert_eq!(root_utility.to_bits(), (-(natural as f64)).to_bits());
        assert_ne!(root_utility.to_bits(), (natural as f64).to_bits());
    }

    #[test]
    fn provenance_boundary_seals_preregister_and_both_new_sources() {
        assert_eq!(PREREGISTER_COMMIT, "c08aa68");
        let paths = CRITICAL_SOURCES
            .iter()
            .map(|(path, _)| *path)
            .collect::<Vec<_>>();
        for required in [
            PREREGISTER_DOCUMENT,
            "bin/cb_qat1_headroom.rs",
            "bin/cb_qat1_headroom/stats.rs",
            "bin/cb_gh1_graph_census/corpus.rs",
            "bin/cb_gh1_graph_census/provenance.rs",
        ] {
            assert!(
                paths.contains(&required),
                "missing critical source {required}"
            );
        }
    }

    #[test]
    fn registered_lock_and_toolchain_contract_is_exact() {
        let rustc = json!({
            "rustc_vv": format!(
                "rustc 1.88.0\n{REGISTERED_RUSTC_COMMIT}\n{REGISTERED_RUSTC_RELEASE}\n\
                 {REGISTERED_LLVM}\n"
            )
        });
        let cargo = json!({"cargo_v": format!("{REGISTERED_CARGO}\n")});
        validate_registered_build_contract(&rustc, &cargo).unwrap();

        let wrong_cargo = json!({"cargo_v": "cargo 1.88.1 (wrong)"});
        assert!(validate_registered_build_contract(&rustc, &wrong_cargo).is_err());
    }

    #[test]
    fn compiled_critical_sources_match_the_build_worktree() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        for &(relative, compiled_bytes) in CRITICAL_SOURCES {
            let disk = std::fs::read(root.join(relative)).unwrap();
            assert_eq!(
                disk, compiled_bytes,
                "compiled source bytes differ for {relative}"
            );
        }
    }
}
