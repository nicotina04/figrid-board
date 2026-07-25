#[path = "cb_gh1_graph_census/corpus.rs"]
mod corpus;
#[path = "cb_gh1_graph_census/graph.rs"]
mod graph;
#[path = "cb_al1_selector/hash.rs"]
mod hash;
#[path = "cb_al1_selector/prepared.rs"]
mod prepared;
#[path = "cb_gh1_graph_census/provenance.rs"]
mod provenance;
#[path = "cb_al1_selector/reveal.rs"]
mod reveal;
#[path = "cb_al1_selector/stats.rs"]
mod stats;

use figrid_board::board::{Move, Stone};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const PREREGISTER_COMMIT: &str = "5c63f04";
const EXECUTABLE_STEM: &str = "cb-al1-selector";
const CANONICAL_RUSTFLAGS: &str = "-C target-cpu=x86-64-v3";
const CANONICAL_BUILD: &str =
    "cargo build --release --locked --features cb-al1-audit --bin cb-al1-selector";
const REGISTERED_WORKING_DIRECTORY: &str =
    r"C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2";
const PREPARED_UNITS_PATH: &str = r"C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-11\rq615c_prepared_units_1000.jsonl";
const PHASE2_MANIFEST_PATH: &str = r"C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-11\rq615c_phase2_prepared_manifest.json";
const PRODUCT_MODEL_PATH: &str = r"C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2\models\gomoku_codebook_v1_swapclosed.json";
const PRODUCT_CBF_PATH: &str = r"C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2\models\gomoku_codebook_v1_swapclosed_factored.cbf";
const TOPK_PATH: &str = r"C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2\data\topk.bin";
const P0A_OUTPUT_PATH: &str = r"C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2-artifacts\2026-07-26\cb-al1-p0a\cb_al1_p0a_selector.json";
const TRAIN_PATH: &str = r"C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-11\rq615c_k6_train.jsonl";
const FINAL_MANIFEST_PATH: &str = r"C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-11\rq615c_final_corpus_manifest.json";
const LINEAGE_MODEL_PATH: &str = r"C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-08\rq569_codebook_full_matefirst_ep3_model_swapclosed.json";
const P0B_OUTPUT_PATH: &str = r"C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2-artifacts\2026-07-26\cb-al1-p0b\cb_al1_p0b_reveal.json";
const COMPILE_TIME_RUSTFLAGS: Option<&str> = option_env!("RUSTFLAGS");
const COMPILE_TIME_FORBIDDEN: &[(&str, Option<&str>)] = &[
    ("LLVM_PROFILE_FILE", option_env!("LLVM_PROFILE_FILE")),
    ("GCOV_PREFIX", option_env!("GCOV_PREFIX")),
    ("GCOV_PREFIX_STRIP", option_env!("GCOV_PREFIX_STRIP")),
    ("RUSTC_WRAPPER", option_env!("RUSTC_WRAPPER")),
    (
        "RUSTC_WORKSPACE_WRAPPER",
        option_env!("RUSTC_WORKSPACE_WRAPPER"),
    ),
    ("RUSTDOCFLAGS", option_env!("RUSTDOCFLAGS")),
    (
        "CARGO_ENCODED_RUSTFLAGS",
        option_env!("CARGO_ENCODED_RUSTFLAGS"),
    ),
    ("RUSTC_BOOTSTRAP", option_env!("RUSTC_BOOTSTRAP")),
    ("CARGO_INCREMENTAL", option_env!("CARGO_INCREMENTAL")),
    ("RAYON_NUM_THREADS", option_env!("RAYON_NUM_THREADS")),
    ("RAYON_STACK_SIZE", option_env!("RAYON_STACK_SIZE")),
    (
        "CARGO_PROFILE_RELEASE_OPT_LEVEL",
        option_env!("CARGO_PROFILE_RELEASE_OPT_LEVEL"),
    ),
    (
        "CARGO_PROFILE_RELEASE_LTO",
        option_env!("CARGO_PROFILE_RELEASE_LTO"),
    ),
    (
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
        option_env!("CARGO_PROFILE_RELEASE_CODEGEN_UNITS"),
    ),
    (
        "CARGO_PROFILE_RELEASE_DEBUG",
        option_env!("CARGO_PROFILE_RELEASE_DEBUG"),
    ),
    (
        "CARGO_PROFILE_RELEASE_STRIP",
        option_env!("CARGO_PROFILE_RELEASE_STRIP"),
    ),
    (
        "CARGO_PROFILE_RELEASE_PANIC",
        option_env!("CARGO_PROFILE_RELEASE_PANIC"),
    ),
];

const PREREGISTER_DOCUMENT: &str =
    "experiments/2026-07-26/cb_al1_active_distillation_preregister.md";
const FROZEN_PREREG_BYTES: usize = 33_675;
const FROZEN_PREREG_SHA256: &str =
    "DFA3151CE7CBEE0483964264CFE74DDF13DE906E046EF45770BC1F88C121FD6B";
const FROZEN_CARGO_LOCK_BYTES: usize = 11_841;
const FROZEN_CARGO_LOCK_SHA256: &str =
    "3F90AA762C0D7B1F0172C22397588835C79B9C924BB5A931D162B2A5714A202C";

const CRITICAL_SOURCES: &[(&str, &[u8])] = &[
    ("Cargo.toml", include_bytes!("../Cargo.toml")),
    ("Cargo.lock", include_bytes!("../Cargo.lock")),
    ("src/lib.rs", include_bytes!("../src/lib.rs")),
    ("src/board.rs", include_bytes!("../src/board.rs")),
    (
        "src/codebook_eval.rs",
        include_bytes!("../src/codebook_eval.rs"),
    ),
    (
        "src/factored_codebook.rs",
        include_bytes!("../src/factored_codebook.rs"),
    ),
    (
        "src/pattern_table.rs",
        include_bytes!("../src/pattern_table.rs"),
    ),
    ("src/d4_hash.rs", include_bytes!("../src/d4_hash.rs")),
    ("src/search.rs", include_bytes!("../src/search.rs")),
    (
        "src/token_delta.rs",
        include_bytes!("../src/token_delta.rs"),
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
        "bin/cb_gh1_graph_census/corpus.rs",
        include_bytes!("cb_gh1_graph_census/corpus.rs"),
    ),
    (
        "bin/cb_al1_selector.rs",
        include_bytes!("cb_al1_selector.rs"),
    ),
    (
        "bin/cb_al1_selector/hash.rs",
        include_bytes!("cb_al1_selector/hash.rs"),
    ),
    (
        "bin/cb_al1_selector/prepared.rs",
        include_bytes!("cb_al1_selector/prepared.rs"),
    ),
    (
        "bin/cb_al1_selector/reveal.rs",
        include_bytes!("cb_al1_selector/reveal.rs"),
    ),
    (
        "bin/cb_al1_selector/stats.rs",
        include_bytes!("cb_al1_selector/stats.rs"),
    ),
    (
        "bin/cb_al1_selector/tests.rs",
        include_bytes!("cb_al1_selector/tests.rs"),
    ),
    (
        PREREGISTER_DOCUMENT,
        include_bytes!("../experiments/2026-07-26/cb_al1_active_distillation_preregister.md"),
    ),
    ("data/topk.bin", include_bytes!("../data/topk.bin")),
];

#[derive(Clone, Debug)]
struct SharedPaths {
    prepared_units: PathBuf,
    phase2_manifest: PathBuf,
    product_model: PathBuf,
    product_cbf: PathBuf,
    topk: PathBuf,
}

#[derive(Clone, Debug)]
struct P0aArgs {
    shared: SharedPaths,
    out_selector: PathBuf,
}

#[derive(Clone, Debug)]
struct P0bArgs {
    selector: PathBuf,
    expected_p0a_bytes: u64,
    expected_p0a_sha256: String,
    shared: SharedPaths,
    train: PathBuf,
    final_manifest: PathBuf,
    lineage_model: PathBuf,
    out_reveal: PathBuf,
}

#[derive(Clone, Debug)]
enum StageArgs {
    P0a(P0aArgs),
    P0b(P0bArgs),
}

fn main() {
    if let Err(error) = run() {
        eprintln!("CB-AL1 INVALID_CB_AL1_P0: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let arguments = env::args_os().skip(1).collect::<Vec<_>>();
    validate_registered_arguments(&arguments)?;
    match parse_args_from(arguments.into_iter())? {
        StageArgs::P0a(args) => run_p0a(args),
        StageArgs::P0b(args) => run_p0b(args),
    }
}

fn run_p0a(args: P0aArgs) -> Result<(), String> {
    let output_target = output_target_identity(&args.out_selector)?;
    let started_unix_ms = provenance::unix_millis()?;
    let provenance_before = stage_provenance("p0a")?;
    let paths = prepared_paths(&args.shared);
    let analysis = prepared::analyze_p0a(&paths)?;
    let selector_analysis = analysis.to_json();
    let rechecked = prepared::recheck_p0a_inputs(&paths)?;
    if rechecked != analysis.input_seals {
        return Err("P0A input seal changed after selector analysis".to_string());
    }
    let provenance_after = stage_provenance("p0a")?;
    if provenance_before != provenance_after {
        return Err("P0A provenance changed during analysis".to_string());
    }
    let finished_unix_ms = provenance::unix_millis()?;
    let status = analysis.status.name();
    let implementation = implementation_identity(&provenance_after)?;
    let report = json!({
        "format": "cb-al1-p0a-selector-v1",
        "status": status,
        "preregister_commit": PREREGISTER_COMMIT,
        "implementation_identity": implementation,
        "stage_timing": {
            "started_unix_ms": started_unix_ms.to_string(),
            "finished_unix_ms": finished_unix_ms.to_string(),
        },
        "output_target": output_target,
        "selector_analysis": selector_analysis,
        "shared_input_seals": {
            "before": prepared_seals_json(&analysis.input_seals),
            "after": prepared_seals_json(&rechecked),
        },
        "provenance": {
            "before": provenance_before,
            "after": provenance_after,
        },
        "label_inputs_opened": false,
        "training_or_product_mutation": false,
    });
    let seal = write_new_json(&args.out_selector, &report)?;
    println!(
        "CB-AL1 P0A {status} bytes={} sha256={}",
        seal.bytes, seal.sha256
    );
    Ok(())
}

fn run_p0b(args: P0bArgs) -> Result<(), String> {
    let output_target = output_target_identity(&args.out_reveal)?;
    let selector_seal_before = provenance::seal_file(&args.selector)?;
    if selector_seal_before.bytes != args.expected_p0a_bytes
        || selector_seal_before.sha256 != args.expected_p0a_sha256
    {
        return Err(format!(
            "P0A literal seal mismatch: observed bytes={} sha256={}",
            selector_seal_before.bytes, selector_seal_before.sha256
        ));
    }
    let selector_bytes = fs::read(&args.selector).map_err(|error| {
        format!(
            "failed to read selector {}: {error}",
            args.selector.display()
        )
    })?;
    let selector_report: Value = serde_json::from_slice(&selector_bytes)
        .map_err(|error| format!("P0A selector JSON is invalid: {error}"))?;
    let mut canonical_selector = serde_json::to_vec_pretty(&selector_report)
        .map_err(|error| format!("failed to canonicalize P0A selector JSON: {error}"))?;
    canonical_selector.push(b'\n');
    if canonical_selector != selector_bytes {
        return Err(
            "P0A selector bytes are not canonical pretty JSON plus one terminal LF".to_string(),
        );
    }
    require_selector_report_contract(&selector_report, &args.selector)?;

    let started_unix_ms = provenance::unix_millis()?;
    let provenance_before = stage_provenance("p0b")?;
    let current_implementation = implementation_identity(&provenance_before)?;
    if selector_report.get("implementation_identity") != Some(&current_implementation) {
        return Err("P0A/P0B implementation identity mismatch".to_string());
    }

    let paths = prepared_paths(&args.shared);
    let prepared_analysis = prepared::analyze_p0a(&paths)?;
    if prepared_analysis.status != prepared::SelectorSupportStatus::ReadyForReveal {
        return Err("P0A recomputation is not READY_FOR_REVEAL".to_string());
    }
    let recomputed_selector = prepared_analysis.to_json();
    if selector_report.get("selector_analysis") != Some(&recomputed_selector) {
        return Err(
            "P0A selector analysis is not byte-semantic identical on P0B replay".to_string(),
        );
    }
    let shared_recheck = prepared::recheck_p0a_inputs(&paths)?;
    if shared_recheck != prepared_analysis.input_seals {
        return Err("shared P0A inputs changed before label reveal".to_string());
    }
    let selector_seal_gate = provenance::seal_file(&args.selector)?;
    if selector_seal_gate != selector_seal_before {
        return Err("P0A selector changed before label reveal".to_string());
    }

    // The label-bearing files are first opened below, after every P0A seal,
    // selector replay, implementation, and output-target gate has passed.
    let label_seals_before = p0b_label_seals(&args)?;
    let corpus_paths = corpus::InputPaths {
        product_model: args.shared.product_model.clone(),
        topk: args.shared.topk.clone(),
        train: args.train.clone(),
        manifest: args.final_manifest.clone(),
        lineage_model: args.lineage_model.clone(),
    };
    let label_index = reveal::load_label_index(&corpus_paths)?;
    let (stat_units, join_diagnostics) = build_stat_units(&prepared_analysis, &label_index)?;
    let stats::AnalysisOutcome {
        final_label,
        report: statistics,
    } = stats::analyze(&stat_units)?;

    let label_seals_after = p0b_label_seals(&args)?;
    if label_seals_after != label_seals_before {
        return Err("P0B label input changed during reveal".to_string());
    }
    corpus::recheck_inputs(&corpus_paths)?;
    let final_shared = prepared::recheck_p0a_inputs(&paths)?;
    if final_shared != shared_recheck {
        return Err("shared P0A input changed during P0B".to_string());
    }
    let selector_seal_after = provenance::seal_file(&args.selector)?;
    if selector_seal_after != selector_seal_before {
        return Err("P0A selector changed during P0B".to_string());
    }
    let provenance_after = stage_provenance("p0b")?;
    if provenance_before != provenance_after {
        return Err("P0B provenance changed during reveal".to_string());
    }
    let finished_unix_ms = provenance::unix_millis()?;
    let output = json!({
        "format": "cb-al1-p0b-reveal-v1",
        "status": final_label,
        "preregister_commit": PREREGISTER_COMMIT,
        "implementation_identity": current_implementation,
        "stage_timing": {
            "started_unix_ms": started_unix_ms.to_string(),
            "finished_unix_ms": finished_unix_ms.to_string(),
        },
        "output_target": output_target,
        "p0a_selector_seals": {
            "before": selector_seal_before.json(),
            "pre_label_gate": selector_seal_gate.json(),
            "after": selector_seal_after.json(),
        },
        "p0a_selector_status": "P0A_READY_FOR_REVEAL",
        "shared_answer_opaque_analysis": recomputed_selector,
        "shared_input_seals": {
            "before": prepared_seals_json(&shared_recheck),
            "after": prepared_seals_json(&final_shared),
        },
        "label_input_seals": {
            "before": label_seals_json(&label_seals_before),
            "after": label_seals_json(&label_seals_after),
        },
        "label_corpus_diagnostics": label_index.diagnostics,
        "join_diagnostics": join_diagnostics,
        "statistics": statistics,
        "provenance": {
            "before": provenance_before,
            "after": provenance_after,
        },
        "new_teacher_queries": 0,
        "training_or_product_mutation": false,
    });
    let seal = write_new_json(&args.out_reveal, &output)?;
    println!(
        "CB-AL1 P0B {final_label} bytes={} sha256={}",
        seal.bytes, seal.sha256
    );
    Ok(())
}

fn prepared_paths(shared: &SharedPaths) -> prepared::PreparedPaths {
    prepared::PreparedPaths {
        prepared_units: shared.prepared_units.clone(),
        phase2_manifest: shared.phase2_manifest.clone(),
        product_model: shared.product_model.clone(),
        product_cbf: shared.product_cbf.clone(),
        topk: shared.topk.clone(),
    }
}

fn implementation_identity(provenance: &Value) -> Result<Value, String> {
    require_exact_object_keys(
        provenance,
        &[
            "stage",
            "preregister_commit",
            "source",
            "critical_source_stream",
            "environment",
            "toolchain",
            "cargo",
            "cpu",
            "executable",
            "working_directory",
            "argv",
            "target_profile",
            "debug_assertions",
            "enabled_features",
            "canonical_build",
        ],
        "stage provenance",
    )?;
    let get = |key: &str| {
        provenance
            .get(key)
            .cloned()
            .ok_or_else(|| format!("stage provenance omitted {key}"))
    };
    Ok(json!({
        "preregister_commit": get("preregister_commit")?,
        "source": get("source")?,
        "critical_source_stream": get("critical_source_stream")?,
        "environment": get("environment")?,
        "toolchain": get("toolchain")?,
        "cargo": get("cargo")?,
        "cpu": get("cpu")?,
        "executable": get("executable")?,
        "working_directory": get("working_directory")?,
        "target_profile": get("target_profile")?,
        "debug_assertions": get("debug_assertions")?,
        "enabled_features": get("enabled_features")?,
        "canonical_build": get("canonical_build")?,
    }))
}

fn require_selector_report_contract(report: &Value, selector_path: &Path) -> Result<(), String> {
    require_exact_object_keys(
        report,
        &[
            "format",
            "status",
            "preregister_commit",
            "implementation_identity",
            "stage_timing",
            "output_target",
            "selector_analysis",
            "shared_input_seals",
            "provenance",
            "label_inputs_opened",
            "training_or_product_mutation",
        ],
        "P0A selector report",
    )?;
    require_json_string(report, "format", "cb-al1-p0a-selector-v1")?;
    require_json_string(report, "status", "P0A_READY_FOR_REVEAL")?;
    require_json_string(report, "preregister_commit", PREREGISTER_COMMIT)?;
    require_json_bool(report, "label_inputs_opened", false)?;
    require_json_bool(report, "training_or_product_mutation", false)?;

    let selector = report
        .get("selector_analysis")
        .ok_or_else(|| "P0A selector report omitted selector_analysis".to_string())?;
    require_json_string(selector, "status", "P0A_READY_FOR_REVEAL")?;
    let shared_seals = report
        .get("shared_input_seals")
        .ok_or_else(|| "P0A selector report omitted shared_input_seals".to_string())?;
    require_exact_object_keys(shared_seals, &["before", "after"], "P0A shared input seals")?;
    let shared_before = shared_seals
        .get("before")
        .ok_or_else(|| "P0A shared input seals omitted before".to_string())?;
    let shared_after = shared_seals
        .get("after")
        .ok_or_else(|| "P0A shared input seals omitted after".to_string())?;
    if shared_before != shared_after || selector.get("input_artifacts") != Some(shared_before) {
        return Err("P0A shared input pre/post/analysis seals differ".to_string());
    }

    let provenance = report
        .get("provenance")
        .ok_or_else(|| "P0A selector report omitted provenance".to_string())?;
    require_exact_object_keys(provenance, &["before", "after"], "P0A provenance")?;
    let before = provenance
        .get("before")
        .ok_or_else(|| "P0A provenance omitted before".to_string())?;
    let after = provenance
        .get("after")
        .ok_or_else(|| "P0A provenance omitted after".to_string())?;
    if before != after {
        return Err("P0A before/after provenance differs".to_string());
    }
    require_json_string(before, "stage", "p0a")?;
    validate_recorded_p0a_argv(before)?;
    let recorded_implementation = report
        .get("implementation_identity")
        .ok_or_else(|| "P0A selector report omitted implementation_identity".to_string())?;
    if recorded_implementation != &implementation_identity(before)? {
        return Err("P0A implementation identity does not match its provenance".to_string());
    }

    let timing = report
        .get("stage_timing")
        .ok_or_else(|| "P0A selector report omitted stage_timing".to_string())?;
    require_exact_object_keys(
        timing,
        &["started_unix_ms", "finished_unix_ms"],
        "P0A stage timing",
    )?;
    let started = parse_decimal_json_string(timing, "started_unix_ms")?;
    let finished = parse_decimal_json_string(timing, "finished_unix_ms")?;
    if finished < started {
        return Err("P0A finish time precedes start time".to_string());
    }
    let output_target = report
        .get("output_target")
        .ok_or_else(|| "P0A selector report omitted output_target".to_string())?;
    require_exact_object_keys(output_target, &["path"], "P0A output target")?;
    let recorded_path = output_target
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| "P0A output target path must be a string".to_string())?;
    let observed_path = selector_path
        .canonicalize()
        .map_err(|error| format!("failed to canonicalize P0A selector path: {error}"))?;
    if recorded_path != observed_path.display().to_string() {
        return Err(format!(
            "P0A selector path mismatch: recorded={recorded_path:?} observed={:?}",
            observed_path.display().to_string()
        ));
    }
    Ok(())
}

fn p0b_label_seals(args: &P0bArgs) -> Result<BTreeMap<&'static str, provenance::FileSeal>, String> {
    let mut seals = BTreeMap::new();
    seals.insert("train", provenance::seal_file(&args.train)?);
    seals.insert(
        "final_manifest",
        provenance::seal_file(&args.final_manifest)?,
    );
    seals.insert("lineage_model", provenance::seal_file(&args.lineage_model)?);
    Ok(seals)
}

fn label_seals_json(seals: &BTreeMap<&'static str, provenance::FileSeal>) -> Value {
    Value::Object(
        seals
            .iter()
            .map(|(name, seal)| ((*name).to_string(), seal.json()))
            .collect(),
    )
}

fn prepared_seals_json(seals: &BTreeMap<&'static str, hash::FileSeal>) -> Value {
    Value::Object(
        seals
            .iter()
            .map(|(name, seal)| {
                (
                    (*name).to_string(),
                    json!({"bytes": seal.bytes, "sha256": seal.sha256}),
                )
            })
            .collect(),
    )
}

fn build_stat_units(
    analysis: &prepared::PreparedAnalysis,
    labels: &reveal::LabelIndex,
) -> Result<(Vec<stats::Unit>, Value), String> {
    const ORDINALS: [usize; 5] = [1, 2, 4, 6, 8];
    const SUPPORT_PER_ORDINAL: usize = 100;
    const ARM_PER_ORDINAL: usize = 25;

    let active = analysis
        .selector
        .active
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let control = analysis
        .selector
        .control
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if active.len() != ORDINALS.len() * ARM_PER_ORDINAL
        || control.len() != ORDINALS.len() * ARM_PER_ORDINAL
    {
        return Err(format!(
            "selector arm cardinality mismatch: active={} control={}",
            active.len(),
            control.len()
        ));
    }

    let mut output = Vec::with_capacity(ORDINALS.len() * SUPPORT_PER_ORDINAL);
    let mut seen_support = BTreeSet::new();
    let mut joined_keys = BTreeSet::new();
    let mut matched_units = 0usize;
    let mut matched_components = BTreeSet::new();
    let mut measurable = [[0usize; 2]; 2];
    let mut teacher_tied = [[0usize; 2]; 2];

    for ordinal in ORDINALS {
        let support = analysis
            .selector
            .support_by_ordinal
            .get(&ordinal)
            .ok_or_else(|| format!("selector omitted support ordinal {ordinal}"))?;
        if support.len() != SUPPORT_PER_ORDINAL {
            return Err(format!(
                "support ordinal {ordinal} has {}, expected {SUPPORT_PER_ORDINAL}",
                support.len()
            ));
        }
        for (rank, uid) in support.iter().enumerate() {
            if !seen_support.insert(uid.as_str()) {
                return Err(format!("duplicate support UID {uid}"));
            }
            let prepared_unit = analysis
                .unit_by_uid(uid)
                .ok_or_else(|| format!("support UID {uid} is absent from prepared units"))?;
            if prepared_unit.figrid_ordinal != ordinal
                || prepared_unit.split != prepared::Split::Train
                || !prepared_unit.quiet_eligible()
            {
                return Err(format!(
                    "support UID {uid} violates ordinal/train/quiet contract"
                ));
            }

            let key = (prepared_unit.opening_group_hash.clone(), ordinal);
            let mut observations = [[stats::Observation::unmeasurable(); 2]; 2];
            let matched_component_uid = if let Some(label_unit) = labels.units.get(&key) {
                if !joined_keys.insert(key.clone()) {
                    return Err(format!("label unit key was reused: {key:?}"));
                }
                validate_joined_unit(prepared_unit, label_unit)?;
                for (color, prepared_parent, label_parent) in [
                    (stats::Color::Black, &prepared_unit.black, &label_unit.black),
                    (stats::Color::White, &prepared_unit.white, &label_unit.white),
                ] {
                    let static_observation = observation_for_move(
                        label_parent,
                        prepared_parent.diagnostics.static_top.mv,
                        false,
                    )?;
                    let actual_observation = observation_for_move(
                        label_parent,
                        prepared_parent.figrid_actual_move,
                        true,
                    )?;
                    observations[stats::ChoiceFamily::StaticTop.index()][color.index()] =
                        static_observation;
                    observations[stats::ChoiceFamily::ArchivedActual.index()][color.index()] =
                        actual_observation;
                }
                matched_units += 1;
                matched_components.insert(label_unit.component_uid.clone());
                Some(label_unit.component_uid.clone())
            } else {
                None
            };

            for choice in stats::ChoiceFamily::ALL {
                for color in stats::Color::ALL {
                    let observation = observations[choice.index()][color.index()];
                    measurable[choice.index()][color.index()] +=
                        usize::from(observation.measurable);
                    teacher_tied[choice.index()][color.index()] +=
                        usize::from(observation.teacher_max_tied);
                }
            }
            output.push(stats::Unit {
                uid: uid.clone(),
                ordinal: u8::try_from(ordinal)
                    .map_err(|_| format!("ordinal does not fit u8: {ordinal}"))?,
                support_rank: u8::try_from(rank)
                    .map_err(|_| format!("support rank does not fit u8: {rank}"))?,
                opening_group_hash: prepared_unit.opening_group_hash.clone(),
                parent_d4_side_hashes: [
                    prepared_unit.black.parent_d4_side_hash.clone(),
                    prepared_unit.white.parent_d4_side_hash.clone(),
                ],
                legal_child_d4_side_hashes: [
                    prepared_unit
                        .black
                        .legal_inventory
                        .iter()
                        .map(|entry| entry.child_d4_side_hash.clone())
                        .collect(),
                    prepared_unit
                        .white
                        .legal_inventory
                        .iter()
                        .map(|entry| entry.child_d4_side_hash.clone())
                        .collect(),
                ],
                matched_component_uid,
                complete_pair: labels.units.contains_key(&key),
                active: active.contains(uid.as_str()),
                deterministic_control: control.contains(uid.as_str()),
                observations,
            });
        }
    }
    if output.len() != 500 || seen_support.len() != 500 {
        return Err(format!(
            "joined support cardinality mismatch: output={} unique={}",
            output.len(),
            seen_support.len()
        ));
    }
    let family_color_json = |counts: [[usize; 2]; 2]| {
        json!({
            "static_top": {
                "black": counts[stats::ChoiceFamily::StaticTop.index()][stats::Color::Black.index()],
                "white": counts[stats::ChoiceFamily::StaticTop.index()][stats::Color::White.index()],
            },
            "archived_actual": {
                "black": counts[stats::ChoiceFamily::ArchivedActual.index()][stats::Color::Black.index()],
                "white": counts[stats::ChoiceFamily::ArchivedActual.index()][stats::Color::White.index()],
            },
        })
    };
    Ok((
        output,
        json!({
            "support_units": seen_support.len(),
            "matched_complete_pair_units": matched_units,
            "missing_train_units": seen_support.len() - matched_units,
            "joined_label_unit_keys_unique": joined_keys.len(),
            "matched_component_uids_unique": matched_components.len(),
            "label_index_train_units": labels.units.len(),
            "measurable_slots": family_color_json(measurable),
            "tied_teacher_maximum_slots": family_color_json(teacher_tied),
            "exact_join_mismatches": 0,
        }),
    ))
}

fn validate_joined_unit(
    prepared: &prepared::PreparedUnit,
    label: &reveal::LabelUnit,
) -> Result<(), String> {
    if label.unit_uid != prepared.unit_uid
        || label.opening_hash != prepared.opening_group_hash
        || label.ordinal != prepared.figrid_ordinal
    {
        return Err(format!(
            "unit identity mismatch for prepared UID {}",
            prepared.unit_uid
        ));
    }
    validate_joined_parent(&prepared.black, &label.black, Stone::Black)?;
    validate_joined_parent(&prepared.white, &label.white, Stone::White)?;
    Ok(())
}

fn validate_joined_parent(
    prepared: &prepared::PreparedParent,
    label: &reveal::LabelParent,
    expected_side: Stone,
) -> Result<(), String> {
    if prepared.side_to_move != expected_side
        || label.side != expected_side
        || prepared.parent_d4_side_hash != label.parent_hash
    {
        return Err(format!(
            "joined parent side/hash mismatch at {}",
            label.row_uid
        ));
    }
    if prepared.history.len() != label.history.len()
        || prepared
            .history
            .iter()
            .zip(&label.history)
            .any(|(left, right)| (left.mv, left.color) != *right)
    {
        return Err(format!("joined history mismatch at {}", label.row_uid));
    }
    if prepared.legal_inventory.len() != label.inventory.len() {
        return Err(format!(
            "joined legal inventory length mismatch at {}",
            label.row_uid
        ));
    }
    for (left, right) in prepared.legal_inventory.iter().zip(&label.inventory) {
        if left.mv != right.mv
            || left.child_d4_side_hash != right.child_hash
            || left.legacy_black_logit_bits != right.legacy_black_logit_bits
        {
            return Err(format!(
                "joined legal inventory identity mismatch at {} move {}",
                label.row_uid, left.mv
            ));
        }
    }
    for candidate in &label.candidates {
        let inventory_matches = label
            .inventory
            .iter()
            .filter(|entry| {
                entry.mv == candidate.mv
                    && entry.child_hash == candidate.child_hash
                    && entry.legacy_black_logit_bits == candidate.legacy_black_logit_bits
            })
            .count();
        if inventory_matches != 1 {
            return Err(format!(
                "candidate/inventory identity count {inventory_matches} at {} move {}",
                label.row_uid, candidate.mv
            ));
        }
    }
    let deployed_actual = label.deployed_actual_move()?;
    if deployed_actual != prepared.figrid_actual_move {
        return Err(format!(
            "archived/deployed actual mismatch at {}",
            label.row_uid
        ));
    }
    Ok(())
}

fn observation_for_move(
    parent: &reveal::LabelParent,
    mv: Move,
    must_be_present: bool,
) -> Result<stats::Observation, String> {
    if !parent.q_teacher.iter().all(|value| value.is_finite()) {
        return Err(format!("non-finite q_teacher at {}", parent.row_uid));
    }
    let q_max = parent
        .q_teacher
        .iter()
        .copied()
        .reduce(f64::max)
        .ok_or_else(|| format!("empty q_teacher at {}", parent.row_uid))?;
    let teacher_max_tied = parent
        .q_teacher
        .iter()
        .filter(|&&value| value == q_max)
        .count()
        > 1;
    let Some(index) = parent.candidate_index(mv) else {
        if must_be_present {
            return Err(format!(
                "required archived actual move {mv} is absent from K=6 at {}",
                parent.row_uid
            ));
        }
        return Ok(stats::Observation {
            teacher_max_tied,
            ..stats::Observation::unmeasurable()
        });
    };
    let q_choice = parent.q_teacher[index];
    let regret = q_max - q_choice;
    if !regret.is_finite() || regret < 0.0 {
        return Err(format!(
            "invalid regret {regret:?} at {} move {mv}",
            parent.row_uid
        ));
    }
    Ok(stats::Observation {
        measurable: true,
        error: q_choice < q_max,
        regret,
        teacher_max_tied,
    })
}

fn require_exact_object_keys(value: &Value, expected: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let observed = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if observed != expected {
        return Err(format!(
            "{label} key mismatch: observed={observed:?} expected={expected:?}"
        ));
    }
    Ok(())
}

fn require_json_string(value: &Value, key: &str, expected: &str) -> Result<(), String> {
    let observed = value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} must be a string"))?;
    if observed != expected {
        return Err(format!(
            "{key} mismatch: observed={observed:?} expected={expected:?}"
        ));
    }
    Ok(())
}

fn require_json_bool(value: &Value, key: &str, expected: bool) -> Result<(), String> {
    let observed = value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{key} must be a bool"))?;
    if observed != expected {
        return Err(format!(
            "{key} mismatch: observed={observed} expected={expected}"
        ));
    }
    Ok(())
}

fn parse_decimal_json_string(value: &Value, key: &str) -> Result<u128, String> {
    let text = value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} must be a decimal string"))?;
    let parsed = text
        .parse::<u128>()
        .map_err(|error| format!("{key} is not decimal: {error}"))?;
    if parsed.to_string() != text {
        return Err(format!("{key} is not canonical decimal"));
    }
    Ok(parsed)
}

fn validate_registered_arguments(arguments: &[OsString]) -> Result<(), String> {
    let rendered = arguments
        .iter()
        .map(|value| {
            value
                .to_str()
                .map(str::to_string)
                .ok_or_else(|| "arguments must be valid UTF-8".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_registered_rendered_arguments(&rendered)
}

fn validate_registered_rendered_arguments(rendered: &[String]) -> Result<(), String> {
    let Some(stage) = rendered.first().map(String::as_str) else {
        return Err(usage().to_string());
    };
    match stage {
        "p0a" => {
            const EXPECTED: &[&str] = &[
                "p0a",
                "--prepared-units",
                PREPARED_UNITS_PATH,
                "--phase2-manifest",
                PHASE2_MANIFEST_PATH,
                "--product-model",
                PRODUCT_MODEL_PATH,
                "--product-cbf",
                PRODUCT_CBF_PATH,
                "--topk",
                TOPK_PATH,
                "--out-selector",
                P0A_OUTPUT_PATH,
            ];
            require_exact_argument_vector(&rendered, EXPECTED, &[], "P0A")
        }
        "p0b" => {
            const EXPECTED: &[&str] = &[
                "p0b",
                "--selector",
                P0A_OUTPUT_PATH,
                "--expected-p0a-bytes",
                "<P0A_BYTES>",
                "--expected-p0a-sha256",
                "<P0A_SHA256>",
                "--prepared-units",
                PREPARED_UNITS_PATH,
                "--phase2-manifest",
                PHASE2_MANIFEST_PATH,
                "--product-model",
                PRODUCT_MODEL_PATH,
                "--product-cbf",
                PRODUCT_CBF_PATH,
                "--topk",
                TOPK_PATH,
                "--train",
                TRAIN_PATH,
                "--final-manifest",
                FINAL_MANIFEST_PATH,
                "--lineage-model",
                LINEAGE_MODEL_PATH,
                "--out-reveal",
                P0B_OUTPUT_PATH,
            ];
            require_exact_argument_vector(&rendered, EXPECTED, &[4, 6], "P0B")
        }
        _ => Err(format!(
            "argument vector does not use a registered stage: {stage:?}"
        )),
    }
}

fn validate_recorded_p0a_argv(provenance: &Value) -> Result<(), String> {
    let argv = provenance
        .get("argv")
        .and_then(Value::as_array)
        .ok_or_else(|| "P0A provenance argv must be an array".to_string())?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("P0A provenance argv[{index}] must be a string"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if argv.len() < 2 {
        return Err("P0A provenance argv omitted executable or arguments".to_string());
    }
    validate_registered_rendered_arguments(&argv[1..])?;

    let working_directory = provenance
        .get("working_directory")
        .and_then(Value::as_str)
        .ok_or_else(|| "P0A provenance working_directory must be a string".to_string())?;
    let executable_path = provenance
        .get("executable")
        .and_then(|value| value.get("path"))
        .and_then(Value::as_str)
        .ok_or_else(|| "P0A provenance executable.path must be a string".to_string())?;
    let argv0 = PathBuf::from(&argv[0]);
    let resolved_argv0 = if argv0.is_absolute() {
        argv0
    } else {
        Path::new(working_directory).join(argv0)
    }
    .canonicalize()
    .map_err(|error| format!("failed to canonicalize recorded P0A argv[0]: {error}"))?;
    if resolved_argv0.display().to_string() != executable_path {
        return Err(format!(
            "recorded P0A argv[0]/executable mismatch: argv0={:?} executable={executable_path:?}",
            resolved_argv0.display().to_string()
        ));
    }
    Ok(())
}

fn require_exact_argument_vector(
    observed: &[String],
    expected: &[&str],
    dynamic_indices: &[usize],
    label: &str,
) -> Result<(), String> {
    if observed.len() != expected.len() {
        return Err(format!(
            "{label} argument count mismatch: observed={} expected={}",
            observed.len(),
            expected.len()
        ));
    }
    for (index, (observed, expected)) in observed.iter().zip(expected).enumerate() {
        if !dynamic_indices.contains(&index) && observed != expected {
            return Err(format!(
                "{label} argument[{index}] mismatch: observed={observed:?} expected={expected:?}"
            ));
        }
    }
    Ok(())
}

fn parse_args_from(values: impl Iterator<Item = OsString>) -> Result<StageArgs, String> {
    let mut values = values
        .map(|value| {
            value
                .into_string()
                .map_err(|_| "arguments must be valid UTF-8".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err(usage().to_string());
    }
    let stage = values.remove(0);
    if values.len() % 2 != 0 {
        return Err(format!("every option needs one value\n{}", usage()));
    }
    let mut seen = BTreeSet::new();
    let mut pairs = Vec::with_capacity(values.len() / 2);
    for pair in values.chunks_exact(2) {
        if !pair[0].starts_with("--") || !seen.insert(pair[0].clone()) {
            return Err(format!(
                "invalid or duplicate option {:?}\n{}",
                pair[0],
                usage()
            ));
        }
        pairs.push((pair[0].clone(), pair[1].clone()));
    }
    let mut take = |name: &str| -> Result<PathBuf, String> {
        let index = pairs
            .iter()
            .position(|(key, _)| key == name)
            .ok_or_else(|| format!("missing {name}\n{}", usage()))?;
        Ok(PathBuf::from(pairs.remove(index).1))
    };

    match stage.as_str() {
        "p0a" => {
            let shared = SharedPaths {
                prepared_units: take("--prepared-units")?,
                phase2_manifest: take("--phase2-manifest")?,
                product_model: take("--product-model")?,
                product_cbf: take("--product-cbf")?,
                topk: take("--topk")?,
            };
            let out_selector = take("--out-selector")?;
            if !pairs.is_empty() {
                return Err(format!("unexpected P0A options {pairs:?}\n{}", usage()));
            }
            Ok(StageArgs::P0a(P0aArgs {
                shared,
                out_selector,
            }))
        }
        "p0b" => {
            let selector = take("--selector")?;
            let expected_p0a_bytes_raw = take("--expected-p0a-bytes")?;
            let expected_p0a_bytes_text = expected_p0a_bytes_raw
                .to_str()
                .ok_or_else(|| "--expected-p0a-bytes is not UTF-8".to_string())?
                .to_string();
            let expected_p0a_bytes = expected_p0a_bytes_text
                .parse::<u64>()
                .map_err(|error| format!("invalid --expected-p0a-bytes: {error}"))?;
            if expected_p0a_bytes.to_string() != expected_p0a_bytes_text {
                return Err(
                    "--expected-p0a-bytes must be the canonical decimal literal printed by P0A"
                        .to_string(),
                );
            }
            let expected_p0a_sha256 = take("--expected-p0a-sha256")?
                .to_str()
                .ok_or_else(|| "--expected-p0a-sha256 is not UTF-8".to_string())?
                .to_string();
            require_upper_hash(&expected_p0a_sha256, "--expected-p0a-sha256")?;
            let shared = SharedPaths {
                prepared_units: take("--prepared-units")?,
                phase2_manifest: take("--phase2-manifest")?,
                product_model: take("--product-model")?,
                product_cbf: take("--product-cbf")?,
                topk: take("--topk")?,
            };
            let train = take("--train")?;
            let final_manifest = take("--final-manifest")?;
            let lineage_model = take("--lineage-model")?;
            let out_reveal = take("--out-reveal")?;
            if !pairs.is_empty() {
                return Err(format!("unexpected P0B options {pairs:?}\n{}", usage()));
            }
            Ok(StageArgs::P0b(P0bArgs {
                selector,
                expected_p0a_bytes,
                expected_p0a_sha256,
                shared,
                train,
                final_manifest,
                lineage_model,
                out_reveal,
            }))
        }
        _ => Err(format!("unknown stage {stage:?}\n{}", usage())),
    }
}

fn usage() -> &'static str {
    "usage:\n\
cb-al1-selector p0a --prepared-units PREPARED.jsonl --phase2-manifest MANIFEST.json \
--product-model MODEL.json --product-cbf PRODUCT.cbf --topk topk.bin \
--out-selector NEW.json\n\
cb-al1-selector p0b --selector P0A.json --expected-p0a-bytes N \
--expected-p0a-sha256 SHA256 --prepared-units PREPARED.jsonl \
--phase2-manifest MANIFEST.json --product-model MODEL.json \
--product-cbf PRODUCT.cbf --topk topk.bin --train TRAIN.jsonl \
--final-manifest MANIFEST.json --lineage-model MODEL.json --out-reveal NEW.json"
}

fn stage_provenance(stage: &str) -> Result<Value, String> {
    let source = provenance::source_identity(PREREGISTER_COMMIT, CRITICAL_SOURCES)?;
    let stream = critical_source_stream_identity()?;
    let environment = environment_identity()?;
    let toolchain = provenance::toolchain_identity()?;
    validate_toolchain_identity(&toolchain)?;
    let cargo = cargo_identity()?;
    let cpu = provenance::cpu_identity()?;
    require_registered_executable_path()?;
    let executable = provenance::executable_identity(EXECUTABLE_STEM)?;
    let working_directory = env::current_dir()
        .map_err(|error| format!("current_dir failed: {error}"))?
        .canonicalize()
        .map_err(|error| format!("working-directory canonicalization failed: {error}"))?;
    let expected = Path::new(env!("CARGO_MANIFEST_DIR"))
        .canonicalize()
        .map_err(|error| format!("manifest-directory canonicalization failed: {error}"))?;
    let registered = Path::new(REGISTERED_WORKING_DIRECTORY)
        .canonicalize()
        .map_err(|error| format!("registered-directory canonicalization failed: {error}"))?;
    if working_directory != expected || working_directory != registered {
        return Err(format!(
            "working directory mismatch: observed={} manifest={} registered={}",
            working_directory.display(),
            expected.display(),
            registered.display(),
        ));
    }
    Ok(json!({
        "stage": stage,
        "preregister_commit": PREREGISTER_COMMIT,
        "source": source,
        "critical_source_stream": stream,
        "environment": environment,
        "toolchain": toolchain,
        "cargo": cargo,
        "cpu": cpu,
        "executable": executable,
        "working_directory": working_directory.display().to_string(),
        "argv": env::args_os().map(|value| value.to_string_lossy().into_owned()).collect::<Vec<_>>(),
        "target_profile": "release",
        "debug_assertions": cfg!(debug_assertions),
        "enabled_features": ["cb-al1-audit", "codebook-eval"],
        "canonical_build": CANONICAL_BUILD,
    }))
}

fn require_registered_executable_path() -> Result<(), String> {
    let observed = env::current_exe()
        .map_err(|error| format!("current_exe failed: {error}"))?
        .canonicalize()
        .map_err(|error| format!("current executable canonicalization failed: {error}"))?;
    let expected = Path::new(REGISTERED_WORKING_DIRECTORY)
        .join("target")
        .join("release")
        .join(format!("{EXECUTABLE_STEM}.exe"))
        .canonicalize()
        .map_err(|error| format!("registered executable canonicalization failed: {error}"))?;
    if observed != expected {
        return Err(format!(
            "executable path mismatch: observed={} registered={}",
            observed.display(),
            expected.display()
        ));
    }
    Ok(())
}

fn critical_source_stream_identity() -> Result<Value, String> {
    require_frozen_compiled_file(
        "Cargo.lock",
        include_bytes!("../Cargo.lock"),
        FROZEN_CARGO_LOCK_BYTES,
        FROZEN_CARGO_LOCK_SHA256,
    )?;
    require_frozen_compiled_file(
        PREREGISTER_DOCUMENT,
        include_bytes!("../experiments/2026-07-26/cb_al1_active_distillation_preregister.md"),
        FROZEN_PREREG_BYTES,
        FROZEN_PREREG_SHA256,
    )?;
    let mut bytes = Vec::new();
    let mut paths = Vec::with_capacity(CRITICAL_SOURCES.len());
    for &(path, payload) in CRITICAL_SOURCES {
        let path_bytes = path.as_bytes();
        let path_len = u32::try_from(path_bytes.len())
            .map_err(|_| format!("critical path too long: {path}"))?;
        let payload_len = u64::try_from(payload.len())
            .map_err(|_| format!("critical payload too long: {path}"))?;
        bytes.extend_from_slice(&path_len.to_le_bytes());
        bytes.extend_from_slice(path_bytes);
        bytes.extend_from_slice(&payload_len.to_le_bytes());
        bytes.extend_from_slice(payload);
        paths.push(path);
    }
    Ok(json!({
        "encoding": "u32_le(path_len)||path_utf8||u64_le(file_len)||file_bytes",
        "ordered_paths": paths,
        "bytes": bytes.len(),
        "sha256": provenance::sha256_hex(&bytes),
    }))
}

fn require_frozen_compiled_file(
    label: &str,
    bytes: &[u8],
    expected_bytes: usize,
    expected_sha256: &str,
) -> Result<(), String> {
    let observed_sha256 = provenance::sha256_hex(bytes);
    if bytes.len() != expected_bytes || observed_sha256 != expected_sha256 {
        return Err(format!(
            "{label} frozen identity mismatch: bytes={} sha256={observed_sha256}",
            bytes.len()
        ));
    }
    Ok(())
}

fn environment_identity() -> Result<Value, String> {
    if cfg!(debug_assertions) {
        return Err("debug_assertions must be false".to_string());
    }
    if !cfg!(all(
        target_feature = "avx2",
        target_feature = "bmi2",
        target_feature = "fma"
    )) {
        return Err("compiled target must expose AVX2/BMI2/FMA".to_string());
    }
    if COMPILE_TIME_RUSTFLAGS != Some(CANONICAL_RUSTFLAGS) {
        return Err(format!(
            "compile-time RUSTFLAGS mismatch: {COMPILE_TIME_RUSTFLAGS:?}"
        ));
    }
    let compile_forbidden = COMPILE_TIME_FORBIDDEN
        .iter()
        .filter_map(|(name, value)| value.map(|_| *name))
        .collect::<Vec<_>>();
    if !compile_forbidden.is_empty() {
        return Err(format!(
            "compile-time forbidden environment variables: {compile_forbidden:?}"
        ));
    }
    let feature_census = [
        ("avx512", cfg!(feature = "avx512")),
        ("cb-al1-audit", cfg!(feature = "cb-al1-audit")),
        ("cb-f1-flat-asset", cfg!(feature = "cb-f1-flat-asset")),
        ("codebook-eval", cfg!(feature = "codebook-eval")),
        ("embed-weights", cfg!(feature = "embed-weights")),
    ];
    let enabled = feature_census
        .iter()
        .filter_map(|(name, value)| value.then_some(*name))
        .collect::<Vec<_>>();
    if enabled != ["cb-al1-audit", "codebook-eval"] {
        return Err(format!("Cargo feature census mismatch: {feature_census:?}"));
    }
    #[cfg(target_arch = "x86_64")]
    if !std::is_x86_feature_detected!("avx2")
        || !std::is_x86_feature_detected!("bmi2")
        || !std::is_x86_feature_detected!("fma")
    {
        return Err("runtime CPU must expose AVX2/BMI2/FMA".to_string());
    }

    const EXACT_FORBIDDEN: &[&str] = &[
        "LLVM_PROFILE_FILE",
        "GCOV_PREFIX",
        "GCOV_PREFIX_STRIP",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "RUSTDOCFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTC_BOOTSTRAP",
        "CARGO_INCREMENTAL",
    ];
    const PREFIX_FORBIDDEN: &[&str] = &["NORU_", "FIGRID_", "RAYON_", "CARGO_PROFILE_"];

    let mut rustflags = Vec::new();
    let mut forbidden = Vec::new();
    for (name, value) in env::vars_os() {
        let rendered = name.to_string_lossy().into_owned();
        let upper = rendered.to_ascii_uppercase();
        if upper == "RUSTFLAGS" {
            rustflags.push((rendered.clone(), value.to_string_lossy().into_owned()));
        }
        if EXACT_FORBIDDEN.contains(&upper.as_str())
            || PREFIX_FORBIDDEN
                .iter()
                .any(|prefix| upper.starts_with(prefix))
        {
            forbidden.push(rendered);
        }
    }
    rustflags.sort();
    forbidden.sort();
    if rustflags.len() != 1 || rustflags[0].1 != CANONICAL_RUSTFLAGS {
        return Err(format!(
            "case-insensitive RUSTFLAGS census mismatch: {rustflags:?}"
        ));
    }
    if !forbidden.is_empty() {
        return Err(format!("forbidden environment variables: {forbidden:?}"));
    }
    Ok(json!({
        "rustflags_entries": rustflags,
        "forbidden_entries": forbidden,
        "compile_time_rustflags": COMPILE_TIME_RUSTFLAGS,
        "compile_time_forbidden_census": COMPILE_TIME_FORBIDDEN,
        "compiled_target_features": {
            "avx2": cfg!(target_feature = "avx2"),
            "bmi2": cfg!(target_feature = "bmi2"),
            "fma": cfg!(target_feature = "fma"),
        },
        "runtime_target_features": {
            "avx2": std::is_x86_feature_detected!("avx2"),
            "bmi2": std::is_x86_feature_detected!("bmi2"),
            "fma": std::is_x86_feature_detected!("fma"),
        },
        "cargo_feature_census": feature_census,
        "enabled_features": enabled,
    }))
}

fn validate_toolchain_identity(value: &Value) -> Result<(), String> {
    let rustc = value
        .get("rustc_vv")
        .and_then(Value::as_str)
        .ok_or_else(|| "toolchain identity omitted rustc_vv".to_string())?;
    for required in [
        "rustc 1.88.0 (6b00bc388 2025-06-23)",
        "commit-hash: 6b00bc3880198600130e1cf62b8f8a93494488cc",
        "host: x86_64-pc-windows-msvc",
        "release: 1.88.0",
        "LLVM version: 20.1.5",
    ] {
        if !rustc.lines().any(|line| line == required) {
            return Err(format!("rustc -Vv missing registered line {required:?}"));
        }
    }
    Ok(())
}

fn cargo_identity() -> Result<Value, String> {
    let output = Command::new("cargo")
        .arg("-V")
        .output()
        .map_err(|error| format!("cargo -V failed to launch: {error}"))?;
    if !output.status.success() {
        return Err(format!("cargo -V failed with {}", output.status));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("cargo -V emitted non-UTF-8: {error}"))?
        .trim()
        .to_string();
    if stdout != "cargo 1.88.0 (873a06493 2025-05-10)" {
        return Err(format!("cargo identity mismatch: {stdout:?}"));
    }
    Ok(json!({"version": stdout}))
}

fn output_target_identity(path: &Path) -> Result<Value, String> {
    if path.exists() {
        return Err(format!("refusing to overwrite {}", path.display()));
    }
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("output path has no UTF-8 file name: {}", path.display()))?;
    let parent = path
        .parent()
        .ok_or_else(|| format!("output path has no parent: {}", path.display()))?
        .canonicalize()
        .map_err(|error| {
            format!(
                "failed to canonicalize output parent {}: {error}",
                path.display()
            )
        })?;
    let resolved = parent.join(file_name);
    if resolved.exists() {
        return Err(format!(
            "resolved output already exists: {}",
            resolved.display()
        ));
    }
    Ok(json!({"path": resolved.display().to_string()}))
}

fn write_new_json(path: &Path, value: &Value) -> Result<provenance::FileSeal, String> {
    let mut payload =
        serde_json::to_vec_pretty(value).map_err(|error| format!("JSON serialization: {error}"))?;
    payload.push(b'\n');
    let duplicate = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("repeat serialization: {error}"))?;
    if duplicate.as_slice() != &payload[..payload.len() - 1] {
        return Err("second in-process serialization was not byte-identical".to_string());
    }
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    if let Err(error) = output
        .write_all(&payload)
        .and_then(|_| output.flush())
        .and_then(|_| output.sync_all())
    {
        drop(output);
        let _ = fs::remove_file(path);
        return Err(format!("failed to commit {}: {error}", path.display()));
    }
    drop(output);
    let seal = match provenance::seal_file(path) {
        Ok(seal) => seal,
        Err(error) => {
            let _ = fs::remove_file(path);
            return Err(error);
        }
    };
    let reread = match fs::read(path) {
        Ok(reread) => reread,
        Err(error) => {
            let _ = fs::remove_file(path);
            return Err(format!("failed to re-read {}: {error}", path.display()));
        }
    };
    if reread != payload {
        let _ = fs::remove_file(path);
        return Err("create-new output re-read mismatch".to_string());
    }
    Ok(seal)
}

fn require_upper_hash(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'A'..=b'F').contains(&byte))
    {
        return Err(format!("{field} must be 64 uppercase hex characters"));
    }
    Ok(())
}

#[cfg(test)]
#[path = "cb_al1_selector/tests.rs"]
mod tests;
