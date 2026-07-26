#![cfg(feature = "cb-p1-audit")]

//! CB-P1 exact bounded-DFPN precondition census.
//!
//! This executable deliberately has one sealed row-bearing input and one
//! create-new output.  It does not deserialize or emit any historical label,
//! game result, engine score, actual move, or prior VCT verdict.

#[allow(dead_code)]
#[path = "cb_al1_selector/hash.rs"]
mod hash;

use figrid_board::board::{BOARD_SIZE, BitBoard, Board, NUM_CELLS, RuleSet, Stone};
use figrid_board::vct::dfpn::{
    BoundedDfpnConfig, BoundedDfpnSession, DfpnCheckpoint, DfpnError, DfpnStatus,
};
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const FORMAT: &str = "cb-p1-bounded-dfpn-census-v1";
const INPUT_FORMAT: &str = "rq547-tactical-position-v1";
const PREREGISTER_COMMIT: &str = "0f0c1e483582a3586a8530342bad8a6019c775ad";
const PREREGISTER_DOCUMENT: &str = "experiments/2026-07-26/cb_p1_bounded_dfpn_preregister.md";
const REGISTERED_CWD: &str =
    r"C:\Users\concreate\.codex\worktrees\06f2\noru-tactic\target\figrid-release-0.8.2";
const REGISTERED_INPUT: &str = r"C:\Users\concreate\Documents\workspace\noru-tactic\experiments\2026-07-05\rq547a_tactical_positions.jsonl";
const REGISTERED_OUTPUT: &str = "experiments/2026-07-26/cb_p1_bounded_dfpn_census.json";
const INPUT_BYTES: u64 = 309_683;
const INPUT_SHA256: &str = "F02663E51716A13F54E0AB22829F7B6FBC7D237F843FAA79BCF62CE3A8EA171F";
const PREREGISTER_BYTES: u64 = 16_275;
const PREREGISTER_SHA256: &str = "655E71928F41FF469D095AB1E30F08A3C1FBD5AA49D283C4FF2A809604802DD0";
const CARGO_LOCK_BYTES: u64 = 11_841;
const CARGO_LOCK_SHA256: &str = "6A6B62449A235ABA53C777484C5D34E18EDB556155B1964A4B2BA6DA7DE2059C";
const EXPECTED_ROWS: usize = 307;
const CANONICAL_RUSTFLAGS: &str = "-C target-cpu=x86-64-v3";
const CANONICAL_BUILD: &str =
    "cargo build --release --locked --features cb-p1-audit --bin cb-p1-census";
const CAPS: [u64; 5] = [1_024, 4_096, 16_384, 65_536, 262_144];
const REFERENCE_CAP_INDEX: usize = 3;
const CEILING_CAP_INDEX: usize = 4;

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
];

const CRITICAL_SOURCES: &[(&str, &[u8])] = &[
    ("Cargo.toml", include_bytes!("../Cargo.toml")),
    ("Cargo.lock", include_bytes!("../Cargo.lock")),
    ("src/lib.rs", include_bytes!("../src/lib.rs")),
    ("src/board.rs", include_bytes!("../src/board.rs")),
    ("src/vct.rs", include_bytes!("../src/vct.rs")),
    ("src/vct/dfpn.rs", include_bytes!("../src/vct/dfpn.rs")),
    (
        "src/pattern_table.rs",
        include_bytes!("../src/pattern_table.rs"),
    ),
    ("bin/cb_p1_census.rs", include_bytes!("cb_p1_census.rs")),
    (
        "bin/cb_al1_selector/hash.rs",
        include_bytes!("cb_al1_selector/hash.rs"),
    ),
    (
        PREREGISTER_DOCUMENT,
        include_bytes!("../experiments/2026-07-26/cb_p1_bounded_dfpn_preregister.md"),
    ),
];

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,
}

#[derive(Clone)]
struct Root {
    ordinal: usize,
    uid: String,
    side: Stone,
    board: Board,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BoardSnapshot {
    black: (u128, u128),
    white: (u128, u128),
    side: u8,
    move_count: usize,
    last_move: Option<usize>,
    history: Vec<usize>,
    zobrist: u64,
    rule: u8,
    exact5: bool,
    line_patterns: Vec<u16>,
}

#[derive(Clone)]
struct CheckpointRecord {
    cap: u64,
    status: &'static str,
    expansions: u64,
    scientific: Value,
    diagnostic: Value,
}

#[derive(Clone)]
struct CertificateRecord {
    scientific: Value,
}

#[derive(Clone)]
struct RootRun {
    ordinal: usize,
    uid: String,
    side: Stone,
    checkpoints: Vec<CheckpointRecord>,
    certificate: Option<CertificateRecord>,
    scientific: Value,
}

#[derive(Clone, Debug)]
struct OracleResult {
    reference_budget: u64,
    reference_proofs: usize,
    oracle_proofs: usize,
    added_proofs: usize,
    assigned_caps: Vec<u64>,
    assigned_cost: u64,
}

fn main() {
    let arguments = env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() == 1 && matches!(arguments[0].to_str(), Some("-h" | "--help")) {
        print_help();
        return;
    }
    if let Err(error) = run(arguments) {
        if error.starts_with("UNSOUND ") {
            eprintln!("CB-P1 UNSOUND: {}", error.trim_start_matches("UNSOUND "));
            std::process::exit(2);
        }
        eprintln!("CB-P1 INVALID_CB_P1_P0: {error}");
        std::process::exit(1);
    }
}

fn run(arguments: Vec<OsString>) -> Result<(), String> {
    let args = parse_args(&arguments)?;
    validate_registered_paths(&args)?;
    refuse_existing(&args.output)?;

    let started_unix_ms = unix_millis()?;
    let provenance_before = provenance_identity(&arguments)?;
    let input_before =
        hash::require_file_seal(&args.input, INPUT_BYTES, INPUT_SHA256, "CB-P1 input")?;
    let roots = load_roots(&args.input)?;

    let mut runs = Vec::with_capacity(roots.len());
    for root in &roots {
        runs.push(run_root(root)?);
    }
    if runs.len() != EXPECTED_ROWS {
        return Err(format!(
            "incomplete run: got {} roots, expected {EXPECTED_ROWS}",
            runs.len()
        ));
    }

    let abort_count = runs
        .iter()
        .filter(|run| run.checkpoints[CEILING_CAP_INDEX].status == "UnknownAbort")
        .count();
    if abort_count != 0 {
        return Err(format!(
            "implementation watchdog aborted {abort_count} roots; census is incomplete"
        ));
    }

    let rerun_indices = determinism_indices(&runs);
    let mut determinism_errors = Vec::new();
    for &index in &rerun_indices {
        let rerun = run_root(&roots[index])?;
        if rerun.scientific != runs[index].scientific {
            determinism_errors.push(json!({
                "ordinal": index,
                "root_uid": roots[index].uid,
                "first_digest": hash::sha256_hex(
                    &serde_json::to_vec(&runs[index].scientific)
                        .map_err(|error| format!("determinism serialize first: {error}"))?
                ),
                "rerun_digest": hash::sha256_hex(
                    &serde_json::to_vec(&rerun.scientific)
                        .map_err(|error| format!("determinism serialize rerun: {error}"))?
                ),
            }));
        }
    }
    if !determinism_errors.is_empty() {
        return Err(format!(
            "determinism mismatch on {} roots: {}",
            determinism_errors.len(),
            serde_json::to_string(&determinism_errors)
                .map_err(|error| format!("determinism error serialization: {error}"))?
        ));
    }

    let oracle = perfect_oracle(&runs)?;
    let gates = evaluate_gates(&runs, &oracle);
    let final_label = gates
        .get("final_label")
        .and_then(Value::as_str)
        .ok_or_else(|| "gate report omitted final_label".to_string())?;

    let input_after = hash::require_file_seal(
        &args.input,
        INPUT_BYTES,
        INPUT_SHA256,
        "CB-P1 input recheck",
    )?;
    if input_after != input_before {
        return Err("input seal changed during census".to_string());
    }
    let provenance_after = provenance_identity(&arguments)?;
    if provenance_after != provenance_before {
        return Err("source, executable, environment, or git provenance changed during run".into());
    }
    let finished_unix_ms = unix_millis()?;

    let roots_json = runs.iter().map(root_run_json).collect::<Vec<_>>();
    let scientific_payload = json!({
        "format": FORMAT,
        "final_label": final_label,
        "claim_boundary": {
            "bounded_query_only": true,
            "root_attacker_forces_five_within_further_plies": 14,
            "all_legal_defender_replies": true,
            "full_game_win_claim": false,
            "exhausted_is_full_game_loss": false,
            "learned_head_present": false,
            "product_search_changed": false,
            "historical_labels_opened": false,
        },
        "preregistration": {
            "commit": PREREGISTER_COMMIT,
            "document": PREREGISTER_DOCUMENT,
        },
        "input": {
            "bytes": input_before.bytes.to_string(),
            "sha256": input_before.sha256,
            "rows": roots.len(),
            "format": INPUT_FORMAT,
            "deserialized_root_fields": [
                "format", "source_path", "game_id", "ply",
                "side_to_move", "position_history"
            ],
            "history_fields": ["x", "y", "color"],
        },
        "policy": {
            "horizon": 14,
            "attacker_vocabulary": "registered product-default forcing classifier",
            "defender_vocabulary": "all legal moves ascending cell",
            "fast_classify": true,
            "reach_mask": true,
            "jump_three_attack_defense": false,
            "jump_three_counter": false,
            "gap_four": false,
            "threat_index": false,
            "fast_immediate_five": false,
            "scratch_buffer_reuse": false,
            "terminal": "complete-bitboard Freestyle five before draw/horizon/generation",
            "memory_accounting_bytes": "144*states + 24*edges + 32*fingerprints + 24*collisions",
            "memory_cap": 64 * 1024 * 1024,
            "checkpoints": CAPS,
        },
        "roots": roots_json,
        "determinism": {
            "rerun_count": rerun_indices.len(),
            "rerun_ordinals": rerun_indices,
            "mismatches": 0,
        },
        "oracle": oracle_json(&oracle, &runs),
        "gates": gates,
    });
    let scientific_bytes = serde_json::to_vec(&scientific_payload)
        .map_err(|error| format!("scientific payload serialization: {error}"))?;
    let report = json!({
        "scientific": scientific_payload,
        "provenance": {
            "started_unix_ms": started_unix_ms.to_string(),
            "finished_unix_ms": finished_unix_ms.to_string(),
            "before": provenance_before,
            "after": provenance_after,
        },
        "output_seal": {
            "domain": "canonical compact JSON bytes of the scientific member",
            "bytes": scientific_bytes.len().to_string(),
            "sha256": hash::sha256_hex(&scientific_bytes),
        },
    });
    let output_seal = write_new_json(&args.output, &report)?;
    println!(
        "CB-P1 {final_label}: roots={} proofs={} oracle_added={} output_bytes={} output_sha256={}",
        roots.len(),
        runs.iter()
            .filter(|run| run.checkpoints[CEILING_CAP_INDEX].status == "ProvenWin")
            .count(),
        oracle.added_proofs,
        output_seal.bytes,
        output_seal.sha256,
    );
    Ok(())
}

/// The sole adapter between the audit executable and the public DFPN session.
fn run_root(root: &Root) -> Result<RootRun, String> {
    let mut board = root.board.clone();
    let root_snapshot = snapshot_board(&board);
    let config = BoundedDfpnConfig::registered();
    let mut session = BoundedDfpnSession::new(&board, config)
        .map_err(|error| format!("root {} session creation: {error:?}", root.ordinal))?;
    let mut checkpoints = Vec::with_capacity(CAPS.len());
    for cap in CAPS {
        let before = snapshot_board(&board);
        let started = Instant::now();
        let checkpoint = session
            .advance_to(&mut board, cap)
            .map_err(|error| classify_dfpn_error(root.ordinal, cap, error))?;
        let wall_nanos = started.elapsed().as_nanos();
        let after = snapshot_board(&board);
        if before != after || after != root_snapshot {
            return Err(format!(
                "UNSOUND board restoration mismatch at root {} cap {cap}",
                root.ordinal
            ));
        }
        checkpoints.push(checkpoint_record(checkpoint, wall_nanos)?);
    }

    let ceiling_status = checkpoints[CEILING_CAP_INDEX].status;
    let certificate = if matches!(ceiling_status, "ProvenWin" | "ExhaustedBounded") {
        let before = snapshot_board(&board);
        match session.verify_terminal_certificate(&mut board) {
            Ok(replay) => {
                let after = snapshot_board(&board);
                if before != after || after != root_snapshot || !replay.root_restored {
                    return Err(format!(
                        "UNSOUND certificate restoration mismatch at root {}",
                        root.ordinal
                    ));
                }
                Some(CertificateRecord {
                    scientific: json!({
                        "status": status_name(replay.status),
                        "visited_nodes": replay.visited_nodes.to_string(),
                        "visited_edges": replay.visited_edges.to_string(),
                        "certificate_digest": replay.certificate_digest,
                        "root_restored": replay.root_restored,
                    }),
                })
            }
            Err(error) => {
                return Err(format!(
                    "UNSOUND certificate replay failed at root {}: {error:?}",
                    root.ordinal
                ));
            }
        }
    } else {
        None
    };
    if snapshot_board(&board) != root_snapshot {
        return Err(format!(
            "UNSOUND final board restoration mismatch at root {}",
            root.ordinal
        ));
    }
    let checkpoint_science = checkpoints
        .iter()
        .map(|checkpoint| checkpoint.scientific.clone())
        .collect::<Vec<_>>();
    let scientific = json!({
        "ordinal": root.ordinal,
        "root_uid": root.uid,
        "root_side": stone_name(root.side),
        "checkpoints": checkpoint_science,
        "certificate": certificate.as_ref().map(|value| value.scientific.clone()),
    });
    Ok(RootRun {
        ordinal: root.ordinal,
        uid: root.uid.clone(),
        side: root.side,
        checkpoints,
        certificate,
        scientific,
    })
}

fn classify_dfpn_error(ordinal: usize, cap: u64, error: DfpnError) -> String {
    match error {
        DfpnError::RootMismatch | DfpnError::Certificate(_) | DfpnError::RestorationMismatch => {
            format!("UNSOUND root {ordinal} cap {cap}: {error}")
        }
        other => format!("root {ordinal} cap {cap}: {other}"),
    }
}

fn checkpoint_record(
    checkpoint: DfpnCheckpoint,
    outer_elapsed_nanos: u128,
) -> Result<CheckpointRecord, String> {
    let status = status_name(checkpoint.status);
    let width_histogram = checkpoint
        .width_histogram
        .iter()
        .map(|bin| {
            json!({
                "width": bin.width,
                "or_expansions": bin.or_count.to_string(),
                "and_expansions": bin.and_count.to_string(),
            })
        })
        .collect::<Vec<_>>();
    let scientific = json!({
        "expansion_cap": checkpoint.expansion_cap.to_string(),
        "pn": checkpoint.pn.to_string(),
        "dn": checkpoint.dn.to_string(),
        "status": status,
        "expansions": checkpoint.expansions.to_string(),
        "calls": checkpoint.calls.to_string(),
        "threshold_returns": checkpoint.threshold_returns.to_string(),
        "exact_states": checkpoint.exact_states.to_string(),
        "stored_edges": checkpoint.stored_edges.to_string(),
        "or_expansions": checkpoint.or_expansions.to_string(),
        "and_expansions": checkpoint.and_expansions.to_string(),
        "width_histogram": width_histogram,
        "exact_transposition_hits": checkpoint.exact_transposition_hits.to_string(),
        "fingerprint_collisions": checkpoint.fingerprint_collisions.to_string(),
        "distinct_fingerprints": checkpoint.distinct_fingerprints.to_string(),
        "collision_entries": checkpoint.collision_entries.to_string(),
        "exact_alias_errors": checkpoint.exact_alias_errors.to_string(),
        "accounted_bytes": checkpoint.accounted_bytes.to_string(),
        "root_state_digest": checkpoint.root_state_digest,
        "scientific_digest": checkpoint.scientific_digest,
    });
    let diagnostic = json!({
        "session_elapsed_nanos": checkpoint.elapsed_nanos.to_string(),
        "outer_advance_elapsed_nanos": outer_elapsed_nanos.to_string(),
        "process_peak_working_set_bytes": process_peak_working_set()
            .map(|value| value.to_string()),
    });
    Ok(CheckpointRecord {
        cap: checkpoint.expansion_cap,
        status,
        expansions: checkpoint.expansions,
        scientific,
        diagnostic,
    })
}

fn status_name(status: DfpnStatus) -> &'static str {
    match status {
        DfpnStatus::ProvenWin => "ProvenWin",
        DfpnStatus::ExhaustedBounded => "ExhaustedBounded",
        DfpnStatus::UnknownNodeBudget => "UnknownNodeBudget",
        DfpnStatus::UnknownMemory => "UnknownMemory",
        DfpnStatus::UnknownAbort => "UnknownAbort",
    }
}

fn root_run_json(run: &RootRun) -> Value {
    json!({
        "ordinal": run.ordinal,
        "root_uid": run.uid,
        "root_side": stone_name(run.side),
        "checkpoints": run.checkpoints.iter().map(|checkpoint| json!({
            "scientific": checkpoint.scientific,
            "diagnostic": checkpoint.diagnostic,
        })).collect::<Vec<_>>(),
        "certificate": run.certificate.as_ref().map(|value| value.scientific.clone()),
    })
}

fn determinism_indices(runs: &[RootRun]) -> Vec<usize> {
    let mut indices = BTreeSet::new();
    indices.extend(0..runs.len().min(32));
    for (index, run) in runs.iter().enumerate() {
        if run.checkpoints[CEILING_CAP_INDEX].status == "ProvenWin" {
            indices.insert(index);
        }
    }
    indices.into_iter().collect()
}

fn load_roots(path: &Path) -> Result<Vec<Root>, String> {
    let file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let mut roots = Vec::with_capacity(EXPECTED_ROWS);
    let mut source_game_pairs = BTreeSet::new();
    let mut exact_roots = BTreeSet::new();
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|error| format!("input line {} read error: {error}", line_index + 1))?;
        if line.trim().is_empty() {
            return Err(format!("blank input line {}", line_index + 1));
        }
        let object = project_allowed_root_fields(line.as_bytes())
            .map_err(|error| format!("input line {} JSON projection: {error}", line_index + 1))?;
        require_string(&object, "format", line_index + 1).and_then(|format| {
            (format == INPUT_FORMAT).then_some(()).ok_or_else(|| {
                format!(
                    "input line {} format {:?}, expected {INPUT_FORMAT:?}",
                    line_index + 1,
                    format
                )
            })
        })?;
        let source_path = require_string(&object, "source_path", line_index + 1)?;
        let game_id = object
            .get("game_id")
            .ok_or_else(|| format!("input line {} missing game_id", line_index + 1))?;
        if !(game_id.is_string() || game_id.is_u64() || game_id.is_i64()) {
            return Err(format!(
                "input line {} game_id must be string or integer",
                line_index + 1
            ));
        }
        let pair = format!(
            "{}\0{}",
            source_path,
            serde_json::to_string(game_id)
                .map_err(|error| format!("game_id serialization: {error}"))?
        );
        if !source_game_pairs.insert(pair) {
            return Err(format!(
                "input line {} duplicates (source_path, game_id)",
                line_index + 1
            ));
        }

        let ply_u64 = object
            .get("ply")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("input line {} ply is not u64", line_index + 1))?;
        let ply = usize::try_from(ply_u64)
            .map_err(|_| format!("input line {} ply exceeds usize", line_index + 1))?;
        let side = parse_stone(
            require_string(&object, "side_to_move", line_index + 1)?,
            &format!("input line {} side_to_move", line_index + 1),
        )?;
        let history_values = object
            .get("position_history")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                format!(
                    "input line {} position_history is not an array",
                    line_index + 1
                )
            })?;
        if ply != history_values.len() {
            return Err(format!(
                "input line {} ply {ply} != history length {}",
                line_index + 1,
                history_values.len()
            ));
        }
        let board = replay_history(history_values, line_index + 1)?;
        if board.side_to_move != side {
            return Err(format!(
                "input line {} stored side {} != replay side {}",
                line_index + 1,
                stone_name(side),
                stone_name(board.side_to_move)
            ));
        }
        let (black_win, white_win) = complete_winners(&board);
        if black_win || white_win || board.move_count == NUM_CELLS {
            return Err(format!(
                "input line {} root is terminal: black_win={black_win} white_win={white_win} full={}",
                line_index + 1,
                board.move_count == NUM_CELLS
            ));
        }
        let exact_key = exact_root_key(&board);
        if !exact_roots.insert(exact_key.clone()) {
            return Err(format!(
                "input line {} duplicates exact (black, white, side, rule) root",
                line_index + 1
            ));
        }
        let uid = hash::sha256_hex(
            [b"CB-P1-exact-root-v1\0".as_slice(), exact_key.as_slice()]
                .concat()
                .as_slice(),
        );
        roots.push(Root {
            ordinal: line_index,
            uid,
            side,
            board,
        });
    }
    if roots.len() != EXPECTED_ROWS {
        return Err(format!(
            "input contains {} rows, expected {EXPECTED_ROWS}",
            roots.len()
        ));
    }
    if source_game_pairs.len() != EXPECTED_ROWS || exact_roots.len() != EXPECTED_ROWS {
        return Err("input uniqueness census did not retain all 307 roots".to_string());
    }
    Ok(roots)
}

/// Project a sealed top-level JSON object without materializing forbidden
/// values. Unknown values are only traversed by the structural skipper. Only
/// the six preregistered raw slices are handed to serde_json.
fn project_allowed_root_fields(input: &[u8]) -> Result<Map<String, Value>, String> {
    const ALLOWED: [&str; 6] = [
        "format",
        "source_path",
        "game_id",
        "ply",
        "side_to_move",
        "position_history",
    ];
    let mut cursor = 0usize;
    skip_json_ws(input, &mut cursor);
    expect_byte(input, &mut cursor, b'{', "top-level object")?;
    let mut seen = BTreeSet::new();
    let mut projected = Map::new();
    skip_json_ws(input, &mut cursor);
    if consume_byte(input, &mut cursor, b'}') {
        return Err("top-level object is empty".to_string());
    }
    loop {
        skip_json_ws(input, &mut cursor);
        let key_start = cursor;
        skip_json_string(input, &mut cursor)?;
        let key: String = serde_json::from_slice(&input[key_start..cursor])
            .map_err(|error| format!("object key is not valid JSON string: {error}"))?;
        if !seen.insert(key.clone()) {
            return Err(format!("duplicate top-level key {key:?}"));
        }
        skip_json_ws(input, &mut cursor);
        expect_byte(input, &mut cursor, b':', "object colon")?;
        skip_json_ws(input, &mut cursor);
        let value_start = cursor;
        skip_json_value(input, &mut cursor, 0)?;
        if ALLOWED.contains(&key.as_str()) {
            let value = serde_json::from_slice(&input[value_start..cursor])
                .map_err(|error| format!("allowed field {key:?} is invalid JSON: {error}"))?;
            projected.insert(key, value);
        }
        skip_json_ws(input, &mut cursor);
        if consume_byte(input, &mut cursor, b'}') {
            break;
        }
        expect_byte(input, &mut cursor, b',', "object comma")?;
    }
    skip_json_ws(input, &mut cursor);
    if cursor != input.len() {
        return Err(format!(
            "trailing bytes after top-level object at offset {cursor}"
        ));
    }
    for field in ALLOWED {
        if !projected.contains_key(field) {
            return Err(format!("missing allowed field {field:?}"));
        }
    }
    Ok(projected)
}

fn skip_json_value(input: &[u8], cursor: &mut usize, depth: usize) -> Result<(), String> {
    if depth > 256 {
        return Err("JSON nesting exceeds 256".to_string());
    }
    skip_json_ws(input, cursor);
    match input.get(*cursor).copied() {
        Some(b'"') => skip_json_string(input, cursor),
        Some(b'{') => {
            *cursor += 1;
            skip_json_ws(input, cursor);
            if consume_byte(input, cursor, b'}') {
                return Ok(());
            }
            loop {
                skip_json_ws(input, cursor);
                skip_json_string(input, cursor)?;
                skip_json_ws(input, cursor);
                expect_byte(input, cursor, b':', "nested object colon")?;
                skip_json_value(input, cursor, depth + 1)?;
                skip_json_ws(input, cursor);
                if consume_byte(input, cursor, b'}') {
                    return Ok(());
                }
                expect_byte(input, cursor, b',', "nested object comma")?;
            }
        }
        Some(b'[') => {
            *cursor += 1;
            skip_json_ws(input, cursor);
            if consume_byte(input, cursor, b']') {
                return Ok(());
            }
            loop {
                skip_json_value(input, cursor, depth + 1)?;
                skip_json_ws(input, cursor);
                if consume_byte(input, cursor, b']') {
                    return Ok(());
                }
                expect_byte(input, cursor, b',', "array comma")?;
            }
        }
        Some(b't') => expect_literal(input, cursor, b"true"),
        Some(b'f') => expect_literal(input, cursor, b"false"),
        Some(b'n') => expect_literal(input, cursor, b"null"),
        Some(b'-' | b'0'..=b'9') => skip_json_number(input, cursor),
        Some(byte) => Err(format!(
            "unexpected JSON byte 0x{byte:02X} at offset {}",
            *cursor
        )),
        None => Err("unexpected EOF while reading JSON value".to_string()),
    }
}

fn skip_json_string(input: &[u8], cursor: &mut usize) -> Result<(), String> {
    expect_byte(input, cursor, b'"', "string quote")?;
    while let Some(byte) = input.get(*cursor).copied() {
        *cursor += 1;
        match byte {
            b'"' => return Ok(()),
            b'\\' => {
                let escaped = input
                    .get(*cursor)
                    .copied()
                    .ok_or_else(|| "EOF after JSON escape".to_string())?;
                *cursor += 1;
                if escaped == b'u' {
                    for _ in 0..4 {
                        let hex = input
                            .get(*cursor)
                            .copied()
                            .ok_or_else(|| "EOF in JSON unicode escape".to_string())?;
                        if !hex.is_ascii_hexdigit() {
                            return Err(format!(
                                "invalid unicode escape byte 0x{hex:02X} at offset {}",
                                *cursor
                            ));
                        }
                        *cursor += 1;
                    }
                } else if !matches!(
                    escaped,
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't'
                ) {
                    return Err(format!("invalid JSON escape 0x{escaped:02X}"));
                }
            }
            0x00..=0x1F => return Err("control byte in JSON string".to_string()),
            _ => {}
        }
    }
    Err("unterminated JSON string".to_string())
}

fn skip_json_number(input: &[u8], cursor: &mut usize) -> Result<(), String> {
    consume_byte(input, cursor, b'-');
    match input.get(*cursor).copied() {
        Some(b'0') => {
            *cursor += 1;
            if matches!(input.get(*cursor), Some(b'0'..=b'9')) {
                return Err("leading zero in JSON number".to_string());
            }
        }
        Some(b'1'..=b'9') => {
            *cursor += 1;
            while matches!(input.get(*cursor), Some(b'0'..=b'9')) {
                *cursor += 1;
            }
        }
        _ => return Err("invalid JSON integer".to_string()),
    }
    if consume_byte(input, cursor, b'.') {
        let start = *cursor;
        while matches!(input.get(*cursor), Some(b'0'..=b'9')) {
            *cursor += 1;
        }
        if *cursor == start {
            return Err("JSON fraction has no digits".to_string());
        }
    }
    if matches!(input.get(*cursor), Some(b'e' | b'E')) {
        *cursor += 1;
        if matches!(input.get(*cursor), Some(b'+' | b'-')) {
            *cursor += 1;
        }
        let start = *cursor;
        while matches!(input.get(*cursor), Some(b'0'..=b'9')) {
            *cursor += 1;
        }
        if *cursor == start {
            return Err("JSON exponent has no digits".to_string());
        }
    }
    Ok(())
}

fn skip_json_ws(input: &[u8], cursor: &mut usize) {
    while matches!(input.get(*cursor), Some(b' ' | b'\t' | b'\r' | b'\n')) {
        *cursor += 1;
    }
}

fn consume_byte(input: &[u8], cursor: &mut usize, expected: u8) -> bool {
    if input.get(*cursor) == Some(&expected) {
        *cursor += 1;
        true
    } else {
        false
    }
}

fn expect_byte(
    input: &[u8],
    cursor: &mut usize,
    expected: u8,
    context: &str,
) -> Result<(), String> {
    if consume_byte(input, cursor, expected) {
        Ok(())
    } else {
        Err(format!(
            "expected {context} byte 0x{expected:02X} at offset {}",
            *cursor
        ))
    }
}

fn expect_literal(input: &[u8], cursor: &mut usize, literal: &[u8]) -> Result<(), String> {
    if input.get(*cursor..(*cursor).saturating_add(literal.len())) == Some(literal) {
        *cursor += literal.len();
        Ok(())
    } else {
        Err(format!(
            "invalid JSON literal at offset {}, expected {}",
            *cursor,
            String::from_utf8_lossy(literal)
        ))
    }
}

fn require_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    line: usize,
) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("input line {line} field {field:?} is not a string"))
}

fn replay_history(values: &[Value], input_line: usize) -> Result<Board, String> {
    let mut board = Board::new();
    board.set_rule_set(RuleSet::Freestyle);
    for (ply, value) in values.iter().enumerate() {
        let (black_win, white_win) = complete_winners(&board);
        if black_win || white_win || board.move_count == NUM_CELLS {
            return Err(format!(
                "input line {input_line} contains move {ply} after terminal position"
            ));
        }
        let object = value
            .as_object()
            .ok_or_else(|| format!("input line {input_line} history ply {ply} is not an object"))?;
        let expected = ["color", "x", "y"].into_iter().collect::<BTreeSet<_>>();
        let observed = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
        if observed != expected {
            return Err(format!(
                "input line {input_line} history ply {ply} keys {observed:?}, expected {expected:?}"
            ));
        }
        let color = parse_stone(
            object.get("color").and_then(Value::as_str).ok_or_else(|| {
                format!("input line {input_line} history ply {ply} color is not a string")
            })?,
            &format!("input line {input_line} history ply {ply} color"),
        )?;
        if color != board.side_to_move {
            return Err(format!(
                "input line {input_line} history ply {ply} color {} != alternating side {}",
                stone_name(color),
                stone_name(board.side_to_move)
            ));
        }
        let x = json_usize(object.get("x"), input_line, ply, "x")?;
        let y = json_usize(object.get("y"), input_line, ply, "y")?;
        if x >= BOARD_SIZE || y >= BOARD_SIZE {
            return Err(format!(
                "input line {input_line} history ply {ply} out of range ({x},{y})"
            ));
        }
        let cell = y * BOARD_SIZE + x;
        if !board.is_legal_move(cell) || !board.is_empty(cell) {
            return Err(format!(
                "input line {input_line} history ply {ply} illegal/occupied cell {cell}"
            ));
        }
        let before = snapshot_board(&board);
        board.make_move(cell);
        if board.move_count != before.move_count + 1
            || board.history.last().copied() != Some(cell)
            || board.last_move != Some(cell)
        {
            return Err(format!(
                "input line {input_line} history ply {ply} make_move state mismatch"
            ));
        }
    }
    Ok(board)
}

fn json_usize(
    value: Option<&Value>,
    input_line: usize,
    ply: usize,
    field: &str,
) -> Result<usize, String> {
    let value = value.and_then(Value::as_u64).ok_or_else(|| {
        format!("input line {input_line} history ply {ply} field {field} is not u64")
    })?;
    usize::try_from(value).map_err(|_| {
        format!("input line {input_line} history ply {ply} field {field} exceeds usize")
    })
}

fn parse_stone(value: &str, field: &str) -> Result<Stone, String> {
    match value {
        "B" | "Black" | "black" => Ok(Stone::Black),
        "W" | "White" | "white" => Ok(Stone::White),
        _ => Err(format!("{field} has unknown color {value:?}")),
    }
}

fn stone_name(stone: Stone) -> &'static str {
    match stone {
        Stone::Black => "Black",
        Stone::White => "White",
    }
}

fn complete_winners(board: &Board) -> (bool, bool) {
    (
        bitboard_has_five(&board.black),
        bitboard_has_five(&board.white),
    )
}

fn bitboard_has_five(stones: &BitBoard) -> bool {
    const DIRECTIONS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
    for cell in stones.iter_ones() {
        let row = (cell / BOARD_SIZE) as i32;
        let col = (cell % BOARD_SIZE) as i32;
        for (dr, dc) in DIRECTIONS {
            let previous_row = row - dr;
            let previous_col = col - dc;
            if in_board(previous_row, previous_col)
                && stones.get(previous_row as usize * BOARD_SIZE + previous_col as usize)
            {
                continue;
            }
            let mut count = 0usize;
            let mut scan_row = row;
            let mut scan_col = col;
            while in_board(scan_row, scan_col)
                && stones.get(scan_row as usize * BOARD_SIZE + scan_col as usize)
            {
                count += 1;
                if count >= 5 {
                    return true;
                }
                scan_row += dr;
                scan_col += dc;
            }
        }
    }
    false
}

fn in_board(row: i32, col: i32) -> bool {
    row >= 0 && row < BOARD_SIZE as i32 && col >= 0 && col < BOARD_SIZE as i32
}

fn exact_root_key(board: &Board) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(66);
    bytes.extend_from_slice(&board.black.lo.to_le_bytes());
    bytes.extend_from_slice(&board.black.hi.to_le_bytes());
    bytes.extend_from_slice(&board.white.lo.to_le_bytes());
    bytes.extend_from_slice(&board.white.hi.to_le_bytes());
    bytes.push(match board.side_to_move {
        Stone::Black => 0,
        Stone::White => 1,
    });
    bytes.push(rule_code(board.effective_rule_set()));
    bytes
}

fn snapshot_board(board: &Board) -> BoardSnapshot {
    BoardSnapshot {
        black: (board.black.lo, board.black.hi),
        white: (board.white.lo, board.white.hi),
        side: match board.side_to_move {
            Stone::Black => 0,
            Stone::White => 1,
        },
        move_count: board.move_count,
        last_move: board.last_move,
        history: board.history.clone(),
        zobrist: board.zobrist,
        rule: rule_code(board.effective_rule_set()),
        exact5: board.exact5,
        line_patterns: board
            .line_pattern_ids
            .iter()
            .flat_map(|directions| directions.iter().copied())
            .collect(),
    }
}

fn rule_code(rule: RuleSet) -> u8 {
    match rule {
        RuleSet::Freestyle => 0,
        RuleSet::Standard => 1,
        RuleSet::Caro => 2,
        RuleSet::Renju => 3,
    }
}

fn perfect_oracle(runs: &[RootRun]) -> Result<OracleResult, String> {
    let reference_budget = runs.iter().try_fold(0u64, |total, run| {
        total
            .checked_add(run.checkpoints[REFERENCE_CAP_INDEX].expansions)
            .ok_or_else(|| "reference actual-cost sum overflow".to_string())
    })?;
    let mut assigned_caps = vec![0u64; runs.len()];
    let mut mandatory_cost = 0u64;
    let mut reference_proofs = 0usize;
    let mut optional_costs = vec![None; runs.len()];

    for (index, run) in runs.iter().enumerate() {
        let reference_proved = run.checkpoints[REFERENCE_CAP_INDEX].status == "ProvenWin";
        let first_proof = run
            .checkpoints
            .iter()
            .find(|checkpoint| checkpoint.status == "ProvenWin")
            .map(|checkpoint| (checkpoint.cap, checkpoint.expansions));
        if reference_proved {
            let (cap, cost) = first_proof.ok_or_else(|| {
                format!("root {index} is reference-proved without a first proof cap")
            })?;
            reference_proofs += 1;
            assigned_caps[index] = cap;
            mandatory_cost = mandatory_cost
                .checked_add(cost)
                .ok_or_else(|| "mandatory oracle cost overflow".to_string())?;
        } else if let Some((cap, cost)) = first_proof {
            optional_costs[index] = Some((cap, cost));
        }
    }
    if mandatory_cost > reference_budget {
        return Err(format!(
            "mandatory reference-proof cost {mandatory_cost} exceeds reference budget {reference_budget}"
        ));
    }
    let remaining_budget = reference_budget - mandatory_cost;

    // Suffix DP: minimum actual cost for exactly k optional proofs.  It is
    // budget-size independent (O(N^2)) and later supports exact
    // lexicographic reconstruction in input order.
    const UNREACHABLE: u64 = u64::MAX;
    let n = runs.len();
    let mut suffix = vec![vec![UNREACHABLE; n + 1]; n + 1];
    suffix[n][0] = 0;
    for index in (0..n).rev() {
        let tail_row = suffix[index + 1].clone();
        suffix[index].copy_from_slice(&tail_row);
        if let Some((_cap, cost)) = optional_costs[index] {
            for count in 1..=n - index {
                let tail = suffix[index + 1][count - 1];
                if tail != UNREACHABLE {
                    let candidate = tail.saturating_add(cost);
                    suffix[index][count] = suffix[index][count].min(candidate);
                }
            }
        }
    }
    let optional_count = (0..=n)
        .rev()
        .find(|&count| suffix[0][count] <= remaining_budget)
        .ok_or_else(|| "oracle DP could not represent the zero-proof assignment".to_string())?;

    let mut proofs_left = optional_count;
    let mut budget_left = remaining_budget;
    for index in 0..n {
        let Some((cap, cost)) = optional_costs[index] else {
            continue;
        };
        // Cap zero is lexicographically smaller. Skip this root whenever the
        // suffix can still realize the optimal proof count.
        if suffix[index + 1][proofs_left] <= budget_left {
            continue;
        }
        if proofs_left == 0
            || cost > budget_left
            || suffix[index + 1][proofs_left - 1] > budget_left - cost
        {
            return Err(format!(
                "oracle lexicographic reconstruction failed at root {index}"
            ));
        }
        assigned_caps[index] = cap;
        budget_left -= cost;
        proofs_left -= 1;
    }
    if proofs_left != 0 {
        return Err(format!(
            "oracle reconstruction left {proofs_left} proofs unassigned"
        ));
    }
    let assigned_cost = reference_budget - budget_left;
    let oracle_proofs = reference_proofs + optional_count;
    Ok(OracleResult {
        reference_budget,
        reference_proofs,
        oracle_proofs,
        added_proofs: optional_count,
        assigned_caps,
        assigned_cost,
    })
}

fn oracle_json(oracle: &OracleResult, runs: &[RootRun]) -> Value {
    let assignments = oracle
        .assigned_caps
        .iter()
        .enumerate()
        .map(|(index, &cap)| {
            let cost = if cap == 0 {
                0
            } else {
                runs[index]
                    .checkpoints
                    .iter()
                    .find(|checkpoint| checkpoint.cap == cap)
                    .map(|checkpoint| checkpoint.expansions)
                    .unwrap_or(u64::MAX)
            };
            json!({
                "ordinal": index,
                "root_uid": runs[index].uid,
                "cap": cap.to_string(),
                "actual_cost": cost.to_string(),
                "proof": cap != 0,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "optimization": "integer O(N^2) suffix DP over exact proof count and minimum actual cost",
        "tie_break": "lexicographically smallest assigned-cap vector in input order",
        "reference_cap": CAPS[REFERENCE_CAP_INDEX].to_string(),
        "reference_actual_cost_budget": oracle.reference_budget.to_string(),
        "reference_proofs": oracle.reference_proofs,
        "oracle_proofs": oracle.oracle_proofs,
        "added_proofs": oracle.added_proofs,
        "assigned_actual_cost": oracle.assigned_cost.to_string(),
        "within_budget": oracle.assigned_cost <= oracle.reference_budget,
        "preserves_reference_proofs": true,
        "assignments": assignments,
    })
}

fn evaluate_gates(runs: &[RootRun], oracle: &OracleResult) -> Value {
    let ceiling_proofs = runs
        .iter()
        .filter(|run| run.checkpoints[CEILING_CAP_INDEX].status == "ProvenWin")
        .count();
    let memory_roots = runs
        .iter()
        .filter(|run| run.checkpoints[CEILING_CAP_INDEX].status == "UnknownMemory")
        .count();
    let budget_sensitive = runs
        .iter()
        .filter(|run| {
            run.checkpoints[1].status != "ProvenWin"
                && run.checkpoints[2..]
                    .iter()
                    .any(|checkpoint| checkpoint.status == "ProvenWin")
        })
        .collect::<Vec<_>>();
    let sensitive_black = budget_sensitive
        .iter()
        .filter(|run| run.side == Stone::Black)
        .count();
    let sensitive_white = budget_sensitive
        .iter()
        .filter(|run| run.side == Stone::White)
        .count();
    let fingerprint_collisions = runs
        .iter()
        .map(|run| {
            scientific_u64(
                &run.checkpoints[CEILING_CAP_INDEX].scientific,
                "fingerprint_collisions",
            )
        })
        .sum::<u64>();
    let collision_entries = runs
        .iter()
        .map(|run| {
            scientific_u64(
                &run.checkpoints[CEILING_CAP_INDEX].scientific,
                "collision_entries",
            )
        })
        .sum::<u64>();
    let alias_errors = runs
        .iter()
        .map(|run| {
            scientific_u64(
                &run.checkpoints[CEILING_CAP_INDEX].scientific,
                "exact_alias_errors",
            )
        })
        .sum::<u64>();
    let missing_certificates = runs
        .iter()
        .filter(|run| {
            matches!(
                run.checkpoints[CEILING_CAP_INDEX].status,
                "ProvenWin" | "ExhaustedBounded"
            ) && run.certificate.is_none()
        })
        .count();
    let unsound = alias_errors != 0 || missing_certificates != 0;
    let within_budget = oracle.assigned_cost <= oracle.reference_budget;
    let all_upper = !unsound
        && memory_roots <= 15
        && ceiling_proofs >= 30
        && budget_sensitive.len() >= 10
        && sensitive_black >= 3
        && sensitive_white >= 3
        && within_budget
        && oracle.added_proofs >= 10;
    let final_label = if unsound {
        "UNSOUND"
    } else if memory_roots > 15 {
        "NO_GO_STATE_EXPLOSION"
    } else if all_upper {
        "GO_PROTOTYPE"
    } else {
        "NO_GO_PRECONDITION"
    };
    json!({
        "final_label": final_label,
        "correctness_and_certificate_errors_zero": !unsound,
        "fingerprint_collisions": fingerprint_collisions.to_string(),
        "collision_entries": collision_entries.to_string(),
        "observed_collisions_separated_by_exact_equality": alias_errors == 0,
        "exact_alias_errors": alias_errors.to_string(),
        "missing_certificates": missing_certificates,
        "memory_roots_at_most_15": memory_roots <= 15,
        "memory_roots": memory_roots,
        "ceiling_proofs_at_least_30": ceiling_proofs >= 30,
        "ceiling_proofs": ceiling_proofs,
        "budget_sensitive_at_least_10": budget_sensitive.len() >= 10,
        "budget_sensitive_roots": budget_sensitive.len(),
        "budget_sensitive_black_at_least_3": sensitive_black >= 3,
        "budget_sensitive_black": sensitive_black,
        "budget_sensitive_white_at_least_3": sensitive_white >= 3,
        "budget_sensitive_white": sensitive_white,
        "oracle_within_actual_reference_budget": within_budget,
        "oracle_preserves_reference_proofs": true,
        "oracle_adds_at_least_10": oracle.added_proofs >= 10,
        "oracle_added_proofs": oracle.added_proofs,
    })
}

fn scientific_u64(value: &Value, field: &str) -> u64 {
    value
        .get(field)
        .and_then(Value::as_str)
        .and_then(|value| value.parse().ok())
        .unwrap_or(u64::MAX)
}

fn parse_args(arguments: &[OsString]) -> Result<Args, String> {
    let mut input = None;
    let mut output = None;
    let mut index = 0usize;
    while index < arguments.len() {
        match arguments[index].to_str() {
            Some("--input") => {
                index += 1;
                input = Some(PathBuf::from(
                    arguments
                        .get(index)
                        .ok_or_else(|| "--input requires a path".to_string())?,
                ));
            }
            Some("--output") => {
                index += 1;
                output = Some(PathBuf::from(
                    arguments
                        .get(index)
                        .ok_or_else(|| "--output requires a path".to_string())?,
                ));
            }
            Some(flag) => return Err(format!("unknown argument {flag:?}")),
            None => return Err("arguments must be valid Unicode".to_string()),
        }
        index += 1;
    }
    Ok(Args {
        input: input.ok_or_else(|| "missing --input".to_string())?,
        output: output.ok_or_else(|| "missing --output".to_string())?,
    })
}

fn print_help() {
    println!(
        "Usage: cb-p1-census --input \"{REGISTERED_INPUT}\" --output {REGISTERED_OUTPUT}\n\
         The input, output, build, environment, git state, and CPU requirements are sealed by the CB-P1 preregistration."
    );
}

fn validate_registered_paths(args: &Args) -> Result<(), String> {
    let cwd = env::current_dir()
        .map_err(|error| format!("current_dir failed: {error}"))?
        .canonicalize()
        .map_err(|error| format!("working-directory canonicalize failed: {error}"))?;
    let expected_cwd = PathBuf::from(REGISTERED_CWD)
        .canonicalize()
        .map_err(|error| format!("registered working directory unavailable: {error}"))?;
    if cwd != expected_cwd {
        return Err(format!(
            "working directory {} != registered {}",
            cwd.display(),
            expected_cwd.display()
        ));
    }
    let input = args
        .input
        .canonicalize()
        .map_err(|error| format!("input canonicalize failed: {error}"))?;
    let expected_input = PathBuf::from(REGISTERED_INPUT)
        .canonicalize()
        .map_err(|error| format!("registered input unavailable: {error}"))?;
    if input != expected_input {
        return Err(format!(
            "input {} != registered {}",
            input.display(),
            expected_input.display()
        ));
    }
    let output = absolute_output_path(&args.output, &cwd)?;
    let expected_output = cwd.join(REGISTERED_OUTPUT);
    if output != expected_output {
        return Err(format!(
            "output {} != registered {}",
            output.display(),
            expected_output.display()
        ));
    }
    Ok(())
}

fn absolute_output_path(path: &Path, cwd: &Path) -> Result<PathBuf, String> {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    let name = joined
        .file_name()
        .ok_or_else(|| format!("output has no file name: {}", joined.display()))?;
    let parent = joined
        .parent()
        .ok_or_else(|| format!("output has no parent: {}", joined.display()))?
        .canonicalize()
        .map_err(|error| format!("output parent canonicalize failed: {error}"))?;
    Ok(parent.join(name))
}

fn refuse_existing(path: &Path) -> Result<(), String> {
    let cwd = env::current_dir().map_err(|error| format!("current_dir: {error}"))?;
    let absolute = absolute_output_path(path, &cwd)?;
    if absolute.exists() {
        return Err(format!("refusing to overwrite {}", absolute.display()));
    }
    let invalid = invalid_output_path(&absolute)?;
    if invalid.exists() {
        return Err(format!(
            "refusing run while prior invalid artifact exists: {}",
            invalid.display()
        ));
    }
    Ok(())
}

fn invalid_output_path(path: &Path) -> Result<PathBuf, String> {
    let file = path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| format!("output file name is not Unicode: {}", path.display()))?;
    Ok(path.with_file_name(format!("{file}.invalid")))
}

fn provenance_identity(arguments: &[OsString]) -> Result<Value, String> {
    validate_build_environment()?;
    let cwd = env::current_dir()
        .map_err(|error| format!("current_dir: {error}"))?
        .canonicalize()
        .map_err(|error| format!("cwd canonicalize: {error}"))?;
    let source = source_identity(&cwd)?;
    let executable = executable_identity()?;
    let git = git_identity(&cwd)?;
    let toolchain = toolchain_identity()?;
    let command_line = arguments
        .iter()
        .map(|value| {
            value
                .to_str()
                .map(str::to_string)
                .ok_or_else(|| "non-Unicode command line".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "canonical_build": CANONICAL_BUILD,
        "canonical_rustflags": CANONICAL_RUSTFLAGS,
        "working_directory": cwd.display().to_string(),
        "command_arguments": command_line,
        "source": source,
        "executable": executable,
        "git": git,
        "toolchain": toolchain,
        "cpu": {
            "compile_avx2": cfg!(target_feature = "avx2"),
            "compile_bmi2": cfg!(target_feature = "bmi2"),
            "compile_fma": cfg!(target_feature = "fma"),
            "runtime_avx2": std::is_x86_feature_detected!("avx2"),
            "runtime_bmi2": std::is_x86_feature_detected!("bmi2"),
            "runtime_fma": std::is_x86_feature_detected!("fma"),
        },
        "cargo_features": {
            "cb_p1_audit": cfg!(feature = "cb-p1-audit"),
            "codebook_eval": cfg!(feature = "codebook-eval"),
            "embed_weights": cfg!(feature = "embed-weights"),
            "cb_al1_audit": cfg!(feature = "cb-al1-audit"),
            "cb_f1_flat_asset": cfg!(feature = "cb-f1-flat-asset"),
            "avx512": cfg!(feature = "avx512"),
        },
    }))
}

fn validate_build_environment() -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("release mode with debug_assertions=false is required".to_string());
    }
    if !cfg!(target_arch = "x86_64")
        || !cfg!(target_feature = "avx2")
        || !cfg!(target_feature = "bmi2")
        || !cfg!(target_feature = "fma")
    {
        return Err("compile-time x86-64 AVX2/BMI2/FMA are required".to_string());
    }
    if !std::is_x86_feature_detected!("avx2")
        || !std::is_x86_feature_detected!("bmi2")
        || !std::is_x86_feature_detected!("fma")
    {
        return Err("runtime AVX2/BMI2/FMA are required".to_string());
    }
    if COMPILE_TIME_RUSTFLAGS != Some(CANONICAL_RUSTFLAGS) {
        return Err(format!(
            "compile-time RUSTFLAGS {:?} != registered {:?}",
            COMPILE_TIME_RUSTFLAGS, CANONICAL_RUSTFLAGS
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
    let enabled_features = [
        ("avx512", cfg!(feature = "avx512")),
        ("cb-al1-audit", cfg!(feature = "cb-al1-audit")),
        ("cb-f1-flat-asset", cfg!(feature = "cb-f1-flat-asset")),
        ("cb-p1-audit", cfg!(feature = "cb-p1-audit")),
        ("codebook-eval", cfg!(feature = "codebook-eval")),
        ("embed-weights", cfg!(feature = "embed-weights")),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect::<Vec<_>>();
    if enabled_features != ["cb-p1-audit"] {
        return Err(format!(
            "Cargo feature census mismatch: {enabled_features:?}"
        ));
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
    const PREFIX_FORBIDDEN: &[&str] = &["NORU_", "FIGRID_", "RAYON_", "CARGO_PROFILE_", "COV"];
    let mut rustflags = Vec::new();
    let mut forbidden = Vec::new();
    for (name, value) in env::vars_os() {
        let name = name.to_string_lossy().into_owned();
        let upper = name.to_ascii_uppercase();
        if upper == "RUSTFLAGS" {
            rustflags.push(value.to_string_lossy().into_owned());
        }
        if EXACT_FORBIDDEN.contains(&upper.as_str())
            || PREFIX_FORBIDDEN
                .iter()
                .any(|prefix| upper.starts_with(prefix))
        {
            forbidden.push(name);
        }
    }
    if rustflags != [CANONICAL_RUSTFLAGS] {
        return Err(format!("runtime RUSTFLAGS census mismatch: {rustflags:?}"));
    }
    if !forbidden.is_empty() {
        return Err(format!(
            "forbidden runtime environment variables: {forbidden:?}"
        ));
    }
    Ok(())
}

fn source_identity(cwd: &Path) -> Result<Value, String> {
    hash::require_file_seal(
        &cwd.join(PREREGISTER_DOCUMENT),
        PREREGISTER_BYTES,
        PREREGISTER_SHA256,
        "CB-P1 preregistration",
    )?;
    hash::require_file_seal(
        &cwd.join("Cargo.lock"),
        CARGO_LOCK_BYTES,
        CARGO_LOCK_SHA256,
        "registered Cargo.lock",
    )?;
    let mut aggregate = hash::Sha256::new();
    let mut files = Vec::with_capacity(CRITICAL_SOURCES.len());
    for &(relative, compiled) in CRITICAL_SOURCES {
        let disk_path = cwd.join(relative);
        let disk = fs::read(&disk_path)
            .map_err(|error| format!("critical source {} read: {error}", disk_path.display()))?;
        if disk != compiled {
            return Err(format!(
                "compiled/disk critical source mismatch for {relative}"
            ));
        }
        let relative_bytes = relative.as_bytes();
        aggregate.update(&(relative_bytes.len() as u64).to_le_bytes());
        aggregate.update(relative_bytes);
        aggregate.update(&(compiled.len() as u64).to_le_bytes());
        aggregate.update(compiled);
        files.push(json!({
            "path": relative,
            "compiled_bytes": compiled.len().to_string(),
            "compiled_sha256": hash::sha256_hex(compiled),
            "disk_bytes": disk.len().to_string(),
            "disk_sha256": hash::sha256_hex(&disk),
            "compiled_matches_disk": true,
        }));
    }
    Ok(json!({
        "stream_encoding": "u64le(path_len)||path||u64le(content_len)||content",
        "file_count": CRITICAL_SOURCES.len(),
        "aggregate_sha256": hash::hex_upper(&aggregate.finalize()),
        "files": files,
    }))
}

fn executable_identity() -> Result<Value, String> {
    let path = env::current_exe()
        .map_err(|error| format!("current_exe: {error}"))?
        .canonicalize()
        .map_err(|error| format!("current_exe canonicalize: {error}"))?;
    let stem = path
        .file_stem()
        .and_then(OsStr::to_str)
        .ok_or_else(|| format!("executable has no Unicode stem: {}", path.display()))?;
    if stem != "cb-p1-census" {
        return Err(format!("executable stem {stem:?} != \"cb-p1-census\""));
    }
    let seal = hash::seal_file(&path)?;
    Ok(json!({
        "path": path.display().to_string(),
        "bytes": seal.bytes.to_string(),
        "sha256": seal.sha256,
    }))
}

fn git_identity(cwd: &Path) -> Result<Value, String> {
    let safe = git_safe_directory(cwd);
    let head = git_stdout(cwd, &safe, &["rev-parse", "HEAD"])?;
    if head.len() != 40 || !head.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("git HEAD is not 40 hex: {head:?}"));
    }
    let status = git_stdout(
        cwd,
        &safe,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    if !status.is_empty() {
        return Err(format!(
            "worktree must be clean before output creation: {status}"
        ));
    }
    let ancestor_status = Command::new("git")
        .current_dir(cwd)
        .args(["-c", &format!("safe.directory={safe}")])
        .args(["merge-base", "--is-ancestor", PREREGISTER_COMMIT, &head])
        .status()
        .map_err(|error| format!("git merge-base launch: {error}"))?;
    if !ancestor_status.success() {
        return Err(format!(
            "HEAD {head} does not descend from preregistration {PREREGISTER_COMMIT}"
        ));
    }
    Ok(json!({
        "head": head,
        "preregister_commit": PREREGISTER_COMMIT,
        "descends_from_preregister": true,
        "worktree_clean": true,
    }))
}

fn git_safe_directory(path: &Path) -> String {
    let displayed = path.display().to_string();
    displayed
        .strip_prefix(r"\\?\")
        .unwrap_or(&displayed)
        .replace('\\', "/")
}

fn git_stdout(cwd: &Path, safe: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .current_dir(cwd)
        .args(["-c", &format!("safe.directory={safe}")])
        .args(args)
        .output()
        .map_err(|error| format!("git {args:?} launch: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("git {args:?} non-UTF8: {error}"))
        .map(|value| value.trim().to_string())
}

fn toolchain_identity() -> Result<Value, String> {
    let rustc = command_stdout("rustc", &["-Vv"])?;
    for line in [
        "rustc 1.88.0 (6b00bc388 2025-06-23)",
        "commit-hash: 6b00bc3880198600130e1cf62b8f8a93494488cc",
        "host: x86_64-pc-windows-msvc",
        "release: 1.88.0",
        "LLVM version: 20.1.5",
    ] {
        if !rustc.lines().any(|observed| observed == line) {
            return Err(format!("rustc -Vv missing registered line {line:?}"));
        }
    }
    let cargo = command_stdout("cargo", &["-V"])?;
    if cargo.trim() != "cargo 1.88.0 (873a06493 2025-05-10)" {
        return Err(format!("cargo identity mismatch: {cargo:?}"));
    }
    Ok(json!({
        "rustc_vv": rustc,
        "cargo_v": cargo.trim(),
    }))
}

fn command_stdout(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("{program} {args:?} launch: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("{program} {args:?} non-UTF8: {error}"))
        .map(|value| value.trim().to_string())
}

fn write_new_json(path: &Path, value: &Value) -> Result<hash::FileSeal, String> {
    let cwd = env::current_dir().map_err(|error| format!("current_dir: {error}"))?;
    let absolute = absolute_output_path(path, &cwd)?;
    let invalid = invalid_output_path(&absolute)?;
    let mut payload =
        serde_json::to_vec_pretty(value).map_err(|error| format!("JSON serialize: {error}"))?;
    payload.push(b'\n');
    let repeated =
        serde_json::to_vec_pretty(value).map_err(|error| format!("repeat serialize: {error}"))?;
    if repeated.as_slice() != &payload[..payload.len() - 1] {
        return Err("in-process JSON serialization was not byte-identical".to_string());
    }
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&absolute)
        .map_err(|error| format!("create-new {}: {error}", absolute.display()))?;
    if let Err(error) = output
        .write_all(&payload)
        .and_then(|_| output.flush())
        .and_then(|_| output.sync_all())
    {
        drop(output);
        let retained = fs::rename(&absolute, &invalid)
            .map(|_| invalid.display().to_string())
            .unwrap_or_else(|rename_error| format!("rename also failed: {rename_error}"));
        return Err(format!(
            "output write failed: {error}; partial retained as {retained}"
        ));
    }
    drop(output);
    let seal = match hash::seal_file(&absolute) {
        Ok(seal) => seal,
        Err(error) => {
            let _ = fs::rename(&absolute, &invalid);
            return Err(format!(
                "output seal failed: {error}; partial moved to {}",
                invalid.display()
            ));
        }
    };
    let reread = fs::read(&absolute).map_err(|error| {
        let _ = fs::rename(&absolute, &invalid);
        format!(
            "output reread failed: {error}; moved to {}",
            invalid.display()
        )
    })?;
    if reread != payload {
        fs::rename(&absolute, &invalid).map_err(|error| {
            format!(
                "output reread mismatch and failed to retain {}: {error}",
                invalid.display()
            )
        })?;
        return Err(format!(
            "output reread mismatch; retained as {}",
            invalid.display()
        ));
    }
    Ok(seal)
}

fn unix_millis() -> Result<u128, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .map_err(|error| format!("system time before UNIX epoch: {error}"))
}

#[cfg(target_os = "windows")]
fn process_peak_working_set() -> Option<u64> {
    #[repr(C)]
    struct ProcessMemoryCountersEx {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut core::ffi::c_void,
            counters: *mut ProcessMemoryCountersEx,
            size: u32,
        ) -> i32;
    }
    let mut counters = ProcessMemoryCountersEx {
        cb: std::mem::size_of::<ProcessMemoryCountersEx>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
        private_usage: 0,
    };
    // SAFETY: both functions are called with the current pseudo-handle and a
    // correctly sized writable PROCESS_MEMORY_COUNTERS_EX buffer.
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    (ok != 0).then_some(counters.peak_working_set_size as u64)
}

#[cfg(not(target_os = "windows"))]
fn process_peak_working_set() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_run(
        ordinal: usize,
        side: Stone,
        statuses: [&'static str; 5],
        expansions: [u64; 5],
    ) -> RootRun {
        let checkpoints = CAPS
            .into_iter()
            .zip(statuses)
            .zip(expansions)
            .map(|((cap, status), expansions)| CheckpointRecord {
                cap,
                status,
                expansions,
                scientific: json!({
                    "fingerprint_collisions": "0",
                    "collision_entries": "0",
                    "exact_alias_errors": "0",
                }),
                diagnostic: Value::Null,
            })
            .collect::<Vec<_>>();
        RootRun {
            ordinal,
            uid: format!("ROOT-{ordinal}"),
            side,
            checkpoints,
            certificate: None,
            scientific: Value::Null,
        }
    }

    #[test]
    fn oracle_uses_actual_cost_and_lexicographically_skips_earlier_optional_root() {
        let runs = vec![
            fake_run(
                0,
                Stone::Black,
                [
                    "ProvenWin",
                    "ProvenWin",
                    "ProvenWin",
                    "ProvenWin",
                    "ProvenWin",
                ],
                [10, 10, 10, 10, 10],
            ),
            fake_run(
                1,
                Stone::White,
                [
                    "UnknownNodeBudget",
                    "UnknownNodeBudget",
                    "UnknownNodeBudget",
                    "UnknownNodeBudget",
                    "ProvenWin",
                ],
                [5, 10, 12, 15, 20],
            ),
            fake_run(
                2,
                Stone::Black,
                [
                    "UnknownNodeBudget",
                    "UnknownNodeBudget",
                    "UnknownNodeBudget",
                    "UnknownNodeBudget",
                    "ProvenWin",
                ],
                [5, 10, 12, 15, 20],
            ),
        ];
        let oracle = perfect_oracle(&runs).unwrap();
        assert_eq!(oracle.reference_budget, 40);
        assert_eq!(oracle.reference_proofs, 1);
        assert_eq!(oracle.added_proofs, 1);
        assert_eq!(oracle.assigned_cost, 30);
        assert_eq!(oracle.assigned_caps, vec![1_024, 0, 262_144]);
    }

    #[test]
    fn history_replay_rejects_move_after_completed_five() {
        let moves = [
            (0, 0, "B"),
            (0, 1, "W"),
            (1, 0, "B"),
            (1, 1, "W"),
            (2, 0, "B"),
            (2, 1, "W"),
            (3, 0, "B"),
            (3, 1, "W"),
            (4, 0, "B"),
            (4, 1, "W"),
        ];
        let history = moves
            .into_iter()
            .map(|(x, y, color)| json!({"x": x, "y": y, "color": color}))
            .collect::<Vec<_>>();
        let error = match replay_history(&history, 1) {
            Ok(_) => panic!("post-terminal history unexpectedly accepted"),
            Err(error) => error,
        };
        assert!(error.contains("after terminal position"), "{error}");
    }

    #[test]
    fn projector_materializes_only_registered_fields() {
        let line = br#"{
            "format":"rq547-tactical-position-v1",
            "source_path":"a",
            "game_id":7,
            "ply":0,
            "side_to_move":"B",
            "position_history":[],
            "rapfi_score":{"deep":[1,2,{"secret":"not projected"}]},
            "game_result":"forbidden"
        }"#;
        let projected = project_allowed_root_fields(line).unwrap();
        assert_eq!(projected.len(), 6);
        assert!(!projected.contains_key("rapfi_score"));
        assert!(!projected.contains_key("game_result"));
    }

    #[test]
    fn git_safe_directory_strips_windows_verbatim_prefix() {
        assert_eq!(
            git_safe_directory(Path::new(r"\\?\C:\Users\concreate\workspace\figrid-board")),
            "C:/Users/concreate/workspace/figrid-board"
        );
    }

    #[test]
    fn complete_scan_does_not_depend_on_last_move() {
        let mut board = Board::new();
        for cell in [0, 15, 1, 16, 2, 17, 3, 18, 4] {
            board.make_move(cell);
        }
        board.last_move = None;
        assert_eq!(complete_winners(&board), (true, false));
    }
}
