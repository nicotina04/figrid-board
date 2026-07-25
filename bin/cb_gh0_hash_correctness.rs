//! Authoritative CB-GH0 P1-H exact D4 hash correctness harness.
//!
//! This binary deliberately contains no timing. It implements only the
//! frozen correctness workload from
//! `experiments/2026-07-25/cb_gh0_p1h_correctness_amendment.md`.

use figrid_board::board::{
    BOARD_SIZE, BitBoard, Board, BoardSearchState, Move, NUM_CELLS, RuleSet, Stone,
};
use figrid_board::d4_hash::{
    D4_COMPOSE, D4_INVERSE, D4_MAP, D4HashState, canonical_context_from_hashes,
    exact_canonical_state,
};
use figrid_board::pattern_table::{pack_window, read_window};
use figrid_board::{GOMOKU_NNUE_CONFIG, SearchResult, Searcher, to_idx};
use noru::network::NnueWeights;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const FORMAT: &str = "cb-gh0-p1h-hash-correctness-v1";
const STATUS_PASS: &str = "OPEN_GH0_HASH_COST_GATE";
const STATUS_FAIL: &str = "REJECT_GH0_HASH_EXACTNESS";
const TRANSFORMS: usize = 8;
const TRANSITIONS: u64 = 100_000;
const RULE_PERIOD: u64 = 251;
const D4_RELATION_PERIOD: u64 = 97;
const MAX_STONES: usize = 180;
const PRNG_SEED: u64 = 0xCB60_2026_0725_0001;
const SPLITMIX_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_MUL1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_MUL2: u64 = 0x94D0_49BB_1331_11EB;
const SIDE_KEY_SEED: u64 = 0xCAFE_BABE_DEAD_BEEF;
const RULE_KEY_SEEDS: [u64; 4] = [
    0xD4C0_0000_0000_0000,
    0xD4C0_0000_0000_0001,
    0xD4C0_0000_0000_0002,
    0xD4C0_0000_0000_0003,
];
const FROZEN_INVERSE: [u8; 8] = [0, 3, 2, 1, 4, 5, 6, 7];
const FROZEN_COMPOSE: [[u8; 8]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 0, 7, 6, 4, 5],
    [2, 3, 0, 1, 5, 4, 7, 6],
    [3, 0, 1, 2, 6, 7, 5, 4],
    [4, 6, 5, 7, 0, 2, 1, 3],
    [5, 7, 4, 6, 2, 0, 3, 1],
    [6, 5, 7, 4, 3, 1, 0, 2],
    [7, 4, 6, 5, 1, 3, 2, 0],
];
const FAILURE_CLASSES: [&str; 22] = [
    "source_identity",
    "protocol_counts",
    "map_bijection_or_composition",
    "rule_domain",
    "sidecar_access",
    "incremental_hash",
    "full_rebuild_hash",
    "canonical_context",
    "exact_canonical_state",
    "prediction_hash",
    "prediction_context",
    "true_hash_collision",
    "intra_orbit_hash_collision",
    "d4_relation",
    "d4_transform_semantics",
    "symmetry_fixture",
    "synthetic_tie",
    "default_off",
    "sidecar_composition",
    "fixed_depth_search",
    "unwind",
    "empty_undo",
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    out_report: PathBuf,
}

#[derive(Default)]
struct FailureClass {
    count: u64,
    first: Option<Value>,
}

struct Failures {
    classes: BTreeMap<&'static str, FailureClass>,
}

impl Failures {
    fn new() -> Self {
        let mut classes = BTreeMap::new();
        for name in FAILURE_CLASSES {
            classes.insert(name, FailureClass::default());
        }
        Self { classes }
    }

    fn add(&mut self, class: &'static str, witness: Value) {
        let entry = self
            .classes
            .get_mut(class)
            .unwrap_or_else(|| panic!("unregistered failure class {class}"));
        entry.count += 1;
        if entry.first.is_none() {
            entry.first = Some(witness);
        }
    }

    fn total(&self) -> u64 {
        self.classes.values().map(|entry| entry.count).sum()
    }

    fn report(&self) -> Value {
        let mut map = serde_json::Map::new();
        for (&name, entry) in &self.classes {
            map.insert(
                name.to_string(),
                json!({
                    "count": entry.count,
                    "first_witness": entry.first
                }),
            );
        }
        Value::Object(map)
    }
}

#[derive(Default)]
struct Counts {
    transitions: u64,
    makes: u64,
    undos: u64,
    prng_draws: u64,
    rule_switches: u64,
    registered_state_audits: u64,
    unwind_state_audits: u64,
    hash_lane_comparisons: u64,
    retained_vs_full_rebuild_lane_comparisons: u64,
    full_rebuilds: u64,
    canonical_context_checks: u64,
    exact_state_checks: u64,
    collision_observations: u64,
    d4_equivalent_repeats: u64,
    true_collisions: u64,
    intra_orbit_collisions: u64,
    prediction_checks: u64,
    prediction_hash_lane_comparisons: u64,
    registered_d4_relation_states: u64,
    fixture_d4_relation_states: u64,
    transformed_boards: u64,
    d4_relation_pairs: u64,
    mapped_move_roundtrips: u64,
    map_bijection_checks: u64,
    composition_checks: u64,
    rule_gate_switches: u64,
    named_symmetry_fixtures: u64,
    synthetic_tie_checks: u64,
    default_off_checks: u64,
    composition_transitions: u64,
    search_smoke_cases: u64,
    unwind_undos: u64,
    empty_undo_checks: u64,
}

impl Counts {
    fn report(&self) -> Value {
        json!({
            "transition_tape": {
                "transitions": self.transitions,
                "makes": self.makes,
                "undos": self.undos,
                "prng_draws": self.prng_draws,
                "rule_switches": self.rule_switches
            },
            "state_correctness": {
                "registered_state_audits": self.registered_state_audits,
                "unwind_state_audits": self.unwind_state_audits,
                "hash_lane_comparisons": self.hash_lane_comparisons,
                "retained_vs_full_rebuild_lane_comparisons":
                    self.retained_vs_full_rebuild_lane_comparisons,
                "full_rebuilds": self.full_rebuilds,
                "canonical_context_checks": self.canonical_context_checks,
                "exact_state_checks": self.exact_state_checks,
                "collision_observations": self.collision_observations,
                "predictions": self.prediction_checks,
                "prediction_hash_lane_comparisons": self.prediction_hash_lane_comparisons
            },
            "collisions": {
                "d4_equivalent_repeats": self.d4_equivalent_repeats,
                "true_hash_collisions": self.true_collisions,
                "intra_orbit_hash_collisions": self.intra_orbit_collisions
            },
            "d4": {
                "registered_relation_states": self.registered_d4_relation_states,
                "fixture_relation_states": self.fixture_d4_relation_states,
                "transformed_boards": self.transformed_boards,
                "relation_pairs": self.d4_relation_pairs,
                "mapped_move_roundtrips": self.mapped_move_roundtrips,
                "map_bijection_checks": self.map_bijection_checks,
                "composition_checks": self.composition_checks
            },
            "focused_gates": {
                "rule_gate_switches": self.rule_gate_switches,
                "named_symmetry_fixtures": self.named_symmetry_fixtures,
                "synthetic_tie_checks": self.synthetic_tie_checks,
                "default_off_checks": self.default_off_checks,
                "sidecar_composition_transitions": self.composition_transitions,
                "fixed_depth_search_cases": self.search_smoke_cases,
                "unwind_undos": self.unwind_undos,
                "empty_undo_checks": self.empty_undo_checks
            }
        })
    }
}

struct SplitMix64 {
    state: u64,
    draws: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            draws: 0,
        }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX_INCREMENT);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MUL1);
        z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MUL2);
        self.draws += 1;
        z ^ (z >> 31)
    }
}

struct CollisionAudit {
    by_key: BTreeMap<u64, [u8; 66]>,
}

impl CollisionAudit {
    fn new() -> Self {
        Self {
            by_key: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Copy)]
struct IndependentContext {
    key: u64,
    transform: u8,
}

#[derive(Clone)]
struct NamedFixture {
    name: &'static str,
    board: Board,
    expected_stabilizers: Vec<u8>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("CB-GH0 P1-H INVALID: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args_from(env::args().skip(1))?;
    refuse_existing(&args.out_report)?;

    let source = source_identity()?;
    let executable = executable_identity()?;
    let mut failures = Failures::new();
    let mut counts = Counts::default();
    if source["tracked_worktree_dirty_at_execution"] != Value::Bool(false) {
        failures.add(
            "source_identity",
            json!({
                "kind": "tracked_worktree_dirty",
                "observed": source["tracked_worktree_dirty_at_execution"]
            }),
        );
    }

    audit_registered_maps(&mut failures, &mut counts);
    let rule_report = audit_rule_domains(&mut failures, &mut counts);
    let fixture_report = audit_named_fixtures(&mut failures, &mut counts);
    let default_off_report = audit_default_off_and_composition(&mut failures, &mut counts);
    let search_report = audit_fixed_depth_search(&mut failures, &mut counts);
    let tape_report = audit_transition_tape(&mut failures, &mut counts);
    let protocol_count_report = audit_protocol_counts(&mut failures, &counts);

    let status = if failures.total() == 0 {
        STATUS_PASS
    } else {
        STATUS_FAIL
    };
    let report = json!({
        "format": FORMAT,
        "status": status,
        "claim_boundary": {
            "authoritative_stage": "CB-GH0 P1-H correctness only",
            "default_off_hash_sidecar_only": true,
            "canonical_tt_score_bound_move_sharing": false,
            "proof_cache_identity_authorized_by_u64_alone": false,
            "timing_collected": false,
            "p2_cost_gate_opened": status == STATUS_PASS,
            "product_promotion_opened": false,
            "benchmark_or_arena_opened": false
        },
        "source": source,
        "executable": executable,
        "registered_constants": registered_constants(),
        "counts": counts.report(),
        "rule_and_legacy_gate": rule_report,
        "named_symmetry_fixtures": fixture_report,
        "default_off_and_composition": default_off_report,
        "fixed_depth_search_smoke": search_report,
        "transition_tape": tape_report,
        "protocol_count_gate": protocol_count_report,
        "failures": failures.report(),
        "decision": {
            "status": status,
            "total_failures": failures.total(),
            "true_hash_collisions": counts.true_collisions,
            "intra_orbit_hash_collisions": counts.intra_orbit_collisions,
            "p2_hash_maintenance_cost_opened": status == STATUS_PASS,
            "tt_score_bound_branch_remains_blocked": true
        }
    });
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("failed to serialize report: {error}"))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.out_report)
        .map_err(|error| {
            format!(
                "failed to create report {}: {error}",
                args.out_report.display()
            )
        })?;
    output
        .write_all(&bytes)
        .and_then(|_| output.write_all(b"\n"))
        .map_err(|error| {
            format!(
                "failed to write report {}: {error}",
                args.out_report.display()
            )
        })?;
    println!(
        "CB-GH0 P1-H {status}: transitions={} failures={}",
        counts.transitions,
        failures.total()
    );
    Ok(())
}

fn parse_args_from<I, S>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = args.into_iter().map(Into::into);
    let mut out_report = None;
    while let Some(option) = args.next() {
        if option != "--out-report" {
            return Err(format!(
                "unknown or forbidden option {option:?}\n{}",
                usage()
            ));
        }
        if out_report.is_some() {
            return Err(format!("duplicate option --out-report\n{}", usage()));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for --out-report\n{}", usage()))?;
        if value.starts_with("--") {
            return Err(format!(
                "missing value for --out-report before option-like token {value:?}\n{}",
                usage()
            ));
        }
        out_report = Some(PathBuf::from(value));
    }
    Ok(Args {
        out_report: out_report
            .ok_or_else(|| format!("missing required --out-report\n{}", usage()))?,
    })
}

fn usage() -> &'static str {
    "usage: cb-gh0-hash-correctness --out-report NEW.json"
}

fn refuse_existing(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Err(format!("refusing to overwrite {}", path.display()));
    }
    Ok(())
}

fn registered_constants() -> Value {
    let rule_keys: Vec<String> = RULE_KEY_SEEDS
        .iter()
        .map(|&seed| format!("{:016X}", splitmix_once(seed)))
        .collect();
    json!({
        "target_cpu_contract": "x86-64-v3",
        "transition_tape": {
            "seed_hex": format!("{PRNG_SEED:016X}"),
            "transitions": TRANSITIONS,
            "max_stones_before_forced_undo": MAX_STONES,
            "undo_decision_mask": 3,
            "legal_move_order": "Board::legal_moves ascending",
            "rule_period": RULE_PERIOD,
            "rule_cycle": ["Standard", "Caro", "Renju", "legacy Standard", "Freestyle"],
            "d4_relation_period": D4_RELATION_PERIOD
        },
        "splitmix64": {
            "increment_hex": format!("{SPLITMIX_INCREMENT:016X}"),
            "multiplier_1_hex": format!("{SPLITMIX_MUL1:016X}"),
            "multiplier_2_hex": format!("{SPLITMIX_MUL2:016X}")
        },
        "hash": {
            "side_key_seed_hex": format!("{SIDE_KEY_SEED:016X}"),
            "side_key_hex": format!("{:016X}", independent_side_key()),
            "rule_key_seed_hex": RULE_KEY_SEEDS.map(|seed| format!("{seed:016X}")),
            "rule_key_hex": rule_keys,
            "stone_seed": "(color * 0x9E3779B97F4A7C15) XOR cell",
            "canonical_tie_rule": "lowest transform index"
        },
        "d4": {
            "coordinate_maps": [
                "(r,c)", "(c,14-r)", "(14-r,14-c)", "(14-c,r)",
                "(r,14-c)", "(14-r,c)", "(c,r)", "(14-c,14-r)"
            ],
            "inverse": FROZEN_INVERSE,
            "composition": FROZEN_COMPOSE
        },
        "exact_state": {
            "bytes": 66,
            "layout": [
                "black.lo:u128:big-endian", "black.hi:u128:big-endian",
                "white.lo:u128:big-endian", "white.hi:u128:big-endian",
                "side:Black=0,White=1",
                "effective_rule:Freestyle=0,Standard=1,Caro=2,Renju=3"
            ],
            "canonical_rule": "lexicographic minimum over all 8 transforms, lowest-index tie"
        },
        "search_smoke": {
            "cases": 3,
            "fixed_depth": 2,
            "weights": "NnueWeights::zeros(GOMOKU_NNUE_CONFIG)",
            "time_limit": null,
            "root_vct": "structurally disabled per Searcher arm"
        }
    })
}

fn source_identity() -> Result<Value, String> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let safe = format!(
        "safe.directory={}",
        manifest.to_string_lossy().replace('\\', "/")
    );
    let invoke = |args: &[&str]| -> Result<String, String> {
        let output = Command::new("git")
            .arg("-c")
            .arg(&safe)
            .args(args)
            .current_dir(manifest)
            .output()
            .map_err(|error| format!("failed to invoke git: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "git {:?} failed: {}",
                args,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        String::from_utf8(output.stdout)
            .map(|text| text.trim().to_string())
            .map_err(|error| format!("git output was not UTF-8: {error}"))
    };
    let head = invoke(&["rev-parse", "HEAD"])?;
    if head.len() != 40 || !head.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("unexpected git HEAD {head:?}"));
    }
    let critical_paths = [
        "Cargo.toml",
        "Cargo.lock",
        "src/lib.rs",
        "src/board.rs",
        "src/search.rs",
        "src/d4_hash.rs",
        "bin/cb_gh0_hash_correctness.rs",
        "experiments/2026-07-25/cb_gh0_exact_d4_hash_preregister.md",
        "experiments/2026-07-25/cb_gh0_p1h_correctness_amendment.md",
    ];
    let mut critical_sources = Vec::with_capacity(critical_paths.len());
    for relative in critical_paths {
        invoke(&["ls-files", "--error-unmatch", "--", relative]).map_err(|error| {
            format!("critical source is not tracked in git HEAD ({relative}): {error}")
        })?;
        let path = manifest.join(relative);
        let bytes = fs::read(&path)
            .map_err(|error| format!("failed to read critical source {relative}: {error}"))?;
        critical_sources.push(json!({
            "path": relative,
            "bytes": bytes.len(),
            "sha256": sha256_hex(&bytes)
        }));
    }
    let tracked_status = invoke(&["status", "--porcelain", "--untracked-files=no"])?;
    Ok(json!({
        "git_head": head,
        "tracked_worktree_dirty_at_execution": !tracked_status.is_empty(),
        "critical_sources_tracked_in_head": true,
        "critical_source_seals": critical_sources,
        "crate_name": env!("CARGO_PKG_NAME"),
        "crate_version": env!("CARGO_PKG_VERSION"),
        "binary": "cb-gh0-hash-correctness"
    }))
}

fn executable_identity() -> Result<Value, String> {
    let path = env::current_exe().map_err(|error| format!("current_exe failed: {error}"))?;
    let bytes = fs::read(&path).map_err(|error| {
        format!(
            "failed to read current executable {}: {error}",
            path.display()
        )
    })?;
    let file_name = path
        .file_name()
        .ok_or_else(|| "current executable has no file name".to_string())?
        .to_string_lossy()
        .into_owned();
    Ok(json!({
        "file_name": file_name,
        "bytes": bytes.len(),
        "sha256": sha256_hex(&bytes)
    }))
}

#[inline]
fn splitmix_once(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(SPLITMIX_INCREMENT);
    z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MUL1);
    z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MUL2);
    z ^ (z >> 31)
}

#[inline]
fn independent_side_key() -> u64 {
    splitmix_once(SIDE_KEY_SEED)
}

#[inline]
fn independent_rule_key(rule: RuleSet) -> u64 {
    splitmix_once(RULE_KEY_SEEDS[rule_tag(rule) as usize])
}

#[inline]
fn independent_stone_key(stone: Stone, cell: usize) -> u64 {
    let color = match stone {
        Stone::Black => 0u64,
        Stone::White => 1u64,
    };
    splitmix_once(color.wrapping_mul(SPLITMIX_INCREMENT) ^ cell as u64)
}

#[inline]
fn independent_map(transform: usize, cell: usize) -> usize {
    let row = cell / BOARD_SIZE;
    let col = cell % BOARD_SIZE;
    let last = BOARD_SIZE - 1;
    let (mapped_row, mapped_col) = match transform {
        0 => (row, col),
        1 => (col, last - row),
        2 => (last - row, last - col),
        3 => (last - col, row),
        4 => (row, last - col),
        5 => (last - row, col),
        6 => (col, row),
        7 => (last - col, last - row),
        _ => panic!("transform outside 0..8"),
    };
    mapped_row * BOARD_SIZE + mapped_col
}

fn independent_hashes(board: &Board) -> [u64; 8] {
    let domain = independent_rule_key(board.effective_rule_set())
        ^ if board.side_to_move == Stone::White {
            independent_side_key()
        } else {
            0
        };
    let mut hashes = [domain; 8];
    for cell in 0..NUM_CELLS {
        if board.black.get(cell) {
            for (transform, hash) in hashes.iter_mut().enumerate() {
                *hash ^= independent_stone_key(Stone::Black, independent_map(transform, cell));
            }
        }
        if board.white.get(cell) {
            for (transform, hash) in hashes.iter_mut().enumerate() {
                *hash ^= independent_stone_key(Stone::White, independent_map(transform, cell));
            }
        }
    }
    hashes
}

fn independent_predicted_child_hashes(board: &Board) -> impl Fn(Move) -> [u64; 8] + '_ {
    let hashes = independent_hashes(board);
    move |mv| {
        let mut child = hashes;
        for (transform, hash) in child.iter_mut().enumerate() {
            *hash ^= independent_stone_key(board.side_to_move, independent_map(transform, mv))
                ^ independent_side_key();
        }
        child
    }
}

fn independent_context(hashes: &[u64; 8]) -> IndependentContext {
    let mut key = hashes[0];
    let mut transform = 0u8;
    for (index, &candidate) in hashes.iter().enumerate().skip(1) {
        if candidate < key {
            key = candidate;
            transform = index as u8;
        }
    }
    IndependentContext { key, transform }
}

fn transformed_bitboard(source: &BitBoard, transform: usize) -> BitBoard {
    let mut result = BitBoard::EMPTY;
    for cell in 0..NUM_CELLS {
        if source.get(cell) {
            result.set(independent_map(transform, cell));
        }
    }
    result
}

fn independent_exact_transformed(board: &Board, transform: usize) -> [u8; 66] {
    let black = transformed_bitboard(&board.black, transform);
    let white = transformed_bitboard(&board.white, transform);
    let mut bytes = [0u8; 66];
    bytes[0..16].copy_from_slice(&black.lo.to_be_bytes());
    bytes[16..32].copy_from_slice(&black.hi.to_be_bytes());
    bytes[32..48].copy_from_slice(&white.lo.to_be_bytes());
    bytes[48..64].copy_from_slice(&white.hi.to_be_bytes());
    bytes[64] = side_tag(board.side_to_move);
    bytes[65] = rule_tag(board.effective_rule_set());
    bytes
}

fn independent_exact_canonical(board: &Board) -> ([u8; 66], u8) {
    let mut bytes = independent_exact_transformed(board, 0);
    let mut transform = 0u8;
    for index in 1..8 {
        let candidate = independent_exact_transformed(board, index);
        if candidate < bytes {
            bytes = candidate;
            transform = index as u8;
        }
    }
    (bytes, transform)
}

#[inline]
const fn side_tag(side: Stone) -> u8 {
    match side {
        Stone::Black => 0,
        Stone::White => 1,
    }
}

#[inline]
const fn rule_tag(rule: RuleSet) -> u8 {
    match rule {
        RuleSet::Freestyle => 0,
        RuleSet::Standard => 1,
        RuleSet::Caro => 2,
        RuleSet::Renju => 3,
    }
}

fn board_semantics_equal(left: &Board, right: &Board) -> bool {
    left.black == right.black
        && left.white == right.white
        && left.side_to_move == right.side_to_move
        && left.move_count == right.move_count
        && left.last_move == right.last_move
        && left.history == right.history
        && left.zobrist == right.zobrist
        && left.line_pattern_ids == right.line_pattern_ids
        && left.rule_set == right.rule_set
        && left.exact5 == right.exact5
}

fn audit_registered_maps(failures: &mut Failures, counts: &mut Counts) {
    if D4_INVERSE != FROZEN_INVERSE {
        failures.add(
            "map_bijection_or_composition",
            json!({
                "kind": "inverse_vector",
                "expected": FROZEN_INVERSE,
                "observed": D4_INVERSE
            }),
        );
    }
    if D4_COMPOSE != FROZEN_COMPOSE {
        failures.add(
            "map_bijection_or_composition",
            json!({
                "kind": "composition_table",
                "expected": FROZEN_COMPOSE,
                "observed": D4_COMPOSE
            }),
        );
    }

    for transform in 0..TRANSFORMS {
        let mut seen = [false; NUM_CELLS];
        for cell in 0..NUM_CELLS {
            counts.map_bijection_checks += 1;
            let observed = D4_MAP[transform][cell] as usize;
            let expected = independent_map(transform, cell);
            if observed >= NUM_CELLS || observed != expected || seen[observed] {
                failures.add(
                    "map_bijection_or_composition",
                    json!({
                        "kind": "map_bijection",
                        "transform": transform,
                        "cell": cell,
                        "expected": expected,
                        "observed": observed,
                        "duplicate": observed < NUM_CELLS && seen[observed]
                    }),
                );
            } else {
                seen[observed] = true;
            }
            let roundtrip =
                D4_MAP[D4_INVERSE[transform] as usize][D4_MAP[transform][cell] as usize] as usize;
            if roundtrip != cell {
                failures.add(
                    "map_bijection_or_composition",
                    json!({
                        "kind": "inverse_roundtrip",
                        "transform": transform,
                        "cell": cell,
                        "observed": roundtrip
                    }),
                );
            }
        }
        if seen.iter().any(|&present| !present) {
            failures.add(
                "map_bijection_or_composition",
                json!({"kind": "map_not_surjective", "transform": transform}),
            );
        }
    }

    for a in 0..TRANSFORMS {
        for b in 0..TRANSFORMS {
            let composed = D4_COMPOSE[a][b] as usize;
            for cell in 0..NUM_CELLS {
                counts.composition_checks += 1;
                let observed = D4_MAP[a][D4_MAP[b][cell] as usize] as usize;
                let expected = D4_MAP[composed][cell] as usize;
                if observed != expected {
                    failures.add(
                        "map_bijection_or_composition",
                        json!({
                            "kind": "composition_semantics",
                            "a": a,
                            "b": b,
                            "cell": cell,
                            "composed": composed,
                            "expected": expected,
                            "observed": observed
                        }),
                    );
                }
            }
        }
    }
}

fn audit_state(
    board: &Board,
    state: &BoardSearchState,
    phase: &str,
    collision: &mut CollisionAudit,
    failures: &mut Failures,
    counts: &mut Counts,
    unwind: bool,
) {
    if unwind {
        counts.unwind_state_audits += 1;
    } else {
        counts.registered_state_audits += 1;
    }
    let expected_hashes = independent_hashes(board);
    let maintained = match state.d4_hashes(board) {
        Some(hashes) => hashes,
        None => {
            failures.add(
                "sidecar_access",
                json!({"phase": phase, "kind": "missing_synchronized_hashes"}),
            );
            [0; 8]
        }
    };
    counts.hash_lane_comparisons += 8;
    for transform in 0..8 {
        if maintained[transform] != expected_hashes[transform] {
            failures.add(
                "incremental_hash",
                json!({
                    "phase": phase,
                    "transform": transform,
                    "expected_hex": format!("{:016X}", expected_hashes[transform]),
                    "observed_hex": format!("{:016X}", maintained[transform])
                }),
            );
        }
    }

    let rebuilt = D4HashState::rebuild(board);
    counts.full_rebuilds += 1;
    for transform in 0..8 {
        counts.retained_vs_full_rebuild_lane_comparisons += 1;
        if maintained[transform] != rebuilt.hashes()[transform] {
            failures.add(
                "incremental_hash",
                json!({
                    "phase": phase,
                    "kind": "retained_vs_library_full_rebuild",
                    "transform": transform,
                    "retained_hex": format!("{:016X}", maintained[transform]),
                    "full_rebuild_hex": format!("{:016X}", rebuilt.hashes()[transform])
                }),
            );
        }
        if rebuilt.hashes()[transform] != expected_hashes[transform] {
            failures.add(
                "full_rebuild_hash",
                json!({
                    "phase": phase,
                    "transform": transform,
                    "expected_hex": format!("{:016X}", expected_hashes[transform]),
                    "observed_hex": format!("{:016X}", rebuilt.hashes()[transform])
                }),
            );
        }
    }

    let expected_context = independent_context(&expected_hashes);
    counts.canonical_context_checks += 1;
    match state.d4_canonical_context(board) {
        Some(context)
            if context.key == expected_context.key
                && context.to_canonical == expected_context.transform => {}
        Some(context) => failures.add(
            "canonical_context",
            json!({
                "phase": phase,
                "expected_key_hex": format!("{:016X}", expected_context.key),
                "expected_transform": expected_context.transform,
                "observed_key_hex": format!("{:016X}", context.key),
                "observed_transform": context.to_canonical
            }),
        ),
        None => failures.add(
            "sidecar_access",
            json!({"phase": phase, "kind": "missing_synchronized_context"}),
        ),
    }

    let (exact_bytes, exact_transform) = independent_exact_canonical(board);
    let library_exact = exact_canonical_state(board);
    counts.exact_state_checks += 1;
    if library_exact.bytes != exact_bytes || library_exact.to_canonical != exact_transform {
        failures.add(
            "exact_canonical_state",
            json!({
                "phase": phase,
                "expected_transform": exact_transform,
                "observed_transform": library_exact.to_canonical,
                "expected_hex": hex_bytes(&exact_bytes),
                "observed_hex": hex_bytes(&library_exact.bytes)
            }),
        );
    }

    counts.collision_observations += 1;
    match collision.by_key.get(&expected_context.key) {
        None => {
            collision.by_key.insert(expected_context.key, exact_bytes);
        }
        Some(previous) if previous == &exact_bytes => {
            counts.d4_equivalent_repeats += 1;
        }
        Some(previous) => {
            counts.true_collisions += 1;
            failures.add(
                "true_hash_collision",
                json!({
                    "phase": phase,
                    "canonical_key_hex": format!("{:016X}", expected_context.key),
                    "first_exact_hex": hex_bytes(previous),
                    "current_exact_hex": hex_bytes(&exact_bytes)
                }),
            );
        }
    }

    let min_key = expected_context.key;
    let tied: Vec<usize> = expected_hashes
        .iter()
        .enumerate()
        .filter_map(|(transform, &key)| (key == min_key).then_some(transform))
        .collect();
    for left_index in 0..tied.len() {
        for right_index in (left_index + 1)..tied.len() {
            let left = independent_exact_transformed(board, tied[left_index]);
            let right = independent_exact_transformed(board, tied[right_index]);
            if left != right {
                counts.intra_orbit_collisions += 1;
                failures.add(
                    "intra_orbit_hash_collision",
                    json!({
                        "phase": phase,
                        "canonical_key_hex": format!("{min_key:016X}"),
                        "left_transform": tied[left_index],
                        "right_transform": tied[right_index],
                        "left_exact_hex": hex_bytes(&left),
                        "right_exact_hex": hex_bytes(&right)
                    }),
                );
            }
        }
    }
}

fn transformed_board(source: &Board, transform: usize) -> Board {
    let mut result = Board::new();
    result.set_rule_set(source.effective_rule_set());
    for &mv in &source.history {
        result.make_move(independent_map(transform, mv));
    }
    result
}

fn audit_d4_relation(
    board: &Board,
    phase: &str,
    registered: bool,
    failures: &mut Failures,
    counts: &mut Counts,
) {
    if registered {
        counts.registered_d4_relation_states += 1;
    } else {
        counts.fixture_d4_relation_states += 1;
    }
    let source_hashes = D4HashState::rebuild(board);
    let source_context = source_hashes.canonical_context();
    let (source_exact, _) = independent_exact_canonical(board);

    for g in 0..TRANSFORMS {
        let transformed = transformed_board(board, g);
        counts.transformed_boards += 1;

        let expected_black = transformed_bitboard(&board.black, g);
        let expected_white = transformed_bitboard(&board.white, g);
        if transformed.black != expected_black
            || transformed.white != expected_white
            || transformed.side_to_move != board.side_to_move
            || transformed.effective_rule_set() != board.effective_rule_set()
        {
            failures.add(
                "d4_transform_semantics",
                json!({
                    "phase": phase,
                    "g": g,
                    "black_matches": transformed.black == expected_black,
                    "white_matches": transformed.white == expected_white,
                    "source_side": side_tag(board.side_to_move),
                    "transformed_side": side_tag(transformed.side_to_move),
                    "source_rule": rule_tag(board.effective_rule_set()),
                    "transformed_rule": rule_tag(transformed.effective_rule_set())
                }),
            );
        }

        let transformed_hashes = D4HashState::rebuild(&transformed);
        for t in 0..TRANSFORMS {
            counts.d4_relation_pairs += 1;
            let composed = D4_COMPOSE[t][g] as usize;
            let expected = source_hashes.hashes()[composed];
            let observed = transformed_hashes.hashes()[t];
            if observed != expected {
                failures.add(
                    "d4_relation",
                    json!({
                        "phase": phase,
                        "g": g,
                        "t": t,
                        "composed": composed,
                        "expected_hex": format!("{expected:016X}"),
                        "observed_hex": format!("{observed:016X}")
                    }),
                );
            }
        }
        let transformed_context = transformed_hashes.canonical_context();
        if transformed_context.key != source_context.key {
            failures.add(
                "d4_relation",
                json!({
                    "phase": phase,
                    "kind": "canonical_key",
                    "g": g,
                    "source_hex": format!("{:016X}", source_context.key),
                    "transformed_hex": format!("{:016X}", transformed_context.key)
                }),
            );
        }
        let (transformed_exact, _) = independent_exact_canonical(&transformed);
        if transformed_exact != source_exact {
            failures.add(
                "d4_relation",
                json!({
                    "phase": phase,
                    "kind": "exact_canonical_bytes",
                    "g": g,
                    "source_hex": hex_bytes(&source_exact),
                    "transformed_hex": hex_bytes(&transformed_exact)
                }),
            );
        }

        for mv in 0..NUM_CELLS {
            counts.mapped_move_roundtrips += 1;
            let mapped = D4_MAP[g][mv] as usize;
            let roundtrip = D4_MAP[D4_INVERSE[g] as usize][mapped] as usize;
            if roundtrip != mv {
                failures.add(
                    "d4_relation",
                    json!({
                        "phase": phase,
                        "kind": "mapped_move_roundtrip",
                        "g": g,
                        "move": mv,
                        "mapped": mapped,
                        "roundtrip": roundtrip
                    }),
                );
            }
        }
    }
}

fn orbit(cell: Move) -> Vec<Move> {
    let mut cells = BTreeSet::new();
    for transform in 0..TRANSFORMS {
        cells.insert(independent_map(transform, cell));
    }
    cells.into_iter().collect()
}

fn interleaved_board(black: &[Move], white: &[Move]) -> Board {
    assert!(black.len() == white.len() || black.len() == white.len() + 1);
    let mut board = Board::new();
    for index in 0..black.len() {
        board.make_move(black[index]);
        if let Some(&white_move) = white.get(index) {
            board.make_move(white_move);
        }
    }
    board
}

fn named_fixtures() -> Vec<NamedFixture> {
    let empty = Board::new();
    let mut center = Board::new();
    center.make_move(to_idx(7, 7));

    let full_black = orbit(to_idx(2, 5));
    let full_white = orbit(to_idx(1, 3));
    assert_eq!(full_black.len(), 8);
    assert_eq!(full_white.len(), 8);
    let full_d4 = interleaved_board(&full_black, &full_white);

    let vertical = interleaved_board(
        &[to_idx(2, 3), to_idx(2, 11), to_idx(5, 1), to_idx(5, 13)],
        &[to_idx(3, 4), to_idx(3, 10), to_idx(9, 2), to_idx(9, 12)],
    );
    let half_turn = interleaved_board(
        &[to_idx(2, 3), to_idx(12, 11), to_idx(4, 1), to_idx(10, 13)],
        &[to_idx(3, 6), to_idx(11, 8), to_idx(5, 2), to_idx(9, 12)],
    );
    let asymmetric = interleaved_board(
        &[to_idx(7, 7), to_idx(5, 9), to_idx(11, 2), to_idx(4, 13)],
        &[to_idx(2, 3), to_idx(8, 6), to_idx(12, 10)],
    );

    Vec::from([
        NamedFixture {
            name: "empty_board",
            board: empty,
            expected_stabilizers: (0u8..8).collect(),
        },
        NamedFixture {
            name: "one_center_stone",
            board: center,
            expected_stabilizers: (0u8..8).collect(),
        },
        NamedFixture {
            name: "full_d4_symmetry",
            board: full_d4,
            expected_stabilizers: (0u8..8).collect(),
        },
        NamedFixture {
            name: "vertical_reflection_only_symmetry",
            board: vertical,
            expected_stabilizers: vec![0, 4],
        },
        NamedFixture {
            name: "half_turn_180_only_symmetry",
            board: half_turn,
            expected_stabilizers: vec![0, 2],
        },
        NamedFixture {
            name: "asymmetric_state",
            board: asymmetric,
            expected_stabilizers: vec![0],
        },
    ])
}

fn audit_named_fixtures(failures: &mut Failures, counts: &mut Counts) -> Value {
    let mut reports = Vec::new();
    for fixture in named_fixtures() {
        counts.named_symmetry_fixtures += 1;
        let identity = independent_exact_transformed(&fixture.board, 0);
        let observed_stabilizers: Vec<u8> = (0..TRANSFORMS)
            .filter(|&transform| {
                independent_exact_transformed(&fixture.board, transform) == identity
            })
            .map(|transform| transform as u8)
            .collect();
        let passed = observed_stabilizers == fixture.expected_stabilizers;
        if !passed {
            failures.add(
                "symmetry_fixture",
                json!({
                    "fixture": fixture.name,
                    "expected_stabilizers": fixture.expected_stabilizers,
                    "observed_stabilizers": observed_stabilizers
                }),
            );
        }
        audit_d4_relation(&fixture.board, fixture.name, false, failures, counts);
        reports.push(json!({
            "name": fixture.name,
            "move_count": fixture.board.move_count,
            "expected_stabilizers": fixture.expected_stabilizers,
            "observed_stabilizers": observed_stabilizers,
            "passed": passed
        }));
    }

    let synthetic = [9u64, 3, 3, 7, 8, 5, 6, 4];
    let synthetic_context = independent_context(&synthetic);
    let production_context = canonical_context_from_hashes(&synthetic);
    counts.synthetic_tie_checks += 1;
    let tie_passed = synthetic_context.key == 3
        && synthetic_context.transform == 1
        && production_context.key == 3
        && production_context.to_canonical == 1;
    if !tie_passed {
        failures.add(
            "synthetic_tie",
            json!({
                "hashes": synthetic,
                "expected_key": 3,
                "expected_transform": 1,
                "observed_key": synthetic_context.key,
                "observed_transform": synthetic_context.transform,
                "production_key": production_context.key,
                "production_transform": production_context.to_canonical
            }),
        );
    }
    json!({
        "fixtures": reports,
        "synthetic_equal_minimum_hash_tie": {
            "hashes": synthetic,
            "expected_key": 3,
            "expected_transform": 1,
            "observed_key": synthetic_context.key,
            "observed_transform": synthetic_context.transform,
            "production_key": production_context.key,
            "production_transform": production_context.to_canonical,
            "passed": tie_passed
        }
    })
}

fn set_rule_mode(board: &mut Board, mode: usize) -> &'static str {
    match mode % 5 {
        0 => {
            board.set_rule_set(RuleSet::Standard);
            "Standard"
        }
        1 => {
            board.set_rule_set(RuleSet::Caro);
            "Caro"
        }
        2 => {
            board.set_rule_set(RuleSet::Renju);
            "Renju"
        }
        3 => {
            board.set_rule_set(RuleSet::Freestyle);
            board.exact5 = true;
            "legacy Standard"
        }
        4 => {
            board.set_rule_set(RuleSet::Freestyle);
            "Freestyle"
        }
        _ => unreachable!(),
    }
}

fn audit_rule_domains(failures: &mut Failures, counts: &mut Counts) -> Value {
    let rules = [
        RuleSet::Freestyle,
        RuleSet::Standard,
        RuleSet::Caro,
        RuleSet::Renju,
    ];
    let mut empty_keys = Vec::new();
    for rule in rules {
        let mut board = Board::new();
        board.set_rule_set(rule);
        let hashes = D4HashState::rebuild(&board);
        let expected = independent_rule_key(rule);
        let all_equal = hashes.hashes().iter().all(|&key| key == expected);
        if !all_equal {
            failures.add(
                "rule_domain",
                json!({
                    "kind": "empty_rule_key",
                    "rule": rule_tag(rule),
                    "expected_hex": format!("{expected:016X}"),
                    "observed_hex": hashes.hashes().map(|key| format!("{key:016X}"))
                }),
            );
        }
        empty_keys.push(expected);
    }
    let distinct = empty_keys.iter().copied().collect::<BTreeSet<_>>().len() == 4;
    if !distinct {
        failures.add(
            "rule_domain",
            json!({"kind": "empty_rule_keys_not_distinct", "keys": empty_keys}),
        );
    }

    let mut board = interleaved_board(
        &[to_idx(7, 7), to_idx(6, 8), to_idx(9, 5), to_idx(3, 12)],
        &[to_idx(2, 2), to_idx(8, 7), to_idx(10, 11), to_idx(1, 4)],
    );
    let mut state = BoardSearchState::new();
    state.set_d4_hash_enabled(&board, true);
    let mut named_hashes = BTreeMap::<String, [u64; 8]>::new();
    for (name, rule) in [
        ("Freestyle", RuleSet::Freestyle),
        ("Standard", RuleSet::Standard),
        ("Caro", RuleSet::Caro),
        ("Renju", RuleSet::Renju),
    ] {
        board.set_rule_set(rule);
        state.synchronize(&board);
        counts.rule_gate_switches += 1;
        let observed = state.d4_hashes(&board).unwrap_or([0; 8]);
        let expected = independent_hashes(&board);
        if observed != expected {
            failures.add(
                "rule_domain",
                json!({
                    "kind": "nonempty_formal_rule",
                    "rule": name,
                    "expected": expected.map(|key| format!("{key:016X}")),
                    "observed": observed.map(|key| format!("{key:016X}"))
                }),
            );
        }
        named_hashes.insert(name.to_string(), observed);
    }

    board.set_rule_set(RuleSet::Freestyle);
    board.exact5 = true;
    state.synchronize(&board);
    counts.rule_gate_switches += 1;
    let legacy_standard = state.d4_hashes(&board).unwrap_or([0; 8]);
    let formal_standard = named_hashes["Standard"];
    let legacy_matches =
        legacy_standard == formal_standard && board.effective_rule_set() == RuleSet::Standard;
    if !legacy_matches {
        failures.add(
            "rule_domain",
            json!({
                "kind": "legacy_standard_equivalence",
                "formal": formal_standard.map(|key| format!("{key:016X}")),
                "legacy": legacy_standard.map(|key| format!("{key:016X}")),
                "legacy_effective_rule": rule_tag(board.effective_rule_set())
            }),
        );
    }

    board.set_rule_set(RuleSet::Freestyle);
    state.synchronize(&board);
    counts.rule_gate_switches += 1;
    let restored = state.d4_hashes(&board).unwrap_or([0; 8]);
    let freestyle = named_hashes["Freestyle"];
    let reset_passed =
        !board.exact5 && board.effective_rule_set() == RuleSet::Freestyle && restored == freestyle;
    if !reset_passed {
        failures.add(
            "rule_domain",
            json!({
                "kind": "freestyle_reset",
                "exact5": board.exact5,
                "effective_rule": rule_tag(board.effective_rule_set()),
                "expected": freestyle.map(|key| format!("{key:016X}")),
                "observed": restored.map(|key| format!("{key:016X}"))
            }),
        );
    }

    json!({
        "empty_rule_keys_hex": empty_keys
            .iter()
            .map(|key| format!("{key:016X}"))
            .collect::<Vec<_>>(),
        "empty_rule_keys_pairwise_distinct": distinct,
        "nonempty_formal_rules_checked": 4,
        "legacy_standard_matches_formal_standard": legacy_matches,
        "return_to_freestyle_clears_exact5_and_restores_hashes": reset_passed
    })
}

fn assert_packed_windows(
    board: &Board,
    state: &BoardSearchState,
    phase: &str,
    failures: &mut Failures,
) -> bool {
    const DIRECTIONS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
    let mut passed = true;
    for cell in 0..NUM_CELLS {
        let row = (cell / BOARD_SIZE) as i32;
        let col = (cell % BOARD_SIZE) as i32;
        for (direction, &(dr, dc)) in DIRECTIONS.iter().enumerate() {
            let expected = pack_window(&read_window(&board.black, &board.white, row, col, dr, dc));
            let observed = state.packed_line_window(cell, direction);
            if observed != Some(expected) {
                passed = false;
                failures.add(
                    "sidecar_composition",
                    json!({
                        "phase": phase,
                        "kind": "packed_window",
                        "cell": cell,
                        "direction": direction,
                        "expected": expected,
                        "observed": observed
                    }),
                );
            }
        }
    }
    passed
}

fn audit_default_off_and_composition(failures: &mut Failures, counts: &mut Counts) -> Value {
    let mut checks = BTreeMap::<String, bool>::new();

    let mut ordinary_board = Board::new();
    let mut ordinary_state = BoardSearchState::new();
    let fresh_sidecar_off =
        !ordinary_state.d4_hash_enabled() && ordinary_state.d4_hashes(&ordinary_board).is_none();
    counts.default_off_checks += 1;
    checks.insert("fresh_board_search_state_d4_off".into(), fresh_sidecar_off);
    if !fresh_sidecar_off {
        failures.add(
            "default_off",
            json!({"kind": "fresh_board_search_state_d4_off"}),
        );
    }

    ordinary_board.make_move(to_idx(7, 7));
    ordinary_board.undo_move();
    ordinary_state.synchronize(&ordinary_board);
    let ordinary_make_undo_did_not_enable =
        !ordinary_state.d4_hash_enabled() && ordinary_state.d4_hashes(&ordinary_board).is_none();
    counts.default_off_checks += 1;
    checks.insert(
        "ordinary_board_make_undo_does_not_enable_sidecar".into(),
        ordinary_make_undo_did_not_enable,
    );
    if !ordinary_make_undo_did_not_enable {
        failures.add(
            "default_off",
            json!({"kind": "ordinary_board_make_undo_enabled_sidecar"}),
        );
    }

    let searcher = Searcher::new();
    let searcher_default_off = !searcher.d4_hash_sidecar_requested();
    counts.default_off_checks += 1;
    checks.insert("fresh_searcher_selector_off".into(), searcher_default_off);
    if !searcher_default_off {
        failures.add("default_off", json!({"kind": "fresh_searcher_selector_on"}));
    }

    // This exhaustive literal is intentionally not replaced by `..template`;
    // compilation is the compatibility gate for the public Board shape.
    let template = Board::new();
    let literal = Board {
        black: template.black,
        white: template.white,
        side_to_move: template.side_to_move,
        move_count: template.move_count,
        last_move: template.last_move,
        history: template.history.clone(),
        zobrist: template.zobrist,
        line_pattern_ids: template.line_pattern_ids.clone(),
        rule_set: template.rule_set,
        exact5: template.exact5,
    };
    let literal_passed = board_semantics_equal(&literal, &template);
    counts.default_off_checks += 1;
    checks.insert("public_exhaustive_board_literal".into(), literal_passed);
    if !literal_passed {
        failures.add(
            "default_off",
            json!({"kind": "public_exhaustive_board_literal_semantics"}),
        );
    }

    let sequence = [
        to_idx(7, 7),
        to_idx(0, 0),
        to_idx(14, 14),
        to_idx(6, 8),
        to_idx(2, 12),
        to_idx(11, 3),
        to_idx(4, 9),
        to_idx(12, 6),
    ];
    let mut composed_board = Board::new();
    let mut composed_state = BoardSearchState::new();
    composed_state.set_packed_line_windows_enabled(&composed_board, true);
    composed_state.set_candidate_frontier_enabled(&composed_board, true);
    composed_state.set_d4_hash_enabled(&composed_board, true);
    let all_enabled = composed_state.packed_line_windows_enabled()
        && composed_state.candidate_frontier_enabled()
        && composed_state.d4_hash_enabled()
        && composed_state.is_synchronized(&composed_board);
    if !all_enabled {
        failures.add(
            "sidecar_composition",
            json!({"phase": "root", "kind": "all_three_not_enabled"}),
        );
    }
    let root_packed_passed = assert_packed_windows(
        &composed_board,
        &composed_state,
        "composition_root",
        failures,
    );
    let mut composition_passed = all_enabled && root_packed_passed;
    for (index, &mv) in sequence.iter().enumerate() {
        composed_state.make_move(&mut composed_board, mv);
        counts.composition_transitions += 1;
        let expected_hashes = independent_hashes(&composed_board);
        let synchronized = composed_state.is_synchronized(&composed_board)
            && composed_state.packed_line_windows_enabled()
            && composed_state.candidate_frontier_enabled()
            && composed_state.d4_hash_enabled()
            && composed_state.d4_hashes(&composed_board) == Some(expected_hashes)
            && composed_state.candidate_moves(&composed_board) == composed_board.candidate_moves();
        composition_passed &= synchronized;
        if !synchronized {
            failures.add(
                "sidecar_composition",
                json!({"phase": format!("make_{}", index + 1), "move": mv}),
            );
        }
        composition_passed &= assert_packed_windows(
            &composed_board,
            &composed_state,
            &format!("composition_make_{}", index + 1),
            failures,
        );
    }
    for index in (0..sequence.len()).rev() {
        composed_state.undo_move(&mut composed_board);
        counts.composition_transitions += 1;
        let expected_hashes = independent_hashes(&composed_board);
        let synchronized = composed_state.is_synchronized(&composed_board)
            && composed_state.packed_line_windows_enabled()
            && composed_state.candidate_frontier_enabled()
            && composed_state.d4_hash_enabled()
            && composed_state.d4_hashes(&composed_board) == Some(expected_hashes)
            && composed_state.candidate_moves(&composed_board) == composed_board.candidate_moves();
        composition_passed &= synchronized;
        if !synchronized {
            failures.add(
                "sidecar_composition",
                json!({"phase": format!("undo_{index}")}),
            );
        }
        composition_passed &= assert_packed_windows(
            &composed_board,
            &composed_state,
            &format!("composition_undo_{index}"),
            failures,
        );
    }
    checks.insert(
        "packed_candidate_d4_composition_make_undo".into(),
        composition_passed,
    );

    json!({
        "checks": checks,
        "composition_sequence": sequence,
        "composition_transitions": counts.composition_transitions,
        "all_passed": checks.values().all(|&passed| passed) && composition_passed
    })
}

fn search_roots() -> Vec<(&'static str, Board)> {
    let empty = Board::new();
    let sparse = interleaved_board(
        &[to_idx(7, 7), to_idx(8, 8), to_idx(7, 8)],
        &[to_idx(3, 3), to_idx(3, 4), to_idx(4, 4)],
    );
    let edge = interleaved_board(
        &[to_idx(0, 0), to_idx(1, 1), to_idx(2, 1), to_idx(4, 2)],
        &[
            to_idx(14, 14),
            to_idx(13, 13),
            to_idx(12, 13),
            to_idx(10, 12),
        ],
    );
    vec![("empty", empty), ("sparse", sparse), ("edge", edge)]
}

fn result_equal(left: &SearchResult, right: &SearchResult) -> bool {
    left.best_move == right.best_move
        && left.score == right.score
        && left.depth == right.depth
        && left.nodes == right.nodes
}

fn result_json(result: &SearchResult) -> Value {
    json!({
        "best_move": result.best_move,
        "score": result.score,
        "completed_depth": result.depth,
        "nodes": result.nodes
    })
}

fn audit_fixed_depth_search(failures: &mut Failures, counts: &mut Counts) -> Value {
    let weights = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);
    let mut cases = Vec::new();
    for (name, root) in search_roots() {
        counts.search_smoke_cases += 1;
        let expected_root = root.clone();
        let mut off_board = root.clone();
        let mut on_board = root;

        let mut off = Searcher::new();
        let off_default = !off.d4_hash_sidecar_requested();
        let off_vct_default = off.root_vct_requested_for_audit();
        off.set_use_root_vct_for_audit(false);
        let off_result = off.search(&mut off_board, &weights, 2, None);
        let mut on = Searcher::new();
        let on_vct_default = on.root_vct_requested_for_audit();
        on.set_use_d4_hash_sidecar(true);
        on.set_use_root_vct_for_audit(false);
        let on_result = on.search(&mut on_board, &weights, 2, None);

        let result_match = result_equal(&off_result, &on_result);
        let final_boards_match = board_semantics_equal(&off_board, &expected_root)
            && board_semantics_equal(&on_board, &expected_root)
            && board_semantics_equal(&off_board, &on_board);
        let vct_disabled =
            !off.root_vct_requested_for_audit() && !on.root_vct_requested_for_audit();
        let passed = off_default
            && off_vct_default
            && on_vct_default
            && on.d4_hash_sidecar_requested()
            && vct_disabled
            && result_match
            && final_boards_match;
        if !passed {
            failures.add(
                "fixed_depth_search",
                json!({
                    "case": name,
                    "off_default": off_default,
                    "off_root_vct_default": off_vct_default,
                    "on_root_vct_default": on_vct_default,
                    "on_requested": on.d4_hash_sidecar_requested(),
                    "root_vct_disabled_both_arms": vct_disabled,
                    "off_result": result_json(&off_result),
                    "on_result": result_json(&on_result),
                    "result_match": result_match,
                    "final_boards_match": final_boards_match
                }),
            );
        }
        cases.push(json!({
            "name": name,
            "depth": 2,
            "off_result": result_json(&off_result),
            "on_result": result_json(&on_result),
            "off_root_vct_default": off_vct_default,
            "on_root_vct_default": on_vct_default,
            "result_match": result_match,
            "final_boards_match": final_boards_match,
            "root_vct_disabled_both_arms": vct_disabled,
            "passed": passed
        }));
    }
    json!({
        "weights": "all-zero NnueWeights",
        "depth": 2,
        "time_limit": null,
        "root_vct": "disabled per Searcher arm",
        "cases": cases,
        "all_passed": cases.iter().all(|case| case["passed"] == Value::Bool(true))
    })
}

fn audit_prediction_before_make(
    board: &Board,
    state: &BoardSearchState,
    mv: Move,
    transition: u64,
    failures: &mut Failures,
    counts: &mut Counts,
) -> ([u64; 8], IndependentContext) {
    counts.prediction_checks += 1;
    let expected_hashes = independent_predicted_child_hashes(board)(mv);
    let expected_context = independent_context(&expected_hashes);
    match state.d4_predicted_child_hashes(board, mv) {
        Some(observed) => {
            counts.prediction_hash_lane_comparisons += 8;
            for transform in 0..8 {
                if observed[transform] != expected_hashes[transform] {
                    failures.add(
                        "prediction_hash",
                        json!({
                            "transition": transition,
                            "move": mv,
                            "phase": "before_make",
                            "transform": transform,
                            "expected_hex": format!("{:016X}", expected_hashes[transform]),
                            "observed_hex": format!("{:016X}", observed[transform])
                        }),
                    );
                }
            }
        }
        None => failures.add(
            "prediction_hash",
            json!({
                "transition": transition,
                "move": mv,
                "phase": "before_make_missing"
            }),
        ),
    }
    match state.d4_predicted_child_context(board, mv) {
        Some(observed)
            if observed.key == expected_context.key
                && observed.to_canonical == expected_context.transform => {}
        Some(observed) => failures.add(
            "prediction_context",
            json!({
                "transition": transition,
                "move": mv,
                "phase": "before_make",
                "expected_key_hex": format!("{:016X}", expected_context.key),
                "expected_transform": expected_context.transform,
                "observed_key_hex": format!("{:016X}", observed.key),
                "observed_transform": observed.to_canonical
            }),
        ),
        None => failures.add(
            "prediction_context",
            json!({
                "transition": transition,
                "move": mv,
                "phase": "before_make_missing"
            }),
        ),
    }
    (expected_hashes, expected_context)
}

fn audit_prediction_after_make(
    board: &Board,
    state: &BoardSearchState,
    mv: Move,
    transition: u64,
    expected_hashes: [u64; 8],
    expected_context: IndependentContext,
    failures: &mut Failures,
    counts: &mut Counts,
) {
    match state.d4_hashes(board) {
        Some(observed) => {
            counts.prediction_hash_lane_comparisons += 8;
            for transform in 0..8 {
                if observed[transform] != expected_hashes[transform] {
                    failures.add(
                        "prediction_hash",
                        json!({
                            "transition": transition,
                            "move": mv,
                            "phase": "after_make",
                            "transform": transform,
                            "expected_hex": format!("{:016X}", expected_hashes[transform]),
                            "observed_hex": format!("{:016X}", observed[transform])
                        }),
                    );
                }
            }
        }
        None => failures.add(
            "prediction_hash",
            json!({
                "transition": transition,
                "move": mv,
                "phase": "after_make_missing"
            }),
        ),
    }
    match state.d4_canonical_context(board) {
        Some(observed)
            if observed.key == expected_context.key
                && observed.to_canonical == expected_context.transform => {}
        Some(observed) => failures.add(
            "prediction_context",
            json!({
                "transition": transition,
                "move": mv,
                "phase": "after_make",
                "expected_key_hex": format!("{:016X}", expected_context.key),
                "expected_transform": expected_context.transform,
                "observed_key_hex": format!("{:016X}", observed.key),
                "observed_transform": observed.to_canonical
            }),
        ),
        None => failures.add(
            "prediction_context",
            json!({
                "transition": transition,
                "move": mv,
                "phase": "after_make_missing"
            }),
        ),
    }
}

fn audit_transition_tape(failures: &mut Failures, counts: &mut Counts) -> Value {
    let mut board = Board::new();
    let mut state = BoardSearchState::new();
    state.set_d4_hash_enabled(&board, true);
    let mut collision = CollisionAudit::new();
    let mut rng = SplitMix64::new(PRNG_SEED);
    let mut maximum_move_count = 0usize;
    let mut rule_cycle_counts = BTreeMap::<String, u64>::new();

    audit_state(
        &board,
        &state,
        "initial",
        &mut collision,
        failures,
        counts,
        false,
    );
    audit_d4_relation(&board, "initial", true, failures, counts);

    for transition in 1..=TRANSITIONS {
        if transition % RULE_PERIOD == 0 {
            let ordinal = counts.rule_switches as usize;
            let name = set_rule_mode(&mut board, ordinal);
            *rule_cycle_counts.entry(name.to_string()).or_default() += 1;
            counts.rule_switches += 1;
            state.synchronize(&board);
            audit_state(
                &board,
                &state,
                &format!("rule_switch_before_transition_{transition}_{name}"),
                &mut collision,
                failures,
                counts,
                false,
            );
        }

        let decision = rng.next();
        let should_undo =
            !board.history.is_empty() && (board.move_count >= MAX_STONES || (decision & 3) == 0);
        if should_undo {
            state.undo_move(&mut board);
            counts.undos += 1;
        } else {
            let legal = board.legal_moves();
            let pick = rng.next();
            let mv = legal[(pick as usize) % legal.len()];
            let (predicted_hashes, predicted_context) =
                audit_prediction_before_make(&board, &state, mv, transition, failures, counts);
            state.make_move(&mut board, mv);
            counts.makes += 1;
            audit_prediction_after_make(
                &board,
                &state,
                mv,
                transition,
                predicted_hashes,
                predicted_context,
                failures,
                counts,
            );
        }
        counts.transitions += 1;
        maximum_move_count = maximum_move_count.max(board.move_count);
        audit_state(
            &board,
            &state,
            &format!("post_transition_{transition}"),
            &mut collision,
            failures,
            counts,
            false,
        );
        if transition % D4_RELATION_PERIOD == 0 || transition == TRANSITIONS {
            audit_d4_relation(
                &board,
                &format!("post_transition_{transition}"),
                true,
                failures,
                counts,
            );
        }
    }
    counts.prng_draws = rng.draws;

    let move_count_before_unwind = board.move_count;
    while !board.history.is_empty() {
        state.undo_move(&mut board);
        counts.unwind_undos += 1;
        audit_state(
            &board,
            &state,
            &format!("unwind_after_undo_{}", counts.unwind_undos),
            &mut collision,
            failures,
            counts,
            true,
        );
    }
    board.set_rule_set(RuleSet::Freestyle);
    state.synchronize(&board);
    audit_state(
        &board,
        &state,
        "final_freestyle_empty",
        &mut collision,
        failures,
        counts,
        true,
    );

    let fresh = Board::new();
    let mut fresh_state = BoardSearchState::new();
    fresh_state.set_d4_hash_enabled(&fresh, true);
    let final_hashes = state.d4_hashes(&board);
    let fresh_hashes = fresh_state.d4_hashes(&fresh);
    let final_context = state.d4_canonical_context(&board);
    let fresh_context = fresh_state.d4_canonical_context(&fresh);
    let final_exact = independent_exact_canonical(&board);
    let fresh_exact = independent_exact_canonical(&fresh);
    let unwind_passed = board_semantics_equal(&board, &fresh)
        && final_hashes == fresh_hashes
        && final_context == fresh_context
        && final_exact == fresh_exact;
    if !unwind_passed {
        failures.add(
            "unwind",
            json!({
                "move_count_before_unwind": move_count_before_unwind,
                "final_move_count": board.move_count,
                "board_matches_fresh": board_semantics_equal(&board, &fresh),
                "hashes_match_fresh": final_hashes == fresh_hashes,
                "context_matches_fresh": final_context == fresh_context,
                "exact_matches_fresh": final_exact == fresh_exact
            }),
        );
    }

    let before_empty_undo = board.clone();
    let hashes_before_empty_undo = state.d4_hashes(&board);
    let context_before_empty_undo = state.d4_canonical_context(&board);
    let exact_before_empty_undo = independent_exact_canonical(&board);
    state.undo_move(&mut board);
    counts.empty_undo_checks += 1;
    let empty_undo_passed = board_semantics_equal(&board, &before_empty_undo)
        && state.d4_hashes(&board) == hashes_before_empty_undo
        && state.d4_canonical_context(&board) == context_before_empty_undo
        && independent_exact_canonical(&board) == exact_before_empty_undo
        && state.is_synchronized(&board);
    if !empty_undo_passed {
        failures.add(
            "empty_undo",
            json!({
                "board_unchanged": board_semantics_equal(&board, &before_empty_undo),
                "hashes_unchanged": state.d4_hashes(&board) == hashes_before_empty_undo,
                "context_unchanged": state.d4_canonical_context(&board)
                    == context_before_empty_undo,
                "exact_unchanged": independent_exact_canonical(&board)
                    == exact_before_empty_undo,
                "synchronized": state.is_synchronized(&board)
            }),
        );
    }

    json!({
        "seed_hex": format!("{PRNG_SEED:016X}"),
        "final_prng_state_hex": format!("{:016X}", rng.state),
        "prng_draws": rng.draws,
        "transitions": counts.transitions,
        "makes": counts.makes,
        "undos": counts.undos,
        "maximum_move_count": maximum_move_count,
        "final_move_count_before_unwind": move_count_before_unwind,
        "rule_switches": counts.rule_switches,
        "rule_cycle_counts": rule_cycle_counts,
        "collision_map_unique_keys": collision.by_key.len(),
        "d4_equivalent_repeats": counts.d4_equivalent_repeats,
        "true_hash_collisions": counts.true_collisions,
        "intra_orbit_hash_collisions": counts.intra_orbit_collisions,
        "complete_unwind_to_fresh_freestyle_empty": unwind_passed,
        "empty_undo_noop": empty_undo_passed
    })
}

fn audit_protocol_counts(failures: &mut Failures, counts: &Counts) -> Value {
    const EXPECTED_RULE_SWITCHES: u64 = TRANSITIONS / RULE_PERIOD;
    const EXPECTED_REGISTERED_STATE_AUDITS: u64 = 1 + TRANSITIONS + EXPECTED_RULE_SWITCHES;
    const EXPECTED_RELATION_STATES: u64 = 1 + (TRANSITIONS / D4_RELATION_PERIOD) + 1;
    const EXPECTED_FIXTURES: u64 = 6;
    const EXPECTED_MAKES: u64 = 50_090;
    const EXPECTED_UNDOS: u64 = 49_910;
    const EXPECTED_UNWIND_UNDOS: u64 = 180;
    let total_state_audits = counts.registered_state_audits + counts.unwind_state_audits;
    let total_relation_states =
        counts.registered_d4_relation_states + counts.fixture_d4_relation_states;

    let checks = [
        ("transitions", counts.transitions, TRANSITIONS),
        ("makes_plus_undos", counts.makes + counts.undos, TRANSITIONS),
        ("frozen_tape_makes", counts.makes, EXPECTED_MAKES),
        ("frozen_tape_undos", counts.undos, EXPECTED_UNDOS),
        (
            "prng_draws_equal_transitions_plus_makes",
            counts.prng_draws,
            counts.transitions + counts.makes,
        ),
        (
            "frozen_tape_prng_draws",
            counts.prng_draws,
            TRANSITIONS + EXPECTED_MAKES,
        ),
        (
            "rule_switches",
            counts.rule_switches,
            EXPECTED_RULE_SWITCHES,
        ),
        (
            "registered_state_audits",
            counts.registered_state_audits,
            EXPECTED_REGISTERED_STATE_AUDITS,
        ),
        ("unwind_undos", counts.unwind_undos, EXPECTED_UNWIND_UNDOS),
        (
            "unwind_state_audits",
            counts.unwind_state_audits,
            EXPECTED_UNWIND_UNDOS + 1,
        ),
        (
            "hash_lane_comparisons",
            counts.hash_lane_comparisons,
            total_state_audits * TRANSFORMS as u64,
        ),
        ("full_rebuilds", counts.full_rebuilds, total_state_audits),
        (
            "canonical_context_checks",
            counts.canonical_context_checks,
            total_state_audits,
        ),
        (
            "exact_state_checks",
            counts.exact_state_checks,
            total_state_audits,
        ),
        (
            "collision_observations",
            counts.collision_observations,
            total_state_audits,
        ),
        (
            "registered_d4_relation_states",
            counts.registered_d4_relation_states,
            EXPECTED_RELATION_STATES,
        ),
        (
            "named_symmetry_fixtures",
            counts.named_symmetry_fixtures,
            EXPECTED_FIXTURES,
        ),
        (
            "fixture_d4_relation_states",
            counts.fixture_d4_relation_states,
            EXPECTED_FIXTURES,
        ),
        (
            "predictions_equal_makes",
            counts.prediction_checks,
            counts.makes,
        ),
        (
            "prediction_hash_lane_comparisons",
            counts.prediction_hash_lane_comparisons,
            counts.makes * TRANSFORMS as u64 * 2,
        ),
        (
            "transformed_boards",
            counts.transformed_boards,
            total_relation_states * TRANSFORMS as u64,
        ),
        (
            "d4_relation_pairs",
            counts.d4_relation_pairs,
            counts.transformed_boards * TRANSFORMS as u64,
        ),
        (
            "mapped_move_roundtrips",
            counts.mapped_move_roundtrips,
            counts.transformed_boards * NUM_CELLS as u64,
        ),
        (
            "map_bijection_checks",
            counts.map_bijection_checks,
            (TRANSFORMS * NUM_CELLS) as u64,
        ),
        (
            "composition_checks",
            counts.composition_checks,
            (TRANSFORMS * TRANSFORMS * NUM_CELLS) as u64,
        ),
        ("true_hash_collisions", counts.true_collisions, 0),
        (
            "intra_orbit_hash_collisions",
            counts.intra_orbit_collisions,
            0,
        ),
        ("rule_gate_switches", counts.rule_gate_switches, 6),
        ("synthetic_tie_checks", counts.synthetic_tie_checks, 1),
        ("default_off_checks", counts.default_off_checks, 4),
        (
            "sidecar_composition_transitions",
            counts.composition_transitions,
            16,
        ),
        ("fixed_depth_search_cases", counts.search_smoke_cases, 3),
        ("empty_undo_checks", counts.empty_undo_checks, 1),
    ];
    let mut report = Vec::new();
    for (name, observed, expected) in checks {
        let passed = observed == expected;
        if !passed {
            failures.add(
                "protocol_counts",
                json!({
                    "name": name,
                    "expected": expected,
                    "observed": observed
                }),
            );
        }
        report.push(json!({
            "name": name,
            "expected": expected,
            "observed": observed,
            "passed": passed
        }));
    }
    json!({
        "checks": report,
        "all_passed": report.iter().all(|item| item["passed"] == Value::Bool(true))
    })
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02X}").expect("write to String cannot fail");
    }
    output
}

fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut schedule = [0u32; 64];
        for index in 0..16 {
            schedule[index] = u32::from_be_bytes(
                chunk[index * 4..index * 4 + 4]
                    .try_into()
                    .expect("four-byte SHA-256 word"),
            );
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let big1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(big1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let big0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = big0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
    state.iter().map(|word| format!("{word:08X}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_is_exactly_one_create_new_report_option() {
        assert_eq!(
            parse_args_from(["--out-report", "new.json"]).unwrap(),
            Args {
                out_report: PathBuf::from("new.json")
            }
        );
        assert!(parse_args_from(std::iter::empty::<&str>()).is_err());
        assert!(parse_args_from(["--output", "new.json"]).is_err());
        assert!(parse_args_from(["--out-report"]).is_err());
        assert!(parse_args_from(["--out-report", "--other"]).is_err());
        assert!(parse_args_from(["--out-report", "a", "--out-report", "b"]).is_err());
        assert!(parse_args_from(["new.json"]).is_err());
    }

    #[test]
    fn report_path_is_strictly_create_new() {
        let path = env::temp_dir().join(format!(
            "cb-gh0-p1h-create-new-test-{}.json",
            std::process::id()
        ));
        if path.exists() {
            fs::remove_file(&path).expect("remove stale exact test path");
        }
        assert!(refuse_existing(&path).is_ok());
        let _file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("first create_new succeeds");
        assert!(refuse_existing(&path).is_err());
        assert!(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .is_err()
        );
        fs::remove_file(path).expect("remove exact test path");
    }

    #[test]
    fn frozen_splitmix_tape_matches_known_prefix() {
        let mut rng = SplitMix64::new(PRNG_SEED);
        assert_eq!(rng.next(), 0x1DB4_7FC4_6A91_FD75);
        assert_eq!(rng.next(), 0xC1FA_45DC_C1A7_3DC0);
        assert_eq!(rng.next(), 0xABA5_D439_C554_1E62);
        assert_eq!(rng.next(), 0x0F2F_4152_1FE9_C54C);
        assert_eq!(rng.next(), 0x3116_5E6C_24DD_2F65);
        assert_eq!(rng.draws, 5);
        assert_eq!(rng.state, 0xE275_80C5_8399_6C6A);
    }

    #[test]
    fn independent_hash_and_exact_state_match_library_on_sample() {
        let mut board = Board::new();
        board.set_rule_set(RuleSet::Caro);
        for mv in [to_idx(7, 7), 0, 224, 17, 53, 190, 91] {
            board.make_move(mv);
        }
        assert_eq!(
            D4HashState::rebuild(&board).hashes(),
            &independent_hashes(&board)
        );
        let exact = exact_canonical_state(&board);
        let independent = independent_exact_canonical(&board);
        assert_eq!(exact.bytes, independent.0);
        assert_eq!(exact.to_canonical, independent.1);
    }

    #[test]
    fn named_fixture_stabilizers_are_exactly_registered() {
        let names: Vec<_> = named_fixtures()
            .into_iter()
            .map(|fixture| {
                let identity = independent_exact_transformed(&fixture.board, 0);
                let observed: Vec<u8> = (0..8)
                    .filter(|&transform| {
                        independent_exact_transformed(&fixture.board, transform) == identity
                    })
                    .map(|transform| transform as u8)
                    .collect();
                assert_eq!(observed, fixture.expected_stabilizers, "{}", fixture.name);
                fixture.name
            })
            .collect();
        assert_eq!(
            names,
            [
                "empty_board",
                "one_center_stone",
                "full_d4_symmetry",
                "vertical_reflection_only_symmetry",
                "half_turn_180_only_symmetry",
                "asymmetric_state"
            ]
        );
    }

    #[test]
    fn synthetic_equal_minimum_chooses_lower_index() {
        let context = independent_context(&[9, 3, 3, 7, 8, 5, 6, 4]);
        let production = canonical_context_from_hashes(&[9, 3, 3, 7, 8, 5, 6, 4]);
        assert_eq!(context.key, 3);
        assert_eq!(context.transform, 1);
        assert_eq!(production.key, 3);
        assert_eq!(production.to_canonical, 1);
        let empty_context = independent_context(&[5; 8]);
        let production_empty = canonical_context_from_hashes(&[5; 8]);
        assert_eq!(empty_context.transform, 0);
        assert_eq!(production_empty.to_canonical, 0);
    }

    #[test]
    fn legacy_standard_cycle_and_freestyle_reset_are_exact() {
        let mut board = Board::new();
        assert_eq!(set_rule_mode(&mut board, 0), "Standard");
        assert_eq!(board.effective_rule_set(), RuleSet::Standard);
        assert_eq!(set_rule_mode(&mut board, 1), "Caro");
        assert_eq!(board.effective_rule_set(), RuleSet::Caro);
        assert_eq!(set_rule_mode(&mut board, 2), "Renju");
        assert_eq!(board.effective_rule_set(), RuleSet::Renju);
        assert_eq!(set_rule_mode(&mut board, 3), "legacy Standard");
        assert_eq!(board.rule_set, RuleSet::Freestyle);
        assert!(board.exact5);
        assert_eq!(board.effective_rule_set(), RuleSet::Standard);
        assert_eq!(set_rule_mode(&mut board, 4), "Freestyle");
        assert_eq!(board.effective_rule_set(), RuleSet::Freestyle);
        assert!(!board.exact5);
    }

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
        );
    }

    #[test]
    fn sha256_matches_million_a_multiblock_vector() {
        let input = vec![b'a'; 1_000_000];
        assert_eq!(
            sha256_hex(&input),
            "CDC76E5C9914FB9281A1C7E284D73E67F1809A48A497200E046D39CCC7112CD0"
        );
    }
}
