//! Authoritative CB-GH0 P2-H exact D4 hash-maintenance cost harness.
//!
//! The protocol is frozen in
//! `experiments/2026-07-25/cb_gh0_p2h_cost_amendment.md`.
//! This binary deliberately exposes no workload-selection knobs.

use figrid_board::board::{
    BOARD_SIZE, Board, BoardSearchState, GameResult, Move, NUM_CELLS, RuleSet, Stone, to_idx,
};
use figrid_board::codebook_eval::QuantizedCodebookWeights;
use figrid_board::d4_hash::{D4HashState, ExactCanonicalState, exact_canonical_state};
use figrid_board::factored_codebook::{PackedCodebookArtifact, PackedCodebookKind};
use figrid_board::{GOMOKU_NNUE_CONFIG, SearchResult, SearchShapeStats, Searcher};
use noru::network::NnueWeights;
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const FORMAT: &str = "cb-gh0-p2h-hash-cost-v1";
const STATUS_INVALID: &str = "INVALID_CB_GH0_P2";
const STATUS_REJECT: &str = "REJECT_GH0_HASH_EXACTNESS";
const STATUS_TOO_COSTLY: &str = "HASH_CORRECT_BUT_TOO_COSTLY";
const STATUS_GO: &str = "GO_GH0_HASH_SIDECAR";

const CANONICAL_RUSTFLAGS: &str = "-C target-cpu=x86-64-v3";
const EXPECTED_GIT_PARENT: &str = "2ec6755";
const EXPECTED_RUSTC: &str = "rustc 1.88.0";
const EXPECTED_LLVM: &str = "LLVM version: 20.1.5";
const EXPECTED_HOST: &str = "host: x86_64-pc-windows-msvc";
const EXPECTED_CPU_VENDOR: &str = "AuthenticAMD";
const EXPECTED_CPU_FAMILY: u32 = 25;
const EXPECTED_CPU_MODEL: u32 = 97;
const EXPECTED_CPU_STEPPING: u32 = 2;
const EXPECTED_LOGICAL_PROCESSORS: usize = 16;

const TRANSITIONS: usize = 100_000;
const MAKES: usize = 50_090;
const UNDOS: usize = 49_910;
const PRNG_DRAWS: u64 = 150_090;
const RULE_SWITCHES: usize = 398;
const MAX_STONES: usize = 180;
const FINAL_MOVE_COUNT: usize = 180;
const TRANSITION_CLUSTERS: usize = 64;
const TRANSITION_REPETITIONS: usize = 8;
const RULE_PERIOD: usize = 251;
const TAPE_SEED: u64 = 0xCB60_2026_0725_0001;
const TAPE_FINAL_STATE: u64 = 0x840B_ED25_52B4_F013;
const SPLITMIX_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_MUL1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_MUL2: u64 = 0x94D0_49BB_1331_11EB;

const EXPECTED_GAMES: usize = 64;
const EXPECTED_ROOTS: usize = 1_022;
const SEARCH_DEPTH: u32 = 4;
const SEARCH_MIN_ROOTS_PER_GAME: usize = 8;
const SEARCH_MAX_ROOTS_PER_GAME: usize = 36;

const BOOTSTRAP_REPLICATES: usize = 100_000;
const BOOTSTRAP_DRAWS: usize = 64;
const BOOTSTRAP_INDEX: usize = 94_999;
const TRANSITION_BOOTSTRAP_SEED: u64 = 0xCB60_2026_0725_0201;
const SEARCH_BOOTSTRAP_SEED: u64 = 0xCB60_2026_0725_0202;

const TRANSITION_WARMUP_ORDER: [&str; 4] = ["A", "B", "B", "A"];
const MEASURED_ARM_ORDER: [&str; 4] = ["A1", "B1", "B2", "A2"];

const CRITICAL_SOURCES: [&str; 16] = [
    "Cargo.toml",
    "Cargo.lock",
    "src/lib.rs",
    "src/board.rs",
    "src/d4_hash.rs",
    "src/search.rs",
    "src/transposition.rs",
    "src/codebook_eval.rs",
    "src/token_delta.rs",
    "src/pattern_table.rs",
    "src/factored_codebook.rs",
    "bin/cb_gh0_hash_cost.rs",
    "experiments/2026-07-25/cb_gh0_exact_d4_hash_preregister.md",
    "experiments/2026-07-25/cb_gh0_p1h_correctness_amendment.md",
    "experiments/2026-07-25/cb_gh0_p1h_correctness_results.md",
    "experiments/2026-07-25/cb_gh0_p2h_cost_amendment.md",
];

#[derive(Clone, Copy)]
struct ArtifactSpec {
    name: &'static str,
    relative_path: &'static str,
    bytes: u64,
    sha256: &'static str,
}

const INPUT_SPECS: [ArtifactSpec; 5] = [
    ArtifactSpec {
        name: "raw_codebook_json",
        relative_path: "models/gomoku_codebook_v1_swapclosed.json",
        bytes: 1_410_562,
        sha256: "42968FDAB01BA8CCD1DE3DED05C532E4B237DD47EEFFD7AE1C2F264D77BA7DA2",
    },
    ArtifactSpec {
        name: "compact_flat_codebook",
        relative_path: "models/gomoku_codebook_v1_swapclosed_compact_flat.cbf",
        bytes: 417_412,
        sha256: "9A5E3D3FC47EEF79468F021F78E9130F5842764F579EE68A2FD270E8289B3250",
    },
    ArtifactSpec {
        name: "topk_vocabulary",
        relative_path: "data/topk.bin",
        bytes: 17_060,
        sha256: "103891DCD1DCD978C654593ABE78EF32C56E2E350B500EE665BC45AC051AA16D",
    },
    ArtifactSpec {
        name: "flat_nnue",
        relative_path: "models/gomoku_v52_5stone_conv_93k.bin",
        bytes: 14_960_159,
        sha256: "A961F378A3E73B3CF66C3D15B9A9AB857FA1B81123D98855EE04180A71EAFEFD",
    },
    ArtifactSpec {
        name: "frozen_trace",
        relative_path: "../figrid-dp-campaign/experiments/2026-07-25/dp_a1_fresh_holdout_64g.jsonl",
        bytes: 317_511,
        sha256: "1FD40D8948F113AD236FA44F5EEADCA1907C0C3103987CB4C704B67A9B47531A",
    },
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    out_report: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileSeal {
    name: String,
    path: String,
    bytes: u64,
    sha256: String,
}

impl FileSeal {
    fn report(&self) -> Value {
        json!({
            "name": self.name,
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceSeal {
    git_head: String,
    tracked_status: String,
    critical_sources: Vec<FileSeal>,
}

impl SourceSeal {
    fn report(&self) -> Value {
        json!({
            "git_head": self.git_head,
            "expected_parent_preregistration_commit": EXPECTED_GIT_PARENT,
            "tracked_worktree_clean": self.tracked_status.is_empty(),
            "tracked_status": self.tracked_status,
            "critical_sources_tracked_in_head": true,
            "critical_sources": self
                .critical_sources
                .iter()
                .map(FileSeal::report)
                .collect::<Vec<_>>(),
        })
    }
}

#[derive(Default)]
struct FailureLedger {
    invalid: BTreeMap<String, Value>,
    reject: BTreeMap<String, Value>,
}

impl FailureLedger {
    fn invalid(&mut self, class: &str, witness: Value) {
        self.invalid.entry(class.to_string()).or_insert(witness);
    }

    fn reject(&mut self, class: &str, witness: Value) {
        self.reject.entry(class.to_string()).or_insert(witness);
    }

    fn decision(&self, cost_passed: Option<bool>) -> &'static str {
        if !self.invalid.is_empty() {
            STATUS_INVALID
        } else if !self.reject.is_empty() {
            STATUS_REJECT
        } else if cost_passed == Some(true) {
            STATUS_GO
        } else {
            STATUS_TOO_COSTLY
        }
    }

    fn report(&self) -> Value {
        json!({
            "invalid_failure_classes": self.invalid,
            "exactness_failure_classes": self.reject,
            "invalid_failure_count": self.invalid.len(),
            "exactness_failure_count": self.reject.len(),
        })
    }
}

#[derive(Clone, Copy, Debug)]
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

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

struct FrozenInputs {
    seals: Vec<FileSeal>,
    raw_codebook: Vec<u8>,
    compact_codebook: Vec<u8>,
    topk: Vec<u8>,
    nnue: Vec<u8>,
    trace: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuleMode {
    Standard,
    Caro,
    Renju,
    LegacyStandard,
    Freestyle,
}

impl RuleMode {
    fn from_ordinal(ordinal: usize) -> Self {
        match ordinal % 5 {
            0 => Self::Standard,
            1 => Self::Caro,
            2 => Self::Renju,
            3 => Self::LegacyStandard,
            4 => Self::Freestyle,
            _ => unreachable!(),
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::Freestyle => 0,
            Self::Standard => 1,
            Self::Caro => 2,
            Self::Renju => 3,
            Self::LegacyStandard => 4,
        }
    }

    fn apply(self, board: &mut Board) {
        match self {
            Self::Standard => board.set_rule_set(RuleSet::Standard),
            Self::Caro => board.set_rule_set(RuleSet::Caro),
            Self::Renju => board.set_rule_set(RuleSet::Renju),
            Self::LegacyStandard => {
                board.set_rule_set(RuleSet::Freestyle);
                board.exact5 = true;
            }
            Self::Freestyle => board.set_rule_set(RuleSet::Freestyle),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TapeAction {
    Make,
    Undo,
}

impl TapeAction {
    fn tag(self) -> u8 {
        match self {
            Self::Make => 0,
            Self::Undo => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TapeEntry {
    index: usize,
    rule_before: Option<RuleMode>,
    action: TapeAction,
    mv: Move,
    color: Stone,
}

#[derive(Clone)]
struct BlockReference {
    start: usize,
    end: usize,
    board: Board,
    digest: String,
}

struct TransitionTape {
    entries: Vec<TapeEntry>,
    blocks: Vec<BlockReference>,
    serialized_sha256: String,
    makes: usize,
    undos: usize,
    rule_switches: usize,
    maximum_move_count: usize,
    final_move_count: usize,
    final_rng_state: u64,
    rng_draws: u64,
}

#[derive(Clone)]
struct TransitionRepetition {
    primary_cluster_ticks: Vec<u128>,
    initial_selector_ticks: u64,
    rule_rebuild_ticks: Vec<u64>,
    block_state_digests: Vec<String>,
    final_state_digest: String,
    unwind_state_digest: String,
}

impl TransitionRepetition {
    fn report(&self, repetition: usize) -> Result<Value, String> {
        let transition_total = checked_u128_sum(
            self.primary_cluster_ticks.iter().copied(),
            "transition report",
        )?;
        let rule_total = checked_tick_sum(
            self.rule_rebuild_ticks.iter().copied(),
            "rule rebuild report",
        )?;
        let rebuild_total = u128::from(self.initial_selector_ticks)
            .checked_add(rule_total)
            .ok_or_else(|| "initial+rule rebuild report tick overflow".to_string())?;
        let combined_total = transition_total
            .checked_add(rebuild_total)
            .ok_or_else(|| "combined lifecycle report tick overflow".to_string())?;
        Ok(json!({
            "repetition": repetition,
            "primary_transition_cluster_ticks": self.primary_cluster_ticks
                .iter()
                .copied()
                .map(u128_string)
                .collect::<Vec<_>>(),
            "initial_selector_rebuild_ticks": self.initial_selector_ticks,
            "rule_synchronization_rebuild_ticks": self.rule_rebuild_ticks,
            "block_state_digests": self.block_state_digests,
            "transition_only_total_ticks": u128_string(transition_total),
            "rebuild_only_total_ticks": u128_string(rebuild_total),
            "combined_lifecycle_total_ticks": u128_string(combined_total),
            "final_state_digest": self.final_state_digest,
            "unwind_state_digest": self.unwind_state_digest,
        }))
    }
}

struct TransitionArmRun {
    label: &'static str,
    d4_enabled: bool,
    repetitions: Vec<TransitionRepetition>,
}

impl TransitionArmRun {
    fn report(&self) -> Result<Value, String> {
        let repetitions = self
            .repetitions
            .iter()
            .enumerate()
            .map(|(index, repetition)| repetition.report(index))
            .collect::<Result<Vec<_>, _>>()?;
        let mut cluster_totals = Vec::with_capacity(TRANSITION_CLUSTERS);
        for cluster in 0..TRANSITION_CLUSTERS {
            let mut total = 0u128;
            for repetition in &self.repetitions {
                let value = repetition
                    .primary_cluster_ticks
                    .get(cluster)
                    .copied()
                    .ok_or_else(|| {
                        format!(
                            "arm {} repetition is missing transition cluster {cluster}",
                            self.label
                        )
                    })?;
                checked_add_assign(&mut total, value, "per-arm transition cluster total")?;
            }
            cluster_totals.push(u128_string(total));
        }
        Ok(json!({
            "arm": self.label,
            "d4_hash_sidecar": self.d4_enabled,
            "cluster_totals_across_eight_repetitions": cluster_totals,
            "repetitions": repetitions,
        }))
    }
}

#[derive(Clone)]
struct TraceRoot {
    root_index: usize,
    game_index: usize,
    game_id: u64,
    seed: u64,
    ply: usize,
    actual_move: Move,
    board: Board,
}

struct TraceGame {
    game_index: usize,
    game_id: u64,
    seed: u64,
    roots: Vec<TraceRoot>,
}

struct FrozenTrace {
    games: Vec<TraceGame>,
    root_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BoardSignature {
    serialized: Vec<u8>,
    sha256: String,
}

impl BoardSignature {
    fn from_board(board: &Board) -> Self {
        let serialized = board_signature_bytes(board);
        let sha256 = sha256_hex(&serialized);
        Self { serialized, sha256 }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SearchSignature {
    best_move: Option<Move>,
    score: i32,
    completed_depth: u32,
    returned_nodes: u64,
    main_nodes: u64,
    qsearch_nodes: u64,
    main_plus_qsearch: u128,
    tt_shape_probes: u64,
    tt_shape_hits: u64,
    tt_cutoffs: u64,
    tt_probes: u64,
    tt_hits: u64,
    tt_stores: u64,
    tt_displaced_depth_pref: u64,
    tt_stored_to_always: u64,
    tt_depth_hist: [u64; 16],
    tt_occupancy: (usize, usize, usize),
    final_board: BoardSignature,
}

struct SearchObservation {
    root_index: usize,
    game_index: usize,
    game_id: u64,
    seed: u64,
    ply: usize,
    actual_move: Move,
    root_zobrist: u64,
    ticks: u64,
    signature: SearchSignature,
    final_board_report: Value,
}

impl SearchObservation {
    fn report(&self) -> Value {
        json!({
            "root_index": self.root_index,
            "identity": {
                "game_index": self.game_index,
                "game_id": self.game_id,
                "seed": self.seed,
                "ply": self.ply,
                "actual_move": self.actual_move,
                "root_zobrist": format!("{:016X}", self.root_zobrist),
            },
            "ticks": self.ticks,
            "search": {
                "best_move": self.signature.best_move,
                "score": self.signature.score,
                "completed_depth": self.signature.completed_depth,
                "returned_nodes": self.signature.returned_nodes,
                "main_nodes": self.signature.main_nodes,
                "qsearch_nodes": self.signature.qsearch_nodes,
                "main_plus_qsearch": u128_string(self.signature.main_plus_qsearch),
            },
            "tt": {
                "shape_probes": self.signature.tt_shape_probes,
                "shape_hits": self.signature.tt_shape_hits,
                "cutoffs": self.signature.tt_cutoffs,
                "probes": self.signature.tt_probes,
                "hits": self.signature.tt_hits,
                "stores": self.signature.tt_stores,
                "displaced_depth_preferred": self.signature.tt_displaced_depth_pref,
                "stored_to_always_replace": self.signature.tt_stored_to_always,
                "depth_histogram": self.signature.tt_depth_hist,
                "occupancy": {
                    "depth_preferred": self.signature.tt_occupancy.0,
                    "always_replace": self.signature.tt_occupancy.1,
                    "buckets": self.signature.tt_occupancy.2,
                },
            },
            "final_board_signature_sha256": self.signature.final_board.sha256,
            "final_board": self.final_board_report,
        })
    }
}

struct SearchArmRun {
    label: &'static str,
    d4_enabled: bool,
    observations: Vec<SearchObservation>,
}

impl SearchArmRun {
    fn report(&self) -> Value {
        json!({
            "arm": self.label,
            "d4_hash_sidecar": self.d4_enabled,
            "roots": self
                .observations
                .iter()
                .map(SearchObservation::report)
                .collect::<Vec<_>>(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rational {
    numerator: u128,
    denominator: u128,
}

impl Rational {
    fn report(&self) -> Value {
        json!({
            "numerator": u128_string(self.numerator),
            "denominator": u128_string(self.denominator),
            "decimal": ratio_decimal(self.numerator, self.denominator),
        })
    }
}

struct BootstrapResult {
    seed: u64,
    final_rng_state: u64,
    rng_draws: u64,
    point: Rational,
    upper: Rational,
    point_gate: bool,
    upper_gate: bool,
}

impl BootstrapResult {
    fn passed(&self) -> bool {
        self.point_gate && self.upper_gate
    }

    fn report(&self) -> Value {
        json!({
            "replicates": BOOTSTRAP_REPLICATES,
            "draws_per_replicate": BOOTSTRAP_DRAWS,
            "seed_hex": format!("{:016X}", self.seed),
            "final_rng_state_hex": format!("{:016X}", self.final_rng_state),
            "rng_draws": self.rng_draws,
            "sorted_quantile_zero_based_index": BOOTSTRAP_INDEX,
            "point_ratio": self.point.report(),
            "one_sided_95_upper": self.upper.report(),
            "point_gate": {
                "rule": "B*200 <= A*201",
                "passed": self.point_gate,
            },
            "upper_gate": {
                "rule": "upper_num*100 < upper_den*101",
                "passed": self.upper_gate,
            },
            "passed": self.passed(),
        })
    }
}

#[cfg(windows)]
mod win32 {
    use std::ffi::c_void;

    pub type Handle = *mut c_void;
    pub const HIGH_PRIORITY_CLASS: u32 = 0x0000_0080;

    #[link(name = "Kernel32")]
    unsafe extern "system" {
        pub fn QueryPerformanceCounter(value: *mut i64) -> i32;
        pub fn QueryPerformanceFrequency(value: *mut i64) -> i32;
        pub fn GetCurrentProcess() -> Handle;
        pub fn GetCurrentThread() -> Handle;
        pub fn GetProcessAffinityMask(
            process: Handle,
            process_mask: *mut usize,
            system_mask: *mut usize,
        ) -> i32;
        pub fn SetThreadAffinityMask(thread: Handle, mask: usize) -> usize;
        pub fn GetCurrentProcessorNumber() -> u32;
        pub fn GetPriorityClass(process: Handle) -> u32;
        pub fn SetPriorityClass(process: Handle, priority_class: u32) -> i32;
    }
}

#[derive(Clone, Debug)]
struct SchedulingSnapshot {
    inherited_process_mask: usize,
    inherited_system_mask: usize,
    previous_thread_mask: usize,
    selected_thread_mask: usize,
    selected_processor: u32,
    previous_priority_class: u32,
    measured_priority_class: u32,
}

impl SchedulingSnapshot {
    fn report(&self, restored: bool) -> Value {
        json!({
            "inherited_process_affinity_mask_hex": format!("{:X}", self.inherited_process_mask),
            "inherited_system_affinity_mask_hex": format!("{:X}", self.inherited_system_mask),
            "previous_thread_affinity_mask_hex": format!("{:X}", self.previous_thread_mask),
            "selected_thread_affinity_mask_hex": format!("{:X}", self.selected_thread_mask),
            "selected_processor": self.selected_processor,
            "previous_priority_class_hex": format!("{:08X}", self.previous_priority_class),
            "measured_priority_class_hex": format!("{:08X}", self.measured_priority_class),
            "requested_priority_class": "HIGH_PRIORITY_CLASS",
            "no_worker_threads": true,
            "restored": restored,
        })
    }
}

struct SchedulingGuard {
    #[cfg(windows)]
    process: win32::Handle,
    #[cfg(windows)]
    thread: win32::Handle,
    snapshot: SchedulingSnapshot,
    restored: bool,
}

impl SchedulingGuard {
    #[cfg(windows)]
    fn setup() -> Result<Self, String> {
        let process = unsafe { win32::GetCurrentProcess() };
        let thread = unsafe { win32::GetCurrentThread() };
        let mut process_mask = 0usize;
        let mut system_mask = 0usize;
        if unsafe { win32::GetProcessAffinityMask(process, &mut process_mask, &mut system_mask) }
            == 0
        {
            return Err("GetProcessAffinityMask failed".to_string());
        }
        if process_mask == 0 || process_mask & !system_mask != 0 {
            return Err(format!(
                "invalid inherited affinity masks process={process_mask:X} system={system_mask:X}"
            ));
        }
        let selected_processor = usize::BITS - 1 - process_mask.leading_zeros();
        let selected_thread_mask = 1usize.checked_shl(selected_processor).ok_or_else(|| {
            "selected processor cannot be represented in affinity mask".to_string()
        })?;
        let previous_priority_class = unsafe { win32::GetPriorityClass(process) };
        if previous_priority_class == 0 {
            return Err("GetPriorityClass failed before setup".to_string());
        }

        let previous_thread_mask =
            unsafe { win32::SetThreadAffinityMask(thread, selected_thread_mask) };
        if previous_thread_mask == 0 {
            return Err(format!(
                "SetThreadAffinityMask failed for mask {selected_thread_mask:X}"
            ));
        }
        let verified_previous =
            unsafe { win32::SetThreadAffinityMask(thread, selected_thread_mask) };
        if verified_previous != selected_thread_mask {
            let _ = unsafe { win32::SetThreadAffinityMask(thread, previous_thread_mask) };
            return Err(format!(
                "thread affinity verification failed expected={selected_thread_mask:X} observed={verified_previous:X}"
            ));
        }
        let observed_processor = unsafe { win32::GetCurrentProcessorNumber() };
        if observed_processor != selected_processor {
            let _ = unsafe { win32::SetThreadAffinityMask(thread, previous_thread_mask) };
            return Err(format!(
                "thread ran on processor {observed_processor}, expected {selected_processor}"
            ));
        }
        if unsafe { win32::SetPriorityClass(process, win32::HIGH_PRIORITY_CLASS) } == 0 {
            let _ = unsafe { win32::SetThreadAffinityMask(thread, previous_thread_mask) };
            return Err("SetPriorityClass(HIGH_PRIORITY_CLASS) failed".to_string());
        }
        let measured_priority_class = unsafe { win32::GetPriorityClass(process) };
        if measured_priority_class != win32::HIGH_PRIORITY_CLASS {
            let _ = unsafe { win32::SetPriorityClass(process, previous_priority_class) };
            let _ = unsafe { win32::SetThreadAffinityMask(thread, previous_thread_mask) };
            return Err(format!(
                "priority verification failed expected={:08X} observed={measured_priority_class:08X}",
                win32::HIGH_PRIORITY_CLASS
            ));
        }

        Ok(Self {
            process,
            thread,
            snapshot: SchedulingSnapshot {
                inherited_process_mask: process_mask,
                inherited_system_mask: system_mask,
                previous_thread_mask,
                selected_thread_mask,
                selected_processor,
                previous_priority_class,
                measured_priority_class,
            },
            restored: false,
        })
    }

    #[cfg(not(windows))]
    fn setup() -> Result<Self, String> {
        Err("CB-GH0 P2-H is registered only for Windows".to_string())
    }

    #[cfg(windows)]
    fn restore(&mut self) -> Result<(), String> {
        let mut errors = Vec::new();

        let observed_previous = unsafe {
            win32::SetThreadAffinityMask(self.thread, self.snapshot.previous_thread_mask)
        };
        if observed_previous != self.snapshot.selected_thread_mask {
            errors.push(format!(
                "affinity restore returned {observed_previous:X}, expected prior selected mask {:X}",
                self.snapshot.selected_thread_mask
            ));
        }
        let verified_previous = unsafe {
            win32::SetThreadAffinityMask(self.thread, self.snapshot.previous_thread_mask)
        };
        if verified_previous != self.snapshot.previous_thread_mask {
            errors.push(format!(
                "affinity restore verification returned {verified_previous:X}, expected {:X}",
                self.snapshot.previous_thread_mask
            ));
        }

        if unsafe { win32::SetPriorityClass(self.process, self.snapshot.previous_priority_class) }
            == 0
        {
            errors.push("priority restore SetPriorityClass failed".to_string());
        }
        let restored_priority = unsafe { win32::GetPriorityClass(self.process) };
        if restored_priority != self.snapshot.previous_priority_class {
            errors.push(format!(
                "priority restore verification observed={restored_priority:08X} expected={:08X}",
                self.snapshot.previous_priority_class
            ));
        }

        self.restored = errors.is_empty();
        if self.restored {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }

    #[cfg(not(windows))]
    fn restore(&mut self) -> Result<(), String> {
        Err("CB-GH0 P2-H is registered only for Windows".to_string())
    }

    #[cfg(windows)]
    fn best_effort_restore(&mut self) {
        let _ = unsafe {
            win32::SetThreadAffinityMask(self.thread, self.snapshot.previous_thread_mask)
        };
        let _ =
            unsafe { win32::SetPriorityClass(self.process, self.snapshot.previous_priority_class) };
    }

    #[cfg(not(windows))]
    fn best_effort_restore(&mut self) {}
}

impl Drop for SchedulingGuard {
    fn drop(&mut self) {
        if !self.restored {
            self.best_effort_restore();
        }
    }
}

#[derive(Clone, Debug)]
struct ClockCalibration {
    frequency_before: u64,
    zero_deltas: usize,
    p50_ticks: u64,
    p95_ticks: u64,
    p99_ticks: u64,
    maximum_ticks: u64,
}

impl ClockCalibration {
    fn report(&self, frequency_after: Option<u64>) -> Value {
        json!({
            "clock": "Windows QueryPerformanceCounter",
            "frequency_before": self.frequency_before,
            "frequency_after": frequency_after,
            "frequency_unchanged": frequency_after == Some(self.frequency_before),
            "calibration_pairs": 10_000,
            "quantile_rule": "nearest-rank: zero-based ceil(p*n)-1",
            "zero_deltas": self.zero_deltas,
            "p50_ticks": self.p50_ticks,
            "p95_ticks": self.p95_ticks,
            "p99_ticks": self.p99_ticks,
            "maximum_ticks": self.maximum_ticks,
            "overhead_subtracted": false,
            "primary_arithmetic": "integer QPC ticks with checked u128 sums/products",
        })
    }
}

#[cfg(windows)]
fn qpc() -> Result<u64, String> {
    let mut value = 0i64;
    if unsafe { win32::QueryPerformanceCounter(&mut value) } == 0 {
        return Err("QueryPerformanceCounter failed".to_string());
    }
    u64::try_from(value).map_err(|_| format!("negative QueryPerformanceCounter value {value}"))
}

#[cfg(not(windows))]
fn qpc() -> Result<u64, String> {
    Err("QueryPerformanceCounter is unavailable off Windows".to_string())
}

#[cfg(windows)]
fn qpf() -> Result<u64, String> {
    let mut value = 0i64;
    if unsafe { win32::QueryPerformanceFrequency(&mut value) } == 0 {
        return Err("QueryPerformanceFrequency failed".to_string());
    }
    let frequency =
        u64::try_from(value).map_err(|_| format!("non-positive QPC frequency {value}"))?;
    if frequency == 0 {
        return Err("zero QPC frequency".to_string());
    }
    Ok(frequency)
}

#[cfg(not(windows))]
fn qpf() -> Result<u64, String> {
    Err("QueryPerformanceFrequency is unavailable off Windows".to_string())
}

fn qpc_elapsed(start: u64, end: u64, label: &str) -> Result<u64, String> {
    end.checked_sub(start)
        .ok_or_else(|| format!("QPC regressed in {label}: start={start} end={end}"))
}

fn calibrate_clock(frequency_before: u64) -> Result<ClockCalibration, String> {
    let mut deltas = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let start = qpc()?;
        let end = qpc()?;
        deltas.push(qpc_elapsed(start, end, "clock calibration")?);
    }
    deltas.sort_unstable();
    Ok(ClockCalibration {
        frequency_before,
        zero_deltas: deltas.partition_point(|&value| value == 0),
        p50_ticks: deltas[4_999],
        p95_ticks: deltas[9_499],
        p99_ticks: deltas[9_899],
        maximum_ticks: deltas[9_999],
    })
}

fn parse_args_from<I, S>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut iter = args.into_iter().map(Into::into);
    let option = iter
        .next()
        .ok_or_else(|| format!("missing --out-report\n{}", usage()))?;
    if option != "--out-report" {
        return Err(format!(
            "unknown option `{}`\n{}",
            option.to_string_lossy(),
            usage()
        ));
    }
    let out_report = iter
        .next()
        .ok_or_else(|| format!("--out-report requires a path\n{}", usage()))?;
    if out_report.is_empty() || out_report.to_string_lossy().starts_with("--") {
        return Err(format!(
            "--out-report requires a non-option path\n{}",
            usage()
        ));
    }
    if let Some(extra) = iter.next() {
        return Err(format!(
            "unexpected extra argument `{}`\n{}",
            extra.to_string_lossy(),
            usage()
        ));
    }
    Ok(Args {
        out_report: PathBuf::from(out_report),
    })
}

fn usage() -> &'static str {
    "usage: cb-gh0-hash-cost --out-report NEW.json"
}

fn refuse_existing(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Err(format!("refusing to overwrite {}", path.display()));
    }
    Ok(())
}

fn manifest_dir() -> Result<PathBuf, String> {
    fs::canonicalize(env!("CARGO_MANIFEST_DIR")).map_err(|error| {
        format!(
            "failed to canonicalize CARGO_MANIFEST_DIR {}: {error}",
            env!("CARGO_MANIFEST_DIR")
        )
    })
}

fn invoke_git(manifest: &Path, args: &[&str]) -> Result<String, String> {
    let safe = format!(
        "safe.directory={}",
        manifest.to_string_lossy().replace('\\', "/")
    );
    let output = Command::new("git")
        .arg("-c")
        .arg(safe)
        .args(args)
        .current_dir(manifest)
        .output()
        .map_err(|error| format!("failed to invoke git {:?}: {error}", args))?;
    if !output.status.success() {
        return Err(format!(
            "git {:?} failed status={}: stdout={} stderr={}",
            args,
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .map_err(|error| format!("git {:?} emitted non-UTF-8 output: {error}", args))
}

fn seal_file(name: &str, path: &Path) -> Result<(FileSeal, Vec<u8>), String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("failed to read {} at {}: {error}", name, path.display()))?;
    let canonical = path
        .canonicalize()
        .map_err(|error| {
            format!(
                "failed to canonicalize sealed file {} at {}: {error}",
                name,
                path.display()
            )
        })?
        .display()
        .to_string();
    let seal = FileSeal {
        name: name.to_string(),
        path: canonical,
        bytes: u64::try_from(bytes.len()).map_err(|_| format!("{name} length cannot fit u64"))?,
        sha256: sha256_hex(&bytes),
    };
    Ok((seal, bytes))
}

fn seal_source(manifest: &Path) -> Result<SourceSeal, String> {
    let git_head = invoke_git(manifest, &["rev-parse", "HEAD"])?;
    if git_head.len() != 40 || !git_head.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("unexpected git HEAD {git_head:?}"));
    }
    invoke_git(
        manifest,
        &["merge-base", "--is-ancestor", EXPECTED_GIT_PARENT, "HEAD"],
    )
    .map_err(|error| {
        format!(
            "HEAD does not descend from frozen preregistration commit {EXPECTED_GIT_PARENT}: {error}"
        )
    })?;

    let mut critical_sources = Vec::with_capacity(CRITICAL_SOURCES.len());
    for relative in CRITICAL_SOURCES {
        invoke_git(manifest, &["ls-files", "--error-unmatch", "--", relative]).map_err(
            |error| format!("critical source is not tracked in HEAD ({relative}): {error}"),
        )?;
        let (seal, _) = seal_file(relative, &manifest.join(relative))?;
        critical_sources.push(seal);
    }
    let tracked_status = invoke_git(manifest, &["status", "--porcelain", "--untracked-files=no"])?;
    if !tracked_status.is_empty() {
        return Err(format!(
            "tracked worktree is dirty before measurement: {tracked_status}"
        ));
    }
    Ok(SourceSeal {
        git_head,
        tracked_status,
        critical_sources,
    })
}

fn seal_executable() -> Result<FileSeal, String> {
    let path = env::current_exe().map_err(|error| format!("current_exe failed: {error}"))?;
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "current executable has no UTF-8 file name".to_string())?;
    if !file_name.eq_ignore_ascii_case("cb-gh0-hash-cost.exe")
        && !file_name.eq_ignore_ascii_case("cb-gh0-hash-cost")
    {
        return Err(format!(
            "unexpected executable name {file_name:?}, expected cb-gh0-hash-cost"
        ));
    }
    seal_file("cb-gh0-hash-cost executable", &path).map(|(seal, _)| seal)
}

fn load_frozen_inputs(manifest: &Path) -> Result<FrozenInputs, String> {
    let mut seals = Vec::with_capacity(INPUT_SPECS.len());
    let mut payloads = Vec::with_capacity(INPUT_SPECS.len());
    for spec in INPUT_SPECS {
        let path = manifest.join(spec.relative_path);
        let (seal, bytes) = seal_file(spec.name, &path)?;
        if seal.bytes != spec.bytes || !seal.sha256.eq_ignore_ascii_case(spec.sha256) {
            return Err(format!(
                "sealed input mismatch {}: bytes={} expected={} sha256={} expected={}",
                spec.name, seal.bytes, spec.bytes, seal.sha256, spec.sha256
            ));
        }
        seals.push(seal);
        payloads.push(bytes);
    }
    let mut iter = payloads.into_iter();
    let raw_codebook = iter.next().ok_or("missing raw codebook payload")?;
    let compact_codebook = iter.next().ok_or("missing compact codebook payload")?;
    let topk = iter.next().ok_or("missing topk payload")?;
    let nnue = iter.next().ok_or("missing NNUE payload")?;
    let trace = iter.next().ok_or("missing trace payload")?;
    if iter.next().is_some() {
        return Err("unexpected extra frozen input payload".to_string());
    }
    Ok(FrozenInputs {
        seals,
        raw_codebook,
        compact_codebook,
        topk,
        nnue,
        trace,
    })
}

fn reseal_inputs(manifest: &Path) -> Result<Vec<FileSeal>, String> {
    let mut seals = Vec::with_capacity(INPUT_SPECS.len());
    for spec in INPUT_SPECS {
        let (seal, _) = seal_file(spec.name, &manifest.join(spec.relative_path))?;
        seals.push(seal);
    }
    Ok(seals)
}

fn environment_identity() -> Result<Value, String> {
    let rustflags = env::var("RUSTFLAGS")
        .map_err(|_| format!("runtime RUSTFLAGS must equal {CANONICAL_RUSTFLAGS:?}"))?;
    if rustflags != CANONICAL_RUSTFLAGS {
        return Err(format!(
            "runtime RUSTFLAGS mismatch: observed={rustflags:?} expected={CANONICAL_RUSTFLAGS:?}"
        ));
    }
    let mut noru_names = env::vars_os()
        .filter_map(|(name, _)| {
            let rendered = name.to_string_lossy().into_owned();
            rendered
                .to_ascii_uppercase()
                .starts_with("NORU_")
                .then_some(rendered)
        })
        .collect::<Vec<_>>();
    noru_names.sort();
    if !noru_names.is_empty() {
        return Err(format!(
            "NORU_* environment overrides are forbidden: {noru_names:?}"
        ));
    }
    Ok(json!({
        "runtime_RUSTFLAGS": rustflags,
        "canonical_build": "cargo build --release --locked --features codebook-eval --bin cb-gh0-hash-cost",
        "noru_prefixed_variables": noru_names,
    }))
}

fn toolchain_identity() -> Result<Value, String> {
    let output = Command::new("rustc")
        .arg("-Vv")
        .output()
        .map_err(|error| format!("failed to invoke rustc -Vv: {error}"))?;
    if !output.status.success() {
        return Err(format!("rustc -Vv failed with {}", output.status));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("rustc -Vv emitted non-UTF-8 output: {error}"))?;
    let release = stdout
        .lines()
        .find(|line| line.starts_with("release: "))
        .map(|line| format!("rustc {}", line.trim_start_matches("release: ")))
        .ok_or_else(|| "rustc -Vv missing release line".to_string())?;
    if release != EXPECTED_RUSTC
        || !stdout.lines().any(|line| line == EXPECTED_HOST)
        || !stdout.lines().any(|line| line == EXPECTED_LLVM)
    {
        return Err(format!(
            "toolchain identity mismatch: release={release:?}\n{stdout}"
        ));
    }
    Ok(json!({
        "release": release,
        "host": EXPECTED_HOST.trim_start_matches("host: "),
        "llvm": EXPECTED_LLVM.trim_start_matches("LLVM version: "),
        "rustc_vv": stdout,
    }))
}

#[cfg(target_arch = "x86_64")]
fn cpu_identity() -> Result<Value, String> {
    use std::arch::x86_64::{__cpuid, __cpuid_count};

    let leaf0 = unsafe { __cpuid(0) };
    let mut vendor_bytes = Vec::with_capacity(12);
    vendor_bytes.extend_from_slice(&leaf0.ebx.to_le_bytes());
    vendor_bytes.extend_from_slice(&leaf0.edx.to_le_bytes());
    vendor_bytes.extend_from_slice(&leaf0.ecx.to_le_bytes());
    let vendor = String::from_utf8(vendor_bytes)
        .map_err(|error| format!("CPUID vendor is not UTF-8: {error}"))?;

    let leaf1 = unsafe { __cpuid(1) };
    let base_family = (leaf1.eax >> 8) & 0x0f;
    let extended_family = (leaf1.eax >> 20) & 0xff;
    let family = if base_family == 0x0f {
        base_family + extended_family
    } else {
        base_family
    };
    let base_model = (leaf1.eax >> 4) & 0x0f;
    let extended_model = (leaf1.eax >> 16) & 0x0f;
    let model = if matches!(base_family, 0x06 | 0x0f) {
        base_model | (extended_model << 4)
    } else {
        base_model
    };
    let stepping = leaf1.eax & 0x0f;
    let logical = std::thread::available_parallelism()
        .map_err(|error| format!("available_parallelism failed: {error}"))?
        .get();
    let _extended_leaf = unsafe { __cpuid_count(7, 0) };

    if vendor != EXPECTED_CPU_VENDOR
        || family != EXPECTED_CPU_FAMILY
        || model != EXPECTED_CPU_MODEL
        || stepping != EXPECTED_CPU_STEPPING
        || logical != EXPECTED_LOGICAL_PROCESSORS
    {
        return Err(format!(
            "CPU identity mismatch vendor={vendor} family={family} model={model} stepping={stepping} logical={logical}"
        ));
    }
    Ok(json!({
        "vendor": vendor,
        "family": family,
        "model": model,
        "stepping": stepping,
        "logical_processors": logical,
        "architecture": env::consts::ARCH,
    }))
}

#[cfg(not(target_arch = "x86_64"))]
fn cpu_identity() -> Result<Value, String> {
    Err("CB-GH0 P2-H requires x86_64".to_string())
}

fn stone_tag(stone: Stone) -> u8 {
    match stone {
        Stone::Black => 0,
        Stone::White => 1,
    }
}

fn rule_tag(rule: RuleSet) -> u8 {
    match rule {
        RuleSet::Freestyle => 0,
        RuleSet::Standard => 1,
        RuleSet::Caro => 2,
        RuleSet::Renju => 3,
    }
}

fn result_name(result: GameResult) -> &'static str {
    match result {
        GameResult::BlackWin => "black_win",
        GameResult::WhiteWin => "white_win",
        GameResult::Draw => "draw",
        GameResult::Ongoing => "ongoing",
    }
}

fn board_signature_bytes(board: &Board) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(71 + board.history.len() * 2 + NUM_CELLS * 8);
    bytes.extend_from_slice(&board.black.lo.to_be_bytes());
    bytes.extend_from_slice(&board.black.hi.to_be_bytes());
    bytes.extend_from_slice(&board.white.lo.to_be_bytes());
    bytes.extend_from_slice(&board.white.hi.to_be_bytes());
    bytes.push(stone_tag(board.side_to_move));
    bytes.push(rule_tag(board.rule_set));
    bytes.push(u8::from(board.exact5));
    bytes.extend_from_slice(&(board.move_count as u16).to_be_bytes());
    bytes.extend_from_slice(
        &board
            .last_move
            .map(|value| value as u16)
            .unwrap_or(u16::MAX)
            .to_be_bytes(),
    );
    bytes.extend_from_slice(&(board.history.len() as u16).to_be_bytes());
    for &mv in &board.history {
        bytes.extend_from_slice(&(mv as u16).to_be_bytes());
    }
    bytes.extend_from_slice(&board.zobrist.to_be_bytes());
    for pattern in board.line_pattern_ids.iter() {
        for &id in pattern {
            bytes.extend_from_slice(&id.to_be_bytes());
        }
    }
    bytes
}

fn board_report(board: &Board) -> Value {
    json!({
        "black": {
            "lo_hex": format!("{:032X}", board.black.lo),
            "hi_hex": format!("{:032X}", board.black.hi),
        },
        "white": {
            "lo_hex": format!("{:032X}", board.white.lo),
            "hi_hex": format!("{:032X}", board.white.hi),
        },
        "side_to_move": match board.side_to_move {
            Stone::Black => "Black",
            Stone::White => "White",
        },
        "formal_rule": match board.rule_set {
            RuleSet::Freestyle => "Freestyle",
            RuleSet::Standard => "Standard",
            RuleSet::Caro => "Caro",
            RuleSet::Renju => "Renju",
        },
        "effective_rule": match board.effective_rule_set() {
            RuleSet::Freestyle => "Freestyle",
            RuleSet::Standard => "Standard",
            RuleSet::Caro => "Caro",
            RuleSet::Renju => "Renju",
        },
        "exact5": board.exact5,
        "move_count": board.move_count,
        "last_move": board.last_move,
        "history": board.history,
        "zobrist_hex": format!("{:016X}", board.zobrist),
        "pattern4_ids": board
            .line_pattern_ids
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
        "signature_sha256": sha256_hex(&board_signature_bytes(board)),
    })
}

fn exact_state_report(state: ExactCanonicalState) -> Value {
    json!({
        "bytes_hex": hex_bytes(&state.bytes),
        "to_canonical": state.to_canonical,
    })
}

fn block_start(cluster: usize) -> usize {
    let q = TRANSITIONS / TRANSITION_CLUSTERS;
    let r = TRANSITIONS % TRANSITION_CLUSTERS;
    cluster * q + cluster.min(r)
}

fn generate_transition_tape() -> Result<TransitionTape, String> {
    let mut board = Board::new();
    board.set_rule_set(RuleSet::Freestyle);
    let mut rng = SplitMix64::new(TAPE_SEED);
    let mut entries = Vec::with_capacity(TRANSITIONS);
    let mut blocks = Vec::with_capacity(TRANSITION_CLUSTERS);
    let mut serialized = Vec::with_capacity(TRANSITIONS * 9);
    let mut makes = 0usize;
    let mut undos = 0usize;
    let mut rule_switches = 0usize;
    let mut maximum_move_count = 0usize;

    for transition_one_based in 1..=TRANSITIONS {
        let rule_before = if transition_one_based % RULE_PERIOD == 0 {
            let rule = RuleMode::from_ordinal(rule_switches);
            rule.apply(&mut board);
            rule_switches += 1;
            Some(rule)
        } else {
            None
        };
        let decision = rng.next();
        let should_undo =
            !board.history.is_empty() && (board.move_count >= MAX_STONES || decision & 3 == 0);
        let (action, mv, color) = if should_undo {
            let mv = *board
                .history
                .last()
                .ok_or_else(|| "generated undo without history".to_string())?;
            let color = board.side_to_move.opponent();
            board.undo_move();
            undos += 1;
            (TapeAction::Undo, mv, color)
        } else {
            let legal = board.legal_moves();
            if legal.is_empty() {
                return Err(format!(
                    "tape generator has no legal move at transition {transition_one_based}"
                ));
            }
            let pick = rng.next();
            let mv = legal[(pick as usize) % legal.len()];
            let color = board.side_to_move;
            board.make_move(mv);
            makes += 1;
            (TapeAction::Make, mv, color)
        };
        maximum_move_count = maximum_move_count.max(board.move_count);
        let entry = TapeEntry {
            index: transition_one_based - 1,
            rule_before,
            action,
            mv,
            color,
        };
        serialized.extend_from_slice(&(entry.index as u32).to_be_bytes());
        serialized.push(entry.rule_before.map(RuleMode::tag).unwrap_or(0xff));
        serialized.push(entry.action.tag());
        serialized.extend_from_slice(&(entry.mv as u16).to_be_bytes());
        serialized.push(stone_tag(entry.color));
        entries.push(entry);

        let next_index = transition_one_based;
        if blocks.len() < TRANSITION_CLUSTERS && next_index == block_start(blocks.len() + 1) {
            let cluster = blocks.len();
            blocks.push(BlockReference {
                start: block_start(cluster),
                end: next_index,
                digest: sha256_hex(&board_signature_bytes(&board)),
                board: board.clone(),
            });
        }
    }

    let tape = TransitionTape {
        serialized_sha256: sha256_hex(&serialized),
        entries,
        blocks,
        makes,
        undos,
        rule_switches,
        maximum_move_count,
        final_move_count: board.move_count,
        final_rng_state: rng.state,
        rng_draws: rng.draws,
    };
    validate_transition_tape(&tape)?;
    Ok(tape)
}

fn validate_transition_tape(tape: &TransitionTape) -> Result<(), String> {
    if tape.entries.len() != TRANSITIONS
        || tape.makes != MAKES
        || tape.undos != UNDOS
        || tape.rule_switches != RULE_SWITCHES
        || tape.maximum_move_count != MAX_STONES
        || tape.final_move_count != FINAL_MOVE_COUNT
        || tape.final_rng_state != TAPE_FINAL_STATE
        || tape.rng_draws != PRNG_DRAWS
        || tape.blocks.len() != TRANSITION_CLUSTERS
    {
        return Err(format!(
            "transition tape constants mismatch entries={} makes={} undos={} switches={} max={} final={} rng_state={:016X} draws={} blocks={}",
            tape.entries.len(),
            tape.makes,
            tape.undos,
            tape.rule_switches,
            tape.maximum_move_count,
            tape.final_move_count,
            tape.final_rng_state,
            tape.rng_draws,
            tape.blocks.len(),
        ));
    }
    for (cluster, block) in tape.blocks.iter().enumerate() {
        let expected_start = block_start(cluster);
        let expected_end = block_start(cluster + 1);
        let expected_len = if cluster < 32 { 1_563 } else { 1_562 };
        if block.start != expected_start
            || block.end != expected_end
            || block.end - block.start != expected_len
        {
            return Err(format!(
                "transition block geometry mismatch cluster={cluster} start={} end={} expected_start={expected_start} expected_end={expected_end} expected_len={expected_len}",
                block.start, block.end,
            ));
        }
    }
    Ok(())
}

fn apply_tape_entry(board: &mut Board, state: &mut BoardSearchState, entry: &TapeEntry) {
    match entry.action {
        TapeAction::Make => state.make_move(board, entry.mv),
        TapeAction::Undo => state.undo_move(board),
    }
}

fn verify_transition_state(
    board: &Board,
    state: &BoardSearchState,
    d4_enabled: bool,
    expected: &Board,
    phase: &str,
    ledger: &mut FailureLedger,
) {
    let board_matches = board_signature_bytes(board) == board_signature_bytes(expected);
    if !board_matches {
        ledger.reject(
            "transition_board_state",
            json!({
                "phase": phase,
                "observed": board_report(board),
                "expected": board_report(expected),
            }),
        );
    }
    if !state.is_synchronized(board) {
        ledger.reject(
            "transition_sidecar_synchronization",
            json!({"phase": phase, "d4_enabled": d4_enabled}),
        );
    }
    if state.d4_hash_enabled() != d4_enabled {
        ledger.reject(
            "transition_sidecar_selector",
            json!({
                "phase": phase,
                "expected": d4_enabled,
                "observed": state.d4_hash_enabled(),
            }),
        );
    }
    if d4_enabled {
        let rebuilt = D4HashState::rebuild(board);
        let observed_hashes = state.d4_hashes(board);
        let observed_context = state.d4_canonical_context(board);
        if observed_hashes != Some(*rebuilt.hashes())
            || observed_context != Some(rebuilt.canonical_context())
        {
            ledger.reject(
                "transition_d4_rebuild",
                json!({
                    "phase": phase,
                    "observed_hashes": observed_hashes,
                    "rebuilt_hashes": rebuilt.hashes(),
                    "observed_context": observed_context.map(|value| json!({
                        "key": format!("{:016X}", value.key),
                        "to_canonical": value.to_canonical,
                    })),
                    "rebuilt_context": {
                        "key": format!("{:016X}", rebuilt.canonical_context().key),
                        "to_canonical": rebuilt.canonical_context().to_canonical,
                    },
                }),
            );
        }
    } else if state.d4_hashes(board).is_some() || state.d4_canonical_context(board).is_some() {
        ledger.reject("transition_off_sidecar_exposed", json!({"phase": phase}));
    }
}

fn run_transition_repetition(
    tape: &TransitionTape,
    d4_enabled: bool,
    measured: bool,
    arm: &str,
    repetition: usize,
    ledger: &mut FailureLedger,
) -> Result<TransitionRepetition, String> {
    let mut board = Board::new();
    board.set_rule_set(RuleSet::Freestyle);
    let mut state = BoardSearchState::new();

    let initial_selector_ticks = if measured {
        let started = qpc()?;
        state.set_d4_hash_enabled(&board, d4_enabled);
        qpc_elapsed(started, qpc()?, "initial sidecar selector/rebuild")?
    } else {
        state.set_d4_hash_enabled(&board, d4_enabled);
        0
    };
    verify_transition_state(
        &board,
        &state,
        d4_enabled,
        &Board::new(),
        &format!("{arm}/rep{repetition}/initial"),
        ledger,
    );

    let mut primary_cluster_ticks = Vec::with_capacity(TRANSITION_CLUSTERS);
    let mut rule_rebuild_ticks = Vec::with_capacity(RULE_SWITCHES);
    let mut block_state_digests = Vec::with_capacity(TRANSITION_CLUSTERS);
    for (cluster, reference) in tape.blocks.iter().enumerate() {
        let mut cluster_ticks = 0u128;
        let mut cursor = reference.start;
        while cursor < reference.end {
            if let Some(rule) = tape.entries[cursor].rule_before {
                let rebuild_ticks = if measured {
                    let started = qpc()?;
                    rule.apply(&mut board);
                    state.synchronize(&board);
                    qpc_elapsed(started, qpc()?, "rule synchronization/rebuild")?
                } else {
                    rule.apply(&mut board);
                    state.synchronize(&board);
                    0
                };
                rule_rebuild_ticks.push(rebuild_ticks);
            }

            let mut segment_end = cursor + 1;
            while segment_end < reference.end && tape.entries[segment_end].rule_before.is_none() {
                segment_end += 1;
            }
            if measured {
                let started = qpc()?;
                for entry in &tape.entries[cursor..segment_end] {
                    apply_tape_entry(&mut board, &mut state, entry);
                }
                let elapsed = qpc_elapsed(started, qpc()?, "transition segment")?;
                cluster_ticks = cluster_ticks
                    .checked_add(u128::from(elapsed))
                    .ok_or_else(|| "transition cluster tick overflow".to_string())?;
            } else {
                for entry in &tape.entries[cursor..segment_end] {
                    apply_tape_entry(&mut board, &mut state, entry);
                }
            }
            cursor = segment_end;
        }
        if measured && cluster_ticks == 0 {
            ledger.invalid(
                "zero_transition_cluster_ticks",
                json!({"arm": arm, "repetition": repetition, "cluster": cluster}),
            );
        }
        primary_cluster_ticks.push(cluster_ticks);
        verify_transition_state(
            &board,
            &state,
            d4_enabled,
            &reference.board,
            &format!("{arm}/rep{repetition}/cluster{cluster}"),
            ledger,
        );
        let observed_digest = sha256_hex(&board_signature_bytes(&board));
        block_state_digests.push(observed_digest.clone());
        if observed_digest != reference.digest {
            ledger.reject(
                "transition_block_digest",
                json!({
                    "arm": arm,
                    "repetition": repetition,
                    "cluster": cluster,
                    "observed": observed_digest,
                    "expected": reference.digest,
                }),
            );
        }
    }
    if rule_rebuild_ticks.len() != RULE_SWITCHES {
        ledger.invalid(
            "rule_rebuild_count",
            json!({
                "arm": arm,
                "repetition": repetition,
                "observed": rule_rebuild_ticks.len(),
                "expected": RULE_SWITCHES,
            }),
        );
    }
    let final_state_digest = sha256_hex(&board_signature_bytes(&board));

    while !board.history.is_empty() {
        state.undo_move(&mut board);
    }
    board.set_rule_set(RuleSet::Freestyle);
    state.synchronize(&board);
    let fresh = Board::new();
    let mut fresh_state = BoardSearchState::new();
    fresh_state.set_d4_hash_enabled(&fresh, d4_enabled);
    verify_transition_state(
        &board,
        &state,
        d4_enabled,
        &fresh,
        &format!("{arm}/rep{repetition}/unwind"),
        ledger,
    );
    if state.d4_hashes(&board) != fresh_state.d4_hashes(&fresh)
        || state.d4_canonical_context(&board) != fresh_state.d4_canonical_context(&fresh)
        || exact_canonical_state(&board) != exact_canonical_state(&fresh)
    {
        ledger.reject(
            "transition_unwind",
            json!({
                "arm": arm,
                "repetition": repetition,
                "observed_board": board_report(&board),
                "fresh_board": board_report(&fresh),
                "observed_hashes": state.d4_hashes(&board),
                "fresh_hashes": fresh_state.d4_hashes(&fresh),
                "observed_context": state.d4_canonical_context(&board).map(|value| json!({
                    "key": format!("{:016X}", value.key),
                    "to_canonical": value.to_canonical,
                })),
                "fresh_context": fresh_state.d4_canonical_context(&fresh).map(|value| json!({
                    "key": format!("{:016X}", value.key),
                    "to_canonical": value.to_canonical,
                })),
                "observed_exact": exact_state_report(exact_canonical_state(&board)),
                "fresh_exact": exact_state_report(exact_canonical_state(&fresh)),
            }),
        );
    }
    let unwind_state_digest = sha256_hex(&board_signature_bytes(&board));

    Ok(TransitionRepetition {
        primary_cluster_ticks,
        initial_selector_ticks,
        rule_rebuild_ticks,
        block_state_digests,
        final_state_digest,
        unwind_state_digest,
    })
}

fn run_transition_warmup(
    tape: &TransitionTape,
    ledger: &mut FailureLedger,
) -> Result<Value, String> {
    let mut reports = Vec::with_capacity(TRANSITION_WARMUP_ORDER.len());
    for (index, label) in TRANSITION_WARMUP_ORDER.iter().enumerate() {
        let d4_enabled = *label == "B";
        let repetition = run_transition_repetition(tape, d4_enabled, false, label, index, ledger)?;
        reports.push(json!({
            "order_index": index,
            "arm": label,
            "d4_hash_sidecar": d4_enabled,
            "final_state_digest": repetition.final_state_digest,
            "unwind_state_digest": repetition.unwind_state_digest,
            "contributes_to_timing": false,
        }));
    }
    Ok(json!({
        "order": TRANSITION_WARMUP_ORDER,
        "complete_tapes": reports,
    }))
}

fn run_transition_measured(
    tape: &TransitionTape,
    ledger: &mut FailureLedger,
) -> Result<Vec<TransitionArmRun>, String> {
    let mut arms = Vec::with_capacity(MEASURED_ARM_ORDER.len());
    for &label in &MEASURED_ARM_ORDER {
        let d4_enabled = label.starts_with('B');
        let mut repetitions = Vec::with_capacity(TRANSITION_REPETITIONS);
        for repetition in 0..TRANSITION_REPETITIONS {
            repetitions.push(run_transition_repetition(
                tape, d4_enabled, true, label, repetition, ledger,
            )?);
        }
        arms.push(TransitionArmRun {
            label,
            d4_enabled,
            repetitions,
        });
    }
    Ok(arms)
}

fn is_figrid(name: &str) -> bool {
    name.to_ascii_lowercase().contains("figrid")
}

fn parse_trace_stone(raw: &str) -> Option<Stone> {
    match raw {
        "B" => Some(Stone::Black),
        "W" => Some(Stone::White),
        _ => None,
    }
}

fn required_string<'a>(value: &'a Value, field: &str, context: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{context} missing string field {field}"))
}

fn required_u64(value: &Value, field: &str, context: &str) -> Result<u64, String> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{context} missing u64 field {field}"))
}

fn parse_frozen_trace(trace: &[u8]) -> Result<FrozenTrace, String> {
    let mut games = Vec::with_capacity(EXPECTED_GAMES);
    let mut processed_roots = 0usize;

    for line_result in BufReader::new(Cursor::new(trace)).lines() {
        let line = line_result.map_err(|error| format!("trace is not valid UTF-8: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let game_index = games.len();
        let game: Value = serde_json::from_str(&line).map_err(|error| {
            format!(
                "invalid trace JSON at nonblank game line {}: {error}",
                game_index + 1
            )
        })?;
        let context = format!("trace game line {}", game_index + 1);
        let game_id = required_u64(&game, "game_id", &context)?;
        let seed = required_u64(&game, "seed", &context)?;
        let black_engine = required_string(&game, "black_engine", &context)?;
        let white_engine = required_string(&game, "white_engine", &context)?;
        let product_side = match (is_figrid(black_engine), is_figrid(white_engine)) {
            (true, false) => Stone::Black,
            (false, true) => Stone::White,
            other => {
                return Err(format!(
                    "expected exactly one case-insensitive figrid engine game={game_id}, got {other:?}"
                ));
            }
        };
        let moves = game
            .get("moves")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("trace game={game_id} missing moves array"))?;
        let declared_result = required_string(&game, "result", &context)?;
        if !matches!(declared_result, "black_win" | "white_win" | "draw") {
            return Err(format!(
                "trace game={game_id} has invalid final result {declared_result:?}"
            ));
        }
        let declared_move_count =
            required_u64(&game, "move_count", &context).and_then(|value| {
                usize::try_from(value)
                    .map_err(|_| format!("trace game={game_id} move_count exceeds usize"))
            })?;
        if declared_move_count != moves.len() {
            return Err(format!(
                "trace move_count mismatch game={game_id}: declared={declared_move_count} rows={}",
                moves.len()
            ));
        }

        let mut board = Board::new();
        board.set_rule_set(RuleSet::Freestyle);
        let mut roots = Vec::new();
        for (ply, move_json) in moves.iter().enumerate() {
            if board.game_result() != GameResult::Ongoing {
                return Err(format!(
                    "trace contains move after terminal state game={game_id} ply={ply}"
                ));
            }
            let move_context = format!("trace game={game_id} ply={ply}");
            let x = usize::try_from(required_u64(move_json, "x", &move_context)?)
                .map_err(|_| format!("{move_context} x exceeds usize"))?;
            let y = usize::try_from(required_u64(move_json, "y", &move_context)?)
                .map_err(|_| format!("{move_context} y exceeds usize"))?;
            if x >= BOARD_SIZE || y >= BOARD_SIZE {
                return Err(format!(
                    "trace move outside board game={game_id} ply={ply} x={x} y={y}"
                ));
            }
            let color_raw = required_string(move_json, "color", &move_context)?;
            let color = parse_trace_stone(color_raw).ok_or_else(|| {
                format!("trace move invalid color game={game_id} ply={ply}: {color_raw}")
            })?;
            let source = required_string(move_json, "source", &move_context)?;
            if !matches!(source, "engine" | "opening") {
                return Err(format!(
                    "trace move invalid source game={game_id} ply={ply}: {source:?}"
                ));
            }
            let mv = to_idx(y, x);

            if source == "engine"
                && board.side_to_move == product_side
                && processed_roots < EXPECTED_ROOTS
            {
                roots.push(TraceRoot {
                    root_index: processed_roots,
                    game_index,
                    game_id,
                    seed,
                    ply,
                    actual_move: mv,
                    board: board.clone(),
                });
                processed_roots += 1;
            }
            if color != board.side_to_move {
                return Err(format!(
                    "trace color/STM mismatch game={game_id} ply={ply}: color={color_raw} stm={:?}",
                    board.side_to_move
                ));
            }
            if !board.is_legal_move(mv) {
                return Err(format!(
                    "trace illegal or occupied move game={game_id} ply={ply} idx={mv}"
                ));
            }
            board.make_move(mv);
        }
        if board.move_count != declared_move_count {
            return Err(format!(
                "trace replay count mismatch game={game_id}: replay={} declared={declared_move_count}",
                board.move_count
            ));
        }
        if result_name(board.game_result()) != declared_result {
            return Err(format!(
                "trace result mismatch game={game_id}: replay={} declared={declared_result}",
                result_name(board.game_result())
            ));
        }
        games.push(TraceGame {
            game_index,
            game_id,
            seed,
            roots,
        });
    }

    if games.len() != EXPECTED_GAMES {
        return Err(format!(
            "frozen trace game count mismatch: got {}, expected {EXPECTED_GAMES}",
            games.len()
        ));
    }
    if processed_roots != EXPECTED_ROOTS {
        return Err(format!(
            "frozen trace root count mismatch: got {processed_roots}, expected {EXPECTED_ROOTS}"
        ));
    }
    for game in &games {
        if !(SEARCH_MIN_ROOTS_PER_GAME..=SEARCH_MAX_ROOTS_PER_GAME).contains(&game.roots.len()) {
            return Err(format!(
                "game {} root contribution out of range: got {}, expected {}..={}",
                game.game_id,
                game.roots.len(),
                SEARCH_MIN_ROOTS_PER_GAME,
                SEARCH_MAX_ROOTS_PER_GAME
            ));
        }
        for root in &game.roots {
            if root.game_index != game.game_index
                || root.game_id != game.game_id
                || root.seed != game.seed
                || root.board.effective_rule_set() != RuleSet::Freestyle
            {
                return Err(format!(
                    "root identity/rule mismatch game={} root={}",
                    game.game_id, root.root_index
                ));
            }
        }
    }
    Ok(FrozenTrace {
        games,
        root_count: processed_roots,
    })
}

fn configured_product_searcher(d4_enabled: bool) -> Result<Searcher, String> {
    let mut searcher = Searcher::new();
    if searcher.d4_hash_sidecar_requested() || !searcher.root_vct_requested_for_audit() {
        return Err(format!(
            "Searcher defaults changed before audit overrides: d4={} root_vct={}",
            searcher.d4_hash_sidecar_requested(),
            searcher.root_vct_requested_for_audit()
        ));
    }
    searcher.set_use_threat_field(false);
    searcher.set_use_lazy_threat_field(false);
    searcher.set_use_move_picker(false);
    searcher.set_use_tail_threat_materialize(false);
    searcher.set_use_packed_line_windows(true);
    searcher.set_use_candidate_frontier(true);
    searcher.set_use_codebook_directional_delta(true);
    searcher.set_white_root_order_enabled(true)?;
    searcher.set_node_limit(None);
    searcher.set_use_d4_hash_sidecar(d4_enabled);
    searcher.set_use_root_vct_for_audit(false);
    if searcher.use_threat_field()
        || !searcher.white_root_order_enabled()
        || searcher.d4_hash_sidecar_requested() != d4_enabled
        || searcher.root_vct_requested_for_audit()
    {
        return Err(format!(
            "configured Searcher selector verification failed d4={d4_enabled}"
        ));
    }
    Ok(searcher)
}

fn capture_search_signature(
    result: &SearchResult,
    shape: SearchShapeStats,
    searcher: &Searcher,
    board: &Board,
) -> Result<SearchSignature, String> {
    let tt = searcher.tt_stats();
    Ok(SearchSignature {
        best_move: result.best_move,
        score: result.score,
        completed_depth: result.depth,
        returned_nodes: result.nodes,
        main_nodes: shape.main_nodes,
        qsearch_nodes: shape.qsearch_nodes,
        main_plus_qsearch: checked_node_sum(shape.main_nodes, shape.qsearch_nodes)?,
        tt_shape_probes: shape.tt_probes,
        tt_shape_hits: shape.tt_hits,
        tt_cutoffs: shape.tt_cutoffs,
        tt_probes: tt.probes,
        tt_hits: tt.hits,
        tt_stores: tt.stores,
        tt_displaced_depth_pref: tt.displaced_depth_pref,
        tt_stored_to_always: tt.stored_to_always,
        tt_depth_hist: tt.depth_hist,
        tt_occupancy: searcher.tt_occupancy(),
        final_board: BoardSignature::from_board(board),
    })
}

fn checked_node_sum(main_nodes: u64, qsearch_nodes: u64) -> Result<u128, String> {
    u128::from(main_nodes)
        .checked_add(u128::from(qsearch_nodes))
        .ok_or_else(|| "u128 main+qsearch node sum overflow".to_string())
}

fn run_search_root(
    searcher: &mut Searcher,
    root: &TraceRoot,
    ordering_weights: &NnueWeights,
    codebook_weights: &QuantizedCodebookWeights,
    measured: bool,
    arm: &str,
    ledger: &mut FailureLedger,
) -> Result<SearchObservation, String> {
    let mut board = root.board.clone();
    let root_signature = BoardSignature::from_board(&board);
    let root_zobrist = board.zobrist;
    let (result, ticks) = if measured {
        let started = qpc()?;
        let result = searcher.search_codebook_eval_quantized(
            &mut board,
            ordering_weights,
            codebook_weights,
            SEARCH_DEPTH,
            None,
        );
        let ticks = qpc_elapsed(started, qpc()?, "whole search root")?;
        (result, ticks)
    } else {
        (
            searcher.search_codebook_eval_quantized(
                &mut board,
                ordering_weights,
                codebook_weights,
                SEARCH_DEPTH,
                None,
            ),
            0,
        )
    };
    if measured && ticks == 0 {
        ledger.invalid(
            "zero_search_root_ticks",
            json!({"arm": arm, "root_index": root.root_index}),
        );
    }
    let shape = searcher.search_shape_stats();
    let signature = capture_search_signature(&result, shape, searcher, &board)?;
    if signature.main_plus_qsearch != u128::from(signature.returned_nodes)
        || signature.tt_shape_probes != signature.tt_probes
        || signature.tt_shape_hits != signature.tt_hits
    {
        ledger.invalid(
            "search_internal_counters",
            json!({
                "arm": arm,
                "root_index": root.root_index,
                "returned_nodes": signature.returned_nodes,
                "main_nodes": signature.main_nodes,
                "qsearch_nodes": signature.qsearch_nodes,
                "main_plus_qsearch": u128_string(signature.main_plus_qsearch),
                "shape_tt_probes": signature.tt_shape_probes,
                "tt_stats_probes": signature.tt_probes,
                "shape_tt_hits": signature.tt_shape_hits,
                "tt_stats_hits": signature.tt_hits,
            }),
        );
    }
    if signature.final_board != root_signature {
        ledger.reject(
            "search_board_restore",
            json!({
                "arm": arm,
                "root_index": root.root_index,
                "root": board_report(&root.board),
                "after_search": board_report(&board),
            }),
        );
    }
    Ok(SearchObservation {
        root_index: root.root_index,
        game_index: root.game_index,
        game_id: root.game_id,
        seed: root.seed,
        ply: root.ply,
        actual_move: root.actual_move,
        root_zobrist,
        ticks,
        signature,
        final_board_report: board_report(&board),
    })
}

fn compare_search_signatures(
    baseline: &SearchObservation,
    observed: &SearchObservation,
    arm: &str,
    ledger: &mut FailureLedger,
) {
    let identity_matches = baseline.root_index == observed.root_index
        && baseline.game_index == observed.game_index
        && baseline.game_id == observed.game_id
        && baseline.seed == observed.seed
        && baseline.ply == observed.ply
        && baseline.actual_move == observed.actual_move
        && baseline.root_zobrist == observed.root_zobrist;
    if !identity_matches {
        ledger.invalid(
            "search_root_pairing",
            json!({
                "arm": arm,
                "baseline_root": baseline.root_index,
                "observed_root": observed.root_index,
            }),
        );
        return;
    }
    let left = &baseline.signature;
    let right = &observed.signature;
    if (
        left.best_move,
        left.score,
        left.completed_depth,
        left.returned_nodes,
    ) != (
        right.best_move,
        right.score,
        right.completed_depth,
        right.returned_nodes,
    ) {
        ledger.reject(
            "search_result",
            json!({
                "arm": arm,
                "root_index": baseline.root_index,
                "baseline": {
                    "best_move": left.best_move,
                    "score": left.score,
                    "depth": left.completed_depth,
                    "nodes": left.returned_nodes,
                },
                "observed": {
                    "best_move": right.best_move,
                    "score": right.score,
                    "depth": right.completed_depth,
                    "nodes": right.returned_nodes,
                },
            }),
        );
    }
    if (
        left.main_nodes,
        left.qsearch_nodes,
        left.main_plus_qsearch,
        left.tt_shape_probes,
        left.tt_shape_hits,
        left.tt_cutoffs,
    ) != (
        right.main_nodes,
        right.qsearch_nodes,
        right.main_plus_qsearch,
        right.tt_shape_probes,
        right.tt_shape_hits,
        right.tt_cutoffs,
    ) {
        ledger.reject(
            "search_node_shape",
            json!({
                "arm": arm,
                "root_index": baseline.root_index,
                "baseline": {
                    "main": left.main_nodes,
                    "qsearch": left.qsearch_nodes,
                    "tt_probes": left.tt_shape_probes,
                    "tt_hits": left.tt_shape_hits,
                    "tt_cutoffs": left.tt_cutoffs,
                },
                "observed": {
                    "main": right.main_nodes,
                    "qsearch": right.qsearch_nodes,
                    "tt_probes": right.tt_shape_probes,
                    "tt_hits": right.tt_shape_hits,
                    "tt_cutoffs": right.tt_cutoffs,
                },
            }),
        );
    }
    if (
        left.tt_probes,
        left.tt_hits,
        left.tt_stores,
        left.tt_displaced_depth_pref,
        left.tt_stored_to_always,
        left.tt_depth_hist,
        left.tt_occupancy,
    ) != (
        right.tt_probes,
        right.tt_hits,
        right.tt_stores,
        right.tt_displaced_depth_pref,
        right.tt_stored_to_always,
        right.tt_depth_hist,
        right.tt_occupancy,
    ) {
        ledger.reject(
            "search_tt_state",
            json!({
                "arm": arm,
                "root_index": baseline.root_index,
                "baseline_signature": format!("{left:?}"),
                "observed_signature": format!("{right:?}"),
            }),
        );
    }
    if left.final_board != right.final_board {
        ledger.reject(
            "search_final_board",
            json!({
                "arm": arm,
                "root_index": baseline.root_index,
                "baseline_sha256": left.final_board.sha256,
                "observed_sha256": right.final_board.sha256,
            }),
        );
    }
}

fn run_search_warmup(
    trace: &FrozenTrace,
    ordering_weights: &NnueWeights,
    codebook_weights: &QuantizedCodebookWeights,
    ledger: &mut FailureLedger,
) -> Result<Value, String> {
    let mut arm_observations = Vec::with_capacity(TRANSITION_WARMUP_ORDER.len());
    for &arm in &TRANSITION_WARMUP_ORDER {
        let mut observations = Vec::with_capacity(EXPECTED_GAMES);
        for game in &trace.games {
            let root = game
                .roots
                .first()
                .ok_or_else(|| format!("game {} has no warmup root", game.game_id))?;
            let d4_enabled = arm == "B";
            let mut searcher = configured_product_searcher(d4_enabled)?;
            observations.push(run_search_root(
                &mut searcher,
                root,
                ordering_weights,
                codebook_weights,
                false,
                arm,
                ledger,
            )?);
        }
        arm_observations.push(observations);
    }
    if let Some(baseline) = arm_observations.first() {
        for observations in arm_observations.iter().skip(1) {
            for (left, right) in baseline.iter().zip(observations) {
                compare_search_signatures(left, right, "warmup", ledger);
            }
        }
    }

    let mut games = Vec::with_capacity(EXPECTED_GAMES);
    for (game_index, game) in trace.games.iter().enumerate() {
        let root = game
            .roots
            .first()
            .ok_or_else(|| format!("game {} has no warmup root", game.game_id))?;
        games.push(json!({
            "game_index": game.game_index,
            "game_id": game.game_id,
            "seed": game.seed,
            "root_index": root.root_index,
            "order": TRANSITION_WARMUP_ORDER,
            "outputs": arm_observations.iter().map(|observations| &observations[game_index]).map(|observation| json!({
                "best_move": observation.signature.best_move,
                "score": observation.signature.score,
                "depth": observation.signature.completed_depth,
                "nodes": observation.signature.returned_nodes,
                "board_sha256": observation.signature.final_board.sha256,
            })).collect::<Vec<_>>(),
            "contributes_to_timing": false,
        }));
    }
    let game_count = games.len();
    Ok(json!({
        "outermost_arm_order": TRANSITION_WARMUP_ORDER,
        "games": games,
        "game_count": game_count,
    }))
}

fn run_search_measured(
    trace: &FrozenTrace,
    ordering_weights: &NnueWeights,
    codebook_weights: &QuantizedCodebookWeights,
    ledger: &mut FailureLedger,
) -> Result<Vec<SearchArmRun>, String> {
    let mut arms = Vec::with_capacity(MEASURED_ARM_ORDER.len());
    for &label in &MEASURED_ARM_ORDER {
        let d4_enabled = label.starts_with('B');
        let mut observations = Vec::with_capacity(EXPECTED_ROOTS);
        for game in &trace.games {
            let mut searcher = configured_product_searcher(d4_enabled)?;
            for root in &game.roots {
                observations.push(run_search_root(
                    &mut searcher,
                    root,
                    ordering_weights,
                    codebook_weights,
                    true,
                    label,
                    ledger,
                )?);
            }
        }
        if observations.len() != EXPECTED_ROOTS {
            ledger.invalid(
                "search_arm_root_count",
                json!({
                    "arm": label,
                    "observed": observations.len(),
                    "expected": EXPECTED_ROOTS,
                }),
            );
        }
        arms.push(SearchArmRun {
            label,
            d4_enabled,
            observations,
        });
    }

    if let Some(baseline) = arms.first() {
        for arm in arms.iter().skip(1) {
            if arm.observations.len() != baseline.observations.len() {
                ledger.invalid(
                    "search_arm_pairing_count",
                    json!({
                        "arm": arm.label,
                        "observed": arm.observations.len(),
                        "baseline": baseline.observations.len(),
                    }),
                );
                continue;
            }
            for (left, right) in baseline.observations.iter().zip(&arm.observations) {
                compare_search_signatures(left, right, arm.label, ledger);
            }
        }
    }
    Ok(arms)
}

fn checked_tick_sum<I>(values: I, context: &str) -> Result<u128, String>
where
    I: IntoIterator<Item = u64>,
{
    let mut sum = 0u128;
    for value in values {
        sum = sum
            .checked_add(u128::from(value))
            .ok_or_else(|| format!("u128 tick sum overflow in {context}"))?;
    }
    Ok(sum)
}

fn checked_u128_sum<I>(values: I, context: &str) -> Result<u128, String>
where
    I: IntoIterator<Item = u128>,
{
    let mut sum = 0u128;
    for value in values {
        sum = sum
            .checked_add(value)
            .ok_or_else(|| format!("u128 tick sum overflow in {context}"))?;
    }
    Ok(sum)
}

fn checked_add_assign(target: &mut u128, value: u128, context: &str) -> Result<(), String> {
    *target = target
        .checked_add(value)
        .ok_or_else(|| format!("u128 tick sum overflow in {context}"))?;
    Ok(())
}

fn transition_cluster_pairs(arms: &[TransitionArmRun]) -> Result<Vec<(u128, u128)>, String> {
    if arms.len() != MEASURED_ARM_ORDER.len()
        || arms
            .iter()
            .zip(MEASURED_ARM_ORDER)
            .any(|(arm, expected)| arm.label != expected)
    {
        return Err("transition measured arm order/count mismatch".to_string());
    }
    for arm in arms {
        if arm.repetitions.len() != TRANSITION_REPETITIONS
            || arm
                .repetitions
                .iter()
                .any(|repetition| repetition.primary_cluster_ticks.len() != TRANSITION_CLUSTERS)
        {
            return Err(format!(
                "transition arm {} repetition/cluster count mismatch",
                arm.label
            ));
        }
    }

    let mut pairs = Vec::with_capacity(TRANSITION_CLUSTERS);
    for cluster in 0..TRANSITION_CLUSTERS {
        let mut a = 0u128;
        let mut b = 0u128;
        for arm in arms {
            let target = if arm.d4_enabled { &mut b } else { &mut a };
            for repetition in &arm.repetitions {
                checked_add_assign(
                    target,
                    repetition.primary_cluster_ticks[cluster],
                    "transition ABBA cluster",
                )?;
            }
        }
        if a == 0 || b == 0 {
            return Err(format!(
                "non-positive transition cluster total cluster={cluster} A={a} B={b}"
            ));
        }
        pairs.push((a, b));
    }
    Ok(pairs)
}

fn search_game_pairs(arms: &[SearchArmRun]) -> Result<Vec<(u128, u128)>, String> {
    if arms.len() != MEASURED_ARM_ORDER.len()
        || arms
            .iter()
            .zip(MEASURED_ARM_ORDER)
            .any(|(arm, expected)| arm.label != expected)
    {
        return Err("search measured arm order/count mismatch".to_string());
    }
    let mut a = vec![0u128; EXPECTED_GAMES];
    let mut b = vec![0u128; EXPECTED_GAMES];
    for arm in arms {
        if arm.observations.len() != EXPECTED_ROOTS {
            return Err(format!(
                "search arm {} root count mismatch: {}",
                arm.label,
                arm.observations.len()
            ));
        }
        for observation in &arm.observations {
            let target = if arm.d4_enabled {
                b.get_mut(observation.game_index)
            } else {
                a.get_mut(observation.game_index)
            }
            .ok_or_else(|| {
                format!(
                    "search observation game index out of range: {}",
                    observation.game_index
                )
            })?;
            checked_add_assign(
                target,
                u128::from(observation.ticks),
                "search ABBA game cluster",
            )?;
        }
    }
    let mut pairs = Vec::with_capacity(EXPECTED_GAMES);
    for game in 0..EXPECTED_GAMES {
        if a[game] == 0 || b[game] == 0 {
            return Err(format!(
                "non-positive search game total game={game} A={} B={}",
                a[game], b[game]
            ));
        }
        pairs.push((a[game], b[game]));
    }
    Ok(pairs)
}

fn sum_pairs(pairs: &[(u128, u128)], context: &str) -> Result<Rational, String> {
    let mut denominator = 0u128;
    let mut numerator = 0u128;
    for &(a, b) in pairs {
        checked_add_assign(&mut denominator, a, context)?;
        checked_add_assign(&mut numerator, b, context)?;
    }
    if denominator == 0 || numerator == 0 {
        return Err(format!(
            "non-positive global ratio in {context}: numerator={numerator} denominator={denominator}"
        ));
    }
    Ok(Rational {
        numerator,
        denominator,
    })
}

fn rational_cmp(left: &Rational, right: &Rational, overflow: &std::cell::Cell<bool>) -> Ordering {
    let lhs = left.numerator.checked_mul(right.denominator);
    let rhs = right.numerator.checked_mul(left.denominator);
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => lhs
            .cmp(&rhs)
            .then_with(|| left.numerator.cmp(&right.numerator))
            .then_with(|| left.denominator.cmp(&right.denominator)),
        _ => {
            overflow.set(true);
            left.numerator
                .cmp(&right.numerator)
                .then_with(|| left.denominator.cmp(&right.denominator))
        }
    }
}

fn paired_bootstrap(
    pairs: &[(u128, u128)],
    seed: u64,
    context: &str,
) -> Result<BootstrapResult, String> {
    if pairs.len() != BOOTSTRAP_DRAWS {
        return Err(format!(
            "{context} bootstrap cluster count mismatch: got {}, expected {BOOTSTRAP_DRAWS}",
            pairs.len()
        ));
    }
    let point = sum_pairs(pairs, context)?;
    let mut rng = SplitMix64::new(seed);
    let mut ratios = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    for replicate in 0..BOOTSTRAP_REPLICATES {
        let mut denominator = 0u128;
        let mut numerator = 0u128;
        for _ in 0..BOOTSTRAP_DRAWS {
            let index = (rng.next() & 63) as usize;
            let (a, b) = pairs[index];
            denominator = denominator
                .checked_add(a)
                .ok_or_else(|| format!("{context} bootstrap A overflow replicate={replicate}"))?;
            numerator = numerator
                .checked_add(b)
                .ok_or_else(|| format!("{context} bootstrap B overflow replicate={replicate}"))?;
        }
        if denominator == 0 || numerator == 0 {
            return Err(format!(
                "{context} bootstrap non-positive ratio replicate={replicate}"
            ));
        }
        ratios.push(Rational {
            numerator,
            denominator,
        });
    }
    let expected_draws = (BOOTSTRAP_REPLICATES as u64)
        .checked_mul(BOOTSTRAP_DRAWS as u64)
        .ok_or_else(|| format!("{context} bootstrap draw-count overflow"))?;
    if rng.draws != expected_draws {
        return Err(format!(
            "{context} bootstrap RNG draw mismatch got={} expected={expected_draws}",
            rng.draws
        ));
    }
    let maximum_numerator = ratios
        .iter()
        .map(|ratio| ratio.numerator)
        .max()
        .ok_or_else(|| format!("{context} bootstrap has no ratios"))?;
    let maximum_denominator = ratios
        .iter()
        .map(|ratio| ratio.denominator)
        .max()
        .ok_or_else(|| format!("{context} bootstrap has no denominators"))?;
    maximum_numerator
        .checked_mul(maximum_denominator)
        .ok_or_else(|| {
            format!("{context} exact rational comparator cross-product bound overflow")
        })?;
    let overflow = std::cell::Cell::new(false);
    ratios.sort_by(|left, right| rational_cmp(left, right, &overflow));
    if overflow.get() {
        return Err(format!(
            "{context} exact rational comparator cross-product overflow"
        ));
    }
    let upper = ratios
        .get(BOOTSTRAP_INDEX)
        .copied()
        .ok_or_else(|| format!("{context} bootstrap quantile index missing"))?;

    let point_left = point
        .numerator
        .checked_mul(200)
        .ok_or_else(|| format!("{context} point gate left overflow"))?;
    let point_right = point
        .denominator
        .checked_mul(201)
        .ok_or_else(|| format!("{context} point gate right overflow"))?;
    let upper_left = upper
        .numerator
        .checked_mul(100)
        .ok_or_else(|| format!("{context} upper gate left overflow"))?;
    let upper_right = upper
        .denominator
        .checked_mul(101)
        .ok_or_else(|| format!("{context} upper gate right overflow"))?;
    Ok(BootstrapResult {
        seed,
        final_rng_state: rng.state,
        rng_draws: rng.draws,
        point,
        upper,
        point_gate: point_left <= point_right,
        upper_gate: upper_left < upper_right,
    })
}

fn pair_report(pairs: &[(u128, u128)], kind: &str) -> Vec<Value> {
    pairs
        .iter()
        .enumerate()
        .map(|(index, &(a, b))| {
            json!({
                "cluster_kind": kind,
                "cluster_index": index,
                "A_ticks": u128_string(a),
                "B_ticks": u128_string(b),
            })
        })
        .collect()
}

fn ticks_to_rounded_ns(ticks: u128, frequency: u64) -> Result<u128, String> {
    let scaled = ticks
        .checked_mul(1_000_000_000)
        .ok_or_else(|| "tick-to-nanosecond multiplication overflow".to_string())?;
    let half = u128::from(frequency / 2);
    scaled
        .checked_add(half)
        .ok_or_else(|| "tick-to-nanosecond rounding overflow".to_string())
        .map(|value| value / u128::from(frequency))
}

fn sorted_tick_diagnostics(values: &[u64], frequency: u64) -> Result<Value, String> {
    if values.is_empty() {
        return Ok(json!({
            "count": 0,
            "p50_ticks": null,
            "p95_ticks": null,
            "maximum_ticks": null,
            "p50_rounded_ns": null,
            "p95_rounded_ns": null,
            "maximum_rounded_ns": null,
        }));
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let p50 = (sorted.len() * 50).div_ceil(100).saturating_sub(1);
    let p95 = (sorted.len() * 95).div_ceil(100).saturating_sub(1);
    let p50_ticks = sorted[p50];
    let p95_ticks = sorted[p95];
    let maximum_ticks = sorted[sorted.len() - 1];
    Ok(json!({
        "count": sorted.len(),
        "quantile_rule": "nearest-rank: zero-based ceil(p*n)-1",
        "p50_ticks": p50_ticks,
        "p95_ticks": p95_ticks,
        "maximum_ticks": maximum_ticks,
        "p50_rounded_ns": u128_string(ticks_to_rounded_ns(u128::from(p50_ticks), frequency)?),
        "p95_rounded_ns": u128_string(ticks_to_rounded_ns(u128::from(p95_ticks), frequency)?),
        "maximum_rounded_ns": u128_string(ticks_to_rounded_ns(u128::from(maximum_ticks), frequency)?),
    }))
}

fn transition_lifecycle_report(
    arms: &[TransitionArmRun],
    frequency: u64,
    pairs: &[(u128, u128)],
) -> Result<Value, String> {
    let point = sum_pairs(pairs, "transition lifecycle report")?;
    let a_ns = ticks_to_rounded_ns(point.denominator, frequency)?;
    let b_ns = ticks_to_rounded_ns(point.numerator, frequency)?;
    let per_transition_delta_ns = (point.numerator as f64 - point.denominator as f64)
        * 1_000_000_000.0
        / frequency as f64
        / ((TRANSITIONS * TRANSITION_REPETITIONS * 2) as f64);
    let mut arm_reports = Vec::with_capacity(arms.len());
    for arm in arms {
        let mut initial = Vec::with_capacity(TRANSITION_REPETITIONS);
        let mut rules = Vec::with_capacity(TRANSITION_REPETITIONS * RULE_SWITCHES);
        let mut transition_total = 0u128;
        let mut rebuild_total = 0u128;
        for repetition in &arm.repetitions {
            initial.push(repetition.initial_selector_ticks);
            rules.extend_from_slice(&repetition.rule_rebuild_ticks);
            let transition = checked_u128_sum(
                repetition.primary_cluster_ticks.iter().copied(),
                "transition lifecycle per-arm",
            )?;
            let rule = checked_tick_sum(
                repetition.rule_rebuild_ticks.iter().copied(),
                "rule lifecycle",
            )?;
            checked_add_assign(
                &mut transition_total,
                transition,
                "transition lifecycle arm total",
            )?;
            checked_add_assign(
                &mut rebuild_total,
                u128::from(repetition.initial_selector_ticks),
                "initial lifecycle arm total",
            )?;
            checked_add_assign(&mut rebuild_total, rule, "rule lifecycle arm total")?;
        }
        let combined = transition_total
            .checked_add(rebuild_total)
            .ok_or_else(|| "combined per-arm lifecycle overflow".to_string())?;
        arm_reports.push(json!({
            "arm": arm.label,
            "initial_enable_rebuild": sorted_tick_diagnostics(&initial, frequency)?,
            "rule_synchronization_rebuild": sorted_tick_diagnostics(&rules, frequency)?,
            "transition_only_total_ticks": u128_string(transition_total),
            "rebuild_only_total_ticks": u128_string(rebuild_total),
            "combined_lifecycle_total_ticks": u128_string(combined),
        }));
    }
    Ok(json!({
        "arms": arm_reports,
        "primary_transition_A_rounded_ns": u128_string(a_ns),
        "primary_transition_B_rounded_ns": u128_string(b_ns),
        "per_transition_B_minus_A_ns": per_transition_delta_ns,
        "primary_gate_excludes_rebuilds": true,
    }))
}

fn u128_string(value: u128) -> String {
    value.to_string()
}

fn ratio_decimal(numerator: u128, denominator: u128) -> String {
    format!("{:.12}", numerator as f64 / denominator as f64)
}

fn registered_constants() -> Value {
    json!({
        "format": FORMAT,
        "outcomes": [STATUS_INVALID, STATUS_REJECT, STATUS_TOO_COSTLY, STATUS_GO],
        "decision_precedence": [
            STATUS_INVALID,
            STATUS_REJECT,
            "both registered cost gates",
        ],
        "transition": {
            "seed_hex": format!("{TAPE_SEED:016X}"),
            "transitions": TRANSITIONS,
            "makes": MAKES,
            "undos": UNDOS,
            "prng_draws": PRNG_DRAWS,
            "rule_period": RULE_PERIOD,
            "rule_switches": RULE_SWITCHES,
            "maximum_move_count": MAX_STONES,
            "final_move_count": FINAL_MOVE_COUNT,
            "final_rng_state_hex": format!("{TAPE_FINAL_STATE:016X}"),
            "clusters": TRANSITION_CLUSTERS,
            "repetitions_per_arm": TRANSITION_REPETITIONS,
            "warmup_order": TRANSITION_WARMUP_ORDER,
            "measured_outermost_arm_order": MEASURED_ARM_ORDER,
        },
        "search": {
            "games": EXPECTED_GAMES,
            "roots": EXPECTED_ROOTS,
            "roots_per_game_min": SEARCH_MIN_ROOTS_PER_GAME,
            "roots_per_game_max": SEARCH_MAX_ROOTS_PER_GAME,
            "depth": SEARCH_DEPTH,
            "time_limit": null,
            "node_limit": null,
            "warmup_order": TRANSITION_WARMUP_ORDER,
            "measured_outermost_arm_order": MEASURED_ARM_ORDER,
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "draws_per_replicate": BOOTSTRAP_DRAWS,
            "transition_seed_hex": format!("{TRANSITION_BOOTSTRAP_SEED:016X}"),
            "search_seed_hex": format!("{SEARCH_BOOTSTRAP_SEED:016X}"),
            "quantile_zero_based_index": BOOTSTRAP_INDEX,
            "point_gate": "B*200 <= A*201",
            "upper_gate": "upper_num*100 < upper_den*101",
        },
        "inputs": INPUT_SPECS.iter().map(|spec| json!({
            "name": spec.name,
            "relative_path": spec.relative_path,
            "bytes": spec.bytes,
            "sha256": spec.sha256,
        })).collect::<Vec<_>>(),
        "critical_sources": CRITICAL_SOURCES,
    })
}

fn product_policy_report() -> Value {
    json!({
        "rule": "Freestyle",
        "runtime": "compact-flat quantized codebook",
        "directional_delta_CB_D1_CB_TD1": true,
        "packed_pattern4_windows": true,
        "exact_order_candidate_frontier": true,
        "white_root_ordering": true,
        "factored_CB_F1_runtime": false,
        "eager_threat_field": false,
        "lazy_threat_field": false,
        "move_picker": false,
        "tail_threat_materialization": false,
        "node_limit": null,
        "fixed_depth": SEARCH_DEPTH,
        "time_limit": null,
        "root_vct": "structurally OFF through per-Searcher audit selector",
        "defensive_root_vct_veto": "structurally OFF through same per-Searcher audit selector",
        "tt": "orientation-specific Zobrist TT; existing reset clears before every search",
        "A_d4_hash_sidecar": false,
        "B_d4_hash_sidecar": "ON, maintained, unconsumed",
    })
}

fn load_product_weights(
    inputs: &FrozenInputs,
) -> Result<(NnueWeights, QuantizedCodebookWeights, Value), String> {
    let raw_sha = sha256_hex(&inputs.raw_codebook);
    let artifact = PackedCodebookArtifact::parse(&inputs.compact_codebook)
        .map_err(|error| format!("compact-flat CBF parser failed: {error}"))?;
    if artifact.kind() != PackedCodebookKind::Flat {
        return Err(format!(
            "compact-flat artifact has kind {:?}, expected Flat",
            artifact.kind()
        ));
    }
    let linked_source_sha = hex_bytes(artifact.source_sha256());
    if !linked_source_sha.eq_ignore_ascii_case(&raw_sha) {
        return Err(format!(
            "compact-flat artifact source SHA mismatch linked={linked_source_sha} raw={raw_sha}"
        ));
    }
    let artifact_payload_len = artifact.artifact_payload_len();
    let codebook = artifact.into_flat_quantized()?;
    let ordering = NnueWeights::load_from_bytes(&inputs.nnue, Some(GOMOKU_NNUE_CONFIG))
        .map_err(|error| format!("flat NNUE parser failed: {error}"))?;
    let report = json!({
        "raw_codebook_identity_only": true,
        "raw_codebook_parsed": false,
        "raw_codebook_sha256": raw_sha,
        "compact_artifact_kind": "Flat",
        "compact_artifact_bytes": inputs.compact_codebook.len(),
        "compact_artifact_payload_bytes": artifact_payload_len,
        "compact_artifact_linked_source_sha256": linked_source_sha,
        "topk_identity_only": true,
        "topk_bytes": inputs.topk.len(),
        "flat_nnue_bytes": inputs.nnue.len(),
    });
    Ok((ordering, codebook, report))
}

fn run() -> Result<(), String> {
    let args = parse_args_from(env::args_os().skip(1))?;
    refuse_existing(&args.out_report)?;

    let manifest = manifest_dir()?;
    let source_pre = seal_source(&manifest)?;
    let executable_pre = seal_executable()?;
    let inputs = load_frozen_inputs(&manifest)?;
    let input_pre = inputs.seals.clone();
    let environment_pre = environment_identity()?;
    let toolchain = toolchain_identity()?;
    let cpu = cpu_identity()?;
    let tape = generate_transition_tape()?;
    let trace = parse_frozen_trace(&inputs.trace)?;
    let (ordering_weights, codebook_weights, weights_report) = load_product_weights(&inputs)?;

    let mut scheduling = SchedulingGuard::setup()?;
    let frequency_before = match qpf() {
        Ok(value) => value,
        Err(error) => {
            let restore = scheduling.restore();
            return Err(format!(
                "pre-timing QPF failure: {error}; explicit restore={restore:?}"
            ));
        }
    };
    let calibration = match calibrate_clock(frequency_before) {
        Ok(value) => value,
        Err(error) => {
            let restore = scheduling.restore();
            return Err(format!(
                "pre-timing clock calibration failure: {error}; explicit restore={restore:?}"
            ));
        }
    };

    let mut ledger = FailureLedger::default();
    let mut runtime_open = true;

    let transition_warmup = match run_transition_warmup(&tape, &mut ledger) {
        Ok(value) => value,
        Err(error) => {
            ledger.invalid("transition_warmup_runtime", json!({"error": error}));
            runtime_open = false;
            Value::Null
        }
    };
    let transition_arms = if runtime_open {
        match run_transition_measured(&tape, &mut ledger) {
            Ok(value) => value,
            Err(error) => {
                ledger.invalid("transition_measured_runtime", json!({"error": error}));
                runtime_open = false;
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    let search_warmup = if runtime_open {
        match run_search_warmup(&trace, &ordering_weights, &codebook_weights, &mut ledger) {
            Ok(value) => value,
            Err(error) => {
                ledger.invalid("search_warmup_runtime", json!({"error": error}));
                runtime_open = false;
                Value::Null
            }
        }
    } else {
        Value::Null
    };
    let search_arms = if runtime_open {
        match run_search_measured(&trace, &ordering_weights, &codebook_weights, &mut ledger) {
            Ok(value) => value,
            Err(error) => {
                ledger.invalid("search_measured_runtime", json!({"error": error}));
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    let transition_pairs = match transition_cluster_pairs(&transition_arms) {
        Ok(value) => Some(value),
        Err(error) => {
            ledger.invalid("transition_abba_aggregation", json!({"error": error}));
            None
        }
    };
    let search_pairs = match search_game_pairs(&search_arms) {
        Ok(value) => Some(value),
        Err(error) => {
            ledger.invalid("search_abba_aggregation", json!({"error": error}));
            None
        }
    };
    let transition_bootstrap = match transition_pairs.as_deref() {
        Some(pairs) => match paired_bootstrap(pairs, TRANSITION_BOOTSTRAP_SEED, "transition") {
            Ok(value) => Some(value),
            Err(error) => {
                ledger.invalid("transition_bootstrap_protocol", json!({"error": error}));
                None
            }
        },
        None => None,
    };
    let search_bootstrap = match search_pairs.as_deref() {
        Some(pairs) => match paired_bootstrap(pairs, SEARCH_BOOTSTRAP_SEED, "search") {
            Ok(value) => Some(value),
            Err(error) => {
                ledger.invalid("search_bootstrap_protocol", json!({"error": error}));
                None
            }
        },
        None => None,
    };

    let frequency_after = match qpf() {
        Ok(value) => {
            if value != frequency_before {
                ledger.invalid(
                    "qpc_frequency_changed",
                    json!({"before": frequency_before, "after": value}),
                );
            }
            Some(value)
        }
        Err(error) => {
            ledger.invalid("qpc_postflight", json!({"error": error}));
            None
        }
    };
    let restore_error = scheduling.restore().err();
    if let Some(error) = &restore_error {
        ledger.invalid("scheduling_restore", json!({"error": error}));
    }
    let scheduling_report = scheduling.snapshot.report(scheduling.restored);

    let source_post = match seal_source(&manifest) {
        Ok(value) => {
            if value != source_pre {
                ledger.invalid(
                    "source_postflight_mismatch",
                    json!({
                        "pre": source_pre.report(),
                        "post": value.report(),
                    }),
                );
            }
            Some(value)
        }
        Err(error) => {
            ledger.invalid("source_postflight", json!({"error": error}));
            None
        }
    };
    let executable_post = match seal_executable() {
        Ok(value) => {
            if value != executable_pre {
                ledger.invalid(
                    "executable_postflight_mismatch",
                    json!({
                        "pre": executable_pre.report(),
                        "post": value.report(),
                    }),
                );
            }
            Some(value)
        }
        Err(error) => {
            ledger.invalid("executable_postflight", json!({"error": error}));
            None
        }
    };
    let input_post = match reseal_inputs(&manifest) {
        Ok(value) => {
            if value != input_pre {
                ledger.invalid(
                    "input_postflight_mismatch",
                    json!({
                        "pre": input_pre.iter().map(FileSeal::report).collect::<Vec<_>>(),
                        "post": value.iter().map(FileSeal::report).collect::<Vec<_>>(),
                    }),
                );
            }
            Some(value)
        }
        Err(error) => {
            ledger.invalid("input_postflight", json!({"error": error}));
            None
        }
    };
    let environment_post = match environment_identity() {
        Ok(value) => {
            if value != environment_pre {
                ledger.invalid(
                    "environment_postflight_mismatch",
                    json!({"pre": environment_pre, "post": value}),
                );
            }
            Some(value)
        }
        Err(error) => {
            ledger.invalid("environment_postflight", json!({"error": error}));
            None
        }
    };

    let transition_raw = match transition_arms
        .iter()
        .map(TransitionArmRun::report)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(value) => Value::Array(value),
        Err(error) => {
            ledger.invalid("transition_report_arithmetic", json!({"error": error}));
            Value::Null
        }
    };
    let transition_lifecycle = match transition_pairs.as_deref() {
        Some(pairs) => match transition_lifecycle_report(&transition_arms, frequency_before, pairs)
        {
            Ok(value) => value,
            Err(error) => {
                ledger.invalid("transition_lifecycle_arithmetic", json!({"error": error}));
                Value::Null
            }
        },
        None => Value::Null,
    };
    let search_raw = Value::Array(search_arms.iter().map(SearchArmRun::report).collect());

    let cost_passed = transition_bootstrap
        .as_ref()
        .zip(search_bootstrap.as_ref())
        .map(|(transition, search)| transition.passed() && search.passed());
    let decision = ledger.decision(cost_passed);
    let report = json!({
        "format": FORMAT,
        "claim_boundary": {
            "candidate": "default-OFF exact D4 hash sidecar maintained but unconsumed",
            "canonical_TT_score_bound_move_sharing": false,
            "proof_cache_identity_by_u64": false,
            "arena_or_playing_strength_claim": false,
            "pbrain_environment_switch": false,
            "product_default_change": false,
        },
        "decision": decision,
        "cost_gates_evaluated": cost_passed.is_some(),
        "cost_gates_passed": cost_passed,
        "failures": ledger.report(),
        "registered_constants": registered_constants(),
        "product_policy": product_policy_report(),
        "preflight": {
            "source": source_pre.report(),
            "executable": executable_pre.report(),
            "inputs": input_pre.iter().map(FileSeal::report).collect::<Vec<_>>(),
            "environment": environment_pre,
            "toolchain": toolchain,
            "cpu": cpu,
            "weights": weights_report,
        },
        "scheduling": scheduling_report,
        "clock": calibration.report(frequency_after),
        "transition_tape": {
            "serialization_sha256": tape.serialized_sha256,
            "entries": tape.entries.len(),
            "makes": tape.makes,
            "undos": tape.undos,
            "rule_switches": tape.rule_switches,
            "maximum_move_count": tape.maximum_move_count,
            "final_move_count": tape.final_move_count,
            "rng_draws": tape.rng_draws,
            "final_rng_state_hex": format!("{:016X}", tape.final_rng_state),
            "blocks": tape.blocks.iter().enumerate().map(|(index, block)| json!({
                "cluster": index,
                "start": block.start,
                "end": block.end,
                "transitions": block.end - block.start,
                "state_digest_sha256": block.digest,
            })).collect::<Vec<_>>(),
        },
        "trace": {
            "games": trace.games.len(),
            "roots": trace.root_count,
            "selection_rule": "first 1,022 states in file order before moves with source==engine && side_to_move==unique case-insensitive figrid engine side; stride=1",
            "per_game": trace.games.iter().map(|game| json!({
                "game_index": game.game_index,
                "game_id": game.game_id,
                "seed": game.seed,
                "selected_roots": game.roots.len(),
                "first_root_index": game.roots.first().map(|root| root.root_index),
                "last_root_index": game.roots.last().map(|root| root.root_index),
            })).collect::<Vec<_>>(),
        },
        "warmups": {
            "transition": transition_warmup,
            "search": search_warmup,
        },
        "raw_measurements": {
            "outermost_arm_order": MEASURED_ARM_ORDER,
            "transition_arms": transition_raw,
            "transition_lifecycle_diagnostics": transition_lifecycle,
            "search_arms": search_raw,
        },
        "paired_clusters": {
            "transition": transition_pairs.as_deref().map(|pairs| pair_report(pairs, "transition_cluster")),
            "search": search_pairs.as_deref().map(|pairs| pair_report(pairs, "game_cluster")),
        },
        "bootstrap": {
            "transition": transition_bootstrap.as_ref().map(BootstrapResult::report),
            "search": search_bootstrap.as_ref().map(BootstrapResult::report),
        },
        "postflight": {
            "source": source_post.as_ref().map(SourceSeal::report),
            "executable": executable_post.as_ref().map(FileSeal::report),
            "inputs": input_post.as_ref().map(|seals| seals.iter().map(FileSeal::report).collect::<Vec<_>>()),
            "environment": environment_post,
            "qpc_frequency": frequency_after,
            "scheduling_restore_error": restore_error,
        },
        "independent_artifact_audit": {
            "status": "REQUIRED_BEFORE_RESULT_COMMIT",
            "must_reproduce": [
                "counts",
                "ABBA sums",
                "exact cross-product decisions",
                "final label",
            ],
        },
    });

    let serialized = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("failed to serialize authoritative report: {error}"))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.out_report)
        .map_err(|error| {
            format!(
                "failed to create-new report {}: {error}",
                args.out_report.display()
            )
        })?;
    output
        .write_all(&serialized)
        .map_err(|error| format!("failed to write authoritative report: {error}"))?;
    output
        .write_all(b"\n")
        .map_err(|error| format!("failed to terminate authoritative report: {error}"))?;
    output
        .flush()
        .map_err(|error| format!("failed to flush authoritative report: {error}"))?;
    output
        .sync_all()
        .map_err(|error| format!("failed to sync authoritative report: {error}"))?;
    println!(
        "created authoritative report {} decision={decision}",
        args.out_report.display()
    );
    Ok(())
}

fn hex_bytes(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789ABCDEF";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
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
            let offset = index * 4;
            schedule[index] = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
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
    let mut digest = [0u8; 32];
    for (index, word) in state.iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    hex_bytes(&digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_accepts_only_one_create_new_output_option() {
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
    fn transition_cluster_geometry_is_exact() {
        assert_eq!(block_start(0), 0);
        assert_eq!(block_start(32), 50_016);
        assert_eq!(block_start(64), TRANSITIONS);
        for cluster in 0..TRANSITION_CLUSTERS {
            assert_eq!(
                block_start(cluster + 1) - block_start(cluster),
                if cluster < 32 { 1_563 } else { 1_562 }
            );
        }
    }

    #[test]
    fn transition_tape_matches_frozen_counts() {
        let tape = generate_transition_tape().unwrap();
        assert_eq!(tape.entries.len(), TRANSITIONS);
        assert_eq!(tape.makes, MAKES);
        assert_eq!(tape.undos, UNDOS);
        assert_eq!(tape.rule_switches, RULE_SWITCHES);
        assert_eq!(tape.maximum_move_count, MAX_STONES);
        assert_eq!(tape.final_move_count, FINAL_MOVE_COUNT);
        assert_eq!(tape.rng_draws, PRNG_DRAWS);
        assert_eq!(tape.final_rng_state, TAPE_FINAL_STATE);
        assert_eq!(tape.serialized_sha256.len(), 64);
    }

    #[test]
    fn unmeasured_transition_replay_restores_exact_root() {
        let tape = generate_transition_tape().unwrap();
        let mut ledger = FailureLedger::default();
        let replay =
            run_transition_repetition(&tape, true, false, "test-B", 0, &mut ledger).unwrap();
        assert!(ledger.invalid.is_empty(), "{:?}", ledger.invalid);
        assert!(ledger.reject.is_empty(), "{:?}", ledger.reject);
        assert_eq!(replay.block_state_digests.len(), TRANSITION_CLUSTERS);
        assert_eq!(replay.rule_rebuild_ticks.len(), RULE_SWITCHES);
        assert!(replay.primary_cluster_ticks.iter().all(|&ticks| ticks == 0));
        assert_eq!(
            replay.unwind_state_digest,
            sha256_hex(&board_signature_bytes(&Board::new()))
        );
    }

    #[test]
    fn frozen_trace_parser_selects_registered_roots() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(INPUT_SPECS[4].relative_path);
        let bytes = fs::read(path).unwrap();
        let trace = parse_frozen_trace(&bytes).unwrap();
        assert_eq!(trace.games.len(), EXPECTED_GAMES);
        assert_eq!(trace.root_count, EXPECTED_ROOTS);
        assert!(trace.games.iter().all(|game| {
            (SEARCH_MIN_ROOTS_PER_GAME..=SEARCH_MAX_ROOTS_PER_GAME).contains(&game.roots.len())
        }));
    }

    #[test]
    fn board_signature_covers_rule_history_and_pattern_cache() {
        let empty = Board::new();
        let mut board = empty.clone();
        board.make_move(to_idx(7, 7));
        assert_ne!(board_signature_bytes(&empty), board_signature_bytes(&board));
        board.undo_move();
        assert_eq!(board_signature_bytes(&empty), board_signature_bytes(&board));
        board.exact5 = true;
        assert_ne!(board_signature_bytes(&empty), board_signature_bytes(&board));
    }

    #[test]
    fn rational_comparison_and_exact_gates_are_ordered() {
        let overflow = std::cell::Cell::new(false);
        let a = Rational {
            numerator: 201,
            denominator: 200,
        };
        let b = Rational {
            numerator: 101,
            denominator: 100,
        };
        assert_eq!(rational_cmp(&a, &b, &overflow), Ordering::Less);
        assert!(!overflow.get());
        assert!(a.numerator * 200 <= a.denominator * 201);
        assert!(b.numerator * 100 >= b.denominator * 101);
    }

    #[test]
    fn node_sum_preserves_two_u64_maxima_exactly() {
        assert_eq!(
            checked_node_sum(u64::MAX, u64::MAX).unwrap(),
            u128::from(u64::MAX) + u128::from(u64::MAX)
        );
    }

    #[test]
    fn paired_bootstrap_keeps_identical_clusters_at_exact_unity() {
        let pairs = vec![(100u128, 100u128); BOOTSTRAP_DRAWS];
        let result = paired_bootstrap(&pairs, TRANSITION_BOOTSTRAP_SEED, "test").unwrap();
        assert_eq!(
            result.point,
            Rational {
                numerator: 6_400,
                denominator: 6_400,
            }
        );
        assert_eq!(result.upper, result.point);
        assert_eq!(
            result.rng_draws,
            (BOOTSTRAP_REPLICATES * BOOTSTRAP_DRAWS) as u64
        );
        assert!(result.passed());
    }

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
        );
    }

    #[test]
    fn product_searcher_selectors_are_explicit() {
        for d4_enabled in [false, true] {
            let searcher = configured_product_searcher(d4_enabled).unwrap();
            assert_eq!(searcher.d4_hash_sidecar_requested(), d4_enabled);
            assert!(!searcher.root_vct_requested_for_audit());
            assert!(!searcher.use_threat_field());
            assert!(searcher.white_root_order_enabled());
        }
    }
}
