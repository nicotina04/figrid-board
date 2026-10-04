use crate::board::RuleSet;
/// ????????ㅻ깹???????????????袁ｋ쨨?? ??????轅붽틓?????(NNUE ???)
///
/// - ?????獄쏅챶留덌┼???????????(Iterative Deepening)
/// - ????????ㅻ깹???????????????袁ｋ쨨?? ???????ル???????븐뼐???????????筌뤾퍓愿?????????ъ몴??/// - ??????????뀀?????? ????????????????????遺얘턁????????????(4??????? ???????ル????)
/// - ?????????+ ????????⑤벡????????????????????????紐껊짍
/// - ??????????????
use crate::board::{
    BOARD_SIZE, BitBoard, Board, BoardSearchState, GameResult, Move, NUM_CELLS, Stone,
};
#[cfg(feature = "codebook-eval")]
use crate::codebook_eval::{
    CodebookWeights, IncrementalCodebookEval, IncrementalQuantizedCodebookEval,
    QuantizedCodebookAccess, QuantizedCodebookWeights,
};
use crate::eval::IncrementalEval;
#[cfg(feature = "codebook-eval")]
use crate::factored_codebook::FactoredQuantizedCodebookWeights;
use crate::heuristic::{DIR, scan_line};
use crate::threat_field::{IncrementalThreatField, ThreatFieldUpdateMode};
use crate::transposition::{Bound, TranspositionTable, TtStats};
use crate::vct::{
    THREAT_KIND_COUNT, ThreatKind, VctConfig, classify_move_fast, search_vct,
    search_vct_with_board_search_state,
};
#[cfg(feature = "codebook-eval")]
use crate::white_root_order::WhiteRootOrder;
use noru::network::NnueWeights;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const INF: i32 = 1_000_000;
const WIN_SCORE: i32 = 999_000;

/// Root VCT ????????????????????(time_limit???????????대첉????????. ??????袁⑸즴筌?씛彛?????????????depth ??????
const ROOT_VCT_BUDGET_MS: u64 = 150;
/// Root VCT ????븐뼐????????? ??? ?????????????(???????????????????.
const ROOT_VCT_DEPTH: u32 = 14;
/// Root VCT???????ル??? ????????????????븐뼐??????⑤슢?????壤굿??띾????????? 5s ????VCT 625ms, 30s ????3.75s.
const ROOT_VCT_BUDGET_FRACTION: u32 = 8;
/// Root VCT ?????????????⑥ル츧癲??(?????????????遺얘턁?????????. 2??
const ROOT_VCT_BUDGET_CAP_MS: u64 = 2_000;
/// Root VCT ?????????????(????????븐뼐???????????????????TT warmup??????.
const ROOT_VCT_BUDGET_FLOOR_MS: u64 = 100;

fn env_flag_enabled(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|raw| {
            let trimmed = raw.trim();
            // An empty value means "unset" (wrappers blank variables to clear them).
            if trimmed.is_empty() {
                return default;
            }
            !(trimmed == "0"
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no"))
        })
        .unwrap_or(default)
}

fn search_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("NORU_SEARCH_PROFILE", false))
}

/// P-CB1 policy-order table (`PCB1v1` binary: magic, u32 n, f32 A[n], f32
/// D[n]). Loaded once from `NORU_POLICY_ORDER`; default OFF. A set-but-bad
/// path panics loudly so an arena can never silently run without it.
/// NG-P1: n = 199,827 selects the untruncated full-vocab id space (ids
/// computed from the bitboards per candidate cell); n = 4,266 keeps the
/// legacy board-maintained ids unchanged.
pub struct PolicyOrderTable {
    attack: Vec<f32>,
    defense: Vec<f32>,
    full_vocab: bool,
    /// Auto-desaturation divisor applied at load (`None` = table untouched).
    desat_k: Option<f64>,
}

/// Quiet-score saturation guard. The runtime quiet score is
/// `clamp((s * 10_000) as i32, ±300_000)` with `s` a sum of 8 entries
/// (A[my] + D[opp] over 4 directions), so any |s| >= 30 collapses onto the
/// clamp and loses its ordering. If `8 * max|entry| >= 30` both tables are
/// divided by `k = 8 * max|entry| / 29` (f64 math, stored back as f32) so
/// every reachable sum stays strictly inside the band; otherwise the table
/// is left bit-identical. Returns the applied `k`.
fn desaturate_policy_tables(attack: &mut [f32], defense: &mut [f32]) -> Option<f64> {
    let max_abs = attack
        .iter()
        .chain(defense.iter())
        .fold(0.0f64, |m, &v| m.max((v as f64).abs()));
    let bound = 8.0 * max_abs;
    if bound < 30.0 {
        return None;
    }
    let k = bound / 29.0;
    for v in attack.iter_mut().chain(defense.iter_mut()) {
        *v = ((*v as f64) / k) as f32;
    }
    Some(k)
}

impl PolicyOrderTable {
    fn load(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        if bytes.len() < 12 || &bytes[..8] != b"PCB1v1\0\0" {
            return Err(format!("{path}: bad PCB1 header"));
        }
        let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let full_vocab = n == crate::pattern_table::FULL_PATTERN_NUM_IDS;
        if bytes.len() != 12 + n * 8
            || (!full_vocab && n != crate::pattern_table::PATTERN_NUM_IDS)
        {
            return Err(format!("{path}: bad PCB1 length (n={n})"));
        }
        if full_vocab {
            crate::pattern_table::warm_full_vocab();
        }
        let read = |off: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    f32::from_le_bytes(bytes[off + i * 4..off + i * 4 + 4].try_into().unwrap())
                })
                .collect()
        };
        let mut attack = read(12);
        let mut defense = read(12 + n * 4);
        let desat_k = desaturate_policy_tables(&mut attack, &mut defense);
        Ok(Self {
            attack,
            defense,
            full_vocab,
            desat_k,
        })
    }

    /// Divisor applied by load-time auto-desaturation, if any.
    pub fn desat_k(&self) -> Option<f64> {
        self.desat_k
    }

    /// Learned quiet score scaled into the sub-tier integer band.
    #[inline]
    fn quiet_score(&self, board: &Board, mv: Move) -> i32 {
        let white_to_move = board.side_to_move == Stone::White;
        let mut s = 0.0f32;
        if self.full_vocab {
            let ids = crate::pattern_table::full_ids_for_cell(&board.black, &board.white, mv);
            for &id in &ids {
                let swapped = crate::pattern_table::full_swap_id(id);
                let (my_id, opp_id) = if white_to_move {
                    (swapped, id)
                } else {
                    (id, swapped)
                };
                s += self.attack[my_id as usize] + self.defense[opp_id as usize];
            }
        } else {
            let ids = board.line_pattern_ids[mv];
            for &id in &ids {
                let (my_id, opp_id) = if white_to_move {
                    (crate::pattern_table::swap_mapped_id(id), id)
                } else {
                    (id, crate::pattern_table::swap_mapped_id(id))
                };
                s += self.attack[my_id as usize] + self.defense[opp_id as usize];
            }
        }
        ((s * 10_000.0) as i32).clamp(-300_000, 300_000)
    }
}

/// Loads the `NORU_POLICY_ORDER` table once. `Ok(None)` = unset/empty,
/// `Err` = set but unreadable/invalid.
fn policy_order_table_init() -> &'static Result<Option<(String, PolicyOrderTable)>, String> {
    static TABLE: OnceLock<Result<Option<(String, PolicyOrderTable)>, String>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let Ok(path) = std::env::var("NORU_POLICY_ORDER") else {
            return Ok(None);
        };
        let path = path.trim();
        if path.is_empty() {
            return Ok(None);
        }
        PolicyOrderTable::load(path)
            .map(|table| Some((path.to_string(), table)))
            .map_err(|e| format!("NORU_POLICY_ORDER: {e}"))
    })
}

fn policy_order_table() -> Option<&'static PolicyOrderTable> {
    match policy_order_table_init() {
        Ok(table) => table.as_ref().map(|(_, table)| table),
        Err(e) => panic!("{e}"),
    }
}

/// P-CB2: policy-rank-conditioned reduction/pruning.
/// Active only when `NORU_POLICY_REDUCE` is truthy AND the policy-order
/// table is loaded (quiet band is then policy-sorted, so move_idx IS the
/// policy rank). A set flag without a table fails loudly.
fn policy_reduce_init() -> &'static Result<bool, String> {
    static ENABLED: OnceLock<Result<bool, String>> = OnceLock::new();
    ENABLED.get_or_init(|| {
        let on = std::env::var("NORU_POLICY_REDUCE")
            .map(|raw| {
                let t = raw.trim();
                !(t.is_empty()
                    || t.eq_ignore_ascii_case("0")
                    || t.eq_ignore_ascii_case("false")
                    || t.eq_ignore_ascii_case("off")
                    || t.eq_ignore_ascii_case("no"))
            })
            .unwrap_or(false);
        if on && !matches!(policy_order_table_init(), Ok(Some(_))) {
            return Err("NORU_POLICY_REDUCE requires NORU_POLICY_ORDER to be loaded".to_string());
        }
        Ok(on)
    })
}

fn policy_reduce_enabled() -> bool {
    match policy_reduce_init() {
        Ok(on) => *on,
        Err(e) => panic!("{e}"),
    }
}

fn forced_restriction_enabled_by_env() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("NORU_FORCED_REPLY_RESTRICTION", false))
}

/// Resolves every env-driven search switch now (instead of lazily at the
/// first search) and returns their effective state as `key=value` pairs
/// joined by `"; "`. A set-but-invalid `NORU_POLICY_ORDER` or a
/// `NORU_POLICY_REDUCE` without a table is reported as `Err` here rather
/// than panicking mid-game.
pub fn runtime_config_summary() -> Result<String, String> {
    let policy_order = match policy_order_table_init() {
        Ok(None) => "off".to_string(),
        Ok(Some((path, table))) => match table.desat_k() {
            Some(k) => format!("{path} (desat k={k:.4})"),
            None => format!("{path} (desat none)"),
        },
        Err(e) => return Err(e.clone()),
    };
    let policy_reduce = policy_reduce_init().clone()?;
    let on_off = |b: bool| if b { "on" } else { "off" };
    Ok(format!(
        "policy_order={policy_order}; policy_reduce={}; forced_reply_restriction={}; search_profile={}",
        on_off(policy_reduce),
        on_off(forced_restriction_enabled_by_env()),
        on_off(search_profile_enabled()),
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchThreatFieldMode {
    Off,
    Eager,
    Lazy,
}

#[cfg(feature = "codebook-eval")]
const DEFAULT_CODEBOOK_EVAL_SCALE: f32 = 15.720162;

/// Effective codebook eval scale (`NORU_CODEBOOK_EVAL_SCALE` if it parses
/// to a finite positive float, else the built-in default).
#[cfg(feature = "codebook-eval")]
pub fn codebook_eval_scale() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NORU_CODEBOOK_EVAL_SCALE")
            .ok()
            .and_then(|raw| raw.trim().parse::<f32>().ok())
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .unwrap_or(DEFAULT_CODEBOOK_EVAL_SCALE)
    })
}

/// ????TT ??????????????= 2^N. 18 ??262 144 ????????????= 8 MB.
/// 0.6.1 ????븐뼐????????2026-04-27)?????16 bits(2 MB)???????displaced 28.5% / always-replace
/// ????2.1%??collision-driven eviction ?????獄쏅챶留덌┼????????ル뒌嶺뚮씮????????? 18 bits?????????ル???? ??癲됱빖???嶺????????/// displaced 5~10% ?????? Piskvork ????????븐뼐?????????????獄쏅챶留?????50 MB ???? ??????????????????????????대첉??
const TT_BUCKET_BITS: u32 = 18;

/// Aspiration windows: ?????????대첐??iteration score ??????獄쏅챷??? ???????筌띿솘?? window. depth ??????/// ???????깆궔?????????????泥?? 1~3 ply??score ??????⑤벡瑜??????????숈?????????????????widening cost???????ル??? ???????????????????????????????
const ASPIRATION_MIN_DEPTH: u32 = 4;
/// ??????嶺뚮∥????window half-width (centipawn). ?????????耀붾굝????癲ル슢???苡??????widening ????? ??????????????????
/// ?????? ??癲됱빖???嶺??????븐뼐????傭?끆?????Β?ｊ콞???癲ル슢?????????멸괜??chess engine 50~100 ???遺얘턁???????? ?????????녳븢??BCE eval scale ??????? 50.
const ASPIRATION_INITIAL_DELTA: i32 = 50;

/// Quiescence lite ????븐뼐????????? ply. depth==0 ??????袁⑸즴筌?씛彛?????NNUE static eval ?????獄쏅챶留덌┼???????????
/// ???????ル??????????븐뼐????傭?끆?????Β?ｊ콞?轅붽틓?????⑸걦????????븐뼐????傭?끆?????Β?ｊ콞?轅붽틓?????⑸걦?????????????밸븶筌믩끃??獄???輿???????????쇈궘?/????癲???/4-3/?????????쇈궘? ????븐뼐??????⑤슢?????壤굿??띾????????ply ???耀붾굝???????????????몃뒌?????썹땟戮녹????
/// ??????熬곣몿??? ????????????源낅???horizon effect ??????????뀀?? ????????????????stand-pat. Codex ??????/// (2026-04-26): "win/must-block/open-four/double-four/four-three ???遺얘턁????????????????援온?????????/// 2-4 ply ????????. ????????leaf ???? ?????耀붾굝????癲ル슢???苡??????????????????????????곗뿨????嚥▲굧?먩뤆??.
const QSEARCH_MAX_PLY: u32 = 4;

/// Threat-gated LMR (Late Move Reductions): non-PV / non-killer / non-forcing
/// ?????耀붾굝????????r ply ??????ш끽踰椰????????????????????????fail-high ??full depth???????
/// "naive LMR -43%p" ??????????? ???????ル???????????繹먮굞??????븐뼐?????? reduce?????????????tier ???????????????/// gating??????????????????뀀?????????濡?씀?濾???? ??? ??????ш끽踰椰???????????????? ?????????몃뒇??? 5??????????????ply ????????????????????ル?????/// ????????????????????ル??? ????븐뼐????????????????泥????????諛몃마嶺뚮??????????????轅붽틓????
const LMR_MIN_DEPTH: u32 = 3;
const LMR_MIN_MOVE_IDX: usize = 3;

/// IIR (Internal Iterative Reduction): TT-miss ???遺얘턁????????ㅼ뒧??띤겫??눫??+ non-PV + ???汝뷴젆?琉??誘↔덱??????????????????????/// 1 ply ??????ш끽踰椰??????????????????????????⑤벡瑜??꿔꺂?????? TT-miss?????????? PV ?????耀붾굝????????????븐뼐???????????븐뼔???????????椰????????遺얘턁???????????ㅿ폍??/// search ??ordering?????????cutoff ?????? 1 ply ??????ш끽踰椰??????????????????????耀붾굝?????????/// store??entry?????????롮쾸?椰???iteration?????PV ???遺얘턁????????? chess engine???????癲됱빖???嶺??????븐뼐????傭?끆?????Β?ｊ콞???癲ル슢?????????멸괜??/// cheap ????????????(~+30 elo).
const IIR_MIN_DEPTH: u32 = 4;

/// LMP (Late Move Pruning): ??? ??PV ???遺얘턁????????ㅼ뒧??띤겫??눫??????move_idx???????ル??? ??????????????????????????/// ??forcing / ??killer ?????skip. count ???????????????????razoring/futility???????ル???
/// ?????????곕츧???????嚥▲굧?먩뤆??NNUE eval scale ???????筌띿솘??????????????????熬곣몿??? tier ???遺얘턁???????????????????밸븶筌믩끃??獄?????????????????quiet
/// move?????????? ??????袁⑸즴筌?씛彛???????????ル???????????????븐뼐????????????
const LMP_MIN_DEPTH: u32 = 1;
const LMP_MAX_DEPTH: u32 = 3;
const LMP_BASE: usize = 8;
const LMP_PER_DEPTH: usize = 4;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: u32,
    pub nodes: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EvalStateStepProfile {
    pub dirty_list_ns: u128,
    pub dirty_list_calls: u64,
    pub frame_write_ns: u128,
    pub frame_write_calls: u64,
    pub backup_ns: u128,
    pub backup_calls: u64,
    pub recompute_ns: u128,
    pub recompute_calls: u64,
    pub aggregate_ns: u128,
    pub aggregate_calls: u64,
    pub restore_ns: u128,
    pub restore_calls: u64,
    pub forward_ns: u128,
    pub forward_calls: u64,
    pub push_calls: u64,
    pub pop_calls: u64,
}

impl EvalStateStepProfile {
    #[inline]
    pub(crate) fn start(profile_enabled: bool) -> Option<Instant> {
        profile_enabled.then(Instant::now)
    }

    #[inline]
    pub(crate) fn elapsed(start: Option<Instant>) -> u128 {
        start.map(|t| t.elapsed().as_nanos()).unwrap_or(0)
    }

    #[inline]
    pub(crate) fn add_backup(&mut self, start: Option<Instant>) {
        if start.is_some() {
            self.backup_ns += Self::elapsed(start);
            self.backup_calls += 1;
        }
    }

    #[inline]
    pub(crate) fn add_dirty_list(&mut self, start: Option<Instant>) {
        if start.is_some() {
            self.dirty_list_ns += Self::elapsed(start);
            self.dirty_list_calls += 1;
        }
    }

    #[inline]
    pub(crate) fn add_frame_write(&mut self, start: Option<Instant>) {
        if start.is_some() {
            self.frame_write_ns += Self::elapsed(start);
            self.frame_write_calls += 1;
        }
    }

    #[inline]
    pub(crate) fn add_recompute(&mut self, start: Option<Instant>) {
        if start.is_some() {
            self.recompute_ns += Self::elapsed(start);
            self.recompute_calls += 1;
        }
    }

    #[inline]
    pub(crate) fn add_aggregate(&mut self, start: Option<Instant>) {
        if start.is_some() {
            self.aggregate_ns += Self::elapsed(start);
            self.aggregate_calls += 1;
        }
    }

    #[inline]
    pub(crate) fn add_restore(&mut self, start: Option<Instant>) {
        if start.is_some() {
            self.restore_ns += Self::elapsed(start);
            self.restore_calls += 1;
        }
    }

    #[inline]
    pub(crate) fn add_forward(&mut self, start: Option<Instant>) {
        if start.is_some() {
            self.forward_ns += Self::elapsed(start);
            self.forward_calls += 1;
        }
    }
}
#[derive(Debug, Clone, Default)]
pub struct SearchProfileSnapshot {
    pub enabled: bool,
    pub total_ns: u128,
    pub eval_ns: u128,
    pub eval_calls: u64,
    pub movegen_order_ns: u128,
    pub movegen_order_calls: u64,
    pub make_undo_ns: u128,
    pub make_undo_calls: u64,
    pub board_make_undo_ns: u128,
    pub board_make_undo_calls: u64,
    pub eval_state_push_pop_ns: u128,
    pub eval_state_push_pop_calls: u64,
    pub eval_state_dirty_list_ns: u128,
    pub eval_state_dirty_list_calls: u64,
    pub eval_state_frame_write_ns: u128,
    pub eval_state_frame_write_calls: u64,
    pub eval_state_backup_ns: u128,
    pub eval_state_backup_calls: u64,
    pub eval_state_recompute_ns: u128,
    pub eval_state_recompute_calls: u64,
    pub eval_state_aggregate_ns: u128,
    pub eval_state_aggregate_calls: u64,
    pub eval_state_restore_ns: u128,
    pub eval_state_restore_calls: u64,
    pub eval_state_forward_ns: u128,
    pub eval_state_forward_calls: u64,
    pub eval_state_push_calls: u64,
    pub eval_state_pop_calls: u64,
    pub tt_ns: u128,
    pub tt_calls: u64,
    pub root_vct_ns: u128,
    pub root_vct_calls: u64,
    pub qsearch_ns: u128,
    pub qsearch_calls: u64,
}

pub const MOVE_PICKER_STAGE_COUNT: usize = 5;
pub const MOVE_PICKER_DIRTY_HIST_BUCKETS: usize = 10;

#[derive(Debug, Clone, Copy, Default)]
pub struct MovePickerStats {
    pub enabled_nodes: u64,
    pub legacy_nodes: u64,
    pub stage_reached: [u64; MOVE_PICKER_STAGE_COUNT],
    pub stage_moves: [u64; MOVE_PICKER_STAGE_COUNT],
    pub stage_cutoffs: [u64; MOVE_PICKER_STAGE_COUNT],
    pub duplicate_suppressed: u64,
    pub l1_materialize_nodes: u64,
    pub l1_materialize_dirty_cells: u64,
    pub direct_urgent_nodes: u64,
    pub direct_urgent_moves: u64,
    pub tail_l1_query_nodes: u64,
    pub tail_l1_query_dirty_cells: u64,
    pub tail_l1_query_dirty_hist: [u64; MOVE_PICKER_DIRTY_HIST_BUCKETS],
    pub quiet_generated_nodes: u64,
    pub quiet_skipped_nodes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SearchShapeStats {
    pub main_nodes: u64,
    pub qsearch_nodes: u64,
    pub tt_probes: u64,
    pub tt_hits: u64,
    pub tt_cutoffs: u64,
}

#[inline]
fn move_picker_dirty_hist_bucket(dirty_cells: u32) -> usize {
    match dirty_cells {
        0 => 0,
        1..=16 => 1,
        17..=32 => 2,
        33..=64 => 3,
        65..=96 => 4,
        97..=128 => 5,
        129..=160 => 6,
        161..=192 => 7,
        193..=224 => 8,
        _ => 9,
    }
}

#[derive(Debug, Clone, Copy)]
enum SearchProfileBucket {
    Eval,
    MovegenOrder,
    MakeUndo,
    BoardMakeUndo,
    EvalStatePushPop,
    Tt,
    RootVct,
    QSearch,
}

#[derive(Debug, Clone, Default)]
struct SearchProfile {
    enabled: bool,
    started_at: Option<Instant>,
    snapshot: SearchProfileSnapshot,
}

impl SearchProfile {
    fn reset(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.started_at = enabled.then(Instant::now);
        self.snapshot = SearchProfileSnapshot {
            enabled,
            ..SearchProfileSnapshot::default()
        };
    }

    #[inline]
    fn start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    #[inline]
    fn add(&mut self, bucket: SearchProfileBucket, started_at: Option<Instant>) {
        let Some(started_at) = started_at else {
            return;
        };
        let elapsed = started_at.elapsed().as_nanos();
        match bucket {
            SearchProfileBucket::Eval => {
                self.snapshot.eval_ns += elapsed;
                self.snapshot.eval_calls += 1;
            }
            SearchProfileBucket::MovegenOrder => {
                self.snapshot.movegen_order_ns += elapsed;
                self.snapshot.movegen_order_calls += 1;
            }
            SearchProfileBucket::MakeUndo => {
                self.snapshot.make_undo_ns += elapsed;
                self.snapshot.make_undo_calls += 1;
            }
            SearchProfileBucket::BoardMakeUndo => {
                self.snapshot.board_make_undo_ns += elapsed;
                self.snapshot.board_make_undo_calls += 1;
            }
            SearchProfileBucket::EvalStatePushPop => {
                self.snapshot.eval_state_push_pop_ns += elapsed;
                self.snapshot.eval_state_push_pop_calls += 1;
            }
            SearchProfileBucket::Tt => {
                self.snapshot.tt_ns += elapsed;
                self.snapshot.tt_calls += 1;
            }
            SearchProfileBucket::RootVct => {
                self.snapshot.root_vct_ns += elapsed;
                self.snapshot.root_vct_calls += 1;
            }
            SearchProfileBucket::QSearch => {
                self.snapshot.qsearch_ns += elapsed;
                self.snapshot.qsearch_calls += 1;
            }
        }
    }

    #[inline]
    fn add_eval_state_detail(&mut self, detail: EvalStateStepProfile) {
        if !self.enabled {
            return;
        }
        self.snapshot.eval_state_dirty_list_ns += detail.dirty_list_ns;
        self.snapshot.eval_state_dirty_list_calls += detail.dirty_list_calls;
        self.snapshot.eval_state_frame_write_ns += detail.frame_write_ns;
        self.snapshot.eval_state_frame_write_calls += detail.frame_write_calls;
        self.snapshot.eval_state_backup_ns += detail.backup_ns;
        self.snapshot.eval_state_backup_calls += detail.backup_calls;
        self.snapshot.eval_state_recompute_ns += detail.recompute_ns;
        self.snapshot.eval_state_recompute_calls += detail.recompute_calls;
        self.snapshot.eval_state_aggregate_ns += detail.aggregate_ns;
        self.snapshot.eval_state_aggregate_calls += detail.aggregate_calls;
        self.snapshot.eval_state_restore_ns += detail.restore_ns;
        self.snapshot.eval_state_restore_calls += detail.restore_calls;
        self.snapshot.eval_state_forward_ns += detail.forward_ns;
        self.snapshot.eval_state_forward_calls += detail.forward_calls;
        self.snapshot.eval_state_push_calls += detail.push_calls;
        self.snapshot.eval_state_pop_calls += detail.pop_calls;
    }

    fn finish(&mut self) {
        if let Some(started_at) = self.started_at {
            self.snapshot.total_ns = started_at.elapsed().as_nanos();
        }
    }

    fn snapshot(&self) -> SearchProfileSnapshot {
        self.snapshot.clone()
    }
}

/// ??????轅붽틓?????/// Bound on the magnitude of any continuation-history entry. The gravity
/// update keeps stored values asymptotically inside `[-HISTORY_MAX,
/// HISTORY_MAX]` and the move-ordering reads scale by `HISTORY_SCORE_SHIFT`
/// to stay within the existing tier budget. The clamp pair caps the per-read
/// contribution so accumulated history can never crash through the killer
/// or tier separators above. The shipped values were validated by a
/// SPSA-style sweep over the shift exponent (Phase C.3-lite, 2026-05-07):
/// the configuration below was the local optimum among `{0, 1, 2, 3}`.
const HISTORY_MAX: i32 = 16_384;
const HISTORY_SCORE_SHIFT: u32 = 2;
const HISTORY_CLAMP_1: i32 = 20_000;
const HISTORY_CLAMP_2: i32 = 15_000;

/// `(prev_move, curr_move) -> i32` continuation-history table. Allocated
/// once per `Searcher`, zeroed per `search()`.
type ContHist = Box<[[i32; NUM_CELLS]]>;

#[inline]
fn new_cont_hist() -> ContHist {
    vec![[0i32; NUM_CELLS]; NUM_CELLS].into_boxed_slice()
}

/// Stockfish-style gravity update. The pull toward zero is proportional to
/// the current value times `|bonus|`, so the table self-bounds at HISTORY_MAX
/// without an explicit clamp and is responsive to recent updates.
#[inline]
fn history_gravity_update(slot: &mut i32, bonus: i32) {
    let abs_bonus = bonus.unsigned_abs() as i64;
    let cur = *slot as i64;
    *slot = (cur + bonus as i64 - cur * abs_bonus / HISTORY_MAX as i64) as i32;
}

trait SearchEvalState {
    fn push_move(&mut self, board: &Board, mv: Move, profile_enabled: bool)
    -> EvalStateStepProfile;
    fn pop_move(&mut self, profile_enabled: bool) -> EvalStateStepProfile;
    fn eval(&mut self, board: &Board, profile_enabled: bool) -> (i32, EvalStateStepProfile);
}

struct FlatEvalState<'a> {
    weights: &'a NnueWeights,
    inc: IncrementalEval,
}

impl<'a> FlatEvalState<'a> {
    fn new(board: &Board, weights: &'a NnueWeights) -> Self {
        let mut inc = IncrementalEval::new(weights);
        inc.refresh(board, weights);
        Self { weights, inc }
    }
}

impl SearchEvalState for FlatEvalState<'_> {
    fn push_move(
        &mut self,
        board: &Board,
        mv: Move,
        _profile_enabled: bool,
    ) -> EvalStateStepProfile {
        self.inc.push_move(board, mv, self.weights);
        EvalStateStepProfile::default()
    }

    fn pop_move(&mut self, _profile_enabled: bool) -> EvalStateStepProfile {
        self.inc.pop_move();
        EvalStateStepProfile::default()
    }

    fn eval(&mut self, board: &Board, _profile_enabled: bool) -> (i32, EvalStateStepProfile) {
        (
            self.inc.eval(self.weights, board),
            EvalStateStepProfile::default(),
        )
    }
}

#[cfg(feature = "codebook-eval")]
struct CodebookEvalState<'a> {
    weights: &'a CodebookWeights,
    inc: IncrementalCodebookEval,
    scale: f32,
}

#[cfg(feature = "codebook-eval")]
impl<'a> CodebookEvalState<'a> {
    fn new(board: &Board, weights: &'a CodebookWeights, scale: f32) -> Self {
        let mut inc = IncrementalCodebookEval::new(weights);
        inc.refresh(board, weights);
        Self {
            weights,
            inc,
            scale,
        }
    }

    fn scaled_value(&self, board: &Board) -> i32 {
        (self.inc.value(board, self.weights) * self.scale)
            .round()
            .clamp(-(WIN_SCORE as f32 - 1.0), WIN_SCORE as f32 - 1.0) as i32
    }
}

#[cfg(feature = "codebook-eval")]
impl SearchEvalState for CodebookEvalState<'_> {
    fn push_move(
        &mut self,
        board: &Board,
        mv: Move,
        _profile_enabled: bool,
    ) -> EvalStateStepProfile {
        self.inc.push_move(board, mv, self.weights);
        EvalStateStepProfile::default()
    }

    fn pop_move(&mut self, _profile_enabled: bool) -> EvalStateStepProfile {
        self.inc.pop_move(self.weights);
        EvalStateStepProfile::default()
    }

    fn eval(&mut self, board: &Board, _profile_enabled: bool) -> (i32, EvalStateStepProfile) {
        (self.scaled_value(board), EvalStateStepProfile::default())
    }
}

#[cfg(feature = "codebook-eval")]
struct QuantizedCodebookEvalState<'a, W: QuantizedCodebookAccess> {
    weights: &'a W,
    inc: IncrementalQuantizedCodebookEval,
    scale: f32,
}

#[cfg(feature = "codebook-eval")]
impl<'a, W: QuantizedCodebookAccess> QuantizedCodebookEvalState<'a, W> {
    fn new(board: &Board, weights: &'a W, scale: f32, use_directional_delta: bool) -> Self {
        let mut inc =
            IncrementalQuantizedCodebookEval::new_with_access(weights, use_directional_delta);
        inc.refresh_with_access(board, weights);
        Self {
            weights,
            inc,
            scale,
        }
    }
}

#[cfg(feature = "codebook-eval")]
impl<W: QuantizedCodebookAccess> SearchEvalState for QuantizedCodebookEvalState<'_, W> {
    fn push_move(
        &mut self,
        board: &Board,
        mv: Move,
        profile_enabled: bool,
    ) -> EvalStateStepProfile {
        self.inc
            .push_move_profiled_with_access(board, mv, self.weights, profile_enabled)
    }

    fn pop_move(&mut self, profile_enabled: bool) -> EvalStateStepProfile {
        self.inc
            .pop_move_profiled_with_access(self.weights, profile_enabled)
    }

    fn eval(&mut self, board: &Board, profile_enabled: bool) -> (i32, EvalStateStepProfile) {
        let (value, detail) =
            self.inc
                .value_profiled_with_access(board, self.weights, profile_enabled);
        (
            (value * self.scale)
                .round()
                .clamp(-(WIN_SCORE as f32 - 1.0), WIN_SCORE as f32 - 1.0) as i32,
            detail,
        )
    }
}

#[cfg(feature = "codebook-eval")]
#[derive(Clone, Copy, Debug)]
struct WhiteRootOrderEntry {
    residual: f32,
    quiet_ongoing: bool,
}

#[cfg(feature = "codebook-eval")]
#[derive(Clone, Debug)]
struct WhiteRootOrderCache {
    root_zobrist: u64,
    entries: [Option<WhiteRootOrderEntry>; NUM_CELLS],
}

#[cfg(feature = "codebook-eval")]
impl WhiteRootOrderCache {
    fn new(root_zobrist: u64) -> Self {
        Self {
            root_zobrist,
            entries: [None; NUM_CELLS],
        }
    }

    fn entry(&self, mv: Move) -> Option<WhiteRootOrderEntry> {
        self.entries.get(mv).copied().flatten()
    }
}

pub struct Searcher {
    pub nodes: u64,
    /// TT cutoff ?????????용맧硅 ??probe()?????????ル??????遺얘턁???????꿔꺂????釉먯춱?entry???????ル??? depth/bound ???汝뷴젆?琉??誘↔덱?????????????源낅???    /// ????븐뼐????傭?끆?????Β?ｊ콞?轅붽틓?????⑸걦???score ?????獄쏅챶留덌┼?????????????????용맧硅. TT ??????????????????븐뼐???????????????????븐뼐????????
    pub tt_cutoffs: u64,
    killers: [[Option<Move>; 2]; 64],
    history: [[i32; NUM_CELLS]; 2],
    /// 1-ply continuation (countermove) history: `[prev_move][curr_move]`.
    /// Bonus accrues when `curr_move` causes a beta-cutoff in a node whose
    /// parent move was `prev_move`; siblings tried earlier in the same node
    /// receive symmetric penalties.
    cont_hist_1: ContHist,
    /// 2-ply follow-up history: `[prev_prev_move][curr_move]`. Same update
    /// scheme, different context ??captures plans that span two of our moves.
    cont_hist_2: ContHist,
    deadline: Option<Instant>,
    aborted: bool,
    node_limit: Option<u64>,
    configured_node_limit: Option<u64>,
    node_limit_hit: bool,
    /// ???????遺얘턁????????ㅼ뒧??띤겫??눫????癲됱빖???嶺????????? ???????ル???? ?????????⑤슢堉??곕?????????????????????뀀맩鍮???????
    /// search ???遺얘턁??????????????????⑤벡瑜??????耀붾굝????????iterative deepening ???????롮쾸?椰???iteration?????    /// ?????????대첐??iteration??PV/cutoff ???遺얘턁?????????????????ル탛????????耀붾굝???????????
    tt: TranspositionTable,
    profile: SearchProfile,
    threat_field_mode: SearchThreatFieldMode,
    stress_threat_field: bool,
    use_move_picker: bool,
    use_tail_threat_materialize: bool,
    /// Enable the incremental candidate frontier only after root VCT has
    /// failed, so VCT nodes that do not consume it pay no maintenance.
    use_candidate_frontier: bool,
    /// Enable packed Pattern4 window maintenance in search-local sidecar
    /// state. Keeping this out of `Board` preserves its public layout.
    use_packed_line_windows: bool,
    /// Enable incremental D4 hashes in search-local sidecar state.
    ///
    /// This is an experimental observer only: no TT, VCT, or move-selection
    /// key consumes it.
    use_d4_hash_sidecar: bool,
    /// Per-searcher root-VCT selector for deterministic audit harnesses.
    ///
    /// Product behavior remains enabled and still obeys `NORU_ROOT_VCT`.
    use_root_vct_for_audit: bool,
    /// Exact quantized-codebook `(cell, direction)` embedding deltas.
    ///
    /// Experimental and OFF by default until the CB-D1 release gates pass.
    #[cfg(feature = "codebook-eval")]
    use_codebook_directional_delta: bool,
    /// Codebook eval scale for this searcher; `None` uses the process-wide [`codebook_eval_scale`].
    #[cfg(feature = "codebook-eval")]
    codebook_eval_scale_override: Option<f32>,
    /// Per-search incremental state. It is rebuilt at the root and dropped
    /// before returning so protocol moves cannot leave it stale.
    board_search_state: Option<BoardSearchState>,
    move_picker_stats: MovePickerStats,
    shape_stats: SearchShapeStats,
    threat_field: Option<IncrementalThreatField>,
    #[cfg(feature = "codebook-eval")]
    white_root_order: Option<WhiteRootOrder>,
    #[cfg(feature = "codebook-eval")]
    white_root_order_cache: Option<WhiteRootOrderCache>,
    /// P4-R1: restrict the move list at nodes whose reply set the rules fix.
    use_forced_restriction: bool,
    /// P4-R0 forced-node census. Counting only; never affects search output.
    pub census: crate::p4r0_census::ForcedCensus,
    census_scratch: Vec<crate::board::Move>,
}

impl Searcher {
    pub fn new() -> Self {
        Self {
            nodes: 0,
            tt_cutoffs: 0,
            killers: [[None; 2]; 64],
            history: [[0; NUM_CELLS]; 2],
            cont_hist_1: new_cont_hist(),
            cont_hist_2: new_cont_hist(),
            deadline: None,
            aborted: false,
            node_limit: None,
            configured_node_limit: None,
            node_limit_hit: false,
            tt: TranspositionTable::new(TT_BUCKET_BITS),
            profile: SearchProfile::default(),
            // RQ595 threat-field / staged move-picker stack: OFF by default,
            // reachable only through the setters (rq582 profiling bin).
            threat_field_mode: SearchThreatFieldMode::Off,
            stress_threat_field: false,
            threat_field: None,
            use_move_picker: false,
            use_tail_threat_materialize: false,
            use_candidate_frontier: false,
            use_packed_line_windows: false,
            use_d4_hash_sidecar: false,
            use_root_vct_for_audit: true,
            // 0.8.6: directional delta is the default eval path (bit-exact,
            // ~2x cheaper materialization); the setter remains for A/B only.
            #[cfg(feature = "codebook-eval")]
            use_codebook_directional_delta: true,
            #[cfg(feature = "codebook-eval")]
            codebook_eval_scale_override: None,
            board_search_state: None,
            move_picker_stats: MovePickerStats::default(),
            shape_stats: SearchShapeStats::default(),
            #[cfg(feature = "codebook-eval")]
            white_root_order: None,
            #[cfg(feature = "codebook-eval")]
            white_root_order_cache: None,
            use_forced_restriction: forced_restriction_enabled_by_env(),
            census: crate::p4r0_census::ForcedCensus::default(),
            census_scratch: Vec::with_capacity(16),
        }
    }

    /// Enable the shipped White root quiet-move ordering model.
    ///
    /// The model is defined for the quantized evaluator bundled with the
    /// tournament engine. The pbrain adapter enables it automatically only
    /// for that exact embedded path; library callers must opt in explicitly.
    #[cfg(feature = "codebook-eval")]
    pub fn set_white_root_order_enabled(&mut self, enabled: bool) -> Result<(), String> {
        self.white_root_order = enabled.then(WhiteRootOrder::production);
        self.white_root_order_cache = None;
        Ok(())
    }

    #[cfg(feature = "codebook-eval")]
    pub fn white_root_order_enabled(&self) -> bool {
        self.white_root_order.is_some()
    }

    /// TT ????븐뼐?????????????筌뤾퍓愿??????????獄쏅챶留덌┼?????????search() ???耀붾굝????????????븐뼐???????????遺얘턁??????????????ル?????
    fn check_node_limit(&mut self) -> bool {
        if let Some(limit) = self.node_limit {
            if self.nodes >= limit {
                self.node_limit_hit = true;
                self.aborted = true;
                return true;
            }
        }
        false
    }

    pub fn tt_stats(&self) -> TtStats {
        self.tt.stats()
    }

    /// TT ????????`(non-empty depth_pref, non-empty always_replace, ??bucket)`.
    pub fn tt_occupancy(&self) -> (usize, usize, usize) {
        self.tt.occupancy()
    }

    pub fn search_profile(&self) -> SearchProfileSnapshot {
        self.profile.snapshot()
    }

    pub fn move_picker_stats(&self) -> MovePickerStats {
        self.move_picker_stats
    }

    pub fn search_shape_stats(&self) -> SearchShapeStats {
        let mut stats = self.shape_stats;
        stats.tt_cutoffs = self.tt_cutoffs;
        stats
    }

    pub fn set_use_threat_field(&mut self, enabled: bool) {
        self.threat_field_mode = if enabled {
            SearchThreatFieldMode::Eager
        } else {
            SearchThreatFieldMode::Off
        };
        if !enabled {
            self.threat_field = None;
        }
    }

    pub fn set_use_lazy_threat_field(&mut self, enabled: bool) {
        self.threat_field_mode = if enabled {
            SearchThreatFieldMode::Lazy
        } else {
            SearchThreatFieldMode::Off
        };
        if !enabled {
            self.threat_field = None;
        }
    }

    pub fn set_stress_threat_field(&mut self, enabled: bool) {
        self.stress_threat_field = enabled;
    }

    /// Enable the sound forced-reply restriction (P4-R1): at nodes where
    /// the opponent threatens an immediate five, search only the reply the
    /// rules leave (our own five, else the single blocking cell; two or more
    /// completing cells score as a loss). Freestyle and Standard only.
    /// Defaults to `NORU_FORCED_REPLY_RESTRICTION` (off when unset).
    pub fn set_use_forced_restriction(&mut self, enabled: bool) {
        self.use_forced_restriction = enabled;
    }

    pub fn set_node_limit(&mut self, limit: Option<u64>) {
        self.configured_node_limit = limit;
    }

    pub fn node_limit_hit(&self) -> bool {
        self.node_limit_hit
    }

    pub fn set_use_move_picker(&mut self, enabled: bool) {
        self.use_move_picker = enabled;
        if enabled && self.threat_field_mode == SearchThreatFieldMode::Off {
            // The staged picker consumes L1 threat sources. Default to the
            // already accepted lazy field when callers only toggle the picker.
            self.threat_field_mode = SearchThreatFieldMode::Lazy;
        }
    }

    pub fn set_use_tail_threat_materialize(&mut self, enabled: bool) {
        self.use_tail_threat_materialize = enabled;
        if enabled {
            self.set_use_move_picker(true);
        }
    }

    /// Enable exact-order incremental candidate generation for the main
    /// alpha-beta search. Root VCT stays on the legacy board path.
    pub fn set_use_candidate_frontier(&mut self, enabled: bool) {
        self.use_candidate_frontier = enabled;
    }

    /// Enable packed Pattern4 windows for search make/undo operations.
    ///
    /// The state is search-local; ordinary `Board` construction and mutation
    /// remain layout- and behavior-compatible for library callers.
    pub fn set_use_packed_line_windows(&mut self, enabled: bool) {
        self.use_packed_line_windows = enabled;
    }

    /// Enable incremental D4 hashes in the search-local sidecar.
    ///
    /// The hashes are maintained across searched make/undo operations but are
    /// deliberately not connected to any transposition or proof cache.
    pub fn set_use_d4_hash_sidecar(&mut self, enabled: bool) {
        self.use_d4_hash_sidecar = enabled;
    }

    /// Report whether D4 hash maintenance was explicitly requested.
    #[doc(hidden)]
    #[inline]
    pub fn d4_hash_sidecar_requested(&self) -> bool {
        self.use_d4_hash_sidecar
    }

    /// Override root VCT for deterministic, no-deadline audit searches.
    ///
    /// This does not alter the process-level product switch and defaults to
    /// enabled. Production callers should leave it unchanged.
    #[doc(hidden)]
    pub fn set_use_root_vct_for_audit(&mut self, enabled: bool) {
        self.use_root_vct_for_audit = enabled;
    }

    #[doc(hidden)]
    #[inline]
    pub fn root_vct_requested_for_audit(&self) -> bool {
        self.use_root_vct_for_audit
    }

    /// Enable the exact directional-delta quantized evaluator.
    #[cfg(feature = "codebook-eval")]
    pub fn set_use_codebook_directional_delta(&mut self, enabled: bool) {
        self.use_codebook_directional_delta = enabled;
    }

    /// Set the codebook eval scale used by this searcher (e.g. a per-rule model); `None` restores the
    /// process-wide [`codebook_eval_scale`].
    #[cfg(feature = "codebook-eval")]
    pub fn set_codebook_eval_scale(&mut self, scale: Option<f32>) {
        self.codebook_eval_scale_override = scale.filter(|s| s.is_finite() && *s > 0.0);
    }

    fn begin_board_search_state(&mut self, board: &Board) {
        if !self.use_packed_line_windows
            && !self.use_candidate_frontier
            && !self.use_d4_hash_sidecar
        {
            self.board_search_state = None;
            return;
        }
        let state = self
            .board_search_state
            .get_or_insert_with(BoardSearchState::new);
        state.synchronize(board);
        state.set_packed_line_windows_enabled(board, self.use_packed_line_windows);
        // A3 starts only after root VCT fails.
        state.set_candidate_frontier_enabled(board, false);
        state.set_d4_hash_enabled(board, self.use_d4_hash_sidecar);
    }

    #[inline]
    fn enable_main_search_candidate_frontier(&mut self, board: &Board) {
        if let Some(state) = self.board_search_state.as_mut() {
            state.set_candidate_frontier_enabled(board, self.use_candidate_frontier);
        }
    }

    #[inline]
    fn end_board_search_state(&mut self, board: &Board) {
        if let Some(state) = self.board_search_state.as_mut() {
            // A3 is profitable only inside the main search. Retain synchronized
            // A2 or D4-hash state for repeated searches of an unchanged root.
            state.set_candidate_frontier_enabled(board, false);
            if !state.packed_line_windows_enabled() && !state.d4_hash_enabled() {
                self.board_search_state = None;
            }
        }
    }

    #[inline]
    fn make_board_move(&mut self, board: &mut Board, mv: Move) {
        if let Some(state) = self.board_search_state.as_mut() {
            state.make_move_synchronized(board, mv);
        } else {
            board.make_move(mv);
        }
    }

    #[inline]
    fn undo_board_move(&mut self, board: &mut Board) {
        if let Some(state) = self.board_search_state.as_mut() {
            state.undo_move_synchronized(board);
        } else {
            board.undo_move();
        }
    }

    #[inline]
    fn board_candidate_moves(&self, board: &Board) -> Vec<Move> {
        let mut moves = self.board_search_state.as_ref().map_or_else(
            || board.candidate_moves(),
            |state| state.candidate_moves_synchronized(board),
        );
        // Renju: black never plays a forbidden point. Status is non-local, so filter after enumeration.
        if board.side_to_move == Stone::Black && board.effective_rule_set() == RuleSet::Renju {
            moves.retain(|&mv| !board.is_forbidden_for_side_to_move(mv));
        }
        moves
    }
    #[inline]
    pub fn use_threat_field(&self) -> bool {
        self.threat_field_mode != SearchThreatFieldMode::Off
    }

    #[inline]
    fn profile_start(&self) -> Option<Instant> {
        self.profile.start()
    }

    #[inline]
    fn profile_add(&mut self, bucket: SearchProfileBucket, started_at: Option<Instant>) {
        self.profile.add(bucket, started_at);
    }

    fn reset_threat_field(&mut self, board: &Board) {
        match self.threat_field_mode {
            SearchThreatFieldMode::Off => {
                self.threat_field = None;
            }
            SearchThreatFieldMode::Eager | SearchThreatFieldMode::Lazy => {
                let mode = match self.threat_field_mode {
                    SearchThreatFieldMode::Eager => ThreatFieldUpdateMode::Eager,
                    SearchThreatFieldMode::Lazy => ThreatFieldUpdateMode::Lazy,
                    SearchThreatFieldMode::Off => unreachable!(),
                };
                self.threat_field = Some(IncrementalThreatField::with_mode(board, mode));
            }
        }
    }

    #[inline]
    fn push_threat_field(&mut self, board: &Board, mv: Move) {
        if let Some(field) = self.threat_field.as_mut() {
            field.push_move(board, mv);
        }
    }

    #[inline]
    fn pop_threat_field(&mut self, board: &Board) {
        if let Some(field) = self.threat_field.as_mut() {
            field.pop_move(board);
        }
    }

    #[inline]
    fn stress_threat_field_query(&mut self, board: &Board) {
        if self.stress_threat_field {
            if let Some(field) = self.threat_field.as_mut() {
                let _ = field.immediate_five(board, Stone::Black);
                let _ = field.immediate_five(board, Stone::White);
            }
        }
    }

    fn reset_for_search(&mut self, time_limit: Option<Duration>) {
        self.nodes = 0;
        self.tt_cutoffs = 0;
        self.aborted = false;
        self.node_limit = self.configured_node_limit;
        self.node_limit_hit = false;
        self.killers = [[None; 2]; 64];
        self.history = [[0; NUM_CELLS]; 2];
        for row in self.cont_hist_1.iter_mut() {
            row.fill(0);
        }
        for row in self.cont_hist_2.iter_mut() {
            row.fill(0);
        }
        self.deadline = time_limit.map(|d| Instant::now() + d);
        self.tt.reset_stats();
        self.tt.clear();
        self.profile.reset(search_profile_enabled());
        self.threat_field = None;
        self.move_picker_stats = MovePickerStats::default();
        self.shape_stats = SearchShapeStats::default();
        #[cfg(feature = "codebook-eval")]
        {
            self.white_root_order_cache = None;
        }
    }

    #[cfg(feature = "codebook-eval")]
    fn prepare_white_root_order_cache<W: QuantizedCodebookAccess>(
        &mut self,
        board: &Board,
        weights: &W,
    ) -> Result<(), String> {
        let Some(policy) = self.white_root_order.as_ref() else {
            return Ok(());
        };
        if board.effective_rule_set() != RuleSet::Freestyle
            || board.side_to_move != Stone::White
            || board.game_result() != GameResult::Ongoing
        {
            return Ok(());
        }

        let opponent = board.side_to_move.opponent();
        let forced_defense_root = board
            .legal_moves()
            .into_iter()
            .any(|mv| classify_move_fast(board, mv, opponent) == ThreatKind::Five);
        if forced_defense_root {
            return Ok(());
        }

        let mut cache = WhiteRootOrderCache::new(board.zobrist);
        let mut scratch = board.clone();
        let mut extractor = IncrementalQuantizedCodebookEval::new_with_access(
            weights,
            self.use_codebook_directional_delta,
        );
        extractor.refresh_with_access(&scratch, weights);
        let candidates = self.board_candidate_moves(board);
        for mv in candidates {
            let attack = classify_move_fast(board, mv, board.side_to_move);
            let block = classify_move_fast(board, mv, opponent);
            if let Some(state) = self.board_search_state.as_mut() {
                state.make_move_synchronized(&mut scratch, mv);
            } else {
                scratch.make_move(mv);
            }
            extractor.push_move_profiled_with_access(&scratch, mv, weights, false);
            let quiet_ongoing = scratch.game_result() == GameResult::Ongoing
                && attack == ThreatKind::None
                && block == ThreatKind::None;
            let residual = if quiet_ongoing {
                let orbit = extractor.explicit_orbit48_with_access(weights, Stone::White)?;
                policy.score_orbit48(&orbit)?
            } else {
                0.0
            };
            extractor.pop_move_profiled_with_access(weights, false);
            if let Some(state) = self.board_search_state.as_mut() {
                state.undo_move_synchronized(&mut scratch);
            } else {
                scratch.undo_move();
            }
            cache.entries[mv] = Some(WhiteRootOrderEntry {
                residual,
                quiet_ongoing,
            });
        }

        if scratch.black != board.black
            || scratch.white != board.white
            || scratch.side_to_move != board.side_to_move
            || scratch.history != board.history
            || scratch.zobrist != board.zobrist
            || scratch.line_pattern_ids.as_ref() != board.line_pattern_ids.as_ref()
        {
            return Err("white root ordering scratch state did not restore the root".to_string());
        }
        self.white_root_order_cache = Some(cache);
        Ok(())
    }

    #[cfg(feature = "codebook-eval")]
    fn apply_white_root_order(
        &self,
        board: &Board,
        moves: &mut [(Move, bool)],
        previous_pv: Option<Move>,
    ) {
        let Some(policy) = self.white_root_order.as_ref() else {
            return;
        };
        let Some(cache) = self.white_root_order_cache.as_ref() else {
            return;
        };
        if board.effective_rule_set() != RuleSet::Freestyle
            || board.side_to_move != Stone::White
            || board.game_result() != GameResult::Ongoing
        {
            return;
        }
        assert_eq!(
            cache.root_zobrist, board.zobrist,
            "white root ordering cache does not match the current root"
        );

        let root_killers = self.killers[0];
        let original_position: [usize; NUM_CELLS] = {
            let mut positions = [usize::MAX; NUM_CELLS];
            for (index, &(mv, _)) in moves.iter().enumerate() {
                positions[mv] = index;
            }
            positions
        };
        let eligible = |mv: Move| {
            cache.entry(mv).is_some_and(|entry| {
                entry.quiet_ongoing
                    && Some(mv) != previous_pv
                    && Some(mv) != root_killers[0]
                    && Some(mv) != root_killers[1]
            })
        };

        let mut start = 0usize;
        while start < moves.len() {
            if !eligible(moves[start].0) {
                start += 1;
                continue;
            }
            let mut end = start + 1;
            while end < moves.len() && eligible(moves[end].0) {
                end += 1;
            }
            if end - start >= 2 {
                let mut score_by_move = [0.0f32; NUM_CELLS];
                for (run_index, &(mv, _)) in moves[start..end].iter().enumerate() {
                    let residual = cache
                        .entry(mv)
                        .expect("eligible move must be cached")
                        .residual;
                    score_by_move[mv] = policy
                        .add_anchor_to_residual(residual, run_index, end - start)
                        .unwrap_or_else(|error| {
                            panic!("invalid white root ordering score: {error}")
                        });
                }
                moves[start..end].sort_by(|&(a, _), &(b, _)| {
                    score_by_move[b]
                        .total_cmp(&score_by_move[a])
                        .then_with(|| original_position[a].cmp(&original_position[b]))
                });
            }
            start = end;
        }
    }

    fn try_root_vct(
        &mut self,
        board: &mut Board,
        time_limit: Option<Duration>,
    ) -> Option<SearchResult> {
        if !self.use_root_vct_for_audit {
            return None;
        }
        let vct_budget = match time_limit {
            Some(d) => (d / ROOT_VCT_BUDGET_FRACTION)
                .max(Duration::from_millis(ROOT_VCT_BUDGET_FLOOR_MS))
                .min(Duration::from_millis(ROOT_VCT_BUDGET_CAP_MS)),
            None => Duration::from_millis(ROOT_VCT_BUDGET_MS),
        };
        let vct_cfg = VctConfig {
            max_depth: ROOT_VCT_DEPTH,
            time_budget: Some(vct_budget),
            node_budget: None,
            enable_jump_three: false,
            enable_jump_three_attack_defense: false,
            enable_jump_three_counter: false,
            enable_jump_three_kind_scoped_defense: false,
            jump_attack_max_or_levels: u32::MAX,
            enable_gap_four: false,
            gap_four_attack_max_or_levels: u32::MAX,
            use_fast_classify: true,
            use_threat_index: false,
            profile: false,
            use_reach_mask: true,
            use_fast_immediate_five: false,
            use_vct_scratch_buffers: false,
        };
        let profile_start = self.profile_start();
        let seq = if let Some(state) = self.board_search_state.as_mut() {
            search_vct_with_board_search_state(board, &vct_cfg, state)
        } else {
            search_vct(board, &vct_cfg)
        };
        self.profile_add(SearchProfileBucket::RootVct, profile_start);
        if let Some(seq) = seq {
            if let Some(&first) = seq.first() {
                let result = SearchResult {
                    best_move: Some(first),
                    score: WIN_SCORE,
                    depth: seq.len() as u32,
                    nodes: self.nodes,
                };
                self.profile.finish();
                return Some(result);
            }
        }
        None
    }

    fn search_with_eval_state<E: SearchEvalState>(
        &mut self,
        board: &mut Board,
        weights: &NnueWeights,
        inc: &mut E,
        max_depth: u32,
    ) -> SearchResult {
        self.reset_threat_field(board);
        let mut best_result = SearchResult {
            best_move: None,
            score: 0,
            depth: 0,
            nodes: 0,
        };
        let mut prev_best: Option<Move> = None;
        let mut prev_score: Option<i32> = None;

        for depth in 1..=max_depth {
            let aspirate = depth >= ASPIRATION_MIN_DEPTH && prev_score.is_some();
            let (alpha_init, beta_init) = if aspirate {
                let s = prev_score.unwrap();
                (s - ASPIRATION_INITIAL_DELTA, s + ASPIRATION_INITIAL_DELTA)
            } else {
                (-INF, INF)
            };

            let mut alpha = alpha_init;
            let mut beta = beta_init;
            let mut delta = ASPIRATION_INITIAL_DELTA;

            let iter_result = loop {
                let current =
                    self.root_pvs_iteration(board, weights, inc, depth, alpha, beta, prev_best);
                if self.aborted {
                    break current;
                }
                let score = current.1;
                if !aspirate {
                    break current;
                }
                if score <= alpha {
                    delta = (delta * 2).min(INF / 4);
                    alpha = (alpha - delta).max(-INF);
                    if alpha == -INF {
                        break self
                            .root_pvs_iteration(board, weights, inc, depth, -INF, INF, prev_best);
                    }
                } else if score >= beta {
                    delta = (delta * 2).min(INF / 4);
                    beta = (beta + delta).min(INF);
                    if beta == INF {
                        break self
                            .root_pvs_iteration(board, weights, inc, depth, -INF, INF, prev_best);
                    }
                } else {
                    break current;
                }
            };

            if self.aborted {
                break;
            }

            let (best_move, score) = iter_result;
            best_result = SearchResult {
                best_move,
                score,
                depth,
                nodes: self.nodes,
            };

            if score.abs() > WIN_SCORE - 100 {
                break;
            }

            prev_best = best_move;
            prev_score = Some(score);
        }

        self.profile.finish();
        best_result
    }

    /// Root search entry point.
    pub fn search(
        &mut self,
        board: &mut Board,
        weights: &NnueWeights,
        max_depth: u32,
        time_limit: Option<Duration>,
    ) -> SearchResult {
        #[cfg(feature = "codebook-eval")]
        assert!(
            self.white_root_order.is_none(),
            "white root ordering requires the quantized codebook search path"
        );
        self.nodes = 0;
        self.tt_cutoffs = 0;
        self.aborted = false;
        self.node_limit = self.configured_node_limit;
        self.node_limit_hit = false;
        self.killers = [[None; 2]; 64];
        self.history = [[0; NUM_CELLS]; 2];
        for row in self.cont_hist_1.iter_mut() {
            row.fill(0);
        }
        for row in self.cont_hist_2.iter_mut() {
            row.fill(0);
        }
        self.deadline = time_limit.map(|d| Instant::now() + d);
        // TT ????븐뼐?????????????筌뤾퍓愿??????????ш끽紐?????????醫딇떍????????search() ???遺얘턁??????????????븐뼐?????????????????⑤벡苑?
        self.tt.reset_stats();
        // ???????롮쾸?椰???search ???遺얘턁????????????癲됱빖???嶺???????????????????????????롮쾸?椰?嚥▲굧???븍툖????????곗뵰?????TT ????.
        // ???????ル???? search() ???遺얘턁?????????iterative deepening ????????TT???????ル??? ??????耀붾굝????????        // ???????????? iteration????? iteration??cutoff/PV ???遺얘턁?????????????????ル탛????????耀붾굝???????????
        self.tt.clear();
        self.profile.reset(search_profile_enabled());
        self.threat_field = None;
        self.move_picker_stats = MovePickerStats::default();
        self.shape_stats = SearchShapeStats::default();
        #[cfg(feature = "codebook-eval")]
        {
            self.white_root_order_cache = None;
        }
        self.begin_board_search_state(board);

        // Root VCT: ????븐뼐?????????? ????????????????????????ル???????????諛몃마嶺뚮?????꾩렯???????耀붾굝????????????븐뼐?????????????????????????븐뼐?곭춯?竊???????.
        // Dynamic budget ???????????1/ROOT_VCT_BUDGET_FRACTION (cap/floor ?????????泥??.
        if self.use_root_vct_for_audit {
            let vct_budget = match time_limit {
                Some(d) => (d / ROOT_VCT_BUDGET_FRACTION)
                    .max(Duration::from_millis(ROOT_VCT_BUDGET_FLOOR_MS))
                    .min(Duration::from_millis(ROOT_VCT_BUDGET_CAP_MS)),
                None => Duration::from_millis(ROOT_VCT_BUDGET_MS),
            };
            let vct_cfg = VctConfig {
                max_depth: ROOT_VCT_DEPTH,
                time_budget: Some(vct_budget),
                node_budget: None,
                enable_jump_three: false,
                enable_jump_three_attack_defense: false,
                enable_jump_three_counter: false,
                enable_jump_three_kind_scoped_defense: false,
                jump_attack_max_or_levels: u32::MAX,
                enable_gap_four: false,
                gap_four_attack_max_or_levels: u32::MAX,
                use_fast_classify: true,
                use_threat_index: false,
                profile: false,
                use_reach_mask: true,
                use_fast_immediate_five: false,
                use_vct_scratch_buffers: false,
            };
            let profile_start = self.profile_start();
            let seq = if let Some(state) = self.board_search_state.as_mut() {
                search_vct_with_board_search_state(board, &vct_cfg, state)
            } else {
                search_vct(board, &vct_cfg)
            };
            self.profile_add(SearchProfileBucket::RootVct, profile_start);
            if let Some(seq) = seq {
                if let Some(&first) = seq.first() {
                    let result = SearchResult {
                        best_move: Some(first),
                        score: WIN_SCORE,
                        depth: seq.len() as u32,
                        nodes: self.nodes,
                    };
                    self.profile.finish();
                    self.end_board_search_state(board);
                    return result;
                }
            }
        }

        self.enable_main_search_candidate_frontier(board);
        let mut best_result = SearchResult {
            best_move: None,
            score: 0,
            depth: 0,
            nodes: 0,
        };

        // Incremental NNUE state ??????????耀붾굝?????傭?끆????椰???????full refresh, ???????꾩룆梨띰쭕??        // make_move/undo_move?? ?????????딅즹???push/pop????????源낅?????leaf?????full
        // compute_active_features????? ??????????Accumulator forward??????????
        self.reset_threat_field(board);
        let mut inc = FlatEvalState::new(board, weights);

        // PV-move priority: the best move from iteration depth-1 becomes the
        // first move we try at iteration depth. Combined with PVS + Aspiration,
        // this drastically reduces re-search cost.
        let mut prev_best: Option<Move> = None;
        let mut prev_score: Option<i32> = None;

        for depth in 1..=max_depth {
            // Aspiration windows: depth ??4 ???????깆궔?????????????대첐??iteration score ??????獄쏅챷???
            // ???????筌띿솘?? [s-delta, s+delta] ?????耀붾굝?????傭?끆????椰? fail-low/high ???????
            // ???遺얘턁???????????怨뺤름???score????????⑤벡瑜???????????筌띿솘?? window ?????????耀붾굝???????????cutoff ??????????????
            // ???????ル탛????? window ????????????????widening.
            let aspirate = depth >= ASPIRATION_MIN_DEPTH && prev_score.is_some();
            let (alpha_init, beta_init) = if aspirate {
                let s = prev_score.unwrap();
                (s - ASPIRATION_INITIAL_DELTA, s + ASPIRATION_INITIAL_DELTA)
            } else {
                (-INF, INF)
            };

            let mut alpha = alpha_init;
            let mut beta = beta_init;
            let mut delta = ASPIRATION_INITIAL_DELTA;

            // Aspiration re-search loop. fail-high/low ????븐뼐????????window widen.
            let iter_result = loop {
                let current = self
                    .root_pvs_iteration(board, weights, &mut inc, depth, alpha, beta, prev_best);
                if self.aborted {
                    break current;
                }
                let score = current.1;
                if !aspirate {
                    break current;
                }
                if score <= alpha {
                    // fail-low: alpha ??????
                    delta = (delta * 2).min(INF / 4);
                    alpha = (alpha - delta).max(-INF);
                    if alpha == -INF {
                        // ?????????耀붾굝????????????full window??break???????
                        // ????癲됱빖???嶺????????????롮쾸?椰???iteration????????? ?????break ??                        // ????????袁④뎬??-INF/INF??
                        // re-search with full window once
                        break self.root_pvs_iteration(
                            board, weights, &mut inc, depth, -INF, INF, prev_best,
                        );
                    }
                } else if score >= beta {
                    delta = (delta * 2).min(INF / 4);
                    beta = (beta + delta).min(INF);
                    if beta == INF {
                        break self.root_pvs_iteration(
                            board, weights, &mut inc, depth, -INF, INF, prev_best,
                        );
                    }
                } else {
                    // window ????OK
                    break current;
                }
            };

            if self.aborted {
                break;
            }

            let (best_move, score) = iter_result;
            best_result = SearchResult {
                best_move,
                score,
                depth,
                nodes: self.nodes,
            };

            if score.abs() > WIN_SCORE - 100 {
                break;
            }

            prev_best = best_move;
            prev_score = Some(score);
        }

        self.profile.finish();
        self.end_board_search_state(board);
        best_result
    }

    #[cfg(feature = "codebook-eval")]
    pub fn search_codebook_eval(
        &mut self,
        board: &mut Board,
        ordering_weights: &NnueWeights,
        codebook_weights: &CodebookWeights,
        max_depth: u32,
        time_limit: Option<Duration>,
    ) -> SearchResult {
        assert!(
            self.white_root_order.is_none(),
            "white root ordering cannot run on the float codebook search path"
        );
        self.reset_for_search(time_limit);
        self.begin_board_search_state(board);
        if let Some(result) = self.try_root_vct(board, time_limit) {
            self.end_board_search_state(board);
            return result;
        }
        self.enable_main_search_candidate_frontier(board);
        let scale = self.codebook_eval_scale_override.unwrap_or_else(codebook_eval_scale);
        let mut inc = CodebookEvalState::new(board, codebook_weights, scale);
        let result = self.search_with_eval_state(board, ordering_weights, &mut inc, max_depth);
        self.end_board_search_state(board);
        result
    }

    #[cfg(feature = "codebook-eval")]
    pub fn search_codebook_eval_quantized(
        &mut self,
        board: &mut Board,
        ordering_weights: &NnueWeights,
        codebook_weights: &QuantizedCodebookWeights,
        max_depth: u32,
        time_limit: Option<Duration>,
    ) -> SearchResult {
        self.search_codebook_eval_quantized_with(
            board,
            ordering_weights,
            codebook_weights,
            max_depth,
            time_limit,
        )
    }

    /// Search with a CB-F1 class-base plus residual quantized codebook.
    ///
    /// This is a separate generic instantiation of the evaluator and search
    /// adapter. The factored table is never expanded and the node hot path
    /// performs no representation tag dispatch.
    #[cfg(feature = "codebook-eval")]
    pub fn search_codebook_eval_quantized_factored(
        &mut self,
        board: &mut Board,
        ordering_weights: &NnueWeights,
        codebook_weights: &FactoredQuantizedCodebookWeights,
        max_depth: u32,
        time_limit: Option<Duration>,
    ) -> SearchResult {
        self.search_codebook_eval_quantized_with(
            board,
            ordering_weights,
            codebook_weights,
            max_depth,
            time_limit,
        )
    }

    #[cfg(feature = "codebook-eval")]
    fn search_codebook_eval_quantized_with<W: QuantizedCodebookAccess>(
        &mut self,
        board: &mut Board,
        ordering_weights: &NnueWeights,
        codebook_weights: &W,
        max_depth: u32,
        time_limit: Option<Duration>,
    ) -> SearchResult {
        self.reset_for_search(time_limit);
        self.begin_board_search_state(board);
        if let Some(result) = self.try_root_vct(board, time_limit) {
            self.end_board_search_state(board);
            return result;
        }
        self.enable_main_search_candidate_frontier(board);
        self.prepare_white_root_order_cache(board, codebook_weights)
            .unwrap_or_else(|error| panic!("invalid white root ordering state: {error}"));
        let scale = self.codebook_eval_scale_override.unwrap_or_else(codebook_eval_scale);
        let mut inc = QuantizedCodebookEvalState::new(
            board,
            codebook_weights,
            scale,
            self.use_codebook_directional_delta,
        );
        let result = self.search_with_eval_state(board, ordering_weights, &mut inc, max_depth);
        self.end_board_search_state(board);
        result
    }

    /// ??iteration??root-level PVS ?????
    /// `[alpha_init, beta_init]` window ????????????????븐뼐???????????븐뼔????root move???????????棺堉?뤃????    /// best move + alpha ?????獄쏅챶留덌┼??????? Aspiration loop??inner step.
    fn root_pvs_iteration<E: SearchEvalState>(
        &mut self,
        board: &mut Board,
        weights: &NnueWeights,
        inc: &mut E,
        depth: u32,
        alpha_init: i32,
        beta_init: i32,
        prev_best: Option<Move>,
    ) -> (Option<Move>, i32) {
        let mut alpha = alpha_init;
        let beta = beta_init;
        let mut best_move: Option<Move> = None;
        let mut leader_score = -INF;

        let profile_start = self.profile_start();
        let mut moves = self.order_moves(board, 0);
        #[cfg(feature = "codebook-eval")]
        self.apply_white_root_order(board, &mut moves, prev_best);
        self.profile_add(SearchProfileBucket::MovegenOrder, profile_start);
        if let Some(pv) = prev_best {
            if let Some(pos) = moves.iter().position(|&(m, _)| m == pv) {
                if pos != 0 {
                    moves.swap(0, pos);
                }
            }
        }

        for (move_idx, &(mv, is_forcing)) in moves.iter().enumerate() {
            // TT prefetch ??same trick as in alpha_beta: warm the child's
            // TT bucket while make_move + accumulator delta runs.
            let next_zob = board.zobrist
                ^ crate::board::zobrist_stone_key(board.side_to_move, mv)
                ^ crate::board::ZOBRIST_SIDE;
            let profile_start = self.profile_start();
            self.tt.prefetch(next_zob);
            self.profile_add(SearchProfileBucket::Tt, profile_start);

            let make_undo_profile_start = self.profile_start();
            let profile_start = self.profile_start();
            self.make_board_move(board, mv);
            self.profile_add(SearchProfileBucket::BoardMakeUndo, profile_start);
            self.push_threat_field(board, mv);
            self.stress_threat_field_query(board);
            let profile_start = self.profile_start();
            let detail = inc.push_move(board, mv, self.profile.enabled);
            self.profile_add(SearchProfileBucket::EvalStatePushPop, profile_start);
            self.profile.add_eval_state_detail(detail);
            self.profile_add(SearchProfileBucket::MakeUndo, make_undo_profile_start);

            let is_killer = self.killers[0][0] == Some(mv) || self.killers[0][1] == Some(mv);

            let score = if move_idx == 0 {
                -self.alpha_beta(board, weights, inc, depth - 1, 1, -beta, -alpha)
            } else {
                let reduction = lmr_reduction(depth, move_idx, is_forcing, is_killer);
                let reduced_depth = (depth - 1).saturating_sub(reduction);
                let mut null =
                    -self.alpha_beta(board, weights, inc, reduced_depth, 1, -alpha - 1, -alpha);
                // LMR re-search (same null window): reduced ??癲됱빖???嶺??????轅붽틓????? alpha ??????⑤슢堉??곕????轅붽틓?????獒뺣폍??                // full depth??????????袁④뎬????????⑤벡瑜??꿔꺂????????????븐뼐??????????룸㈇而???fail-high??? ??癲됱빖???嶺????
                if !self.aborted && reduction > 0 && null > alpha {
                    null = -self.alpha_beta(board, weights, inc, depth - 1, 1, -alpha - 1, -alpha);
                }
                if !self.aborted && null > alpha && null < beta {
                    -self.alpha_beta(board, weights, inc, depth - 1, 1, -beta, -alpha)
                } else {
                    null
                }
            };

            let make_undo_profile_start = self.profile_start();
            let profile_start = self.profile_start();
            let detail = inc.pop_move(self.profile.enabled);
            self.profile_add(SearchProfileBucket::EvalStatePushPop, profile_start);
            self.profile.add_eval_state_detail(detail);
            let profile_start = self.profile_start();
            self.undo_board_move(board);
            self.profile_add(SearchProfileBucket::BoardMakeUndo, profile_start);
            self.pop_threat_field(board);
            self.profile_add(SearchProfileBucket::MakeUndo, make_undo_profile_start);

            if self.aborted {
                break;
            }

            if score > leader_score {
                leader_score = score;
                best_move = Some(mv);
            }

            if score > alpha {
                alpha = score;
            }
        }

        (best_move, alpha)
    }

    fn alpha_beta<E: SearchEvalState>(
        &mut self,
        board: &mut Board,
        weights: &NnueWeights,
        inc: &mut E,
        depth: u32,
        ply: usize,
        mut alpha: i32,
        beta: i32,
    ) -> i32 {
        let qsearch_profile_start = self.profile_start();
        self.nodes += 1;
        self.shape_stats.main_nodes += 1;
        if self.check_node_limit() {
            self.profile_add(SearchProfileBucket::QSearch, qsearch_profile_start);
            return 0;
        }

        // ??????????븐뼐???????????(1024 ???遺얘턁????????ㅼ뒧??띤겫??눫?????????????
        if self.nodes & 127 == 0 {
            if let Some(deadline) = self.deadline {
                if Instant::now() >= deadline {
                    self.aborted = true;
                    return 0;
                }
            }
        }

        match board.game_result() {
            GameResult::BlackWin | GameResult::WhiteWin => {
                return -(WIN_SCORE - ply as i32);
            }
            GameResult::Draw => return 0,
            GameResult::Ongoing => {}
        }

        // ???????????????????袁⑸즴筌?씛彛?????quiescence lite??forcing line??????븐뼐??????? ????
        // stand-pat = NNUE static eval. ???????ル??????? stand-pat????????????遺얘턁????????
        if depth == 0 {
            return self.qsearch(board, weights, inc, 0, ply, alpha, beta);
        }

        // === TT lookup ===
        // ???????ル???? zobrist key + ???汝뷴젆?琉??誘↔덱???????depth?????? ?????????????⑤챷竊???????
        // bound ????????源낅펰???????????산뭐???alpha/beta cutoff ???????ル?????
        let original_alpha = alpha;
        let profile_start = self.profile_start();
        let tt_hit = self.tt.probe(board.zobrist);
        self.shape_stats.tt_probes += 1;
        if tt_hit.is_some() {
            self.shape_stats.tt_hits += 1;
        }
        self.profile_add(SearchProfileBucket::Tt, profile_start);
        let mut tt_move: Option<Move> = None;
        if let Some(entry) = tt_hit {
            tt_move = if entry.best_move == u16::MAX {
                None
            } else {
                Some(entry.best_move as Move)
            };
            if entry.depth as u32 >= depth {
                let cached = entry.score;
                match entry.bound {
                    Bound::Exact => {
                        self.tt_cutoffs += 1;
                        return cached;
                    }
                    Bound::Lower if cached >= beta => {
                        self.tt_cutoffs += 1;
                        return cached;
                    }
                    Bound::Upper if cached <= alpha => {
                        self.tt_cutoffs += 1;
                        return cached;
                    }
                    _ => {}
                }
            }
        }

        // === IIR (Internal Iterative Reduction) ===
        // TT-miss + ???????????? ??PV ???遺얘턁????????ㅼ뒧??띤겫??눫???1 ply ??????ш끽踰椰?????????????????⑤벡瑜??꿔꺂?????? ordering???????        // ???遺얘턁????????ㅼ뒧??띤겫??눫??????遺얘턁???????????ㅿ폍??????????????????⑥ル???????耀붾굝????????cutoff ?????? ??????ш끽踰椰?????????????????????耀붾굝?????????store??        // entry???????ル??? ???????롮쾸?椰???iteration??PV ???????ル????????????
        let is_pv = beta - alpha > 1;
        let depth = if depth >= IIR_MIN_DEPTH && tt_move.is_none() && !is_pv {
            depth - 1
        } else {
            depth
        };

        let mut best_score = -INF;
        let mut best_move_at_node: Option<Move> = None;
        let side = board.side_to_move as usize;

        // Continuation-history context: read the moves played to reach this
        // node BEFORE the loop starts making/undoing moves. `prev1` is the
        // immediate parent move; `prev2` is the move played two plies ago.
        // When the table indices are absent (root and ply 1) the updates and
        // reads simply skip the corresponding table.
        let prev1: Option<Move> = board.history.last().copied();
        let prev2: Option<Move> = if board.history.len() >= 2 {
            Some(board.history[board.history.len() - 2])
        } else {
            None
        };

        // Quiet (non-forcing) moves tried before a beta-cutoff in this node;
        // they receive negative bonuses on cutoff to discourage futures from
        // ordering them above the actual cutter.
        let mut quiets_tried: Vec<Move> = Vec::new();

        // PVS + Threat-gated LMR:
        // order_moves[0] = PV ???? ??full window ?????
        // ???????꾩룆梨띰쭕??????븐뼐???????????븐뼔??????????null-window???????????????fail-high??full re-search.
        // LMR ??????熬곣몿???: ??PV / ??killer / ??forcing ??????reduction r ply ??????ш끽踰椰???????????        // ??????⑤벡瑜??꿔꺂?????? ??????ш끽踰椰????????????耀붾굝???????alpha ??????⑤슢堉??곕????轅붽틓?????獒뺣폍??full depth??????? tier ???????????????gating????????        // ???????ル????????뀀맩鍮??????룸챶猷??????? reduce??? ?????關?쒎첎?嫄??怨룻돫?? horizon effect ???.
        let mut searched_moves = 0usize;
        // P4-R0 census: classify this node before any move is tried.
        let census_forced = if self.census.enabled {
            self.census.observe_node(board, &mut self.census_scratch)
        } else {
            None
        };

        // === P4-R1: sound forced-reply restriction ===
        // Where the opponent already threatens an immediate five, the rules fix
        // the non-losing reply set: our own five if we have one, else the single
        // completing cell, else the node is lost. Restricting to it is a theorem
        // rather than a heuristic, which is what separates this from the naive
        // LMR that reduced forcing moves and cost -43%p.
        let mut forced_only: Option<Move> = None;
        if self.use_forced_restriction {
            use crate::p4r0_census::SoundReply;
            match crate::p4r0_census::sound_reply(board, &mut self.census_scratch) {
                SoundReply::NotForced => {}
                SoundReply::WinNow(mv) | SoundReply::OnlyBlock(mv) => forced_only = Some(mv),
                SoundReply::Lost => {
                    // We reply, they complete five: the loss is exactly two plies out.
                    return -(WIN_SCORE - (ply as i32 + 2));
                }
            }
        }
        if let Some(only) = forced_only {
            let _ = self.search_alpha_beta_child(
                board,
                weights,
                inc,
                depth,
                ply,
                searched_moves,
                only,
                true,
                is_pv,
                &mut alpha,
                beta,
                side,
                prev1,
                prev2,
                &mut quiets_tried,
                &mut best_score,
                &mut best_move_at_node,
            );
            searched_moves += 1;
        } else if self.use_move_picker && ply > 0 {
            self.move_picker_stats.enabled_nodes += 1;
            let mut emitted = [false; NUM_CELLS];
            let mut stop = false;
            for stage in 0..MOVE_PICKER_STAGE_COUNT {
                let profile_start = self.profile_start();
                let stage_moves = self.generate_move_picker_stage(
                    board,
                    ply,
                    depth,
                    weights,
                    tt_move,
                    stage,
                    &mut emitted,
                );
                self.profile_add(SearchProfileBucket::MovegenOrder, profile_start);
                self.move_picker_stats.stage_reached[stage] += 1;
                self.move_picker_stats.stage_moves[stage] += stage_moves.len() as u64;
                if stage == 4 && !stage_moves.is_empty() {
                    self.move_picker_stats.quiet_generated_nodes += 1;
                }

                for (mv, is_forcing) in stage_moves {
                    stop = self.search_alpha_beta_child(
                        board,
                        weights,
                        inc,
                        depth,
                        ply,
                        searched_moves,
                        mv,
                        is_forcing,
                        is_pv,
                        &mut alpha,
                        beta,
                        side,
                        prev1,
                        prev2,
                        &mut quiets_tried,
                        &mut best_score,
                        &mut best_move_at_node,
                    );
                    searched_moves += 1;
                    if stop {
                        self.move_picker_stats.stage_cutoffs[stage] += 1;
                        if stage < 4 {
                            self.move_picker_stats.quiet_skipped_nodes += 1;
                        }
                        break;
                    }
                }

                if stop || self.aborted {
                    break;
                }
            }
        } else {
            self.move_picker_stats.legacy_nodes += 1;
            let profile_start = self.profile_start();
            let mut moves = self.order_moves(board, ply);
            self.profile_add(SearchProfileBucket::MovegenOrder, profile_start);
            if moves.is_empty() {
                return 0;
            }

            // TT-best move first for legacy generate-all ordering.
            if let Some(tt_mv) = tt_move {
                if let Some(pos) = moves.iter().position(|&(m, _)| m == tt_mv) {
                    if pos != 0 {
                        moves.swap(0, pos);
                    }
                }
            }

            for &(mv, is_forcing) in moves.iter() {
                let stop = self.search_alpha_beta_child(
                    board,
                    weights,
                    inc,
                    depth,
                    ply,
                    searched_moves,
                    mv,
                    is_forcing,
                    is_pv,
                    &mut alpha,
                    beta,
                    side,
                    prev1,
                    prev2,
                    &mut quiets_tried,
                    &mut best_score,
                    &mut best_move_at_node,
                );
                searched_moves += 1;
                if stop || self.aborted {
                    break;
                }
            }
        }

        if searched_moves == 0 {
            return 0;
        }
        if self.aborted {
            return 0;
        }
        // === TT store ===
        // bound ??????????已???????
        //   - best_score <= original_alpha ??fail-low (Upper bound, true value ??
        //   - best_score >= beta            ??fail-high (Lower bound, true value ??
        //   - ????                         ??Exact PV node
        let bound = if best_score <= original_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        if self.census.enabled {
            let observed = match bound {
                Bound::Upper => crate::p4r0_census::NodeBound::All,
                Bound::Lower => crate::p4r0_census::NodeBound::Cut,
                Bound::Exact => crate::p4r0_census::NodeBound::Exact,
            };
            self.census.record(census_forced, searched_moves, observed);
        }
        // depth???????ル??? u8????????????????????ル?????saturate. ?????????녳븢??max_depth ??20??????????????????????대첉??
        let profile_start = self.profile_start();
        self.tt.store(
            board.zobrist,
            best_score,
            depth.min(255) as u8,
            bound,
            best_move_at_node,
        );
        self.profile_add(SearchProfileBucket::Tt, profile_start);

        best_score
    }

    fn search_alpha_beta_child<E: SearchEvalState>(
        &mut self,
        board: &mut Board,
        weights: &NnueWeights,
        inc: &mut E,
        depth: u32,
        ply: usize,
        move_idx: usize,
        mv: Move,
        is_forcing: bool,
        is_pv: bool,
        alpha: &mut i32,
        beta: i32,
        side: usize,
        prev1: Option<Move>,
        prev2: Option<Move>,
        quiets_tried: &mut Vec<Move>,
        best_score: &mut i32,
        best_move_at_node: &mut Option<Move>,
    ) -> bool {
        let is_killer =
            ply < 64 && (self.killers[ply][0] == Some(mv) || self.killers[ply][1] == Some(mv));

        if !is_pv && !is_forcing && !is_killer && depth >= LMP_MIN_DEPTH && depth <= LMP_MAX_DEPTH {
            // P-CB2: policy-sorted quiets earn an earlier count cutoff
            // (frozen 6 + 2*depth vs the default 8 + 4*depth).
            let lmp_threshold = if policy_reduce_enabled() {
                6 + 2 * depth as usize
            } else {
                LMP_BASE + LMP_PER_DEPTH * depth as usize
            };
            if move_idx >= lmp_threshold {
                return false;
            }
        }

        let next_zob = board.zobrist
            ^ crate::board::zobrist_stone_key(board.side_to_move, mv)
            ^ crate::board::ZOBRIST_SIDE;
        let profile_start = self.profile_start();
        self.tt.prefetch(next_zob);
        self.profile_add(SearchProfileBucket::Tt, profile_start);

        if !is_forcing {
            quiets_tried.push(mv);
        }

        let make_undo_profile_start = self.profile_start();
        let profile_start = self.profile_start();
        self.make_board_move(board, mv);
        self.profile_add(SearchProfileBucket::BoardMakeUndo, profile_start);
        self.push_threat_field(board, mv);
        self.stress_threat_field_query(board);
        let profile_start = self.profile_start();
        let detail = inc.push_move(board, mv, self.profile.enabled);
        self.profile_add(SearchProfileBucket::EvalStatePushPop, profile_start);
        self.profile.add_eval_state_detail(detail);
        self.profile_add(SearchProfileBucket::MakeUndo, make_undo_profile_start);

        let score = if move_idx == 0 {
            -self.alpha_beta(board, weights, inc, depth - 1, ply + 1, -beta, -(*alpha))
        } else {
            let reduction = lmr_reduction(depth, move_idx, is_forcing, is_killer);
            let reduced_depth = (depth - 1).saturating_sub(reduction);
            let mut null_score = -self.alpha_beta(
                board,
                weights,
                inc,
                reduced_depth,
                ply + 1,
                -(*alpha) - 1,
                -(*alpha),
            );
            if !self.aborted && reduction > 0 && null_score > *alpha {
                null_score = -self.alpha_beta(
                    board,
                    weights,
                    inc,
                    depth - 1,
                    ply + 1,
                    -(*alpha) - 1,
                    -(*alpha),
                );
            }
            if !self.aborted && null_score > *alpha && null_score < beta {
                -self.alpha_beta(board, weights, inc, depth - 1, ply + 1, -beta, -(*alpha))
            } else {
                null_score
            }
        };

        let make_undo_profile_start = self.profile_start();
        let profile_start = self.profile_start();
        let detail = inc.pop_move(self.profile.enabled);
        self.profile_add(SearchProfileBucket::EvalStatePushPop, profile_start);
        self.profile.add_eval_state_detail(detail);
        let profile_start = self.profile_start();
        self.undo_board_move(board);
        self.profile_add(SearchProfileBucket::BoardMakeUndo, profile_start);
        self.pop_threat_field(board);
        self.profile_add(SearchProfileBucket::MakeUndo, make_undo_profile_start);

        if self.aborted {
            return true;
        }

        if score > *best_score {
            *best_score = score;
            *best_move_at_node = Some(mv);
        }
        if score > *alpha {
            *alpha = score;
            self.history[side][mv] += (depth * depth) as i32;
        }
        if *alpha >= beta {
            if ply < 64 {
                self.killers[ply][1] = self.killers[ply][0];
                self.killers[ply][0] = Some(mv);
            }
            if !is_forcing {
                let bonus = ((depth * depth) as i32).min(HISTORY_MAX);
                if let Some(p1) = prev1 {
                    history_gravity_update(&mut self.cont_hist_1[p1][mv], bonus);
                }
                if let Some(p2) = prev2 {
                    history_gravity_update(&mut self.cont_hist_2[p2][mv], bonus);
                }
                for &qm in &quiets_tried[..quiets_tried.len().saturating_sub(1)] {
                    if let Some(p1) = prev1 {
                        history_gravity_update(&mut self.cont_hist_1[p1][qm], -bonus);
                    }
                    if let Some(p2) = prev2 {
                        history_gravity_update(&mut self.cont_hist_2[p2][qm], -bonus);
                    }
                }
            }
            return true;
        }
        false
    }
    /// Quiescence lite. ???????ル?????????????遺얘턁?????????horizon effect ??????????뀀??
    /// - stand-pat (NNUE static eval) ??fail-high ?????cutoff
    /// Quiescence lite search for forcing replies at the horizon.
    fn qsearch<E: SearchEvalState>(
        &mut self,
        board: &mut Board,
        weights: &NnueWeights,
        inc: &mut E,
        qply: u32,
        ply: usize,
        mut alpha: i32,
        beta: i32,
    ) -> i32 {
        let qsearch_profile_start = self.profile_start();
        self.nodes += 1;
        self.shape_stats.qsearch_nodes += 1;
        if self.check_node_limit() {
            self.profile_add(SearchProfileBucket::QSearch, qsearch_profile_start);
            return 0;
        }

        if self.nodes & 127 == 0 {
            if let Some(deadline) = self.deadline {
                if Instant::now() >= deadline {
                    self.aborted = true;
                    self.profile_add(SearchProfileBucket::QSearch, qsearch_profile_start);
                    return 0;
                }
            }
        }

        match board.game_result() {
            GameResult::BlackWin | GameResult::WhiteWin => {
                let score = -(WIN_SCORE - ply as i32);
                self.profile_add(SearchProfileBucket::QSearch, qsearch_profile_start);
                return score;
            }
            GameResult::Draw => {
                self.profile_add(SearchProfileBucket::QSearch, qsearch_profile_start);
                return 0;
            }
            GameResult::Ongoing => {}
        }

        let profile_start = self.profile_start();
        let (stand_pat, detail) = inc.eval(board, self.profile.enabled);
        self.profile_add(SearchProfileBucket::Eval, profile_start);
        self.profile.add_eval_state_detail(detail);
        if qply >= QSEARCH_MAX_PLY {
            self.profile_add(SearchProfileBucket::QSearch, qsearch_profile_start);
            return stand_pat;
        }
        if stand_pat >= beta {
            self.profile_add(SearchProfileBucket::QSearch, qsearch_profile_start);
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let candidates = self.board_candidate_moves(board);
        if candidates.is_empty() {
            self.profile_add(SearchProfileBucket::QSearch, qsearch_profile_start);
            return stand_pat;
        }

        let (my, opp) = match board.side_to_move {
            Stone::Black => (&board.black, &board.white),
            Stone::White => (&board.white, &board.black),
        };

        // 0.6.9: cache opp_kind by scanning candidates once instead of letting
        // (a) the must-block precheck and (b) the per-move OpenFour-block
        // check each call `classify_move(opp, my, mv)` again ??together they
        // had been costing up to 2N calls. 0.7.0 swaps the per-call body for
        // the Pattern4 fast path (~10x cheaper per call), so we keep the
        // cache to lock the call count at N as well.
        let opp_side = match board.side_to_move {
            Stone::Black => Stone::White,
            Stone::White => Stone::Black,
        };
        let _ = (my, opp); // moved into classify_move_fast(board, mv, side) form
        let opp_kinds: Vec<ThreatKind> = candidates
            .iter()
            .map(|&m| classify_move_fast(board, m, opp_side))
            .collect();
        let opp_has_five = opp_kinds.iter().any(|&k| matches!(k, ThreatKind::Five));

        let mut forcing: Vec<(Move, i32)> = Vec::new();
        for (i, &mv) in candidates.iter().enumerate() {
            let opp_kind = opp_kinds[i];
            let my_kind = classify_move_fast(board, mv, board.side_to_move);

            // ????븐뼐????傭?끆?????Β?ｊ콞?轅붽틓?????⑸걦?????????諛몃마嶺뚮?????꾩렯?????must-block ????? ????轅붽틓???壤굿??덊뒌??????????怨뺤떪???????????????.
            if matches!(my_kind, ThreatKind::Five) {
                forcing.push((mv, 1_000_000));
                continue;
            }

            if opp_has_five {
                // Must-block ????븐뼐???????????븐뼔???? ??? Five ????븐뼐??????⑤슢?????壤굿??띾?????????
                if matches!(opp_kind, ThreatKind::Five) {
                    forcing.push((mv, 900_000));
                }
                continue;
            }

            // ?????????????ル??????????????????已???????packed table.
            let attack = QS_ATTACK_TABLE[my_kind as usize];
            if attack > 0 {
                forcing.push((mv, attack));
                continue;
            }

            // ??? OpenFour ????븐뼐??????⑤슢?????壤굿??띾????????????ル??????(??????opp_kind ?????.
            if matches!(opp_kind, ThreatKind::OpenFour) {
                forcing.push((mv, 700_000));
            }
        }

        if forcing.is_empty() {
            self.profile_add(SearchProfileBucket::QSearch, qsearch_profile_start);
            return stand_pat;
        }

        forcing.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        let mut best = stand_pat;
        for &(mv, _) in &forcing {
            let make_undo_profile_start = self.profile_start();
            let profile_start = self.profile_start();
            self.make_board_move(board, mv);
            self.profile_add(SearchProfileBucket::BoardMakeUndo, profile_start);
            self.push_threat_field(board, mv);
            self.stress_threat_field_query(board);
            let profile_start = self.profile_start();
            let detail = inc.push_move(board, mv, self.profile.enabled);
            self.profile_add(SearchProfileBucket::EvalStatePushPop, profile_start);
            self.profile.add_eval_state_detail(detail);
            self.profile_add(SearchProfileBucket::MakeUndo, make_undo_profile_start);
            let score = -self.qsearch(board, weights, inc, qply + 1, ply + 1, -beta, -alpha);
            let make_undo_profile_start = self.profile_start();
            let profile_start = self.profile_start();
            let detail = inc.pop_move(self.profile.enabled);
            self.profile_add(SearchProfileBucket::EvalStatePushPop, profile_start);
            self.profile.add_eval_state_detail(detail);
            let profile_start = self.profile_start();
            self.undo_board_move(board);
            self.profile_add(SearchProfileBucket::BoardMakeUndo, profile_start);
            self.pop_threat_field(board);
            self.profile_add(SearchProfileBucket::MakeUndo, make_undo_profile_start);

            if self.aborted {
                self.profile_add(SearchProfileBucket::QSearch, qsearch_profile_start);
                return 0;
            }

            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }

        self.profile_add(SearchProfileBucket::QSearch, qsearch_profile_start);
        best
    }

    /// ?????遺얘턁???????????? ??????????뀀?????? ?????????????????????⑤벡????????????????????????紐껊짍.
    /// ?????獄쏅챶留덌┼???????? (mv, is_forcing) ??is_forcing?? LMR gating?????????뀀맩鍮??????룸챶猷??????????????뀀??????????곕츥???μ떜媛?걫?繹먃??.
    ///
    /// Packs (score, is_forcing, mv) into a single u64 so the hot sort path
    /// runs on a primitive-integer slice (pdqsort kernel) instead of a
    /// struct-comparison lambda. On 30-50 candidate moves this saves ~30%
    /// of the order_moves time vs the previous `Vec<(Move, i32, bool)>`
    /// + `sort_unstable_by(|a,b| b.1.cmp(&a.1))` form. Search throughput
    /// gain ~3-5%.
    fn generate_move_picker_stage(
        &mut self,
        board: &Board,
        ply: usize,
        depth: u32,
        _weights: &NnueWeights,
        tt_move: Option<Move>,
        stage: usize,
        emitted: &mut [bool; NUM_CELLS],
    ) -> Vec<(Move, bool)> {
        let side = board.side_to_move as usize;
        let (my, opp) = match board.side_to_move {
            Stone::Black => (&board.black, &board.white),
            Stone::White => (&board.white, &board.black),
        };
        if self.use_tail_threat_materialize {
            match stage {
                0 => self.generate_tt_stage(board, ply, side, my, opp, emitted, tt_move),
                1 => self.generate_direct_urgent_stage(board, ply, side, my, opp, emitted),
                2 => self.generate_killer_stage(board, ply, side, my, opp, emitted),
                3 => self.generate_l1_threat_stage(board, ply, depth, side, my, opp, emitted, true),
                4 => self.generate_quiet_stage(board, ply, side, my, opp, emitted),
                _ => Vec::new(),
            }
        } else {
            match stage {
                0 => self.generate_tt_stage(board, ply, side, my, opp, emitted, tt_move),
                1 => {
                    self.generate_l1_threat_stage(board, ply, depth, side, my, opp, emitted, false)
                }
                2 => self.generate_forcing_stage(board, ply, side, my, opp, emitted),
                3 => self.generate_killer_stage(board, ply, side, my, opp, emitted),
                4 => self.generate_quiet_stage(board, ply, side, my, opp, emitted),
                _ => Vec::new(),
            }
        }
    }

    fn generate_tt_stage(
        &mut self,
        board: &Board,
        ply: usize,
        side: usize,
        my: &BitBoard,
        opp: &BitBoard,
        emitted: &mut [bool; NUM_CELLS],
        tt_move: Option<Move>,
    ) -> Vec<(Move, bool)> {
        let mut out = Vec::with_capacity(1);
        if let Some(mv) = tt_move {
            if let Some((_, mv, is_forcing)) =
                self.score_stage_move(board, ply, side, my, opp, emitted, mv)
            {
                out.push((mv, is_forcing));
            }
        }
        out
    }

    fn generate_l1_threat_stage(
        &mut self,
        board: &Board,
        ply: usize,
        depth: u32,
        side: usize,
        my: &BitBoard,
        opp: &BitBoard,
        emitted: &mut [bool; NUM_CELLS],
        tail_query: bool,
    ) -> Vec<(Move, bool)> {
        if tail_query && depth < 2 {
            return Vec::new();
        }
        let Some(field) = self.threat_field.as_mut() else {
            return Vec::new();
        };
        let pending = field.pending_dirty_count();
        if tail_query {
            self.move_picker_stats.tail_l1_query_nodes += 1;
            self.move_picker_stats.tail_l1_query_dirty_cells += pending as u64;
            let bucket = move_picker_dirty_hist_bucket(pending);
            self.move_picker_stats.tail_l1_query_dirty_hist[bucket] += 1;
        }
        if pending > 0 {
            self.move_picker_stats.l1_materialize_nodes += 1;
            self.move_picker_stats.l1_materialize_dirty_cells += pending as u64;
        }
        let us = board.side_to_move;
        let them = us.opponent();
        let mut sources = Vec::with_capacity(10);
        for &(source_side, kind) in &[
            (us, ThreatKind::Five),
            (them, ThreatKind::Five),
            (us, ThreatKind::OpenFour),
            (them, ThreatKind::OpenFour),
            (us, ThreatKind::DoubleFour),
            (them, ThreatKind::DoubleFour),
            (us, ThreatKind::FourThree),
            (them, ThreatKind::FourThree),
            (us, ThreatKind::ClosedFour),
            (them, ThreatKind::ClosedFour),
        ] {
            sources.push(field.tier_sources(board, source_side, kind));
        }
        let mut packed = Vec::new();
        for source in sources {
            for mv in source.iter_ones() {
                if let Some(entry) = self.score_stage_move(board, ply, side, my, opp, emitted, mv) {
                    packed.push(entry);
                }
            }
        }
        sort_packed_stage_moves(&mut packed);
        packed.into_iter().map(|(_, mv, f)| (mv, f)).collect()
    }

    fn generate_direct_urgent_stage(
        &mut self,
        board: &Board,
        ply: usize,
        side: usize,
        my: &BitBoard,
        opp: &BitBoard,
        emitted: &mut [bool; NUM_CELLS],
    ) -> Vec<(Move, bool)> {
        self.move_picker_stats.direct_urgent_nodes += 1;
        let us = board.side_to_move;
        let them = us.opponent();
        let mut packed = Vec::new();
        for mv in self.board_candidate_moves(board) {
            if emitted[mv] {
                continue;
            }
            let my_kind = classify_move_fast(board, mv, us);
            let opp_kind = classify_move_fast(board, mv, them);
            let urgent = matches!(my_kind, ThreatKind::Five | ThreatKind::OpenFour)
                || matches!(opp_kind, ThreatKind::Five | ThreatKind::OpenFour);
            if urgent {
                let (score, is_forcing) =
                    self.move_score_and_forcing(mv, ply, side, my, opp, board);
                emitted[mv] = true;
                packed.push((score, mv, is_forcing));
            }
        }
        self.move_picker_stats.direct_urgent_moves += packed.len() as u64;
        sort_packed_stage_moves(&mut packed);
        packed.into_iter().map(|(_, mv, f)| (mv, f)).collect()
    }

    fn generate_forcing_stage(
        &mut self,
        board: &Board,
        ply: usize,
        side: usize,
        my: &BitBoard,
        opp: &BitBoard,
        emitted: &mut [bool; NUM_CELLS],
    ) -> Vec<(Move, bool)> {
        let mut packed = Vec::new();
        for mv in self.board_candidate_moves(board) {
            if emitted[mv] {
                continue;
            }
            let (score, is_forcing) = self.move_score_and_forcing(mv, ply, side, my, opp, board);
            if is_forcing {
                emitted[mv] = true;
                packed.push((score, mv, is_forcing));
            }
        }
        sort_packed_stage_moves(&mut packed);
        packed.into_iter().map(|(_, mv, f)| (mv, f)).collect()
    }

    fn generate_killer_stage(
        &mut self,
        board: &Board,
        ply: usize,
        side: usize,
        my: &BitBoard,
        opp: &BitBoard,
        emitted: &mut [bool; NUM_CELLS],
    ) -> Vec<(Move, bool)> {
        let mut out = Vec::with_capacity(2);
        if ply < 64 {
            for mv in self.killers[ply] {
                if let Some(mv) = mv {
                    if let Some((_, mv, is_forcing)) =
                        self.score_stage_move(board, ply, side, my, opp, emitted, mv)
                    {
                        out.push((mv, is_forcing));
                    }
                }
            }
        }
        out
    }

    fn generate_quiet_stage(
        &mut self,
        board: &Board,
        ply: usize,
        side: usize,
        my: &BitBoard,
        opp: &BitBoard,
        emitted: &mut [bool; NUM_CELLS],
    ) -> Vec<(Move, bool)> {
        let mut packed = Vec::new();
        for mv in self.board_candidate_moves(board) {
            if emitted[mv] {
                continue;
            }
            let (score, is_forcing) = self.move_score_and_forcing(mv, ply, side, my, opp, board);
            emitted[mv] = true;
            packed.push((score, mv, is_forcing));
        }
        sort_packed_stage_moves(&mut packed);
        packed.into_iter().map(|(_, mv, f)| (mv, f)).collect()
    }

    fn score_stage_move(
        &mut self,
        board: &Board,
        ply: usize,
        side: usize,
        my: &BitBoard,
        opp: &BitBoard,
        emitted: &mut [bool; NUM_CELLS],
        mv: Move,
    ) -> Option<(i32, Move, bool)> {
        if mv >= NUM_CELLS || !board.is_legal_move(mv) {
            return None;
        }
        if emitted[mv] {
            self.move_picker_stats.duplicate_suppressed += 1;
            return None;
        }
        let (score, is_forcing) = self.move_score_and_forcing(mv, ply, side, my, opp, board);
        emitted[mv] = true;
        Some((score, mv, is_forcing))
    }
    fn order_moves(&self, board: &Board, ply: usize) -> Vec<(Move, bool)> {
        let candidates = self.board_candidate_moves(board);
        let side = board.side_to_move as usize;

        let (my, opp) = match board.side_to_move {
            Stone::Black => (&board.black, &board.white),
            Stone::White => (&board.white, &board.black),
        };

        // Layout (highest ??lowest bit):
        //   [bits 16..64]: score + SCORE_BIAS (i32 range easily fits 48 bits)
        //   [bit  9]      : is_forcing flag
        //   [bits 0..9]   : mv index (0..225 ??9 bits)
        const SCORE_BIAS: i64 = 1 << 30;
        const MV_MASK: u64 = (1 << 9) - 1;
        const FORCING_BIT: u64 = 1 << 9;

        let mut packed: Vec<u64> = candidates
            .into_iter()
            .map(|m| {
                let (s, f) = self.move_score_and_forcing(m, ply, side, my, opp, board);
                let score_u = (s as i64 + SCORE_BIAS) as u64;
                (score_u << 16) | (if f { FORCING_BIT } else { 0 }) | (m as u64)
            })
            .collect();

        // Descending order = best score first. sort_unstable on u64 hits
        // the optimized pdqsort code path directly.
        packed.sort_unstable_by(|a, b| b.cmp(a));

        packed
            .into_iter()
            .map(|p| {
                let mv = (p & MV_MASK) as Move;
                let f = (p & FORCING_BIT) != 0;
                (mv, f)
            })
            .collect()
    }

    /// ???????????遺얘턁?????????????????+ LMR-gating ??is_forcing ?????????????????
    /// is_forcing = ?????????븐뼐?????????OpenThree ?????????????????뀀???????true.
    /// ??????????LMR??reduce??? ?????????몃뒇???
    fn move_score_and_forcing(
        &self,
        mv: Move,
        ply: usize,
        side: usize,
        my: &crate::board::BitBoard,
        opp: &crate::board::BitBoard,
        board: &Board,
    ) -> (i32, bool) {
        let row = (mv / BOARD_SIZE) as i32;
        let col = (mv % BOARD_SIZE) as i32;

        let opp_side = match board.side_to_move {
            Stone::Black => Stone::White,
            Stone::White => Stone::Black,
        };
        let my_kind = classify_move_fast(board, mv, board.side_to_move);
        let opp_kind = classify_move_fast(board, mv, opp_side);

        // 0.6.9: ??????????已????????轅붽틓?????packed-table tier scoring. TIER ?????????泥????buffer???????ル???
        // ??100 000?????????????????????? max() ??癲됱빖???嶺??????轅붽틓????? if-else ????븐뼐??????????轅붽틓??????????????꾨굴????????類?킅????????⑤벡瑜????
        // (?? my OpenFour=8M > opp OpenFour=7M > my DoubleFour=6M ...)
        let attack_tier = MOVE_ATTACK_TABLE[my_kind as usize];
        let block_tier = MOVE_BLOCK_TABLE[opp_kind as usize];
        let tier_score = attack_tier.max(block_tier);

        let is_forcing = is_forcing_kind(my_kind) || is_forcing_kind(opp_kind);

        // Five ?????亦껋꼦維????????濚밸Ŧ援???early-return: ???????猷몄굡??????? score ???????밸븶筌믩끃????????????????(TIER_WIN/BLOCK_WIN
        // ????????????????뼿??tier?? ???汝뷴젆?琉??誘↔덱?????????? killer/history ??????????? ?????????대첉??.
        if matches!(my_kind, ThreatKind::Five) {
            return (TIER_WIN, true);
        }
        if matches!(opp_kind, ThreatKind::Five) {
            return (TIER_BLOCK_WIN, true);
        }

        // P-CB1: with a policy table loaded, QUIET moves (no attack and no
        // block classification) are ranked by the learned pattern-conditioned
        // score instead of history/cont-history. Killers and every tier above
        // stay untouched; with NORU_POLICY_ORDER unset this branch is dead.
        if matches!(my_kind, ThreatKind::None) && matches!(opp_kind, ThreatKind::None) {
            if let Some(table) = policy_order_table() {
                let mut score = 0i32;
                if ply < 64 {
                    if self.killers[ply][0] == Some(mv) {
                        score += 80_000;
                    } else if self.killers[ply][1] == Some(mv) {
                        score += 40_000;
                    }
                }
                score += table.quiet_score(board, mv);
                return (score, false);
            }
        }

        let mut score = tier_score;

        if ply < 64 {
            if self.killers[ply][0] == Some(mv) {
                score += 80_000;
            } else if self.killers[ply][1] == Some(mv) {
                score += 40_000;
            }
        }
        score += self.history[side][mv].min(50_000);

        // Continuation-history bonuses. The reads share the same source of
        // `prev1` / `prev2` as the alpha_beta cutoff updates: each is the
        // last (or second-last) move actually played to reach this node.
        // The right-shift trims the table's range to a budget that doesn't
        // crash through the tier separators above.
        if let Some(p1) = board.history.last() {
            score += (self.cont_hist_1[*p1][mv] >> HISTORY_SCORE_SHIFT)
                .clamp(-HISTORY_CLAMP_1, HISTORY_CLAMP_1);
        }
        if board.history.len() >= 2 {
            let p2 = board.history[board.history.len() - 2];
            score += (self.cont_hist_2[p2][mv] >> HISTORY_SCORE_SHIFT)
                .clamp(-HISTORY_CLAMP_2, HISTORY_CLAMP_2);
        }

        for &(dr, dc) in &DIR {
            let my_info = scan_line(my, opp, row, col, dr, dc);
            if my_info.count == 2 && my_info.open_front && my_info.open_back {
                score += 200;
            }
            let opp_info = scan_line(opp, my, row, col, dr, dc);
            if opp_info.count == 2 && opp_info.open_front && opp_info.open_back {
                score += 150;
            }
        }

        // 0.6.5 (2026-04-27): center bonus ???? quiet move ordering?????        // ????遺용퉻???????? ???????????耀붾굝????????+ ??????꾩룆梨띰쭕????opening (Pela swap2 ?? ?????????        // ????븐뼐??????????癲됱빖???嶺???? 14 - center_dist????????⑤벡瑜??????????룸ı?????????댁삩?????븐뼐?????? 14 ????븐뼐??????⑤슢?????壤굿??띾????????????濡?씀?濾????        // ????븐뼐???????? ??????quiet ??????killer/history/scan-line(open-2)????븐뼐???????????        // tie-break.

        (score, is_forcing)
    }
}

fn sort_packed_stage_moves(moves: &mut [(i32, Move, bool)]) {
    moves.sort_unstable_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| b.1.cmp(&a.1))
    });
}
// === Move ordering tier ?????===
// ??tier ????buffer???????ル??? ??100 000??????????history/killer/center ???????밸븶筌믩끃?????// ?????????뼿??tier?? ??? ???汝뷴젆?琉??誘↔덱???????? ???????繹먮굞???
const TIER_WIN: i32 = 10_000_000;
const TIER_BLOCK_WIN: i32 = 9_000_000;
const TIER_OPEN_FOUR: i32 = 8_000_000;
const TIER_BLOCK_OPEN_FOUR: i32 = 7_000_000;
const TIER_DOUBLE_FOUR: i32 = 6_000_000;
const TIER_BLOCK_DOUBLE_FOUR: i32 = 5_000_000;
const TIER_DOUBLE_THREE: i32 = 4_000_000;
const TIER_BLOCK_DOUBLE_THREE: i32 = 3_000_000;
const TIER_CLOSED_FOUR: i32 = 1_500_000;
const TIER_BLOCK_CLOSED_FOUR: i32 = 1_400_000;
const TIER_OPEN_THREE: i32 = 1_000_000;
const TIER_BLOCK_OPEN_THREE: i32 = 900_000;

// === Branchless threat-score tables (0.6.9) ===
// `ThreatKind as usize` ???遺얘턁???????? ???????vct.rs??#[repr(u8)] discriminant?? ??????濡?씀?濾????源낆졒??
//   0=None  1=ClosedFour  2=OpenThree  3=Five
//   4=OpenFour  5=DoubleFour  6=FourThree  7=DoubleThree
// ??????⑤벡瑜??????vct.rs??ThreatKind discriminant???????????????ろ떀????????⑤슢堉??곕?????????濚밸Ŧ援욃퐲????

/// Move ordering: ????????????뀀????????븐뼐??????????attack tier ?????
const MOVE_ATTACK_TABLE: [i32; THREAT_KIND_COUNT] = [
    0,                 // None
    TIER_CLOSED_FOUR,  // ClosedFour
    TIER_OPEN_THREE,   // OpenThree
    TIER_WIN,          // Five
    TIER_OPEN_FOUR,    // OpenFour
    TIER_DOUBLE_FOUR,  // DoubleFour
    TIER_DOUBLE_FOUR,  // FourThree
    TIER_DOUBLE_THREE, // DoubleThree
    TIER_OPEN_THREE,   // JumpThree
];

/// Move ordering: ??? ??????????뀀???????븐뼐??????⑤슢?????壤굿??띾????tier ?????
const MOVE_BLOCK_TABLE: [i32; THREAT_KIND_COUNT] = [
    0,                       // None
    TIER_BLOCK_CLOSED_FOUR,  // ClosedFour
    TIER_BLOCK_OPEN_THREE,   // OpenThree
    TIER_BLOCK_WIN,          // Five
    TIER_BLOCK_OPEN_FOUR,    // OpenFour
    TIER_BLOCK_DOUBLE_FOUR,  // DoubleFour
    TIER_BLOCK_DOUBLE_FOUR,  // FourThree
    TIER_BLOCK_DOUBLE_THREE, // DoubleThree
    TIER_BLOCK_OPEN_THREE,   // JumpThree
];

/// `is_forcing` ???????癲ル슢?ο㎖?????????⑤벡???? bit i set ??ThreatKind discriminant i???????ル??? forcing.
/// ??????????뀀???嶺?forcing ???遺얘턁?????????? ClosedFour, OpenThree, Five, OpenFour, DoubleFour, FourThree.
/// (DoubleThree?????耀붾굝??????????????????????關?쒎첎?????蹂?????.)
const FORCING_MASK: u16 =
    (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 8);

/// qsearch attack-tier ?????(????????????뀀???. Five????????⑤벡瑜?????????븐뼐???????????癲ル슢??룸퀬苑???????獄쏅챶留덌┼???????????븐뼐????傭?끆?????Β?ｊ콞?轅붽틓?????⑸걦???cutoff).
const QS_ATTACK_TABLE: [i32; THREAT_KIND_COUNT] = [
    0,       // None
    0,       // ClosedFour
    0,       // OpenThree
    0,       // Five; immediate wins are handled by caller
    800_000, // OpenFour
    600_000, // DoubleFour
    600_000, // FourThree
    0,       // DoubleThree
    0,       // JumpThree
];

#[inline]
fn is_forcing_kind(kind: ThreatKind) -> bool {
    (FORCING_MASK >> (kind as u16)) & 1 != 0
}

/// Threat-gated LMR reduction ????????????
/// ???????ル??????/ killer / ??LMR_MIN_MOVE_IDX ?????/ ??? depth??0 (????????ш끽踰椰??????????.
/// ??????forcing tier 0 ????? depth/idx??????????산뭐???1~2 ply.
/// reduction?? depth-2????? ??????癲ル슢?????cap (qsearch ????븐뼐??????????롮쾸?椰?????????獄쏅챶留??逆곷틳源븃떋?).
fn lmr_reduction(depth: u32, move_idx: usize, is_forcing: bool, is_killer: bool) -> u32 {
    if depth < LMR_MIN_DEPTH || move_idx < LMR_MIN_MOVE_IDX || is_forcing || is_killer {
        return 0;
    }
    let mut r = 1u32;
    if depth >= 6 {
        r += 1;
    }
    if move_idx >= 6 {
        r += 1;
    }
    // P-CB2: with policy-sorted quiets, deep-late = policy-junk. Frozen
    // rank thresholds 10 (+1) and 18 (+1), same depth-2 cap.
    if policy_reduce_enabled() {
        if move_idx >= 10 {
            r += 1;
        }
        if move_idx >= 18 {
            r += 1;
        }
    }
    r.min(depth.saturating_sub(2))
}

#[allow(dead_code)]
fn threat_priority(kind: ThreatKind, defending: bool) -> i32 {
    // ???遺얘턁?????꿔꺂?????諛ㅻ???stub ????move_score??inline tier??????븐뼐????????????븐뼐???????????癲ル슢??룸퀬苑??
    let base = match kind {
        ThreatKind::Five => 1_000_000,
        ThreatKind::OpenFour => 500_000,
        ThreatKind::DoubleFour | ThreatKind::FourThree => 300_000,
        ThreatKind::DoubleThree => 200_000,
        ThreatKind::ClosedFour => 100_000,
        ThreatKind::OpenThree => 30_000,
        ThreatKind::JumpThree => 30_000,
        ThreatKind::None => 0,
    };
    // ????????????????????????????????? ????(??? ?????關?쒎첎?嫄??怨몄겮???????? ??? ????븐뼐???????????癲됱빖???嶺??????????????).
    if defending { base * 9 / 10 } else { base }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_table_desaturation_rescales_only_saturating_tables() {
        // Saturating table: max|entry| = 45 -> 8*45 = 360 >= 30.
        let mut attack: Vec<f32> = (0..64).map(|i| ((i % 19) as f32 - 9.0) * 5.0).collect();
        let mut defense: Vec<f32> = (0..64).map(|i| ((i % 13) as f32 - 6.0) * 3.5).collect();
        attack[7] = 45.0;
        defense[3] = -44.0;
        let orig_a = attack.clone();
        let orig_d = defense.clone();
        let sum8 = |a: &[f32], d: &[f32], ids: [usize; 4]| -> f32 {
            // 4 directions x (A[my] + D[opp]) = 8 entries, as in quiet_score.
            ids.iter().map(|&i| a[i] + d[(i * 5 + 1) % 64]).sum::<f32>()
        };
        let samples: [[usize; 4]; 5] = [
            [7, 7, 7, 7],
            [0, 1, 2, 3],
            [10, 20, 30, 40],
            [5, 9, 17, 33],
            [60, 61, 62, 63],
        ];
        let before: Vec<f32> = samples.iter().map(|&ids| sum8(&orig_a, &orig_d, ids)).collect();
        assert!(before.iter().any(|s| s.abs() >= 30.0), "fixture must saturate");

        let k = desaturate_policy_tables(&mut attack, &mut defense).expect("must rescale");
        assert!((k - 8.0 * 45.0 / 29.0).abs() < 1e-9);
        let max_after = attack
            .iter()
            .chain(defense.iter())
            .fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(8.0 * max_after < 30.0, "8*max after = {}", 8.0 * max_after);
        let after: Vec<f32> = samples.iter().map(|&ids| sum8(&attack, &defense, ids)).collect();
        for i in 0..samples.len() {
            for j in 0..samples.len() {
                if before[i] < before[j] {
                    assert!(after[i] < after[j], "order flipped for samples {i},{j}");
                }
            }
            assert!(after[i].abs() < 30.0, "sample {i} still saturates: {}", after[i]);
        }

        // Non-saturating table: max|entry| = 3 -> 24 < 30, bit-identical.
        let mut small_a: Vec<f32> = (0..64).map(|i| ((i % 7) as f32 - 3.0) * 0.99).collect();
        let mut small_d: Vec<f32> = (0..64).map(|i| ((i % 5) as f32 - 2.0) * 1.5).collect();
        small_a[0] = 3.0;
        let (ref_a, ref_d) = (small_a.clone(), small_d.clone());
        assert!(desaturate_policy_tables(&mut small_a, &mut small_d).is_none());
        assert!(small_a.iter().zip(&ref_a).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(small_d.iter().zip(&ref_d).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    use crate::board::{Board, to_idx};
    #[cfg(feature = "codebook-eval")]
    use crate::factored_codebook::PackedCodebookArtifact;
    use crate::features::GOMOKU_NNUE_CONFIG;

    #[cfg(feature = "codebook-eval")]
    fn install_white_root_order_entries(
        searcher: &mut Searcher,
        board: &Board,
        entries: &[(Move, f32, bool)],
    ) {
        searcher.white_root_order = Some(WhiteRootOrder::production());
        let mut cache = WhiteRootOrderCache::new(board.zobrist);
        for &(mv, residual, quiet_ongoing) in entries {
            cache.entries[mv] = Some(WhiteRootOrderEntry {
                residual,
                quiet_ongoing,
            });
        }
        searcher.white_root_order_cache = Some(cache);
    }

    #[cfg(feature = "codebook-eval")]
    #[test]
    fn directional_delta_selector_supports_rollback() {
        // 0.8.6: directional delta is the default; the setter still allows
        // rolling back to the legacy path for A/B measurement.
        let mut searcher = Searcher::new();
        assert!(searcher.use_codebook_directional_delta);

        searcher.set_use_codebook_directional_delta(false);
        assert!(!searcher.use_codebook_directional_delta);

        searcher.set_use_codebook_directional_delta(true);
        assert!(searcher.use_codebook_directional_delta);
    }

    #[cfg(feature = "codebook-eval")]
    #[test]
    fn factored_quantized_search_matches_reconstructed_flat_smoke() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models/gomoku_codebook_v1_swapclosed_factored.cbf");
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let factored = PackedCodebookArtifact::parse(&bytes)
            .expect("valid CB-F1 artifact")
            .into_factored_quantized()
            .expect("factored payload");
        let flat = factored.reconstruct_flat();
        let ordering = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);

        let mut flat_board = Board::new();
        flat_board.make_move(to_idx(7, 7));
        let mut factored_board = flat_board.clone();
        let mut flat_searcher = Searcher::new();
        let mut factored_searcher = Searcher::new();
        flat_searcher.set_use_codebook_directional_delta(true);
        factored_searcher.set_use_codebook_directional_delta(true);
        flat_searcher.set_white_root_order_enabled(true).unwrap();
        factored_searcher
            .set_white_root_order_enabled(true)
            .unwrap();

        let flat_result = flat_searcher.search_codebook_eval_quantized(
            &mut flat_board,
            &ordering,
            &flat,
            2,
            None,
        );
        let factored_result = factored_searcher.search_codebook_eval_quantized_factored(
            &mut factored_board,
            &ordering,
            &factored,
            2,
            None,
        );
        assert_eq!(factored_result.best_move, flat_result.best_move);
        assert_eq!(factored_result.score, flat_result.score);
        assert_eq!(factored_result.depth, flat_result.depth);
        assert_eq!(factored_result.nodes, flat_result.nodes);
        assert!(factored_board.black == flat_board.black);
        assert!(factored_board.white == flat_board.white);
        assert_eq!(factored_board.history, flat_board.history);
        assert_eq!(factored_board.zobrist, flat_board.zobrist);
    }

    #[cfg(feature = "codebook-eval")]
    #[test]
    fn white_root_order_preserves_barriers_and_reorders_only_quiet_runs() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        assert_eq!(board.side_to_move, Stone::White);

        let mut searcher = Searcher::new();
        searcher.killers[0][0] = Some(54);
        install_white_root_order_entries(
            &mut searcher,
            &board,
            &[
                (50, 0.0, true),
                (51, 2.0, true),
                (52, 9.0, true),
                (53, 8.0, false),
                (54, 7.0, true),
                (55, 0.0, true),
                (56, 2.0, true),
            ],
        );
        let mut moves = (50..=56).map(|mv| (mv, false)).collect::<Vec<_>>();
        searcher.apply_white_root_order(&board, &mut moves, Some(52));
        assert_eq!(
            moves.iter().map(|&(mv, _)| mv).collect::<Vec<_>>(),
            vec![51, 50, 52, 53, 54, 56, 55]
        );
    }

    #[cfg(feature = "codebook-eval")]
    #[test]
    fn white_root_order_builds_one_reusable_cache_for_a_white_root() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        let codebook = CodebookWeights::deterministic(16, 8).quantize_i16_s32_s64();
        let mut searcher = Searcher::new();
        searcher.set_white_root_order_enabled(true).unwrap();
        searcher
            .prepare_white_root_order_cache(&board, &codebook)
            .unwrap();

        let cache = searcher.white_root_order_cache.as_ref().unwrap();
        assert_eq!(cache.root_zobrist, board.zobrist);
        assert!(
            board
                .candidate_moves()
                .into_iter()
                .any(|mv| cache.entry(mv).is_some_and(|entry| entry.quiet_ongoing))
        );
        let baseline = searcher.order_moves(&board, 0);
        let mut first = baseline.clone();
        let mut repeated = baseline;
        searcher.apply_white_root_order(&board, &mut first, None);
        searcher.apply_white_root_order(&board, &mut repeated, None);
        assert_eq!(first, repeated);
    }

    #[cfg(feature = "codebook-eval")]
    #[test]
    fn white_root_order_is_unobservable_on_black_forced_and_terminal_roots() {
        let weights = CodebookWeights::deterministic(16, 8).quantize_i16_s32_s64();

        let black = Board::new();
        let mut black_searcher = Searcher::new();
        black_searcher.set_white_root_order_enabled(true).unwrap();
        black_searcher
            .prepare_white_root_order_cache(&black, &weights)
            .unwrap();
        assert!(black_searcher.white_root_order_cache.is_none());

        let mut standard = Board::new();
        standard.set_rule_set(RuleSet::Standard);
        standard.make_move(to_idx(7, 7));
        let mut standard_searcher = Searcher::new();
        standard_searcher
            .set_white_root_order_enabled(true)
            .unwrap();
        standard_searcher
            .prepare_white_root_order_cache(&standard, &weights)
            .unwrap();
        assert!(standard_searcher.white_root_order_cache.is_none());

        let mut forced = Board::new();
        for mv in [
            to_idx(7, 3),
            to_idx(0, 0),
            to_idx(7, 4),
            to_idx(0, 2),
            to_idx(7, 5),
            to_idx(0, 4),
            to_idx(7, 6),
        ] {
            forced.make_move(mv);
        }
        let mut forced_searcher = Searcher::new();
        forced_searcher.set_white_root_order_enabled(true).unwrap();
        forced_searcher
            .prepare_white_root_order_cache(&forced, &weights)
            .unwrap();
        assert!(forced_searcher.white_root_order_cache.is_none());

        let mut terminal = Board::new();
        for mv in [
            to_idx(7, 3),
            to_idx(0, 0),
            to_idx(7, 4),
            to_idx(0, 2),
            to_idx(7, 5),
            to_idx(0, 4),
            to_idx(7, 6),
            to_idx(0, 6),
            to_idx(7, 7),
        ] {
            terminal.make_move(mv);
        }
        assert_ne!(terminal.game_result(), GameResult::Ongoing);
        let mut terminal_searcher = Searcher::new();
        terminal_searcher
            .set_white_root_order_enabled(true)
            .unwrap();
        terminal_searcher
            .prepare_white_root_order_cache(&terminal, &weights)
            .unwrap();
        assert!(terminal_searcher.white_root_order_cache.is_none());
    }

    #[cfg(feature = "codebook-eval")]
    #[test]
    #[should_panic(expected = "requires the quantized codebook search path")]
    fn white_root_order_rejects_generic_nnue_search() {
        let mut board = Board::new();
        let ordering = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);
        let mut searcher = Searcher::new();
        searcher.set_white_root_order_enabled(true).unwrap();
        let _ = searcher.search(&mut board, &ordering, 1, None);
    }

    #[cfg(feature = "codebook-eval")]
    #[test]
    #[should_panic(expected = "cannot run on the float codebook search path")]
    fn white_root_order_rejects_float_codebook_search() {
        let mut board = Board::new();
        let ordering = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);
        let codebook = CodebookWeights::deterministic(16, 8);
        let mut searcher = Searcher::new();
        searcher.set_white_root_order_enabled(true).unwrap();
        let _ = searcher.search_codebook_eval(&mut board, &ordering, &codebook, 1, None);
    }

    #[test]
    fn test_search_finds_winning_move() {
        let mut board = Board::new();
        let weights = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);

        // Black has four in a row; search should find a winning endpoint.
        board.make_move(to_idx(7, 3));
        board.make_move(to_idx(8, 3));
        board.make_move(to_idx(7, 4));
        board.make_move(to_idx(8, 4));
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(8, 5));
        board.make_move(to_idx(7, 6));
        board.make_move(to_idx(8, 6));

        let mut searcher = Searcher::new();
        let result = searcher.search(&mut board, &weights, 2, None);

        let winning_moves = [to_idx(7, 7), to_idx(7, 2)];
        assert!(result.best_move.is_some());
        assert!(
            winning_moves.contains(&result.best_move.unwrap()),
            "should find the winning move, got {:?}",
            result.best_move
        );
    }

    #[test]
    fn test_search_depth_1() {
        let mut board = Board::new();
        let weights = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);
        let mut searcher = Searcher::new();
        let result = searcher.search(&mut board, &weights, 1, None);
        assert!(result.best_move.is_some());
    }

    #[test]
    fn search_resynchronizes_retained_sidecar_for_replacement_board() {
        let weights = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);
        let mut accelerated = Searcher::new();
        accelerated.set_use_packed_line_windows(true);
        accelerated.set_use_candidate_frontier(true);

        // Seed a retained A2 sidecar for a different root.
        let mut first_root = Board::new();
        let _ = accelerated.search(&mut first_root, &weights, 1, None);
        assert!(
            accelerated
                .board_search_state
                .as_ref()
                .is_some_and(|state| state.is_synchronized(&first_root))
        );

        let mut replacement = Board::new();
        for mv in [to_idx(7, 7), to_idx(0, 0), to_idx(14, 14), to_idx(0, 14)] {
            replacement.make_move(mv);
        }
        let expected_root = replacement.clone();
        let mut baseline_board = replacement.clone();
        let mut baseline_searcher = Searcher::new();

        let actual = accelerated.search(&mut replacement, &weights, 2, None);
        let expected = baseline_searcher.search(&mut baseline_board, &weights, 2, None);

        assert_eq!(actual.best_move, expected.best_move);
        assert_eq!(actual.score, expected.score);
        assert_eq!(actual.depth, expected.depth);
        assert_eq!(actual.nodes, expected.nodes);

        assert!(replacement.black == expected_root.black);
        assert!(replacement.white == expected_root.white);
        assert_eq!(replacement.side_to_move, expected_root.side_to_move);
        assert_eq!(replacement.move_count, expected_root.move_count);
        assert_eq!(replacement.last_move, expected_root.last_move);
        assert_eq!(replacement.history, expected_root.history);
        assert_eq!(replacement.zobrist, expected_root.zobrist);
        assert_eq!(replacement.line_pattern_ids, expected_root.line_pattern_ids);
        let retained = accelerated
            .board_search_state
            .as_ref()
            .expect("A2 sidecar should remain retained");
        assert!(retained.is_synchronized(&replacement));
        assert!(!retained.candidate_frontier_enabled());
    }

    #[test]
    fn d4_hash_sidecar_selector_defaults_off_and_supports_rollback() {
        let mut searcher = Searcher::new();
        assert!(!searcher.d4_hash_sidecar_requested());
        assert!(searcher.root_vct_requested_for_audit());

        searcher.set_use_d4_hash_sidecar(true);
        assert!(searcher.d4_hash_sidecar_requested());
        searcher.set_use_root_vct_for_audit(false);
        assert!(!searcher.root_vct_requested_for_audit());

        searcher.set_use_d4_hash_sidecar(false);
        assert!(!searcher.d4_hash_sidecar_requested());
        searcher.set_use_root_vct_for_audit(true);
        assert!(searcher.root_vct_requested_for_audit());
    }

    #[test]
    fn d4_hash_sidecar_is_search_transparent_and_retained() {
        let weights = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);
        let mut root = Board::new();
        for mv in [
            to_idx(7, 7),
            to_idx(3, 3),
            to_idx(8, 8),
            to_idx(3, 4),
            to_idx(7, 8),
            to_idx(4, 4),
        ] {
            root.make_move(mv);
        }
        let expected_root = root.clone();
        let mut off_board = root.clone();
        let mut on_board = root;

        let mut off = Searcher::new();
        assert!(!off.d4_hash_sidecar_requested());
        off.set_use_root_vct_for_audit(false);
        let off_result = off.search(&mut off_board, &weights, 2, None);
        assert!(
            off.board_search_state.is_none(),
            "default search must not allocate a sidecar"
        );

        let mut on = Searcher::new();
        on.set_use_d4_hash_sidecar(true);
        on.set_use_root_vct_for_audit(false);
        let on_result = on.search(&mut on_board, &weights, 2, None);

        assert_eq!(on_result.best_move, off_result.best_move);
        assert_eq!(on_result.score, off_result.score);
        assert_eq!(on_result.depth, off_result.depth);
        assert_eq!(on_result.nodes, off_result.nodes);

        let assert_root_unchanged = |board: &Board| {
            assert!(board.black == expected_root.black);
            assert!(board.white == expected_root.white);
            assert_eq!(board.side_to_move, expected_root.side_to_move);
            assert_eq!(board.move_count, expected_root.move_count);
            assert_eq!(board.last_move, expected_root.last_move);
            assert_eq!(board.history, expected_root.history);
            assert_eq!(board.zobrist, expected_root.zobrist);
            assert_eq!(board.line_pattern_ids, expected_root.line_pattern_ids);
            assert_eq!(board.rule_set, expected_root.rule_set);
            assert_eq!(board.exact5, expected_root.exact5);
        };
        assert_root_unchanged(&off_board);
        assert_root_unchanged(&on_board);

        let retained = on
            .board_search_state
            .as_ref()
            .expect("D4 hash sidecar should remain retained");
        assert!(retained.is_synchronized(&on_board));
        assert!(retained.d4_hash_enabled());
        assert_eq!(
            retained.d4_hashes(&on_board),
            Some(*crate::d4_hash::D4HashState::rebuild(&on_board).hashes())
        );
        assert!(!retained.packed_line_windows_enabled());
        assert!(!retained.candidate_frontier_enabled());
    }
}
