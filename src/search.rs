/// ????????ㅻ깹???????????????袁ｋ쨨?? ??????轅붽틓?????(NNUE ???)
///
/// - ?????獄쏅챶留덌┼???????????(Iterative Deepening)
/// - ????????ㅻ깹???????????????袁ｋ쨨?? ???????ル???????븐뼐???????????筌뤾퍓愿?????????ъ몴??/// - ??????????뀀?????? ????????????????????遺얘턁????????????(4??????? ???????ル????)
/// - ?????????+ ????????⑤벡????????????????????????紐껊짍
/// - ??????????????
use crate::board::{BOARD_SIZE, Board, GameResult, Move, NUM_CELLS, Stone};
#[cfg(feature = "codebook-eval")]
use crate::codebook_eval::{CodebookWeights, IncrementalCodebookEval};
use crate::eval::IncrementalEval;
use crate::heuristic::{DIR, scan_line};
use crate::transposition::{Bound, TranspositionTable, TtStats};
use crate::vct::{THREAT_KIND_COUNT, ThreatKind, VctConfig, classify_move_fast, search_vct};
use noru::network::NnueWeights;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const INF: i32 = 1_000_000;
const WIN_SCORE: i32 = 999_000;

#[inline]
fn is_win_score(score: i32) -> bool {
    score.abs() >= WIN_SCORE - 1_000
}

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

fn root_vct_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NORU_ROOT_VCT")
            .map(|raw| {
                let trimmed = raw.trim();
                !(trimmed == "0"
                    || trimmed.eq_ignore_ascii_case("false")
                    || trimmed.eq_ignore_ascii_case("off")
                    || trimmed.eq_ignore_ascii_case("no"))
            })
            .unwrap_or(true)
    })
}

#[cfg(feature = "codebook-eval")]
fn codebook_eval_scale() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NORU_CODEBOOK_EVAL_SCALE")
            .ok()
            .and_then(|raw| raw.trim().parse::<f32>().ok())
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .unwrap_or(400.0)
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

fn candidate_ranker_order_topk() -> usize {
    static TOPK: OnceLock<usize> = OnceLock::new();
    *TOPK.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_RANKER_ORDER_TOPK")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(0)
    })
}

fn candidate_ranker_order_tiebreak_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_RANKER_ORDER_TIEBREAK")
            .map(|raw| {
                let trimmed = raw.trim();
                !(trimmed.is_empty()
                    || trimmed.eq_ignore_ascii_case("0")
                    || trimmed.eq_ignore_ascii_case("false")
                    || trimmed.eq_ignore_ascii_case("off")
                    || trimmed.eq_ignore_ascii_case("no"))
            })
            .unwrap_or(false)
    })
}

fn candidate_ranker_order_tie_margin() -> u64 {
    static MARGIN: OnceLock<u64> = OnceLock::new();
    *MARGIN.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_RANKER_ORDER_TIE_MARGIN")
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .unwrap_or(0)
    })
}

fn candidate_local_ab_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_LOCAL_ROOT_AB_PROBE")
            .map(|raw| {
                let trimmed = raw.trim();
                !(trimmed.is_empty()
                    || trimmed.eq_ignore_ascii_case("0")
                    || trimmed.eq_ignore_ascii_case("false")
                    || trimmed.eq_ignore_ascii_case("off")
                    || trimmed.eq_ignore_ascii_case("no"))
            })
            .unwrap_or(false)
    })
}

fn candidate_local_ab_probe_depth() -> u32 {
    static DEPTH: OnceLock<u32> = OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_LOCAL_ROOT_AB_PROBE_DEPTH")
            .ok()
            .and_then(|raw| raw.trim().parse::<u32>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(2)
    })
}

fn candidate_local_ab_probe_topk() -> usize {
    static TOPK: OnceLock<usize> = OnceLock::new();
    *TOPK.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_LOCAL_ROOT_AB_PROBE_TOPK")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(8)
    })
}

fn candidate_local_ab_probe_candidates() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_LOCAL_ROOT_AB_PROBE_CANDIDATES")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(3)
    })
}

fn candidate_local_ab_probe_min_ply() -> usize {
    static MIN_PLY: OnceLock<usize> = OnceLock::new();
    *MIN_PLY.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_LOCAL_ROOT_AB_PROBE_MIN_PLY")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(0)
    })
}

fn candidate_local_ab_probe_min_depth() -> u32 {
    static MIN_DEPTH: OnceLock<u32> = OnceLock::new();
    *MIN_DEPTH.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_LOCAL_ROOT_AB_PROBE_MIN_DEPTH")
            .ok()
            .and_then(|raw| raw.trim().parse::<u32>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(2)
    })
}

fn candidate_local_ab_probe_margin() -> i32 {
    static MARGIN: OnceLock<i32> = OnceLock::new();
    *MARGIN.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_LOCAL_ROOT_AB_PROBE_MARGIN")
            .ok()
            .and_then(|raw| raw.trim().parse::<i32>().ok())
            .filter(|value| *value >= 0)
            .unwrap_or(0)
    })
}

fn candidate_local_ab_probe_min_local_delta() -> i32 {
    static DELTA: OnceLock<i32> = OnceLock::new();
    *DELTA.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_LOCAL_ROOT_AB_PROBE_MIN_LOCAL_DELTA")
            .ok()
            .and_then(|raw| raw.trim().parse::<i32>().ok())
            .unwrap_or(0)
    })
}

fn candidate_local_ab_probe_node_limit() -> Option<u64> {
    static LIMIT: OnceLock<Option<u64>> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_LOCAL_ROOT_AB_PROBE_NODE_LIMIT")
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .filter(|value| *value > 0)
    })
}

fn candidate_ranker_order_gate_allows_pair(
    board: &Board,
    a: Move,
    b: Move,
    mode: crate::candidate_ranker::RootGateMode,
) -> bool {
    let a_gate = crate::candidate_ranker::root_gate_key(board, a);
    let b_gate = crate::candidate_ranker::root_gate_key(board, b);
    crate::candidate_ranker::gate_allows(mode, a_gate, b_gate)
        && crate::candidate_ranker::gate_allows(mode, b_gate, a_gate)
}

fn candidate_ranker_root_tiebreak_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_RANKER_ROOT_TIEBREAK")
            .map(|raw| {
                let trimmed = raw.trim();
                !(trimmed.is_empty()
                    || trimmed.eq_ignore_ascii_case("0")
                    || trimmed.eq_ignore_ascii_case("false")
                    || trimmed.eq_ignore_ascii_case("off")
                    || trimmed.eq_ignore_ascii_case("no"))
            })
            .unwrap_or(true)
    })
}

/// Defensive open-four probe mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DefensiveOpen4ProbeMode {
    Off,
    Trace,
    Demote,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum DefensiveOpen4Risk {
    Safe,
    ForcedBlockThenWinningThreat,
    ImmediateWinningThreat,
    ImmediateFive,
}

impl DefensiveOpen4Risk {
    fn label(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::ForcedBlockThenWinningThreat => "forced-block-then-winning-threat",
            Self::ImmediateWinningThreat => "immediate-winning-threat",
            Self::ImmediateFive => "immediate-five",
        }
    }
}

fn candidate_ranker_root_final_only_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_RANKER_ROOT_FINAL_ONLY")
            .map(|raw| {
                let trimmed = raw.trim();
                !(trimmed.is_empty()
                    || trimmed.eq_ignore_ascii_case("0")
                    || trimmed.eq_ignore_ascii_case("false")
                    || trimmed.eq_ignore_ascii_case("off")
                    || trimmed.eq_ignore_ascii_case("no"))
            })
            .unwrap_or(false)
    })
}

fn candidate_ranker_root_rescue_only_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_RANKER_ROOT_RESCUE_ONLY")
            .map(|raw| {
                let trimmed = raw.trim();
                !(trimmed.is_empty()
                    || trimmed.eq_ignore_ascii_case("0")
                    || trimmed.eq_ignore_ascii_case("false")
                    || trimmed.eq_ignore_ascii_case("off")
                    || trimmed.eq_ignore_ascii_case("no"))
            })
            .unwrap_or(false)
    })
}

fn candidate_ranker_root_min_stones() -> usize {
    static MIN_STONES: OnceLock<usize> = OnceLock::new();
    *MIN_STONES.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_RANKER_ROOT_MIN_STONES")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(0)
    })
}

fn candidate_ranker_root_rescue_min_block_tier() -> i32 {
    static TIER: OnceLock<i32> = OnceLock::new();
    *TIER.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_RANKER_ROOT_RESCUE_MIN_BLOCK")
            .ok()
            .and_then(|raw| {
                let trimmed = raw.trim();
                if trimmed.eq_ignore_ascii_case("open-three")
                    || trimmed.eq_ignore_ascii_case("open_three")
                    || trimmed.eq_ignore_ascii_case("openthree")
                {
                    Some(TIER_BLOCK_OPEN_THREE)
                } else if trimmed.eq_ignore_ascii_case("closed-four")
                    || trimmed.eq_ignore_ascii_case("closed_four")
                    || trimmed.eq_ignore_ascii_case("closedfour")
                {
                    Some(TIER_BLOCK_CLOSED_FOUR)
                } else if trimmed.eq_ignore_ascii_case("double-three")
                    || trimmed.eq_ignore_ascii_case("double_three")
                    || trimmed.eq_ignore_ascii_case("doublethree")
                {
                    Some(TIER_BLOCK_DOUBLE_THREE)
                } else {
                    trimmed.parse::<i32>().ok()
                }
            })
            .filter(|tier| *tier > 0)
            .unwrap_or(TIER_BLOCK_CLOSED_FOUR)
    })
}

fn candidate_ranker_root_rescue_allows(board: &Board, candidate: Move, incumbent: Move) -> bool {
    let defender = board.side_to_move.opponent();
    let candidate_block = classify_move_fast(board, candidate, defender);
    let incumbent_block = classify_move_fast(board, incumbent, defender);
    let candidate_tier = MOVE_BLOCK_TABLE[candidate_block as usize];
    let incumbent_tier = MOVE_BLOCK_TABLE[incumbent_block as usize];
    candidate_tier >= candidate_ranker_root_rescue_min_block_tier()
        && candidate_tier > incumbent_tier
}

fn defensive_open4_probe_mode() -> DefensiveOpen4ProbeMode {
    static MODE: OnceLock<DefensiveOpen4ProbeMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let Ok(raw) = std::env::var("NORU_DEFENSIVE_OPEN4_PROBE") else {
            return DefensiveOpen4ProbeMode::Off;
        };
        let trimmed = raw.trim();
        if trimmed.is_empty()
            || trimmed == "0"
            || trimmed.eq_ignore_ascii_case("off")
            || trimmed.eq_ignore_ascii_case("false")
            || trimmed.eq_ignore_ascii_case("no")
        {
            return DefensiveOpen4ProbeMode::Off;
        }
        if trimmed.eq_ignore_ascii_case("trace") || trimmed.eq_ignore_ascii_case("log") {
            DefensiveOpen4ProbeMode::Trace
        } else {
            DefensiveOpen4ProbeMode::Demote
        }
    })
}

fn defensive_open4_probe_depth() -> u32 {
    static DEPTH: OnceLock<u32> = OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("NORU_DEFENSIVE_OPEN4_PROBE_DEPTH")
            .ok()
            .and_then(|raw| raw.trim().parse::<u32>().ok())
            .unwrap_or(2)
            .clamp(1, 2)
    })
}

fn defensive_open4_probe_max_ply() -> usize {
    static MAX_PLY: OnceLock<usize> = OnceLock::new();
    *MAX_PLY.get_or_init(|| {
        std::env::var("NORU_DEFENSIVE_OPEN4_PROBE_MAX_PLY")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(0)
    })
}

fn defensive_open4_probe_penalty() -> i32 {
    static PENALTY: OnceLock<i32> = OnceLock::new();
    *PENALTY.get_or_init(|| {
        std::env::var("NORU_DEFENSIVE_OPEN4_PROBE_PENALTY")
            .ok()
            .and_then(|raw| raw.trim().parse::<i32>().ok())
            .filter(|v| *v >= 0)
            .unwrap_or(90_000)
    })
}

fn defensive_open4_probe_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NORU_DEFENSIVE_OPEN4_PROBE_TRACE")
            .map(|raw| {
                let trimmed = raw.trim();
                !(trimmed.is_empty()
                    || trimmed == "0"
                    || trimmed.eq_ignore_ascii_case("off")
                    || trimmed.eq_ignore_ascii_case("false")
                    || trimmed.eq_ignore_ascii_case("no"))
            })
            .unwrap_or(false)
    })
}

#[inline]
fn defensive_open4_probe_enabled_for_ply(ply: usize) -> bool {
    defensive_open4_probe_mode() != DefensiveOpen4ProbeMode::Off
        && ply <= defensive_open4_probe_max_ply()
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: u32,
    pub nodes: u64,
}

#[derive(Debug, Clone)]
pub struct RootCandidateAudit {
    pub mv: Move,
    pub search_score: i32,
    pub relation_score: Option<i32>,
    pub candidate_rank_score: Option<i32>,
    pub codebook_score: Option<i32>,
    pub is_forcing: bool,
}

#[derive(Debug, Clone)]
pub struct RootSearchAudit {
    pub result: SearchResult,
    pub candidates: Vec<RootCandidateAudit>,
}

fn final_root_candidate_tiebreak(
    board: &Board,
    candidates: &[RootCandidateAudit],
    incumbent: Option<Move>,
    leader_score: i32,
) -> Option<Move> {
    let mut best_move = incumbent?;
    if is_win_score(leader_score) {
        return Some(best_move);
    }
    let candidate_rank_margin = crate::candidate_ranker::root_margin();
    let candidate_rank_gate_mode = crate::candidate_ranker::root_gate_mode();
    let mut best_candidate_rank_score = candidates
        .iter()
        .find(|candidate| candidate.mv == best_move)
        .and_then(|candidate| candidate.candidate_rank_score);
    let mut best_candidate_rank_gate = crate::candidate_ranker::root_gate_key(board, best_move);

    for candidate in candidates {
        if candidate.mv == best_move || is_win_score(candidate.search_score) {
            continue;
        }
        let within_margin = if candidate_rank_margin == 0 {
            candidate.search_score == leader_score
        } else {
            leader_score.saturating_sub(candidate.search_score) <= candidate_rank_margin
        };
        if !within_margin {
            continue;
        }
        let candidate_rank_gate = crate::candidate_ranker::root_gate_key(board, candidate.mv);
        if candidate_ranker_root_rescue_only_enabled()
            && !candidate_ranker_root_rescue_allows(board, candidate.mv, best_move)
        {
            continue;
        }
        if !crate::candidate_ranker::gate_allows(
            candidate_rank_gate_mode,
            candidate_rank_gate,
            best_candidate_rank_gate,
        ) {
            continue;
        }
        if crate::candidate_ranker::score_prefers(
            candidate.candidate_rank_score,
            best_candidate_rank_score,
        ) {
            best_move = candidate.mv;
            best_candidate_rank_score = candidate.candidate_rank_score;
            best_candidate_rank_gate = candidate_rank_gate;
        }
    }

    Some(best_move)
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

#[inline]
fn relation_score_prefers(candidate: Option<i32>, incumbent: Option<i32>) -> bool {
    match (candidate, incumbent) {
        (Some(candidate), Some(incumbent)) => candidate > incumbent,
        (Some(_), None) => true,
        _ => false,
    }
}

trait SearchEvalState {
    fn push_move(&mut self, board: &Board, mv: Move);
    fn pop_move(&mut self);
    fn eval(&self, board: &Board) -> i32;
    fn eval_base(&self, board: &Board) -> i32 {
        self.eval(board)
    }
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
    fn push_move(&mut self, board: &Board, mv: Move) {
        self.inc.push_move(board, mv, self.weights);
    }

    fn pop_move(&mut self) {
        self.inc.pop_move();
    }

    fn eval(&self, board: &Board) -> i32 {
        self.inc.eval(self.weights, board)
    }

    fn eval_base(&self, board: &Board) -> i32 {
        self.inc.eval_base(self.weights, board)
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
    fn push_move(&mut self, board: &Board, mv: Move) {
        self.inc.push_move(board, mv, self.weights);
    }

    fn pop_move(&mut self) {
        self.inc.pop_move(self.weights);
    }

    fn eval(&self, board: &Board) -> i32 {
        self.scaled_value(board)
    }
}

fn probe_root_move_with(
    searcher: &mut Searcher,
    board: &Board,
    weights: &NnueWeights,
    mv: Move,
    depth: u32,
) -> i32 {
    let mut child = board.clone();
    child.make_move(mv);
    let mut inc = FlatEvalState::new(&child, weights);
    -searcher.alpha_beta(
        &mut child,
        weights,
        &mut inc,
        depth.saturating_sub(1),
        1,
        -INF,
        INF,
    )
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct RootRelationGate {
    attack: ThreatKind,
    block: ThreatKind,
}

#[inline]
fn root_relation_gate_key(board: &Board, mv: Move) -> Option<RootRelationGate> {
    let attack = classify_move_fast(board, mv, board.side_to_move);
    let block = classify_move_fast(board, mv, board.side_to_move.opponent());
    if attack == ThreatKind::None && block == ThreatKind::None {
        None
    } else {
        Some(RootRelationGate { attack, block })
    }
}

#[inline]
fn root_relation_strict_gate() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NORU_RELATION_LITE_ROOT_GATE")
            .map(|raw| {
                let trimmed = raw.trim();
                trimmed.eq_ignore_ascii_case("strict")
                    || trimmed.eq_ignore_ascii_case("tactical")
                    || trimmed.eq_ignore_ascii_case("same-threat")
                    || trimmed.eq_ignore_ascii_case("same_threat")
            })
            .unwrap_or(false)
    })
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
    node_limit_hit: bool,
    /// ???????遺얘턁????????ㅼ뒧??띤겫??눫????癲됱빖???嶺????????? ???????ル???? ?????????⑤슢堉??곕?????????????????????뀀맩鍮???????
    /// search ???遺얘턁??????????????????⑤벡瑜??????耀붾굝????????iterative deepening ???????롮쾸?椰???iteration?????    /// ?????????대첐??iteration??PV/cutoff ???遺얘턁?????????????????ル탛????????耀붾굝???????????
    tt: TranspositionTable,
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
            node_limit_hit: false,
            tt: TranspositionTable::new(TT_BUCKET_BITS),
        }
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

    fn reset_for_search(&mut self, time_limit: Option<Duration>) {
        self.nodes = 0;
        self.tt_cutoffs = 0;
        self.aborted = false;
        self.node_limit = None;
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
    }

    fn try_root_vct(
        &mut self,
        board: &mut Board,
        time_limit: Option<Duration>,
        root_search_decision_audit: bool,
    ) -> Option<SearchResult> {
        if !root_vct_enabled() {
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
        };
        if let Some(seq) = search_vct(board, &vct_cfg) {
            if let Some(&first) = seq.first() {
                let result = SearchResult {
                    best_move: Some(first),
                    score: WIN_SCORE,
                    depth: seq.len() as u32,
                    nodes: self.nodes,
                };
                if root_search_decision_audit {
                    crate::candidate_local_ensemble::append_root_search_decision_audit(
                        board,
                        result.best_move,
                        result.best_move,
                        result.score,
                        result.depth,
                        result.nodes,
                        self.aborted,
                        &[],
                    );
                }
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
        root_search_decision_audit: bool,
    ) -> SearchResult {
        let mut best_result = SearchResult {
            best_move: None,
            score: 0,
            depth: 0,
            nodes: 0,
        };
        let mut prev_best: Option<Move> = None;
        let mut prev_score: Option<i32> = None;
        let final_only_candidate_ranker = candidate_ranker_root_final_only_enabled()
            || crate::candidate_local_ensemble::root_tiebreak_enabled_for(board);
        let collect_root_candidates = final_only_candidate_ranker || root_search_decision_audit;
        let allow_root_iteration_tiebreak =
            !final_only_candidate_ranker && !candidate_ranker_root_final_only_enabled();
        let mut final_candidates = Vec::new();

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
                let current = if collect_root_candidates {
                    final_candidates.clear();
                    self.root_pvs_iteration_with_audit(
                        board,
                        weights,
                        inc,
                        depth,
                        alpha,
                        beta,
                        prev_best,
                        Some(&mut final_candidates),
                        allow_root_iteration_tiebreak,
                    )
                } else {
                    self.root_pvs_iteration(board, weights, inc, depth, alpha, beta, prev_best)
                };
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
                        break if collect_root_candidates {
                            final_candidates.clear();
                            self.root_pvs_iteration_with_audit(
                                board,
                                weights,
                                inc,
                                depth,
                                -INF,
                                INF,
                                prev_best,
                                Some(&mut final_candidates),
                                allow_root_iteration_tiebreak,
                            )
                        } else {
                            self.root_pvs_iteration(
                                board, weights, inc, depth, -INF, INF, prev_best,
                            )
                        };
                    }
                } else if score >= beta {
                    delta = (delta * 2).min(INF / 4);
                    beta = (beta + delta).min(INF);
                    if beta == INF {
                        break if collect_root_candidates {
                            final_candidates.clear();
                            self.root_pvs_iteration_with_audit(
                                board,
                                weights,
                                inc,
                                depth,
                                -INF,
                                INF,
                                prev_best,
                                Some(&mut final_candidates),
                                allow_root_iteration_tiebreak,
                            )
                        } else {
                            self.root_pvs_iteration(
                                board, weights, inc, depth, -INF, INF, prev_best,
                            )
                        };
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

        let search_best_move = best_result.best_move;
        if final_only_candidate_ranker {
            if let Some(best_move) = crate::candidate_local_ensemble::final_root_tiebreak(
                board,
                weights,
                &final_candidates,
                best_result.best_move,
                best_result.score,
            ) {
                best_result.best_move = Some(best_move);
            } else if let Some(best_move) = final_root_candidate_tiebreak(
                board,
                &final_candidates,
                best_result.best_move,
                best_result.score,
            ) {
                best_result.best_move = Some(best_move);
            }
        }
        if root_search_decision_audit {
            crate::candidate_local_ensemble::append_root_search_decision_audit(
                board,
                search_best_move,
                best_result.best_move,
                best_result.score,
                best_result.depth,
                best_result.nodes,
                self.aborted,
                &final_candidates,
            );
        }

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
        self.nodes = 0;
        self.tt_cutoffs = 0;
        self.aborted = false;
        self.node_limit = None;
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
        let root_search_decision_audit =
            crate::candidate_local_ensemble::root_search_decision_audit_enabled();

        // Root VCT: ????븐뼐?????????? ????????????????????????ル???????????諛몃마嶺뚮?????꾩렯???????耀붾굝????????????븐뼐?????????????????????????븐뼐?곭춯?竊???????.
        // Dynamic budget ???????????1/ROOT_VCT_BUDGET_FRACTION (cap/floor ?????????泥??.
        if root_vct_enabled() {
            let vct_budget = match time_limit {
                Some(d) => (d / ROOT_VCT_BUDGET_FRACTION)
                    .max(Duration::from_millis(ROOT_VCT_BUDGET_FLOOR_MS))
                    .min(Duration::from_millis(ROOT_VCT_BUDGET_CAP_MS)),
                None => Duration::from_millis(ROOT_VCT_BUDGET_MS),
            };
            let vct_cfg = VctConfig {
                max_depth: ROOT_VCT_DEPTH,
                time_budget: Some(vct_budget),
            };
            if let Some(seq) = search_vct(board, &vct_cfg) {
                if let Some(&first) = seq.first() {
                    let result = SearchResult {
                        best_move: Some(first),
                        score: WIN_SCORE,
                        depth: seq.len() as u32,
                        nodes: self.nodes,
                    };
                    if root_search_decision_audit {
                        crate::candidate_local_ensemble::append_root_search_decision_audit(
                            board,
                            result.best_move,
                            result.best_move,
                            result.score,
                            result.depth,
                            result.nodes,
                            self.aborted,
                            &[],
                        );
                    }
                    return result;
                }
            }
        }

        let mut best_result = SearchResult {
            best_move: None,
            score: 0,
            depth: 0,
            nodes: 0,
        };

        // Incremental NNUE state ??????????耀붾굝?????傭?끆????椰???????full refresh, ???????꾩룆梨띰쭕??        // make_move/undo_move?? ?????????딅즹???push/pop????????源낅?????leaf?????full
        // compute_active_features????? ??????????Accumulator forward??????????
        let mut inc = FlatEvalState::new(board, weights);

        // PV-move priority: the best move from iteration depth-1 becomes the
        // first move we try at iteration depth. Combined with PVS + Aspiration,
        // this drastically reduces re-search cost.
        let mut prev_best: Option<Move> = None;
        let mut prev_score: Option<i32> = None;
        let final_only_candidate_ranker = candidate_ranker_root_final_only_enabled()
            || crate::candidate_local_ensemble::root_tiebreak_enabled_for(board);
        let collect_root_candidates = final_only_candidate_ranker || root_search_decision_audit;
        let allow_root_iteration_tiebreak =
            !final_only_candidate_ranker && !candidate_ranker_root_final_only_enabled();
        let mut final_candidates = Vec::new();

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
                let current = if collect_root_candidates {
                    final_candidates.clear();
                    self.root_pvs_iteration_with_audit(
                        board,
                        weights,
                        &mut inc,
                        depth,
                        alpha,
                        beta,
                        prev_best,
                        Some(&mut final_candidates),
                        allow_root_iteration_tiebreak,
                    )
                } else {
                    self.root_pvs_iteration(board, weights, &mut inc, depth, alpha, beta, prev_best)
                };
                if self.aborted {
                    break current;
                }
                let score = current.1;
                if !aspirate {
                    break current;
                }
                if score <= alpha {
                    // fail-low: alpha ??????                    delta = (delta * 2).min(INF / 4);
                    alpha = (alpha - delta).max(-INF);
                    if alpha == -INF {
                        // ?????????耀붾굝????????????full window??break???????
                        // ????癲됱빖???嶺????????????롮쾸?椰???iteration????????? ?????break ??                        // ????????袁④뎬??-INF/INF??
                        // re-search with full window once
                        break if collect_root_candidates {
                            final_candidates.clear();
                            self.root_pvs_iteration_with_audit(
                                board,
                                weights,
                                &mut inc,
                                depth,
                                -INF,
                                INF,
                                prev_best,
                                Some(&mut final_candidates),
                                allow_root_iteration_tiebreak,
                            )
                        } else {
                            self.root_pvs_iteration(
                                board, weights, &mut inc, depth, -INF, INF, prev_best,
                            )
                        };
                    }
                } else if score >= beta {
                    delta = (delta * 2).min(INF / 4);
                    beta = (beta + delta).min(INF);
                    if beta == INF {
                        break if collect_root_candidates {
                            final_candidates.clear();
                            self.root_pvs_iteration_with_audit(
                                board,
                                weights,
                                &mut inc,
                                depth,
                                -INF,
                                INF,
                                prev_best,
                                Some(&mut final_candidates),
                                allow_root_iteration_tiebreak,
                            )
                        } else {
                            self.root_pvs_iteration(
                                board, weights, &mut inc, depth, -INF, INF, prev_best,
                            )
                        };
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

        let search_best_move = best_result.best_move;
        if final_only_candidate_ranker {
            if let Some(best_move) = crate::candidate_local_ensemble::final_root_tiebreak(
                board,
                weights,
                &final_candidates,
                best_result.best_move,
                best_result.score,
            ) {
                best_result.best_move = Some(best_move);
            } else if let Some(best_move) = final_root_candidate_tiebreak(
                board,
                &final_candidates,
                best_result.best_move,
                best_result.score,
            ) {
                best_result.best_move = Some(best_move);
            }
        }
        if root_search_decision_audit {
            crate::candidate_local_ensemble::append_root_search_decision_audit(
                board,
                search_best_move,
                best_result.best_move,
                best_result.score,
                best_result.depth,
                best_result.nodes,
                self.aborted,
                &final_candidates,
            );
        }

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
        self.reset_for_search(time_limit);
        let root_search_decision_audit =
            crate::candidate_local_ensemble::root_search_decision_audit_enabled();
        if let Some(result) = self.try_root_vct(board, time_limit, root_search_decision_audit) {
            return result;
        }
        let scale = codebook_eval_scale();
        let mut inc = CodebookEvalState::new(board, codebook_weights, scale);
        self.search_with_eval_state(
            board,
            ordering_weights,
            &mut inc,
            max_depth,
            root_search_decision_audit,
        )
    }

    /// Search normally, but keep the root candidate table from the last
    /// completed iterative-deepening iteration. This is for offline engine
    /// diagnostics; the normal pbrain path uses `search()`.
    pub fn audit_root_candidates(
        &mut self,
        board: &mut Board,
        weights: &NnueWeights,
        max_depth: u32,
        time_limit: Option<Duration>,
    ) -> RootSearchAudit {
        self.nodes = 0;
        self.tt_cutoffs = 0;
        self.aborted = false;
        self.deadline = time_limit.map(|d| Instant::now() + d);
        self.killers = [[None; 2]; 64];
        self.history = [[0; NUM_CELLS]; 2];
        self.cont_hist_1.fill([0; NUM_CELLS]);
        self.cont_hist_2.fill([0; NUM_CELLS]);
        self.tt.reset_stats();
        self.tt.clear();

        let mut inc = FlatEvalState::new(board, weights);

        if root_vct_enabled() {
            let vct_budget = match time_limit {
                Some(total) => (total / ROOT_VCT_BUDGET_FRACTION)
                    .max(Duration::from_millis(ROOT_VCT_BUDGET_FLOOR_MS))
                    .min(Duration::from_millis(ROOT_VCT_BUDGET_CAP_MS)),
                None => Duration::from_millis(ROOT_VCT_BUDGET_MS),
            };
            let vct_cfg = VctConfig {
                max_depth: ROOT_VCT_DEPTH,
                time_budget: Some(vct_budget),
            };
            if let Some(seq) = search_vct(board, &vct_cfg) {
                if let Some(&first) = seq.first() {
                    return RootSearchAudit {
                        result: SearchResult {
                            best_move: Some(first),
                            score: WIN_SCORE,
                            depth: seq.len() as u32,
                            nodes: self.nodes,
                        },
                        candidates: Vec::new(),
                    };
                }
            }
        }

        let mut best_result = SearchResult {
            best_move: None,
            score: 0,
            depth: 0,
            nodes: 0,
        };
        let mut best_candidates = Vec::new();
        let mut prev_best: Option<Move> = None;
        let mut prev_score: Option<i32> = None;

        for depth in 1..=max_depth {
            let (mut alpha, mut beta) = if let Some(s) = prev_score {
                if depth >= ASPIRATION_MIN_DEPTH {
                    (s - ASPIRATION_INITIAL_DELTA, s + ASPIRATION_INITIAL_DELTA)
                } else {
                    (-INF, INF)
                }
            } else {
                (-INF, INF)
            };

            let mut delta = ASPIRATION_INITIAL_DELTA;
            let mut iter_candidates = Vec::new();

            let iter_result = loop {
                iter_candidates.clear();
                let current = self.root_pvs_iteration_with_audit(
                    board,
                    weights,
                    &mut inc,
                    depth,
                    alpha,
                    beta,
                    prev_best,
                    Some(&mut iter_candidates),
                    !candidate_ranker_root_final_only_enabled(),
                );
                if self.aborted {
                    break current;
                }
                let score = current.1;
                if score <= alpha {
                    delta = (delta * 2).min(INF / 4);
                    alpha = (alpha - delta).max(-INF);
                    if alpha == -INF {
                        iter_candidates.clear();
                        break self.root_pvs_iteration_with_audit(
                            board,
                            weights,
                            &mut inc,
                            depth,
                            -INF,
                            INF,
                            prev_best,
                            Some(&mut iter_candidates),
                            !candidate_ranker_root_final_only_enabled(),
                        );
                    }
                } else if score >= beta {
                    delta = (delta * 2).min(INF / 4);
                    beta = (beta + delta).min(INF);
                    if beta == INF {
                        iter_candidates.clear();
                        break self.root_pvs_iteration_with_audit(
                            board,
                            weights,
                            &mut inc,
                            depth,
                            -INF,
                            INF,
                            prev_best,
                            Some(&mut iter_candidates),
                            !candidate_ranker_root_final_only_enabled(),
                        );
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
            best_candidates = iter_candidates;

            if score.abs() > WIN_SCORE - 100 {
                break;
            }

            prev_best = best_move;
            prev_score = Some(score);
        }

        RootSearchAudit {
            result: best_result,
            candidates: best_candidates,
        }
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
        self.root_pvs_iteration_with_audit(
            board,
            weights,
            inc,
            depth,
            alpha_init,
            beta_init,
            prev_best,
            None,
            !candidate_ranker_root_final_only_enabled(),
        )
    }

    fn root_pvs_iteration_with_audit<E: SearchEvalState>(
        &mut self,
        board: &mut Board,
        weights: &NnueWeights,
        inc: &mut E,
        depth: u32,
        alpha_init: i32,
        beta_init: i32,
        prev_best: Option<Move>,
        mut audit: Option<&mut Vec<RootCandidateAudit>>,
        allow_candidate_rank_tiebreak: bool,
    ) -> (Option<Move>, i32) {
        let mut alpha = alpha_init;
        let beta = beta_init;
        let mut best_move: Option<Move> = None;
        let mut leader_score = -INF;
        let mut best_relation_score: Option<i32> = None;
        let mut best_relation_gate: Option<RootRelationGate> = None;
        let mut best_candidate_rank_score: Option<i32> = None;
        let mut best_candidate_rank_gate: Option<crate::candidate_ranker::RootGateKey> = None;
        let mut best_codebook_score: Option<i32> = None;
        let mut best_codebook_gate: Option<crate::candidate_ranker::RootGateKey> = None;
        let use_root_relation = crate::relation_lite::root_enabled()
            && (board.move_count as usize) >= crate::relation_lite::root_min_ply();
        let strict_root_relation = use_root_relation && root_relation_strict_gate();
        let root_relation_margin = if use_root_relation {
            crate::relation_lite::root_margin()
        } else {
            0
        };
        let candidate_rank_min_stones_ok =
            (board.move_count as usize) >= candidate_ranker_root_min_stones();
        let score_candidate_ranker = candidate_rank_min_stones_ok
            && crate::candidate_ranker::root_score_enabled_for(board)
            && candidate_ranker_root_tiebreak_enabled();
        let use_candidate_ranker = score_candidate_ranker && allow_candidate_rank_tiebreak;
        let candidate_rank_margin = if use_candidate_ranker {
            crate::candidate_ranker::root_margin()
        } else {
            0
        };
        let candidate_rank_score_margin = if use_candidate_ranker {
            crate::candidate_ranker::root_score_margin()
        } else {
            0
        };
        let candidate_rank_gate_mode = crate::candidate_ranker::root_gate_mode();
        let use_codebook_tiebreak = crate::codebook_sidecar::root_tiebreak_enabled_for(board)
            && !use_candidate_ranker
            && !use_root_relation;
        let use_codebook_audit = crate::codebook_sidecar::root_audit_enabled_for(board);
        let use_codebook_final_tiebreak =
            crate::codebook_sidecar::root_final_tiebreak_enabled_for(board);
        let codebook_margin = if use_codebook_tiebreak {
            crate::codebook_sidecar::root_margin()
        } else {
            0
        };
        let codebook_gate_mode = crate::codebook_sidecar::root_gate_mode();

        let mut moves = self.order_moves(board, 0, weights);
        let relation_fusion_order = if crate::relation_fusion_gate::enabled_for(board) {
            Some(moves.clone())
        } else {
            None
        };
        if let Some(pv) = prev_best {
            if let Some(pos) = moves.iter().position(|&(m, _)| m == pv) {
                if pos != 0 {
                    moves.swap(0, pos);
                }
            }
        }
        self.apply_candidate_local_ab_probe_order(board, weights, &mut moves, prev_best, depth);
        if self.aborted {
            return (best_move, alpha);
        }
        let mut root_codebook_scores: Vec<Option<i32>> = Vec::new();
        let mut root_codebook_best_score: Option<i32> = None;
        if use_codebook_tiebreak && crate::codebook_sidecar::root_require_global_best() {
            root_codebook_scores = moves
                .iter()
                .map(|&(mv, _)| crate::codebook_sidecar::root_candidate_score(board, mv))
                .collect();
            for &score in &root_codebook_scores {
                if crate::candidate_ranker::score_prefers(score, root_codebook_best_score) {
                    root_codebook_best_score = score;
                }
            }
        }

        for (move_idx, &(mv, is_forcing)) in moves.iter().enumerate() {
            let root_relation_gate = if strict_root_relation {
                crate::relation_lite::root_move_allowed(mv)
                    .then(|| root_relation_gate_key(board, mv))
                    .flatten()
            } else {
                None
            };

            // TT prefetch ??same trick as in alpha_beta: warm the child's
            // TT bucket while make_move + accumulator delta runs.
            let next_zob = board.zobrist
                ^ crate::board::zobrist_stone_key(board.side_to_move, mv)
                ^ crate::board::ZOBRIST_SIDE;
            self.tt.prefetch(next_zob);

            board.make_move(mv);
            inc.push_move(board, mv);

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

            let root_relation_score = if use_root_relation
                && crate::relation_lite::root_move_allowed(mv)
                && !self.aborted
                && !is_win_score(score)
            {
                let child_base = inc.eval_base(board);
                crate::relation_lite::root_candidate_eval(board, child_base).map(|child| -child)
            } else {
                None
            };

            inc.pop_move();
            board.undo_move();

            let candidate_rank_gate = if use_candidate_ranker {
                crate::candidate_ranker::root_gate_key(board, mv)
            } else {
                None
            };
            let candidate_rank_score =
                if use_candidate_ranker && !self.aborted && !is_win_score(score) {
                    crate::candidate_ranker::root_candidate_score(board, mv, weights)
                } else {
                    None
                };
            let codebook_gate = if use_codebook_tiebreak {
                crate::candidate_ranker::root_gate_key(board, mv)
            } else {
                None
            };
            let codebook_score =
                if (use_codebook_tiebreak || use_codebook_audit || use_codebook_final_tiebreak)
                    && !self.aborted
                    && !is_win_score(score)
                {
                    root_codebook_scores
                        .get(move_idx)
                        .copied()
                        .flatten()
                        .or_else(|| crate::codebook_sidecar::root_candidate_score(board, mv))
                } else {
                    None
                };

            if self.aborted {
                break;
            }

            if let Some(audit) = audit.as_mut() {
                audit.push(RootCandidateAudit {
                    mv,
                    search_score: score,
                    relation_score: root_relation_score,
                    candidate_rank_score,
                    codebook_score,
                    is_forcing,
                });
            }

            if score > leader_score {
                leader_score = score;
                best_move = Some(mv);
                best_relation_score = root_relation_score;
                best_relation_gate = root_relation_gate;
                best_candidate_rank_score = candidate_rank_score;
                best_candidate_rank_gate = candidate_rank_gate;
                best_codebook_score = codebook_score;
                best_codebook_gate = codebook_gate;
            } else if use_candidate_ranker
                && !is_win_score(score)
                && !is_win_score(leader_score)
                && (if candidate_rank_margin == 0 {
                    score == leader_score
                } else {
                    leader_score.saturating_sub(score) <= candidate_rank_margin
                })
                && crate::candidate_ranker::gate_allows(
                    candidate_rank_gate_mode,
                    candidate_rank_gate,
                    best_candidate_rank_gate,
                )
                && (!candidate_ranker_root_rescue_only_enabled()
                    || best_move
                        .map(|incumbent| candidate_ranker_root_rescue_allows(board, mv, incumbent))
                        .unwrap_or(false))
                && crate::candidate_ranker::score_prefers_with_margin(
                    candidate_rank_score,
                    best_candidate_rank_score,
                    candidate_rank_score_margin,
                )
            {
                best_move = Some(mv);
                best_relation_score = root_relation_score;
                best_relation_gate = root_relation_gate;
                best_candidate_rank_score = candidate_rank_score;
                best_candidate_rank_gate = candidate_rank_gate;
                best_codebook_score = codebook_score;
                best_codebook_gate = codebook_gate;
            } else if use_codebook_tiebreak
                && !is_win_score(score)
                && !is_win_score(leader_score)
                && (if codebook_margin == 0 {
                    score == leader_score
                } else {
                    leader_score.saturating_sub(score) <= codebook_margin
                })
                && crate::candidate_ranker::gate_allows(
                    codebook_gate_mode,
                    codebook_gate,
                    best_codebook_gate,
                )
                && crate::codebook_sidecar::root_global_best_allows(
                    codebook_score,
                    root_codebook_best_score,
                )
                && crate::candidate_ranker::score_prefers(codebook_score, best_codebook_score)
            {
                best_move = Some(mv);
                best_relation_score = root_relation_score;
                best_relation_gate = root_relation_gate;
                best_candidate_rank_score = candidate_rank_score;
                best_candidate_rank_gate = candidate_rank_gate;
                best_codebook_score = codebook_score;
                best_codebook_gate = codebook_gate;
            } else if use_root_relation
                && !is_win_score(score)
                && !is_win_score(leader_score)
                && (if strict_root_relation {
                    score == leader_score
                        && root_relation_gate.is_some()
                        && root_relation_gate == best_relation_gate
                } else if root_relation_margin == 0 {
                    score == leader_score
                } else {
                    leader_score.saturating_sub(score) <= root_relation_margin
                })
                && relation_score_prefers(root_relation_score, best_relation_score)
            {
                best_move = Some(mv);
                best_relation_score = root_relation_score;
                best_relation_gate = root_relation_gate;
                best_candidate_rank_score = candidate_rank_score;
                best_candidate_rank_gate = candidate_rank_gate;
                best_codebook_score = codebook_score;
                best_codebook_gate = codebook_gate;
            }

            if score > alpha {
                alpha = score;
            }
        }

        if let Some(order) = relation_fusion_order.as_deref() {
            if let Some(replacement) =
                crate::relation_fusion_gate::choose_replacement(board, order, best_move)
            {
                best_move = Some(replacement);
            }
        }

        (best_move, alpha)
    }

    fn apply_candidate_local_ab_probe_order(
        &mut self,
        board: &Board,
        weights: &NnueWeights,
        moves: &mut Vec<(Move, bool)>,
        prev_best: Option<Move>,
        root_depth: u32,
    ) {
        if !candidate_local_ab_probe_enabled()
            || moves.len() < 2
            || board.move_count < candidate_local_ab_probe_min_ply()
            || root_depth < candidate_local_ab_probe_min_depth()
        {
            return;
        }

        let Some(scores) =
            crate::candidate_local_ensemble::root_candidate_score_map(board, weights)
        else {
            return;
        };
        if scores.is_empty() {
            return;
        }

        let topk = candidate_local_ab_probe_topk();
        let split = if topk == 0 {
            moves.len()
        } else {
            moves.len().min(topk)
        };
        let start = match prev_best {
            Some(pv) if moves.first().map(|&(mv, _)| mv) == Some(pv) => 1,
            _ => 0,
        };
        if start >= split {
            return;
        }

        let mut local_ranked = (start..split)
            .filter_map(|idx| scores.get(&moves[idx].0).copied().map(|score| (idx, score)))
            .collect::<Vec<_>>();
        if local_ranked.is_empty() {
            return;
        }
        local_ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let incumbent_local_score = scores.get(&moves[start].0).copied().unwrap_or(i32::MIN);
        let min_local_delta = candidate_local_ab_probe_min_local_delta();
        let mut probe_indices = vec![start];
        let max_relation_candidates = candidate_local_ab_probe_candidates();
        for (idx, local_score) in local_ranked {
            if idx == start {
                continue;
            }
            if local_score < incumbent_local_score.saturating_add(min_local_delta) {
                continue;
            }
            probe_indices.push(idx);
            if probe_indices.len() > max_relation_candidates {
                break;
            }
        }
        probe_indices.sort_unstable();
        probe_indices.dedup();
        if probe_indices.len() < 2 {
            return;
        }

        let depth = candidate_local_ab_probe_depth();
        let before = moves[start..split]
            .iter()
            .map(|&(mv, _)| mv)
            .collect::<Vec<_>>();
        let mut probe_searcher = Searcher::new();
        probe_searcher.deadline = self.deadline;
        probe_searcher.node_limit = candidate_local_ab_probe_node_limit();
        let mut probe_incomplete = false;
        let mut probed = Vec::with_capacity(probe_indices.len());
        for &idx in &probe_indices {
            let mv = moves[idx].0;
            let score = probe_root_move_with(&mut probe_searcher, board, weights, mv, depth);
            if probe_searcher.aborted {
                if probe_searcher.node_limit_hit {
                    probe_incomplete = true;
                } else {
                    self.aborted = true;
                }
                break;
            }
            probed.push((idx, mv, score));
        }
        self.nodes = self.nodes.saturating_add(probe_searcher.nodes);
        self.tt_cutoffs = self.tt_cutoffs.saturating_add(probe_searcher.tt_cutoffs);
        if self.aborted || probe_incomplete {
            return;
        }

        let Some((_, incumbent_mv, incumbent_score)) =
            probed.iter().find(|(idx, _, _)| *idx == start).copied()
        else {
            return;
        };
        let mut best = (start, incumbent_mv, incumbent_score);
        for &(idx, mv, score) in &probed {
            let local_score = scores.get(&mv).copied().unwrap_or(i32::MIN);
            let best_local_score = scores.get(&best.1).copied().unwrap_or(i32::MIN);
            if score > best.2 || (score == best.2 && local_score > best_local_score) {
                best = (idx, mv, score);
            }
        }

        let margin = candidate_local_ab_probe_margin();
        let beats_incumbent = if margin == 0 {
            best.2 > incumbent_score
        } else {
            best.2 >= incumbent_score.saturating_add(margin)
        };
        if beats_incumbent && best.0 != start {
            let item = moves.remove(best.0);
            moves.insert(start, item);
        }

        if crate::candidate_local_ensemble::root_order_audit_enabled() {
            let after = moves[start..split]
                .iter()
                .map(|&(mv, _)| mv)
                .collect::<Vec<_>>();
            let probe_scores = probed
                .into_iter()
                .map(|(_, mv, score)| (mv, score))
                .collect::<Vec<_>>();
            crate::candidate_local_ensemble::append_root_ab_probe_audit(
                board,
                start,
                split,
                depth,
                &before,
                &after,
                &scores,
                &probe_scores,
            );
        }
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
        self.nodes += 1;
        if self.check_node_limit() {
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
        let tt_hit = self.tt.probe(board.zobrist);
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

        let mut moves = self.order_moves(board, ply, weights);
        if moves.is_empty() {
            return 0;
        }

        // TT-best move???????????(PVS????????????ル???????cutoff ??????????????.
        if let Some(tt_mv) = tt_move {
            if let Some(pos) = moves.iter().position(|&(m, _)| m == tt_mv) {
                if pos != 0 {
                    moves.swap(0, pos);
                }
            }
        }

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
        for (move_idx, &(mv, is_forcing)) in moves.iter().enumerate() {
            let is_killer =
                ply < 64 && (self.killers[ply][0] == Some(mv) || self.killers[ply][1] == Some(mv));

            // === LMP (Late Move Pruning) ===
            // ??PV / ??forcing / ??killer / ??? depth?????move_idx???????ル???
            // ?????????????????????????????quiet move skip. count-based??eval ??????????已???            // ????轅붽틓???壤굿??덊뒌??, ????trigger ??????⑤벡瑜???? tier ???遺얘턁???????????????????quiet move??????????
            // ??????袁⑸즴筌?씛彛??????????ル???????????????븐뼐????????????
            if !is_pv
                && !is_forcing
                && !is_killer
                && depth >= LMP_MIN_DEPTH
                && depth <= LMP_MAX_DEPTH
            {
                let lmp_threshold = LMP_BASE + LMP_PER_DEPTH * depth as usize;
                if move_idx >= lmp_threshold {
                    continue;
                }
            }

            // TT prefetch: hint the CPU to load the child node's TT bucket
            // into L1 while we run the (cache-cold) make_move + accumulator
            // delta below. The child's first action is a TT probe, so by
            // the time it gets there the line is already warm. Worth ~5-10%
            // search throughput on cache-bound positions.
            let next_zob = board.zobrist
                ^ crate::board::zobrist_stone_key(board.side_to_move, mv)
                ^ crate::board::ZOBRIST_SIDE;
            self.tt.prefetch(next_zob);

            // Track quiet move ordering for continuation-history penalties on
            // a later cutoff (skipped for forcing moves ??those carry their
            // own tier signal and shouldn't get history bonuses).
            if !is_forcing {
                quiets_tried.push(mv);
            }

            board.make_move(mv);
            inc.push_move(board, mv);

            let score = if move_idx == 0 {
                -self.alpha_beta(board, weights, inc, depth - 1, ply + 1, -beta, -alpha)
            } else {
                let reduction = lmr_reduction(depth, move_idx, is_forcing, is_killer);
                let reduced_depth = (depth - 1).saturating_sub(reduction);
                let mut null_score = -self.alpha_beta(
                    board,
                    weights,
                    inc,
                    reduced_depth,
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                );
                if !self.aborted && reduction > 0 && null_score > alpha {
                    null_score = -self.alpha_beta(
                        board,
                        weights,
                        inc,
                        depth - 1,
                        ply + 1,
                        -alpha - 1,
                        -alpha,
                    );
                }
                if !self.aborted && null_score > alpha && null_score < beta {
                    -self.alpha_beta(board, weights, inc, depth - 1, ply + 1, -beta, -alpha)
                } else {
                    null_score
                }
            };

            inc.pop_move();
            board.undo_move();

            if self.aborted {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move_at_node = Some(mv);
            }
            if score > alpha {
                alpha = score;
                self.history[side][mv] += (depth * depth) as i32;
            }
            if alpha >= beta {
                if ply < 64 {
                    self.killers[ply][1] = self.killers[ply][0];
                    self.killers[ply][0] = Some(mv);
                }
                // Continuation-history bonus on beta-cutoff. Only quiet
                // (non-forcing) cutters earn history; forcing cutters are
                // already prioritized by their tier score in the move
                // ordering and don't need the table to remember them. Quiet
                // moves tried earlier in this node receive a symmetric
                // penalty so the next ordering pass demotes them.
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
                break;
            }
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
        // depth???????ル??? u8????????????????????ル?????saturate. ?????????녳븢??max_depth ??20??????????????????????대첉??
        self.tt.store(
            board.zobrist,
            best_score,
            depth.min(255) as u8,
            bound,
            best_move_at_node,
        );

        best_score
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
        self.nodes += 1;
        if self.check_node_limit() {
            return 0;
        }

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

        let stand_pat = inc.eval(board);
        if qply >= QSEARCH_MAX_PLY {
            return stand_pat;
        }
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let candidates = board.candidate_moves();
        if candidates.is_empty() {
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
            return stand_pat;
        }

        forcing.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        let mut best = stand_pat;
        for &(mv, _) in &forcing {
            board.make_move(mv);
            inc.push_move(board, mv);
            let score = -self.qsearch(board, weights, inc, qply + 1, ply + 1, -beta, -alpha);
            inc.pop_move();
            board.undo_move();

            if self.aborted {
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
    fn order_moves(&self, board: &Board, ply: usize, weights: &NnueWeights) -> Vec<(Move, bool)> {
        let candidates = board.candidate_moves();
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

        let topk = candidate_ranker_order_topk();
        if ply == 0
            && crate::candidate_ranker::root_score_enabled_for(board)
            && candidate_ranker_order_tiebreak_enabled()
            && !candidate_ranker_root_final_only_enabled()
            && !candidate_ranker_root_rescue_only_enabled()
        {
            let split = if topk == 0 {
                packed.len()
            } else {
                packed.len().min(topk)
            };
            let tie_margin = candidate_ranker_order_tie_margin();
            let order_gate_mode = crate::candidate_ranker::order_gate_mode();
            let mut start = 0;
            while start < split {
                let group_score = packed[start] >> 16;
                let mut end = start + 1;
                while end < split && group_score.saturating_sub(packed[end] >> 16) <= tie_margin {
                    end += 1;
                }
                if end - start > 1 {
                    packed[start..end].sort_unstable_by(|a, b| {
                        let a_mv = (*a & MV_MASK) as Move;
                        let b_mv = (*b & MV_MASK) as Move;
                        if !candidate_ranker_order_gate_allows_pair(
                            board,
                            a_mv,
                            b_mv,
                            order_gate_mode,
                        ) {
                            return b.cmp(a);
                        }
                        let a_score =
                            crate::candidate_ranker::root_candidate_score(board, a_mv, weights)
                                .unwrap_or(i32::MIN);
                        let b_score =
                            crate::candidate_ranker::root_candidate_score(board, b_mv, weights)
                                .unwrap_or(i32::MIN);
                        b_score.cmp(&a_score).then_with(|| b.cmp(a))
                    });
                }
                start = end;
            }
        }

        if ply == 0
            && crate::codebook_sidecar::root_tiebreak_enabled_for(board)
            && crate::codebook_sidecar::root_order_tiebreak_enabled()
            && !candidate_ranker_order_tiebreak_enabled()
            && !candidate_ranker_root_final_only_enabled()
            && !candidate_ranker_root_rescue_only_enabled()
        {
            let tie_margin = crate::codebook_sidecar::root_tie_margin();
            let order_gate_mode = crate::codebook_sidecar::root_gate_mode();
            let mut start = 0;
            while start < packed.len() {
                let group_score = packed[start] >> 16;
                let mut end = start + 1;
                while end < packed.len()
                    && group_score.saturating_sub(packed[end] >> 16) <= tie_margin
                {
                    end += 1;
                }
                if end - start > 1 {
                    packed[start..end].sort_unstable_by(|a, b| {
                        let a_mv = (*a & MV_MASK) as Move;
                        let b_mv = (*b & MV_MASK) as Move;
                        if !candidate_ranker_order_gate_allows_pair(
                            board,
                            a_mv,
                            b_mv,
                            order_gate_mode,
                        ) {
                            return b.cmp(a);
                        }
                        let a_score = crate::codebook_sidecar::root_candidate_score(board, a_mv)
                            .unwrap_or(i32::MIN);
                        let b_score = crate::codebook_sidecar::root_candidate_score(board, b_mv)
                            .unwrap_or(i32::MIN);
                        b_score.cmp(&a_score).then_with(|| b.cmp(a))
                    });
                }
                start = end;
            }
        }

        if ply == 0
            && crate::relation_fusion_gate::root_order_tiebreak_enabled_for(board)
            && !candidate_ranker_order_tiebreak_enabled()
            && !crate::codebook_sidecar::root_order_tiebreak_enabled()
            && !candidate_ranker_root_final_only_enabled()
            && !candidate_ranker_root_rescue_only_enabled()
        {
            let tie_margin = crate::relation_fusion_gate::root_order_tie_margin();
            let mut start = 0;
            while start < packed.len() {
                let group_score = packed[start] >> 16;
                let mut end = start + 1;
                while end < packed.len()
                    && group_score.saturating_sub(packed[end] >> 16) <= tie_margin
                {
                    end += 1;
                }
                if end - start > 1 {
                    packed[start..end].sort_unstable_by(|a, b| {
                        let a_mv = (*a & MV_MASK) as Move;
                        let b_mv = (*b & MV_MASK) as Move;
                        let a_score =
                            crate::relation_fusion_gate::root_candidate_score(board, a_mv)
                                .unwrap_or(i32::MIN);
                        let b_score =
                            crate::relation_fusion_gate::root_candidate_score(board, b_mv)
                                .unwrap_or(i32::MIN);
                        b_score.cmp(&a_score).then_with(|| b.cmp(a))
                    });
                }
                start = end;
            }
        }

        if ply == 0 {
            let candidate_local_order_enabled =
                crate::candidate_local_ensemble::root_order_tiebreak_enabled_for(board);
            let candidate_ranker_order_enabled = candidate_ranker_order_tiebreak_enabled();
            let codebook_order_enabled = crate::codebook_sidecar::root_order_tiebreak_enabled();
            let relation_fusion_order_enabled =
                crate::relation_fusion_gate::root_order_tiebreak_enabled_for(board);
            let candidate_ranker_final_only = candidate_ranker_root_final_only_enabled();
            let candidate_ranker_rescue_only = candidate_ranker_root_rescue_only_enabled();
            let active = candidate_local_order_enabled
                && !candidate_ranker_order_enabled
                && !codebook_order_enabled
                && !relation_fusion_order_enabled
                && !candidate_ranker_final_only
                && !candidate_ranker_rescue_only;
            let topk = crate::candidate_local_ensemble::root_order_topk();
            let split = if topk == 0 {
                packed.len()
            } else {
                packed.len().min(topk)
            };
            let tie_margin = crate::candidate_local_ensemble::root_order_tie_margin();
            let mut score_count = None;
            let mut group_count = 0usize;
            let mut changed_group_count = 0usize;
            if active {
                let order_gate_mode = crate::candidate_local_ensemble::root_order_gate_mode();
                let scores =
                    crate::candidate_local_ensemble::root_candidate_score_map(board, weights);
                score_count = scores.as_ref().map(|scores| scores.len());
                if let Some(scores) = scores {
                    let mut start = 0;
                    while start < split {
                        let group_score = packed[start] >> 16;
                        let mut end = start + 1;
                        while end < split
                            && group_score.saturating_sub(packed[end] >> 16) <= tie_margin
                        {
                            end += 1;
                        }
                        if end - start > 1 {
                            group_count += 1;
                            let before = packed[start..end]
                                .iter()
                                .map(|p| (*p & MV_MASK) as Move)
                                .collect::<Vec<_>>();
                            packed[start..end].sort_unstable_by(|a, b| {
                                let a_mv = (*a & MV_MASK) as Move;
                                let b_mv = (*b & MV_MASK) as Move;
                                if !candidate_ranker_order_gate_allows_pair(
                                    board,
                                    a_mv,
                                    b_mv,
                                    order_gate_mode,
                                ) {
                                    return b.cmp(a);
                                }
                                let a_score = scores.get(&a_mv).copied().unwrap_or(i32::MIN);
                                let b_score = scores.get(&b_mv).copied().unwrap_or(i32::MIN);
                                b_score.cmp(&a_score).then_with(|| b.cmp(a))
                            });
                            let after = packed[start..end]
                                .iter()
                                .map(|p| (*p & MV_MASK) as Move)
                                .collect::<Vec<_>>();
                            if before != after {
                                changed_group_count += 1;
                            }
                            crate::candidate_local_ensemble::append_root_order_audit(
                                board,
                                start,
                                group_score,
                                tie_margin,
                                &before,
                                &after,
                                &scores,
                            );
                        }
                        start = end;
                    }
                }
            }
            if crate::candidate_local_ensemble::root_order_audit_enabled() {
                crate::candidate_local_ensemble::append_root_order_attempt_audit(
                    board,
                    packed.len(),
                    split,
                    tie_margin,
                    candidate_local_order_enabled,
                    candidate_ranker_order_enabled,
                    codebook_order_enabled,
                    relation_fusion_order_enabled,
                    candidate_ranker_final_only,
                    candidate_ranker_rescue_only,
                    active,
                    score_count,
                    group_count,
                    changed_group_count,
                );
            }
        }

        let mut moves: Vec<(Move, bool)> = packed
            .into_iter()
            .map(|p| {
                let mv = (p & MV_MASK) as Move;
                let f = (p & FORCING_BIT) != 0;
                (mv, f)
            })
            .collect();

        let topk = candidate_ranker_order_topk();
        if ply == 0
            && topk > 0
            && crate::candidate_ranker::root_score_enabled_for(board)
            && !candidate_ranker_order_tiebreak_enabled()
            && !candidate_ranker_root_final_only_enabled()
            && !candidate_ranker_root_rescue_only_enabled()
        {
            let split = moves.len().min(topk);
            let order_gate_mode = crate::candidate_ranker::order_gate_mode();
            let mut ranked = moves[..split]
                .iter()
                .map(|&(mv, is_forcing)| {
                    let score = match order_gate_mode {
                        crate::candidate_ranker::RootGateMode::None => {
                            crate::candidate_ranker::root_candidate_score(board, mv, weights)
                                .unwrap_or(i32::MIN)
                        }
                        crate::candidate_ranker::RootGateMode::Tactical
                        | crate::candidate_ranker::RootGateMode::Strict => {
                            if crate::candidate_ranker::root_gate_key(board, mv).is_some() {
                                crate::candidate_ranker::root_candidate_score(board, mv, weights)
                                    .unwrap_or(i32::MIN)
                            } else {
                                i32::MIN
                            }
                        }
                    };
                    (mv, is_forcing, score)
                })
                .collect::<Vec<_>>();
            ranked.sort_unstable_by(|a, b| b.2.cmp(&a.2).then_with(|| b.0.cmp(&a.0)));
            for (dst, (mv, is_forcing, _)) in moves.iter_mut().take(split).zip(ranked) {
                *dst = (mv, is_forcing);
            }
        }

        moves
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

        let mut score =
            apply_weak_attack_cap(tier_score, attack_tier, block_tier, my_kind, opp_kind, ply);

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

        if defensive_open4_probe_enabled_for_ply(ply) {
            let risk = defensive_open4_risk_after_move(board, mv, defensive_open4_probe_depth());
            if risk != DefensiveOpen4Risk::Safe {
                let penalty = defensive_open4_risk_penalty(risk);
                let mode = defensive_open4_probe_mode();
                if mode == DefensiveOpen4ProbeMode::Demote {
                    score = score.saturating_sub(penalty);
                }
                if mode == DefensiveOpen4ProbeMode::Trace || defensive_open4_probe_trace_enabled() {
                    eprintln!(
                        "noru defensive-open4 probe ply={ply} mv=({row},{col}) risk={} penalty={penalty}",
                        risk.label()
                    );
                }
            }
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
];

/// `is_forcing` ???????癲ル슢?ο㎖?????????⑤벡???? bit i set ??ThreatKind discriminant i???????ル??? forcing.
/// ??????????뀀???嶺?forcing ???遺얘턁?????????? ClosedFour, OpenThree, Five, OpenFour, DoubleFour, FourThree.
/// (DoubleThree?????耀붾굝??????????????????????關?쒎첎?????蹂?????.)
const FORCING_MASK: u8 = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6);

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
];

#[inline]
fn is_forcing_kind(kind: ThreatKind) -> bool {
    (FORCING_MASK >> (kind as u8)) & 1 != 0
}

fn defensive_open4_risk_after_move(board: &Board, mv: Move, depth: u32) -> DefensiveOpen4Risk {
    let mut child = board.clone();
    child.make_move(mv);
    defensive_open4_risk_for_side_to_move(&mut child, depth)
}

fn defensive_open4_risk_for_side_to_move(board: &mut Board, depth: u32) -> DefensiveOpen4Risk {
    let attacker = board.side_to_move;
    let candidates = board.candidate_moves();
    let mut best = DefensiveOpen4Risk::Safe;

    for &attack_mv in &candidates {
        let kind = classify_move_fast(board, attack_mv, attacker);
        if kind == ThreatKind::Five {
            return DefensiveOpen4Risk::ImmediateFive;
        }
        if kind.is_winning() {
            best = best.max(DefensiveOpen4Risk::ImmediateWinningThreat);
        }
    }

    if best != DefensiveOpen4Risk::Safe || depth <= 1 {
        return best;
    }

    for &attack_mv in &candidates {
        board.make_move(attack_mv);
        let mut forced_block = None;
        let mut block_count = 0usize;
        for block_mv in board.candidate_moves() {
            if classify_move_fast(board, block_mv, attacker) == ThreatKind::Five {
                forced_block = Some(block_mv);
                block_count += 1;
                if block_count > 1 {
                    break;
                }
            }
        }

        if block_count == 1 {
            let block_mv = forced_block.expect("single forced block must be present");
            board.make_move(block_mv);
            for attack2_mv in board.candidate_moves() {
                let kind = classify_move_fast(board, attack2_mv, attacker);
                if kind == ThreatKind::Five {
                    board.undo_move();
                    board.undo_move();
                    return DefensiveOpen4Risk::ImmediateFive;
                }
                if kind.is_winning() {
                    board.undo_move();
                    board.undo_move();
                    return DefensiveOpen4Risk::ForcedBlockThenWinningThreat;
                }
            }
            board.undo_move();
        }

        board.undo_move();
    }

    DefensiveOpen4Risk::Safe
}

fn defensive_open4_risk_penalty(risk: DefensiveOpen4Risk) -> i32 {
    let penalty = defensive_open4_probe_penalty();
    match risk {
        DefensiveOpen4Risk::Safe => 0,
        DefensiveOpen4Risk::ForcedBlockThenWinningThreat => penalty / 2,
        DefensiveOpen4Risk::ImmediateWinningThreat | DefensiveOpen4Risk::ImmediateFive => penalty,
    }
}

#[inline]
fn apply_weak_attack_cap(
    score: i32,
    attack_tier: i32,
    block_tier: i32,
    my_kind: ThreatKind,
    opp_kind: ThreatKind,
    ply: usize,
) -> i32 {
    let Some(cap) = weak_attack_cap() else {
        return score;
    };
    if weak_attack_root_only() && ply != 0 {
        return score;
    }
    let weak_attack = block_tier == 0
        && attack_tier == score
        && opp_kind == ThreatKind::None
        && matches!(my_kind, ThreatKind::ClosedFour | ThreatKind::OpenThree);
    if weak_attack { score.min(cap) } else { score }
}

#[inline]
fn weak_attack_cap() -> Option<i32> {
    static CAP: OnceLock<Option<i32>> = OnceLock::new();
    *CAP.get_or_init(|| {
        let Ok(raw) = std::env::var("NORU_WEAK_ATTACK_CAP") else {
            return None;
        };
        let trimmed = raw.trim();
        if trimmed.is_empty()
            || trimmed == "0"
            || trimmed.eq_ignore_ascii_case("off")
            || trimmed.eq_ignore_ascii_case("false")
        {
            return None;
        }
        if trimmed.eq_ignore_ascii_case("demote") {
            return Some(0);
        }
        trimmed.parse::<i32>().ok().filter(|v| *v >= 0)
    })
}

#[inline]
fn weak_attack_root_only() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NORU_WEAK_ATTACK_CAP_ROOT_ONLY")
            .map(|v| {
                let trimmed = v.trim();
                !(trimmed.is_empty()
                    || trimmed == "0"
                    || trimmed.eq_ignore_ascii_case("off")
                    || trimmed.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(false)
    })
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
        ThreatKind::None => 0,
    };
    // ????????????????????????????????? ????(??? ?????關?쒎첎?嫄??怨몄겮???????? ??? ????븐뼐???????????癲됱빖???嶺??????????????).
    if defending { base * 9 / 10 } else { base }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, to_idx};
    use crate::features::GOMOKU_NNUE_CONFIG;

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
    fn defensive_open4_probe_detects_next_open_four() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 5));
        board.make_move(to_idx(5, 5));
        board.make_move(to_idx(7, 6));
        board.make_move(to_idx(5, 6));
        board.make_move(to_idx(7, 7));

        assert_eq!(board.side_to_move, Stone::White);
        let quiet = to_idx(5, 7);
        let risk = defensive_open4_risk_after_move(&board, quiet, 1);
        assert_eq!(risk, DefensiveOpen4Risk::ImmediateWinningThreat);
    }
}
