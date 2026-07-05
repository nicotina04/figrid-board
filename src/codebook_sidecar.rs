//! Pattern4 codebook sidecar used for root move-order tie-breaking.
//!
//! This is deliberately not a leaf evaluator. It scores only child boards
//! after a root move and is meant to break ties between already-equivalent
//! tactical ordering buckets.

use crate::board::{BOARD_SIZE, Board, GameResult, Move, NUM_CELLS, Stone};
use crate::pattern_table::{PATTERN_NUM_IDS, swap_mapped_id};
use serde_json::Value;
use std::sync::OnceLock;

const FORMAT: &str = "noru-pattern4-codebook-eval-v1";
const DEFAULT_MIN_PLY: usize = 5;
const DEFAULT_SCORE_SCALE: f32 = 10_000.0;
const TERMINAL_LOGIT: f32 = 8.0;

#[derive(Clone, Copy, Eq, PartialEq)]
enum HeadKind {
    Linear,
    Fm,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ScoreMode {
    Prob,
    Logit,
}

struct CodebookSidecar {
    default_score_mode: ScoreMode,
    head_kind: HeadKind,
    regions: usize,
    dim: usize,
    fm_rank: usize,
    embeddings: Vec<f32>,
    head: Vec<f32>,
    factors: Vec<f32>,
    bias: f32,
}

pub(crate) fn root_tiebreak_enabled_for(board: &Board) -> bool {
    root_tiebreak_enabled() && board.move_count >= min_ply() && sidecar().is_some()
}

pub(crate) fn root_audit_enabled_for(board: &Board) -> bool {
    root_audit_enabled() && board.move_count >= min_ply() && sidecar().is_some()
}

pub(crate) fn root_final_tiebreak_enabled_for(board: &Board) -> bool {
    root_final_tiebreak_enabled() && board.move_count >= min_ply() && sidecar().is_some()
}

pub(crate) fn root_order_tiebreak_enabled() -> bool {
    root_order_tiebreak_switch_enabled() && sidecar().is_some()
}

fn root_order_tiebreak_switch_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_env_bool_default("NORU_CODEBOOK_ROOT_ORDER_TIEBREAK", true))
}

pub(crate) fn root_tie_margin() -> u64 {
    static MARGIN: OnceLock<u64> = OnceLock::new();
    *MARGIN.get_or_init(|| parse_env_u64("NORU_CODEBOOK_ROOT_TIE_MARGIN").unwrap_or(0))
}

pub(crate) fn root_margin() -> i32 {
    static MARGIN: OnceLock<i32> = OnceLock::new();
    *MARGIN.get_or_init(|| {
        parse_env_i32("NORU_CODEBOOK_ROOT_MARGIN")
            .filter(|v| *v >= 0)
            .unwrap_or(0)
    })
}

pub(crate) fn root_require_global_best() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_env_bool_default("NORU_CODEBOOK_ROOT_REQUIRE_GLOBAL_BEST", false))
}

pub(crate) fn root_global_best_margin() -> i32 {
    static MARGIN: OnceLock<i32> = OnceLock::new();
    *MARGIN.get_or_init(|| {
        parse_env_i32("NORU_CODEBOOK_ROOT_GLOBAL_BEST_MARGIN")
            .filter(|v| *v >= 0)
            .unwrap_or(0)
    })
}

pub(crate) fn root_global_best_allows(candidate: Option<i32>, best: Option<i32>) -> bool {
    if !root_require_global_best() {
        return true;
    }
    match (candidate, best) {
        (Some(candidate), Some(best)) => {
            candidate.saturating_add(root_global_best_margin()) >= best
        }
        _ => false,
    }
}

pub(crate) fn root_final_max_search_delta() -> i32 {
    static DELTA: OnceLock<i32> = OnceLock::new();
    *DELTA.get_or_init(|| {
        parse_env_i32("NORU_CODEBOOK_ROOT_FINAL_MAX_SEARCH_DELTA")
            .filter(|v| *v >= 0)
            .unwrap_or(0)
    })
}

pub(crate) fn root_final_score_margin() -> i32 {
    static MARGIN: OnceLock<i32> = OnceLock::new();
    *MARGIN.get_or_init(|| {
        parse_env_i32("NORU_CODEBOOK_ROOT_FINAL_SCORE_MARGIN")
            .filter(|v| *v >= 0)
            .unwrap_or(0)
    })
}

pub(crate) fn root_final_max_local_deficit() -> i32 {
    static DEFICIT: OnceLock<i32> = OnceLock::new();
    *DEFICIT.get_or_init(|| {
        parse_env_i32("NORU_CODEBOOK_ROOT_FINAL_MAX_LOCAL_DEFICIT")
            .filter(|v| *v >= 0)
            .unwrap_or(0)
    })
}

pub(crate) fn root_final_require_global_best() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        parse_env_bool_default("NORU_CODEBOOK_ROOT_FINAL_REQUIRE_GLOBAL_BEST", true)
    })
}

pub(crate) fn root_final_global_best_margin() -> i32 {
    static MARGIN: OnceLock<i32> = OnceLock::new();
    *MARGIN.get_or_init(|| {
        parse_env_i32("NORU_CODEBOOK_ROOT_FINAL_GLOBAL_BEST_MARGIN")
            .filter(|v| *v >= 0)
            .unwrap_or(0)
    })
}

pub(crate) fn root_gate_mode() -> crate::candidate_ranker::RootGateMode {
    static MODE: OnceLock<crate::candidate_ranker::RootGateMode> = OnceLock::new();
    *MODE.get_or_init(|| parse_gate_mode_env("NORU_CODEBOOK_ROOT_GATE"))
}

pub(crate) fn root_candidate_score(board: &Board, mv: Move) -> Option<i32> {
    let sidecar = sidecar()?;
    let mode = score_mode_override().unwrap_or(sidecar.default_score_mode);
    let raw = sidecar.score_child(board, mv, mode);
    if !raw.is_finite() {
        return None;
    }
    let scaled = (raw * score_scale()).round();
    if !scaled.is_finite() {
        return None;
    }
    Some(scaled.clamp(i32::MIN as f32 + 1.0, i32::MAX as f32 - 1.0) as i32)
}

fn sidecar() -> Option<&'static CodebookSidecar> {
    static SIDECAR: OnceLock<Option<CodebookSidecar>> = OnceLock::new();
    SIDECAR.get_or_init(load_sidecar).as_ref()
}

fn root_tiebreak_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_env_bool_default("NORU_CODEBOOK_ROOT_TIEBREAK", true))
}

fn root_audit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_env_bool_default("NORU_CODEBOOK_ROOT_AUDIT", false))
}

fn root_final_tiebreak_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_env_bool_default("NORU_CODEBOOK_ROOT_FINAL_TIEBREAK", false))
}

fn min_ply() -> usize {
    static MIN_PLY: OnceLock<usize> = OnceLock::new();
    *MIN_PLY.get_or_init(|| parse_env_usize("NORU_CODEBOOK_MIN_PLY").unwrap_or(DEFAULT_MIN_PLY))
}

fn score_scale() -> f32 {
    static SCALE: OnceLock<f32> = OnceLock::new();
    *SCALE.get_or_init(|| {
        parse_env_f32("NORU_CODEBOOK_SCORE_SCALE")
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(DEFAULT_SCORE_SCALE)
    })
}

fn score_mode_override() -> Option<ScoreMode> {
    static MODE: OnceLock<Option<ScoreMode>> = OnceLock::new();
    *MODE.get_or_init(|| {
        let Ok(raw) = std::env::var("NORU_CODEBOOK_SCORE_MODE") else {
            return None;
        };
        let trimmed = raw.trim();
        if is_disabled_value(trimmed) || trimmed.eq_ignore_ascii_case("auto") {
            return None;
        }
        match trimmed.to_ascii_lowercase().as_str() {
            "prob" | "probability" | "sigmoid" => Some(ScoreMode::Prob),
            "logit" | "margin" => Some(ScoreMode::Logit),
            other => {
                panic!("invalid NORU_CODEBOOK_SCORE_MODE={other:?}; expected auto, prob, or logit")
            }
        }
    })
}

fn load_sidecar() -> Option<CodebookSidecar> {
    let Ok(path) = std::env::var("NORU_CODEBOOK_SIDECAR") else {
        return None;
    };
    let trimmed = path.trim();
    if is_disabled_value(trimmed) {
        return None;
    }
    let text = std::fs::read_to_string(trimmed)
        .unwrap_or_else(|e| panic!("failed to read NORU_CODEBOOK_SIDECAR={trimmed}: {e}"));
    let value: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse codebook sidecar {trimmed}: {e}"));
    Some(
        parse_sidecar(&value).unwrap_or_else(|e| panic!("invalid codebook sidecar {trimmed}: {e}")),
    )
}

fn parse_sidecar(value: &Value) -> Result<CodebookSidecar, String> {
    let format = value
        .get("format")
        .and_then(Value::as_str)
        .ok_or("missing format")?;
    if format != FORMAT {
        return Err(format!("unsupported format {format:?}"));
    }

    let head_kind = match value.get("head").and_then(Value::as_str).unwrap_or("") {
        "linear" => HeadKind::Linear,
        "fm" => HeadKind::Fm,
        other => return Err(format!("unsupported head {other:?}")),
    };
    let objective = value
        .get("training")
        .and_then(|t| t.get("objective"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let default_score_mode =
        if objective.starts_with("rapfi_child_gt") || objective.starts_with("root_choice") {
            ScoreMode::Logit
        } else {
            ScoreMode::Prob
        };

    let regions = required_usize(value, "regions")?;
    if regions != 1 && regions != 9 {
        return Err(format!("unsupported regions={regions}; expected 1 or 9"));
    }
    let dim = required_usize(value, "embedding_dim")?;
    let fm_rank = required_usize(value, "fm_rank")?;
    let pattern_num_ids = required_usize(value, "pattern_num_ids")?;
    if pattern_num_ids != PATTERN_NUM_IDS {
        return Err(format!(
            "pattern_num_ids mismatch: checkpoint={} runtime={}",
            pattern_num_ids, PATTERN_NUM_IDS
        ));
    }
    if required_usize(value, "board_size")? != BOARD_SIZE
        || required_usize(value, "num_cells")? != NUM_CELLS
    {
        return Err("board shape mismatch".to_string());
    }

    let weights = value
        .get("weights")
        .ok_or_else(|| "missing weights".to_string())?;
    let embeddings = required_f32_vec(weights, "embeddings")?;
    let head = required_f32_vec(weights, "head")?;
    let factors = required_f32_vec(weights, "factors")?;
    let bias = weights
        .get("bias")
        .and_then(Value::as_f64)
        .ok_or_else(|| "missing weights.bias".to_string())? as f32;

    let expected_emb = PATTERN_NUM_IDS * dim;
    let expected_head = regions * dim;
    let expected_factors = if head_kind == HeadKind::Fm {
        expected_head * fm_rank
    } else {
        0
    };
    if embeddings.len() != expected_emb {
        return Err(format!(
            "embedding length mismatch: got {}, expected {}",
            embeddings.len(),
            expected_emb
        ));
    }
    if head.len() != expected_head {
        return Err(format!(
            "head length mismatch: got {}, expected {}",
            head.len(),
            expected_head
        ));
    }
    if factors.len() != expected_factors {
        return Err(format!(
            "factor length mismatch: got {}, expected {}",
            factors.len(),
            expected_factors
        ));
    }
    if !bias.is_finite()
        || embeddings.iter().any(|v| !v.is_finite())
        || head.iter().any(|v| !v.is_finite())
        || factors.iter().any(|v| !v.is_finite())
    {
        return Err("checkpoint contains non-finite weights".to_string());
    }

    Ok(CodebookSidecar {
        default_score_mode,
        head_kind,
        regions,
        dim,
        fm_rank,
        embeddings,
        head,
        factors,
        bias,
    })
}

impl CodebookSidecar {
    fn score_child(&self, board: &Board, mv: Move, mode: ScoreMode) -> f32 {
        let root_side = board.side_to_move;
        let mut child = board.clone();
        child.make_move(mv);
        if let Some(score) = terminal_score(&child, root_side, mode) {
            return score;
        }
        let logit = self.logit_board(&child);
        match mode {
            ScoreMode::Prob => {
                let pred = sigmoid(logit);
                if child.side_to_move == root_side {
                    pred
                } else {
                    1.0 - pred
                }
            }
            ScoreMode::Logit => {
                if child.side_to_move == root_side {
                    logit
                } else {
                    -logit
                }
            }
        }
    }

    fn logit_board(&self, board: &Board) -> f32 {
        let pool = self.pool_board(board);
        let mut logit = self.bias;
        for (x, w) in pool.iter().zip(&self.head) {
            logit += x * w;
        }
        if self.head_kind == HeadKind::Fm && self.fm_rank > 0 {
            for rank in 0..self.fm_rank {
                let mut sum = 0.0f32;
                let mut square_sum = 0.0f32;
                for (idx, &x) in pool.iter().enumerate() {
                    let vx = self.factors[idx * self.fm_rank + rank] * x;
                    sum += vx;
                    square_sum += vx * vx;
                }
                logit += 0.5 * (sum * sum - square_sum);
            }
        }
        logit
    }

    fn pool_board(&self, board: &Board) -> Vec<f32> {
        let mut pool = vec![0.0f32; self.regions * self.dim];
        let mut cell_pre = vec![0.0f32; self.dim];
        let swap = board.side_to_move == Stone::White;
        for (cell, dirs) in board.line_pattern_ids.iter().enumerate() {
            cell_pre.fill(0.0);
            for &pid in dirs {
                let pid = if swap { swap_mapped_id(pid) } else { pid };
                let emb_base = pid as usize * self.dim;
                for (d, v) in cell_pre.iter_mut().enumerate() {
                    *v += self.embeddings[emb_base + d];
                }
            }

            let region = region_of_cell(cell, self.regions);
            let pool_base = region * self.dim;
            let denom = region_cell_count(region, self.regions) as f32;
            for d in 0..self.dim {
                pool[pool_base + d] += cell_pre[d].max(0.0) / denom;
            }
        }
        pool
    }
}

fn terminal_score(board: &Board, root_side: Stone, mode: ScoreMode) -> Option<f32> {
    match board.game_result() {
        GameResult::BlackWin => Some(side_terminal_score(root_side == Stone::Black, mode)),
        GameResult::WhiteWin => Some(side_terminal_score(root_side == Stone::White, mode)),
        GameResult::Draw => Some(match mode {
            ScoreMode::Prob => 0.5,
            ScoreMode::Logit => 0.0,
        }),
        GameResult::Ongoing => None,
    }
}

fn side_terminal_score(root_won: bool, mode: ScoreMode) -> f32 {
    match mode {
        ScoreMode::Prob => {
            if root_won {
                1.0
            } else {
                0.0
            }
        }
        ScoreMode::Logit => {
            if root_won {
                TERMINAL_LOGIT
            } else {
                -TERMINAL_LOGIT
            }
        }
    }
}

fn region_of_cell(cell: usize, regions: usize) -> usize {
    if regions == 1 {
        return 0;
    }
    let row = cell / BOARD_SIZE;
    let col = cell % BOARD_SIZE;
    let rr = (row / 5).min(2);
    let cc = (col / 5).min(2);
    rr * 3 + cc
}

fn region_cell_count(_region: usize, regions: usize) -> usize {
    match regions {
        1 => NUM_CELLS,
        9 => 25,
        _ => NUM_CELLS,
    }
}

fn required_usize(value: &Value, key: &str) -> Result<usize, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| format!("missing {key}"))
}

fn required_f32_vec(value: &Value, key: &str) -> Result<Vec<f32>, String> {
    let arr = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing weights.{key}"))?;
    arr.iter()
        .map(|x| {
            x.as_f64()
                .map(|v| v as f32)
                .ok_or_else(|| format!("non-float value in weights.{key}"))
        })
        .collect()
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn parse_env_bool_default(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|raw| {
            let trimmed = raw.trim();
            !(trimmed.is_empty()
                || trimmed == "0"
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no"))
        })
        .unwrap_or(default)
}

fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
}

fn parse_env_u64(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
}

fn parse_env_i32(name: &str) -> Option<i32> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<i32>().ok())
}

fn parse_env_f32(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
}

fn parse_gate_mode_env(name: &str) -> crate::candidate_ranker::RootGateMode {
    let Ok(raw) = std::env::var(name) else {
        return crate::candidate_ranker::RootGateMode::Strict;
    };
    let trimmed = raw.trim();
    if is_disabled_value(trimmed) || trimmed.eq_ignore_ascii_case("none") {
        return crate::candidate_ranker::RootGateMode::None;
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "tactical" | "nonquiet" | "non-quiet" => crate::candidate_ranker::RootGateMode::Tactical,
        "strict" | "same-threat" | "same_threat" => crate::candidate_ranker::RootGateMode::Strict,
        other => panic!("invalid {name}={other:?}; expected none, tactical, or strict"),
    }
}

fn is_disabled_value(value: &str) -> bool {
    value.is_empty()
        || value.eq_ignore_ascii_case("0")
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("no")
}
