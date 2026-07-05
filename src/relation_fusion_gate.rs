//! Experimental relation-fusion root gate.
//!
//! Disabled unless `NORU_RELATION_FUSION_RERANKER` points at a JSON checkpoint
//! produced by noru-tactic's `train-relation-fusion-*` tools.

use crate::board::{BOARD_SIZE, Board, Move, NUM_CELLS, Stone, to_idx};
use crate::pattern_table::{PATTERN_NUM_IDS, swap_mapped_id};
use crate::vct::{THREAT_KIND_COUNT, ThreatKind, classify_move_fast};
use serde_json::Value;
use std::sync::OnceLock;

const FORMAT: &str = "noru-relation-fusion-eval-v1";
const WINDOW_LENS: [usize; 3] = [5, 6, 7];
const DEFAULT_MIN_PLY: usize = 5;
const DEFAULT_MARGIN: f32 = 2.0;
const DEFAULT_ORDER_TOP_K: usize = 10;
const DEFAULT_ORDER_TIE_MARGIN: u64 = 0;
const DEFAULT_PROTECT_ORDER_RANK: usize = 5;
const DEFAULT_SAFETY_THRESHOLD: f32 = 0.55;

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
const TIER_SCALE: f32 = TIER_WIN as f32;
const FORCING_MASK: u8 = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6);

const MOVE_ATTACK_TABLE: [i32; THREAT_KIND_COUNT] = [
    0,
    TIER_CLOSED_FOUR,
    TIER_OPEN_THREE,
    TIER_WIN,
    TIER_OPEN_FOUR,
    TIER_DOUBLE_FOUR,
    TIER_DOUBLE_FOUR,
    TIER_DOUBLE_THREE,
];

const MOVE_BLOCK_TABLE: [i32; THREAT_KIND_COUNT] = [
    0,
    TIER_BLOCK_CLOSED_FOUR,
    TIER_BLOCK_OPEN_THREE,
    TIER_BLOCK_WIN,
    TIER_BLOCK_OPEN_FOUR,
    TIER_BLOCK_DOUBLE_FOUR,
    TIER_BLOCK_DOUBLE_FOUR,
    TIER_BLOCK_DOUBLE_THREE,
];

#[derive(Clone, Copy)]
enum ScoreMode {
    RootProbability,
    RootLogit,
}

#[derive(Clone)]
struct WindowSpec {
    len: usize,
    cells: Vec<usize>,
    before: Option<usize>,
    after: Option<usize>,
}

struct ThreatSpec {
    windows: Vec<WindowSpec>,
    window_counts: Vec<(usize, usize)>,
    feature_count: usize,
}

struct RelationFusionModel {
    dim: usize,
    fm_rank: usize,
    codebook_dim: usize,
    score_mode: ScoreMode,
    embeddings: Vec<f32>,
    head: Vec<f32>,
    factors: Vec<f32>,
    bias: f32,
    threat_mean: Vec<f32>,
    threat_std: Vec<f32>,
}

struct Gate {
    reranker: RelationFusionModel,
    safety: Option<RelationFusionModel>,
    spec: ThreatSpec,
}

#[derive(Clone, Copy)]
struct ScoredMove {
    mv: Move,
    score: f32,
}

pub(crate) fn enabled_for(board: &Board) -> bool {
    board.move_count >= min_ply() && gate().is_some()
}

pub(crate) fn root_order_tiebreak_enabled_for(board: &Board) -> bool {
    enabled_for(board) && order_tiebreak_enabled()
}

pub(crate) fn root_order_tie_margin() -> u64 {
    order_tie_margin()
}

pub(crate) fn root_candidate_score(board: &Board, mv: Move) -> Option<i32> {
    if !enabled_for(board) || !board.is_empty(mv) {
        return None;
    }
    let gate = gate()?;
    Some((gate.reranker.score_child(board, mv, &gate.spec) * 1_000_000.0).round() as i32)
}

pub(crate) fn choose_replacement(
    board: &Board,
    ordered: &[(Move, bool)],
    search_best: Option<Move>,
) -> Option<Move> {
    if !replacement_enabled() {
        return None;
    }
    if !enabled_for(board) {
        return None;
    }
    let gate = gate()?;
    let search_best = search_best?;
    if immediate_win(board, search_best) {
        return None;
    }
    let protect_rank = protect_order_rank();
    if protect_rank > 0
        && ordered
            .iter()
            .position(|&(mv, _)| mv == search_best)
            .is_some_and(|idx| idx + 1 <= protect_rank)
    {
        return None;
    }

    let candidates = board.candidate_moves();
    let mut scored = candidates
        .iter()
        .copied()
        .filter(|&mv| board.is_empty(mv))
        .map(|mv| ScoredMove {
            mv,
            score: gate.reranker.score_child(board, mv, &gate.spec),
        })
        .collect::<Vec<_>>();
    scored.sort_by(compare_scored_desc);

    let search_score = scored.iter().find(|s| s.mv == search_best)?.score;
    let margin = margin();
    let top_k = order_top_k().max(1);
    let raw = ordered.iter().take(top_k).find_map(|&(mv, _)| {
        if mv == search_best || !board.is_empty(mv) || immediate_win(board, search_best) {
            return None;
        }
        let candidate_score = scored.iter().find(|s| s.mv == mv)?.score;
        (candidate_score - search_score >= margin).then_some(mv)
    })?;

    if let Some(safety) = &gate.safety {
        let root_score = safety.score_child(board, raw, &gate.spec);
        let child_risk = 1.0 - root_score;
        if child_risk >= safety_threshold() {
            return None;
        }
    }
    Some(raw)
}

fn gate() -> Option<&'static Gate> {
    static GATE: OnceLock<Option<Gate>> = OnceLock::new();
    GATE.get_or_init(load_gate).as_ref()
}

fn load_gate() -> Option<Gate> {
    let Ok(path) = std::env::var("NORU_RELATION_FUSION_RERANKER") else {
        return None;
    };
    let trimmed = path.trim();
    if is_disabled_value(trimmed) {
        return None;
    }
    let spec = ThreatSpec::new();
    let reranker = load_model(trimmed, &spec, "NORU_RELATION_FUSION_RERANKER");
    let safety = std::env::var("NORU_RELATION_FUSION_SAFETY")
        .ok()
        .map(|raw| raw.trim().to_string())
        .filter(|raw| !is_disabled_value(raw))
        .map(|raw| load_model(&raw, &spec, "NORU_RELATION_FUSION_SAFETY"));
    Some(Gate {
        reranker,
        safety,
        spec,
    })
}

fn load_model(path: &str, spec: &ThreatSpec, env_name: &str) -> RelationFusionModel {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {env_name}={path}: {e}"));
    let value: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse {env_name}={path}: {e}"));
    RelationFusionModel::parse(&value, spec.feature_count)
        .unwrap_or_else(|e| panic!("invalid {env_name}={path}: {e}"))
}

impl RelationFusionModel {
    fn parse(value: &Value, expected_threat_dim: usize) -> Result<Self, String> {
        let format = str_req(value, "format")?;
        if format != FORMAT {
            return Err(format!("unsupported format {format:?}"));
        }
        let metadata = value.get("metadata").ok_or("missing metadata")?;
        let weights = value.get("weights").ok_or("missing weights")?;
        let norm = value.get("normalization").ok_or("missing normalization")?;
        let objective = value
            .get("objective")
            .and_then(Value::as_str)
            .unwrap_or("rapfi_value_bce");
        let score_mode = if objective.starts_with("rapfi_child_gt") {
            ScoreMode::RootLogit
        } else {
            ScoreMode::RootProbability
        };
        let dim = usize_req(metadata, "embedding_dim")?;
        let fm_rank = usize_req(metadata, "fm_rank")?;
        let codebook_dim = usize_req(metadata, "codebook_features")?;
        let threat_dim = usize_req(metadata, "threat_features")?;
        if threat_dim != expected_threat_dim {
            return Err(format!(
                "threat_features mismatch: checkpoint={threat_dim} runtime={expected_threat_dim}"
            ));
        }
        let embeddings = f32_array(weights.get("embeddings"), "weights.embeddings")?;
        let head = f32_array(weights.get("head"), "weights.head")?;
        let factors = f32_array(weights.get("factors"), "weights.factors")?;
        let bias = weights
            .get("bias")
            .and_then(Value::as_f64)
            .ok_or("missing weights.bias")? as f32;
        let threat_mean = f32_array(norm.get("threat_mean"), "normalization.threat_mean")?;
        let threat_std = f32_array(norm.get("threat_std"), "normalization.threat_std")?;
        if embeddings.len() != PATTERN_NUM_IDS * dim {
            return Err(format!(
                "bad embeddings length {}; expected {}",
                embeddings.len(),
                PATTERN_NUM_IDS * dim
            ));
        }
        if head.len() != codebook_dim + threat_dim {
            return Err(format!(
                "bad head length {}; expected {}",
                head.len(),
                codebook_dim + threat_dim
            ));
        }
        if factors.len() != head.len() * fm_rank {
            return Err(format!(
                "bad factors length {}; expected {}",
                factors.len(),
                head.len() * fm_rank
            ));
        }
        if threat_mean.len() != threat_dim || threat_std.len() != threat_dim {
            return Err("bad threat normalization lengths".to_string());
        }
        Ok(Self {
            dim,
            fm_rank,
            codebook_dim,
            score_mode,
            embeddings,
            head,
            factors,
            bias,
            threat_mean,
            threat_std,
        })
    }

    fn score_child(&self, board: &Board, mv: Move, spec: &ThreatSpec) -> f32 {
        match self.score_mode {
            ScoreMode::RootProbability => self.score_child_probability(board, mv, spec),
            ScoreMode::RootLogit => self.score_child_logit(board, mv, spec),
        }
    }

    fn score_child_probability(&self, board: &Board, mv: Move, spec: &ThreatSpec) -> f32 {
        let root_side = board.side_to_move;
        let mut child = board.clone();
        child.make_move(mv);
        if child.check_win(mv) {
            return 1.0;
        }
        if child.move_count == NUM_CELLS {
            return 0.5;
        }
        let pred = sigmoid(self.logit(&child, spec));
        if child.side_to_move == root_side {
            pred
        } else {
            1.0 - pred
        }
    }

    fn score_child_logit(&self, board: &Board, mv: Move, spec: &ThreatSpec) -> f32 {
        let root_side = board.side_to_move;
        let mut child = board.clone();
        child.make_move(mv);
        let sign = if child.side_to_move == root_side {
            1.0
        } else {
            -1.0
        };
        sign * self.logit(&child, spec)
    }

    fn logit(&self, board: &Board, spec: &ThreatSpec) -> f32 {
        let mut features = self.extract_codebook_features(board);
        let mut threat = extract_threat_features(board, spec);
        for i in 0..threat.len() {
            threat[i] = (threat[i] - self.threat_mean[i]) / self.threat_std[i].max(1e-4);
        }
        features.extend(threat);
        let mut logit = self.bias;
        for (x, w) in features.iter().zip(&self.head) {
            logit += x * w;
        }
        for rank in 0..self.fm_rank {
            let mut sum = 0.0f32;
            let mut square_sum = 0.0f32;
            for (idx, &x) in features.iter().enumerate() {
                let vx = self.factors[idx * self.fm_rank + rank] * x;
                sum += vx;
                square_sum += vx * vx;
            }
            logit += 0.5 * (sum * sum - square_sum);
        }
        logit
    }

    fn extract_codebook_features(&self, board: &Board) -> Vec<f32> {
        let mut features = vec![0.0f32; self.codebook_dim];
        let mut cell_pre = vec![0.0f32; NUM_CELLS * self.dim];
        let swap = board.side_to_move == Stone::White;
        for (cell, dirs) in board.line_pattern_ids.iter().enumerate() {
            let region = region_of_cell(cell);
            let feature_base = region * self.dim;
            let cell_base = cell * self.dim;
            for &pid in dirs {
                let pid = if swap { swap_mapped_id(pid) } else { pid };
                let emb_base = pid as usize * self.dim;
                for d in 0..self.dim {
                    cell_pre[cell_base + d] += self.embeddings[emb_base + d];
                }
            }
            for d in 0..self.dim {
                features[feature_base + d] += cell_pre[cell_base + d].max(0.0) / 25.0;
            }
        }
        features
    }
}

impl ThreatSpec {
    fn new() -> Self {
        let mut windows = Vec::new();
        let mut window_counts = Vec::new();
        for &len in &WINDOW_LENS {
            let before = windows.len();
            push_windows_for_len(len, &mut windows);
            window_counts.push((len, windows.len() - before));
        }
        let feature_count = 13
            + THREAT_KIND_COUNT * 2
            + THREAT_KIND_COUNT * THREAT_KIND_COUNT
            + WINDOW_LENS
                .iter()
                .map(|&len| 1 + 2 * len * 3)
                .sum::<usize>();
        Self {
            windows,
            window_counts,
            feature_count,
        }
    }
}

fn push_windows_for_len(len: usize, out: &mut Vec<WindowSpec>) {
    let dirs: [(isize, isize); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
    for &(dr, dc) in &dirs {
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                let end_r = r as isize + dr * (len as isize - 1);
                let end_c = c as isize + dc * (len as isize - 1);
                if !in_bounds(end_r, end_c) {
                    continue;
                }
                let cells = (0..len)
                    .map(|offset| {
                        to_idx(
                            (r as isize + dr * offset as isize) as usize,
                            (c as isize + dc * offset as isize) as usize,
                        )
                    })
                    .collect();
                let before = idx_if_in_bounds(r as isize - dr, c as isize - dc);
                let after = idx_if_in_bounds(
                    r as isize + dr * len as isize,
                    c as isize + dc * len as isize,
                );
                out.push(WindowSpec {
                    len,
                    cells,
                    before,
                    after,
                });
            }
        }
    }
}

fn extract_threat_features(board: &Board, spec: &ThreatSpec) -> Vec<f32> {
    let mut features = Vec::with_capacity(spec.feature_count);
    push_candidate_threat_features(board, &mut features);
    push_window_tensor_features(board, spec, &mut features);
    debug_assert_eq!(features.len(), spec.feature_count);
    features
}

fn push_candidate_threat_features(board: &Board, features: &mut Vec<f32>) {
    let candidates = board.candidate_moves();
    let denom = candidates.len().max(1) as f32;
    features.push(match board.side_to_move {
        Stone::Black => 1.0,
        Stone::White => -1.0,
    });
    features.push(board.move_count as f32 / NUM_CELLS as f32);
    features.push(candidates.len() as f32 / NUM_CELLS as f32);

    let side = board.side_to_move;
    let opp = side.opponent();
    let mut attack_counts = [0usize; THREAT_KIND_COUNT];
    let mut block_counts = [0usize; THREAT_KIND_COUNT];
    let mut pair_counts = [0usize; THREAT_KIND_COUNT * THREAT_KIND_COUNT];
    let mut best_attack = 0;
    let mut best_block = 0;
    let mut weak_attack = 0usize;
    let mut weak_closed_four = 0usize;
    let mut weak_open_three = 0usize;
    let mut own_forcing = 0usize;
    let mut opp_forcing = 0usize;
    let mut urgent_block = 0usize;

    for mv in candidates {
        if !board.is_empty(mv) {
            continue;
        }
        let attack = classify_move_fast(board, mv, side);
        let block = classify_move_fast(board, mv, opp);
        let ai = attack as usize;
        let bi = block as usize;
        attack_counts[ai] += 1;
        block_counts[bi] += 1;
        pair_counts[ai * THREAT_KIND_COUNT + bi] += 1;
        best_attack = best_attack.max(MOVE_ATTACK_TABLE[ai]);
        best_block = best_block.max(MOVE_BLOCK_TABLE[bi]);
        if weak_attack_move(attack, block) {
            weak_attack += 1;
            if attack == ThreatKind::ClosedFour {
                weak_closed_four += 1;
            }
            if attack == ThreatKind::OpenThree {
                weak_open_three += 1;
            }
        }
        if is_forcing_kind(attack) {
            own_forcing += 1;
        }
        if is_forcing_kind(block) {
            opp_forcing += 1;
        }
        if matches!(
            block,
            ThreatKind::Five
                | ThreatKind::OpenFour
                | ThreatKind::DoubleFour
                | ThreatKind::FourThree
                | ThreatKind::DoubleThree
        ) {
            urgent_block += 1;
        }
    }

    features.push(best_attack as f32 / TIER_SCALE);
    features.push(best_block as f32 / TIER_SCALE);
    features.push((best_attack - best_block) as f32 / TIER_SCALE);
    features.push(if weak_attack > 0 { 1.0 } else { 0.0 });
    features.push(weak_attack as f32 / denom);
    features.push(weak_closed_four as f32 / denom);
    features.push(weak_open_three as f32 / denom);
    features.push(own_forcing as f32 / denom);
    features.push(opp_forcing as f32 / denom);
    features.push(urgent_block as f32 / denom);
    for count in attack_counts {
        features.push(count as f32 / denom);
    }
    for count in block_counts {
        features.push(count as f32 / denom);
    }
    for count in pair_counts {
        features.push(count as f32 / denom);
    }
}

fn push_window_tensor_features(board: &Board, spec: &ThreatSpec, features: &mut Vec<f32>) {
    for &(len, count) in &spec.window_counts {
        let mut mixed = 0usize;
        let mut own = vec![[0usize; 3]; len + 1];
        let mut opp = vec![[0usize; 3]; len + 1];
        for window in spec.windows.iter().filter(|w| w.len == len) {
            let mut black = 0usize;
            let mut white = 0usize;
            for &cell in &window.cells {
                if board.black.get(cell) {
                    black += 1;
                } else if board.white.get(cell) {
                    white += 1;
                }
            }
            if black > 0 && white > 0 {
                mixed += 1;
                continue;
            }
            let open_ends = window_open_ends(board, window);
            if black > 0 {
                let bucket = if board.side_to_move == Stone::Black {
                    &mut own
                } else {
                    &mut opp
                };
                bucket[black][open_ends] += 1;
            } else if white > 0 {
                let bucket = if board.side_to_move == Stone::White {
                    &mut own
                } else {
                    &mut opp
                };
                bucket[white][open_ends] += 1;
            }
        }
        let denom = count.max(1) as f32;
        features.push(mixed as f32 / denom);
        for owner in [&own, &opp] {
            for stones in 1..=len {
                for open_ends in 0..=2 {
                    features.push(owner[stones][open_ends] as f32 / denom);
                }
            }
        }
    }
}

fn window_open_ends(board: &Board, window: &WindowSpec) -> usize {
    [window.before, window.after]
        .into_iter()
        .flatten()
        .filter(|&idx| board.is_empty(idx))
        .count()
}

fn immediate_win(board: &Board, mv: Move) -> bool {
    classify_move_fast(board, mv, board.side_to_move) == ThreatKind::Five
}

fn weak_attack_move(attack: ThreatKind, block: ThreatKind) -> bool {
    matches!(attack, ThreatKind::ClosedFour | ThreatKind::OpenThree)
        && !matches!(
            block,
            ThreatKind::Five
                | ThreatKind::OpenFour
                | ThreatKind::DoubleFour
                | ThreatKind::FourThree
                | ThreatKind::DoubleThree
        )
}

#[inline]
fn is_forcing_kind(kind: ThreatKind) -> bool {
    (FORCING_MASK >> (kind as u8)) & 1 != 0
}

fn region_of_cell(cell: usize) -> usize {
    let row = cell / BOARD_SIZE;
    let col = cell % BOARD_SIZE;
    let rr = (row / 5).min(2);
    let cc = (col / 5).min(2);
    rr * 3 + cc
}

fn compare_scored_desc(a: &ScoredMove, b: &ScoredMove) -> std::cmp::Ordering {
    b.score.total_cmp(&a.score).then_with(|| a.mv.cmp(&b.mv))
}

fn in_bounds(r: isize, c: isize) -> bool {
    (0..BOARD_SIZE as isize).contains(&r) && (0..BOARD_SIZE as isize).contains(&c)
}

fn idx_if_in_bounds(r: isize, c: isize) -> Option<usize> {
    in_bounds(r, c).then(|| to_idx(r as usize, c as usize))
}

fn min_ply() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE
        .get_or_init(|| parse_env_usize("NORU_RELATION_FUSION_MIN_PLY").unwrap_or(DEFAULT_MIN_PLY))
}

fn margin() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_f32("NORU_RELATION_FUSION_MARGIN")
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(DEFAULT_MARGIN)
    })
}

fn order_top_k() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_usize("NORU_RELATION_FUSION_ORDER_TOP_K").unwrap_or(DEFAULT_ORDER_TOP_K)
    })
}

fn order_tiebreak_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| parse_env_bool("NORU_RELATION_FUSION_ORDER_TIEBREAK").unwrap_or(false))
}

fn replacement_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| parse_env_bool("NORU_RELATION_FUSION_REPLACE").unwrap_or(true))
}

fn order_tie_margin() -> u64 {
    static VALUE: OnceLock<u64> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_usize("NORU_RELATION_FUSION_ORDER_TIE_MARGIN")
            .map(|v| v as u64)
            .unwrap_or(DEFAULT_ORDER_TIE_MARGIN)
    })
}

fn protect_order_rank() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_usize("NORU_RELATION_FUSION_PROTECT_ORDER_RANK")
            .unwrap_or(DEFAULT_PROTECT_ORDER_RANK)
    })
}

fn safety_threshold() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_f32("NORU_RELATION_FUSION_SAFETY_THRESHOLD")
            .filter(|v| v.is_finite())
            .unwrap_or(DEFAULT_SAFETY_THRESHOLD)
    })
}

fn parse_env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
}

fn parse_env_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
}

fn parse_env_bool(key: &str) -> Option<bool> {
    std::env::var(key).ok().and_then(|raw| {
        let raw = raw.trim();
        if is_disabled_value(raw) {
            Some(false)
        } else if raw == "1" || raw.eq_ignore_ascii_case("on") || raw.eq_ignore_ascii_case("true") {
            Some(true)
        } else {
            None
        }
    })
}

fn is_disabled_value(value: &str) -> bool {
    value.is_empty()
        || value == "0"
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("false")
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn str_req<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or invalid {key}"))
}

fn usize_req(value: &Value, key: &str) -> Result<usize, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("missing or invalid {key}"))
}

fn f32_array(value: Option<&Value>, name: &str) -> Result<Vec<f32>, String> {
    let arr = value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing {name}"))?;
    arr.iter()
        .map(|v| {
            let value =
                v.as_f64()
                    .ok_or_else(|| format!("{name} contains non-number"))? as f32;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(format!("{name} contains non-finite value"))
            }
        })
        .collect()
}
