//! Experimental root candidate ranker.
//!
//! Disabled unless `NORU_CANDIDATE_RANKER` points at a JSON file produced by
//! noru-tactic's `candidate-rank-probe`.

use crate::board::{to_idx, to_rc, Board, Move, Stone, BOARD_SIZE};
use crate::eval::evaluate_base;
use crate::heuristic::{scan_line, DIR};
use crate::vct::{classify_move_fast, ThreatKind, THREAT_KIND_COUNT};
use noru::network::NnueWeights;
use serde_json::Value;
use std::sync::OnceLock;

const FORMAT: &str = "noru-candidate-ranker-v1";
const FEATURE_COUNT: usize = 95;

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
const EVAL_SCALE: f32 = 200.0;
const SCORE_SCALE: f32 = 100_000.0;
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

#[derive(Clone)]
struct CandidateRanker {
    mean: Vec<f32>,
    std: Vec<f32>,
    weights: Vec<f32>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum RootGateMode {
    None,
    Tactical,
    Strict,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) struct RootGateKey {
    attack: ThreatKind,
    block: ThreatKind,
}

pub(crate) fn root_enabled() -> bool {
    ranker().is_some()
}

pub(crate) fn root_margin() -> i32 {
    static MARGIN: OnceLock<i32> = OnceLock::new();
    *MARGIN.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_RANKER_ROOT_MARGIN")
            .filter(|v| *v >= 0)
            .unwrap_or(0)
    })
}

pub(crate) fn root_gate_mode() -> RootGateMode {
    static MODE: OnceLock<RootGateMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let Ok(raw) = std::env::var("NORU_CANDIDATE_RANKER_ROOT_GATE") else {
            return RootGateMode::None;
        };
        let trimmed = raw.trim();
        if is_disabled_value(trimmed) || trimmed.eq_ignore_ascii_case("none") {
            return RootGateMode::None;
        }
        match trimmed.to_ascii_lowercase().as_str() {
            "tactical" | "nonquiet" | "non-quiet" => RootGateMode::Tactical,
            "strict" | "same-threat" | "same_threat" => RootGateMode::Strict,
            other => panic!(
                "invalid NORU_CANDIDATE_RANKER_ROOT_GATE={other:?}; expected none, tactical, or strict"
            ),
        }
    })
}

pub(crate) fn root_gate_key(board: &Board, mv: Move) -> Option<RootGateKey> {
    let attack = classify_move_fast(board, mv, board.side_to_move);
    let block = classify_move_fast(board, mv, board.side_to_move.opponent());
    if attack == ThreatKind::None && block == ThreatKind::None {
        None
    } else {
        Some(RootGateKey { attack, block })
    }
}

pub(crate) fn gate_allows(
    mode: RootGateMode,
    candidate: Option<RootGateKey>,
    incumbent: Option<RootGateKey>,
) -> bool {
    match mode {
        RootGateMode::None => true,
        RootGateMode::Tactical => candidate.is_some() && incumbent.is_some(),
        RootGateMode::Strict => candidate.is_some() && candidate == incumbent,
    }
}

pub(crate) fn score_prefers(candidate: Option<i32>, incumbent: Option<i32>) -> bool {
    match (candidate, incumbent) {
        (Some(candidate), Some(incumbent)) => candidate > incumbent,
        (Some(_), None) => true,
        _ => false,
    }
}

pub(crate) fn root_candidate_score(board: &Board, mv: Move, weights: &NnueWeights) -> Option<i32> {
    let ranker = ranker()?;
    let features = candidate_features(board, mv, weights);
    Some((ranker.score(&features) * SCORE_SCALE).round() as i32)
}

impl CandidateRanker {
    fn score(&self, features: &[f32]) -> f32 {
        self.weights
            .iter()
            .zip(features)
            .zip(self.mean.iter().zip(&self.std))
            .map(|((w, x), (mean, std))| w * ((x - mean) / std))
            .sum()
    }
}

fn ranker() -> Option<&'static CandidateRanker> {
    static RANKER: OnceLock<Option<CandidateRanker>> = OnceLock::new();
    RANKER.get_or_init(load_ranker).as_ref()
}

fn load_ranker() -> Option<CandidateRanker> {
    let Ok(path) = std::env::var("NORU_CANDIDATE_RANKER") else {
        return None;
    };
    let trimmed = path.trim();
    if is_disabled_value(trimmed) {
        return None;
    }
    let text = std::fs::read_to_string(trimmed)
        .unwrap_or_else(|e| panic!("failed to read NORU_CANDIDATE_RANKER={trimmed}: {e}"));
    let value: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse candidate ranker {trimmed}: {e}"));
    parse_ranker(&value).unwrap_or_else(|e| panic!("invalid candidate ranker {trimmed}: {e}"))
}

fn parse_ranker(value: &Value) -> Result<Option<CandidateRanker>, String> {
    let model = value.get("ranker_model").unwrap_or(value);
    let format = model
        .get("format")
        .and_then(Value::as_str)
        .ok_or("missing ranker_model.format")?;
    if format != FORMAT {
        return Err(format!("unsupported format {format:?}"));
    }

    let feature_names = str_array(model.get("feature_names"), "ranker_model.feature_names")?;
    let mean = f32_array(model.get("mean"), "ranker_model.mean")?;
    let std = f32_array(model.get("std"), "ranker_model.std")?;
    let weights = f32_array(model.get("weights"), "ranker_model.weights")?;

    if feature_names.len() != FEATURE_COUNT
        || mean.len() != FEATURE_COUNT
        || std.len() != FEATURE_COUNT
        || weights.len() != FEATURE_COUNT
    {
        return Err(format!(
            "length mismatch: names={} mean={} std={} weights={} expected={FEATURE_COUNT}",
            feature_names.len(),
            mean.len(),
            std.len(),
            weights.len()
        ));
    }
    if feature_names.first().map(String::as_str) != Some("child_root_eval") {
        return Err("unexpected feature order: first feature is not child_root_eval".to_string());
    }
    if std.iter().any(|v| !v.is_finite() || *v == 0.0) {
        return Err("standardizer std contains zero or non-finite value".to_string());
    }
    if weights.iter().any(|v| !v.is_finite()) {
        return Err("ranker weights contain non-finite value".to_string());
    }

    Ok(Some(CandidateRanker { mean, std, weights }))
}

fn candidate_features(board: &Board, mv: Move, weights: &NnueWeights) -> Vec<f32> {
    let side = board.side_to_move;
    let opp = side.opponent();
    let attack = classify_move_fast(board, mv, side);
    let block = classify_move_fast(board, mv, opp);
    let attack_tier = MOVE_ATTACK_TABLE[attack as usize];
    let block_tier = MOVE_BLOCK_TABLE[block as usize];
    let order_score = move_order_score(board, mv, attack, block);
    let root_eval = child_root_eval(board, mv, weights);
    let (row, col) = to_rc(mv);
    let center = (BOARD_SIZE as f32 - 1.0) * 0.5;
    let dr = row as f32 - center;
    let dc = col as f32 - center;
    let center_dist = (dr * dr + dc * dc).sqrt() / center.max(1.0);
    let last_dist = board
        .last_move
        .map(|last| {
            let (lr, lc) = to_rc(last);
            let dr = row as f32 - lr as f32;
            let dc = col as f32 - lc as f32;
            (dr * dr + dc * dc).sqrt() / BOARD_SIZE as f32
        })
        .unwrap_or(1.0);
    let (my_r1, opp_r1) = neighbor_counts(board, mv, 1);
    let (my_r2, opp_r2) = neighbor_counts(board, mv, 2);

    let mut features = Vec::with_capacity(FEATURE_COUNT);
    features.push(root_eval / EVAL_SCALE);
    features.push(order_score as f32 / TIER_SCALE);
    features.push(attack_tier as f32 / TIER_SCALE);
    features.push(block_tier as f32 / TIER_SCALE);
    features.push((attack_tier - block_tier) as f32 / TIER_SCALE);
    features.push(if is_forcing_kind(attack) { 1.0 } else { 0.0 });
    features.push(if is_forcing_kind(block) { 1.0 } else { 0.0 });
    features.push(if attack == ThreatKind::None && block == ThreatKind::None {
        1.0
    } else {
        0.0
    });
    features.push(
        if matches!(attack, ThreatKind::ClosedFour | ThreatKind::OpenThree)
            && block == ThreatKind::None
        {
            1.0
        } else {
            0.0
        },
    );
    features.push(center_dist);
    features.push(last_dist);
    features.push(my_r1 as f32 / 8.0);
    features.push(opp_r1 as f32 / 8.0);
    features.push(my_r2 as f32 / 24.0);
    features.push(opp_r2 as f32 / 24.0);
    for kind in threat_kinds() {
        features.push(if attack == kind { 1.0 } else { 0.0 });
    }
    for kind in threat_kinds() {
        features.push(if block == kind { 1.0 } else { 0.0 });
    }
    for a in threat_kinds() {
        for b in threat_kinds() {
            features.push(if attack == a && block == b { 1.0 } else { 0.0 });
        }
    }
    features
}

fn child_root_eval(board: &Board, mv: Move, weights: &NnueWeights) -> f32 {
    let mut child = board.clone();
    child.make_move(mv);
    -evaluate_base(&child, weights) as f32
}

fn move_order_score(board: &Board, mv: Move, attack: ThreatKind, block: ThreatKind) -> i32 {
    let attack_tier = MOVE_ATTACK_TABLE[attack as usize];
    let block_tier = MOVE_BLOCK_TABLE[block as usize];
    let mut score = attack_tier.max(block_tier);
    if attack == ThreatKind::Five {
        return TIER_WIN;
    }
    if block == ThreatKind::Five {
        return TIER_BLOCK_WIN;
    }
    let row = (mv / BOARD_SIZE) as i32;
    let col = (mv % BOARD_SIZE) as i32;
    let (my, opp) = match board.side_to_move {
        Stone::Black => (&board.black, &board.white),
        Stone::White => (&board.white, &board.black),
    };
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
    score
}

fn neighbor_counts(board: &Board, mv: Move, radius: i32) -> (usize, usize) {
    let (row, col) = to_rc(mv);
    let row = row as i32;
    let col = col as i32;
    let (my, opp) = match board.side_to_move {
        Stone::Black => (&board.black, &board.white),
        Stone::White => (&board.white, &board.black),
    };
    let mut my_count = 0;
    let mut opp_count = 0;
    for dr in -radius..=radius {
        for dc in -radius..=radius {
            if dr == 0 && dc == 0 {
                continue;
            }
            let r = row + dr;
            let c = col + dc;
            if r < 0 || c < 0 || r >= BOARD_SIZE as i32 || c >= BOARD_SIZE as i32 {
                continue;
            }
            let idx = to_idx(r as usize, c as usize);
            if my.get(idx) {
                my_count += 1;
            }
            if opp.get(idx) {
                opp_count += 1;
            }
        }
    }
    (my_count, opp_count)
}

fn is_forcing_kind(kind: ThreatKind) -> bool {
    (FORCING_MASK >> (kind as u8)) & 1 != 0
}

fn threat_kinds() -> [ThreatKind; THREAT_KIND_COUNT] {
    [
        ThreatKind::None,
        ThreatKind::ClosedFour,
        ThreatKind::OpenThree,
        ThreatKind::Five,
        ThreatKind::OpenFour,
        ThreatKind::DoubleFour,
        ThreatKind::FourThree,
        ThreatKind::DoubleThree,
    ]
}

fn f32_array(value: Option<&Value>, name: &str) -> Result<Vec<f32>, String> {
    let arr = value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing {name}"))?;
    arr.iter()
        .map(|v| {
            v.as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| format!("{name} contains non-number"))
        })
        .collect()
}

fn str_array(value: Option<&Value>, name: &str) -> Result<Vec<String>, String> {
    let arr = value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing {name}"))?;
    arr.iter()
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("{name} contains non-string"))
        })
        .collect()
}

fn parse_env_i32(name: &str) -> Option<i32> {
    let raw = std::env::var(name).ok()?;
    raw.trim().parse::<i32>().ok()
}

fn is_disabled_value(value: &str) -> bool {
    value.is_empty()
        || value.eq_ignore_ascii_case("0")
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("no")
}
