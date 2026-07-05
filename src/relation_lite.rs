//! Experimental Relation Factor Lite sidecar evaluator.
//!
//! Disabled unless `NORU_RELATION_LITE_SIDECAR` points at a JSON sidecar
//! produced by noru-tactic's `relation-lite-probe`.

use crate::board::{BOARD_SIZE, Board, NUM_CELLS, Stone};
use crate::vct::{THREAT_KIND_COUNT, ThreatKind, classify_move_fast};
use serde_json::Value;
use std::sync::OnceLock;

const FORMAT: &str = "noru-relation-lite-sidecar-v1";
const FEATURE_COUNT: usize = 94;

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

const FORCING_MASK: u8 = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6);

#[derive(Clone)]
struct Sidecar {
    mean: Vec<f32>,
    std: Vec<f32>,
    weights: Vec<f32>,
    bias: f32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum SidecarMode {
    Off,
    Leaf,
    Root,
    Both,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum LeafGateMode {
    All,
    Tactical,
    Urgent,
}

#[derive(Clone, Copy)]
struct RuntimeConfig {
    mode: SidecarMode,
    leaf_gate: LeafGateMode,
    output_scale: f32,
    blend: f32,
    root_margin: i32,
    root_min_ply: usize,
    root_max_center_dist: Option<usize>,
}

impl SidecarMode {
    #[inline]
    fn leaf_enabled(self) -> bool {
        matches!(self, Self::Leaf | Self::Both)
    }

    #[inline]
    fn root_enabled(self) -> bool {
        matches!(self, Self::Root | Self::Both)
    }
}

pub(crate) fn apply_sidecar(board: &Board, base_eval: i32) -> i32 {
    let config = runtime_config();
    if !config.mode.leaf_enabled() {
        return base_eval;
    }
    if !leaf_gate_allows(board, config.leaf_gate) {
        return base_eval;
    }
    let Some(sidecar) = sidecar() else {
        return base_eval;
    };
    let scaled = scaled_prediction(sidecar, board, base_eval, config.output_scale);
    let blend = config.blend;
    ((base_eval as f32) * (1.0 - blend) + (scaled as f32) * blend).round() as i32
}

pub(crate) fn root_enabled() -> bool {
    runtime_config().mode.root_enabled()
}

pub(crate) fn root_candidate_eval(board: &Board, base_eval: i32) -> Option<i32> {
    let config = runtime_config();
    if !config.mode.root_enabled() {
        return None;
    }
    sidecar().map(|sidecar| scaled_prediction(sidecar, board, base_eval, config.output_scale))
}

pub(crate) fn root_margin() -> i32 {
    runtime_config().root_margin
}

pub(crate) fn root_min_ply() -> usize {
    runtime_config().root_min_ply
}

pub(crate) fn root_move_allowed(mv: usize) -> bool {
    let Some(max_dist) = runtime_config().root_max_center_dist else {
        return true;
    };
    let row = mv / BOARD_SIZE;
    let col = mv % BOARD_SIZE;
    let center = BOARD_SIZE / 2;
    row.abs_diff(center) + col.abs_diff(center) <= max_dist
}

fn sidecar() -> Option<&'static Sidecar> {
    static SIDECAR: OnceLock<Option<Sidecar>> = OnceLock::new();
    SIDECAR.get_or_init(load_sidecar).as_ref()
}

fn runtime_config() -> &'static RuntimeConfig {
    static CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();
    CONFIG.get_or_init(load_runtime_config)
}

fn load_runtime_config() -> RuntimeConfig {
    RuntimeConfig {
        mode: parse_sidecar_mode(),
        leaf_gate: parse_leaf_gate_mode(),
        output_scale: parse_env_f32("NORU_RELATION_LITE_SCALE")
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(32.0),
        blend: parse_env_f32("NORU_RELATION_LITE_BLEND")
            .filter(|v| v.is_finite())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0),
        root_margin: parse_env_i32("NORU_RELATION_LITE_ROOT_MARGIN")
            .filter(|v| *v >= 0)
            .unwrap_or(50),
        root_min_ply: parse_env_usize("NORU_RELATION_LITE_ROOT_MIN_PLY")
            .filter(|v| *v > 0)
            .unwrap_or(0),
        root_max_center_dist: parse_env_usize("NORU_RELATION_LITE_ROOT_MAX_CENTER_DIST"),
    }
}

fn parse_sidecar_mode() -> SidecarMode {
    let Ok(raw) = std::env::var("NORU_RELATION_LITE_MODE") else {
        return SidecarMode::Leaf;
    };
    let trimmed = raw.trim();
    if is_disabled_value(trimmed) {
        return SidecarMode::Off;
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "leaf" | "eval" => SidecarMode::Leaf,
        "root" | "rerank" | "tiebreak" | "tie-break" => SidecarMode::Root,
        "both" | "all" => SidecarMode::Both,
        other => {
            panic!("invalid NORU_RELATION_LITE_MODE={other:?}; expected leaf, root, both, or off")
        }
    }
}

fn parse_leaf_gate_mode() -> LeafGateMode {
    let Ok(raw) = std::env::var("NORU_RELATION_LITE_LEAF_GATE") else {
        return LeafGateMode::All;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("all") {
        return LeafGateMode::All;
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "tactical" | "tactic" | "risk" | "risky" => LeafGateMode::Tactical,
        "urgent" | "forcing" | "forced" => LeafGateMode::Urgent,
        other => panic!(
            "invalid NORU_RELATION_LITE_LEAF_GATE={other:?}; expected all, tactical, or urgent"
        ),
    }
}

fn leaf_gate_allows(board: &Board, mode: LeafGateMode) -> bool {
    match mode {
        LeafGateMode::All => true,
        LeafGateMode::Tactical => has_tactical_leaf_signal(board),
        LeafGateMode::Urgent => has_urgent_leaf_signal(board),
    }
}

fn load_sidecar() -> Option<Sidecar> {
    let Ok(path) = std::env::var("NORU_RELATION_LITE_SIDECAR") else {
        return None;
    };
    let trimmed = path.trim();
    if is_disabled_value(trimmed) {
        return None;
    }
    let text = std::fs::read_to_string(trimmed)
        .unwrap_or_else(|e| panic!("failed to read NORU_RELATION_LITE_SIDECAR={trimmed}: {e}"));
    let value: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse relation-lite sidecar {trimmed}: {e}"));
    parse_sidecar(&value).unwrap_or_else(|e| panic!("invalid relation-lite sidecar {trimmed}: {e}"))
}

fn parse_sidecar(value: &Value) -> Result<Option<Sidecar>, String> {
    let format = value
        .get("format")
        .and_then(Value::as_str)
        .ok_or("missing format")?;
    if format != FORMAT {
        return Err(format!("unsupported format {format:?}"));
    }
    let feature_count = value
        .pointer("/features/count")
        .and_then(Value::as_u64)
        .ok_or("missing features.count")? as usize;
    if feature_count != FEATURE_COUNT {
        return Err(format!(
            "feature count mismatch: expected {FEATURE_COUNT}, got {feature_count}"
        ));
    }

    let mean = f32_array(
        value.pointer("/features/standardizer/mean"),
        "features.standardizer.mean",
    )?;
    let std = f32_array(
        value.pointer("/features/standardizer/std"),
        "features.standardizer.std",
    )?;
    let weights = f32_array(value.pointer("/linear/weights"), "linear.weights")?;
    let bias = value
        .pointer("/linear/bias")
        .and_then(Value::as_f64)
        .ok_or("missing linear.bias")? as f32;

    if mean.len() != FEATURE_COUNT || std.len() != FEATURE_COUNT || weights.len() != FEATURE_COUNT {
        return Err(format!(
            "length mismatch: mean={} std={} weights={} expected={FEATURE_COUNT}",
            mean.len(),
            std.len(),
            weights.len()
        ));
    }
    if std.iter().any(|v| !v.is_finite() || *v == 0.0) {
        return Err("standardizer std contains zero or non-finite value".to_string());
    }
    if weights.iter().any(|v| !v.is_finite()) || !bias.is_finite() {
        return Err("linear weights contain non-finite value".to_string());
    }

    Ok(Some(Sidecar {
        mean,
        std,
        weights,
        bias,
    }))
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

impl Sidecar {
    fn predict(&self, board: &Board, base_eval: i32) -> f32 {
        let mut features = Vec::with_capacity(FEATURE_COUNT);
        features.push(base_eval as f32);
        push_relation_features(board, &mut features);
        debug_assert_eq!(features.len(), FEATURE_COUNT);

        let mut out = self.bias;
        for (i, raw) in features.iter().enumerate() {
            let z = (*raw - self.mean[i]) / self.std[i];
            out += self.weights[i] * z;
        }
        out
    }
}

fn scaled_prediction(sidecar: &Sidecar, board: &Board, base_eval: i32, output_scale: f32) -> i32 {
    (sidecar.predict(board, base_eval) * output_scale).round() as i32
}

fn push_relation_features(board: &Board, features: &mut Vec<f32>) {
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
        let weak = matches!(attack, ThreatKind::ClosedFour | ThreatKind::OpenThree)
            && block == ThreatKind::None;
        if weak {
            weak_attack += 1;
            match attack {
                ThreatKind::ClosedFour => weak_closed_four += 1,
                ThreatKind::OpenThree => weak_open_three += 1,
                _ => {}
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

fn has_tactical_leaf_signal(board: &Board) -> bool {
    let side = board.side_to_move;
    let opp = side.opponent();
    for mv in board.candidate_moves() {
        if !board.is_empty(mv) {
            continue;
        }
        let attack = classify_move_fast(board, mv, side);
        let block = classify_move_fast(board, mv, opp);
        if is_forcing_kind(attack) || is_forcing_kind(block) {
            return true;
        }
        if matches!(attack, ThreatKind::ClosedFour | ThreatKind::OpenThree)
            && block == ThreatKind::None
        {
            return true;
        }
    }
    false
}

fn has_urgent_leaf_signal(board: &Board) -> bool {
    let side = board.side_to_move;
    let opp = side.opponent();
    for mv in board.candidate_moves() {
        if !board.is_empty(mv) {
            continue;
        }
        let attack = classify_move_fast(board, mv, side);
        let block = classify_move_fast(board, mv, opp);
        if matches!(
            attack,
            ThreatKind::Five
                | ThreatKind::OpenFour
                | ThreatKind::DoubleFour
                | ThreatKind::FourThree
        ) || matches!(
            block,
            ThreatKind::Five
                | ThreatKind::OpenFour
                | ThreatKind::DoubleFour
                | ThreatKind::FourThree
                | ThreatKind::DoubleThree
        ) {
            return true;
        }
    }
    false
}

#[inline]
fn is_forcing_kind(kind: ThreatKind) -> bool {
    (FORCING_MASK >> (kind as u8)) & 1 != 0
}

fn parse_env_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
}

fn parse_env_i32(key: &str) -> Option<i32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<i32>().ok())
}

fn parse_env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
}

fn is_disabled_value(value: &str) -> bool {
    value.is_empty()
        || value == "0"
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("false")
}
