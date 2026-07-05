//! Experimental root candidate ranker.
//!
//! Disabled unless `NORU_CANDIDATE_RANKER` points at a JSON file produced by
//! noru-tactic's `candidate-rank-probe`.

use crate::board::{BOARD_SIZE, Board, Move, NUM_CELLS, Stone, to_idx, to_rc};
use crate::eval::evaluate_base;
use crate::heuristic::{DIR, scan_line};
use crate::vct::{THREAT_KIND_COUNT, ThreatKind, classify_move_fast};
use noru::network::NnueWeights;
use serde_json::Value;
use std::sync::OnceLock;

const FORMAT_V1: &str = "noru-candidate-ranker-v1";
const FORMAT_V2: &str = "noru-candidate-ranker-v2";
const FORMAT_V3: &str = "noru-candidate-ranker-v3";
const MULTI_HEAD_FORMAT: &str = "noru-multi-head-candidate-v0";
const DEF_REL_FORMAT: &str = "noru-defensive-relation-sidecar-v1";
const BASE_FEATURE_COUNT: usize = 95;
const RELATION_CHILD_FEATURE_COUNT: usize = 188;
const RQ36_RICH_FEATURE_COUNT: usize = 118;
const DEF_REL_FEATURE_COUNT: usize = 25;
const RQ15_REPLY_FEATURE_COUNT: usize = 25;

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
    feature_set: FeatureSet,
    model: RankerModel,
}

#[derive(Clone)]
struct DefensiveRelationSidecar {
    relation_mean: Vec<f32>,
    relation_std: Vec<f32>,
    relation_weights: Vec<f32>,
    relation_bias: f32,
    reference_mean: Vec<f32>,
    reference_std: Vec<f32>,
    reference_weights: Vec<f32>,
    alpha: f32,
}

#[derive(Clone)]
enum RankerModel {
    Linear {
        weights: Vec<f32>,
        bias: f32,
    },
    Mlp {
        w1: Vec<Vec<f32>>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: f32,
    },
    MultiHead {
        w1: Vec<f32>,
        b1: Vec<f32>,
        policy: MultiHeadHead,
        relation: MultiHeadHead,
        value: MultiHeadHead,
        selector: MultiHeadSelector,
    },
}

#[derive(Clone)]
struct MultiHeadHead {
    weights: Vec<f32>,
    input_skip_weights: Vec<f32>,
    bias: f32,
}

#[derive(Clone, Copy)]
enum MultiHeadSelector {
    Policy,
    Relation,
    Value,
    PolicyPlusRelation(f32),
    PolicyPlusValue(f32),
    PolicyPlusRelationValue(f32, f32),
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum FeatureSet {
    Base,
    RelationChild,
    Rq15Reply,
    Rq36ReplyRich,
}

impl FeatureSet {
    fn feature_count(self) -> usize {
        match self {
            Self::Base => BASE_FEATURE_COUNT,
            Self::RelationChild => RELATION_CHILD_FEATURE_COUNT,
            Self::Rq15Reply => RQ15_REPLY_FEATURE_COUNT,
            Self::Rq36ReplyRich => RQ36_RICH_FEATURE_COUNT,
        }
    }
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

pub(crate) fn root_score_enabled_for(board: &Board) -> bool {
    (ranker().is_some() && ranker_applies(board))
        || (defensive_relation_sidecar().is_some() && defensive_relation_applies(board))
}

pub(crate) fn root_margin() -> i32 {
    static MARGIN: OnceLock<i32> = OnceLock::new();
    *MARGIN.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_RANKER_ROOT_MARGIN")
            .filter(|v| *v >= 0)
            .unwrap_or(0)
    })
}

pub(crate) fn root_score_margin() -> i32 {
    static MARGIN: OnceLock<i32> = OnceLock::new();
    *MARGIN.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_RANKER_ROOT_SCORE_MARGIN")
            .filter(|v| *v >= 0)
            .unwrap_or(0)
    })
}

pub(crate) fn root_gate_mode() -> RootGateMode {
    static MODE: OnceLock<RootGateMode> = OnceLock::new();
    *MODE.get_or_init(|| parse_gate_mode_env("NORU_CANDIDATE_RANKER_ROOT_GATE"))
}

pub(crate) fn order_gate_mode() -> RootGateMode {
    static MODE: OnceLock<RootGateMode> = OnceLock::new();
    *MODE.get_or_init(|| parse_gate_mode_env("NORU_CANDIDATE_RANKER_ORDER_GATE"))
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
    score_prefers_with_margin(candidate, incumbent, 0)
}

pub(crate) fn score_prefers_with_margin(
    candidate: Option<i32>,
    incumbent: Option<i32>,
    margin: i32,
) -> bool {
    match (candidate, incumbent) {
        (Some(candidate), Some(incumbent)) if margin > 0 => {
            candidate.saturating_sub(incumbent) >= margin
        }
        (Some(candidate), Some(incumbent)) => candidate > incumbent,
        (Some(_), None) => true,
        _ => false,
    }
}

pub(crate) fn root_candidate_score(board: &Board, mv: Move, weights: &NnueWeights) -> Option<i32> {
    let rank_score = if ranker_applies(board) {
        ranker().map(|ranker| {
            let features = candidate_features(board, mv, weights, ranker.feature_set);
            (ranker.score(&features) * SCORE_SCALE).round() as i32
        })
    } else {
        None
    };
    let Some(sidecar) = defensive_relation_sidecar() else {
        return rank_score;
    };
    if !defensive_relation_applies(board) {
        return rank_score;
    }
    let features = defensive_relation_features(board, mv);
    let relation_bonus =
        (sidecar.alpha * sidecar.relation_logit(&features) * SCORE_SCALE).round() as i32;
    let base = rank_score
        .unwrap_or_else(|| (sidecar.reference_score(&features) * SCORE_SCALE).round() as i32);
    Some(base.saturating_add(relation_bonus))
}

impl CandidateRanker {
    fn score(&self, features: &[f32]) -> f32 {
        let standardized = features
            .iter()
            .zip(self.mean.iter().zip(&self.std))
            .map(|(x, (mean, std))| (*x - mean) / std)
            .collect::<Vec<_>>();
        match &self.model {
            RankerModel::Linear { weights, bias } => *bias + dot(weights, &standardized),
            RankerModel::Mlp { w1, b1, w2, b2 } => {
                let hidden = w1
                    .iter()
                    .zip(b1)
                    .map(|(row, bias)| (dot(row, &standardized) + *bias).max(0.0))
                    .collect::<Vec<_>>();
                *b2 + dot(w2, &hidden)
            }
            RankerModel::MultiHead {
                w1,
                b1,
                policy,
                relation,
                value,
                selector,
            } => {
                let input = standardized.len();
                let hidden = b1
                    .iter()
                    .enumerate()
                    .map(|(row_idx, bias)| {
                        let start = row_idx * input;
                        (dot(&w1[start..start + input], &standardized) + *bias).max(0.0)
                    })
                    .collect::<Vec<_>>();
                let policy_score = policy.score(&standardized, &hidden);
                let relation_score = relation.score(&standardized, &hidden);
                let value_score = value.score(&standardized, &hidden);
                match *selector {
                    MultiHeadSelector::Policy => policy_score,
                    MultiHeadSelector::Relation => relation_score,
                    MultiHeadSelector::Value => value_score,
                    MultiHeadSelector::PolicyPlusRelation(alpha) => {
                        policy_score + alpha * relation_score
                    }
                    MultiHeadSelector::PolicyPlusValue(alpha) => policy_score + alpha * value_score,
                    MultiHeadSelector::PolicyPlusRelationValue(relation_alpha, value_alpha) => {
                        policy_score + relation_alpha * relation_score + value_alpha * value_score
                    }
                }
            }
        }
    }
}

impl MultiHeadHead {
    fn score(&self, input: &[f32], hidden: &[f32]) -> f32 {
        self.bias + dot(&self.weights, hidden) + dot(&self.input_skip_weights, input)
    }
}

fn ranker() -> Option<&'static CandidateRanker> {
    static RANKER: OnceLock<Option<CandidateRanker>> = OnceLock::new();
    RANKER.get_or_init(load_ranker).as_ref()
}

fn ranker_applies(board: &Board) -> bool {
    static WHITE_ONLY: OnceLock<bool> = OnceLock::new();
    let white_only = *WHITE_ONLY.get_or_init(|| parse_env_bool("NORU_CANDIDATE_RANKER_WHITE_ONLY"));
    !white_only || board.side_to_move == Stone::White
}

fn defensive_relation_sidecar() -> Option<&'static DefensiveRelationSidecar> {
    static SIDECAR: OnceLock<Option<DefensiveRelationSidecar>> = OnceLock::new();
    SIDECAR
        .get_or_init(load_defensive_relation_sidecar)
        .as_ref()
}

fn defensive_relation_applies(board: &Board) -> bool {
    static WHITE_ONLY: OnceLock<bool> = OnceLock::new();
    let white_only =
        *WHITE_ONLY.get_or_init(|| parse_env_bool_default("NORU_DEF_RELATION_WHITE_ONLY", true));
    !white_only || board.side_to_move == Stone::White
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

fn load_defensive_relation_sidecar() -> Option<DefensiveRelationSidecar> {
    let Ok(path) = std::env::var("NORU_DEF_RELATION_SIDECAR") else {
        return None;
    };
    let trimmed = path.trim();
    if is_disabled_value(trimmed) {
        return None;
    }
    let text = std::fs::read_to_string(trimmed)
        .unwrap_or_else(|e| panic!("failed to read NORU_DEF_RELATION_SIDECAR={trimmed}: {e}"));
    let value: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse defensive relation sidecar {trimmed}: {e}"));
    parse_defensive_relation_sidecar(&value)
        .unwrap_or_else(|e| panic!("invalid defensive relation sidecar {trimmed}: {e}"))
}

fn parse_ranker(value: &Value) -> Result<Option<CandidateRanker>, String> {
    let model = value.get("ranker_model").unwrap_or(value);
    let format = model
        .get("format")
        .and_then(Value::as_str)
        .ok_or("missing ranker_model.format")?;
    if format == MULTI_HEAD_FORMAT {
        return parse_multi_head_ranker(model);
    }
    let feature_set = match format {
        FORMAT_V1 => FeatureSet::Base,
        FORMAT_V2 | FORMAT_V3 => match model
            .get("feature_set")
            .and_then(Value::as_str)
            .unwrap_or("base")
        {
            "base" => FeatureSet::Base,
            "relation-child" => FeatureSet::RelationChild,
            "rq15_reply_v1" | "rq15-reply" => FeatureSet::Rq15Reply,
            other => return Err(format!("unsupported feature_set {other:?}")),
        },
        _ => {
            return Err(format!("unsupported format {format:?}"));
        }
    };
    let expected = feature_set.feature_count();
    if format == FORMAT_V1 && feature_set != FeatureSet::Base {
        return Err(format!("unsupported format {format:?}"));
    }

    let feature_names = str_array(model.get("feature_names"), "ranker_model.feature_names")?;
    let mean = f32_array(model.get("mean"), "ranker_model.mean")?;
    let std = f32_array(model.get("std"), "ranker_model.std")?;
    if feature_names.len() != expected || mean.len() != expected || std.len() != expected {
        return Err(format!(
            "length mismatch: names={} mean={} std={} expected={expected}",
            feature_names.len(),
            mean.len(),
            std.len()
        ));
    }
    let expected_first_feature = match feature_set {
        FeatureSet::Rq15Reply => "candidate_x_norm",
        _ => "child_root_eval",
    };
    if feature_names.first().map(String::as_str) != Some(expected_first_feature) {
        return Err(format!(
            "unexpected feature order: first feature is not {expected_first_feature}"
        ));
    }
    if std.iter().any(|v| !v.is_finite() || *v == 0.0) {
        return Err("standardizer std contains zero or non-finite value".to_string());
    }
    let model_kind = model
        .get("model_kind")
        .and_then(Value::as_str)
        .unwrap_or("listwise-linear-policy");
    let ranker_model = match model_kind {
        "listwise-linear-policy" => {
            let weights = f32_array(model.get("weights"), "ranker_model.weights")?;
            if weights.len() != expected {
                return Err(format!(
                    "linear weights length mismatch: weights={} expected={expected}",
                    weights.len()
                ));
            }
            let bias = model.get("bias").and_then(Value::as_f64).unwrap_or(0.0) as f32;
            RankerModel::Linear { weights, bias }
        }
        "listwise-mlp-policy" => {
            if format != FORMAT_V3 {
                return Err(format!("MLP ranker requires {FORMAT_V3}, got {format:?}"));
            }
            let w1 = f32_matrix(model.get("w1"), "ranker_model.w1")?;
            let b1 = f32_array(model.get("b1"), "ranker_model.b1")?;
            let w2 = f32_array(model.get("w2"), "ranker_model.w2")?;
            let b2 = model
                .get("b2")
                .and_then(Value::as_f64)
                .ok_or("missing ranker_model.b2")? as f32;
            if w1.is_empty() || w1.iter().any(|row| row.len() != expected) {
                return Err(format!("MLP w1 rows must have input width {expected}"));
            }
            if b1.len() != w1.len() || w2.len() != w1.len() {
                return Err(format!(
                    "MLP hidden length mismatch: w1={} b1={} w2={}",
                    w1.len(),
                    b1.len(),
                    w2.len()
                ));
            }
            if !b2.is_finite() {
                return Err("ranker_model.b2 is non-finite".to_string());
            }
            RankerModel::Mlp { w1, b1, w2, b2 }
        }
        other => return Err(format!("unsupported model_kind {other:?}")),
    };

    Ok(Some(CandidateRanker {
        mean,
        std,
        feature_set,
        model: ranker_model,
    }))
}

fn parse_multi_head_ranker(model: &Value) -> Result<Option<CandidateRanker>, String> {
    let feature_set = match model
        .get("feature_set")
        .and_then(Value::as_str)
        .ok_or("missing feature_set")?
    {
        "rq36_reply_rich_v1" => FeatureSet::Rq36ReplyRich,
        other => return Err(format!("unsupported multi-head feature_set {other:?}")),
    };
    let expected = feature_set.feature_count();
    let feature_count = model
        .get("feature_count")
        .and_then(Value::as_u64)
        .ok_or("missing feature_count")? as usize;
    if feature_count != expected {
        return Err(format!(
            "feature_count mismatch: got={feature_count} expected={expected}"
        ));
    }
    let feature_names = str_array(model.pointer("/features/names"), "features.names")?;
    let mean = f32_array(
        model.pointer("/features/standardizer/mean"),
        "features.standardizer.mean",
    )?;
    let std = f32_array(
        model.pointer("/features/standardizer/std"),
        "features.standardizer.std",
    )?;
    if feature_names.len() != expected || mean.len() != expected || std.len() != expected {
        return Err(format!(
            "multi-head length mismatch: names={} mean={} std={} expected={expected}",
            feature_names.len(),
            mean.len(),
            std.len()
        ));
    }
    if feature_names.first().map(String::as_str) != Some("candidate_x_norm") {
        return Err(
            "unexpected rich feature order: first feature is not candidate_x_norm".to_string(),
        );
    }
    if std.iter().any(|v| !v.is_finite() || *v == 0.0) {
        return Err("multi-head standardizer std contains zero or non-finite value".to_string());
    }

    let w1 = f32_array(model.pointer("/trunk/weights"), "trunk.weights")?;
    let b1 = f32_array(model.pointer("/trunk/bias"), "trunk.bias")?;
    if b1.is_empty() || w1.len() != b1.len() * expected {
        return Err(format!(
            "trunk shape mismatch: weights={} hidden={} input={expected}",
            w1.len(),
            b1.len()
        ));
    }
    let policy = parse_multi_head_head(model, "policy", expected, b1.len())?;
    let relation = parse_multi_head_head(model, "defensive_relation", expected, b1.len())?;
    let value = parse_multi_head_head(model, "candidate_value", expected, b1.len())?;
    let selector_name = std::env::var("NORU_MULTI_HEAD_SELECTOR")
        .ok()
        .filter(|raw| !raw.trim().is_empty())
        .or_else(|| {
            model
                .pointer("/training/selection/best_selector")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "policy_plus_value_a1".to_string());
    let selector = parse_multi_head_selector(&selector_name)?;

    Ok(Some(CandidateRanker {
        mean,
        std,
        feature_set,
        model: RankerModel::MultiHead {
            w1,
            b1,
            policy,
            relation,
            value,
            selector,
        },
    }))
}

fn parse_multi_head_head(
    model: &Value,
    name: &str,
    input: usize,
    hidden: usize,
) -> Result<MultiHeadHead, String> {
    let weights = f32_array(
        model.pointer(&format!("/heads/{name}/weights")),
        &format!("heads.{name}.weights"),
    )?;
    let input_skip_weights = f32_array(
        model.pointer(&format!("/heads/{name}/input_skip_weights")),
        &format!("heads.{name}.input_skip_weights"),
    )?;
    let bias = model
        .pointer(&format!("/heads/{name}/bias"))
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing heads.{name}.bias"))? as f32;
    if weights.len() != hidden || input_skip_weights.len() != input {
        return Err(format!(
            "head {name} shape mismatch: weights={} hidden={hidden} skip={} input={input}",
            weights.len(),
            input_skip_weights.len()
        ));
    }
    if !bias.is_finite()
        || weights.iter().any(|v| !v.is_finite())
        || input_skip_weights.iter().any(|v| !v.is_finite())
    {
        return Err(format!("head {name} contains non-finite value"));
    }
    Ok(MultiHeadHead {
        weights,
        input_skip_weights,
        bias,
    })
}

fn parse_multi_head_selector(raw: &str) -> Result<MultiHeadSelector, String> {
    let trimmed = raw.trim();
    match trimmed {
        "policy" => Ok(MultiHeadSelector::Policy),
        "defensive_relation" | "relation" => Ok(MultiHeadSelector::Relation),
        "candidate_value" | "value" => Ok(MultiHeadSelector::Value),
        _ if trimmed.starts_with("policy_plus_relation_a") && trimmed.contains("_value_a") => {
            let rest = trimmed
                .strip_prefix("policy_plus_relation_a")
                .ok_or_else(|| format!("invalid selector {trimmed:?}"))?;
            let (relation, value) = rest
                .split_once("_value_a")
                .ok_or_else(|| format!("invalid selector {trimmed:?}"))?;
            Ok(MultiHeadSelector::PolicyPlusRelationValue(
                parse_alpha_name(relation)?,
                parse_alpha_name(value)?,
            ))
        }
        _ if trimmed.starts_with("policy_plus_relation_a") => {
            let alpha = trimmed
                .strip_prefix("policy_plus_relation_a")
                .ok_or_else(|| format!("invalid selector {trimmed:?}"))?;
            Ok(MultiHeadSelector::PolicyPlusRelation(parse_alpha_name(
                alpha,
            )?))
        }
        _ if trimmed.starts_with("policy_plus_value_a") => {
            let alpha = trimmed
                .strip_prefix("policy_plus_value_a")
                .ok_or_else(|| format!("invalid selector {trimmed:?}"))?;
            Ok(MultiHeadSelector::PolicyPlusValue(parse_alpha_name(alpha)?))
        }
        _ => Err(format!("unsupported multi-head selector {trimmed:?}")),
    }
}

fn parse_alpha_name(raw: &str) -> Result<f32, String> {
    raw.replace('p', ".")
        .parse::<f32>()
        .map_err(|_| format!("invalid selector alpha {raw:?}"))
}

fn parse_defensive_relation_sidecar(
    value: &Value,
) -> Result<Option<DefensiveRelationSidecar>, String> {
    let format = value
        .get("format")
        .and_then(Value::as_str)
        .ok_or("missing format")?;
    if format != DEF_REL_FORMAT {
        return Err(format!("unsupported format {format:?}"));
    }
    let feature_count = value
        .get("feature_count")
        .and_then(Value::as_u64)
        .ok_or("missing feature_count")? as usize;
    if feature_count != DEF_REL_FEATURE_COUNT {
        return Err(format!(
            "feature count mismatch: expected {DEF_REL_FEATURE_COUNT}, got {feature_count}"
        ));
    }
    let feature_set = value
        .get("feature_set")
        .and_then(Value::as_str)
        .ok_or("missing feature_set")?;
    if feature_set != "rq15_reply_v1" {
        return Err(format!("unsupported feature_set {feature_set:?}"));
    }

    let relation_mean = f32_array(
        value.pointer("/features/standardizer/mean"),
        "features.standardizer.mean",
    )?;
    let relation_std = f32_array(
        value.pointer("/features/standardizer/std"),
        "features.standardizer.std",
    )?;
    let relation_weights = f32_array(
        value.pointer("/heads/defensive_relation/weights"),
        "heads.defensive_relation.weights",
    )?;
    let relation_bias = value
        .pointer("/heads/defensive_relation/bias")
        .and_then(Value::as_f64)
        .ok_or("missing heads.defensive_relation.bias")? as f32;
    let reference_mean = f32_array(
        value.pointer("/heads/reference_reply_policy/standardizer/mean"),
        "heads.reference_reply_policy.standardizer.mean",
    )?;
    let reference_std = f32_array(
        value.pointer("/heads/reference_reply_policy/standardizer/std"),
        "heads.reference_reply_policy.standardizer.std",
    )?;
    let reference_weights = f32_array(
        value.pointer("/heads/reference_reply_policy/weights"),
        "heads.reference_reply_policy.weights",
    )?;

    for (name, values) in [
        ("relation_mean", &relation_mean),
        ("relation_std", &relation_std),
        ("relation_weights", &relation_weights),
        ("reference_mean", &reference_mean),
        ("reference_std", &reference_std),
        ("reference_weights", &reference_weights),
    ] {
        if values.len() != DEF_REL_FEATURE_COUNT {
            return Err(format!(
                "{name} length mismatch: got={} expected={DEF_REL_FEATURE_COUNT}",
                values.len()
            ));
        }
    }
    if relation_std.iter().any(|v| !v.is_finite() || *v == 0.0)
        || reference_std.iter().any(|v| !v.is_finite() || *v == 0.0)
    {
        return Err("standardizer std contains zero or non-finite value".to_string());
    }
    if relation_weights.iter().any(|v| !v.is_finite())
        || reference_weights.iter().any(|v| !v.is_finite())
        || !relation_bias.is_finite()
    {
        return Err("linear weights contain non-finite value".to_string());
    }

    let sidecar_alpha = value
        .pointer("/recommended/blend_alpha")
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(0.5);
    let alpha = parse_env_f32("NORU_DEF_RELATION_ALPHA")
        .filter(|v| v.is_finite())
        .unwrap_or(sidecar_alpha);

    Ok(Some(DefensiveRelationSidecar {
        relation_mean,
        relation_std,
        relation_weights,
        relation_bias,
        reference_mean,
        reference_std,
        reference_weights,
        alpha,
    }))
}

fn candidate_features(
    board: &Board,
    mv: Move,
    weights: &NnueWeights,
    feature_set: FeatureSet,
) -> Vec<f32> {
    if feature_set == FeatureSet::Rq15Reply {
        return defensive_relation_features(board, mv);
    }
    if feature_set == FeatureSet::Rq36ReplyRich {
        return rq36_reply_rich_features(board, mv);
    }
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

    let mut features = Vec::with_capacity(feature_set.feature_count());
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
    if feature_set == FeatureSet::RelationChild {
        let mut child = board.clone();
        child.make_move(mv);
        push_relation_features(&child, &mut features);
    }
    debug_assert_eq!(features.len(), feature_set.feature_count());
    features
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
        if is_weak_attack(attack, block) {
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

impl DefensiveRelationSidecar {
    fn relation_logit(&self, features: &[f32]) -> f32 {
        self.relation_bias
            + standardized_dot(
                features,
                &self.relation_mean,
                &self.relation_std,
                &self.relation_weights,
            )
    }

    fn reference_score(&self, features: &[f32]) -> f32 {
        standardized_dot(
            features,
            &self.reference_mean,
            &self.reference_std,
            &self.reference_weights,
        )
    }
}

fn defensive_relation_features(board: &Board, mv: Move) -> Vec<f32> {
    let (row, col) = to_rc(mv);
    let x = col as f32;
    let y = row as f32;
    let center = (BOARD_SIZE as f32 - 1.0) * 0.5;
    let dx = (x - center) / center;
    let dy = (y - center) / center;
    let edge = row
        .min(col)
        .min(BOARD_SIZE - 1 - row)
        .min(BOARD_SIZE - 1 - col) as f32
        / center;
    let candidate_count = board.candidate_moves().len() as f32;
    let mut features = Vec::with_capacity(DEF_REL_FEATURE_COUNT);
    features.push(x / (BOARD_SIZE as f32 - 1.0));
    features.push(y / (BOARD_SIZE as f32 - 1.0));
    features.push((dx * dx + dy * dy).sqrt());
    features.push(edge);
    features.push(board.move_count as f32 / NUM_CELLS as f32);
    features.push(candidate_count / NUM_CELLS as f32);
    push_rq15_kind_one_hot(
        &mut features,
        rq15_threat_bin(board, mv, board.side_to_move),
    );

    let mut child = board.clone();
    child.make_move(mv);
    let opponent = child.side_to_move;
    let immediate = (0..NUM_CELLS)
        .filter(|&cell| child.is_empty(cell) && rq15_threat_bin(&child, cell, opponent) == 7)
        .count();
    let open_four = (0..NUM_CELLS)
        .filter(|&cell| {
            if !child.is_empty(cell) {
                return false;
            }
            let bin = rq15_threat_bin(&child, cell, opponent);
            bin == 5 || bin == 6
        })
        .count();
    let opponent_candidates = child.candidate_moves();
    let denom = opponent_candidates.len().max(1) as f32;
    let mut kind_counts = [0usize; 8];
    for reply in opponent_candidates.iter().copied() {
        if child.is_empty(reply) {
            kind_counts[rq15_threat_bin(&child, reply, opponent)] += 1;
        }
    }
    features.push(immediate.min(4) as f32 / 4.0);
    features.push(open_four.min(8) as f32 / 8.0);
    features.push(opponent_candidates.len() as f32 / NUM_CELLS as f32);
    for count in kind_counts {
        features.push(count as f32 / denom);
    }
    debug_assert_eq!(features.len(), DEF_REL_FEATURE_COUNT);
    features
}

fn rq36_reply_rich_features(board: &Board, mv: Move) -> Vec<f32> {
    let side = board.side_to_move;
    let opp = side.opponent();
    let attack_bin = rq15_threat_bin(board, mv, side);
    let block_bin = rq15_threat_bin(board, mv, opp);
    let attack_tier = rq36_attack_tier(attack_bin);
    let block_tier = rq36_block_tier(block_bin);
    let order_score = rq36_move_order_score(board, mv, attack_bin, block_bin);
    let last_dist = board
        .last_move
        .map(|last| {
            let (row, col) = to_rc(mv);
            let (lr, lc) = to_rc(last);
            let dr = row as f32 - lr as f32;
            let dc = col as f32 - lc as f32;
            (dr * dr + dc * dc).sqrt() / BOARD_SIZE as f32
        })
        .unwrap_or(1.0);
    let (my_r1, opp_r1) = neighbor_counts(board, mv, 1);
    let (my_r2, opp_r2) = neighbor_counts(board, mv, 2);

    let mut features = defensive_relation_features(board, mv);
    features.push(order_score as f32 / TIER_SCALE);
    features.push(attack_tier as f32 / TIER_SCALE);
    features.push(block_tier as f32 / TIER_SCALE);
    features.push((attack_tier - block_tier) as f32 / TIER_SCALE);
    features.push(if rq36_forcing_bin(attack_bin) {
        1.0
    } else {
        0.0
    });
    features.push(if rq36_forcing_bin(block_bin) {
        1.0
    } else {
        0.0
    });
    features.push(if attack_bin == 0 && block_bin == 0 {
        1.0
    } else {
        0.0
    });
    features.push(if rq36_weak_attack_bin(attack_bin, block_bin) {
        1.0
    } else {
        0.0
    });
    features.push(last_dist);
    features.push(my_r1 as f32 / 8.0);
    features.push(opp_r1 as f32 / 8.0);
    features.push(my_r2 as f32 / 24.0);
    features.push(opp_r2 as f32 / 24.0);
    for bin in 0..8 {
        features.push(if attack_bin == bin { 1.0 } else { 0.0 });
    }
    for bin in 0..8 {
        features.push(if block_bin == bin { 1.0 } else { 0.0 });
    }
    for attack in 0..8 {
        for block in 0..8 {
            features.push(if attack_bin == attack && block_bin == block {
                1.0
            } else {
                0.0
            });
        }
    }
    debug_assert_eq!(features.len(), RQ36_RICH_FEATURE_COUNT);
    features
}

fn rq36_attack_tier(bin: usize) -> i32 {
    match bin {
        1 => TIER_OPEN_THREE,
        2 => TIER_DOUBLE_THREE,
        3 => TIER_CLOSED_FOUR,
        4 => TIER_DOUBLE_FOUR,
        5 => TIER_OPEN_FOUR,
        6 => TIER_DOUBLE_FOUR,
        7 => TIER_WIN,
        _ => 0,
    }
}

fn rq36_block_tier(bin: usize) -> i32 {
    match bin {
        1 => TIER_BLOCK_OPEN_THREE,
        2 => TIER_BLOCK_DOUBLE_THREE,
        3 => TIER_BLOCK_CLOSED_FOUR,
        4 => TIER_BLOCK_DOUBLE_FOUR,
        5 => TIER_BLOCK_OPEN_FOUR,
        6 => TIER_BLOCK_DOUBLE_FOUR,
        7 => TIER_BLOCK_WIN,
        _ => 0,
    }
}

fn rq36_forcing_bin(bin: usize) -> bool {
    bin != 0
}

fn rq36_weak_attack_bin(attack_bin: usize, block_bin: usize) -> bool {
    matches!(attack_bin, 1 | 3) && block_bin == 0
}

fn rq36_move_order_score(board: &Board, mv: Move, attack_bin: usize, block_bin: usize) -> i32 {
    if attack_bin == 7 {
        return TIER_WIN;
    }
    if block_bin == 7 {
        return TIER_BLOCK_WIN;
    }
    let mut score = rq36_attack_tier(attack_bin).max(rq36_block_tier(block_bin));
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

fn push_rq15_kind_one_hot(features: &mut Vec<f32>, bin: usize) {
    for idx in 0..8 {
        features.push(if idx == bin { 1.0 } else { 0.0 });
    }
}

fn rq15_threat_bin(board: &Board, mv: Move, side: Stone) -> usize {
    debug_assert!(board.is_empty(mv));
    let (row, col) = to_rc(mv);
    let row = row as i32;
    let col = col as i32;
    let (mine, opp) = match side {
        Stone::Black => (board.black, board.white),
        Stone::White => (board.white, board.black),
    };
    let mut mine = mine;
    mine.set(mv);
    let mut open_fours = 0usize;
    let mut closed_fours = 0usize;
    let mut open_threes = 0usize;
    let rule_set = board.effective_rule_set();

    for &(dr, dc) in &DIR {
        let info = scan_line(&mine, &opp, row, col, dr, dc);
        let open_ends = info.open_front as u32 + info.open_back as u32;
        if rule_set.line_wins(side, info.count, open_ends) {
            return 7;
        }
        match (info.count, open_ends) {
            (4, 2) => open_fours += 1,
            (4, 1) => closed_fours += 1,
            (3, 2) => open_threes += 1,
            _ => {}
        }
    }

    if open_fours >= 2 {
        6
    } else if open_fours > 0 {
        5
    } else if closed_fours >= 2 {
        4
    } else if closed_fours > 0 {
        3
    } else if open_threes >= 2 {
        2
    } else if open_threes > 0 {
        1
    } else {
        0
    }
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

fn is_weak_attack(attack: ThreatKind, block: ThreatKind) -> bool {
    matches!(attack, ThreatKind::ClosedFour | ThreatKind::OpenThree) && block == ThreatKind::None
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
            let x = v
                .as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| format!("{name} contains non-number"))?;
            if x.is_finite() {
                Ok(x)
            } else {
                Err(format!("{name} contains non-finite value"))
            }
        })
        .collect()
}

fn f32_matrix(value: Option<&Value>, name: &str) -> Result<Vec<Vec<f32>>, String> {
    let arr = value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing {name}"))?;
    arr.iter()
        .map(|row| {
            let row = row
                .as_array()
                .ok_or_else(|| format!("{name} contains non-array row"))?;
            row.iter()
                .map(|v| {
                    let x = v
                        .as_f64()
                        .map(|x| x as f32)
                        .ok_or_else(|| format!("{name} contains non-number"))?;
                    if x.is_finite() {
                        Ok(x)
                    } else {
                        Err(format!("{name} contains non-finite value"))
                    }
                })
                .collect()
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn standardized_dot(features: &[f32], mean: &[f32], std: &[f32], weights: &[f32]) -> f32 {
    features
        .iter()
        .zip(mean.iter().zip(std.iter().zip(weights)))
        .map(|(x, (mean, (std, weight)))| ((*x - *mean) / *std) * *weight)
        .sum()
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

fn parse_env_f32(name: &str) -> Option<f32> {
    let raw = std::env::var(name).ok()?;
    raw.trim().parse::<f32>().ok()
}

fn parse_env_bool(name: &str) -> bool {
    parse_env_bool_default(name, false)
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

fn parse_gate_mode_env(name: &str) -> RootGateMode {
    let Ok(raw) = std::env::var(name) else {
        return RootGateMode::None;
    };
    let trimmed = raw.trim();
    if is_disabled_value(trimmed) || trimmed.eq_ignore_ascii_case("none") {
        return RootGateMode::None;
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "tactical" | "nonquiet" | "non-quiet" => RootGateMode::Tactical,
        "strict" | "same-threat" | "same_threat" => RootGateMode::Strict,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defensive_relation_features_have_rq15_shape() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(7, 8));
        board.make_move(to_idx(8, 7));
        assert_eq!(board.side_to_move, Stone::White);

        let features = defensive_relation_features(&board, to_idx(8, 8));
        assert_eq!(features.len(), DEF_REL_FEATURE_COUNT);
        assert!(features.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rq15_threat_bin_detects_open_three() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 8));
        board.make_move(to_idx(0, 1));

        assert_eq!(rq15_threat_bin(&board, to_idx(7, 9), Stone::Black), 1);
    }
}
