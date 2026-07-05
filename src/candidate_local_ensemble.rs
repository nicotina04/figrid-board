//! Root-batched candidate-local ensemble scorer.
//!
//! This loads the RQ343 `noru-rq343-teacher-played-ensemble-v1` artifact and
//! scores all root candidates as one batch so each head can be z-scored inside
//! the current root before blending.

use crate::board::{BOARD_SIZE, Board, Move, NUM_CELLS, Stone, to_idx, to_rc};
use crate::eval::evaluate;
use crate::heuristic::{DIR, scan_line};
use crate::search::RootCandidateAudit;
use crate::vct::{ThreatKind, classify_move_fast};
use noru::network::NnueWeights;
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::OnceLock;

const FORMAT: &str = "noru-rq343-teacher-played-ensemble-v1";
const RISK_FORMAT: &str = "noru-rq434-paired-causal-search-risk-logistic-v1";
const COMMITMENT_CRITIC_FORMAT: &str = "noru-root-commitment-critic-logistic-v1";
const CANDIDATE_TRUST_FORMAT: &str = "noru-runtime-gate-candidate-trust-logistic-v1";
const KIND_COUNT: usize = 8;
const RICH_COUNT: usize = 118;
const FEATURE_COUNT: usize = 256;
const EXPANDED_VALUE_COUNT: usize = 182;
const VALUE_ONLY_OFFSET: usize = EXPANDED_VALUE_COUNT - VALUE_ONLY_COUNT;
const VALUE_ONLY_COUNT: usize = 14;
const RANK_DELTA_OFFSET: usize = FEATURE_COUNT;
const RANK_DELTA_COUNT: usize = 12;
const POST_REPLY_OFFSET: usize = FEATURE_COUNT + RANK_DELTA_COUNT;
const POST_REPLY_COUNT: usize = 35;
const TRAJECTORY_CHILD_STATIC_OFFSET: usize = FEATURE_COUNT + RANK_DELTA_COUNT + POST_REPLY_COUNT;
const TRAJECTORY_CHILD_STATIC_COUNT: usize = 78;
const TRAJECTORY_POST_REPLY_OFFSET: usize =
    TRAJECTORY_CHILD_STATIC_OFFSET + TRAJECTORY_CHILD_STATIC_COUNT;
const TRAJECTORY_POST_REPLY_COUNT: usize = 140;
const ROLLOUT_FAST_OFFSET: usize = TRAJECTORY_POST_REPLY_OFFSET + TRAJECTORY_POST_REPLY_COUNT;
const ROLLOUT_FAST_COUNT: usize = 34;
const DEFAULT_TOP_REPLIES: usize = 8;
const DEFAULT_TRAJECTORY_POST_REPLY_TOP_REPLIES: usize = 16;
const DEFAULT_ROLLOUT_FAST_PLIES: usize = 8;
const DEFAULT_ROLLOUT_FAST_GAMMA: f32 = 0.92;
const ROLLOUT_FAST_FIRST_OPP_FORCE_DELAY_INDEX: usize = 7;
const DEFAULT_SCORE_SCALE: f32 = 100_000.0;
const WIN_SCORE: i32 = 999_000;

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
const ROLLOUT_FAST_FEATURE_NAMES: [&str; ROLLOUT_FAST_COUNT] = [
    "rollout_min_pressure_delta",
    "rollout_max_pressure_delta",
    "rollout_last_pressure_delta",
    "rollout_area",
    "rollout_debt",
    "rollout_terminal",
    "rollout_first_own_force_delay",
    "rollout_first_opp_force_delay",
    "rollout_first_own_lead_delay",
    "rollout_first_opp_lead_delay",
    "rollout_max_own_force",
    "rollout_max_opp_force",
    "rollout_pressure_delta_p0",
    "rollout_pressure_delta_p2",
    "rollout_pressure_delta_p4",
    "rollout_pressure_delta_p8",
    "rollout_fside_move_count",
    "rollout_opp_move_count",
    "rollout_fside_move_kind_quiet",
    "rollout_fside_move_kind_open_three",
    "rollout_fside_move_kind_double_open_three",
    "rollout_fside_move_kind_closed_four",
    "rollout_fside_move_kind_double_closed_four",
    "rollout_fside_move_kind_open_four",
    "rollout_fside_move_kind_double_open_four",
    "rollout_fside_move_kind_five",
    "rollout_opp_move_kind_quiet",
    "rollout_opp_move_kind_open_three",
    "rollout_opp_move_kind_double_open_three",
    "rollout_opp_move_kind_closed_four",
    "rollout_opp_move_kind_double_closed_four",
    "rollout_opp_move_kind_open_four",
    "rollout_opp_move_kind_double_open_four",
    "rollout_opp_move_kind_five",
];

#[derive(Clone)]
struct Ensemble {
    heads: Vec<Head>,
    raw_scores: bool,
}

#[derive(Clone)]
struct Head {
    weight: f32,
    input_dim: usize,
    input_offset: usize,
    input_indices: Option<Vec<usize>>,
    needs_rank_delta: bool,
    needs_post_reply: bool,
    needs_trajectory_child_static: bool,
    needs_trajectory_post_reply: bool,
    needs_rollout_fast: bool,
    mean: Vec<f32>,
    std: Vec<f32>,
    model: Model,
}

#[derive(Clone)]
enum Model {
    Linear {
        w: Vec<f32>,
        bias: f32,
    },
    Mlp {
        w1: Vec<Vec<f32>>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: f32,
    },
    Fm {
        w: Vec<f32>,
        v: Vec<Vec<f32>>,
        bias: f32,
    },
}

#[derive(Clone)]
struct RiskModel {
    feature_names: Vec<String>,
    mean: Vec<f32>,
    std: Vec<f32>,
    weights: Vec<f32>,
    bias: f32,
}

#[derive(Clone)]
struct CandidateTrustModel {
    feature_names: Vec<String>,
    mean: Vec<f32>,
    std: Vec<f32>,
    weights: Vec<f32>,
    bias: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CandidateTrustMode {
    Off,
    CodebookScorePrimary,
    ScorePrimaryRq423Blocked,
}

#[derive(Clone)]
struct CandidateValue {
    mv: Move,
    child_eval: f32,
    best_reply_eval: f32,
    best_reply: Option<Move>,
}

#[derive(Clone, Debug)]
struct ValueRecord {
    root_eval: f32,
    child_eval: f32,
    child_eval_delta: f32,
    child_eval_best_gap: f32,
    child_eval_worst_gap: f32,
    child_eval_rank_frac: f64,
    child_eval_percentile: f32,
    opp_best_reply_eval: f32,
    opp_best_reply_delta: f32,
    reply_drop: f32,
    reply_eval_best_gap: f32,
    reply_eval_worst_gap: f32,
    reply_eval_rank_frac: f64,
    reply_eval_percentile: f32,
    opp_best_reply: Option<Move>,
}

#[derive(Clone, Copy)]
struct CandidateProbe {
    status: ProbeStatus,
    unsafe_kind: &'static str,
    own_kind: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProbeStatus {
    WinsNow,
    Survives,
    Unsafe,
}

#[derive(Clone, Copy, Default)]
struct ThreatCounts {
    opp_immediate: usize,
    opp_open_four: usize,
    own_immediate: usize,
    own_open_four: usize,
}

#[derive(Clone, Copy, Default)]
struct PressureSnapshot {
    own_immediate: usize,
    own_open_four: usize,
    opp_immediate: usize,
    opp_open_four: usize,
    own_pressure: i32,
    opp_pressure: i32,
    own_kinds: [usize; KIND_COUNT],
    opp_kinds: [usize; KIND_COUNT],
}

#[derive(Clone, Copy)]
struct TrajectoryReplyCandidate {
    mv: Move,
    own_kind: usize,
    block_kind: usize,
    order_score: i32,
    center_dist: i32,
    row: i32,
    col: i32,
}

#[derive(Clone, Copy)]
struct RolloutFastCandidate {
    mv: Move,
    own_kind: usize,
    block_kind: usize,
    order_score: i32,
    center_dist: i32,
    row: i32,
    col: i32,
}

#[derive(Clone, Copy)]
struct RolloutFastMoveMeta {
    actor: Stone,
    own_kind: usize,
}

struct RolloutFastFeatureRecord {
    terminal: i32,
    snapshots: usize,
    moves: usize,
    gamma: f32,
    features: Vec<f32>,
}

#[derive(Clone, Copy)]
struct TrajectoryReplyRecord {
    exists: bool,
    mv: Move,
    own_kind: usize,
    block_kind: usize,
    order_score: i32,
    post: PressureSnapshot,
}

#[derive(Clone, Copy)]
struct ScoredMove {
    mv: Move,
    score: i32,
}

#[derive(Clone, Copy)]
struct OrderGuardEntry {
    rank: usize,
    score: i32,
}

#[derive(Clone, Copy)]
struct PostReplyGuardMetrics {
    incumbent_post_delta: i32,
    candidate_post_delta: i32,
    post_delta_diff: i32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RolloutFastRuleMode {
    Off,
    OwnThreatOppDelay,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Rq423RootAcceptPairMode {
    SearchIncumbent,
    CurrentBest,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SearchFeatureGuardMode {
    Off,
    ChildEvalDelta,
    ChildRankReplyDrop,
    ChildEvalBestGap,
    ReplyBestGapSearchDelta,
    SearchDeltaReplyRank,
}

#[derive(Clone, Copy)]
struct RolloutFastRuleMetrics {
    candidate_own_kind: usize,
    incumbent_opp_force_delay: f32,
    candidate_opp_force_delay: f32,
    opp_force_delay_delta: f32,
}

pub(crate) fn root_tiebreak_enabled_for(board: &Board) -> bool {
    root_tiebreak_enabled() && board.move_count >= min_ply() && ensemble().is_some()
}

pub(crate) fn final_root_tiebreak(
    board: &Board,
    weights: &NnueWeights,
    candidates: &[RootCandidateAudit],
    incumbent: Option<Move>,
    leader_score: i32,
) -> Option<Move> {
    if !root_tiebreak_enabled_for(board) {
        return None;
    }
    let mut best_move = incumbent?;
    if is_win_score(leader_score) {
        return Some(best_move);
    }
    let incumbent_move = best_move;
    let scores = root_score_map(board, weights)?;
    let veto_scores = root_veto_score_map(board, weights);
    let secondary_veto_scores = root_secondary_veto_score_map(board, weights);
    let mut best_score = scores.get(&best_move).copied();
    let root_margin = root_margin();
    let score_margin = root_score_margin();
    let primary_veto_margin = root_veto_margin();
    let primary_veto_confidence = root_veto_confidence();
    let secondary_veto_margin = root_secondary_veto_margin();
    let secondary_veto_confidence = root_secondary_veto_confidence();
    let reply_center_guard_max_delta = root_reply_center_guard_max_delta();
    let post_reply_guard_max_ply = root_post_reply_guard_max_ply();
    let post_reply_guard_max_delta = root_post_reply_guard_max_delta();
    let rollout_fast_rule_mode = if ensemble()
        .map(|ensemble| ensemble.needs_rollout_fast())
        .unwrap_or(false)
    {
        root_rollout_fast_rule_mode()
    } else {
        RolloutFastRuleMode::Off
    };
    let rollout_fast_rule_active = rollout_fast_rule_mode != RolloutFastRuleMode::Off;
    let rollout_fast_rule_min_own_kind = root_rollout_fast_rule_min_own_kind();
    let rq423_root_accept_pair_mode = rq423_root_accept_pair_mode();
    let search_feature_guard_mode = root_search_feature_guard_mode();
    let search_feature_guard_active = search_feature_guard_mode != SearchFeatureGuardMode::Off;
    let search_feature_guard_child_eval_delta = root_search_feature_guard_child_eval_delta();
    let search_feature_guard_min_child_best_gap = root_search_feature_guard_min_child_best_gap();
    let search_feature_guard_max_child_rank = root_search_feature_guard_max_child_rank();
    let search_feature_guard_max_reply_drop = root_search_feature_guard_max_reply_drop();
    let search_feature_guard_min_reply_best_gap = root_search_feature_guard_min_reply_best_gap();
    let search_feature_guard_min_search_delta = root_search_feature_guard_min_search_delta();
    let search_feature_guard_reply_rank_a_search_delta =
        root_search_feature_guard_reply_rank_a_search_delta();
    let search_feature_guard_reply_rank_a = root_search_feature_guard_reply_rank_a();
    let search_feature_guard_reply_rank_b_search_delta =
        root_search_feature_guard_reply_rank_b_search_delta();
    let search_feature_guard_reply_rank_b = root_search_feature_guard_reply_rank_b();
    let root_search_score_max = root_search_score_max();
    let codebook_final_tiebreak = crate::codebook_sidecar::root_final_tiebreak_enabled_for(board);
    let codebook_final_max_search_delta = crate::codebook_sidecar::root_final_max_search_delta();
    let codebook_final_score_margin = crate::codebook_sidecar::root_final_score_margin();
    let codebook_final_max_local_deficit = crate::codebook_sidecar::root_final_max_local_deficit();
    let codebook_final_require_global_best =
        crate::codebook_sidecar::root_final_require_global_best();
    let codebook_final_global_best_margin =
        crate::codebook_sidecar::root_final_global_best_margin();
    let codebook_final_global_best_score =
        if codebook_final_tiebreak && codebook_final_require_global_best {
            candidates
                .iter()
                .filter(|candidate| !is_win_score(candidate.search_score))
                .filter(|candidate| {
                    leader_score.saturating_sub(candidate.search_score)
                        <= codebook_final_max_search_delta
                })
                .filter_map(|candidate| candidate.codebook_score)
                .max()
        } else {
            None
        };
    let commitment_guard = root_commitment_guard();
    let commitment_search_score_min = root_commitment_search_score_min();
    let commitment_candidate_count_min = root_commitment_candidate_count_min();
    let root_risk_model = root_risk_model();
    let root_risk_threshold = root_risk_threshold();
    let root_commitment_critic_model = root_commitment_critic_model();
    let root_commitment_critic_threshold = root_commitment_critic_threshold();
    let candidate_trust_model = root_candidate_trust_model();
    let candidate_trust_threshold = root_candidate_trust_threshold();
    let candidate_trust_mode = root_candidate_trust_mode();
    let gate_mode = root_gate_mode();
    let mut best_gate = crate::candidate_ranker::root_gate_key(board, best_move);
    let rescue_min_order_rank = root_rescue_min_order_rank();
    let order_guard_max_rank = root_order_guard_max_rank();
    let needs_order_entries = root_order_guard_enabled()
        || order_guard_max_rank.is_some()
        || rescue_min_order_rank.is_some();
    let order_guard = needs_order_entries.then(|| root_order_guard_entries(board));
    let incumbent_order = order_guard
        .as_ref()
        .and_then(|entries| entries.get(&best_move).copied());
    let audit_enabled = root_audit_enabled();
    let audit_accepted_only = root_audit_accepted_only();
    let audit_all_candidates = audit_enabled && !audit_accepted_only;
    let audit_rollout_fast = audit_enabled && root_rollout_fast_audit_enabled();
    let rollout_fast = if audit_rollout_fast {
        let mut moves = candidates
            .iter()
            .map(|candidate| candidate.mv)
            .collect::<Vec<_>>();
        moves.push(best_move);
        Some(root_rollout_fast_map(board, moves))
    } else {
        None
    };
    let mut audit_candidates = Vec::new();
    let audit_search_features = audit_enabled && root_search_feature_audit_enabled();
    let search_feature_values = if audit_search_features
        || search_feature_guard_active
        || root_risk_model.is_some()
        || root_commitment_critic_model.is_some()
    {
        let mut moves = candidates
            .iter()
            .map(|candidate| candidate.mv)
            .collect::<Vec<_>>();
        moves.push(best_move);
        moves.sort_unstable();
        moves.dedup();
        Some(value_records(board, weights, &moves))
    } else {
        None
    };
    if !root_rescue_allows_incumbent(incumbent_order, rescue_min_order_rank) {
        if audit_all_candidates {
            append_root_audit(json!({
                "event": "candidate_local_root_tiebreak",
                "reason": "incumbent_not_rescuable",
                "pid": std::process::id(),
                "position": board_context_json(board),
                "move_count": board.move_count,
                "side_to_move": stone_label(board.side_to_move),
                "leader_score": leader_score,
                "incumbent": move_to_json(best_move),
                "incumbent_score": optional_i32_json(best_score),
                "incumbent_order": order_entry_to_json(incumbent_order),
                "incumbent_rollout_fast": rollout_fast_lookup_json(rollout_fast.as_ref(), best_move),
                "rollout_fast_audit": audit_rollout_fast,
                "rollout_fast_plies": root_rollout_fast_plies(),
                "rollout_fast_gamma": root_rollout_fast_gamma(),
                "rollout_fast_rule": rollout_fast_rule_label(rollout_fast_rule_mode),
                "rollout_fast_rule_min_own_kind": rollout_fast_rule_min_own_kind,
                "rescue_min_order_rank": rescue_min_order_rank,
                "order_guard_max_rank": order_guard_max_rank,
                "search_feature_audit": audit_search_features,
                "search_feature_top_replies": top_replies(),
                "search_feature_guard": search_feature_guard_label(search_feature_guard_mode),
                "search_feature_guard_child_eval_delta": search_feature_guard_child_eval_delta,
                "search_feature_guard_min_child_best_gap": search_feature_guard_min_child_best_gap,
                "search_feature_guard_max_child_rank": search_feature_guard_max_child_rank,
                "search_feature_guard_max_reply_drop": search_feature_guard_max_reply_drop,
                "search_feature_guard_min_reply_best_gap": search_feature_guard_min_reply_best_gap,
                "search_feature_guard_min_search_delta": search_feature_guard_min_search_delta,
                "search_feature_guard_reply_rank_a_search_delta": search_feature_guard_reply_rank_a_search_delta,
                "search_feature_guard_reply_rank_a": search_feature_guard_reply_rank_a,
                "search_feature_guard_reply_rank_b_search_delta": search_feature_guard_reply_rank_b_search_delta,
                "search_feature_guard_reply_rank_b": search_feature_guard_reply_rank_b,
                "root_search_score_max": optional_i32_json(root_search_score_max),
                "commitment_guard": commitment_guard,
                "commitment_search_score_min": commitment_search_score_min,
                "commitment_candidate_count_min": commitment_candidate_count_min,
                "root_risk_model": root_risk_model.is_some(),
                "root_risk_threshold": root_risk_threshold,
                "root_commitment_critic_model": root_commitment_critic_model.is_some(),
                "root_commitment_critic_threshold": root_commitment_critic_threshold,
            }));
        }
        return Some(best_move);
    }

    let initial_incumbent = best_move;
    let initial_incumbent_score = scores.get(&initial_incumbent).copied();
    for candidate in candidates {
        if candidate.mv == best_move || is_win_score(candidate.search_score) {
            if audit_all_candidates {
                audit_candidates.push(json!({
                    "move": move_to_json(candidate.mv),
                    "search_score": candidate.search_score,
                    "relation_score": optional_i32_json(candidate.relation_score),
                    "candidate_rank_score": optional_i32_json(candidate.candidate_rank_score),
                    "codebook_score": optional_i32_json(candidate.codebook_score),
                    "is_forcing": candidate.is_forcing,
                    "local_score": score_lookup_json(&scores, candidate.mv),
                    "rollout_fast": rollout_fast_lookup_json(rollout_fast.as_ref(), candidate.mv),
                    "skip": if candidate.mv == best_move { "current_best" } else { "search_win_score" },
                }));
            }
            continue;
        }
        let candidate_score = scores.get(&candidate.mv).copied();
        let best_score_before = best_score;
        let best_move_before = best_move;
        let within_margin = if root_margin == 0 {
            candidate.search_score == leader_score
        } else {
            leader_score.saturating_sub(candidate.search_score) <= root_margin
        };
        if !within_margin {
            if audit_all_candidates {
                audit_candidates.push(json!({
                    "move": move_to_json(candidate.mv),
                    "search_score": candidate.search_score,
                    "search_delta": leader_score.saturating_sub(candidate.search_score),
                    "relation_score": optional_i32_json(candidate.relation_score),
                    "candidate_rank_score": optional_i32_json(candidate.candidate_rank_score),
                    "codebook_score": optional_i32_json(candidate.codebook_score),
                    "is_forcing": candidate.is_forcing,
                    "local_score": optional_i32_json(candidate_score),
                    "best_score_before": optional_i32_json(best_score_before),
                    "rollout_fast": rollout_fast_lookup_json(rollout_fast.as_ref(), candidate.mv),
                    "skip": "outside_root_margin",
                }));
            }
            continue;
        }
        let candidate_gate = crate::candidate_ranker::root_gate_key(board, candidate.mv);
        if !crate::candidate_ranker::gate_allows(gate_mode, candidate_gate, best_gate) {
            if audit_all_candidates {
                audit_candidates.push(json!({
                    "move": move_to_json(candidate.mv),
                    "search_score": candidate.search_score,
                    "search_delta": leader_score.saturating_sub(candidate.search_score),
                    "relation_score": optional_i32_json(candidate.relation_score),
                    "candidate_rank_score": optional_i32_json(candidate.candidate_rank_score),
                    "codebook_score": optional_i32_json(candidate.codebook_score),
                    "is_forcing": candidate.is_forcing,
                    "local_score": optional_i32_json(candidate_score),
                    "best_score_before": optional_i32_json(best_score_before),
                    "rollout_fast": rollout_fast_lookup_json(rollout_fast.as_ref(), candidate.mv),
                    "skip": "gate_blocked",
                }));
            }
            continue;
        }
        let candidate_order = order_guard
            .as_ref()
            .and_then(|entries| entries.get(&candidate.mv).copied());
        if !root_order_guard_allows(candidate_order, incumbent_order) {
            if audit_all_candidates {
                audit_candidates.push(json!({
                    "move": move_to_json(candidate.mv),
                    "search_score": candidate.search_score,
                    "search_delta": leader_score.saturating_sub(candidate.search_score),
                    "relation_score": optional_i32_json(candidate.relation_score),
                    "candidate_rank_score": optional_i32_json(candidate.candidate_rank_score),
                    "codebook_score": optional_i32_json(candidate.codebook_score),
                    "is_forcing": candidate.is_forcing,
                    "local_score": optional_i32_json(candidate_score),
                    "best_score_before": optional_i32_json(best_score_before),
                    "candidate_order": order_entry_to_json(candidate_order),
                    "incumbent_order": order_entry_to_json(incumbent_order),
                    "rollout_fast": rollout_fast_lookup_json(rollout_fast.as_ref(), candidate.mv),
                    "skip": "order_guard_blocked",
                }));
            }
            continue;
        }
        let candidate_search_delta = leader_score.saturating_sub(candidate.search_score);
        let score_prefers = crate::candidate_ranker::score_prefers_with_margin(
            candidate_score,
            best_score,
            score_margin,
        );
        let best_codebook_score_before = candidates
            .iter()
            .find(|entry| entry.mv == best_move_before)
            .and_then(|entry| entry.codebook_score);
        let codebook_global_best_allows =
            if !codebook_final_tiebreak || !codebook_final_require_global_best {
                true
            } else {
                match (candidate.codebook_score, codebook_final_global_best_score) {
                    (Some(candidate_score), Some(global_best_score)) => {
                        candidate_score.saturating_add(codebook_final_global_best_margin)
                            >= global_best_score
                    }
                    _ => false,
                }
            };
        let codebook_local_deficit_allows = if !codebook_final_tiebreak {
            true
        } else {
            match (candidate_score, best_score_before) {
                (Some(candidate_score), Some(best_score_before)) => {
                    candidate_score.saturating_add(codebook_final_max_local_deficit)
                        >= best_score_before
                }
                (Some(_), None) => true,
                _ => false,
            }
        };
        let codebook_prefers = codebook_final_tiebreak
            && candidate_search_delta <= codebook_final_max_search_delta
            && codebook_global_best_allows
            && codebook_local_deficit_allows
            && crate::candidate_ranker::score_prefers_with_margin(
                candidate.codebook_score,
                best_codebook_score_before,
                codebook_final_score_margin,
            );
        let selector_prefers = score_prefers || codebook_prefers;
        let pressure_allows =
            root_pressure_guard_allows(board, veto_scores.as_ref(), incumbent_move, candidate.mv);
        let primary_veto_allows = root_veto_allows(
            veto_scores.as_ref(),
            incumbent_move,
            candidate.mv,
            primary_veto_margin,
            primary_veto_confidence,
        );
        let secondary_veto_allows = root_veto_allows(
            secondary_veto_scores.as_ref(),
            incumbent_move,
            candidate.mv,
            secondary_veto_margin,
            secondary_veto_confidence,
        );
        let reply_center_metrics = reply_center_guard_max_delta
            .and_then(|_| root_reply_center_guard_metrics(board, incumbent_move, candidate.mv));
        let reply_center_guard_allows =
            root_reply_center_guard_allows(reply_center_guard_max_delta, reply_center_metrics);
        let post_reply_metrics = post_reply_guard_max_delta
            .and_then(|_| root_post_reply_guard_metrics(board, incumbent_move, candidate.mv));
        let post_reply_guard_allows = root_post_reply_guard_allows(
            board.move_count + 1,
            post_reply_guard_max_ply,
            post_reply_guard_max_delta,
            post_reply_metrics,
        );
        let rollout_fast_rule_metrics = if selector_prefers && rollout_fast_rule_active {
            root_rollout_fast_rule_metrics(board, incumbent_move, candidate.mv)
        } else {
            None
        };
        let rollout_fast_rule_allows = root_rollout_fast_rule_allows(
            rollout_fast_rule_mode,
            rollout_fast_rule_min_own_kind,
            rollout_fast_rule_metrics,
        );
        let rq423_root_accept_baseline = match rq423_root_accept_pair_mode {
            Rq423RootAcceptPairMode::SearchIncumbent => incumbent_move,
            Rq423RootAcceptPairMode::CurrentBest => best_move,
        };
        let rq423_root_accept = if selector_prefers {
            crate::rq423_root_accept::root_accept_decision(
                board,
                rq423_root_accept_baseline,
                candidate.mv,
            )
        } else {
            None
        };
        let rq423_root_accept_allows = rq423_root_accept
            .map(|decision| decision.allows)
            .unwrap_or(true);
        let search_feature_guard_allows = root_search_feature_guard_allows(
            search_feature_guard_mode,
            search_feature_guard_child_eval_delta,
            search_feature_guard_min_child_best_gap,
            search_feature_guard_max_child_rank,
            search_feature_guard_max_reply_drop,
            search_feature_guard_min_reply_best_gap,
            search_feature_guard_min_search_delta,
            search_feature_guard_reply_rank_a_search_delta,
            search_feature_guard_reply_rank_a,
            search_feature_guard_reply_rank_b_search_delta,
            search_feature_guard_reply_rank_b,
            candidate_search_delta,
            search_feature_values
                .as_ref()
                .and_then(|values| values.get(&candidate.mv)),
        );
        let search_score_cap_allows = root_search_score_max
            .map(|max_score| candidate.search_score < max_score)
            .unwrap_or(true);
        let commitment_features = root_commitment_features(
            board,
            candidate.mv,
            candidates.len(),
            candidate.search_score,
            commitment_search_score_min,
            commitment_candidate_count_min,
        );
        let commitment_guard_allows = !commitment_guard || !commitment_features.risk;
        let candidate_value_record = search_feature_values
            .as_ref()
            .and_then(|values| values.get(&candidate.mv));
        let best_value_record = search_feature_values
            .as_ref()
            .and_then(|values| values.get(&best_move_before));
        let root_risk_feature_values = if selector_prefers
            && (root_risk_model.is_some() || root_commitment_critic_model.is_some())
        {
            Some(root_risk_features(
                board,
                candidate,
                candidate_score,
                best_score_before,
                initial_incumbent_score,
                candidate_score,
                leader_score,
                candidates.len(),
                veto_scores.as_ref(),
                incumbent_move,
                rq423_root_accept.map(|decision| decision.probability),
                candidate_value_record,
                best_value_record,
                candidate.codebook_score,
                best_codebook_score_before,
            ))
        } else {
            None
        };
        let root_risk_score = root_risk_model
            .zip(root_risk_feature_values.as_ref())
            .map(|(model, features)| model.score(features));
        let root_risk_allows = root_risk_score
            .map(|score| score < root_risk_threshold)
            .unwrap_or(true);
        let root_commitment_critic_score = root_commitment_critic_model
            .zip(root_risk_feature_values.as_ref())
            .map(|(model, base_features)| {
                let features = root_commitment_critic_features(
                    board,
                    candidate.mv,
                    incumbent_move,
                    base_features,
                );
                model.score(&features)
            });
        let root_commitment_critic_allows = root_commitment_critic_score
            .map(|score| score < root_commitment_critic_threshold)
            .unwrap_or(true);
        let accepted_without_candidate_trust = selector_prefers
            && pressure_allows
            && primary_veto_allows
            && secondary_veto_allows
            && reply_center_guard_allows
            && post_reply_guard_allows
            && rollout_fast_rule_allows
            && search_feature_guard_allows
            && search_score_cap_allows
            && commitment_guard_allows
            && root_risk_allows
            && root_commitment_critic_allows
            && rq423_root_accept_allows;
        let candidate_trust_feature_values = if selector_prefers && candidate_trust_model.is_some()
        {
            Some(root_candidate_trust_features(
                board,
                candidate,
                candidate_score,
                best_score_before,
                leader_score,
                veto_scores.as_ref(),
                incumbent_move,
                best_codebook_score_before,
                accepted_without_candidate_trust,
                codebook_prefers,
                codebook_global_best_allows,
                score_prefers,
                pressure_allows,
                primary_veto_allows,
                secondary_veto_allows,
                rq423_root_accept_allows,
                search_score_cap_allows,
            ))
        } else {
            None
        };
        let candidate_trust_score = candidate_trust_model
            .zip(candidate_trust_feature_values.as_ref())
            .map(|(model, features)| model.score(features));
        let candidate_trust_reason_allows = candidate_trust_mode_allows(
            candidate_trust_mode,
            codebook_prefers,
            score_prefers,
            primary_veto_allows,
            rq423_root_accept_allows,
        );
        let candidate_trust_allows = candidate_trust_score
            .map(|score| score >= candidate_trust_threshold && candidate_trust_reason_allows)
            .unwrap_or(false);
        let rq423_or_candidate_trust_allows = rq423_root_accept_allows || candidate_trust_allows;
        let accepted = selector_prefers
            && pressure_allows
            && primary_veto_allows
            && secondary_veto_allows
            && reply_center_guard_allows
            && post_reply_guard_allows
            && rollout_fast_rule_allows
            && search_feature_guard_allows
            && search_score_cap_allows
            && commitment_guard_allows
            && root_risk_allows
            && root_commitment_critic_allows
            && rq423_or_candidate_trust_allows;
        if audit_enabled && (!audit_accepted_only || accepted) {
            let mut audit_candidate = json!({
                "move": move_to_json(candidate.mv),
                "search_score": candidate.search_score,
                "search_delta": leader_score.saturating_sub(candidate.search_score),
                "relation_score": optional_i32_json(candidate.relation_score),
                "candidate_rank_score": optional_i32_json(candidate.candidate_rank_score),
                "codebook_score": optional_i32_json(candidate.codebook_score),
                "is_forcing": candidate.is_forcing,
                "local_score": optional_i32_json(candidate_score),
                "best_score_before": optional_i32_json(best_score_before),
                "candidate_order": order_entry_to_json(candidate_order),
                "incumbent_order": order_entry_to_json(incumbent_order),
                "rollout_fast": rollout_fast_lookup_json(rollout_fast.as_ref(), candidate.mv),
                "primary_veto_candidate_score": score_lookup_json_opt(veto_scores.as_ref(), candidate.mv),
                "primary_veto_incumbent_score": score_lookup_json_opt(veto_scores.as_ref(), incumbent_move),
                "secondary_veto_candidate_score": score_lookup_json_opt(secondary_veto_scores.as_ref(), candidate.mv),
                "secondary_veto_incumbent_score": score_lookup_json_opt(secondary_veto_scores.as_ref(), incumbent_move),
                "reply_center_incumbent": optional_i32_json(reply_center_metrics.map(|metrics| metrics.0)),
                "reply_center_candidate": optional_i32_json(reply_center_metrics.map(|metrics| metrics.1)),
                "reply_center_delta": optional_i32_json(reply_center_metrics.map(|metrics| metrics.2)),
                "post_reply_incumbent_post_delta": optional_i32_json(post_reply_metrics.map(|metrics| metrics.incumbent_post_delta)),
                "post_reply_candidate_post_delta": optional_i32_json(post_reply_metrics.map(|metrics| metrics.candidate_post_delta)),
                "post_reply_delta_diff": optional_i32_json(post_reply_metrics.map(|metrics| metrics.post_delta_diff)),
                "rollout_fast_rule_candidate_own_kind": optional_usize_json(rollout_fast_rule_metrics.map(|metrics| metrics.candidate_own_kind)),
                "rollout_fast_rule_incumbent_opp_force_delay": optional_f32_json(rollout_fast_rule_metrics.map(|metrics| metrics.incumbent_opp_force_delay)),
                "rollout_fast_rule_candidate_opp_force_delay": optional_f32_json(rollout_fast_rule_metrics.map(|metrics| metrics.candidate_opp_force_delay)),
                "rollout_fast_rule_opp_force_delay_delta": optional_f32_json(rollout_fast_rule_metrics.map(|metrics| metrics.opp_force_delay_delta)),
                "rq423_root_accept_pair_mode": rq423_root_accept_pair_mode_label(rq423_root_accept_pair_mode),
                "rq423_root_accept_baseline": move_to_json(rq423_root_accept_baseline),
                "rq423_root_accept_probability": optional_f32_json(rq423_root_accept.map(|decision| decision.probability)),
                "rq423_root_accept_threshold": optional_f32_json(rq423_root_accept.map(|decision| decision.threshold)),
                "rq423_root_accept_allows": rq423_root_accept_allows,
                "score_prefers": score_prefers,
                "pressure_allows": pressure_allows,
                "primary_veto_allows": primary_veto_allows,
                "secondary_veto_allows": secondary_veto_allows,
                "reply_center_guard_allows": reply_center_guard_allows,
                "post_reply_guard_allows": post_reply_guard_allows,
                "rollout_fast_rule": rollout_fast_rule_label(rollout_fast_rule_mode),
                "rollout_fast_rule_allows": rollout_fast_rule_allows,
                "accepted": accepted,
            });
            if let Some(object) = audit_candidate.as_object_mut() {
                object.insert(
                    "best_codebook_score_before".to_string(),
                    optional_i32_json(best_codebook_score_before),
                );
                object.insert(
                    "codebook_final_tiebreak".to_string(),
                    json!(codebook_final_tiebreak),
                );
                object.insert(
                    "codebook_final_max_search_delta".to_string(),
                    json!(codebook_final_max_search_delta),
                );
                object.insert(
                    "codebook_final_score_margin".to_string(),
                    json!(codebook_final_score_margin),
                );
                object.insert(
                    "codebook_final_max_local_deficit".to_string(),
                    json!(codebook_final_max_local_deficit),
                );
                object.insert(
                    "codebook_final_require_global_best".to_string(),
                    json!(codebook_final_require_global_best),
                );
                object.insert(
                    "codebook_final_global_best_margin".to_string(),
                    json!(codebook_final_global_best_margin),
                );
                object.insert(
                    "codebook_final_global_best_score".to_string(),
                    optional_i32_json(codebook_final_global_best_score),
                );
                object.insert(
                    "codebook_global_best_allows".to_string(),
                    json!(codebook_global_best_allows),
                );
                object.insert(
                    "codebook_local_deficit_allows".to_string(),
                    json!(codebook_local_deficit_allows),
                );
                object.insert("codebook_prefers".to_string(), json!(codebook_prefers));
                object.insert("selector_prefers".to_string(), json!(selector_prefers));
                object.insert(
                    "rq423_or_candidate_trust_allows".to_string(),
                    json!(rq423_or_candidate_trust_allows),
                );
                object.insert(
                    "candidate_trust_score".to_string(),
                    optional_f32_json(candidate_trust_score),
                );
                object.insert(
                    "candidate_trust_threshold".to_string(),
                    json!(candidate_trust_threshold),
                );
                object.insert(
                    "candidate_trust_mode".to_string(),
                    json!(candidate_trust_mode_label(candidate_trust_mode)),
                );
                object.insert(
                    "candidate_trust_reason_allows".to_string(),
                    json!(candidate_trust_reason_allows),
                );
                object.insert(
                    "candidate_trust_allows".to_string(),
                    json!(candidate_trust_allows),
                );
                object.insert(
                    "search_feature_guard".to_string(),
                    json!(search_feature_guard_label(search_feature_guard_mode)),
                );
                object.insert(
                    "search_feature_guard_child_eval_delta".to_string(),
                    json!(search_feature_guard_child_eval_delta),
                );
                object.insert(
                    "search_feature_guard_min_child_best_gap".to_string(),
                    json!(search_feature_guard_min_child_best_gap),
                );
                object.insert(
                    "search_feature_guard_max_child_rank".to_string(),
                    json!(search_feature_guard_max_child_rank),
                );
                object.insert(
                    "search_feature_guard_max_reply_drop".to_string(),
                    json!(search_feature_guard_max_reply_drop),
                );
                object.insert(
                    "search_feature_guard_min_reply_best_gap".to_string(),
                    json!(search_feature_guard_min_reply_best_gap),
                );
                object.insert(
                    "search_feature_guard_min_search_delta".to_string(),
                    json!(search_feature_guard_min_search_delta),
                );
                object.insert(
                    "search_feature_guard_reply_rank_a_search_delta".to_string(),
                    json!(search_feature_guard_reply_rank_a_search_delta),
                );
                object.insert(
                    "search_feature_guard_reply_rank_a".to_string(),
                    json!(search_feature_guard_reply_rank_a),
                );
                object.insert(
                    "search_feature_guard_reply_rank_b_search_delta".to_string(),
                    json!(search_feature_guard_reply_rank_b_search_delta),
                );
                object.insert(
                    "search_feature_guard_reply_rank_b".to_string(),
                    json!(search_feature_guard_reply_rank_b),
                );
                object.insert(
                    "search_feature_guard_allows".to_string(),
                    json!(search_feature_guard_allows),
                );
                object.insert(
                    "root_search_score_max".to_string(),
                    optional_i32_json(root_search_score_max),
                );
                object.insert(
                    "search_score_cap_allows".to_string(),
                    json!(search_score_cap_allows),
                );
                object.insert("commitment_guard".to_string(), json!(commitment_guard));
                object.insert(
                    "commitment_guard_allows".to_string(),
                    json!(commitment_guard_allows),
                );
                object.insert(
                    "commitment_risk".to_string(),
                    json!(commitment_features.risk),
                );
                object.insert(
                    "commitment_high_search".to_string(),
                    json!(commitment_features.high_search),
                );
                object.insert(
                    "commitment_wide_root".to_string(),
                    json!(commitment_features.wide_root),
                );
                object.insert(
                    "commitment_candidate_nonforcing".to_string(),
                    json!(commitment_features.candidate_nonforcing),
                );
                object.insert(
                    "commitment_candidate_attack".to_string(),
                    json!(threat_kind_label(commitment_features.candidate_attack)),
                );
                object.insert(
                    "commitment_candidate_block".to_string(),
                    json!(threat_kind_label(commitment_features.candidate_block)),
                );
                object.insert(
                    "commitment_search_score_min".to_string(),
                    json!(commitment_search_score_min),
                );
                object.insert(
                    "commitment_candidate_count_min".to_string(),
                    json!(commitment_candidate_count_min),
                );
                object.insert(
                    "root_risk_model".to_string(),
                    json!(root_risk_model.is_some()),
                );
                object.insert(
                    "root_risk_score".to_string(),
                    optional_f32_json(root_risk_score),
                );
                object.insert(
                    "root_risk_threshold".to_string(),
                    json!(root_risk_threshold),
                );
                object.insert("root_risk_allows".to_string(), json!(root_risk_allows));
                object.insert(
                    "root_commitment_critic_model".to_string(),
                    json!(root_commitment_critic_model.is_some()),
                );
                object.insert(
                    "root_commitment_critic_score".to_string(),
                    optional_f32_json(root_commitment_critic_score),
                );
                object.insert(
                    "root_commitment_critic_threshold".to_string(),
                    json!(root_commitment_critic_threshold),
                );
                object.insert(
                    "root_commitment_critic_allows".to_string(),
                    json!(root_commitment_critic_allows),
                );
            }
            if audit_search_features {
                if let Some(object) = audit_candidate.as_object_mut() {
                    object.insert(
                        "best_move_before".to_string(),
                        move_to_json(best_move_before),
                    );
                    object.insert(
                        "best_search".to_string(),
                        value_record_to_json(
                            search_feature_values
                                .as_ref()
                                .and_then(|values| values.get(&best_move_before)),
                        ),
                    );
                    object.insert(
                        "candidate_search".to_string(),
                        value_record_to_json(
                            search_feature_values
                                .as_ref()
                                .and_then(|values| values.get(&candidate.mv)),
                        ),
                    );
                    object.insert(
                        "candidate_minus_best_search".to_string(),
                        value_record_delta_to_json(
                            search_feature_values
                                .as_ref()
                                .and_then(|values| values.get(&candidate.mv)),
                            search_feature_values
                                .as_ref()
                                .and_then(|values| values.get(&best_move_before)),
                        ),
                    );
                }
            }
            audit_candidates.push(audit_candidate);
        }
        if accepted {
            best_move = candidate.mv;
            best_score = candidate_score;
            best_gate = candidate_gate;
        }
    }

    if audit_enabled && (!audit_accepted_only || best_move != initial_incumbent) {
        let codebook_final_audit = json!({
            "enabled": codebook_final_tiebreak,
            "max_search_delta": codebook_final_max_search_delta,
            "score_margin": codebook_final_score_margin,
            "max_local_deficit": codebook_final_max_local_deficit,
            "require_global_best": codebook_final_require_global_best,
            "global_best_margin": codebook_final_global_best_margin,
            "global_best_score": optional_i32_json(codebook_final_global_best_score),
        });
        let mut audit_row = json!({
            "event": "candidate_local_root_tiebreak",
            "pid": std::process::id(),
            "position": board_context_json(board),
            "move_count": board.move_count,
            "side_to_move": stone_label(board.side_to_move),
            "leader_score": leader_score,
            "candidate_count": candidates.len(),
            "incumbent": move_to_json(initial_incumbent),
            "final_move": move_to_json(best_move),
            "changed": best_move != initial_incumbent,
            "candidates": audit_candidates,
        });
        if let Some(object) = audit_row.as_object_mut() {
            object.insert("root_margin".to_string(), json!(root_margin));
            object.insert("score_margin".to_string(), json!(score_margin));
            object.insert(
                "order_guard_max_rank".to_string(),
                json!(order_guard_max_rank),
            );
            object.insert(
                "order_guard_max_rank_delta".to_string(),
                json!(root_order_guard_max_rank_delta()),
            );
            object.insert(
                "order_guard_score_margin".to_string(),
                json!(root_order_guard_score_margin()),
            );
            object.insert(
                "primary_veto_margin".to_string(),
                json!(primary_veto_margin),
            );
            object.insert(
                "primary_veto_confidence".to_string(),
                json!(primary_veto_confidence),
            );
            object.insert(
                "secondary_veto_margin".to_string(),
                json!(secondary_veto_margin),
            );
            object.insert(
                "secondary_veto_confidence".to_string(),
                json!(secondary_veto_confidence),
            );
            object.insert(
                "pressure_guard_min_candidate_score".to_string(),
                json!(root_pressure_guard_min_candidate_score()),
            );
            object.insert(
                "reply_center_guard_max_delta".to_string(),
                json!(reply_center_guard_max_delta),
            );
            object.insert(
                "post_reply_guard_max_ply".to_string(),
                json!(post_reply_guard_max_ply),
            );
            object.insert(
                "post_reply_guard_max_delta".to_string(),
                json!(post_reply_guard_max_delta),
            );
            object.insert("rollout_fast_audit".to_string(), json!(audit_rollout_fast));
            object.insert(
                "rollout_fast_plies".to_string(),
                json!(root_rollout_fast_plies()),
            );
            object.insert(
                "rollout_fast_gamma".to_string(),
                json!(root_rollout_fast_gamma()),
            );
            object.insert(
                "rollout_fast_rule".to_string(),
                json!(rollout_fast_rule_label(rollout_fast_rule_mode)),
            );
            object.insert(
                "rollout_fast_rule_min_own_kind".to_string(),
                json!(rollout_fast_rule_min_own_kind),
            );
            object.insert(
                "search_feature_audit".to_string(),
                json!(audit_search_features),
            );
            object.insert(
                "search_feature_top_replies".to_string(),
                json!(top_replies()),
            );
            object.insert(
                "search_feature_guard".to_string(),
                json!(search_feature_guard_label(search_feature_guard_mode)),
            );
            object.insert(
                "search_feature_guard_child_eval_delta".to_string(),
                json!(search_feature_guard_child_eval_delta),
            );
            object.insert(
                "search_feature_guard_max_child_rank".to_string(),
                json!(search_feature_guard_max_child_rank),
            );
            object.insert(
                "search_feature_guard_max_reply_drop".to_string(),
                json!(search_feature_guard_max_reply_drop),
            );
            object.insert("codebook_final".to_string(), codebook_final_audit);
            object.insert(
                "rq423_root_accept_pair_mode".to_string(),
                json!(rq423_root_accept_pair_mode_label(
                    rq423_root_accept_pair_mode
                )),
            );
            object.insert(
                "candidate_trust_model".to_string(),
                json!(candidate_trust_model.is_some()),
            );
            object.insert(
                "candidate_trust_threshold".to_string(),
                json!(candidate_trust_threshold),
            );
            object.insert(
                "candidate_trust_mode".to_string(),
                json!(candidate_trust_mode_label(candidate_trust_mode)),
            );
            object.insert("min_ply".to_string(), json!(min_ply()));
            object.insert(
                "incumbent_score".to_string(),
                score_lookup_json(&scores, initial_incumbent),
            );
            object.insert(
                "incumbent_rollout_fast".to_string(),
                rollout_fast_lookup_json(rollout_fast.as_ref(), initial_incumbent),
            );
            object.insert(
                "final_score".to_string(),
                score_lookup_json(&scores, best_move),
            );
            object.insert(
                "final_rollout_fast".to_string(),
                rollout_fast_lookup_json(rollout_fast.as_ref(), best_move),
            );
        }
        append_root_audit(audit_row);
    }
    Some(best_move)
}

fn root_audit_enabled() -> bool {
    root_audit_log_path().is_some()
}

pub(crate) fn root_search_decision_audit_enabled() -> bool {
    root_audit_enabled()
        && parse_env_bool_default("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_DECISION_AUDIT", false)
}

pub(crate) fn append_root_search_decision_audit(
    board: &Board,
    search_best_move: Option<Move>,
    final_move: Option<Move>,
    score: i32,
    depth: u32,
    nodes: u64,
    aborted: bool,
    candidates: &[RootCandidateAudit],
) {
    if !root_search_decision_audit_enabled() {
        return;
    }
    let candidates = candidates
        .iter()
        .map(|candidate| {
            json!({
                "move": move_to_json(candidate.mv),
                "search_score": candidate.search_score,
                "relation_score": optional_i32_json(candidate.relation_score),
                "candidate_rank_score": optional_i32_json(candidate.candidate_rank_score),
                "codebook_score": optional_i32_json(candidate.codebook_score),
                "is_forcing": candidate.is_forcing,
            })
        })
        .collect::<Vec<_>>();
    append_root_audit(json!({
        "event": "root_search_decision",
        "pid": std::process::id(),
        "position": board_context_json(board),
        "move_count": board.move_count,
        "side_to_move": stone_label(board.side_to_move),
        "search_best_move": optional_move_json(search_best_move),
        "final_move": optional_move_json(final_move),
        "changed_by_final_tiebreak": search_best_move != final_move,
        "score": score,
        "depth": depth,
        "nodes": nodes,
        "aborted": aborted,
        "candidate_count": candidates.len(),
        "candidates": candidates,
    }));
}

pub(crate) fn root_order_audit_enabled() -> bool {
    root_order_audit_log_path().is_some()
}

pub(crate) fn append_root_order_attempt_audit(
    board: &Board,
    move_count: usize,
    split: usize,
    tie_margin: u64,
    candidate_local_enabled: bool,
    candidate_ranker_order_enabled: bool,
    codebook_order_enabled: bool,
    relation_fusion_order_enabled: bool,
    candidate_ranker_final_only_enabled: bool,
    candidate_ranker_rescue_only_enabled: bool,
    active: bool,
    scores_len: Option<usize>,
    group_count: usize,
    changed_group_count: usize,
) {
    let Some(path) = root_order_audit_log_path() else {
        return;
    };
    append_audit_to_path(
        path,
        json!({
            "event": "candidate_local_root_order_attempt",
            "pid": std::process::id(),
            "position": board_context_json(board),
            "move_count": board.move_count,
            "side_to_move": stone_label(board.side_to_move),
            "candidate_count": move_count,
            "split": split,
            "tie_margin": tie_margin,
            "candidate_local_enabled": candidate_local_enabled,
            "candidate_ranker_order_enabled": candidate_ranker_order_enabled,
            "codebook_order_enabled": codebook_order_enabled,
            "relation_fusion_order_enabled": relation_fusion_order_enabled,
            "candidate_ranker_final_only_enabled": candidate_ranker_final_only_enabled,
            "candidate_ranker_rescue_only_enabled": candidate_ranker_rescue_only_enabled,
            "active": active,
            "scores_len": scores_len,
            "group_count": group_count,
            "changed_group_count": changed_group_count,
        }),
    );
}

pub(crate) fn append_root_order_audit(
    board: &Board,
    group_start: usize,
    group_score: u64,
    tie_margin: u64,
    before: &[Move],
    after: &[Move],
    scores: &BTreeMap<Move, i32>,
) {
    let Some(path) = root_order_audit_log_path() else {
        return;
    };
    let changed = before != after;
    if !changed && !root_order_audit_all() {
        return;
    }
    let entries = after
        .iter()
        .enumerate()
        .map(|(after_rank, mv)| {
            let before_rank = before
                .iter()
                .position(|before_mv| before_mv == mv)
                .unwrap_or(after_rank);
            json!({
                "move": move_to_json(*mv),
                "before_rank": before_rank,
                "after_rank": after_rank,
                "rank_delta": before_rank as i32 - after_rank as i32,
                "local_score": score_lookup_json(scores, *mv),
            })
        })
        .collect::<Vec<_>>();
    append_audit_to_path(
        path,
        json!({
            "event": "candidate_local_root_order",
            "pid": std::process::id(),
            "position": board_context_json(board),
            "move_count": board.move_count,
            "side_to_move": stone_label(board.side_to_move),
            "group_start": group_start,
            "group_len": before.len(),
            "group_score": group_score,
            "tie_margin": tie_margin,
            "changed": changed,
            "entries": entries,
        }),
    );
}

pub(crate) fn append_root_ab_probe_audit(
    board: &Board,
    start: usize,
    split: usize,
    depth: u32,
    before: &[Move],
    after: &[Move],
    local_scores: &BTreeMap<Move, i32>,
    probe_scores: &[(Move, i32)],
) {
    let Some(path) = root_order_audit_log_path() else {
        return;
    };
    let changed = before != after;
    if !changed && !root_order_audit_all() {
        return;
    }
    let entries = after
        .iter()
        .enumerate()
        .map(|(after_rank, mv)| {
            let before_rank = before
                .iter()
                .position(|before_mv| before_mv == mv)
                .unwrap_or(after_rank);
            let probe_score = probe_scores
                .iter()
                .find_map(|(probe_mv, score)| (*probe_mv == *mv).then_some(*score));
            json!({
                "move": move_to_json(*mv),
                "before_rank": before_rank,
                "after_rank": after_rank,
                "rank_delta": before_rank as i32 - after_rank as i32,
                "local_score": score_lookup_json(local_scores, *mv),
                "probe_score": optional_i32_json(probe_score),
            })
        })
        .collect::<Vec<_>>();
    append_audit_to_path(
        path,
        json!({
            "event": "candidate_local_root_ab_probe",
            "pid": std::process::id(),
            "position": board_context_json(board),
            "move_count": board.move_count,
            "side_to_move": stone_label(board.side_to_move),
            "start": start,
            "split": split,
            "depth": depth,
            "changed": changed,
            "entries": entries,
        }),
    );
}

fn root_audit_accepted_only() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_bool_default("NORU_CANDIDATE_LOCAL_ROOT_AUDIT_ACCEPTED_ONLY", false)
    })
}

fn root_search_feature_audit_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_bool_default("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_FEATURE_AUDIT", false)
    })
}

fn root_audit_log_path() -> Option<&'static str> {
    static VALUE: OnceLock<Option<String>> = OnceLock::new();
    VALUE
        .get_or_init(|| {
            std::env::var("NORU_CANDIDATE_LOCAL_ROOT_AUDIT_LOG")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty() && !is_disabled_value(value))
                .map(|value| expand_audit_log_path_for_process(&value))
        })
        .as_deref()
}

fn root_order_audit_log_path() -> Option<&'static str> {
    static VALUE: OnceLock<Option<String>> = OnceLock::new();
    VALUE
        .get_or_init(|| {
            std::env::var("NORU_CANDIDATE_LOCAL_ROOT_ORDER_AUDIT_LOG")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty() && !is_disabled_value(value))
                .map(|value| expand_audit_log_path_for_process(&value))
        })
        .as_deref()
}

fn root_order_audit_all() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE
        .get_or_init(|| parse_env_bool_default("NORU_CANDIDATE_LOCAL_ROOT_ORDER_AUDIT_ALL", false))
}

fn expand_audit_log_path_for_process(path: &str) -> String {
    let pid = std::process::id();
    if path.contains("{pid}") {
        return path.replace("{pid}", &pid.to_string());
    }
    match path.rfind('.') {
        Some(dot) => format!("{}-pid{}{}", &path[..dot], pid, &path[dot..]),
        None => format!("{path}-pid{pid}"),
    }
}

fn append_root_audit(row: Value) {
    let Some(path) = root_audit_log_path() else {
        return;
    };
    append_audit_to_path(path, row);
}

fn append_audit_to_path(path: &str, row: Value) {
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    let _ = writeln!(file, "{row}");
}

fn move_to_json(mv: Move) -> Value {
    let (row, col) = to_rc(mv);
    json!({
        "idx": mv,
        "row": row,
        "col": col,
    })
}

fn board_context_json(board: &Board) -> Value {
    json!({
        "zobrist": format!("{:016x}", board.zobrist),
        "history": board.history.iter().map(|mv| move_to_json(*mv)).collect::<Vec<_>>(),
    })
}

fn stone_label(stone: Stone) -> &'static str {
    match stone {
        Stone::Black => "black",
        Stone::White => "white",
    }
}

fn optional_i32_json(value: Option<i32>) -> Value {
    value.map_or(Value::Null, |value| json!(value))
}

fn optional_usize_json(value: Option<usize>) -> Value {
    value.map_or(Value::Null, |value| json!(value))
}

fn optional_f32_json(value: Option<f32>) -> Value {
    value.map_or(Value::Null, |value| json!(value))
}

fn optional_move_json(value: Option<Move>) -> Value {
    value.map_or(Value::Null, move_to_json)
}

fn value_record_to_json(value: Option<&ValueRecord>) -> Value {
    value.map_or(Value::Null, |value| {
        json!({
            "root_eval": value.root_eval,
            "child_eval": value.child_eval,
            "child_eval_delta": value.child_eval_delta,
            "child_eval_best_gap": value.child_eval_best_gap,
            "child_eval_worst_gap": value.child_eval_worst_gap,
            "child_eval_rank_frac": value.child_eval_rank_frac,
            "child_eval_percentile": value.child_eval_percentile,
            "opp_best_reply_eval": value.opp_best_reply_eval,
            "opp_best_reply_delta": value.opp_best_reply_delta,
            "reply_drop": value.reply_drop,
            "reply_eval_best_gap": value.reply_eval_best_gap,
            "reply_eval_worst_gap": value.reply_eval_worst_gap,
            "reply_eval_rank_frac": value.reply_eval_rank_frac,
            "reply_eval_percentile": value.reply_eval_percentile,
            "opp_best_reply": optional_move_json(value.opp_best_reply),
        })
    })
}

fn value_record_delta_to_json(
    candidate: Option<&ValueRecord>,
    baseline: Option<&ValueRecord>,
) -> Value {
    let (Some(candidate), Some(baseline)) = (candidate, baseline) else {
        return Value::Null;
    };
    json!({
        "child_eval": candidate.child_eval - baseline.child_eval,
        "child_eval_delta": candidate.child_eval_delta - baseline.child_eval_delta,
        "child_eval_best_gap": candidate.child_eval_best_gap - baseline.child_eval_best_gap,
        "child_eval_worst_gap": candidate.child_eval_worst_gap - baseline.child_eval_worst_gap,
        "child_eval_rank_frac": candidate.child_eval_rank_frac - baseline.child_eval_rank_frac,
        "child_eval_percentile": candidate.child_eval_percentile - baseline.child_eval_percentile,
        "opp_best_reply_eval": candidate.opp_best_reply_eval - baseline.opp_best_reply_eval,
        "opp_best_reply_delta": candidate.opp_best_reply_delta - baseline.opp_best_reply_delta,
        "reply_drop": candidate.reply_drop - baseline.reply_drop,
        "reply_eval_best_gap": candidate.reply_eval_best_gap - baseline.reply_eval_best_gap,
        "reply_eval_worst_gap": candidate.reply_eval_worst_gap - baseline.reply_eval_worst_gap,
        "reply_eval_rank_frac": candidate.reply_eval_rank_frac - baseline.reply_eval_rank_frac,
        "reply_eval_percentile": candidate.reply_eval_percentile - baseline.reply_eval_percentile,
    })
}

fn root_search_feature_guard_allows(
    mode: SearchFeatureGuardMode,
    child_eval_delta_threshold: i32,
    min_child_best_gap: i32,
    max_child_rank: f32,
    max_reply_drop: f32,
    min_reply_best_gap: f32,
    min_search_delta: i32,
    reply_rank_a_search_delta: i32,
    reply_rank_a: f32,
    reply_rank_b_search_delta: i32,
    reply_rank_b: f32,
    search_delta: i32,
    candidate: Option<&ValueRecord>,
) -> bool {
    match mode {
        SearchFeatureGuardMode::Off => true,
        SearchFeatureGuardMode::ChildEvalDelta => candidate
            .map(|value| value.child_eval_delta > child_eval_delta_threshold as f32)
            .unwrap_or(true),
        SearchFeatureGuardMode::ChildEvalBestGap => candidate
            .map(|value| value.child_eval_best_gap < min_child_best_gap as f32)
            .unwrap_or(true),
        SearchFeatureGuardMode::ChildRankReplyDrop => candidate
            .map(|value| {
                (value.child_eval_rank_frac as f32) <= max_child_rank
                    && value.reply_drop <= max_reply_drop
            })
            .unwrap_or(true),
        SearchFeatureGuardMode::ReplyBestGapSearchDelta => candidate
            .map(|value| {
                value.reply_eval_best_gap < min_reply_best_gap || search_delta < min_search_delta
            })
            .unwrap_or(true),
        SearchFeatureGuardMode::SearchDeltaReplyRank => candidate
            .map(|value| {
                let reply_rank = value.reply_eval_rank_frac as f32;
                let first_bucket =
                    search_delta >= reply_rank_a_search_delta && reply_rank >= reply_rank_a;
                let second_bucket =
                    search_delta >= reply_rank_b_search_delta && reply_rank >= reply_rank_b;
                !(first_bucket || second_bucket)
            })
            .unwrap_or(true),
    }
}

fn order_entry_to_json(entry: Option<OrderGuardEntry>) -> Value {
    entry.map_or(Value::Null, |entry| {
        json!({
            "rank": entry.rank,
            "score": entry.score,
        })
    })
}

fn score_lookup_json(scores: &BTreeMap<Move, i32>, mv: Move) -> Value {
    optional_i32_json(scores.get(&mv).copied())
}

fn score_lookup_json_opt(scores: Option<&BTreeMap<Move, i32>>, mv: Move) -> Value {
    optional_i32_json(scores.and_then(|scores| scores.get(&mv).copied()))
}

fn rollout_fast_lookup_json(rows: Option<&BTreeMap<Move, Value>>, mv: Move) -> Value {
    rows.and_then(|rows| rows.get(&mv).cloned())
        .unwrap_or(Value::Null)
}

fn root_order_guard_entries(board: &Board) -> BTreeMap<Move, OrderGuardEntry> {
    let side = board.side_to_move;
    let mut rows = candidate_moves_sorted(board)
        .into_iter()
        .filter(|&mv| board.is_empty(mv))
        .map(|mv| {
            let own_kind = threat_bin(board, mv, side);
            let block_kind = threat_bin(board, mv, side.opponent());
            let score = move_order_score(board, mv, side, own_kind, block_kind);
            (mv, score)
        })
        .collect::<Vec<_>>();
    rows.sort_by(|(move_a, score_a), (move_b, score_b)| {
        score_b.cmp(score_a).then_with(|| move_a.cmp(move_b))
    });
    rows.into_iter()
        .enumerate()
        .map(|(idx, (mv, score))| {
            (
                mv,
                OrderGuardEntry {
                    rank: idx + 1,
                    score,
                },
            )
        })
        .collect()
}

fn root_order_guard_allows(
    candidate: Option<OrderGuardEntry>,
    incumbent: Option<OrderGuardEntry>,
) -> bool {
    let Some(candidate) = candidate else {
        return true;
    };
    if let Some(max_rank) = root_order_guard_max_rank() {
        if candidate.rank > max_rank {
            return false;
        }
    }
    let Some(incumbent) = incumbent else {
        return true;
    };
    if candidate.rank
        > incumbent
            .rank
            .saturating_add(root_order_guard_max_rank_delta())
    {
        return false;
    }
    if let Some(score_margin) = root_order_guard_score_margin() {
        return candidate.score.saturating_add(score_margin) >= incumbent.score;
    }
    true
}

fn root_rescue_allows_incumbent(
    incumbent: Option<OrderGuardEntry>,
    min_order_rank: Option<usize>,
) -> bool {
    let Some(min_order_rank) = min_order_rank else {
        return true;
    };
    incumbent
        .map(|entry| entry.rank >= min_order_rank)
        .unwrap_or(false)
}

pub(crate) fn root_candidate_score_map(
    board: &Board,
    weights: &NnueWeights,
) -> Option<BTreeMap<Move, i32>> {
    let ensemble = ensemble()?;
    root_score_map_for(board, weights, ensemble)
}

fn root_score_map(board: &Board, weights: &NnueWeights) -> Option<BTreeMap<Move, i32>> {
    root_candidate_score_map(board, weights)
}

fn root_veto_score_map(board: &Board, weights: &NnueWeights) -> Option<BTreeMap<Move, i32>> {
    let ensemble = veto_ensemble()?;
    root_score_map_for(board, weights, ensemble)
}

fn root_secondary_veto_score_map(
    board: &Board,
    weights: &NnueWeights,
) -> Option<BTreeMap<Move, i32>> {
    let ensemble = secondary_veto_ensemble()?;
    root_score_map_for(board, weights, ensemble)
}

fn root_score_map_for(
    board: &Board,
    weights: &NnueWeights,
    ensemble: &Ensemble,
) -> Option<BTreeMap<Move, i32>> {
    let features = root_feature_rows(
        board,
        weights,
        ensemble.needs_rank_delta(),
        ensemble.needs_post_reply(),
        ensemble.needs_trajectory_child_static(),
        ensemble.needs_trajectory_post_reply(),
        ensemble.needs_rollout_fast(),
    );
    if features.is_empty() {
        return None;
    }
    Some(
        ensemble
            .score_root(features)
            .into_iter()
            .map(|item| (item.mv, item.score))
            .collect(),
    )
}

fn root_risk_features(
    board: &Board,
    candidate: &RootCandidateAudit,
    candidate_score: Option<i32>,
    best_score_before: Option<i32>,
    initial_incumbent_score: Option<i32>,
    prospective_final_score: Option<i32>,
    leader_score: i32,
    candidate_count: usize,
    veto_scores: Option<&BTreeMap<Move, i32>>,
    incumbent_move: Move,
    rq423_probability: Option<f32>,
    candidate_value: Option<&ValueRecord>,
    best_value: Option<&ValueRecord>,
    candidate_codebook_score: Option<i32>,
    incumbent_codebook_score: Option<i32>,
) -> BTreeMap<String, f32> {
    let mut features = BTreeMap::new();
    insert_i32_feature(
        &mut features,
        "candidate.search_score",
        Some(candidate.search_score),
    );
    insert_i32_feature(
        &mut features,
        "candidate.search_delta",
        Some(leader_score.saturating_sub(candidate.search_score)),
    );
    insert_feature(
        &mut features,
        "candidate.is_forcing",
        Some(if candidate.is_forcing { 1.0 } else { 0.0 }),
    );
    insert_i32_feature(&mut features, "candidate.local_score", candidate_score);
    insert_i32_feature(
        &mut features,
        "candidate.best_score_before",
        best_score_before,
    );
    insert_i32_feature(
        &mut features,
        "codebook.candidate_score",
        candidate_codebook_score,
    );
    insert_i32_feature(
        &mut features,
        "codebook.incumbent_score",
        incumbent_codebook_score,
    );
    if let (Some(candidate), Some(incumbent)) = (candidate_codebook_score, incumbent_codebook_score)
    {
        insert_i32_feature(
            &mut features,
            "codebook.candidate_minus_incumbent",
            Some(candidate.saturating_sub(incumbent)),
        );
    }
    let primary_candidate = veto_scores.and_then(|scores| scores.get(&candidate.mv).copied());
    let primary_incumbent = veto_scores.and_then(|scores| scores.get(&incumbent_move).copied());
    insert_i32_feature(
        &mut features,
        "candidate.primary_veto_candidate_score",
        primary_candidate,
    );
    insert_i32_feature(
        &mut features,
        "candidate.primary_veto_incumbent_score",
        primary_incumbent,
    );
    insert_feature(
        &mut features,
        "candidate.rq423_root_accept_probability",
        rq423_probability,
    );
    insert_i32_feature(
        &mut features,
        "root.candidate_count",
        Some(candidate_count as i32),
    );
    insert_i32_feature(&mut features, "root.leader_score", Some(leader_score));
    insert_i32_feature(
        &mut features,
        "root.incumbent_score",
        initial_incumbent_score,
    );
    insert_i32_feature(&mut features, "root.final_score", prospective_final_score);
    insert_i32_feature(
        &mut features,
        "position.move_count",
        Some(board.move_count as i32),
    );
    insert_feature(
        &mut features,
        "position.side_black",
        Some(if board.side_to_move == Stone::Black {
            1.0
        } else {
            0.0
        }),
    );
    insert_value_record_features(&mut features, "search.candidate", candidate_value);
    insert_value_record_features(&mut features, "search.incumbent", best_value);
    insert_value_record_delta_features(&mut features, candidate_value, best_value);
    if let (Some(candidate), Some(incumbent)) = (primary_candidate, primary_incumbent) {
        insert_i32_feature(
            &mut features,
            "derived.primary_veto_margin",
            Some(candidate.saturating_sub(incumbent)),
        );
    }
    if let (Some(candidate), Some(best)) = (candidate_value, best_value) {
        insert_feature(
            &mut features,
            "derived.child_eval_margin",
            Some(candidate.child_eval - best.child_eval),
        );
        insert_feature(
            &mut features,
            "derived.opp_reply_eval_margin",
            Some(candidate.opp_best_reply_eval - best.opp_best_reply_eval),
        );
        insert_feature(
            &mut features,
            "derived.reply_drop_margin",
            Some(candidate.reply_drop - best.reply_drop),
        );
        insert_feature(
            &mut features,
            "derived.reply_best_gap_margin",
            Some(candidate.reply_eval_best_gap - best.reply_eval_best_gap),
        );
    }
    let keys = features.keys().cloned().collect::<Vec<_>>();
    for key in keys {
        if key.ends_with("delta") || key.ends_with("gap") || key.ends_with("score") {
            if let Some(value) = features.get(&key).copied() {
                features.insert(format!("abs.{key}"), value.abs());
            }
        }
    }
    features
}

fn root_candidate_trust_features(
    board: &Board,
    candidate: &RootCandidateAudit,
    candidate_score: Option<i32>,
    best_score_before: Option<i32>,
    leader_score: i32,
    veto_scores: Option<&BTreeMap<Move, i32>>,
    incumbent_move: Move,
    best_codebook_score_before: Option<i32>,
    accepted_without_candidate_trust: bool,
    codebook_prefers: bool,
    codebook_global_best_allows: bool,
    score_prefers: bool,
    pressure_allows: bool,
    primary_veto_allows: bool,
    secondary_veto_allows: bool,
    rq423_root_accept_allows: bool,
    search_score_cap_allows: bool,
) -> BTreeMap<String, f32> {
    let mut features = BTreeMap::new();
    let search_delta = leader_score.saturating_sub(candidate.search_score);
    let primary_candidate = veto_scores.and_then(|scores| scores.get(&candidate.mv).copied());
    let primary_incumbent = veto_scores.and_then(|scores| scores.get(&incumbent_move).copied());

    insert_i32_feature(
        &mut features,
        "runtime.search_score",
        Some(candidate.search_score),
    );
    insert_i32_feature(&mut features, "runtime.search_delta", Some(search_delta));
    insert_i32_feature(&mut features, "runtime.local_score", candidate_score);
    insert_i32_feature(
        &mut features,
        "runtime.best_score_before",
        best_score_before,
    );
    insert_i32_feature(
        &mut features,
        "runtime.codebook_score",
        candidate.codebook_score,
    );
    insert_i32_feature(
        &mut features,
        "runtime.best_codebook_score_before",
        best_codebook_score_before,
    );
    insert_bool_feature(&mut features, "runtime.codebook_prefers", codebook_prefers);
    insert_bool_feature(
        &mut features,
        "runtime.codebook_global_best_allows",
        codebook_global_best_allows,
    );
    insert_bool_feature(&mut features, "runtime.score_prefers", score_prefers);
    insert_bool_feature(&mut features, "runtime.pressure_allows", pressure_allows);
    insert_bool_feature(
        &mut features,
        "runtime.primary_veto_allows",
        primary_veto_allows,
    );
    insert_bool_feature(
        &mut features,
        "runtime.secondary_veto_allows",
        secondary_veto_allows,
    );
    insert_bool_feature(
        &mut features,
        "runtime.rq423_root_accept_allows",
        rq423_root_accept_allows,
    );
    insert_bool_feature(
        &mut features,
        "runtime.search_score_cap_allows",
        search_score_cap_allows,
    );
    insert_i32_feature(
        &mut features,
        "runtime.primary_veto_candidate_score",
        primary_candidate,
    );
    insert_i32_feature(
        &mut features,
        "runtime.primary_veto_incumbent_score",
        primary_incumbent,
    );

    insert_i32_feature(
        &mut features,
        "abs.runtime.search_score",
        Some(candidate.search_score.abs()),
    );
    insert_i32_feature(
        &mut features,
        "abs.runtime.search_delta",
        Some(search_delta.abs()),
    );
    if let Some(value) = candidate_score {
        insert_i32_feature(&mut features, "abs.runtime.local_score", Some(value.abs()));
    }
    if let Some(value) = candidate.codebook_score {
        insert_i32_feature(
            &mut features,
            "abs.runtime.codebook_score",
            Some(value.abs()),
        );
    }

    insert_bool_feature(
        &mut features,
        "reason.accepted",
        accepted_without_candidate_trust,
    );
    insert_bool_feature(&mut features, "reason.codebook_prefers", codebook_prefers);
    insert_bool_feature(
        &mut features,
        "reason.codebook_global_best",
        codebook_global_best_allows,
    );
    insert_bool_feature(
        &mut features,
        "reason.global_score",
        codebook_global_best_allows && score_prefers,
    );
    insert_bool_feature(
        &mut features,
        "reason.score_primary_rq423_blocked",
        score_prefers && primary_veto_allows && !rq423_root_accept_allows,
    );
    insert_bool_feature(
        &mut features,
        "reason.codebook_score_primary",
        codebook_prefers && score_prefers && primary_veto_allows,
    );

    insert_i32_feature(&mut features, "position.ply", Some(board.move_count as i32));
    insert_bool_feature(
        &mut features,
        "position.side_black",
        board.side_to_move == Stone::Black,
    );
    insert_candidate_trust_move_features(&mut features, "candidate_move", candidate.mv);
    insert_candidate_trust_move_features(&mut features, "incumbent_move", incumbent_move);
    insert_i32_feature(
        &mut features,
        "move.candidate_incumbent_manhattan",
        Some(move_manhattan(candidate.mv, incumbent_move)),
    );
    insert_i32_feature(
        &mut features,
        "move.candidate_center_delta_vs_incumbent",
        Some(center_dist_move(candidate.mv) - center_dist_move(incumbent_move)),
    );

    if let (Some(candidate_score), Some(best_score_before)) = (candidate_score, best_score_before) {
        let margin = candidate_score.saturating_sub(best_score_before);
        insert_i32_feature(
            &mut features,
            "derived.local_margin_vs_before",
            Some(margin),
        );
        insert_i32_feature(
            &mut features,
            "abs.derived.local_margin_vs_before",
            Some(margin.abs()),
        );
    }
    if let (Some(candidate_score), Some(best_codebook_score_before)) =
        (candidate.codebook_score, best_codebook_score_before)
    {
        let margin = candidate_score.saturating_sub(best_codebook_score_before);
        insert_i32_feature(
            &mut features,
            "derived.codebook_margin_vs_before",
            Some(margin),
        );
        insert_i32_feature(
            &mut features,
            "abs.derived.codebook_margin_vs_before",
            Some(margin.abs()),
        );
    }
    if let (Some(candidate_score), Some(local_score)) = (candidate.codebook_score, candidate_score)
    {
        insert_i32_feature(
            &mut features,
            "derived.codebook_minus_local",
            Some(candidate_score.saturating_sub(local_score)),
        );
    }
    if let Some(local_score) = candidate_score {
        insert_i32_feature(
            &mut features,
            "derived.search_minus_local",
            Some(candidate.search_score.saturating_sub(local_score)),
        );
    }
    if let (Some(candidate), Some(incumbent)) = (primary_candidate, primary_incumbent) {
        let margin = candidate.saturating_sub(incumbent);
        insert_i32_feature(&mut features, "derived.primary_veto_margin", Some(margin));
        insert_i32_feature(
            &mut features,
            "abs.derived.primary_veto_margin",
            Some(margin.abs()),
        );
    }

    insert_candidate_trust_tactical_pair(
        &mut features,
        board,
        board.side_to_move,
        candidate.mv,
        incumbent_move,
    );
    features
}

fn insert_candidate_trust_move_features(
    features: &mut BTreeMap<String, f32>,
    prefix: &str,
    mv: Move,
) {
    let (row, col) = to_rc(mv);
    insert_i32_feature(features, &format!("{prefix}.x"), Some(col as i32));
    insert_i32_feature(features, &format!("{prefix}.y"), Some(row as i32));
    insert_i32_feature(
        features,
        &format!("{prefix}.center_dist"),
        Some(center_dist_move(mv)),
    );
    insert_i32_feature(
        features,
        &format!("{prefix}.edge_dist"),
        Some(edge_dist_move(mv)),
    );
}

fn insert_candidate_trust_tactical_pair(
    features: &mut BTreeMap<String, f32>,
    board: &Board,
    side: Stone,
    candidate: Move,
    incumbent: Move,
) {
    let Some(candidate_features) = candidate_trust_tactical_move_features(board, side, candidate)
    else {
        return;
    };
    let Some(incumbent_features) = candidate_trust_tactical_move_features(board, side, incumbent)
    else {
        return;
    };
    for (prefix, map) in [
        ("tactical.candidate", &candidate_features),
        ("tactical.incumbent", &incumbent_features),
    ] {
        for (name, value) in map {
            insert_feature(features, &format!("{prefix}.{name}"), Some(*value));
        }
    }
    let mut keys = BTreeSet::new();
    keys.extend(candidate_features.keys().cloned());
    keys.extend(incumbent_features.keys().cloned());
    for key in keys {
        let delta = candidate_features.get(&key).copied().unwrap_or(0.0)
            - incumbent_features.get(&key).copied().unwrap_or(0.0);
        insert_feature(features, &format!("tactical.delta.{key}"), Some(delta));
        if key.ends_with("moves")
            || key.ends_with("count")
            || key == "opp_reply_max_bucket"
            || key == "opp_reply_danger_any"
            || key == "opp_reply_force_count"
        {
            insert_feature(
                features,
                &format!("abs.tactical.delta.{key}"),
                Some(delta.abs()),
            );
        }
    }
}

fn candidate_trust_tactical_move_features(
    board: &Board,
    side: Stone,
    mv: Move,
) -> Option<BTreeMap<String, f32>> {
    if !board.is_empty(mv) {
        return None;
    }
    let (basic, rich) = crate::rq423_root_accept::debug_move_feature_maps(board, side, mv);
    let mut out = BTreeMap::new();
    for key in [
        "x",
        "y",
        "center_dist",
        "edge_dist",
        "zone",
        "attack",
        "attack2",
        "block",
        "block2",
        "max_threat",
        "multi_threat",
        "attack_minus_block",
        "r1_own",
        "r1_opp",
        "r1_empty",
        "r2_own",
        "r2_opp",
        "r2_empty",
        "own_line_best_stones",
        "own_line_second_stones",
        "own_line_best_open",
        "own_line_open_dirs",
        "opp_line_best_stones",
        "opp_line_second_stones",
        "opp_line_best_open",
        "opp_line_open_dirs",
    ] {
        insert_feature(&mut out, key, basic.get(key).copied());
    }
    for key in [
        "r3_own", "r3_opp", "r3_empty", "r4_own", "r4_opp", "r4_empty",
    ] {
        insert_feature(&mut out, key, rich.get(key).copied());
    }
    for (target, source) in [
        ("own_four_windows", "own_win_four_windows"),
        ("own_open_four_dirs", "own_win_dir_four_open2"),
        ("own_open_three_dirs", "own_win_dir_three_open2"),
        ("own_double_four", "own_win_double_four"),
        ("own_four_three", "own_win_four_three"),
        ("own_double_three", "own_win_double_three"),
        ("block_four_windows", "block_win_four_windows"),
        ("block_open_four_dirs", "block_win_dir_four_open2"),
        ("block_open_three_dirs", "block_win_dir_three_open2"),
        ("block_double_four", "block_win_double_four"),
        ("block_four_three", "block_win_four_three"),
        ("block_double_three", "block_win_double_three"),
    ] {
        insert_feature(&mut out, target, rich.get(source).copied());
    }
    for (target, source) in [
        ("opp_reply_points", "opp_reply_r3_points"),
        ("opp_reply_five_moves", "opp_reply_r3_five_moves"),
        ("opp_reply_four_moves", "opp_reply_r3_four_moves"),
        ("opp_reply_open_four_moves", "opp_reply_r3_open_four_moves"),
        ("opp_reply_three_moves", "opp_reply_r3_three_moves"),
        (
            "opp_reply_open_three_moves",
            "opp_reply_r3_open_three_moves",
        ),
        (
            "opp_reply_double_four_moves",
            "opp_reply_r3_double_four_moves",
        ),
        (
            "opp_reply_double_three_moves",
            "opp_reply_r3_double_three_moves",
        ),
    ] {
        insert_feature(&mut out, target, rich.get(source).copied());
    }
    let reply_five = map_value(&out, "opp_reply_five_moves");
    let reply_four = map_value(&out, "opp_reply_four_moves");
    let reply_open_four = map_value(&out, "opp_reply_open_four_moves");
    let reply_open_three = map_value(&out, "opp_reply_open_three_moves");
    let reply_double_four = map_value(&out, "opp_reply_double_four_moves");
    let reply_double_three = map_value(&out, "opp_reply_double_three_moves");
    let reply_four_three = 0.0;
    let max_bucket = if reply_five > 0.0 {
        5.0
    } else if reply_four > 0.0 {
        4.0
    } else if reply_open_three > 0.0 {
        3.0
    } else {
        0.0
    };
    insert_feature(
        &mut out,
        "opp_reply_four_three_moves",
        Some(reply_four_three),
    );
    insert_feature(&mut out, "opp_reply_max_bucket", Some(max_bucket));
    insert_feature(
        &mut out,
        "opp_reply_danger_any",
        Some(
            if reply_five > 0.0
                || reply_four > 0.0
                || reply_open_four > 0.0
                || reply_double_three > 0.0
                || reply_four_three > 0.0
            {
                1.0
            } else {
                0.0
            },
        ),
    );
    insert_feature(
        &mut out,
        "opp_reply_force_count",
        Some(reply_five + reply_four + reply_open_four + reply_double_four + reply_four_three),
    );
    let force_window_balance = map_value(&out, "own_four_windows")
        + map_value(&out, "block_four_windows")
        - reply_four
        - reply_five;
    insert_feature(&mut out, "force_window_balance", Some(force_window_balance));
    insert_feature(
        &mut out,
        "three_window_balance",
        Some(
            map_value(&rich, "own_win_three_windows") + map_value(&rich, "block_win_three_windows")
                - reply_open_three,
        ),
    );
    Some(out)
}

fn map_value(map: &BTreeMap<String, f32>, key: &str) -> f32 {
    map.get(key).copied().unwrap_or(0.0)
}

fn center_dist_move(mv: Move) -> i32 {
    let (row, col) = to_rc(mv);
    let center = (BOARD_SIZE as i32) / 2;
    (row as i32 - center).abs() + (col as i32 - center).abs()
}

fn edge_dist_move(mv: Move) -> i32 {
    let (row, col) = to_rc(mv);
    let row = row as i32;
    let col = col as i32;
    row.min(col)
        .min(BOARD_SIZE as i32 - 1 - row)
        .min(BOARD_SIZE as i32 - 1 - col)
}

fn move_manhattan(left: Move, right: Move) -> i32 {
    let (left_row, left_col) = to_rc(left);
    let (right_row, right_col) = to_rc(right);
    (left_row as i32 - right_row as i32).abs() + (left_col as i32 - right_col as i32).abs()
}

fn insert_i32_feature(features: &mut BTreeMap<String, f32>, name: &str, value: Option<i32>) {
    insert_feature(features, name, value.map(|value| value as f32));
}

fn insert_feature(features: &mut BTreeMap<String, f32>, name: &str, value: Option<f32>) {
    if let Some(value) = value.filter(|value| value.is_finite()) {
        features.insert(name.to_string(), value);
    }
}

fn insert_value_record_features(
    features: &mut BTreeMap<String, f32>,
    prefix: &str,
    value: Option<&ValueRecord>,
) {
    let Some(value) = value else {
        return;
    };
    insert_feature(
        features,
        &format!("{prefix}.root_eval"),
        Some(value.root_eval),
    );
    insert_feature(
        features,
        &format!("{prefix}.child_eval"),
        Some(value.child_eval),
    );
    insert_feature(
        features,
        &format!("{prefix}.child_eval_delta"),
        Some(value.child_eval_delta),
    );
    insert_feature(
        features,
        &format!("{prefix}.child_eval_best_gap"),
        Some(value.child_eval_best_gap),
    );
    insert_feature(
        features,
        &format!("{prefix}.child_eval_worst_gap"),
        Some(value.child_eval_worst_gap),
    );
    insert_feature(
        features,
        &format!("{prefix}.child_eval_rank_frac"),
        Some(value.child_eval_rank_frac as f32),
    );
    insert_feature(
        features,
        &format!("{prefix}.child_eval_percentile"),
        Some(value.child_eval_percentile),
    );
    insert_feature(
        features,
        &format!("{prefix}.opp_best_reply_eval"),
        Some(value.opp_best_reply_eval),
    );
    insert_feature(
        features,
        &format!("{prefix}.opp_best_reply_delta"),
        Some(value.opp_best_reply_delta),
    );
    insert_feature(
        features,
        &format!("{prefix}.reply_drop"),
        Some(value.reply_drop),
    );
    insert_feature(
        features,
        &format!("{prefix}.reply_eval_best_gap"),
        Some(value.reply_eval_best_gap),
    );
    insert_feature(
        features,
        &format!("{prefix}.reply_eval_worst_gap"),
        Some(value.reply_eval_worst_gap),
    );
    insert_feature(
        features,
        &format!("{prefix}.reply_eval_rank_frac"),
        Some(value.reply_eval_rank_frac as f32),
    );
    insert_feature(
        features,
        &format!("{prefix}.reply_eval_percentile"),
        Some(value.reply_eval_percentile),
    );
}

fn insert_value_record_delta_features(
    features: &mut BTreeMap<String, f32>,
    candidate: Option<&ValueRecord>,
    best: Option<&ValueRecord>,
) {
    let (Some(candidate), Some(best)) = (candidate, best) else {
        return;
    };
    insert_feature(
        features,
        "search.delta.child_eval",
        Some(candidate.child_eval - best.child_eval),
    );
    insert_feature(
        features,
        "search.delta.child_eval_delta",
        Some(candidate.child_eval_delta - best.child_eval_delta),
    );
    insert_feature(
        features,
        "search.delta.child_eval_best_gap",
        Some(candidate.child_eval_best_gap - best.child_eval_best_gap),
    );
    insert_feature(
        features,
        "search.delta.child_eval_worst_gap",
        Some(candidate.child_eval_worst_gap - best.child_eval_worst_gap),
    );
    insert_feature(
        features,
        "search.delta.child_eval_rank_frac",
        Some((candidate.child_eval_rank_frac - best.child_eval_rank_frac) as f32),
    );
    insert_feature(
        features,
        "search.delta.child_eval_percentile",
        Some(candidate.child_eval_percentile - best.child_eval_percentile),
    );
    insert_feature(
        features,
        "search.delta.opp_best_reply_eval",
        Some(candidate.opp_best_reply_eval - best.opp_best_reply_eval),
    );
    insert_feature(
        features,
        "search.delta.opp_best_reply_delta",
        Some(candidate.opp_best_reply_delta - best.opp_best_reply_delta),
    );
    insert_feature(
        features,
        "search.delta.reply_drop",
        Some(candidate.reply_drop - best.reply_drop),
    );
    insert_feature(
        features,
        "search.delta.reply_eval_best_gap",
        Some(candidate.reply_eval_best_gap - best.reply_eval_best_gap),
    );
    insert_feature(
        features,
        "search.delta.reply_eval_worst_gap",
        Some(candidate.reply_eval_worst_gap - best.reply_eval_worst_gap),
    );
    insert_feature(
        features,
        "search.delta.reply_eval_rank_frac",
        Some((candidate.reply_eval_rank_frac - best.reply_eval_rank_frac) as f32),
    );
    insert_feature(
        features,
        "search.delta.reply_eval_percentile",
        Some(candidate.reply_eval_percentile - best.reply_eval_percentile),
    );
}
fn root_veto_allows(
    scores: Option<&BTreeMap<Move, i32>>,
    incumbent: Move,
    candidate: Move,
    margin: i32,
    confidence: Option<i32>,
) -> bool {
    let Some(scores) = scores else {
        return true;
    };
    let Some(candidate_score) = scores.get(&candidate).copied() else {
        return false;
    };
    let Some(incumbent_score) = scores.get(&incumbent).copied() else {
        return false;
    };
    if let Some(confidence) = confidence {
        if candidate_score < confidence || incumbent_score > -confidence {
            return false;
        }
    }
    candidate_score >= incumbent_score.saturating_add(margin)
}

fn root_pressure_guard_allows(
    board: &Board,
    scores: Option<&BTreeMap<Move, i32>>,
    incumbent: Move,
    candidate: Move,
) -> bool {
    let min_incumbent_child_pressure = root_pressure_guard_min_incumbent_child();
    let min_child_delta = root_pressure_guard_min_child_delta();
    let min_candidate_score = root_pressure_guard_min_candidate_score();
    if min_incumbent_child_pressure.is_none()
        && min_child_delta.is_none()
        && min_candidate_score.is_none()
    {
        return true;
    }
    let mut incumbent_child_pressure = None;
    if let Some(min_pressure) = min_incumbent_child_pressure {
        let side = board.side_to_move;
        let Some(pressure) = child_pressure_delta(board, incumbent, side) else {
            return false;
        };
        incumbent_child_pressure = Some(pressure);
        if pressure < min_pressure {
            return false;
        }
    }
    if let Some(min_delta) = min_child_delta {
        let side = board.side_to_move;
        let incumbent_pressure = match incumbent_child_pressure {
            Some(pressure) => pressure,
            None => {
                let Some(pressure) = child_pressure_delta(board, incumbent, side) else {
                    return false;
                };
                pressure
            }
        };
        let Some(candidate_pressure) = child_pressure_delta(board, candidate, side) else {
            return false;
        };
        if candidate_pressure.saturating_sub(incumbent_pressure) < min_delta {
            return false;
        }
    }
    if let Some(min_score) = min_candidate_score {
        let Some(scores) = scores else {
            return false;
        };
        let Some(candidate_score) = scores.get(&candidate).copied() else {
            return false;
        };
        if candidate_score < min_score {
            return false;
        }
    }
    true
}

fn root_reply_center_guard_allows(
    max_delta: Option<i32>,
    metrics: Option<(i32, i32, i32)>,
) -> bool {
    let Some(max_delta) = max_delta else {
        return true;
    };
    metrics
        .map(|(_, _, delta)| delta <= max_delta)
        .unwrap_or(false)
}

fn root_reply_center_guard_metrics(
    board: &Board,
    incumbent: Move,
    candidate: Move,
) -> Option<(i32, i32, i32)> {
    let side = board.side_to_move;
    let incumbent_center = root_reply_center_after(board, incumbent, side)?;
    let candidate_center = root_reply_center_after(board, candidate, side)?;
    Some((
        incumbent_center,
        candidate_center,
        candidate_center - incumbent_center,
    ))
}

fn root_reply_center_after(board: &Board, mv: Move, side: Stone) -> Option<i32> {
    if !board.is_empty(mv) {
        return None;
    }
    let mut scratch = board.clone();
    place_stone(&mut scratch, mv, side);
    let reply = best_trajectory_reply(&mut scratch, side);
    if !reply.exists {
        return None;
    }
    let (row, col) = to_rc(reply.mv);
    Some((col as i32 - 7).abs() + (row as i32 - 7).abs())
}

fn root_post_reply_guard_allows(
    ply: usize,
    max_ply: Option<usize>,
    max_delta: Option<i32>,
    metrics: Option<PostReplyGuardMetrics>,
) -> bool {
    if max_ply.is_none() && max_delta.is_none() {
        return true;
    }
    if let Some(max_ply) = max_ply {
        if ply > max_ply {
            return false;
        }
    }
    let Some(max_delta) = max_delta else {
        return true;
    };
    metrics
        .map(|metrics| metrics.post_delta_diff <= max_delta)
        .unwrap_or(false)
}

fn root_post_reply_guard_metrics(
    board: &Board,
    incumbent: Move,
    candidate: Move,
) -> Option<PostReplyGuardMetrics> {
    let side = board.side_to_move;
    let incumbent_post_delta = root_post_reply_post_delta(board, incumbent, side)?;
    let candidate_post_delta = root_post_reply_post_delta(board, candidate, side)?;
    Some(PostReplyGuardMetrics {
        incumbent_post_delta,
        candidate_post_delta,
        post_delta_diff: candidate_post_delta - incumbent_post_delta,
    })
}

fn root_post_reply_post_delta(board: &Board, mv: Move, side: Stone) -> Option<i32> {
    if !board.is_empty(mv) {
        return None;
    }
    let mut scratch = board.clone();
    place_stone(&mut scratch, mv, side);
    let reply = best_trajectory_reply(&mut scratch, side);
    remove_stone(&mut scratch, mv, side);
    if !reply.exists {
        return None;
    }
    Some(reply.post.own_pressure - reply.post.opp_pressure)
}

fn child_pressure_delta(board: &Board, mv: Move, side: Stone) -> Option<i32> {
    if !board.is_empty(mv) {
        return None;
    }
    let mut scratch = board.clone();
    place_stone(&mut scratch, mv, side);
    let child = pressure_snapshot(&scratch, side);
    Some(child.own_pressure - child.opp_pressure)
}

fn root_rollout_fast_map<I>(board: &Board, moves: I) -> BTreeMap<Move, Value>
where
    I: IntoIterator<Item = Move>,
{
    let side = board.side_to_move;
    let plies = root_rollout_fast_plies();
    let gamma = root_rollout_fast_gamma();
    let mut out = BTreeMap::new();
    for mv in moves {
        if board.is_empty(mv) && !out.contains_key(&mv) {
            out.insert(
                mv,
                rollout_fast_features_json(board, mv, side, plies, gamma),
            );
        }
    }
    out
}

fn root_rollout_fast_rule_metrics(
    board: &Board,
    incumbent: Move,
    candidate: Move,
) -> Option<RolloutFastRuleMetrics> {
    if !board.is_empty(incumbent) || !board.is_empty(candidate) {
        return None;
    }
    let side = board.side_to_move;
    let candidate_own_kind = threat_bin(board, candidate, side);
    let plies = root_rollout_fast_plies();
    let gamma = root_rollout_fast_gamma();
    let incumbent_record = rollout_fast_feature_record(board, incumbent, side, plies, gamma)?;
    let candidate_record = rollout_fast_feature_record(board, candidate, side, plies, gamma)?;
    let incumbent_opp_force_delay = rollout_fast_opp_force_delay(&incumbent_record)?;
    let candidate_opp_force_delay = rollout_fast_opp_force_delay(&candidate_record)?;
    Some(RolloutFastRuleMetrics {
        candidate_own_kind,
        incumbent_opp_force_delay,
        candidate_opp_force_delay,
        opp_force_delay_delta: candidate_opp_force_delay - incumbent_opp_force_delay,
    })
}

fn root_rollout_fast_rule_allows(
    mode: RolloutFastRuleMode,
    min_own_kind: usize,
    metrics: Option<RolloutFastRuleMetrics>,
) -> bool {
    match mode {
        RolloutFastRuleMode::Off => true,
        RolloutFastRuleMode::OwnThreatOppDelay => metrics
            .map(|metrics| {
                metrics.candidate_own_kind >= min_own_kind
                    && metrics.opp_force_delay_delta >= -f32::EPSILON
            })
            .unwrap_or(false),
    }
}

fn rollout_fast_opp_force_delay(record: &RolloutFastFeatureRecord) -> Option<f32> {
    record
        .features
        .get(ROLLOUT_FAST_FIRST_OPP_FORCE_DELAY_INDEX)
        .copied()
}

fn rollout_fast_features_json(
    board: &Board,
    mv: Move,
    side: Stone,
    plies: usize,
    gamma: f32,
) -> Value {
    let Some(record) = rollout_fast_feature_record(board, mv, side, plies, gamma) else {
        return Value::Null;
    };
    let mut features = serde_json::Map::new();
    for (name, value) in ROLLOUT_FAST_FEATURE_NAMES
        .iter()
        .zip(record.features.iter().copied())
    {
        features.insert((*name).to_string(), json!(value));
    }
    json!({
        "selector": "order",
        "plies": plies,
        "gamma": record.gamma,
        "terminal": record.terminal,
        "snapshots": record.snapshots,
        "moves": record.moves,
        "features": Value::Object(features),
    })
}

fn rollout_fast_feature_record(
    board: &Board,
    mv: Move,
    side: Stone,
    plies: usize,
    gamma: f32,
) -> Option<RolloutFastFeatureRecord> {
    if !board.is_empty(mv) {
        return None;
    }
    let mut scratch = board.clone();
    place_stone(&mut scratch, mv, side);
    let mut snapshots = vec![(0usize, pressure_snapshot(&scratch, side))];
    let mut metas = Vec::with_capacity(plies);
    let mut terminal = 0i32;
    let mut actor = side.opponent();
    for delay in 1..=plies {
        let Some(candidate) = choose_rollout_fast_move(&scratch, actor) else {
            break;
        };
        place_stone(&mut scratch, candidate.mv, actor);
        metas.push(RolloutFastMoveMeta {
            actor,
            own_kind: candidate.own_kind,
        });
        snapshots.push((delay, pressure_snapshot(&scratch, side)));
        if is_five_at(&scratch, candidate.mv, actor) {
            terminal = if actor == side { 1 } else { -1 };
            break;
        }
        actor = actor.opponent();
    }

    let deltas = snapshots
        .iter()
        .map(|(_, snapshot)| snapshot.own_pressure - snapshot.opp_pressure)
        .collect::<Vec<_>>();
    let own_forces = snapshots
        .iter()
        .map(|(_, snapshot)| snapshot.own_immediate + snapshot.own_open_four)
        .collect::<Vec<_>>();
    let opp_forces = snapshots
        .iter()
        .map(|(_, snapshot)| snapshot.opp_immediate + snapshot.opp_open_four)
        .collect::<Vec<_>>();
    let gamma = if gamma.is_finite() && gamma >= 0.0 {
        gamma
    } else {
        DEFAULT_ROLLOUT_FAST_GAMMA
    };
    let mut discount_sum = 0.0f32;
    let mut area_sum = 0.0f32;
    let mut debt_sum = 0.0f32;
    for (delay, delta) in snapshots
        .iter()
        .map(|(delay, _)| *delay)
        .zip(deltas.iter().copied())
    {
        let discount = gamma.powi(delay as i32);
        discount_sum += discount;
        area_sum += discount * capped_signed(delta as f32, 80.0);
        debt_sum += discount * pos_capped(-(delta as f32), 80.0);
    }
    if discount_sum <= 0.0 {
        discount_sum = 1.0;
    }
    let cap = plies;
    let first_own_force = first_rollout_delay(&snapshots, cap, |snapshot| {
        snapshot.own_immediate + snapshot.own_open_four > 0
    });
    let first_opp_force = first_rollout_delay(&snapshots, cap, |snapshot| {
        snapshot.opp_immediate + snapshot.opp_open_four > 0
    });
    let first_own_lead = first_rollout_delay(&snapshots, cap, |snapshot| {
        snapshot.own_pressure > snapshot.opp_pressure
    });
    let first_opp_lead = first_rollout_delay(&snapshots, cap, |snapshot| {
        snapshot.opp_pressure > snapshot.own_pressure
    });
    let cap_plus = (cap + 1).max(1) as f32;
    let move_cap = cap.max(1) as f32;
    let fside_move_count = metas.iter().filter(|meta| meta.actor == side).count();
    let opp_move_count = metas.iter().filter(|meta| meta.actor != side).count();

    let mut features = vec![
        capped_signed(*deltas.iter().min().unwrap_or(&0) as f32, 80.0),
        capped_signed(*deltas.iter().max().unwrap_or(&0) as f32, 80.0),
        capped_signed(*deltas.last().unwrap_or(&0) as f32, 80.0),
        area_sum / discount_sum,
        debt_sum / discount_sum,
        capped_signed(terminal as f32, 1.0),
        pos_capped(first_own_force as f32, cap_plus),
        pos_capped(first_opp_force as f32, cap_plus),
        pos_capped(first_own_lead as f32, cap_plus),
        pos_capped(first_opp_lead as f32, cap_plus),
        pos_capped(*own_forces.iter().max().unwrap_or(&0) as f32, 6.0),
        pos_capped(*opp_forces.iter().max().unwrap_or(&0) as f32, 6.0),
        capped_signed(rollout_delta_at(&snapshots, &deltas, 0) as f32, 80.0),
        capped_signed(rollout_delta_at(&snapshots, &deltas, 2) as f32, 80.0),
        capped_signed(rollout_delta_at(&snapshots, &deltas, 4) as f32, 80.0),
        capped_signed(rollout_delta_at(&snapshots, &deltas, 8) as f32, 80.0),
        pos_capped(fside_move_count as f32, move_cap),
        pos_capped(opp_move_count as f32, move_cap),
    ];
    for owner in [side, side.opponent()] {
        for bin in 0..KIND_COUNT {
            let count = metas
                .iter()
                .filter(|meta| meta.actor == owner && meta.own_kind == bin)
                .count();
            features.push(pos_capped(count as f32, move_cap));
        }
    }
    debug_assert_eq!(features.len(), ROLLOUT_FAST_COUNT);
    Some(RolloutFastFeatureRecord {
        terminal,
        snapshots: snapshots.len(),
        moves: metas.len(),
        gamma,
        features,
    })
}

fn choose_rollout_fast_move(board: &Board, side: Stone) -> Option<RolloutFastCandidate> {
    let mut candidates = candidate_moves_sorted(board)
        .into_iter()
        .filter(|&mv| board.is_empty(mv))
        .map(|mv| {
            let own_kind = threat_bin(board, mv, side);
            let block_kind = threat_bin(board, mv, side.opponent());
            let order_score = move_order_score(board, mv, side, own_kind, block_kind);
            let (row, col) = to_rc(mv);
            let row = row as i32;
            let col = col as i32;
            let center_dist = (col - 7).abs() + (row - 7).abs();
            RolloutFastCandidate {
                mv,
                own_kind,
                block_kind,
                order_score,
                center_dist,
                row,
                col,
            }
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        b.order_score
            .cmp(&a.order_score)
            .then_with(|| {
                attack_tier(b.own_kind)
                    .max(block_tier(b.block_kind))
                    .cmp(&attack_tier(a.own_kind).max(block_tier(a.block_kind)))
            })
            .then_with(|| a.center_dist.cmp(&b.center_dist))
            .then_with(|| a.row.cmp(&b.row))
            .then_with(|| a.col.cmp(&b.col))
    });
    candidates.into_iter().next()
}

fn first_rollout_delay<F>(
    snapshots: &[(usize, PressureSnapshot)],
    cap: usize,
    predicate: F,
) -> usize
where
    F: Fn(PressureSnapshot) -> bool,
{
    snapshots
        .iter()
        .find_map(|(delay, snapshot)| predicate(*snapshot).then_some(*delay))
        .unwrap_or(cap + 1)
}

fn rollout_delta_at(
    snapshots: &[(usize, PressureSnapshot)],
    deltas: &[i32],
    target_delay: usize,
) -> i32 {
    snapshots
        .iter()
        .zip(deltas.iter())
        .filter(|((delay, _), _)| *delay <= target_delay)
        .map(|(_, delta)| *delta)
        .last()
        .unwrap_or_else(|| *deltas.last().unwrap_or(&0))
}

fn root_feature_rows(
    board: &Board,
    weights: &NnueWeights,
    include_rank_delta: bool,
    include_post_reply: bool,
    include_trajectory_child_static: bool,
    include_trajectory_post_reply: bool,
    include_rollout_fast: bool,
) -> Vec<(Move, Vec<f32>)> {
    let side = board.side_to_move;
    let mut candidates = candidate_moves_sorted(board);
    candidates.retain(|&mv| board.is_empty(mv));
    let candidate_count = candidates.len();
    let values = value_records(board, weights, &candidates);
    let pre = threat_counts(board, side);
    let pre_opp_immediate = immediate_winning_moves(board, side.opponent())
        .into_iter()
        .collect::<BTreeSet<_>>();
    let ply = board.move_count;
    let mut scratch = board.clone();

    let mut rows = candidates
        .into_iter()
        .map(|mv| {
            let probe = evaluate_candidate(&mut scratch, mv, side);
            let value = values.get(&mv);
            let features = candidate_local_features(
                &mut scratch,
                mv,
                side,
                ply,
                candidate_count,
                probe,
                value,
                pre,
                &pre_opp_immediate,
            );
            debug_assert_eq!(features.len(), FEATURE_COUNT);
            (mv, features)
        })
        .collect::<Vec<_>>();
    if include_rank_delta || include_post_reply {
        append_rank_delta_features(&mut rows);
    }
    if include_post_reply {
        append_post_reply_features(board, &mut rows, &values, pre);
    }
    if include_trajectory_child_static {
        append_trajectory_child_static_features(board, &mut rows, side, ply);
    }
    if include_trajectory_post_reply {
        append_trajectory_post_reply_features(board, &mut rows, side, ply);
    }
    if include_rollout_fast {
        append_rollout_fast_features(board, &mut rows, side);
    }
    rows
}

fn append_rank_delta_features(rows: &mut [(Move, Vec<f32>)]) {
    if rows.is_empty() {
        return;
    }
    let count = rows.len().max(1);
    let mut order = rows
        .iter()
        .map(|(mv, features)| (*mv, features.get(25).copied().unwrap_or(0.0)))
        .collect::<Vec<_>>();
    order.sort_by(|(move_a, score_a), (move_b, score_b)| {
        score_b
            .partial_cmp(score_a)
            .unwrap_or(Ordering::Equal)
            .then_with(|| move_a.cmp(move_b))
    });
    let best = order.first().map(|(_, score)| *score).unwrap_or(0.0);
    let worst = order.last().map(|(_, score)| *score).unwrap_or(0.0);
    let span = (best - worst).abs().max(1.0e-6);
    let ranks = order
        .into_iter()
        .enumerate()
        .map(|(idx, (mv, score))| {
            let rank = idx + 1;
            let best_gap = best - score;
            let percentile = (score - worst) / span;
            (mv, (rank, best_gap, percentile))
        })
        .collect::<BTreeMap<_, _>>();

    for (mv, features) in rows {
        let (rank, order_gap, order_percentile) =
            ranks.get(mv).copied().unwrap_or((count, 0.0, 0.0));
        let order_rank_frac = rank as f32 / count as f32;
        let child_rank = features.get(VALUE_ONLY_OFFSET + 5).copied().unwrap_or(1.0);
        let reply_rank = features.get(VALUE_ONLY_OFFSET + 12).copied().unwrap_or(1.0);
        features.extend([
            order_rank_frac,
            capped_signed(order_gap, 1.0),
            order_percentile,
            child_rank,
            reply_rank,
            child_rank - order_rank_frac,
            reply_rank - order_rank_frac,
            child_rank - reply_rank,
            order_rank_frac - child_rank,
            if rank == 1 { 1.0 } else { 0.0 },
            if child_rank <= 0.02 { 1.0 } else { 0.0 },
            if reply_rank <= 0.10 { 1.0 } else { 0.0 },
        ]);
        debug_assert_eq!(features.len(), FEATURE_COUNT + RANK_DELTA_COUNT);
    }
}

fn append_post_reply_features(
    board: &Board,
    rows: &mut [(Move, Vec<f32>)],
    values: &BTreeMap<Move, ValueRecord>,
    pre: ThreatCounts,
) {
    let side = board.side_to_move;
    let mut scratch = board.clone();
    for (mv, features) in rows {
        let post = post_reply_features(&mut scratch, *mv, side, values.get(mv), pre);
        debug_assert_eq!(post.len(), POST_REPLY_COUNT);
        features.extend(post);
        debug_assert_eq!(
            features.len(),
            FEATURE_COUNT + RANK_DELTA_COUNT + POST_REPLY_COUNT
        );
    }
}

fn append_trajectory_child_static_features(
    board: &Board,
    rows: &mut [(Move, Vec<f32>)],
    side: Stone,
    ply: usize,
) {
    let pre = pressure_snapshot(board, side);
    let mut scratch = board.clone();
    for (mv, features) in rows {
        if features.len() < TRAJECTORY_CHILD_STATIC_OFFSET {
            features.resize(TRAJECTORY_CHILD_STATIC_OFFSET, 0.0);
        }
        let trajectory = trajectory_child_static_features(&mut scratch, *mv, side, ply, pre);
        debug_assert_eq!(trajectory.len(), TRAJECTORY_CHILD_STATIC_COUNT);
        features.extend(trajectory);
        debug_assert_eq!(
            features.len(),
            TRAJECTORY_CHILD_STATIC_OFFSET + TRAJECTORY_CHILD_STATIC_COUNT
        );
    }
}

fn append_trajectory_post_reply_features(
    board: &Board,
    rows: &mut [(Move, Vec<f32>)],
    side: Stone,
    ply: usize,
) {
    let pre = pressure_snapshot(board, side);
    let mut scratch = board.clone();
    for (mv, features) in rows {
        if features.len() < TRAJECTORY_POST_REPLY_OFFSET {
            features.resize(TRAJECTORY_POST_REPLY_OFFSET, 0.0);
        }
        let trajectory = trajectory_post_reply_features(&mut scratch, *mv, side, ply, pre);
        debug_assert_eq!(trajectory.len(), TRAJECTORY_POST_REPLY_COUNT);
        features.extend(trajectory);
        debug_assert_eq!(
            features.len(),
            TRAJECTORY_POST_REPLY_OFFSET + TRAJECTORY_POST_REPLY_COUNT
        );
    }
}

fn append_rollout_fast_features(board: &Board, rows: &mut [(Move, Vec<f32>)], side: Stone) {
    let plies = root_rollout_fast_plies();
    let gamma = root_rollout_fast_gamma();
    for (mv, features) in rows {
        if features.len() < ROLLOUT_FAST_OFFSET {
            features.resize(ROLLOUT_FAST_OFFSET, 0.0);
        }
        let rollout = rollout_fast_feature_record(board, *mv, side, plies, gamma)
            .map(|record| record.features)
            .unwrap_or_else(|| vec![0.0; ROLLOUT_FAST_COUNT]);
        debug_assert_eq!(rollout.len(), ROLLOUT_FAST_COUNT);
        features.extend(rollout);
        debug_assert_eq!(features.len(), ROLLOUT_FAST_OFFSET + ROLLOUT_FAST_COUNT);
    }
}

fn value_records(
    board: &Board,
    weights: &NnueWeights,
    candidates: &[Move],
) -> BTreeMap<Move, ValueRecord> {
    let root_eval = eval_root(board, weights);
    let mut values = candidates
        .iter()
        .copied()
        .map(|mv| score_candidate(board, mv, weights, top_replies()))
        .collect::<Vec<_>>();
    values.sort_by(|a, b| {
        b.child_eval
            .partial_cmp(&a.child_eval)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.mv.cmp(&b.mv))
    });
    let best_child_eval = values.first().map(|v| v.child_eval).unwrap_or(root_eval);
    let worst_child_eval = values.last().map(|v| v.child_eval).unwrap_or(root_eval);
    let mut best_reply_sorted = values.clone();
    best_reply_sorted.sort_by(|a, b| {
        b.best_reply_eval
            .partial_cmp(&a.best_reply_eval)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.mv.cmp(&b.mv))
    });
    let best_reply_eval_global = best_reply_sorted
        .first()
        .map(|v| v.best_reply_eval)
        .unwrap_or(root_eval);
    let worst_reply_eval_global = best_reply_sorted
        .last()
        .map(|v| v.best_reply_eval)
        .unwrap_or(root_eval);
    let child_rank = rank_map(&values, |v| v.child_eval);
    let reply_rank = rank_map(&best_reply_sorted, |v| v.best_reply_eval);
    let child_span = (best_child_eval - worst_child_eval).abs().max(1.0);
    let reply_span = (best_reply_eval_global - worst_reply_eval_global)
        .abs()
        .max(1.0);
    let candidate_count_f32 = candidates.len().max(1) as f32;
    let candidate_count_f64 = candidates.len().max(1) as f64;

    values
        .into_iter()
        .map(|value| {
            let child_rank_idx = *child_rank.get(&value.mv).unwrap_or(&candidates.len());
            let reply_rank_idx = *reply_rank.get(&value.mv).unwrap_or(&candidates.len());
            (
                value.mv,
                ValueRecord {
                    root_eval,
                    child_eval: value.child_eval,
                    child_eval_delta: value.child_eval - root_eval,
                    child_eval_best_gap: best_child_eval - value.child_eval,
                    child_eval_worst_gap: value.child_eval - worst_child_eval,
                    child_eval_rank_frac: child_rank_idx as f64 / candidate_count_f64,
                    child_eval_percentile: (value.child_eval - worst_child_eval) / child_span,
                    opp_best_reply_eval: value.best_reply_eval,
                    opp_best_reply_delta: value.best_reply_eval - value.child_eval,
                    reply_drop: value.child_eval - value.best_reply_eval,
                    reply_eval_best_gap: best_reply_eval_global - value.best_reply_eval,
                    reply_eval_worst_gap: value.best_reply_eval - worst_reply_eval_global,
                    reply_eval_rank_frac: reply_rank_idx as f64 / candidate_count_f64,
                    reply_eval_percentile: (value.best_reply_eval - worst_reply_eval_global)
                        / reply_span,
                    opp_best_reply: value.best_reply,
                },
            )
        })
        .inspect(|(_, row)| {
            debug_assert!(row.child_eval_rank_frac.is_finite());
            debug_assert!(row.reply_eval_rank_frac.is_finite());
            debug_assert!(candidate_count_f32 > 0.0);
        })
        .collect()
}

fn score_candidate(
    board: &Board,
    mv: Move,
    weights: &NnueWeights,
    top_replies: usize,
) -> CandidateValue {
    let mut child = board.clone();
    child.make_move(mv);
    let child_eval = -eval_root(&child, weights);
    let mut replies = child.candidate_moves();
    if top_replies > 0 && replies.len() > top_replies {
        replies.truncate(top_replies);
    }
    let mut best_reply_eval = child_eval;
    let mut best_reply = None;
    let mut initialized = false;
    for reply in replies {
        if !child.is_empty(reply) {
            continue;
        }
        let mut reply_child = child.clone();
        reply_child.make_move(reply);
        let root_eval_after_reply = eval_root(&reply_child, weights);
        if !initialized || root_eval_after_reply < best_reply_eval {
            initialized = true;
            best_reply_eval = root_eval_after_reply;
            best_reply = Some(reply);
        }
    }
    CandidateValue {
        mv,
        child_eval,
        best_reply_eval,
        best_reply,
    }
}

fn candidate_local_features(
    board: &mut Board,
    mv: Move,
    side: Stone,
    ply: usize,
    candidate_count: usize,
    probe: CandidateProbe,
    value: Option<&ValueRecord>,
    pre: ThreatCounts,
    pre_opp_immediate: &BTreeSet<Move>,
) -> Vec<f32> {
    let mut features = Vec::with_capacity(FEATURE_COUNT);
    features.extend(expanded_features(
        board,
        mv,
        side,
        ply,
        candidate_count,
        probe,
    ));
    features.extend(value_feature_vector(value));
    features.extend(value_local_features(value));
    features.extend(tactical_local_features(
        board,
        mv,
        side,
        value,
        pre,
        pre_opp_immediate,
    ));
    features
}

fn expanded_features(
    board: &mut Board,
    mv: Move,
    side: Stone,
    ply: usize,
    candidate_count: usize,
    probe: CandidateProbe,
) -> Vec<f32> {
    let own_kind = probe.own_kind;
    let reply = reply_features(board, mv, side, ply, candidate_count, own_kind);
    let rich = reply_rich_features(board, mv, side, own_kind, &reply);
    let tactical = child_tactical_features(board, mv, side, own_kind, probe);
    let line = line_window_features(board, mv, side);
    let mut out = Vec::with_capacity(168);
    out.extend(rich);
    out.extend(tactical);
    out.extend(line);
    out
}

fn reply_features(
    board: &mut Board,
    mv: Move,
    side: Stone,
    ply: usize,
    candidate_count: usize,
    own_kind: usize,
) -> Vec<f32> {
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
    let mut out = vec![
        x / (BOARD_SIZE as f32 - 1.0),
        y / (BOARD_SIZE as f32 - 1.0),
        (dx * dx + dy * dy).sqrt(),
        edge,
        ply as f32 / NUM_CELLS as f32,
        candidate_count as f32 / NUM_CELLS as f32,
    ];
    push_kind_one_hot(&mut out, own_kind);

    let opp = side.opponent();
    place_stone(board, mv, side);
    let immediate = empty_moves(board)
        .into_iter()
        .filter(|&reply| threat_bin(board, reply, opp) == 7)
        .count();
    let open_four = empty_moves(board)
        .into_iter()
        .filter(|&reply| matches!(threat_bin(board, reply, opp), 5 | 6))
        .count();
    let opponent_candidates = candidate_moves_sorted(board);
    let denom = opponent_candidates.len().max(1) as f32;
    let mut kind_counts = [0usize; KIND_COUNT];
    for reply in opponent_candidates.iter().copied() {
        if board.is_empty(reply) {
            kind_counts[threat_bin(board, reply, opp)] += 1;
        }
    }
    remove_stone(board, mv, side);

    out.push(immediate.min(4) as f32 / 4.0);
    out.push(open_four.min(8) as f32 / 8.0);
    out.push(opponent_candidates.len() as f32 / NUM_CELLS as f32);
    for count in kind_counts {
        out.push(count as f32 / denom);
    }
    out
}

fn reply_rich_features(
    board: &Board,
    mv: Move,
    side: Stone,
    own_kind: usize,
    reply: &[f32],
) -> Vec<f32> {
    let opp = side.opponent();
    let block_kind = threat_bin(board, mv, opp);
    let attack_tier = attack_tier(own_kind);
    let block_tier = block_tier(block_kind);
    let order_score = move_order_score(board, mv, side, own_kind, block_kind);
    let (row, col) = to_rc(mv);
    let last_dist = board
        .last_move
        .map(|last| {
            let (lr, lc) = to_rc(last);
            let dr = row as f32 - lr as f32;
            let dc = col as f32 - lc as f32;
            (dr * dr + dc * dc).sqrt() / BOARD_SIZE as f32
        })
        .unwrap_or(1.0);
    let (my_r1, opp_r1) = neighbor_counts(board, mv, side, 1);
    let (my_r2, opp_r2) = neighbor_counts(board, mv, side, 2);
    let mut extras = vec![
        order_score as f32 / TIER_SCALE,
        attack_tier as f32 / TIER_SCALE,
        block_tier as f32 / TIER_SCALE,
        (attack_tier - block_tier) as f32 / TIER_SCALE,
        if is_forcing_bin(own_kind) { 1.0 } else { 0.0 },
        if is_forcing_bin(block_kind) { 1.0 } else { 0.0 },
        if own_kind == 0 && block_kind == 0 {
            1.0
        } else {
            0.0
        },
        if matches!(own_kind, 1 | 3) && block_kind == 0 {
            1.0
        } else {
            0.0
        },
        last_dist,
        my_r1 as f32 / 8.0,
        opp_r1 as f32 / 8.0,
        my_r2 as f32 / 24.0,
        opp_r2 as f32 / 24.0,
    ];
    push_kind_one_hot(&mut extras, own_kind);
    push_kind_one_hot(&mut extras, block_kind);
    for attack in 0..KIND_COUNT {
        for block in 0..KIND_COUNT {
            extras.push(if own_kind == attack && block_kind == block {
                1.0
            } else {
                0.0
            });
        }
    }
    let mut out = Vec::with_capacity(reply.len() + extras.len());
    out.extend_from_slice(reply);
    out.extend(extras);
    out
}

fn child_tactical_features(
    board: &mut Board,
    mv: Move,
    side: Stone,
    own_kind: usize,
    probe: CandidateProbe,
) -> Vec<f32> {
    let opp = side.opponent();
    let block_kind = threat_bin(board, mv, opp);
    let attack_tier = attack_tier(own_kind);
    let block_tier = block_tier(block_kind);
    let before_opp_immediate = immediate_winning_moves(board, opp).len();
    let before_opp_open_four = open_four_replies_count(board, opp);

    place_stone(board, mv, side);
    let child_wins = is_five_at(board, mv, side);
    let opp_moves = candidate_moves_sorted(board);
    let own_moves = opp_moves.clone();
    let opp_immediate = immediate_winning_moves(board, opp).len();
    let own_immediate = immediate_winning_moves(board, side).len();
    let opp_open_four = open_four_replies_count(board, opp);
    let own_open_four = open_four_replies_count(board, side);
    let opp_forcing = forcing_share(board, opp, &opp_moves);
    let own_forcing = forcing_share(board, side, &own_moves);
    let opp_max_tier = max_attack_tier(board, opp, &opp_moves);
    let own_max_tier = max_attack_tier(board, side, &own_moves);
    remove_stone(board, mv, side);

    let (my_r3, opp_r3) = neighbor_counts(board, mv, side, 3);
    let unsafe_move = probe.status == ProbeStatus::Unsafe;
    let unsafe_kind = probe.unsafe_kind;
    let mut out = vec![
        if probe.status == ProbeStatus::WinsNow {
            1.0
        } else {
            0.0
        },
        if probe.status == ProbeStatus::Survives {
            1.0
        } else {
            0.0
        },
        if unsafe_move { 1.0 } else { 0.0 },
        if unsafe_kind.contains("immediate") || unsafe_kind.contains("wins_now") {
            1.0
        } else {
            0.0
        },
        if unsafe_kind.contains("open_four") {
            1.0
        } else {
            0.0
        },
        if unsafe_kind.contains("a1") { 1.0 } else { 0.0 },
        if unsafe_kind.contains("a2") { 1.0 } else { 0.0 },
        block_tier as f32 / TIER_SCALE,
        (attack_tier - block_tier) as f32 / TIER_SCALE,
        if is_forcing_bin(own_kind) && is_forcing_bin(block_kind) {
            1.0
        } else {
            0.0
        },
        if child_wins { 1.0 } else { 0.0 },
        capped_pos(opp_immediate as f32, 4.0),
        capped_pos(opp_open_four as f32, 8.0),
        opp_forcing,
        opp_max_tier as f32 / TIER_SCALE,
        capped_pos(own_immediate as f32, 4.0),
        capped_pos(own_open_four as f32, 8.0),
        own_forcing,
        own_max_tier as f32 / TIER_SCALE,
        capped_pos(own_immediate as f32 - opp_immediate as f32 + 4.0, 8.0),
        capped_pos(own_open_four as f32 - opp_open_four as f32 + 8.0, 16.0),
        (own_forcing - opp_forcing + 1.0) / 2.0,
        (((own_max_tier - opp_max_tier) as f32 / TIER_SCALE) + 1.0) / 2.0,
        if opp_immediate == 0 { 1.0 } else { 0.0 },
        if opp_open_four == 0 { 1.0 } else { 0.0 },
        my_r3 as f32 / 48.0,
        opp_r3 as f32 / 48.0,
        (my_r3 as f32 - opp_r3 as f32 + 48.0) / 96.0,
    ];
    out[19] = capped_pos(
        before_opp_immediate as f32 - opp_immediate as f32 + 4.0,
        8.0,
    );
    out[20] = capped_pos(
        before_opp_open_four as f32 - opp_open_four as f32 + 8.0,
        16.0,
    );
    out
}

fn line_window_features(board: &mut Board, mv: Move, side: Stone) -> Vec<f32> {
    let attack = window_stats_for_color(board, mv, side);
    let block = window_stats_for_color(board, mv, side.opponent());
    let mut out = Vec::with_capacity(22);
    out.extend_from_slice(&attack);
    out.extend_from_slice(&block);
    out.push((attack[0] - block[0] + 1.0) / 2.0);
    out.push((attack[2] - block[2] + 1.0) / 2.0);
    out.push(block[2]);
    out.push(attack[2]);
    out
}

fn window_stats_for_color(board: &mut Board, mv: Move, side: Stone) -> [f32; 9] {
    let (row, col) = to_rc(mv);
    let row = row as isize;
    let col = col as isize;
    let own = stone_code(side);
    let opp = stone_code(side.opponent());
    let mut max_stones = 0usize;
    let mut window4 = 0usize;
    let mut open_window4 = 0usize;
    let mut window3 = 0usize;
    let mut open_window3 = 0usize;
    let mut window2 = 0usize;
    let mut max_open_ends = 0usize;
    let mut best_contiguous = 0usize;
    let mut best_contiguous_open_ends = 0usize;

    place_stone(board, mv, side);
    for &(dr_i32, dc_i32) in &DIR {
        let dr = dr_i32 as isize;
        let dc = dc_i32 as isize;
        let mut line = [0i8; 9];
        for offset in -4isize..=4 {
            line[(offset + 4) as usize] = cell_code(board, row + dr * offset, col + dc * offset);
        }
        for start in 0..5 {
            let window = &line[start..start + 5];
            if window.iter().any(|&cell| cell == -1 || cell == opp) {
                continue;
            }
            let stones = window.iter().filter(|&&cell| cell == own).count();
            let empties = window.iter().filter(|&&cell| cell == 0).count();
            let before = if start > 0 { line[start - 1] } else { -1 };
            let after = if start + 5 < line.len() {
                line[start + 5]
            } else {
                -1
            };
            let open_ends = usize::from(before == 0) + usize::from(after == 0);
            max_stones = max_stones.max(stones);
            max_open_ends = max_open_ends.max(open_ends);
            if stones == 4 && empties == 1 {
                window4 += 1;
                if open_ends > 0 {
                    open_window4 += 1;
                }
            }
            if stones == 3 && empties == 2 {
                window3 += 1;
                if open_ends == 2 {
                    open_window3 += 1;
                }
            }
            if stones == 2 && empties == 3 && open_ends == 2 {
                window2 += 1;
            }
        }

        let mut count = 1usize;
        let mut nr = row + dr;
        let mut nc = col + dc;
        while cell_code(board, nr, nc) == own {
            count += 1;
            nr += dr;
            nc += dc;
        }
        let front_open = cell_code(board, nr, nc) == 0;
        nr = row - dr;
        nc = col - dc;
        while cell_code(board, nr, nc) == own {
            count += 1;
            nr -= dr;
            nc -= dc;
        }
        let back_open = cell_code(board, nr, nc) == 0;
        if count > best_contiguous {
            best_contiguous = count;
            best_contiguous_open_ends = usize::from(front_open) + usize::from(back_open);
        }
    }
    remove_stone(board, mv, side);

    [
        max_stones as f32 / 5.0,
        capped_pos(window4 as f32, 4.0),
        capped_pos(open_window4 as f32, 4.0),
        capped_pos(window3 as f32, 8.0),
        capped_pos(open_window3 as f32, 4.0),
        capped_pos(window2 as f32, 8.0),
        max_open_ends as f32 / 2.0,
        best_contiguous.min(5) as f32 / 5.0,
        best_contiguous_open_ends as f32 / 2.0,
    ]
}

fn value_feature_vector(value: Option<&ValueRecord>) -> Vec<f32> {
    let Some(v) = value else {
        return vec![0.0; 14];
    };
    let scale = 1000.0;
    vec![
        v.root_eval / scale,
        v.child_eval / scale,
        v.child_eval_delta / scale,
        v.child_eval_best_gap / scale,
        v.child_eval_worst_gap / scale,
        v.child_eval_rank_frac as f32,
        v.child_eval_percentile,
        v.opp_best_reply_eval / scale,
        v.opp_best_reply_delta / scale,
        v.reply_drop / scale,
        v.reply_eval_best_gap / scale,
        v.reply_eval_worst_gap / scale,
        v.reply_eval_rank_frac as f32,
        v.reply_eval_percentile,
    ]
}

fn value_local_features(value: Option<&ValueRecord>) -> Vec<f32> {
    let Some(v) = value else {
        return vec![0.0; 10];
    };
    vec![
        capped_signed(v.child_eval - v.opp_best_reply_eval, 1000.0),
        pos_capped(v.reply_drop, 1000.0),
        pos_capped(-v.reply_drop, 1000.0),
        capped_signed(v.opp_best_reply_eval - v.root_eval, 1000.0),
        capped_signed(v.child_eval_best_gap, 1000.0),
        capped_signed(v.reply_eval_best_gap, 1000.0),
        if v.child_eval_rank_frac <= 0.02 {
            1.0
        } else {
            0.0
        },
        if v.reply_eval_rank_frac <= 0.10 {
            1.0
        } else {
            0.0
        },
        v.child_eval_percentile,
        v.reply_eval_percentile,
    ]
}

fn tactical_local_features(
    board: &mut Board,
    mv: Move,
    side: Stone,
    value: Option<&ValueRecord>,
    pre: ThreatCounts,
    pre_opp_immediate: &BTreeSet<Move>,
) -> Vec<f32> {
    let opp = side.opponent();
    let own_kind = threat_bin(board, mv, side);
    let block_kind = threat_bin(board, mv, opp);
    let child_counts;
    let mut after_reply = ThreatCounts::default();
    let mut best_reply_kind = 0usize;
    let mut best_reply_block_kind = 0usize;
    let mut best_reply_wins = false;
    let mut best_reply_blocks_candidate = false;

    place_stone(board, mv, side);
    child_counts = threat_counts(board, side);
    if let Some(reply) = value.and_then(|v| v.opp_best_reply) {
        if board.is_empty(reply) {
            best_reply_kind = threat_bin(board, reply, opp);
            best_reply_block_kind = threat_bin(board, reply, side);
            place_stone(board, reply, opp);
            best_reply_wins = is_five_at(board, reply, opp);
            after_reply = threat_counts(board, side);
            remove_stone(board, reply, opp);
            best_reply_blocks_candidate = is_forcing_bin(best_reply_block_kind);
        }
    }
    remove_stone(board, mv, side);

    let attack_score_tier = attack_tier(own_kind);
    let block_score_tier = block_tier(block_kind);
    let reply_attack_tier = attack_tier(best_reply_kind);
    let reply_block_tier = block_tier(best_reply_block_kind);
    let candidate_wins = own_kind == 7;
    let mut out = vec![
        pos_capped(pre.opp_immediate as f32, 4.0),
        pos_capped(pre.opp_open_four as f32, 8.0),
        pos_capped(pre.own_immediate as f32, 4.0),
        pos_capped(pre.own_open_four as f32, 8.0),
        pos_capped(child_counts.opp_immediate as f32, 4.0),
        pos_capped(child_counts.opp_open_four as f32, 8.0),
        pos_capped(child_counts.own_immediate as f32, 4.0),
        pos_capped(child_counts.own_open_four as f32, 8.0),
        pos_capped(after_reply.own_immediate as f32, 4.0),
        pos_capped(after_reply.own_open_four as f32, 8.0),
        pos_capped(after_reply.opp_immediate as f32, 4.0),
        pos_capped(after_reply.opp_open_four as f32, 8.0),
        attack_score_tier as f32 / TIER_SCALE,
        block_score_tier as f32 / TIER_SCALE,
        (attack_score_tier - block_score_tier) as f32 / TIER_SCALE,
        if pre_opp_immediate.contains(&mv) {
            1.0
        } else {
            0.0
        },
        if candidate_wins { 1.0 } else { 0.0 },
        if is_forcing_bin(own_kind) { 1.0 } else { 0.0 },
        if is_forcing_bin(block_kind) { 1.0 } else { 0.0 },
        reply_attack_tier as f32 / TIER_SCALE,
        reply_block_tier as f32 / TIER_SCALE,
        (reply_attack_tier - reply_block_tier) as f32 / TIER_SCALE,
        if best_reply_wins { 1.0 } else { 0.0 },
        if is_forcing_bin(best_reply_kind) {
            1.0
        } else {
            0.0
        },
        if is_forcing_bin(best_reply_block_kind) {
            1.0
        } else {
            0.0
        },
        if best_reply_blocks_candidate {
            1.0
        } else {
            0.0
        },
    ];
    push_urgency_one_hot(&mut out, pre);
    push_kind_one_hot(&mut out, own_kind);
    push_kind_one_hot(&mut out, block_kind);
    push_kind_one_hot(&mut out, best_reply_kind);
    push_kind_one_hot(&mut out, best_reply_block_kind);
    out
}

fn post_reply_features(
    board: &mut Board,
    mv: Move,
    side: Stone,
    value: Option<&ValueRecord>,
    pre: ThreatCounts,
) -> Vec<f32> {
    let opp = side.opponent();
    let mut child = ThreatCounts::default();
    let mut after_reply = ThreatCounts::default();
    let mut best_reply_kind = 0usize;
    let mut best_reply_block_kind = 0usize;
    let mut best_reply_wins = false;
    let mut best_reply_blocks_candidate = false;

    if board.is_empty(mv) {
        place_stone(board, mv, side);
        child = threat_counts(board, side);
        if let Some(reply) = value.and_then(|v| v.opp_best_reply) {
            if board.is_empty(reply) {
                best_reply_kind = threat_bin(board, reply, opp);
                best_reply_block_kind = threat_bin(board, reply, side);
                place_stone(board, reply, opp);
                best_reply_wins = is_five_at(board, reply, opp);
                after_reply = threat_counts(board, side);
                remove_stone(board, reply, opp);
                best_reply_blocks_candidate = is_forcing_bin(best_reply_block_kind);
            }
        }
        remove_stone(board, mv, side);
    }

    let reply_attack_tier = attack_tier(best_reply_kind);
    let reply_block_tier = block_tier(best_reply_block_kind);
    let mut out = Vec::with_capacity(POST_REPLY_COUNT);
    push_post_count_features(&mut out, pre);
    push_post_count_features(&mut out, child);
    push_post_count_features(&mut out, after_reply);
    out.extend([
        reply_attack_tier as f32 / TIER_SCALE,
        reply_block_tier as f32 / TIER_SCALE,
        (reply_attack_tier - reply_block_tier) as f32 / TIER_SCALE,
        if best_reply_wins { 1.0 } else { 0.0 },
        if is_forcing_bin(best_reply_kind) {
            1.0
        } else {
            0.0
        },
        if is_forcing_bin(best_reply_block_kind) {
            1.0
        } else {
            0.0
        },
        if best_reply_blocks_candidate {
            1.0
        } else {
            0.0
        },
    ]);
    push_kind_one_hot(&mut out, best_reply_kind);
    push_kind_one_hot(&mut out, best_reply_block_kind);
    out
}

fn push_post_count_features(out: &mut Vec<f32>, counts: ThreatCounts) {
    out.extend([
        pos_capped(counts.opp_immediate as f32, 8.0),
        pos_capped(counts.opp_open_four as f32, 8.0),
        pos_capped(counts.own_immediate as f32, 8.0),
        pos_capped(counts.own_open_four as f32, 8.0),
    ]);
}

fn trajectory_child_static_features(
    board: &mut Board,
    mv: Move,
    side: Stone,
    ply: usize,
    pre: PressureSnapshot,
) -> Vec<f32> {
    let (row, col) = to_rc(mv);
    let row_i = row as i32;
    let col_i = col as i32;
    let attack_kind = if board.is_empty(mv) {
        threat_bin(board, mv, side)
    } else {
        0
    };
    let block_kind = if board.is_empty(mv) {
        threat_bin(board, mv, side.opponent())
    } else {
        0
    };
    let attack_tier = attack_tier(attack_kind);
    let block_tier = block_tier(block_kind);
    let mut child = pre;
    if board.is_empty(mv) {
        place_stone(board, mv, side);
        child = pressure_snapshot(board, side);
        remove_stone(board, mv, side);
    }

    let mut out = Vec::with_capacity(TRAJECTORY_CHILD_STATIC_COUNT);
    out.extend([
        if side == Stone::Black { 1.0 } else { 0.0 },
        if side == Stone::White { 1.0 } else { 0.0 },
        capped_signed((ply + 1) as f32, 80.0),
        capped_signed((col_i - 7) as f32, 7.0),
        capped_signed((row_i - 7) as f32, 7.0),
        pos_capped(((col_i - 7).abs() + (row_i - 7).abs()) as f32, 14.0),
        attack_tier as f32 / TIER_SCALE,
        block_tier as f32 / TIER_SCALE,
        (attack_tier - block_tier) as f32 / TIER_SCALE,
        (block_tier - attack_tier) as f32 / TIER_SCALE,
        if attack_kind != 0 || block_kind != 0 {
            1.0
        } else {
            0.0
        },
    ]);
    push_pressure_snapshot_features(&mut out, pre);
    push_kind_one_hot(&mut out, attack_kind);
    push_kind_one_hot(&mut out, block_kind);
    push_pressure_snapshot_features(&mut out, child);

    let pre_own_force = pre.own_immediate + pre.own_open_four;
    let pre_opp_force = pre.opp_immediate + pre.opp_open_four;
    let child_own_force = child.own_immediate + child.own_open_four;
    let child_opp_force = child.opp_immediate + child.opp_open_four;
    out.extend([
        capped_signed((child.own_pressure - pre.own_pressure) as f32, 80.0),
        capped_signed((child.opp_pressure - pre.opp_pressure) as f32, 80.0),
        capped_signed(
            ((child.own_pressure - child.opp_pressure) - (pre.own_pressure - pre.opp_pressure))
                as f32,
            80.0,
        ),
        pos_capped(child_own_force as f32 - pre_own_force as f32, 6.0),
        pos_capped(child_opp_force as f32 - pre_opp_force as f32, 6.0),
    ]);
    out
}

fn trajectory_post_reply_features(
    board: &mut Board,
    mv: Move,
    side: Stone,
    ply: usize,
    pre: PressureSnapshot,
) -> Vec<f32> {
    let mut out = trajectory_child_static_features(board, mv, side, ply, pre);
    let mut child = pre;
    let mut reply = TrajectoryReplyRecord {
        exists: false,
        mv: to_idx(7, 7),
        own_kind: 0,
        block_kind: 0,
        order_score: 0,
        post: PressureSnapshot::default(),
    };
    if board.is_empty(mv) {
        place_stone(board, mv, side);
        child = pressure_snapshot(board, side);
        reply = best_trajectory_reply(board, side);
        remove_stone(board, mv, side);
    }
    let (reply_row, reply_col) = to_rc(reply.mv);
    let reply_row_i = reply_row as i32;
    let reply_col_i = reply_col as i32;
    let center_dist = (reply_col_i - 7).abs() + (reply_row_i - 7).abs();
    let reply_attack_tier = attack_tier(reply.own_kind);
    let reply_block_tier = block_tier(reply.block_kind);
    out.extend([
        if reply.exists { 1.0 } else { 0.0 },
        capped_signed((reply_col_i - 7) as f32, 7.0),
        capped_signed((reply_row_i - 7) as f32, 7.0),
        pos_capped(center_dist as f32, 14.0),
        reply_attack_tier as f32 / TIER_SCALE,
        reply_block_tier as f32 / TIER_SCALE,
        (reply_attack_tier - reply_block_tier) as f32 / TIER_SCALE,
        (reply_block_tier - reply_attack_tier) as f32 / TIER_SCALE,
        reply.order_score as f32 / TIER_SCALE,
        if reply.own_kind != 0 { 1.0 } else { 0.0 },
        if reply.block_kind != 0 { 1.0 } else { 0.0 },
    ]);
    push_kind_one_hot(&mut out, reply.own_kind);
    push_kind_one_hot(&mut out, reply.block_kind);
    push_pressure_snapshot_features(&mut out, reply.post);

    let child_own_force = child.own_immediate + child.own_open_four;
    let child_opp_force = child.opp_immediate + child.opp_open_four;
    let post_own_force = reply.post.own_immediate + reply.post.own_open_four;
    let post_opp_force = reply.post.opp_immediate + reply.post.opp_open_four;
    let child_delta = child.own_pressure - child.opp_pressure;
    let post_delta = reply.post.own_pressure - reply.post.opp_pressure;
    out.extend([
        capped_signed((reply.post.own_pressure - child.own_pressure) as f32, 80.0),
        capped_signed((reply.post.opp_pressure - child.opp_pressure) as f32, 80.0),
        capped_signed((post_delta - child_delta) as f32, 80.0),
        capped_signed(post_own_force as f32 - child_own_force as f32, 6.0),
        capped_signed(post_opp_force as f32 - child_opp_force as f32, 6.0),
        if child_own_force > 0 && post_own_force > 0 {
            1.0
        } else {
            0.0
        },
        if child_opp_force == 0 && post_opp_force > 0 {
            1.0
        } else {
            0.0
        },
        if post_own_force > 0 { 1.0 } else { 0.0 },
        if post_opp_force > 0 { 1.0 } else { 0.0 },
        if post_delta > 0 { 1.0 } else { 0.0 },
        if post_delta < 0 { 1.0 } else { 0.0 },
        if post_delta >= child_delta { 1.0 } else { 0.0 },
    ]);
    out
}

fn best_trajectory_reply(board: &mut Board, side: Stone) -> TrajectoryReplyRecord {
    let reply_side = side.opponent();
    let mut candidates = candidate_moves_sorted(board)
        .into_iter()
        .filter(|&mv| board.is_empty(mv))
        .map(|mv| {
            let own_kind = threat_bin(board, mv, reply_side);
            let block_kind = threat_bin(board, mv, side);
            let order_score = move_order_score(board, mv, reply_side, own_kind, block_kind);
            let (row, col) = to_rc(mv);
            let row = row as i32;
            let col = col as i32;
            let center_dist = (col - 7).abs() + (row - 7).abs();
            TrajectoryReplyCandidate {
                mv,
                own_kind,
                block_kind,
                order_score,
                center_dist,
                row,
                col,
            }
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        b.order_score
            .cmp(&a.order_score)
            .then_with(|| a.center_dist.cmp(&b.center_dist))
            .then_with(|| a.row.cmp(&b.row))
            .then_with(|| a.col.cmp(&b.col))
    });

    let mut best: Option<(TrajectoryReplyRecord, (i32, i32, i32, i32, i32, i32, i32))> = None;
    for candidate in candidates
        .into_iter()
        .take(trajectory_post_reply_top_replies())
    {
        place_stone(board, candidate.mv, reply_side);
        let post = pressure_snapshot(board, side);
        remove_stone(board, candidate.mv, reply_side);
        let post_opp_force = (post.opp_immediate + post.opp_open_four) as i32;
        let post_own_force = (post.own_immediate + post.own_open_four) as i32;
        let pressure_lead = post.opp_pressure - post.own_pressure;
        let key = (
            candidate.order_score,
            post_opp_force,
            pressure_lead,
            -post_own_force,
            -candidate.center_dist,
            -candidate.row,
            -candidate.col,
        );
        let record = TrajectoryReplyRecord {
            exists: true,
            mv: candidate.mv,
            own_kind: candidate.own_kind,
            block_kind: candidate.block_kind,
            order_score: candidate.order_score,
            post,
        };
        if best.map(|(_, best_key)| key > best_key).unwrap_or(true) {
            best = Some((record, key));
        }
    }
    best.map(|(record, _)| record)
        .unwrap_or(TrajectoryReplyRecord {
            exists: false,
            mv: to_idx(7, 7),
            own_kind: 0,
            block_kind: 0,
            order_score: 0,
            post: PressureSnapshot::default(),
        })
}

fn push_pressure_snapshot_features(out: &mut Vec<f32>, snapshot: PressureSnapshot) {
    let own_force = snapshot.own_immediate + snapshot.own_open_four;
    let opp_force = snapshot.opp_immediate + snapshot.opp_open_four;
    out.extend([
        pos_capped(snapshot.own_immediate as f32, 4.0),
        pos_capped(snapshot.own_open_four as f32, 4.0),
        pos_capped(snapshot.opp_immediate as f32, 4.0),
        pos_capped(snapshot.opp_open_four as f32, 4.0),
        pos_capped(own_force as f32, 6.0),
        pos_capped(opp_force as f32, 6.0),
        capped_signed(snapshot.own_pressure as f32, 80.0),
        capped_signed(snapshot.opp_pressure as f32, 80.0),
        capped_signed((snapshot.own_pressure - snapshot.opp_pressure) as f32, 80.0),
    ]);
    for bin in 1..KIND_COUNT {
        out.push(pos_capped(snapshot.own_kinds[bin] as f32, 8.0));
    }
    for bin in 1..KIND_COUNT {
        out.push(pos_capped(snapshot.opp_kinds[bin] as f32, 8.0));
    }
}

fn pressure_snapshot(board: &Board, side: Stone) -> PressureSnapshot {
    let own_kinds = pressure_kind_counts(board, side);
    let opp_kinds = pressure_kind_counts(board, side.opponent());
    PressureSnapshot {
        own_immediate: immediate_winning_moves(board, side).len(),
        own_open_four: open_four_replies_count(board, side),
        opp_immediate: immediate_winning_moves(board, side.opponent()).len(),
        opp_open_four: open_four_replies_count(board, side.opponent()),
        own_pressure: pressure_score(own_kinds),
        opp_pressure: pressure_score(opp_kinds),
        own_kinds,
        opp_kinds,
    }
}

fn pressure_kind_counts(board: &Board, side: Stone) -> [usize; KIND_COUNT] {
    let mut counts = [0usize; KIND_COUNT];
    for mv in candidate_moves_sorted(board) {
        if board.is_empty(mv) {
            let bin = threat_bin(board, mv, side);
            counts[bin] += 1;
        }
    }
    counts
}

fn pressure_score(kinds: [usize; KIND_COUNT]) -> i32 {
    (16 * kinds[7]
        + 12 * kinds[6]
        + 10 * kinds[5]
        + 7 * kinds[4]
        + 5 * kinds[3]
        + 4 * kinds[2]
        + 2 * kinds[1]) as i32
}

fn evaluate_candidate(board: &mut Board, mv: Move, side: Stone) -> CandidateProbe {
    let own_kind = threat_bin(board, mv, side);
    place_stone(board, mv, side);
    if is_five_at(board, mv, side) {
        remove_stone(board, mv, side);
        return CandidateProbe {
            status: ProbeStatus::WinsNow,
            unsafe_kind: "",
            own_kind,
        };
    }
    let force = force_after_candidate(board, side.opponent());
    remove_stone(board, mv, side);
    match force {
        None => CandidateProbe {
            status: ProbeStatus::Survives,
            unsafe_kind: "",
            own_kind,
        },
        Some(unsafe_kind) => CandidateProbe {
            status: ProbeStatus::Unsafe,
            unsafe_kind,
            own_kind,
        },
    }
}

fn force_after_candidate(board: &mut Board, attacker: Stone) -> Option<&'static str> {
    let defender = attacker.opponent();
    if !immediate_winning_moves(board, attacker).is_empty() {
        return Some("opponent_wins_now");
    }
    if open_four_replies_count(board, attacker) > 0 {
        return Some("opponent_open_four_now");
    }
    for attack_mv in candidate_moves_sorted(board) {
        if !board.is_empty(attack_mv) {
            continue;
        }
        place_stone(board, attack_mv, attacker);
        if is_five_at(board, attack_mv, attacker) {
            remove_stone(board, attack_mv, attacker);
            return Some("opponent_a1_wins");
        }
        let a1_wins = immediate_winning_moves(board, attacker);
        if a1_wins.len() >= 2 {
            remove_stone(board, attack_mv, attacker);
            return Some("opponent_a1_double_immediate");
        }
        if a1_wins.len() == 1 {
            let block_mv = a1_wins[0];
            if board.is_empty(block_mv) {
                place_stone(board, block_mv, defender);
                let defender_wins_on_block = is_five_at(board, block_mv, defender);
                if !defender_wins_on_block {
                    if !immediate_winning_moves(board, attacker).is_empty() {
                        remove_stone(board, block_mv, defender);
                        remove_stone(board, attack_mv, attacker);
                        return Some("opponent_a2_wins_after_forced_block");
                    }
                    if open_four_replies_count(board, attacker) > 0 {
                        remove_stone(board, block_mv, defender);
                        remove_stone(board, attack_mv, attacker);
                        return Some("opponent_a2_open_four_after_forced_block");
                    }
                }
                remove_stone(board, block_mv, defender);
            }
        }
        remove_stone(board, attack_mv, attacker);
    }
    None
}

fn threat_counts(board: &Board, side: Stone) -> ThreatCounts {
    let opp = side.opponent();
    ThreatCounts {
        opp_immediate: immediate_winning_moves(board, opp).len(),
        opp_open_four: open_four_replies_count(board, opp),
        own_immediate: immediate_winning_moves(board, side).len(),
        own_open_four: open_four_replies_count(board, side),
    }
}

fn immediate_winning_moves(board: &Board, side: Stone) -> Vec<Move> {
    empty_moves(board)
        .into_iter()
        .filter(|&mv| threat_bin(board, mv, side) == 7)
        .collect()
}

fn open_four_replies_count(board: &Board, side: Stone) -> usize {
    let mut count = 0usize;
    for mv in empty_moves(board) {
        if !matches!(threat_bin(board, mv, side), 5 | 6) {
            continue;
        }
        let mut child = board.clone();
        place_stone(&mut child, mv, side);
        if immediate_winning_moves(&child, side).len() >= 2 {
            count += 1;
        }
    }
    count
}

fn forcing_share(board: &Board, side: Stone, moves: &[Move]) -> f32 {
    if moves.is_empty() {
        return 0.0;
    }
    let forcing = moves
        .iter()
        .copied()
        .filter(|&mv| board.is_empty(mv) && is_forcing_bin(threat_bin(board, mv, side)))
        .count();
    forcing as f32 / moves.len() as f32
}

fn max_attack_tier(board: &Board, side: Stone, moves: &[Move]) -> i32 {
    moves
        .iter()
        .copied()
        .filter(|&mv| board.is_empty(mv))
        .map(|mv| attack_tier(threat_bin(board, mv, side)))
        .max()
        .unwrap_or(0)
}

fn threat_bin(board: &Board, mv: Move, side: Stone) -> usize {
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

fn is_five_at(board: &Board, mv: Move, side: Stone) -> bool {
    let (row, col) = to_rc(mv);
    let row = row as i32;
    let col = col as i32;
    let (mine, opp) = match side {
        Stone::Black => (&board.black, &board.white),
        Stone::White => (&board.white, &board.black),
    };
    let rule_set = board.effective_rule_set();
    for &(dr, dc) in &DIR {
        let info = scan_line(mine, opp, row, col, dr, dc);
        let open_ends = info.open_front as u32 + info.open_back as u32;
        if rule_set.line_wins(side, info.count, open_ends) {
            return true;
        }
    }
    false
}

fn move_order_score(
    board: &Board,
    mv: Move,
    side: Stone,
    own_kind: usize,
    block_kind: usize,
) -> i32 {
    if own_kind == 7 {
        return TIER_WIN;
    }
    if block_kind == 7 {
        return TIER_BLOCK_WIN;
    }
    let attack_tier = attack_tier(own_kind);
    let block_tier = block_tier(block_kind);
    let mut score = attack_tier.max(block_tier);
    let (row, col) = to_rc(mv);
    let row = row as i32;
    let col = col as i32;
    let (my, opp) = match side {
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

fn neighbor_counts(board: &Board, mv: Move, side: Stone, radius: i32) -> (usize, usize) {
    let (row, col) = to_rc(mv);
    let row = row as i32;
    let col = col as i32;
    let opp = side.opponent();
    let mut my_count = 0usize;
    let mut opp_count = 0usize;
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
            if stone_at(board, idx) == Some(side) {
                my_count += 1;
            } else if stone_at(board, idx) == Some(opp) {
                opp_count += 1;
            }
        }
    }
    (my_count, opp_count)
}

fn attack_tier(bin: usize) -> i32 {
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

fn block_tier(bin: usize) -> i32 {
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

fn is_forcing_bin(bin: usize) -> bool {
    bin != 0
}

#[derive(Clone, Copy, Debug)]
struct RootCommitmentFeatures {
    candidate_attack: ThreatKind,
    candidate_block: ThreatKind,
    high_search: bool,
    wide_root: bool,
    candidate_nonforcing: bool,
    risk: bool,
}

#[derive(Clone, Copy, Default)]
struct CommitmentMoveFeatures {
    attack_bin: usize,
    block_bin: usize,
    attack_tier: i32,
    block_tier: i32,
    max_tier: i32,
    order_score: i32,
    after_own_force: usize,
    after_opp_force: usize,
    attack_forcing: bool,
    block_forcing: bool,
    forcing: bool,
    weak_attack: bool,
    blocks_opp_immediate: bool,
    blocks_opp_open_four: bool,
    reduces_opp_force: bool,
    creates_own_force: bool,
}

fn root_commitment_features(
    board: &Board,
    candidate: Move,
    candidate_count: usize,
    search_score: i32,
    search_score_min: i32,
    candidate_count_min: usize,
) -> RootCommitmentFeatures {
    let candidate_attack = classify_move_fast(board, candidate, board.side_to_move);
    let candidate_block = classify_move_fast(board, candidate, board.side_to_move.opponent());
    let candidate_forcing = candidate_attack.is_forcing() || candidate_block.is_forcing();
    let high_search = search_score >= search_score_min;
    let wide_root = candidate_count >= candidate_count_min;
    let candidate_nonforcing = !candidate_forcing;
    RootCommitmentFeatures {
        candidate_attack,
        candidate_block,
        high_search,
        wide_root,
        candidate_nonforcing,
        risk: high_search && wide_root && candidate_nonforcing,
    }
}

fn root_commitment_critic_features(
    board: &Board,
    candidate: Move,
    incumbent: Move,
    base_features: &BTreeMap<String, f32>,
) -> BTreeMap<String, f32> {
    let mut features = BTreeMap::new();
    for (key, value) in base_features {
        features.insert(format!("base.{key}"), *value);
    }
    insert_feature(
        &mut features,
        "base.root.move_count",
        Some(board.move_count as f32),
    );
    insert_feature(
        &mut features,
        "position.move_count",
        Some(board.move_count as f32),
    );
    insert_feature(
        &mut features,
        "position.side_black",
        Some(if board.side_to_move == Stone::Black {
            1.0
        } else {
            0.0
        }),
    );

    let search_score = base_features
        .get("candidate.search_score")
        .copied()
        .unwrap_or(-1.0e6);
    let search_delta = base_features.get("candidate.search_delta").copied();
    let candidate_count = base_features
        .get("root.candidate_count")
        .copied()
        .unwrap_or(0.0);
    let leader_score = base_features.get("root.leader_score").copied();
    let high_search = search_score >= 58.0;
    let wide_root = candidate_count >= 10.0;
    insert_feature(
        &mut features,
        "derived.high_search_58",
        Some(if high_search { 1.0 } else { 0.0 }),
    );
    insert_feature(
        &mut features,
        "derived.wide_root_10",
        Some(if wide_root { 1.0 } else { 0.0 }),
    );
    insert_feature(
        &mut features,
        "derived.high_search_x_wide_root",
        Some(if high_search && wide_root { 1.0 } else { 0.0 }),
    );
    insert_feature(
        &mut features,
        "derived.search_score_x_candidate_count",
        Some(search_score * candidate_count),
    );
    if let Some(search_delta) = search_delta {
        insert_feature(
            &mut features,
            "derived.search_score_x_search_delta",
            Some(search_score * search_delta),
        );
    }
    if let Some(leader_score) = leader_score {
        insert_feature(
            &mut features,
            "derived.leader_score_x_candidate_count",
            Some(leader_score * candidate_count),
        );
    }

    let candidate_features = commitment_move_features(board, candidate);
    let incumbent_features = commitment_move_features(board, incumbent);
    insert_commitment_move_features(&mut features, "commit.candidate", candidate_features);
    insert_commitment_move_features(&mut features, "commit.incumbent", incumbent_features);

    let candidate_forcing = candidate_features.forcing;
    let incumbent_forcing = incumbent_features.forcing;
    let candidate_nonforcing = !candidate_forcing;
    let candidate_blocks_force =
        candidate_features.blocks_opp_immediate || candidate_features.blocks_opp_open_four;
    let incumbent_blocks_force =
        incumbent_features.blocks_opp_immediate || incumbent_features.blocks_opp_open_four;
    let lost_forcing = incumbent_forcing && !candidate_forcing;
    let lost_block = incumbent_blocks_force && !candidate_blocks_force;
    let lost_tier = incumbent_features.max_tier > candidate_features.max_tier;
    let candidate_exposes_more_opp_force =
        candidate_features.after_opp_force > incumbent_features.after_opp_force;
    let candidate_reduces_less_opp_force =
        incumbent_features.after_opp_force < candidate_features.after_opp_force;
    let commitment_risk = high_search
        && wide_root
        && (candidate_nonforcing || lost_forcing || lost_block || candidate_exposes_more_opp_force);

    insert_feature(
        &mut features,
        "commit.candidate_count",
        Some(candidate_count),
    );
    insert_bool_feature(&mut features, "commit.high_search", high_search);
    insert_bool_feature(&mut features, "commit.wide_root", wide_root);
    insert_bool_feature(
        &mut features,
        "commit.high_search_wide",
        high_search && wide_root,
    );
    insert_bool_feature(
        &mut features,
        "commit.high_search_wide_nonforcing",
        high_search && wide_root && candidate_nonforcing,
    );
    insert_bool_feature(&mut features, "commit.commitment_risk", commitment_risk);
    insert_bool_feature(&mut features, "commit.incumbent_forcing", incumbent_forcing);
    insert_bool_feature(&mut features, "commit.candidate_forcing", candidate_forcing);
    insert_bool_feature(
        &mut features,
        "commit.candidate_nonforcing",
        candidate_nonforcing,
    );
    insert_bool_feature(
        &mut features,
        "commit.incumbent_blocks_force",
        incumbent_blocks_force,
    );
    insert_bool_feature(
        &mut features,
        "commit.candidate_blocks_force",
        candidate_blocks_force,
    );
    insert_bool_feature(&mut features, "commit.lost_forcing", lost_forcing);
    insert_bool_feature(&mut features, "commit.lost_block", lost_block);
    insert_bool_feature(&mut features, "commit.lost_tier", lost_tier);
    insert_bool_feature(
        &mut features,
        "commit.candidate_exposes_more_opp_force",
        candidate_exposes_more_opp_force,
    );
    insert_bool_feature(
        &mut features,
        "commit.candidate_reduces_less_opp_force",
        candidate_reduces_less_opp_force,
    );
    insert_bool_feature(
        &mut features,
        "commit.incumbent_weak_attack_candidate_nonforcing",
        incumbent_features.weak_attack && candidate_nonforcing,
    );
    insert_bool_feature(
        &mut features,
        "commit.candidate_weak_attack",
        candidate_features.weak_attack,
    );
    insert_i32_feature(
        &mut features,
        "commit.tier_gap",
        Some(incumbent_features.max_tier - candidate_features.max_tier),
    );
    insert_i32_feature(
        &mut features,
        "commit.attack_tier_gap",
        Some(incumbent_features.attack_tier - candidate_features.attack_tier),
    );
    insert_i32_feature(
        &mut features,
        "commit.block_tier_gap",
        Some(incumbent_features.block_tier - candidate_features.block_tier),
    );
    insert_i32_feature(
        &mut features,
        "commit.order_score_gap",
        Some(incumbent_features.order_score - candidate_features.order_score),
    );
    insert_i32_feature(
        &mut features,
        "commit.own_force_gap",
        Some(incumbent_features.after_own_force as i32 - candidate_features.after_own_force as i32),
    );
    insert_i32_feature(
        &mut features,
        "commit.opp_force_added",
        Some(candidate_features.after_opp_force as i32 - incumbent_features.after_opp_force as i32),
    );
    insert_i32_feature(
        &mut features,
        "commit.delta.max_tier",
        Some(candidate_features.max_tier - incumbent_features.max_tier),
    );
    insert_i32_feature(
        &mut features,
        "commit.delta.attack_tier",
        Some(candidate_features.attack_tier - incumbent_features.attack_tier),
    );
    insert_i32_feature(
        &mut features,
        "commit.delta.block_tier",
        Some(candidate_features.block_tier - incumbent_features.block_tier),
    );
    insert_i32_feature(
        &mut features,
        "commit.delta.order_score",
        Some(candidate_features.order_score - incumbent_features.order_score),
    );
    insert_i32_feature(
        &mut features,
        "commit.delta.after_own_force",
        Some(candidate_features.after_own_force as i32 - incumbent_features.after_own_force as i32),
    );
    insert_i32_feature(
        &mut features,
        "commit.delta.after_opp_force",
        Some(candidate_features.after_opp_force as i32 - incumbent_features.after_opp_force as i32),
    );

    features
}

fn commitment_move_features(board: &Board, mv: Move) -> CommitmentMoveFeatures {
    if !board.is_empty(mv) {
        return CommitmentMoveFeatures::default();
    }
    let side = board.side_to_move;
    let opp = side.opponent();
    let before = threat_counts(board, side);
    let attack_bin = threat_bin(board, mv, side);
    let block_bin = threat_bin(board, mv, opp);
    let attack_tier = attack_tier(attack_bin);
    let block_tier = block_tier(block_bin);
    let blocks_opp_immediate = threat_bin(board, mv, opp) == 7;
    let blocks_opp_open_four = is_open_four_reply_move(board, mv, opp);
    let mut child = board.clone();
    place_stone(&mut child, mv, side);
    let after = threat_counts(&child, side);
    let before_own_force = force_count(before.own_immediate, before.own_open_four);
    let before_opp_force = force_count(before.opp_immediate, before.opp_open_four);
    let after_own_force = force_count(after.own_immediate, after.own_open_four);
    let after_opp_force = force_count(after.opp_immediate, after.opp_open_four);
    let attack_forcing = is_forcing_bin(attack_bin);
    let block_forcing = is_forcing_bin(block_bin);
    CommitmentMoveFeatures {
        attack_bin,
        block_bin,
        attack_tier,
        block_tier,
        max_tier: attack_tier.max(block_tier),
        order_score: move_order_score(board, mv, side, attack_bin, block_bin),
        after_own_force,
        after_opp_force,
        attack_forcing,
        block_forcing,
        forcing: attack_forcing || block_forcing,
        weak_attack: matches!(attack_bin, 1 | 3) && block_bin == 0,
        blocks_opp_immediate,
        blocks_opp_open_four,
        reduces_opp_force: after_opp_force < before_opp_force,
        creates_own_force: after_own_force > before_own_force,
    }
}

fn is_open_four_reply_move(board: &Board, mv: Move, side: Stone) -> bool {
    if !board.is_empty(mv) || !matches!(threat_bin(board, mv, side), 5 | 6) {
        return false;
    }
    let mut child = board.clone();
    place_stone(&mut child, mv, side);
    immediate_winning_moves(&child, side).len() >= 2
}

fn force_count(immediate: usize, open_four: usize) -> usize {
    immediate + open_four
}

fn insert_commitment_move_features(
    features: &mut BTreeMap<String, f32>,
    prefix: &str,
    values: CommitmentMoveFeatures,
) {
    insert_i32_feature(
        features,
        &format!("{prefix}.attack_tier"),
        Some(values.attack_tier),
    );
    insert_i32_feature(
        features,
        &format!("{prefix}.block_tier"),
        Some(values.block_tier),
    );
    insert_i32_feature(
        features,
        &format!("{prefix}.max_tier"),
        Some(values.max_tier),
    );
    insert_i32_feature(
        features,
        &format!("{prefix}.order_score"),
        Some(values.order_score),
    );
    insert_i32_feature(
        features,
        &format!("{prefix}.after_own_force"),
        Some(values.after_own_force as i32),
    );
    insert_i32_feature(
        features,
        &format!("{prefix}.after_opp_force"),
        Some(values.after_opp_force as i32),
    );
    insert_bool_feature(
        features,
        &format!("{prefix}.attack_forcing"),
        values.attack_forcing,
    );
    insert_bool_feature(
        features,
        &format!("{prefix}.block_forcing"),
        values.block_forcing,
    );
    insert_bool_feature(features, &format!("{prefix}.forcing"), values.forcing);
    insert_bool_feature(
        features,
        &format!("{prefix}.weak_attack"),
        values.weak_attack,
    );
    insert_bool_feature(
        features,
        &format!("{prefix}.blocks_opp_immediate"),
        values.blocks_opp_immediate,
    );
    insert_bool_feature(
        features,
        &format!("{prefix}.blocks_opp_open_four"),
        values.blocks_opp_open_four,
    );
    insert_bool_feature(
        features,
        &format!("{prefix}.reduces_opp_force"),
        values.reduces_opp_force,
    );
    insert_bool_feature(
        features,
        &format!("{prefix}.creates_own_force"),
        values.creates_own_force,
    );
    insert_threat_one_hot(features, &format!("{prefix}.attack"), values.attack_bin);
    insert_threat_one_hot(features, &format!("{prefix}.block"), values.block_bin);
}

fn insert_threat_one_hot(features: &mut BTreeMap<String, f32>, prefix: &str, bin: usize) {
    insert_bool_feature(features, &format!("{prefix}.is_quiet"), bin == 0);
    insert_bool_feature(features, &format!("{prefix}.is_open_three"), bin == 1);
    insert_bool_feature(features, &format!("{prefix}.is_closed_four"), bin == 3);
    insert_bool_feature(
        features,
        &format!("{prefix}.is_open_four"),
        matches!(bin, 5 | 6),
    );
    insert_bool_feature(features, &format!("{prefix}.is_five"), bin == 7);
}

fn insert_bool_feature(features: &mut BTreeMap<String, f32>, name: &str, value: bool) {
    insert_feature(features, name, Some(if value { 1.0 } else { 0.0 }));
}

fn threat_kind_label(kind: ThreatKind) -> &'static str {
    match kind {
        ThreatKind::None => "None",
        ThreatKind::ClosedFour => "ClosedFour",
        ThreatKind::OpenThree => "OpenThree",
        ThreatKind::Five => "Five",
        ThreatKind::OpenFour => "OpenFour",
        ThreatKind::DoubleFour => "DoubleFour",
        ThreatKind::FourThree => "FourThree",
        ThreatKind::DoubleThree => "DoubleThree",
    }
}

fn push_kind_one_hot(out: &mut Vec<f32>, bin: usize) {
    for idx in 0..KIND_COUNT {
        out.push(if idx == bin { 1.0 } else { 0.0 });
    }
}

fn push_urgency_one_hot(out: &mut Vec<f32>, counts: ThreatCounts) {
    let label = if counts.opp_immediate > 0 {
        1
    } else if counts.opp_open_four > 0 {
        2
    } else if counts.own_immediate > 0 {
        3
    } else if counts.own_open_four > 0 {
        4
    } else if counts.opp_open_four > 0 && counts.own_open_four > 0 {
        5
    } else {
        0
    };
    for idx in 0..6 {
        out.push(if idx == label { 1.0 } else { 0.0 });
    }
}

fn capped_pos(value: f32, cap: f32) -> f32 {
    if cap <= 0.0 {
        0.0
    } else {
        value.min(cap) / cap
    }
}

fn capped_signed(value: f32, cap: f32) -> f32 {
    if cap <= 0.0 {
        0.0
    } else {
        value.min(cap).max(-cap) / cap
    }
}

fn pos_capped(value: f32, cap: f32) -> f32 {
    if cap <= 0.0 {
        0.0
    } else {
        value.min(cap).max(0.0) / cap
    }
}

fn candidate_moves_sorted(board: &Board) -> Vec<Move> {
    let mut moves = board.candidate_moves();
    moves.sort_unstable();
    moves
}

fn empty_moves(board: &Board) -> Vec<Move> {
    (0..NUM_CELLS).filter(|&idx| board.is_empty(idx)).collect()
}

fn place_stone(board: &mut Board, mv: Move, side: Stone) {
    debug_assert!(board.is_empty(mv));
    match side {
        Stone::Black => board.black.set(mv),
        Stone::White => board.white.set(mv),
    }
    board.move_count += 1;
}

fn remove_stone(board: &mut Board, mv: Move, side: Stone) {
    match side {
        Stone::Black => board.black.clear(mv),
        Stone::White => board.white.clear(mv),
    }
    board.move_count -= 1;
}

fn stone_at(board: &Board, mv: Move) -> Option<Stone> {
    if board.black.get(mv) {
        Some(Stone::Black)
    } else if board.white.get(mv) {
        Some(Stone::White)
    } else {
        None
    }
}

fn stone_code(side: Stone) -> i8 {
    match side {
        Stone::Black => 1,
        Stone::White => 2,
    }
}

fn cell_code(board: &Board, row: isize, col: isize) -> i8 {
    if row < 0 || col < 0 || row >= BOARD_SIZE as isize || col >= BOARD_SIZE as isize {
        return -1;
    }
    let idx = to_idx(row as usize, col as usize);
    if board.black.get(idx) {
        1
    } else if board.white.get(idx) {
        2
    } else {
        0
    }
}

fn eval_root(board: &Board, weights: &NnueWeights) -> f32 {
    evaluate(board, weights) as f32
}

fn rank_map<F>(values: &[CandidateValue], score: F) -> BTreeMap<Move, usize>
where
    F: Fn(&CandidateValue) -> f32,
{
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| {
        score(b)
            .partial_cmp(&score(a))
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.mv.cmp(&b.mv))
    });
    sorted
        .into_iter()
        .enumerate()
        .map(|(idx, value)| (value.mv, idx + 1))
        .collect()
}

impl RiskModel {
    fn load(path: &str) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("failed to read risk model: {e}"))?;
        let value: Value =
            serde_json::from_str(&text).map_err(|e| format!("failed to parse risk model: {e}"))?;
        let format = str_req(&value, "format")?;
        if format != RISK_FORMAT && format != COMMITMENT_CRITIC_FORMAT {
            return Err("unsupported risk model format".to_string());
        }
        let feature_names = value
            .get("feature_names")
            .and_then(Value::as_array)
            .ok_or("missing feature_names")?
            .iter()
            .map(|item| {
                item.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| "feature_names must be strings".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let standardizer = value.get("standardizer").ok_or("missing standardizer")?;
        let mean = f32_array(standardizer.get("mean"), "standardizer.mean")?;
        let std = f32_array(standardizer.get("std"), "standardizer.std")?;
        let model = value.get("model").ok_or("missing model")?;
        if str_req(model, "kind")? != "logistic" {
            return Err("risk model kind must be logistic".to_string());
        }
        let weights = f32_array(model.get("weights"), "model.weights")?;
        let bias = f32_req(model, "bias")?;
        let width = feature_names.len();
        if mean.len() != width || std.len() != width || weights.len() != width {
            return Err(format!(
                "risk model width mismatch: features={width} mean={} std={} weights={}",
                mean.len(),
                std.len(),
                weights.len()
            ));
        }
        if std.iter().any(|value| !value.is_finite() || *value == 0.0) {
            return Err("bad risk model standardizer std".to_string());
        }
        Ok(Self {
            feature_names,
            mean,
            std,
            weights,
            bias,
        })
    }

    fn score(&self, features: &BTreeMap<String, f32>) -> f32 {
        let mut z = self.bias;
        for idx in 0..self.feature_names.len() {
            let raw = features
                .get(&self.feature_names[idx])
                .copied()
                .unwrap_or(self.mean[idx]);
            let x = (raw - self.mean[idx]) / self.std[idx];
            z += x * self.weights[idx];
        }
        sigmoid(z)
    }
}

impl CandidateTrustModel {
    fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read candidate trust model: {e}"))?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| format!("failed to parse candidate trust model: {e}"))?;
        if str_req(&value, "format")? != CANDIDATE_TRUST_FORMAT {
            return Err("unsupported candidate trust model format".to_string());
        }
        let feature_names = value
            .get("feature_names")
            .and_then(Value::as_array)
            .ok_or("missing feature_names")?
            .iter()
            .map(|item| {
                item.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| "feature_names must be strings".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let standardizer = value.get("standardizer").ok_or("missing standardizer")?;
        let mean = f32_array(standardizer.get("mean"), "standardizer.mean")?;
        let std = f32_array(standardizer.get("std"), "standardizer.std")?;
        let model = value.get("model").ok_or("missing model")?;
        let weights = f32_array(model.get("weights"), "model.weights")?;
        let bias = f32_req(model, "bias")?;
        let width = feature_names.len();
        if mean.len() != width || std.len() != width || weights.len() != width {
            return Err(format!(
                "candidate trust model width mismatch: features={width} mean={} std={} weights={}",
                mean.len(),
                std.len(),
                weights.len()
            ));
        }
        if std.iter().any(|value| !value.is_finite() || *value == 0.0) {
            return Err("bad candidate trust model standardizer std".to_string());
        }
        Ok(Self {
            feature_names,
            mean,
            std,
            weights,
            bias,
        })
    }

    fn score(&self, features: &BTreeMap<String, f32>) -> f32 {
        let mut z = self.bias;
        for idx in 0..self.feature_names.len() {
            let raw = features
                .get(&self.feature_names[idx])
                .copied()
                .unwrap_or(0.0);
            let x = (raw - self.mean[idx]) / self.std[idx];
            z += x * self.weights[idx];
        }
        sigmoid(z)
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value.clamp(-40.0, 40.0)).exp())
}
impl Ensemble {
    fn load(path: &str) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("failed to read model: {e}"))?;
        let value: Value =
            serde_json::from_str(&text).map_err(|e| format!("failed to parse model: {e}"))?;
        if str_req(&value, "format")? != FORMAT {
            return Err("unsupported ensemble format".to_string());
        }
        let ensemble = value
            .get("ensemble")
            .and_then(Value::as_object)
            .ok_or("missing ensemble")?;
        let heads_obj = value
            .get("heads")
            .and_then(Value::as_object)
            .ok_or("missing heads")?;
        let raw_scores = match value
            .get("score_mode")
            .and_then(Value::as_str)
            .unwrap_or("zscore")
        {
            "raw" => true,
            "zscore" | "root_zscore" => false,
            other => return Err(format!("unsupported score_mode {other:?}")),
        };
        let mut heads = Vec::new();
        for (name, weight_value) in ensemble {
            let weight = f32_from_value(weight_value, "ensemble weight")?;
            let head_value = heads_obj
                .get(name)
                .ok_or_else(|| format!("missing head {name}"))?;
            heads.push(Head::parse(name, weight, head_value)?);
        }
        if heads.is_empty() {
            return Err("ensemble has no heads".to_string());
        }
        Ok(Self { heads, raw_scores })
    }

    fn needs_rank_delta(&self) -> bool {
        self.heads.iter().any(|head| head.needs_rank_delta)
    }

    fn needs_post_reply(&self) -> bool {
        self.heads.iter().any(|head| head.needs_post_reply)
    }

    fn needs_trajectory_child_static(&self) -> bool {
        self.heads
            .iter()
            .any(|head| head.needs_trajectory_child_static)
    }

    fn needs_trajectory_post_reply(&self) -> bool {
        self.heads
            .iter()
            .any(|head| head.needs_trajectory_post_reply)
    }

    fn needs_rollout_fast(&self) -> bool {
        self.heads.iter().any(|head| head.needs_rollout_fast)
    }

    fn score_root(&self, rows: Vec<(Move, Vec<f32>)>) -> Vec<ScoredMove> {
        if rows.is_empty() {
            return Vec::new();
        }
        let mut raw_by_head = vec![vec![0.0f32; rows.len()]; self.heads.len()];
        for (head_idx, head) in self.heads.iter().enumerate() {
            for (row_idx, (_, features)) in rows.iter().enumerate() {
                raw_by_head[head_idx][row_idx] = head.score(features);
            }
        }
        let score_by_head = if self.raw_scores {
            raw_by_head
        } else {
            raw_by_head
                .iter()
                .map(|scores| zscores(scores))
                .collect::<Vec<_>>()
        };
        rows.into_iter()
            .enumerate()
            .filter_map(|(row_idx, (mv, _))| {
                let mut ensemble_score = 0.0f32;
                for (head_idx, head) in self.heads.iter().enumerate() {
                    ensemble_score += head.weight * score_by_head[head_idx][row_idx];
                }
                if !ensemble_score.is_finite() {
                    return None;
                }
                let scaled = (ensemble_score * score_scale()).round();
                if !scaled.is_finite() {
                    return None;
                }
                Some(ScoredMove {
                    mv,
                    score: scaled.clamp(i32::MIN as f32 + 1.0, i32::MAX as f32 - 1.0) as i32,
                })
            })
            .collect()
    }
}

impl Head {
    fn parse(name: &str, weight: f32, value: &Value) -> Result<Self, String> {
        let feature_set = str_req(value, "feature_set")?;
        let (
            input_offset,
            input_dim,
            input_indices,
            needs_rank_delta,
            needs_post_reply,
            needs_trajectory_child_static,
            needs_trajectory_post_reply,
        ) = match feature_set {
            "expanded_value" => (0, EXPANDED_VALUE_COUNT, None, false, false, false, false),
            "candidate_local" => (0, FEATURE_COUNT, None, false, false, false, false),
            "value_only" | "child_value" => (
                VALUE_ONLY_OFFSET,
                VALUE_ONLY_COUNT,
                None,
                false,
                false,
                false,
                false,
            ),
            "rich_value" => (
                0,
                RICH_COUNT + VALUE_ONLY_COUNT,
                Some(rich_value_indices(false)),
                false,
                false,
                false,
                false,
            ),
            "rank_delta" => (
                0,
                RANK_DELTA_COUNT,
                Some((RANK_DELTA_OFFSET..RANK_DELTA_OFFSET + RANK_DELTA_COUNT).collect()),
                true,
                false,
                false,
                false,
            ),
            "rich_value_rank_delta" => (
                0,
                RICH_COUNT + VALUE_ONLY_COUNT + RANK_DELTA_COUNT,
                Some(rich_value_indices(true)),
                true,
                false,
                false,
                false,
            ),
            "post_reply" => (
                0,
                POST_REPLY_COUNT,
                Some(post_reply_indices(false)),
                false,
                true,
                false,
                false,
            ),
            "post_reply_rank_delta" => (
                0,
                POST_REPLY_COUNT + RANK_DELTA_COUNT,
                Some(post_reply_indices(true)),
                true,
                true,
                false,
                false,
            ),
            "rich_post_reply" => (
                0,
                RICH_COUNT + POST_REPLY_COUNT,
                Some(rich_post_reply_indices(false, false)),
                false,
                true,
                false,
                false,
            ),
            "rich_post_reply_rank_delta" => (
                0,
                RICH_COUNT + POST_REPLY_COUNT + RANK_DELTA_COUNT,
                Some(rich_post_reply_indices(false, true)),
                true,
                true,
                false,
                false,
            ),
            "rich_value_post" => (
                0,
                RICH_COUNT + VALUE_ONLY_COUNT + POST_REPLY_COUNT,
                Some(rich_post_reply_indices(true, false)),
                false,
                true,
                false,
                false,
            ),
            "rich_value_post_rank" => (
                0,
                RICH_COUNT + VALUE_ONLY_COUNT + POST_REPLY_COUNT + RANK_DELTA_COUNT,
                Some(rich_post_reply_indices(true, true)),
                true,
                true,
                false,
                false,
            ),
            "trajectory_child_static" => (
                TRAJECTORY_CHILD_STATIC_OFFSET,
                TRAJECTORY_CHILD_STATIC_COUNT,
                None,
                false,
                false,
                true,
                false,
            ),
            "trajectory_post_reply" => (
                TRAJECTORY_POST_REPLY_OFFSET,
                TRAJECTORY_POST_REPLY_COUNT,
                None,
                false,
                false,
                false,
                true,
            ),
            "rollout_fast" => (
                ROLLOUT_FAST_OFFSET,
                ROLLOUT_FAST_COUNT,
                None,
                false,
                false,
                false,
                false,
            ),
            other => return Err(format!("unsupported feature_set {other:?} for {name}")),
        };
        let needs_rollout_fast = feature_set == "rollout_fast";
        let feature_names = value
            .get("feature_names")
            .and_then(Value::as_array)
            .ok_or("missing feature_names")?;
        if feature_names.len() != input_dim {
            return Err(format!(
                "feature_names width mismatch for {name}: got={} expected={input_dim}",
                feature_names.len()
            ));
        }
        let standardizer = value.get("standardizer").ok_or("missing standardizer")?;
        let mean = f32_array(standardizer.get("mean"), "standardizer.mean")?;
        let std = f32_array(standardizer.get("std"), "standardizer.std")?;
        if mean.len() != input_dim || std.len() != input_dim {
            return Err(format!(
                "standardizer width mismatch for {name}: mean={} std={} expected={input_dim}",
                mean.len(),
                std.len()
            ));
        }
        if std.iter().any(|v| !v.is_finite() || *v == 0.0) {
            return Err(format!("bad standardizer std for {name}"));
        }
        let model = Model::parse(value.get("model").ok_or("missing model")?, input_dim)?;
        Ok(Self {
            weight,
            input_dim,
            input_offset,
            input_indices,
            needs_rank_delta,
            needs_post_reply,
            needs_trajectory_child_static,
            needs_trajectory_post_reply,
            needs_rollout_fast,
            mean,
            std,
            model,
        })
    }

    fn score(&self, features: &[f32]) -> f32 {
        let raw = if let Some(indices) = &self.input_indices {
            let mut values = Vec::with_capacity(indices.len());
            for &idx in indices {
                let Some(value) = features.get(idx) else {
                    return 0.0;
                };
                values.push(*value);
            }
            values
        } else {
            if features.len() < self.input_offset + self.input_dim {
                return 0.0;
            }
            features
                .iter()
                .skip(self.input_offset)
                .take(self.input_dim)
                .copied()
                .collect::<Vec<_>>()
        };
        let x = raw
            .iter()
            .zip(self.mean.iter().zip(&self.std))
            .map(|(value, (mean, std))| (*value - *mean) / *std)
            .collect::<Vec<_>>();
        self.model.score(&x)
    }
}

fn rich_value_indices(include_rank_delta: bool) -> Vec<usize> {
    let mut indices = Vec::with_capacity(
        RICH_COUNT
            + VALUE_ONLY_COUNT
            + if include_rank_delta {
                RANK_DELTA_COUNT
            } else {
                0
            },
    );
    indices.extend(0..RICH_COUNT);
    indices.extend(VALUE_ONLY_OFFSET..VALUE_ONLY_OFFSET + VALUE_ONLY_COUNT);
    if include_rank_delta {
        indices.extend(RANK_DELTA_OFFSET..RANK_DELTA_OFFSET + RANK_DELTA_COUNT);
    }
    indices
}

fn post_reply_indices(include_rank_delta: bool) -> Vec<usize> {
    let mut indices = Vec::with_capacity(
        POST_REPLY_COUNT
            + if include_rank_delta {
                RANK_DELTA_COUNT
            } else {
                0
            },
    );
    indices.extend(POST_REPLY_OFFSET..POST_REPLY_OFFSET + POST_REPLY_COUNT);
    if include_rank_delta {
        indices.extend(RANK_DELTA_OFFSET..RANK_DELTA_OFFSET + RANK_DELTA_COUNT);
    }
    indices
}

fn rich_post_reply_indices(include_value: bool, include_rank_delta: bool) -> Vec<usize> {
    let mut indices = Vec::with_capacity(
        RICH_COUNT
            + if include_value { VALUE_ONLY_COUNT } else { 0 }
            + POST_REPLY_COUNT
            + if include_rank_delta {
                RANK_DELTA_COUNT
            } else {
                0
            },
    );
    indices.extend(0..RICH_COUNT);
    if include_value {
        indices.extend(VALUE_ONLY_OFFSET..VALUE_ONLY_OFFSET + VALUE_ONLY_COUNT);
    }
    indices.extend(POST_REPLY_OFFSET..POST_REPLY_OFFSET + POST_REPLY_COUNT);
    if include_rank_delta {
        indices.extend(RANK_DELTA_OFFSET..RANK_DELTA_OFFSET + RANK_DELTA_COUNT);
    }
    indices
}

impl Model {
    fn parse(value: &Value, input_dim: usize) -> Result<Self, String> {
        match str_req(value, "kind")? {
            "linear" => {
                let w = f32_array(value.get("w"), "model.w")?;
                let bias = f32_req(value, "bias")?;
                if w.len() != input_dim {
                    return Err("Linear width mismatch".to_string());
                }
                Ok(Self::Linear { w, bias })
            }
            "mlp" => {
                let w1 = f32_matrix(value.get("w1"), "model.w1")?;
                let b1 = f32_array(value.get("b1"), "model.b1")?;
                let w2 = f32_array(value.get("w2"), "model.w2")?;
                let b2 = f32_req(value, "b2")?;
                if w1.len() != input_dim || w1.iter().any(|row| row.len() != b1.len()) {
                    return Err("MLP w1 shape mismatch".to_string());
                }
                if w2.len() != b1.len() {
                    return Err("MLP w2 shape mismatch".to_string());
                }
                Ok(Self::Mlp { w1, b1, w2, b2 })
            }
            "fm" => {
                let w = f32_array(value.get("w"), "model.w")?;
                let v = f32_matrix(value.get("v"), "model.v")?;
                let bias = f32_req(value, "bias")?;
                if w.len() != input_dim || v.len() != input_dim {
                    return Err("FM width mismatch".to_string());
                }
                let rank = v.first().map(|row| row.len()).unwrap_or(0);
                if rank == 0 || v.iter().any(|row| row.len() != rank) {
                    return Err("FM rank mismatch".to_string());
                }
                Ok(Self::Fm { w, v, bias })
            }
            other => Err(format!("unsupported model kind {other:?}")),
        }
    }

    fn score(&self, x: &[f32]) -> f32 {
        match self {
            Self::Linear { w, bias } => {
                let mut out = *bias;
                for (value, weight) in x.iter().zip(w) {
                    out += value * weight;
                }
                out
            }
            Self::Mlp { w1, b1, w2, b2 } => {
                let mut out = *b2;
                for hidden_idx in 0..b1.len() {
                    let mut z = b1[hidden_idx];
                    for (feature_idx, value) in x.iter().enumerate() {
                        z += *value * w1[feature_idx][hidden_idx];
                    }
                    out += z.tanh() * w2[hidden_idx];
                }
                out
            }
            Self::Fm { w, v, bias } => {
                let mut out = *bias;
                for (value, weight) in x.iter().zip(w) {
                    out += value * weight;
                }
                let rank = v[0].len();
                for r in 0..rank {
                    let mut sum = 0.0f32;
                    let mut square_sum = 0.0f32;
                    for (idx, value) in x.iter().enumerate() {
                        let vx = value * v[idx][r];
                        sum += vx;
                        square_sum += vx * vx;
                    }
                    out += 0.5 * (sum * sum - square_sum);
                }
                if out.is_finite() {
                    out
                } else if out.is_sign_positive() {
                    1.0e6
                } else {
                    -1.0e6
                }
            }
        }
    }
}

fn ensemble() -> Option<&'static Ensemble> {
    static ENSEMBLE: OnceLock<Option<Ensemble>> = OnceLock::new();
    ENSEMBLE.get_or_init(load_ensemble).as_ref()
}

fn veto_ensemble() -> Option<&'static Ensemble> {
    static ENSEMBLE: OnceLock<Option<Ensemble>> = OnceLock::new();
    ENSEMBLE.get_or_init(load_veto_ensemble).as_ref()
}

fn secondary_veto_ensemble() -> Option<&'static Ensemble> {
    static ENSEMBLE: OnceLock<Option<Ensemble>> = OnceLock::new();
    ENSEMBLE.get_or_init(load_secondary_veto_ensemble).as_ref()
}

fn root_risk_model() -> Option<&'static RiskModel> {
    static MODEL: OnceLock<Option<RiskModel>> = OnceLock::new();
    MODEL.get_or_init(load_root_risk_model).as_ref()
}

fn root_commitment_critic_model() -> Option<&'static RiskModel> {
    static MODEL: OnceLock<Option<RiskModel>> = OnceLock::new();
    MODEL
        .get_or_init(load_root_commitment_critic_model)
        .as_ref()
}

fn root_candidate_trust_model() -> Option<&'static CandidateTrustModel> {
    static MODEL: OnceLock<Option<CandidateTrustModel>> = OnceLock::new();
    MODEL.get_or_init(load_root_candidate_trust_model).as_ref()
}

fn load_root_risk_model() -> Option<RiskModel> {
    let Ok(path) = std::env::var("NORU_CANDIDATE_LOCAL_ROOT_RISK_MODEL") else {
        return None;
    };
    let trimmed = path.trim();
    if is_disabled_value(trimmed) {
        return None;
    }
    Some(
        RiskModel::load(trimmed).unwrap_or_else(|e| {
            panic!("invalid NORU_CANDIDATE_LOCAL_ROOT_RISK_MODEL={trimmed}: {e}")
        }),
    )
}

fn load_root_commitment_critic_model() -> Option<RiskModel> {
    let Ok(path) = std::env::var("NORU_CANDIDATE_LOCAL_ROOT_COMMITMENT_CRITIC_MODEL") else {
        return None;
    };
    let trimmed = path.trim();
    if is_disabled_value(trimmed) {
        return None;
    }
    Some(RiskModel::load(trimmed).unwrap_or_else(|e| {
        panic!("invalid NORU_CANDIDATE_LOCAL_ROOT_COMMITMENT_CRITIC_MODEL={trimmed}: {e}")
    }))
}

fn load_root_candidate_trust_model() -> Option<CandidateTrustModel> {
    let Ok(path) = std::env::var("NORU_CANDIDATE_LOCAL_ROOT_TRUST_MODEL") else {
        return None;
    };
    let trimmed = path.trim();
    if is_disabled_value(trimmed) {
        return None;
    }
    Some(
        CandidateTrustModel::load(trimmed).unwrap_or_else(|e| {
            panic!("invalid NORU_CANDIDATE_LOCAL_ROOT_TRUST_MODEL={trimmed}: {e}")
        }),
    )
}

fn load_ensemble() -> Option<Ensemble> {
    let Ok(path) = std::env::var("NORU_CANDIDATE_LOCAL_ENSEMBLE") else {
        return None;
    };
    let trimmed = path.trim();
    if is_disabled_value(trimmed) {
        return None;
    }
    Some(
        Ensemble::load(trimmed)
            .unwrap_or_else(|e| panic!("invalid NORU_CANDIDATE_LOCAL_ENSEMBLE={trimmed}: {e}")),
    )
}

fn load_veto_ensemble() -> Option<Ensemble> {
    let Ok(path) = std::env::var("NORU_CANDIDATE_LOCAL_ROOT_VETO_MODEL") else {
        return None;
    };
    let trimmed = path.trim();
    if is_disabled_value(trimmed) {
        return None;
    }
    Some(
        Ensemble::load(trimmed).unwrap_or_else(|e| {
            panic!("invalid NORU_CANDIDATE_LOCAL_ROOT_VETO_MODEL={trimmed}: {e}")
        }),
    )
}

fn load_secondary_veto_ensemble() -> Option<Ensemble> {
    let Ok(path) = std::env::var("NORU_CANDIDATE_LOCAL_ROOT_SECONDARY_VETO_MODEL") else {
        return None;
    };
    let trimmed = path.trim();
    if is_disabled_value(trimmed) {
        return None;
    }
    Some(Ensemble::load(trimmed).unwrap_or_else(|e| {
        panic!("invalid NORU_CANDIDATE_LOCAL_ROOT_SECONDARY_VETO_MODEL={trimmed}: {e}")
    }))
}

fn zscores(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let var = values
        .iter()
        .map(|value| {
            let d = *value - mean;
            d * d
        })
        .sum::<f32>()
        / values.len() as f32;
    let std = var.sqrt().max(1.0e-6);
    values.iter().map(|value| (*value - mean) / std).collect()
}

fn root_tiebreak_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_env_bool_default("NORU_CANDIDATE_LOCAL_ROOT_TIEBREAK", true))
}

fn min_ply() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| parse_env_usize("NORU_CANDIDATE_LOCAL_MIN_PLY").unwrap_or(0))
}

fn top_replies() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_usize("NORU_CANDIDATE_LOCAL_TOP_REPLIES").unwrap_or(DEFAULT_TOP_REPLIES)
    })
}

fn trajectory_post_reply_top_replies() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_usize("NORU_CANDIDATE_LOCAL_TRAJECTORY_POST_REPLY_TOP_REPLIES")
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_TRAJECTORY_POST_REPLY_TOP_REPLIES)
    })
}

fn root_margin() -> i32 {
    static VALUE: OnceLock<i32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_MARGIN")
            .filter(|v| *v >= 0)
            .unwrap_or(0)
    })
}

fn root_score_margin() -> i32 {
    static VALUE: OnceLock<i32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_SCORE_MARGIN")
            .filter(|v| *v >= 0)
            .unwrap_or(0)
    })
}

fn score_scale() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_f32("NORU_CANDIDATE_LOCAL_SCORE_SCALE")
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(DEFAULT_SCORE_SCALE)
    })
}

fn root_gate_mode() -> crate::candidate_ranker::RootGateMode {
    static VALUE: OnceLock<crate::candidate_ranker::RootGateMode> = OnceLock::new();
    *VALUE.get_or_init(|| parse_gate_mode_env("NORU_CANDIDATE_LOCAL_ROOT_GATE"))
}

pub(crate) fn root_order_tiebreak_enabled_for(board: &Board) -> bool {
    root_order_tiebreak_enabled() && board.move_count >= min_ply() && ensemble().is_some()
}

fn root_order_tiebreak_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| parse_env_bool_default("NORU_CANDIDATE_LOCAL_ROOT_ORDER_TIEBREAK", false))
}

pub(crate) fn root_order_tie_margin() -> u64 {
    static VALUE: OnceLock<u64> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_usize("NORU_CANDIDATE_LOCAL_ROOT_ORDER_TIE_MARGIN")
            .unwrap_or(0)
            .try_into()
            .unwrap_or(u64::MAX)
    })
}

pub(crate) fn root_order_topk() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| parse_env_usize("NORU_CANDIDATE_LOCAL_ROOT_ORDER_TOPK").unwrap_or(0))
}

pub(crate) fn root_order_gate_mode() -> crate::candidate_ranker::RootGateMode {
    static VALUE: OnceLock<crate::candidate_ranker::RootGateMode> = OnceLock::new();
    *VALUE.get_or_init(|| parse_gate_mode_env("NORU_CANDIDATE_LOCAL_ROOT_ORDER_GATE"))
}

fn root_order_guard_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_env_bool_default("NORU_CANDIDATE_LOCAL_ROOT_ORDER_GUARD", false))
}

fn root_order_guard_max_rank_delta() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_usize("NORU_CANDIDATE_LOCAL_ROOT_ORDER_MAX_RANK_DELTA").unwrap_or(0)
    })
}

fn root_order_guard_max_rank() -> Option<usize> {
    static VALUE: OnceLock<Option<usize>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_usize("NORU_CANDIDATE_LOCAL_ROOT_ORDER_MAX_RANK").filter(|value| *value > 0)
    })
}

fn root_order_guard_score_margin() -> Option<i32> {
    static VALUE: OnceLock<Option<i32>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_ORDER_SCORE_MARGIN").filter(|value| *value >= 0)
    })
}

fn root_rescue_min_order_rank() -> Option<usize> {
    static VALUE: OnceLock<Option<usize>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_usize("NORU_CANDIDATE_LOCAL_ROOT_RESCUE_MIN_ORDER_RANK")
            .filter(|value| *value > 0)
    })
}

fn root_veto_margin() -> i32 {
    static VALUE: OnceLock<i32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_VETO_MARGIN")
            .filter(|value| *value >= 0)
            .unwrap_or(0)
    })
}

fn root_veto_confidence() -> Option<i32> {
    static VALUE: OnceLock<Option<i32>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_VETO_CONFIDENCE").filter(|value| *value >= 0)
    })
}

fn root_secondary_veto_margin() -> i32 {
    static VALUE: OnceLock<i32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_SECONDARY_VETO_MARGIN")
            .filter(|value| *value >= 0)
            .unwrap_or(0)
    })
}

fn root_secondary_veto_confidence() -> Option<i32> {
    static VALUE: OnceLock<Option<i32>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_SECONDARY_VETO_CONFIDENCE")
            .filter(|value| *value >= 0)
    })
}

fn root_reply_center_guard_max_delta() -> Option<i32> {
    static VALUE: OnceLock<Option<i32>> = OnceLock::new();
    *VALUE.get_or_init(|| parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_REPLY_CENTER_MAX_DELTA"))
}

fn root_post_reply_guard_max_ply() -> Option<usize> {
    static VALUE: OnceLock<Option<usize>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_usize("NORU_CANDIDATE_LOCAL_ROOT_POST_REPLY_MAX_PLY").filter(|value| *value > 0)
    })
}

fn root_post_reply_guard_max_delta() -> Option<i32> {
    static VALUE: OnceLock<Option<i32>> = OnceLock::new();
    *VALUE.get_or_init(|| parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_POST_REPLY_MAX_DELTA"))
}

fn root_rollout_fast_audit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        parse_env_bool_default("NORU_CANDIDATE_LOCAL_ROOT_ROLLOUT_FAST_AUDIT", false)
    })
}

fn root_rollout_fast_plies() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_usize("NORU_CANDIDATE_LOCAL_ROOT_ROLLOUT_FAST_PLIES")
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_ROLLOUT_FAST_PLIES)
    })
}

fn root_rollout_fast_gamma() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_f32("NORU_CANDIDATE_LOCAL_ROOT_ROLLOUT_FAST_GAMMA")
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(DEFAULT_ROLLOUT_FAST_GAMMA)
    })
}

fn root_rollout_fast_rule_mode() -> RolloutFastRuleMode {
    static VALUE: OnceLock<RolloutFastRuleMode> = OnceLock::new();
    *VALUE.get_or_init(|| {
        let Ok(raw) = std::env::var("NORU_CANDIDATE_LOCAL_ROOT_ROLLOUT_FAST_RULE") else {
            return RolloutFastRuleMode::Off;
        };
        let trimmed = raw.trim();
        if is_disabled_value(trimmed) || trimmed.eq_ignore_ascii_case("none") {
            return RolloutFastRuleMode::Off;
        }
        match trimmed.to_ascii_lowercase().as_str() {
            "own_threat_opp_delay" | "own-threat-opp-delay" => {
                RolloutFastRuleMode::OwnThreatOppDelay
            }
            other => panic!(
                "invalid NORU_CANDIDATE_LOCAL_ROOT_ROLLOUT_FAST_RULE={other:?}; expected off or own_threat_opp_delay"
            ),
        }
    })
}

fn rollout_fast_rule_label(mode: RolloutFastRuleMode) -> &'static str {
    match mode {
        RolloutFastRuleMode::Off => "off",
        RolloutFastRuleMode::OwnThreatOppDelay => "own_threat_opp_delay",
    }
}

fn root_risk_threshold() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_f32("NORU_CANDIDATE_LOCAL_ROOT_RISK_THRESHOLD")
            .filter(|value| value.is_finite())
            .unwrap_or(0.80)
    })
}

fn root_commitment_critic_threshold() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_f32("NORU_CANDIDATE_LOCAL_ROOT_COMMITMENT_CRITIC_THRESHOLD")
            .filter(|value| value.is_finite())
            .unwrap_or(0.80)
    })
}

fn root_candidate_trust_threshold() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_f32("NORU_CANDIDATE_LOCAL_ROOT_TRUST_THRESHOLD")
            .filter(|value| value.is_finite())
            .unwrap_or(0.85)
    })
}

fn root_candidate_trust_mode() -> CandidateTrustMode {
    static VALUE: OnceLock<CandidateTrustMode> = OnceLock::new();
    *VALUE.get_or_init(|| {
        let Ok(raw) = std::env::var("NORU_CANDIDATE_LOCAL_ROOT_TRUST_MODE") else {
            return CandidateTrustMode::Off;
        };
        let trimmed = raw.trim();
        if is_disabled_value(trimmed) || trimmed.eq_ignore_ascii_case("none") {
            return CandidateTrustMode::Off;
        }
        match trimmed.to_ascii_lowercase().as_str() {
            "codebook_score_primary" | "codebook-score-primary" | "score_primary"
            | "score-primary" => CandidateTrustMode::CodebookScorePrimary,
            "score_primary_rq423_blocked" | "score-primary-rq423-blocked"
            | "score_primary_blocked" | "score-primary-blocked" => {
                CandidateTrustMode::ScorePrimaryRq423Blocked
            }
            other => panic!(
                "invalid NORU_CANDIDATE_LOCAL_ROOT_TRUST_MODE={other:?}; expected off, codebook_score_primary, or score_primary_rq423_blocked"
            ),
        }
    })
}

fn candidate_trust_mode_allows(
    mode: CandidateTrustMode,
    codebook_prefers: bool,
    score_prefers: bool,
    primary_veto_allows: bool,
    rq423_root_accept_allows: bool,
) -> bool {
    match mode {
        CandidateTrustMode::Off => false,
        CandidateTrustMode::CodebookScorePrimary => {
            codebook_prefers && score_prefers && primary_veto_allows
        }
        CandidateTrustMode::ScorePrimaryRq423Blocked => {
            score_prefers && primary_veto_allows && !rq423_root_accept_allows
        }
    }
}

fn candidate_trust_mode_label(mode: CandidateTrustMode) -> &'static str {
    match mode {
        CandidateTrustMode::Off => "off",
        CandidateTrustMode::CodebookScorePrimary => "codebook_score_primary",
        CandidateTrustMode::ScorePrimaryRq423Blocked => "score_primary_rq423_blocked",
    }
}

fn root_search_feature_guard_mode() -> SearchFeatureGuardMode {
    static VALUE: OnceLock<SearchFeatureGuardMode> = OnceLock::new();
    *VALUE.get_or_init(|| {
        let raw = std::env::var("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_GUARD_MODE")
            .or_else(|_| std::env::var("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_GUARD"));
        let Ok(raw) = raw else {
            return SearchFeatureGuardMode::Off;
        };
        let trimmed = raw.trim();
        if is_disabled_value(trimmed) || trimmed.eq_ignore_ascii_case("none") {
            return SearchFeatureGuardMode::Off;
        }
        match trimmed.to_ascii_lowercase().as_str() {
            "child_eval_delta" | "child-eval-delta" | "child_delta" | "child-delta" => {
                SearchFeatureGuardMode::ChildEvalDelta
            }
            "child_rank_reply_drop" | "child-rank-reply-drop" | "rank_reply"
            | "rank-reply" => SearchFeatureGuardMode::ChildRankReplyDrop,
            "child_eval_best_gap" | "child-eval-best-gap" | "child_best_gap"
            | "child-best-gap" => SearchFeatureGuardMode::ChildEvalBestGap,
            "reply_best_gap_search_delta" | "reply-best-gap-search-delta"
            | "reply_gap_search_delta" | "reply-gap-search-delta" => {
                SearchFeatureGuardMode::ReplyBestGapSearchDelta
            }
            "search_delta_reply_rank" | "search-delta-reply-rank"
            | "reply_rank_search_delta" | "reply-rank-search-delta" => {
                SearchFeatureGuardMode::SearchDeltaReplyRank
            }
            other => panic!(
                "invalid NORU_CANDIDATE_LOCAL_ROOT_SEARCH_GUARD_MODE={other:?}; expected off, child_eval_delta, child_rank_reply_drop, child_eval_best_gap, reply_best_gap_search_delta, or search_delta_reply_rank"
            ),
        }
    })
}

fn root_search_feature_guard_child_eval_delta() -> i32 {
    static VALUE: OnceLock<i32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_GUARD_CHILD_DELTA").unwrap_or(-25)
    })
}

fn root_search_feature_guard_min_child_best_gap() -> i32 {
    static VALUE: OnceLock<i32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_GUARD_CHILD_BEST_GAP").unwrap_or(15)
    })
}

fn root_search_feature_guard_max_child_rank() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_f32("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_GUARD_MAX_CHILD_RANK")
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(0.05)
    })
}

fn root_search_feature_guard_max_reply_drop() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_f32("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_GUARD_MAX_REPLY_DROP")
            .filter(|value| value.is_finite())
            .unwrap_or(0.0)
    })
}

fn root_search_feature_guard_min_reply_best_gap() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_f32("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_GUARD_REPLY_BEST_GAP")
            .filter(|value| value.is_finite())
            .unwrap_or(10.0)
    })
}

fn root_search_feature_guard_min_search_delta() -> i32 {
    static VALUE: OnceLock<i32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_GUARD_SEARCH_DELTA").unwrap_or(14)
    })
}

fn root_search_feature_guard_reply_rank_a_search_delta() -> i32 {
    static VALUE: OnceLock<i32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_GUARD_REPLY_RANK_A_SEARCH_DELTA")
            .unwrap_or(10)
    })
}

fn root_search_feature_guard_reply_rank_a() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_f32("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_GUARD_REPLY_RANK_A")
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(0.35)
    })
}

fn root_search_feature_guard_reply_rank_b_search_delta() -> i32 {
    static VALUE: OnceLock<i32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_GUARD_REPLY_RANK_B_SEARCH_DELTA")
            .unwrap_or(12)
    })
}

fn root_search_feature_guard_reply_rank_b() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_f32("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_GUARD_REPLY_RANK_B")
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(0.25)
    })
}

fn root_search_score_max() -> Option<i32> {
    static VALUE: OnceLock<Option<i32>> = OnceLock::new();
    *VALUE.get_or_init(|| parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_SEARCH_SCORE_MAX"))
}

fn root_commitment_guard() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NORU_CANDIDATE_LOCAL_ROOT_COMMITMENT_GUARD")
            .map(|raw| !is_disabled_value(raw.trim()))
            .unwrap_or(false)
    })
}

fn root_commitment_search_score_min() -> i32 {
    static VALUE: OnceLock<i32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_COMMITMENT_SEARCH_SCORE_MIN").unwrap_or(58)
    })
}

fn root_commitment_candidate_count_min() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_usize("NORU_CANDIDATE_LOCAL_ROOT_COMMITMENT_CANDIDATE_COUNT_MIN").unwrap_or(10)
    })
}

fn search_feature_guard_label(mode: SearchFeatureGuardMode) -> &'static str {
    match mode {
        SearchFeatureGuardMode::Off => "off",
        SearchFeatureGuardMode::ChildEvalDelta => "child_eval_delta",
        SearchFeatureGuardMode::ChildRankReplyDrop => "child_rank_reply_drop",
        SearchFeatureGuardMode::ChildEvalBestGap => "child_eval_best_gap",
        SearchFeatureGuardMode::ReplyBestGapSearchDelta => "reply_best_gap_search_delta",
        SearchFeatureGuardMode::SearchDeltaReplyRank => "search_delta_reply_rank",
    }
}

fn rq423_root_accept_pair_mode() -> Rq423RootAcceptPairMode {
    static VALUE: OnceLock<Rq423RootAcceptPairMode> = OnceLock::new();
    *VALUE.get_or_init(|| {
        let Ok(raw) = std::env::var("NORU_RQ423_ROOT_ACCEPT_PAIR_MODE") else {
            return Rq423RootAcceptPairMode::SearchIncumbent;
        };
        let trimmed = raw.trim();
        if is_disabled_value(trimmed)
            || trimmed.eq_ignore_ascii_case("default")
            || trimmed.eq_ignore_ascii_case("search_incumbent")
            || trimmed.eq_ignore_ascii_case("search-incumbent")
            || trimmed.eq_ignore_ascii_case("initial")
        {
            return Rq423RootAcceptPairMode::SearchIncumbent;
        }
        match trimmed.to_ascii_lowercase().as_str() {
            "current_best" | "current-best" | "current" | "dynamic" | "dynamic_incumbent"
            | "dynamic-incumbent" => Rq423RootAcceptPairMode::CurrentBest,
            other => panic!(
                "invalid NORU_RQ423_ROOT_ACCEPT_PAIR_MODE={other:?}; expected search_incumbent or current_best"
            ),
        }
    })
}

fn rq423_root_accept_pair_mode_label(mode: Rq423RootAcceptPairMode) -> &'static str {
    match mode {
        Rq423RootAcceptPairMode::SearchIncumbent => "search_incumbent",
        Rq423RootAcceptPairMode::CurrentBest => "current_best",
    }
}

fn root_rollout_fast_rule_min_own_kind() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_usize("NORU_CANDIDATE_LOCAL_ROOT_ROLLOUT_FAST_RULE_MIN_OWN_KIND")
            .unwrap_or(1)
            .min(KIND_COUNT - 1)
    })
}

fn root_pressure_guard_min_incumbent_child() -> Option<i32> {
    static VALUE: OnceLock<Option<i32>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_PRESSURE_GUARD_MIN_INCUMBENT_CHILD")
    })
}

fn root_pressure_guard_min_child_delta() -> Option<i32> {
    static VALUE: OnceLock<Option<i32>> = OnceLock::new();
    *VALUE.get_or_init(|| parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_PRESSURE_GUARD_MIN_CHILD_DELTA"))
}

fn root_pressure_guard_min_candidate_score() -> Option<i32> {
    static VALUE: OnceLock<Option<i32>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        parse_env_i32("NORU_CANDIDATE_LOCAL_ROOT_PRESSURE_GUARD_MIN_CANDIDATE_SCORE")
    })
}

fn parse_gate_mode_env(name: &str) -> crate::candidate_ranker::RootGateMode {
    let Ok(raw) = std::env::var(name) else {
        return crate::candidate_ranker::RootGateMode::None;
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

fn is_win_score(score: i32) -> bool {
    score.abs() >= WIN_SCORE - 1_000
}

fn str_req<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing {key}"))
}

fn f32_req(value: &Value, key: &str) -> Result<f32, String> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .ok_or_else(|| format!("missing {key}"))
}

fn f32_from_value(value: &Value, name: &str) -> Result<f32, String> {
    value
        .as_f64()
        .map(|v| v as f32)
        .ok_or_else(|| format!("{name} is not a number"))
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
    arr.iter().map(|row| f32_array(Some(row), name)).collect()
}

fn parse_env_i32(name: &str) -> Option<i32> {
    std::env::var(name).ok()?.trim().parse::<i32>().ok()
}

fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse::<usize>().ok()
}

fn parse_env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok()?.trim().parse::<f32>().ok()
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

fn is_disabled_value(value: &str) -> bool {
    value.is_empty()
        || value.eq_ignore_ascii_case("0")
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("no")
}
