//! RQ423 richer relation-UE root accept guard.
//!
//! This is an env-gated pair classifier trained in noru-tactic from runtime
//! first-diff pairs. It never proposes a move by itself; it only decides
//! whether an already preferred root candidate may replace the incumbent.

use crate::board::{BOARD_SIZE, Board, Move, NUM_CELLS, Stone, to_idx, to_rc};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::OnceLock;

const FORMAT: &str = "noru-rq423-richer-relation-ue-root-accept-v1";
const FEATURE_COUNT: usize = 366;
const CENTER: i32 = (BOARD_SIZE as i32) / 2;
const DIRS: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];

const WINDOW_STAT_NAMES: [&str; 27] = [
    "clean_windows",
    "open_end_sum",
    "max_stones",
    "max_open_ends",
    "five_windows",
    "four_windows",
    "four_open_ge1",
    "four_open2",
    "four_broken",
    "four_inner_gap",
    "four_edge_gap",
    "three_windows",
    "three_open_ge1",
    "three_open2",
    "three_broken",
    "three_split",
    "two_windows",
    "two_open2",
    "dir_five",
    "dir_four",
    "dir_four_open_ge1",
    "dir_four_open2",
    "dir_three",
    "dir_three_open2",
    "double_four",
    "four_three",
    "double_three",
];

const REPLY_STAT_NAMES: [&str; 11] = [
    "points",
    "five_moves",
    "four_moves",
    "open_four_moves",
    "three_moves",
    "open_three_moves",
    "double_four_moves",
    "double_three_moves",
    "max_stones",
    "max_four_windows",
    "max_three_windows",
];

#[derive(Clone, Copy)]
pub(crate) struct RootAcceptDecision {
    pub(crate) probability: f32,
    pub(crate) threshold: f32,
    pub(crate) allows: bool,
}

struct RootAcceptModel {
    threshold: f32,
    mean: Vec<f32>,
    std: Vec<f32>,
    weights: Vec<f32>,
    bias: f32,
}

impl RootAcceptModel {
    fn load(path: &str) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("failed to read model: {e}"))?;
        let value: Value =
            serde_json::from_str(&text).map_err(|e| format!("failed to parse model: {e}"))?;
        if str_req(&value, "format")? != FORMAT {
            return Err("unsupported RQ423 root accept format".to_string());
        }
        let feature_names = str_array(value.get("feature_names"), "feature_names")?;
        if feature_names.len() != FEATURE_COUNT {
            return Err(format!(
                "feature width mismatch: expected {FEATURE_COUNT}, got {}",
                feature_names.len()
            ));
        }
        let standardizer = value
            .get("standardizer")
            .and_then(Value::as_object)
            .ok_or("missing standardizer")?;
        let logistic = value
            .get("logistic")
            .and_then(Value::as_object)
            .ok_or("missing logistic")?;
        let mean = f32_array(standardizer.get("mean"), "standardizer.mean")?;
        let std = f32_array(standardizer.get("std"), "standardizer.std")?;
        let weights = f32_array(logistic.get("weights"), "logistic.weights")?;
        let bias = f32_from_value(
            logistic.get("bias").ok_or("missing logistic.bias")?,
            "logistic.bias",
        )?;
        if mean.len() != FEATURE_COUNT
            || std.len() != FEATURE_COUNT
            || weights.len() != FEATURE_COUNT
        {
            return Err("model array width mismatch".to_string());
        }
        let threshold = parse_env_f32("NORU_RQ423_ROOT_ACCEPT_THRESHOLD")
            .filter(|value| value.is_finite())
            .unwrap_or(f32_from_value(
                value.get("threshold").ok_or("missing threshold")?,
                "threshold",
            )?);
        Ok(Self {
            threshold,
            mean,
            std,
            weights,
            bias,
        })
    }

    fn probability(&self, features: &[f32]) -> Option<f32> {
        if features.len() != FEATURE_COUNT {
            return None;
        }
        let mut z = self.bias;
        for idx in 0..FEATURE_COUNT {
            let std = self.std[idx].abs().max(1.0e-6);
            z += ((features[idx] - self.mean[idx]) / std) * self.weights[idx];
        }
        Some(sigmoid(z))
    }
}

pub(crate) fn root_accept_decision(
    board: &Board,
    incumbent: Move,
    candidate: Move,
) -> Option<RootAcceptDecision> {
    let model = model()?;
    let features = pair_features(board, incumbent, candidate);
    let probability = model.probability(&features)?;
    Some(RootAcceptDecision {
        probability,
        threshold: model.threshold,
        allows: probability >= model.threshold,
    })
}

#[doc(hidden)]
pub fn debug_pair_features(board: &Board, incumbent: Move, candidate: Move) -> Vec<f32> {
    pair_features(board, incumbent, candidate)
}

#[doc(hidden)]
pub(crate) fn debug_move_feature_maps(
    board: &Board,
    side: Stone,
    mv: Move,
) -> (BTreeMap<String, f32>, BTreeMap<String, f32>) {
    (
        move_features(board, side, mv),
        rich_move_features(board, side, mv),
    )
}

#[doc(hidden)]
pub fn debug_model_probability(model_path: &str, features: &[f32]) -> Result<(f32, f32), String> {
    let model = RootAcceptModel::load(model_path)?;
    let probability = model
        .probability(features)
        .ok_or_else(|| "feature width mismatch".to_string())?;
    Ok((probability, model.threshold))
}

fn model() -> Option<&'static RootAcceptModel> {
    static MODEL: OnceLock<Option<RootAcceptModel>> = OnceLock::new();
    MODEL.get_or_init(load_model).as_ref()
}

fn load_model() -> Option<RootAcceptModel> {
    let Ok(path) = std::env::var("NORU_RQ423_ROOT_ACCEPT_MODEL") else {
        return None;
    };
    let trimmed = path.trim();
    if is_disabled_value(trimmed) {
        return None;
    }
    Some(
        RootAcceptModel::load(trimmed)
            .unwrap_or_else(|e| panic!("invalid NORU_RQ423_ROOT_ACCEPT_MODEL={trimmed}: {e}")),
    )
}

fn pair_features(board: &Board, baseline: Move, test: Move) -> Vec<f32> {
    let side = board.side_to_move;
    let (by, bx) = to_rc(baseline);
    let (ty, tx) = to_rc(test);
    let bf = move_features(board, side, baseline);
    let tf = move_features(board, side, test);
    let mut out = Vec::with_capacity(FEATURE_COUNT);
    out.extend([
        (board.move_count + 1) as f32,
        if side == Stone::Black { 1.0 } else { 0.0 },
        (tx as i32 - bx as i32).abs() as f32,
        (ty as i32 - by as i32).abs() as f32,
        ((tx as i32 - bx as i32).abs() + (ty as i32 - by as i32).abs()) as f32,
        if zone(test) == zone(baseline) {
            1.0
        } else {
            0.0
        },
        if center_dist(test) <= 2 { 1.0 } else { 0.0 },
        if center_dist(test) <= 3 { 1.0 } else { 0.0 },
        if edge_dist(test) <= 1 { 1.0 } else { 0.0 },
        if get(&tf, "center_dist") > get(&bf, "center_dist") {
            1.0
        } else {
            0.0
        },
        if get(&tf, "attack") > get(&bf, "attack") {
            1.0
        } else {
            0.0
        },
        if get(&tf, "block") < get(&bf, "block") {
            1.0
        } else {
            0.0
        },
    ]);
    push_maps(&mut out, &bf, &tf);

    let rich_bf = rich_move_features(board, side, baseline);
    let rich_tf = rich_move_features(board, side, test);
    push_maps(&mut out, &rich_bf, &rich_tf);
    debug_assert_eq!(out.len(), FEATURE_COUNT);
    out
}

fn push_maps(out: &mut Vec<f32>, baseline: &BTreeMap<String, f32>, test: &BTreeMap<String, f32>) {
    for key in baseline.keys() {
        out.push(get(test_map_side(baseline, key), key));
    }
    for key in baseline.keys() {
        out.push(get(test, key));
    }
    for key in baseline.keys() {
        out.push(get(test, key) - get(baseline, key));
    }
}

fn test_map_side<'a>(baseline: &'a BTreeMap<String, f32>, _key: &str) -> &'a BTreeMap<String, f32> {
    baseline
}

fn move_features(board: &Board, side: Stone, mv: Move) -> BTreeMap<String, f32> {
    let opp = side.opponent();
    let (attack, attack2) = threat_bucket(board, mv, side);
    let (block, block2) = threat_bucket(board, mv, opp);
    let (r1_own, r1_opp, r1_empty) = radius_counts(board, mv, side, 1);
    let (r2_own, r2_opp, r2_empty) = radius_counts(board, mv, side, 2);
    let mut feat = BTreeMap::new();
    feat.insert("x".to_string(), col_of(mv) as f32);
    feat.insert("y".to_string(), row_of(mv) as f32);
    feat.insert("center_dist".to_string(), center_dist(mv) as f32);
    feat.insert("edge_dist".to_string(), edge_dist(mv) as f32);
    feat.insert("zone".to_string(), zone(mv) as f32);
    feat.insert("attack".to_string(), attack as f32);
    feat.insert("attack2".to_string(), attack2 as f32);
    feat.insert("block".to_string(), block as f32);
    feat.insert("block2".to_string(), block2 as f32);
    feat.insert("max_threat".to_string(), attack.max(block) as f32);
    feat.insert("multi_threat".to_string(), attack2.max(block2) as f32);
    feat.insert(
        "attack_minus_block".to_string(),
        attack as f32 - block as f32,
    );
    feat.insert("r1_own".to_string(), r1_own as f32);
    feat.insert("r1_opp".to_string(), r1_opp as f32);
    feat.insert("r1_empty".to_string(), r1_empty as f32);
    feat.insert("r2_own".to_string(), r2_own as f32);
    feat.insert("r2_opp".to_string(), r2_opp as f32);
    feat.insert("r2_empty".to_string(), r2_empty as f32);
    for (prefix, color) in [("own", side), ("opp", opp)] {
        for (name, value) in line_shape_features(board, mv, color) {
            feat.insert(format!("{prefix}_{name}"), value);
        }
    }
    feat
}

fn rich_move_features(board: &Board, side: Stone, mv: Move) -> BTreeMap<String, f32> {
    let opp = side.opponent();
    let mut feat = BTreeMap::new();
    for radius in [3, 4] {
        let (own, opp_count, empty) = radius_counts(board, mv, side, radius);
        feat.insert(format!("r{radius}_own"), own as f32);
        feat.insert(format!("r{radius}_opp"), opp_count as f32);
        feat.insert(format!("r{radius}_empty"), empty as f32);
    }
    for (name, value) in nearest_distances(board, side, mv) {
        feat.insert(name, value);
    }
    let own_windows = window_stats(board, side, mv);
    let block_windows = window_stats(board, opp, mv);
    for (idx, name) in WINDOW_STAT_NAMES.iter().enumerate() {
        feat.insert(format!("own_win_{name}"), own_windows[idx]);
        feat.insert(format!("block_win_{name}"), block_windows[idx]);
    }
    for radius in [2, 3] {
        let reply = local_reply_stats(board, side, mv, radius);
        for (idx, name) in REPLY_STAT_NAMES.iter().enumerate() {
            feat.insert(format!("opp_reply_r{radius}_{name}"), reply[idx]);
        }
    }

    let attack_four = own_windows[4] + own_windows[5];
    let block_four = block_windows[4] + block_windows[5];
    let reply_four = get(&feat, "opp_reply_r3_four_moves") + get(&feat, "opp_reply_r3_five_moves");
    let attack_three = own_windows[11];
    let block_three = block_windows[11];
    let reply_three = get(&feat, "opp_reply_r3_three_moves");
    feat.insert(
        "force_window_balance".to_string(),
        attack_four + block_four - reply_four,
    );
    feat.insert(
        "three_window_balance".to_string(),
        attack_three + block_three - reply_three,
    );
    feat.insert(
        "attack_block_four_any".to_string(),
        if attack_four > 0.0 || block_four > 0.0 {
            1.0
        } else {
            0.0
        },
    );
    feat.insert(
        "attack_block_double_any".to_string(),
        if own_windows[24] > 0.0
            || own_windows[26] > 0.0
            || block_windows[24] > 0.0
            || block_windows[26] > 0.0
        {
            1.0
        } else {
            0.0
        },
    );
    feat.insert(
        "reply_danger_any".to_string(),
        if reply_four > 0.0 || get(&feat, "opp_reply_r3_double_three_moves") > 0.0 {
            1.0
        } else {
            0.0
        },
    );
    feat.insert(
        "center_pressure".to_string(),
        (8 - center_dist(mv)).max(0) as f32 * (1.0 + get(&feat, "r3_own") + get(&feat, "r3_opp")),
    );
    feat
}

fn line_shape_features(board: &Board, mv: Move, side: Stone) -> BTreeMap<String, f32> {
    let mut out = BTreeMap::new();
    out.insert("line_best_stones".to_string(), 0.0);
    out.insert("line_second_stones".to_string(), 0.0);
    out.insert("line_best_open".to_string(), 0.0);
    out.insert("line_open_dirs".to_string(), 0.0);
    if !board.is_empty(mv) {
        return out;
    }
    let mut values = Vec::new();
    for (dc, dr) in DIRS {
        let (a, open_a) = count_dir(board, mv, side, dc, dr);
        let (b, open_b) = count_dir(board, mv, side, -dc, -dr);
        values.push((a + b + 1, open_a as i32 + open_b as i32));
    }
    values.sort_by(|a, b| b.cmp(a));
    out.insert("line_best_stones".to_string(), values[0].0 as f32);
    out.insert("line_second_stones".to_string(), values[1].0 as f32);
    out.insert("line_best_open".to_string(), values[0].1 as f32);
    out.insert(
        "line_open_dirs".to_string(),
        values.iter().filter(|(_, open)| *open > 0).count() as f32,
    );
    out
}

fn threat_bucket(board: &Board, mv: Move, side: Stone) -> (usize, usize) {
    if !board.is_empty(mv) {
        return (0, 0);
    }
    let mut buckets = Vec::with_capacity(4);
    for (dc, dr) in DIRS {
        let (a, open_a) = count_dir(board, mv, side, dc, dr);
        let (b, open_b) = count_dir(board, mv, side, -dc, -dr);
        let stones = a + b + 1;
        let open_ends = open_a as usize + open_b as usize;
        let bucket = if stones >= 5 {
            5
        } else if stones == 4 && open_ends == 2 {
            5
        } else if stones == 4 && open_ends >= 1 {
            4
        } else if stones == 3 && open_ends == 2 {
            3
        } else if stones == 3 && open_ends >= 1 {
            2
        } else if stones == 2 && open_ends == 2 {
            1
        } else {
            0
        };
        buckets.push(bucket);
    }
    buckets.sort_by(|a, b| b.cmp(a));
    (buckets[0], buckets[1])
}

fn count_dir(board: &Board, mv: Move, side: Stone, dc: i32, dr: i32) -> (usize, bool) {
    let mut count = 0;
    let mut row = row_of(mv) + dr;
    let mut col = col_of(mv) + dc;
    while in_bounds(row, col) && stone_at_rc(board, row, col) == Some(side) {
        count += 1;
        row += dr;
        col += dc;
    }
    (
        count,
        in_bounds(row, col) && stone_at_rc(board, row, col).is_none(),
    )
}

fn radius_counts(board: &Board, mv: Move, side: Stone, radius: i32) -> (usize, usize, usize) {
    let opp = side.opponent();
    let mut own = 0;
    let mut opp_count = 0;
    let mut empty = 0;
    let row = row_of(mv);
    let col = col_of(mv);
    for rr in row - radius..=row + radius {
        for cc in col - radius..=col + radius {
            if rr == row && cc == col {
                continue;
            }
            if !in_bounds(rr, cc) {
                continue;
            }
            match stone_at_rc(board, rr, cc) {
                Some(value) if value == side => own += 1,
                Some(value) if value == opp => opp_count += 1,
                Some(_) => {}
                None => empty += 1,
            }
        }
    }
    (own, opp_count, empty)
}

fn nearest_distances(board: &Board, side: Stone, mv: Move) -> BTreeMap<String, f32> {
    let opp = side.opponent();
    let mut own_min = 99;
    let mut opp_min = 99;
    let mut own_count = 0;
    let mut opp_count = 0;
    let row = row_of(mv);
    let col = col_of(mv);
    for idx in 0..NUM_CELLS {
        let Some(value) = stone_at(board, idx) else {
            continue;
        };
        let dist = (row_of(idx) - row).abs() + (col_of(idx) - col).abs();
        if value == side {
            own_count += 1;
            own_min = own_min.min(dist);
        } else if value == opp {
            opp_count += 1;
            opp_min = opp_min.min(dist);
        }
    }
    BTreeMap::from([
        (
            "nearest_own".to_string(),
            if own_min == 99 { 16.0 } else { own_min as f32 },
        ),
        (
            "nearest_opp".to_string(),
            if opp_min == 99 { 16.0 } else { opp_min as f32 },
        ),
        ("own_stones".to_string(), own_count as f32),
        ("opp_stones".to_string(), opp_count as f32),
    ])
}

fn window_stats(board: &Board, side: Stone, mv: Move) -> [f32; 27] {
    let mut stats = [0.0f32; 27];
    if !board.is_empty(mv) {
        return stats;
    }
    let opp = side.opponent();
    let row = row_of(mv);
    let col = col_of(mv);
    let mut dir_five = 0;
    let mut dir_four = 0;
    let mut dir_four_open_ge1 = 0;
    let mut dir_four_open2 = 0;
    let mut dir_three = 0;
    let mut dir_three_open2 = 0;

    for (dc, dr) in DIRS {
        let mut local_five = 0;
        let mut local_four = 0;
        let mut local_four_open_ge1 = 0;
        let mut local_four_open2 = 0;
        let mut local_three = 0;
        let mut local_three_open2 = 0;
        for start in -4..=0 {
            let mut values = [0i8; 5];
            let mut blocked = false;
            let mut has_opp = false;
            for idx in 0..5 {
                let rr = row + (start + idx as i32) * dr;
                let cc = col + (start + idx as i32) * dc;
                let value = cell_value(board, rr, cc, row, col, side);
                values[idx] = value;
                if value == -1 {
                    blocked = true;
                    break;
                }
                if value == stone_code(opp) {
                    has_opp = true;
                    break;
                }
            }
            if blocked || has_opp {
                continue;
            }
            let before = cell_value(
                board,
                row + (start - 1) * dr,
                col + (start - 1) * dc,
                row,
                col,
                side,
            );
            let after = cell_value(
                board,
                row + (start + 5) * dr,
                col + (start + 5) * dc,
                row,
                col,
                side,
            );
            let open_ends = (before == 0) as usize + (after == 0) as usize;
            let stones = values
                .iter()
                .filter(|value| **value == stone_code(side))
                .count();
            let empties = values.iter().filter(|value| **value == 0).count();
            let groups = group_count(&values, stone_code(side));

            stats[0] += 1.0;
            stats[1] += open_ends as f32;
            stats[2] = stats[2].max(stones as f32);
            stats[3] = stats[3].max(open_ends as f32);

            if stones >= 5 {
                stats[4] += 1.0;
                local_five += 1;
            } else if stones == 4 && empties == 1 {
                stats[5] += 1.0;
                local_four += 1;
                if open_ends >= 1 {
                    stats[6] += 1.0;
                    local_four_open_ge1 += 1;
                }
                if open_ends == 2 {
                    stats[7] += 1.0;
                    local_four_open2 += 1;
                }
                if groups >= 2 {
                    stats[8] += 1.0;
                }
                let gap_idx = values.iter().position(|value| *value == 0).unwrap_or(0);
                if (1..=3).contains(&gap_idx) {
                    stats[9] += 1.0;
                } else {
                    stats[10] += 1.0;
                }
            } else if stones == 3 && empties == 2 {
                stats[11] += 1.0;
                local_three += 1;
                if open_ends >= 1 {
                    stats[12] += 1.0;
                }
                if open_ends == 2 {
                    stats[13] += 1.0;
                    local_three_open2 += 1;
                }
                if groups >= 2 {
                    stats[14] += 1.0;
                }
                if groups >= 3 {
                    stats[15] += 1.0;
                }
            } else if stones == 2 && empties == 3 {
                stats[16] += 1.0;
                if open_ends == 2 {
                    stats[17] += 1.0;
                }
            }
        }
        if local_five > 0 {
            dir_five += 1;
        }
        if local_four > 0 {
            dir_four += 1;
        }
        if local_four_open_ge1 > 0 {
            dir_four_open_ge1 += 1;
        }
        if local_four_open2 > 0 {
            dir_four_open2 += 1;
        }
        if local_three > 0 {
            dir_three += 1;
        }
        if local_three_open2 > 0 {
            dir_three_open2 += 1;
        }
    }
    stats[18] = dir_five as f32;
    stats[19] = dir_four as f32;
    stats[20] = dir_four_open_ge1 as f32;
    stats[21] = dir_four_open2 as f32;
    stats[22] = dir_three as f32;
    stats[23] = dir_three_open2 as f32;
    stats[24] = if dir_four >= 2 { 1.0 } else { 0.0 };
    stats[25] = if dir_four >= 1 && dir_three >= 1 {
        1.0
    } else {
        0.0
    };
    stats[26] = if dir_three >= 2 { 1.0 } else { 0.0 };
    stats
}

fn local_reply_stats(board: &Board, side: Stone, mv: Move, radius: i32) -> [f32; 11] {
    let mut stats = [0.0f32; 11];
    if !board.is_empty(mv) {
        return stats;
    }
    let mut child = board.clone();
    match side {
        Stone::Black => child.black.set(mv),
        Stone::White => child.white.set(mv),
    }
    let opp = side.opponent();
    let row = row_of(mv);
    let col = col_of(mv);
    for rr in row - radius..=row + radius {
        for cc in col - radius..=col + radius {
            if !in_bounds(rr, cc) {
                continue;
            }
            let reply = to_idx(rr as usize, cc as usize);
            if !child.is_empty(reply) {
                continue;
            }
            let reply_stats = window_stats(&child, opp, reply);
            stats[0] += 1.0;
            stats[8] = stats[8].max(reply_stats[2]);
            stats[9] = stats[9].max(reply_stats[5]);
            stats[10] = stats[10].max(reply_stats[11]);
            if reply_stats[4] > 0.0 {
                stats[1] += 1.0;
            }
            if reply_stats[5] > 0.0 {
                stats[2] += 1.0;
            }
            if reply_stats[7] > 0.0 {
                stats[3] += 1.0;
            }
            if reply_stats[11] > 0.0 {
                stats[4] += 1.0;
            }
            if reply_stats[13] > 0.0 {
                stats[5] += 1.0;
            }
            if reply_stats[24] > 0.0 {
                stats[6] += 1.0;
            }
            if reply_stats[26] > 0.0 {
                stats[7] += 1.0;
            }
        }
    }
    stats
}

fn group_count(values: &[i8; 5], code: i8) -> usize {
    let mut groups = 0;
    let mut in_group = false;
    for value in values {
        if *value == code {
            if !in_group {
                groups += 1;
                in_group = true;
            }
        } else {
            in_group = false;
        }
    }
    groups
}

fn cell_value(
    board: &Board,
    row: i32,
    col: i32,
    place_row: i32,
    place_col: i32,
    side: Stone,
) -> i8 {
    if !in_bounds(row, col) {
        return -1;
    }
    if row == place_row && col == place_col {
        return stone_code(side);
    }
    stone_at_rc(board, row, col).map(stone_code).unwrap_or(0)
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

fn stone_at_rc(board: &Board, row: i32, col: i32) -> Option<Stone> {
    if !in_bounds(row, col) {
        return None;
    }
    stone_at(board, to_idx(row as usize, col as usize))
}

fn stone_code(side: Stone) -> i8 {
    match side {
        Stone::Black => 1,
        Stone::White => 2,
    }
}

fn row_of(mv: Move) -> i32 {
    (mv / BOARD_SIZE) as i32
}

fn col_of(mv: Move) -> i32 {
    (mv % BOARD_SIZE) as i32
}

fn center_dist(mv: Move) -> i32 {
    (col_of(mv) - CENTER).abs() + (row_of(mv) - CENTER).abs()
}

fn edge_dist(mv: Move) -> i32 {
    let row = row_of(mv);
    let col = col_of(mv);
    row.min(col)
        .min(BOARD_SIZE as i32 - 1 - row)
        .min(BOARD_SIZE as i32 - 1 - col)
}

fn zone(mv: Move) -> i32 {
    let row = row_of(mv) as usize;
    let col = col_of(mv) as usize;
    (row / 5).min(2) as i32 * 3 + (col / 5).min(2) as i32
}

fn in_bounds(row: i32, col: i32) -> bool {
    row >= 0 && col >= 0 && row < BOARD_SIZE as i32 && col < BOARD_SIZE as i32
}

fn get(map: &BTreeMap<String, f32>, key: &str) -> f32 {
    *map.get(key).unwrap_or(&0.0)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value.clamp(-40.0, 40.0)).exp())
}

fn str_req<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string field {key}"))
}

fn f32_from_value(value: &Value, name: &str) -> Result<f32, String> {
    value
        .as_f64()
        .map(|v| v as f32)
        .ok_or_else(|| format!("missing numeric field {name}"))
}

fn f32_array(value: Option<&Value>, name: &str) -> Result<Vec<f32>, String> {
    value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array {name}"))?
        .iter()
        .map(|item| f32_from_value(item, name))
        .collect()
}

fn str_array(value: Option<&Value>, name: &str) -> Result<Vec<String>, String> {
    value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array {name}"))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("non-string entry in {name}"))
        })
        .collect()
}

fn parse_env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok()?.trim().parse().ok()
}

fn is_disabled_value(value: &str) -> bool {
    value.is_empty()
        || value.eq_ignore_ascii_case("0")
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("none")
}
