//! Threat-space search candidate vocabulary.
//!
//! RQ597 is intentionally a static reference implementation. It identifies
//! non-forcing moves that improve the attacker's next-move threat sources.
//! Proof search and main-search integration remain separate, preregistered
//! work.

use crate::board::{BOARD_SIZE, BitBoard, Board, Move, NUM_CELLS, Stone};
use crate::heuristic::DIR;
use crate::pattern_table::{WindowThreat, classify_window_after_play_with_flags, read_window};
use crate::vct::{ThreatKind, VctConfig, classify_move_fast_with_flags, search_vct_with_stats};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuietThreatConfig {
    pub min_gain: u8,
    pub enable_jump_three: bool,
    pub enable_gap_four: bool,
}

impl Default for QuietThreatConfig {
    fn default() -> Self {
        Self {
            min_gain: 1,
            enable_jump_three: true,
            enable_gap_four: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuietThreatCandidate {
    pub mv: Move,
    pub forcing_gains: u8,
    pub winning_gains: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DependencyQuietCandidate {
    pub mv: Move,
    pub forcing_gains: u8,
    pub winning_gains: u8,
    pub dependency_links: u16,
    pub max_reused_support: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencyCandidateArms {
    pub d1: Vec<DependencyQuietCandidate>,
    pub d2: Vec<DependencyQuietCandidate>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseRelevanceAudit {
    pub quiet_move: Move,
    pub forcing_gains: u8,
    pub winning_gains: u8,
    pub gained_sources: Vec<Move>,
    pub gained_line_count: usize,
    pub legal_replies: Vec<Move>,
    pub immediate_replies: Vec<Move>,
    pub defender_forcing_replies: Vec<Move>,
    pub footprint_replies: Vec<Move>,
    pub causal_replies: Vec<Move>,
    pub f1_replies: Vec<Move>,
    pub f2_replies: Vec<Move>,
    pub causal_outside_footprint: Vec<Move>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q1DefenseOutcome {
    Proved,
    Exhausted,
    Deadline,
    NodeBudget,
    ImmediateWin,
}

impl Q1DefenseOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Proved => "proved",
            Self::Exhausted => "exhausted",
            Self::Deadline => "deadline",
            Self::NodeBudget => "node_budget",
            Self::ImmediateWin => "immediate_win",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Q1DefenseAttempt {
    pub mv: Move,
    pub outcome: Q1DefenseOutcome,
    pub child_nodes: u64,
    pub child_first_move: Option<Move>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Q1CandidateAttempt {
    pub mv: Move,
    pub forcing_gains: u8,
    pub winning_gains: u8,
    pub defenses_total: usize,
    pub complete: bool,
    pub defenses: Vec<Q1DefenseAttempt>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q1TssStopReason {
    Proved,
    Exhausted,
    Deadline,
    NodeBudget,
}

impl Q1TssStopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Proved => "proved",
            Self::Exhausted => "exhausted",
            Self::Deadline => "deadline",
            Self::NodeBudget => "node_budget",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Q1TssConfig {
    pub max_candidates: usize,
    pub child_vct_depth: u32,
    pub time_budget: Option<Duration>,
    pub node_budget: Option<u64>,
}

impl Default for Q1TssConfig {
    fn default() -> Self {
        Self {
            max_candidates: 8,
            child_vct_depth: 18,
            time_budget: Some(Duration::from_millis(1000)),
            node_budget: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Q1TssResult {
    pub selected_move: Option<Move>,
    pub candidate_count: usize,
    pub candidates_tested: usize,
    pub child_nodes: u64,
    pub elapsed: Duration,
    pub stop_reason: Q1TssStopReason,
    pub attempts: Vec<Q1CandidateAttempt>,
}

#[derive(Clone, Copy)]
struct Q1ChildResult {
    outcome: Q1DefenseOutcome,
    nodes: u64,
    first_move: Option<Move>,
}

#[derive(Clone, Copy)]
struct ThreatLine {
    footprint: BitBoard,
    support: BitBoard,
}

/// Generate quiet preparation moves using a full-board semantic reference.
///
/// A candidate is not forcing in the current position, does not allow an
/// immediate opponent Five after it is played, and upgrades at least
/// `config.min_gain` next-move threat sources for the original attacker.
pub fn generate_quiet_threat_candidates(
    board: &mut Board,
    config: QuietThreatConfig,
) -> Vec<QuietThreatCandidate> {
    let attacker = board.side_to_move;
    let history_len = board.history.len();
    let zobrist = board.zobrist;
    let last_move = board.last_move;
    let base = classify_all(board, attacker, config);
    let mut out = Vec::new();

    for mv in 0..NUM_CELLS {
        if !board.is_empty(mv) || base[mv].is_forcing() {
            continue;
        }

        board.make_move(mv);
        if has_immediate_five(board, attacker.opponent(), config) {
            board.undo_move();
            continue;
        }

        let after = classify_all(board, attacker, config);
        let mut forcing_gains = 0u8;
        let mut winning_gains = 0u8;
        for cell in 0..NUM_CELLS {
            let before_kind = base[cell];
            let after_kind = after[cell];
            if after_kind.is_forcing() && threat_strength(after_kind) > threat_strength(before_kind)
            {
                forcing_gains = forcing_gains.saturating_add(1);
                if after_kind.is_winning() && !before_kind.is_winning() {
                    winning_gains = winning_gains.saturating_add(1);
                }
            }
        }
        board.undo_move();

        if forcing_gains >= config.min_gain {
            out.push(QuietThreatCandidate {
                mv,
                forcing_gains,
                winning_gains,
            });
        }
    }

    debug_assert_eq!(board.history.len(), history_len);
    debug_assert_eq!(board.zobrist, zobrist);
    debug_assert_eq!(board.last_move, last_move);
    debug_assert_eq!(board.side_to_move, attacker);

    out.sort_unstable_by(|a, b| {
        b.winning_gains
            .cmp(&a.winning_gains)
            .then_with(|| b.forcing_gains.cmp(&a.forcing_gains))
            .then_with(|| a.mv.cmp(&b.mv))
    });
    out
}

/// Generate the two preregistered RQ598 dependency arms in one semantic pass.
///
/// D1 requires an upgraded forcing line to intersect an existing forcing-line
/// footprint. D2 additionally requires at least two pre-existing attacker
/// stones to be shared by one potential/existing line pair.
pub fn generate_dependency_quiet_candidates(
    board: &mut Board,
    config: QuietThreatConfig,
) -> DependencyCandidateArms {
    let attacker = board.side_to_move;
    let history_len = board.history.len();
    let zobrist = board.zobrist;
    let last_move = board.last_move;
    let original_black = board.black;
    let original_white = board.white;
    let original_mine = if attacker == Stone::Black {
        original_black
    } else {
        original_white
    };
    let base = classify_all(board, attacker, config);
    let base_directions = classify_all_directions(board, attacker, config);
    let existing_lines = collect_existing_lines(board, &base, &base_directions, &original_mine);
    let mut d1 = Vec::new();
    let mut d2 = Vec::new();

    for mv in 0..NUM_CELLS {
        if !board.is_empty(mv) || base[mv].is_forcing() {
            continue;
        }

        board.make_move(mv);
        if has_immediate_five(board, attacker.opponent(), config) {
            board.undo_move();
            continue;
        }

        let after = classify_all(board, attacker, config);
        let after_directions = classify_all_directions(board, attacker, config);
        let mut forcing_gains = 0u8;
        let mut winning_gains = 0u8;
        let mut dependency_links = 0u16;
        let mut max_reused_support = 0u8;

        for source in 0..NUM_CELLS {
            let before_kind = base[source];
            let after_kind = after[source];
            if !after_kind.is_forcing()
                || threat_strength(after_kind) <= threat_strength(before_kind)
            {
                continue;
            }
            forcing_gains = forcing_gains.saturating_add(1);
            if after_kind.is_winning() && !before_kind.is_winning() {
                winning_gains = winning_gains.saturating_add(1);
            }

            for dir_idx in 0..DIR.len() {
                let before_line = base_directions[source][dir_idx];
                let after_line = after_directions[source][dir_idx];
                if !direction_is_forcing(after_line)
                    || direction_strength(after_line) <= direction_strength(before_line)
                {
                    continue;
                }
                let potential = make_threat_line(source, dir_idx, &original_mine);
                for existing in &existing_lines {
                    if bitboards_intersect(&potential.footprint, &existing.footprint) {
                        dependency_links = dependency_links.saturating_add(1);
                    }
                    let reused = intersection_count(&potential.support, &existing.support) as u8;
                    max_reused_support = max_reused_support.max(reused);
                }
            }
        }
        board.undo_move();

        let candidate = DependencyQuietCandidate {
            mv,
            forcing_gains,
            winning_gains,
            dependency_links,
            max_reused_support,
        };
        if forcing_gains >= 1 && dependency_links > 0 {
            d1.push(candidate);
        }
        if forcing_gains >= 1 && max_reused_support >= 2 {
            d2.push(candidate);
        }
    }

    debug_assert_eq!(board.history.len(), history_len);
    debug_assert_eq!(board.zobrist, zobrist);
    debug_assert_eq!(board.last_move, last_move);
    debug_assert_eq!(board.side_to_move, attacker);
    debug_assert!(board.black == original_black);
    debug_assert!(board.white == original_white);

    sort_dependency_candidates(&mut d1);
    sort_dependency_candidates(&mut d2);
    debug_assert!(d2.iter().all(|candidate| {
        d1.iter()
            .any(|d1_candidate| d1_candidate.mv == candidate.mv)
    }));
    DependencyCandidateArms { d1, d2 }
}

/// Classify legal responses to one labelled quiet preparation move.
///
/// RQ603 is a static audit only. F1 retains replies in gained directional
/// footprints plus global forcing counters. F2 retains replies that actually
/// weaken a gained aggregate source plus the same global counters.
pub fn audit_quiet_response_relevance(
    board: &mut Board,
    quiet_move: Move,
    config: QuietThreatConfig,
) -> Result<ResponseRelevanceAudit, &'static str> {
    if quiet_move >= NUM_CELLS || !board.is_empty(quiet_move) {
        return Err("quiet move is not legal");
    }

    let attacker = board.side_to_move;
    let history_len = board.history.len();
    let black = board.black;
    let white = board.white;
    let zobrist = board.zobrist;
    let last_move = board.last_move;
    let base = classify_all(board, attacker, config);
    if base[quiet_move].is_forcing() {
        return Err("labelled move is already forcing");
    }
    let base_directions = classify_all_directions(board, attacker, config);

    board.make_move(quiet_move);
    if has_immediate_five(board, attacker.opponent(), config) {
        board.undo_move();
        return Err("quiet move leaves an immediate opponent five");
    }

    let after = classify_all(board, attacker, config);
    let after_directions = classify_all_directions(board, attacker, config);
    let mut forcing_gains = 0u8;
    let mut winning_gains = 0u8;
    let mut gained_sources = Vec::new();
    let mut gained_source_kinds = Vec::new();
    let mut gained_footprint = BitBoard::EMPTY;
    let mut gained_line_count = 0usize;

    for source in 0..NUM_CELLS {
        let before_kind = base[source];
        let after_kind = after[source];
        if !after_kind.is_forcing() || threat_strength(after_kind) <= threat_strength(before_kind) {
            continue;
        }
        forcing_gains = forcing_gains.saturating_add(1);
        if after_kind.is_winning() && !before_kind.is_winning() {
            winning_gains = winning_gains.saturating_add(1);
        }
        gained_sources.push(source);
        gained_source_kinds.push((source, after_kind));

        for dir_idx in 0..DIR.len() {
            let before_line = base_directions[source][dir_idx];
            let after_line = after_directions[source][dir_idx];
            if direction_is_forcing(after_line)
                && direction_strength(after_line) > direction_strength(before_line)
            {
                let line = make_threat_line(
                    source,
                    dir_idx,
                    &if attacker == Stone::Black {
                        black
                    } else {
                        white
                    },
                );
                gained_footprint.lo |= line.footprint.lo;
                gained_footprint.hi |= line.footprint.hi;
                gained_line_count += 1;
            }
        }
    }

    if forcing_gains == 0 {
        board.undo_move();
        return Err("labelled move has no forcing gain");
    }

    let defender = board.side_to_move;
    let legal_replies = board.legal_moves();
    let mut immediate_replies = Vec::new();
    let mut defender_forcing_replies = Vec::new();
    let mut footprint_replies = Vec::new();
    let mut causal_replies = Vec::new();
    let mut f1_replies = Vec::new();
    let mut f2_replies = Vec::new();
    let mut causal_outside_footprint = Vec::new();

    for &reply in &legal_replies {
        let defender_kind = classify_move_fast_with_flags(
            board,
            reply,
            defender,
            config.enable_jump_three,
            config.enable_gap_four,
        );
        let defender_forcing = defender_kind.is_forcing();
        let in_footprint = gained_footprint.get(reply);

        board.make_move(reply);
        let immediate = board.check_win(reply);
        let weakens_gain = gained_source_kinds.iter().any(|&(source, expected)| {
            if !board.is_empty(source) {
                return true;
            }
            let actual = classify_move_fast_with_flags(
                board,
                source,
                attacker,
                config.enable_jump_three,
                config.enable_gap_four,
            );
            threat_strength(actual) < threat_strength(expected)
        });
        board.undo_move();

        if immediate {
            immediate_replies.push(reply);
        }
        if defender_forcing {
            defender_forcing_replies.push(reply);
        }
        if in_footprint {
            footprint_replies.push(reply);
        }
        if weakens_gain {
            causal_replies.push(reply);
            if !in_footprint {
                causal_outside_footprint.push(reply);
            }
        }
        if immediate || defender_forcing || in_footprint {
            f1_replies.push(reply);
        }
        if immediate || defender_forcing || weakens_gain {
            f2_replies.push(reply);
        }
    }

    debug_assert!(
        immediate_replies
            .iter()
            .all(|reply| f1_replies.contains(reply) && f2_replies.contains(reply))
    );
    debug_assert!(
        defender_forcing_replies
            .iter()
            .all(|reply| f1_replies.contains(reply) && f2_replies.contains(reply))
    );

    board.undo_move();
    debug_assert_eq!(board.history.len(), history_len);
    debug_assert!(board.black == black);
    debug_assert!(board.white == white);
    debug_assert_eq!(board.side_to_move, attacker);
    debug_assert_eq!(board.zobrist, zobrist);
    debug_assert_eq!(board.last_move, last_move);

    Ok(ResponseRelevanceAudit {
        quiet_move,
        forcing_gains,
        winning_gains,
        gained_sources,
        gained_line_count,
        legal_replies,
        immediate_replies,
        defender_forcing_replies,
        footprint_replies,
        causal_replies,
        f1_replies,
        f2_replies,
        causal_outside_footprint,
    })
}

/// Try one quiet preparation move and prove every legal defender reply with
/// the canonical root-scoped JumpThree VCT. This is an offline Stage 2B
/// reference path; it is not connected to the main search.
pub fn search_q1_tss_root(board: &mut Board, config: &Q1TssConfig) -> Q1TssResult {
    search_q1_tss_root_with(
        board,
        config,
        |child_board, config, time_budget, node_budget| {
            let mut child_config = VctConfig::default();
            child_config.max_depth = config.child_vct_depth;
            child_config.time_budget = time_budget;
            child_config.node_budget = node_budget;
            child_config.enable_jump_three_attack_defense = true;
            child_config.enable_jump_three_counter = true;
            child_config.jump_attack_max_or_levels = 1;
            child_config.enable_gap_four = false;
            child_config.use_fast_classify = true;
            child_config.use_reach_mask = true;

            let result = search_vct_with_stats(child_board, &child_config);
            let first_move = result
                .sequence
                .as_ref()
                .and_then(|sequence| sequence.first().copied());
            let outcome = if result.sequence.is_some() {
                Q1DefenseOutcome::Proved
            } else if result.stats.hit_node_budget() {
                Q1DefenseOutcome::NodeBudget
            } else if result.stats.hit_deadline() {
                Q1DefenseOutcome::Deadline
            } else {
                Q1DefenseOutcome::Exhausted
            };
            Q1ChildResult {
                outcome,
                nodes: result.stats.nodes,
                first_move,
            }
        },
    )
}

fn search_q1_tss_root_with<F>(
    board: &mut Board,
    config: &Q1TssConfig,
    mut prove_child: F,
) -> Q1TssResult
where
    F: FnMut(&mut Board, &Q1TssConfig, Option<Duration>, Option<u64>) -> Q1ChildResult,
{
    let started = Instant::now();
    let deadline = config.time_budget.map(|budget| started + budget);
    let history_len = board.history.len();
    let black = board.black;
    let white = board.white;
    let side_to_move = board.side_to_move;
    let zobrist = board.zobrist;
    let last_move = board.last_move;

    let mut candidates = generate_quiet_threat_candidates(board, QuietThreatConfig::default());
    candidates.truncate(config.max_candidates);
    let candidate_count = candidates.len();
    let mut attempts = Vec::with_capacity(candidate_count);
    let mut child_nodes = 0u64;
    let mut selected_move = None;
    let mut stop_reason = Q1TssStopReason::Exhausted;

    'candidate: for candidate in candidates {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            stop_reason = Q1TssStopReason::Deadline;
            break;
        }
        if config.node_budget.is_some_and(|limit| child_nodes >= limit) {
            stop_reason = Q1TssStopReason::NodeBudget;
            break;
        }

        board.make_move(candidate.mv);
        let defenses = board.legal_moves();
        let mut attempt = Q1CandidateAttempt {
            mv: candidate.mv,
            forcing_gains: candidate.forcing_gains,
            winning_gains: candidate.winning_gains,
            defenses_total: defenses.len(),
            complete: false,
            defenses: Vec::with_capacity(defenses.len()),
        };

        for defense in defenses {
            let remaining_time = match deadline {
                Some(limit) => {
                    let now = Instant::now();
                    if now >= limit {
                        attempts.push(attempt);
                        board.undo_move();
                        stop_reason = Q1TssStopReason::Deadline;
                        break 'candidate;
                    }
                    Some(limit.saturating_duration_since(now))
                }
                None => None,
            };
            let remaining_nodes = match config.node_budget {
                Some(limit) => {
                    if child_nodes >= limit {
                        attempts.push(attempt);
                        board.undo_move();
                        stop_reason = Q1TssStopReason::NodeBudget;
                        break 'candidate;
                    }
                    Some(limit - child_nodes)
                }
                None => None,
            };

            board.make_move(defense);
            let child = if board.check_win(defense) {
                Q1ChildResult {
                    outcome: Q1DefenseOutcome::ImmediateWin,
                    nodes: 0,
                    first_move: None,
                }
            } else {
                let child_history_len = board.history.len();
                let child_zobrist = board.zobrist;
                let result = prove_child(board, config, remaining_time, remaining_nodes);
                debug_assert_eq!(board.history.len(), child_history_len);
                debug_assert_eq!(board.zobrist, child_zobrist);
                result
            };
            child_nodes = child_nodes.saturating_add(child.nodes);
            attempt.defenses.push(Q1DefenseAttempt {
                mv: defense,
                outcome: child.outcome,
                child_nodes: child.nodes,
                child_first_move: child.first_move,
            });
            board.undo_move();

            match child.outcome {
                Q1DefenseOutcome::Proved => {}
                Q1DefenseOutcome::Deadline => {
                    attempts.push(attempt);
                    board.undo_move();
                    stop_reason = Q1TssStopReason::Deadline;
                    break 'candidate;
                }
                Q1DefenseOutcome::NodeBudget => {
                    attempts.push(attempt);
                    board.undo_move();
                    stop_reason = Q1TssStopReason::NodeBudget;
                    break 'candidate;
                }
                Q1DefenseOutcome::Exhausted | Q1DefenseOutcome::ImmediateWin => {
                    attempts.push(attempt);
                    board.undo_move();
                    continue 'candidate;
                }
            }
        }

        attempt.complete = attempt.defenses.len() == attempt.defenses_total;
        debug_assert!(attempt.complete);
        selected_move = Some(candidate.mv);
        attempts.push(attempt);
        board.undo_move();
        stop_reason = Q1TssStopReason::Proved;
        break;
    }

    debug_assert_eq!(board.history.len(), history_len);
    debug_assert!(board.black == black);
    debug_assert!(board.white == white);
    debug_assert_eq!(board.side_to_move, side_to_move);
    debug_assert_eq!(board.zobrist, zobrist);
    debug_assert_eq!(board.last_move, last_move);

    Q1TssResult {
        selected_move,
        candidate_count,
        candidates_tested: attempts.len(),
        child_nodes,
        elapsed: started.elapsed(),
        stop_reason,
        attempts,
    }
}

/// Count disagreements between the directional semantic reference and the
/// production aggregate classifier for every empty cell and both sides.
pub fn directional_aggregation_mismatches(board: &Board, config: QuietThreatConfig) -> usize {
    [Stone::Black, Stone::White]
        .into_iter()
        .map(|side| {
            (0..NUM_CELLS)
                .filter(|&cell| board.is_empty(cell))
                .filter(|&cell| {
                    aggregate_directional(classify_directions(board, cell, side, config), config)
                        != classify_move_fast_with_flags(
                            board,
                            cell,
                            side,
                            config.enable_jump_three,
                            config.enable_gap_four,
                        )
                })
                .count()
        })
        .sum()
}

/// Return the production-equivalent aggregate kind and its four directional
/// components for one legal empty move. Intended for offline label audits.
pub fn classify_move_with_directions(
    board: &Board,
    mv: Move,
    side: Stone,
    config: QuietThreatConfig,
) -> (ThreatKind, [WindowThreat; 4]) {
    let directions = classify_directions(board, mv, side, config);
    let aggregate = aggregate_directional(directions, config);
    debug_assert_eq!(
        aggregate,
        classify_move_fast_with_flags(
            board,
            mv,
            side,
            config.enable_jump_three,
            config.enable_gap_four,
        )
    );
    (aggregate, directions)
}

fn classify_all(board: &Board, side: Stone, config: QuietThreatConfig) -> [ThreatKind; NUM_CELLS] {
    let mut kinds = [ThreatKind::None; NUM_CELLS];
    for (cell, kind) in kinds.iter_mut().enumerate() {
        if board.is_empty(cell) {
            *kind = classify_move_fast_with_flags(
                board,
                cell,
                side,
                config.enable_jump_three,
                config.enable_gap_four,
            );
        }
    }
    kinds
}

fn classify_all_directions(
    board: &Board,
    side: Stone,
    config: QuietThreatConfig,
) -> [[WindowThreat; 4]; NUM_CELLS] {
    let mut directions = [[WindowThreat::None; 4]; NUM_CELLS];
    for (cell, line_kinds) in directions.iter_mut().enumerate() {
        if board.is_empty(cell) {
            *line_kinds = classify_directions(board, cell, side, config);
        }
    }
    directions
}

fn classify_directions(
    board: &Board,
    source: Move,
    side: Stone,
    config: QuietThreatConfig,
) -> [WindowThreat; 4] {
    debug_assert!(board.is_empty(source));
    let (mine, opp) = if side == Stone::Black {
        (&board.black, &board.white)
    } else {
        (&board.white, &board.black)
    };
    let row = (source / BOARD_SIZE) as i32;
    let col = (source % BOARD_SIZE) as i32;
    std::array::from_fn(|dir_idx| {
        let (dr, dc) = DIR[dir_idx];
        let window = read_window(mine, opp, row, col, dr, dc);
        classify_window_after_play_with_flags(
            &window,
            board.effective_rule_set(),
            side,
            config.enable_jump_three,
            config.enable_gap_four,
        )
    })
}

fn aggregate_directional(lines: [WindowThreat; 4], config: QuietThreatConfig) -> ThreatKind {
    let fives = lines
        .iter()
        .filter(|&&kind| kind == WindowThreat::Five)
        .count();
    let open_fours = lines
        .iter()
        .filter(|&&kind| kind == WindowThreat::OpenFour)
        .count();
    let closed_fours = lines
        .iter()
        .filter(|&&kind| kind == WindowThreat::ClosedFour)
        .count();
    let open_threes = lines
        .iter()
        .filter(|&&kind| kind == WindowThreat::OpenThree)
        .count();
    let jump_threes = lines
        .iter()
        .filter(|&&kind| kind == WindowThreat::JumpThree)
        .count();
    let fours = open_fours + closed_fours;

    if fives >= 1 {
        ThreatKind::Five
    } else if open_fours >= 1 {
        ThreatKind::OpenFour
    } else if fours >= 2 {
        ThreatKind::DoubleFour
    } else if closed_fours >= 1 && open_threes >= 1 {
        ThreatKind::FourThree
    } else if open_threes >= 2 {
        ThreatKind::DoubleThree
    } else if closed_fours >= 1 {
        ThreatKind::ClosedFour
    } else if open_threes >= 1 {
        ThreatKind::OpenThree
    } else if config.enable_jump_three && jump_threes >= 1 {
        ThreatKind::JumpThree
    } else {
        ThreatKind::None
    }
}

fn collect_existing_lines(
    board: &Board,
    aggregate: &[ThreatKind; NUM_CELLS],
    directions: &[[WindowThreat; 4]; NUM_CELLS],
    original_mine: &BitBoard,
) -> Vec<ThreatLine> {
    let mut lines = Vec::new();
    for source in 0..NUM_CELLS {
        if !board.is_empty(source) || !aggregate[source].is_forcing() {
            continue;
        }
        for dir_idx in 0..DIR.len() {
            if direction_is_forcing(directions[source][dir_idx]) {
                lines.push(make_threat_line(source, dir_idx, original_mine));
            }
        }
    }
    lines
}

fn make_threat_line(source: Move, dir_idx: usize, original_mine: &BitBoard) -> ThreatLine {
    let row = (source / BOARD_SIZE) as i32;
    let col = (source % BOARD_SIZE) as i32;
    let (dr, dc) = DIR[dir_idx];
    let mut footprint = BitBoard::EMPTY;
    let mut support = BitBoard::EMPTY;
    for offset in -4i32..=4 {
        let r = row + dr * offset;
        let c = col + dc * offset;
        if r < 0 || c < 0 || r >= BOARD_SIZE as i32 || c >= BOARD_SIZE as i32 {
            continue;
        }
        let cell = r as usize * BOARD_SIZE + c as usize;
        footprint.set(cell);
        if original_mine.get(cell) {
            support.set(cell);
        }
    }
    ThreatLine { footprint, support }
}

fn direction_is_forcing(kind: WindowThreat) -> bool {
    matches!(
        kind,
        WindowThreat::OpenThree
            | WindowThreat::JumpThree
            | WindowThreat::ClosedFour
            | WindowThreat::OpenFour
            | WindowThreat::Five
    )
}

fn direction_strength(kind: WindowThreat) -> u8 {
    match kind {
        WindowThreat::OpenThree | WindowThreat::JumpThree => 1,
        WindowThreat::ClosedFour => 2,
        WindowThreat::OpenFour => 3,
        WindowThreat::Five => 4,
        _ => 0,
    }
}

fn bitboards_intersect(a: &BitBoard, b: &BitBoard) -> bool {
    (a.lo & b.lo) != 0 || (a.hi & b.hi) != 0
}

fn intersection_count(a: &BitBoard, b: &BitBoard) -> u32 {
    (a.lo & b.lo).count_ones() + (a.hi & b.hi).count_ones()
}

fn sort_dependency_candidates(candidates: &mut [DependencyQuietCandidate]) {
    candidates.sort_unstable_by(|a, b| {
        b.winning_gains
            .cmp(&a.winning_gains)
            .then_with(|| b.max_reused_support.cmp(&a.max_reused_support))
            .then_with(|| b.dependency_links.cmp(&a.dependency_links))
            .then_with(|| b.forcing_gains.cmp(&a.forcing_gains))
            .then_with(|| a.mv.cmp(&b.mv))
    });
}

fn has_immediate_five(board: &Board, side: Stone, config: QuietThreatConfig) -> bool {
    (0..NUM_CELLS).any(|cell| {
        board.is_empty(cell)
            && classify_move_fast_with_flags(
                board,
                cell,
                side,
                config.enable_jump_three,
                config.enable_gap_four,
            ) == ThreatKind::Five
    })
}

fn threat_strength(kind: ThreatKind) -> u8 {
    match kind {
        ThreatKind::None => 0,
        ThreatKind::OpenThree | ThreatKind::JumpThree => 1,
        ThreatKind::ClosedFour => 2,
        ThreatKind::DoubleThree => 3,
        ThreatKind::FourThree => 4,
        ThreatKind::OpenFour | ThreatKind::DoubleFour => 5,
        ThreatKind::Five => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::to_idx;

    #[test]
    fn quiet_pair_builder_is_generated_and_board_is_restored() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(0, 0));
        let before_history = board.history.clone();
        let before_black = board.black;
        let before_white = board.white;
        let before_zobrist = board.zobrist;

        let candidates = generate_quiet_threat_candidates(
            &mut board,
            QuietThreatConfig {
                min_gain: 1,
                ..QuietThreatConfig::default()
            },
        );

        assert!(candidates.iter().any(|c| c.mv == to_idx(7, 8)));
        assert_eq!(board.history, before_history);
        assert!(board.black == before_black);
        assert!(board.white == before_white);
        assert_eq!(board.zobrist, before_zobrist);
        assert_eq!(board.side_to_move, Stone::Black);
    }

    #[test]
    fn direct_forcing_move_is_not_a_quiet_candidate() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 8));
        board.make_move(to_idx(0, 1));

        let forcing = to_idx(7, 9);
        assert!(
            classify_move_fast_with_flags(&board, forcing, Stone::Black, true, true).is_forcing()
        );
        let candidates = generate_quiet_threat_candidates(&mut board, QuietThreatConfig::default());
        assert!(!candidates.iter().any(|c| c.mv == forcing));
    }

    #[test]
    fn candidate_does_not_leave_an_opponent_immediate_five() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(0, 0));
        board.make_move(to_idx(7, 8));
        board.make_move(to_idx(0, 1));
        board.make_move(to_idx(8, 7));
        board.make_move(to_idx(0, 2));
        board.make_move(to_idx(8, 8));
        board.make_move(to_idx(0, 3));

        let block = to_idx(0, 4);
        let candidates = generate_quiet_threat_candidates(&mut board, QuietThreatConfig::default());

        assert!(candidates.iter().all(|candidate| candidate.mv == block));
        assert_eq!(board.side_to_move, Stone::Black);
        assert_eq!(board.history.len(), 8);
    }

    #[test]
    fn generation_order_is_deterministic() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(6, 7));
        let a = generate_quiet_threat_candidates(&mut board, QuietThreatConfig::default());
        let b = generate_quiet_threat_candidates(&mut board, QuietThreatConfig::default());
        assert_eq!(a, b);
    }

    #[test]
    fn directional_aggregation_matches_production_classifier() {
        let mut board = Board::new();
        for mv in [
            to_idx(7, 7),
            to_idx(6, 7),
            to_idx(7, 8),
            to_idx(6, 8),
            to_idx(8, 8),
            to_idx(5, 9),
        ] {
            board.make_move(mv);
        }
        assert_eq!(
            directional_aggregation_mismatches(&board, QuietThreatConfig::default()),
            0
        );
    }

    #[test]
    fn dependency_d2_is_a_subset_of_d1_and_board_is_restored() {
        let mut board = Board::new();
        for mv in [
            to_idx(7, 7),
            to_idx(0, 0),
            to_idx(7, 8),
            to_idx(0, 1),
            to_idx(6, 7),
            to_idx(1, 0),
        ] {
            board.make_move(mv);
        }
        let before_history = board.history.clone();
        let before_black = board.black;
        let before_white = board.white;
        let before_zobrist = board.zobrist;
        let arms = generate_dependency_quiet_candidates(&mut board, QuietThreatConfig::default());

        assert!(arms.d2.iter().all(|candidate| {
            arms.d1
                .iter()
                .any(|d1_candidate| d1_candidate.mv == candidate.mv)
        }));
        assert_eq!(board.history, before_history);
        assert!(board.black == before_black);
        assert!(board.white == before_white);
        assert_eq!(board.zobrist, before_zobrist);
    }

    #[test]
    fn q1_full_defense_covers_every_legal_reply_and_restores_board() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(0, 0));
        let before = board.clone();
        let config = Q1TssConfig {
            max_candidates: 1,
            time_budget: None,
            node_budget: None,
            ..Q1TssConfig::default()
        };

        let result =
            search_q1_tss_root_with(&mut board, &config, |_board, _config, _time, _nodes| {
                Q1ChildResult {
                    outcome: Q1DefenseOutcome::Proved,
                    nodes: 1,
                    first_move: Some(to_idx(7, 8)),
                }
            });

        assert_eq!(result.stop_reason, Q1TssStopReason::Proved);
        assert!(result.selected_move.is_some());
        assert_eq!(result.attempts.len(), 1);
        let attempt = &result.attempts[0];
        assert!(attempt.complete);
        assert_eq!(attempt.defenses_total, NUM_CELLS - before.move_count - 1);
        assert_eq!(attempt.defenses.len(), attempt.defenses_total);
        assert!(
            attempt
                .defenses
                .iter()
                .all(|defense| defense.outcome == Q1DefenseOutcome::Proved)
        );
        assert_eq!(board.history, before.history);
        assert!(board.black == before.black);
        assert!(board.white == before.white);
        assert_eq!(board.side_to_move, before.side_to_move);
        assert_eq!(board.zobrist, before.zobrist);
    }

    #[test]
    fn q1_refuted_candidates_do_not_leak_board_state() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(0, 0));
        let before = board.clone();
        let config = Q1TssConfig {
            max_candidates: 2,
            time_budget: None,
            node_budget: None,
            ..Q1TssConfig::default()
        };

        let result =
            search_q1_tss_root_with(&mut board, &config, |_board, _config, _time, _nodes| {
                Q1ChildResult {
                    outcome: Q1DefenseOutcome::Exhausted,
                    nodes: 1,
                    first_move: None,
                }
            });

        assert_eq!(result.stop_reason, Q1TssStopReason::Exhausted);
        assert!(result.selected_move.is_none());
        assert_eq!(result.attempts.len(), 2);
        assert!(result.attempts.iter().all(|attempt| {
            !attempt.complete
                && attempt.defenses.len() == 1
                && attempt.defenses[0].outcome == Q1DefenseOutcome::Exhausted
        }));
        assert_eq!(board.history, before.history);
        assert!(board.black == before.black);
        assert!(board.white == before.white);
        assert_eq!(board.side_to_move, before.side_to_move);
        assert_eq!(board.zobrist, before.zobrist);
    }

    #[test]
    fn response_relevance_includes_global_counters_and_restores_board() {
        let mut board = Board::new();
        board.make_move(to_idx(7, 7));
        board.make_move(to_idx(0, 0));
        let before = board.clone();

        let audit =
            audit_quiet_response_relevance(&mut board, to_idx(7, 8), QuietThreatConfig::default())
                .unwrap();

        assert!(audit.forcing_gains > 0);
        assert!(!audit.gained_sources.is_empty());
        assert!(!audit.f1_replies.is_empty());
        assert!(!audit.f2_replies.is_empty());
        assert!(
            audit.immediate_replies.iter().all(|reply| {
                audit.f1_replies.contains(reply) && audit.f2_replies.contains(reply)
            })
        );
        assert!(
            audit.defender_forcing_replies.iter().all(|reply| {
                audit.f1_replies.contains(reply) && audit.f2_replies.contains(reply)
            })
        );
        assert_eq!(board.history, before.history);
        assert!(board.black == before.black);
        assert!(board.white == before.white);
        assert_eq!(board.side_to_move, before.side_to_move);
        assert_eq!(board.zobrist, before.zobrist);
    }
}
