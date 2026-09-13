//! Regression for the independently refuted White 167 / Black 106 VCT proof.
//!
//! These tests exercise only VCT geometry. They load no model and run no arena.
//! The selected-attack test has no clock deadline: reaching its fixed node cap
//! is a test failure, not evidence that the attack was refuted.

use super::*;
use crate::board::GameResult;

const ROOT_HISTORY: [Move; 27] = [
    112, 110, 140, 124, 152, 95, 64, 137, 83, 53, 105, 91, 126, 98, 154, 168, 151, 153, 82, 81,
    109, 123, 138, 169, 185, 139, 107,
];
const ATTACK: Move = 167;
const COUNTER: Move = 106;

/// All effective public configurations: a threat index is used only when fast
/// classification is enabled. Testing an index with `fast = false` would test
/// an unreachable combination of private-function arguments.
const CLASSIFY_PATHS: [(bool, bool); 3] = [(false, false), (true, false), (true, true)];

fn transform(cell: Move, symmetry: usize) -> Move {
    let (mut row, mut col) = (cell / BOARD_SIZE, cell % BOARD_SIZE);
    if symmetry >= 4 {
        col = BOARD_SIZE - 1 - col;
    }
    for _ in 0..symmetry % 4 {
        (row, col) = (col, BOARD_SIZE - 1 - row);
    }
    row * BOARD_SIZE + col
}

fn replay(history: &[Move]) -> Board {
    let mut board = Board::new();
    for &mv in history {
        assert_eq!(board.game_result(), GameResult::Ongoing);
        assert!(board.is_legal_move(mv), "illegal fixture move {mv}");
        board.make_move(mv);
    }
    assert_eq!(board.game_result(), GameResult::Ongoing);
    board
}

fn assert_board_restored(actual: &Board, expected: &Board) {
    assert_eq!(
        (actual.black.lo, actual.black.hi),
        (expected.black.lo, expected.black.hi)
    );
    assert_eq!(
        (actual.white.lo, actual.white.hi),
        (expected.white.lo, expected.white.hi)
    );
    assert_eq!(actual.side_to_move, expected.side_to_move);
    assert_eq!(actual.move_count, expected.move_count);
    assert_eq!(actual.last_move, expected.last_move);
    assert_eq!(actual.history, expected.history);
    assert_eq!(actual.zobrist, expected.zobrist);
    assert_eq!(actual.line_pattern_ids, expected.line_pattern_ids);
    assert_eq!(actual.rule_set, expected.rule_set);
    assert_eq!(actual.exact5, expected.exact5);
}

fn defenses_for_path(
    board: &Board,
    attack: Move,
    kind: ThreatKind,
    fast: bool,
    indexed: bool,
    use_reach: bool,
) -> Vec<Move> {
    assert!(!indexed || fast);
    let cfg = VctConfig {
        use_fast_classify: fast,
        use_threat_index: indexed,
        use_reach_mask: use_reach,
        ..VctConfig::default()
    };
    assert!(
        !cfg.enable_gap_four,
        "attacker gap-four vocabulary stays disabled"
    );
    let flags = cfg.jump_three_flags();
    let index = indexed.then(|| VctThreatIndex::new(board));
    let reach = use_reach.then(|| reach_mask_for_side(board, board.side_to_move));
    let mut scratch = VctScratch::default();
    let mut defenses = find_defenses_with_counters(
        board,
        attack,
        kind,
        flags,
        index.as_ref(),
        reach.as_ref(),
        None,
        &mut scratch,
        false,
    );
    for &mv in &defenses {
        assert!(board.is_legal_move(mv));
    }
    defenses.sort_unstable();
    assert!(defenses.windows(2).all(|pair| pair[0] != pair[1]));
    defenses
}

#[test]
fn recorded_remote_counter_is_present_in_all_paths_and_d4_images() {
    for symmetry in 0..8 {
        let history: Vec<_> = ROOT_HISTORY
            .iter()
            .map(|&mv| transform(mv, symmetry))
            .collect();
        let mut board = replay(&history);
        assert_eq!(board.side_to_move, Stone::White);
        let attack = transform(ATTACK, symmetry);
        let counter = transform(COUNTER, symmetry);
        let kind = classify_move_rules(
            &board.white,
            &board.black,
            attack,
            Stone::White,
            RuleSet::Freestyle,
        );
        assert!(kind.is_forcing());
        assert!(!is_vct_terminal_win(kind));
        board.make_move(attack);
        let saved = board.clone();
        assert_eq!(board.side_to_move, Stone::Black);
        assert!(!find_defenses(&board, attack, false, false).contains(&counter));
        assert!(reach_mask_for_side(&board, Stone::Black).get(counter));
        let mut reference = None;
        for (fast, indexed) in CLASSIFY_PATHS {
            for use_reach in [false, true] {
                let defenses = defenses_for_path(&board, attack, kind, fast, indexed, use_reach);
                assert!(
                    defenses.contains(&counter),
                    "missing recorded counter: symmetry={symmetry}, fast={fast}, index={indexed}, reach={use_reach}"
                );
                if let Some(expected) = &reference {
                    assert_eq!(
                        &defenses, expected,
                        "counter paths must agree for the recorded position"
                    );
                } else {
                    reference = Some(defenses);
                }
                assert_board_restored(&board, &saved);
            }
        }
    }
}

/// Construct a legal alternating history with three defender stones on the
/// central line and a remote last attack. White fixtures have one extra Black
/// filler, so both colors use ordinary make_move rather than editing bitboards.
fn broken_four_fixture(existing: &[Move; 3], defender: Stone, symmetry: usize) -> Board {
    let remote_attack = 0;
    let fillers = [14, 210, 224];
    let mut history = Vec::new();
    match defender {
        Stone::Black => {
            for index in 0..3 {
                history.push(existing[index]);
                history.push(if index == 2 {
                    remote_attack
                } else {
                    fillers[index]
                });
            }
        }
        Stone::White => {
            for index in 0..3 {
                history.push(fillers[index]);
                history.push(existing[index]);
            }
            history.push(remote_attack);
        }
    }
    let history: Vec<_> = history
        .into_iter()
        .map(|mv| transform(mv, symmetry))
        .collect();
    let board = replay(&history);
    assert_eq!(board.side_to_move, defender);
    board
}

#[test]
fn broken_four_counter_shapes_cover_rotations_and_both_colors() {
    // Segment row 7, columns 3..7. Tuple: three existing stones, new stone,
    // remaining gap. The completed five has no same-color extension outside.
    let cases = [
        ([110, 111, 112], 108, 109), // X.XXX
        ([108, 111, 112], 109, 110), // XX.XX
        ([108, 109, 112], 110, 111), // XXX.X
    ];
    for (existing, new_stone, gap) in cases {
        for defender in [Stone::Black, Stone::White] {
            for symmetry in 0..4 {
                let board = broken_four_fixture(&existing, defender, symmetry);
                let attack = board.last_move.unwrap();
                let counter = transform(new_stone, symmetry);
                let gap = transform(gap, symmetry);
                assert!(!find_defenses(&board, attack, false, false).contains(&counter));
                let (mine, opp) = bb_pair(&board, defender);
                let mut after_counter = *mine;
                after_counter.set(counter);
                // Direct line geometry, not the optional gap-four classifier,
                // establishes that this counter leaves a next-turn five.
                assert!(completes_five_at(
                    &after_counter,
                    opp,
                    gap,
                    defender,
                    RuleSet::Freestyle
                ));
                let mut reference = None;
                for (fast, indexed) in CLASSIFY_PATHS {
                    for use_reach in [false, true] {
                        let defenses = defenses_for_path(
                            &board,
                            attack,
                            ThreatKind::OpenThree,
                            fast,
                            indexed,
                            use_reach,
                        );
                        assert!(
                            defenses.contains(&counter),
                            "missing shape counter={counter}, defender={defender:?}, rotation={symmetry}, fast={fast}, index={indexed}, reach={use_reach}"
                        );
                        if let Some(expected) = &reference {
                            assert_eq!(
                                &defenses, expected,
                                "counter paths must agree for synthetic geometry"
                            );
                        } else {
                            reference = Some(defenses);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn selected_white_167_attack_is_refuted_without_deadline_or_node_stop() {
    let mut board = replay(&ROOT_HISTORY);
    let attacker = board.side_to_move;
    let kind = classify_move_fast(&board, ATTACK, attacker);
    assert!(kind.is_forcing());
    assert!(!is_vct_terminal_win(kind));
    board.make_move(ATTACK);
    let saved = board.clone();
    let cfg = VctConfig {
        max_depth: 14,
        time_budget: None,
        node_budget: Some(25_000),
        ..VctConfig::default()
    };
    let mut sequence = vec![ATTACK];
    let mut tt = TransTable::with_capacity(65_536);
    let mut stats = VctSearchStats::default();
    let mut scratch = VctScratch::default();
    let mut index = None;
    let proved = vct_and(
        &mut board,
        attacker,
        kind,
        13,
        0,
        None,
        cfg.node_budget,
        cfg.jump_three_flags(),
        &mut None,
        &mut index,
        &mut sequence,
        &mut tt,
        &mut stats,
        &mut scratch,
    );
    assert_board_restored(&board, &saved);
    assert!(stats.nodes > 0);
    assert!(
        !stats.hit_stop(),
        "a budget stop is inconclusive, stats={stats:?}"
    );
    assert!(
        !proved,
        "White 167 has the independently proved Black 106 counter"
    );
    assert_eq!(
        sequence,
        vec![ATTACK],
        "failed AND restores the caller's sequence checkpoint"
    );
}
