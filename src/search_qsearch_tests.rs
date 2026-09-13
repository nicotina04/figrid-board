use super::*;
use crate::features::GOMOKU_NNUE_CONFIG;

#[derive(Default)]
struct ConstantEval {
    value: i32,
    stack: Vec<Move>,
    searched: Vec<Move>,
    calls: usize,
}

impl SearchEvalState for ConstantEval {
    fn push_move(&mut self, _: &Board, mv: Move, _: bool) -> EvalStateStepProfile {
        self.stack.push(mv);
        self.searched.push(mv);
        EvalStateStepProfile::default()
    }

    fn pop_move(&mut self, _: bool) -> EvalStateStepProfile {
        self.stack.pop().expect("balanced eval state");
        EvalStateStepProfile::default()
    }

    fn eval(&mut self, _: &Board, _: bool) -> (i32, EvalStateStepProfile) {
        self.calls += 1;
        (self.value, EvalStateStepProfile::default())
    }
}

fn position(history: &[Move], rule: RuleSet) -> Board {
    let mut board = Board::new();
    board.set_rule_set(rule);
    for &mv in history {
        assert_eq!(board.game_result(), GameResult::Ongoing);
        board.make_move(mv);
    }
    assert_eq!(board.game_result(), GameResult::Ongoing);
    board
}

fn probe(
    board: &mut Board,
    qply: u32,
    alpha: i32,
    beta: i32,
    value: i32,
    node_limit: Option<u64>,
) -> (i32, ConstantEval, Searcher) {
    let original = board.clone();
    let weights = NnueWeights::zeros(GOMOKU_NNUE_CONFIG);
    let mut searcher = Searcher::new();
    searcher.set_node_limit(node_limit);
    searcher.reset_for_search(None);
    searcher.set_use_packed_line_windows(true);
    searcher.set_use_candidate_frontier(true);
    searcher.begin_board_search_state(board);
    searcher.enable_main_search_candidate_frontier(board);
    let mut inc = ConstantEval {
        value,
        ..Default::default()
    };
    let score = searcher.qsearch(board, &weights, &mut inc, qply, 5, alpha, beta);
    assert!(board.black == original.black && board.white == original.white);
    assert_eq!(board.history, original.history);
    assert_eq!(board.zobrist, original.zobrist);
    assert_eq!(board.side_to_move, original.side_to_move);
    assert!(inc.stack.is_empty());
    assert!(
        searcher
            .board_search_state
            .as_ref()
            .unwrap()
            .is_synchronized(board)
    );
    (score, inc, searcher)
}

#[test]
fn qsearch_rejects_stand_pat_for_exact_loss_before_cutoff_and_cap() {
    for rule in [RuleSet::Freestyle, RuleSet::Standard] {
        for qply in [0, QSEARCH_MAX_PLY, QSEARCH_MAX_PLY + 1] {
            for (alpha, beta) in [(-INF, INF), (-5, -4)] {
                let mut board = position(&[0, 106, 2, 107, 4, 108, 6, 109], rule);
                // Independently establish loss: every legal move leaves a win
                // for White on its next turn. No model label is involved.
                for mv in board.legal_moves() {
                    let mut child = board.clone();
                    child.make_move(mv);
                    assert_eq!(child.game_result(), GameResult::Ongoing);
                    assert!(
                        [105, 110]
                            .into_iter()
                            .filter(|&m| child.is_empty(m))
                            .any(|m| {
                                let mut reply = child.clone();
                                reply.make_move(m);
                                reply.game_result() == GameResult::WhiteWin
                            })
                    );
                }
                let (score, inc, _) = probe(&mut board, qply, alpha, beta, 1000, None);
                assert_eq!(score, -(WIN_SCORE - 7));
                assert_eq!(inc.calls, 0);
            }
        }
    }
}

#[test]
fn qsearch_replays_real_false_fail_high() {
    let mut board = position(
        &[
            112, 128, 82, 144, 129, 97, 114, 98, 158, 99, 100, 110, 96, 83, 115, 113,
        ],
        RuleSet::Freestyle,
    );
    let (score, _, _) = probe(&mut board, 0, -5, -4, -2, None);
    assert_eq!(score, -998993);
}

#[test]
fn qsearch_forced_block_replaces_static_floor_and_extends_past_cap() {
    for qply in [0, QSEARCH_MAX_PLY, QSEARCH_MAX_PLY + 1] {
        for (alpha, beta) in [(-INF, INF), (-5, -4)] {
            let mut board = position(&[30, 0, 32, 1, 34, 2, 36, 3], RuleSet::Freestyle);
            let (score, inc, _) = probe(&mut board, qply, alpha, beta, 1000, None);
            assert_eq!(score, -1000);
            assert_eq!(inc.searched, [4]);
            assert_eq!(inc.calls, 1, "evaluate the defended child only");
        }
    }
}

#[test]
fn qsearch_winning_counter_precedes_defense_at_cap() {
    let mut board = position(&[0, 15, 1, 16, 2, 17, 3, 18], RuleSet::Freestyle);
    let (score, inc, _) = probe(&mut board, QSEARCH_MAX_PLY, -INF, INF, -1000, None);
    assert_eq!(score, WIN_SCORE - 6);
    assert_eq!(inc.calls, 0);
}

#[test]
fn qsearch_caro_two_winning_cells_can_both_be_defended() {
    let mut board = position(&[106, 108, 113, 109, 0, 110, 2, 111], RuleSet::Caro);
    // White's C8 and H8 wins are both neutralized by closing either endpoint.
    for winning_cell in [107, 112] {
        let mut white = board.clone();
        white.make_move(224);
        white.make_move(winning_cell);
        assert_eq!(white.game_result(), GameResult::WhiteWin);
    }
    let (score, inc, _) = probe(&mut board, QSEARCH_MAX_PLY, -INF, INF, 1000, None);
    assert_eq!(
        score, -1000,
        "two Caro threats are not necessarily a forced loss"
    );
    let mut searched = inc.searched;
    searched.sort_unstable();
    assert_eq!(searched, [107, 112]);
}

#[test]
fn qsearch_caro_includes_defense_at_the_other_end() {
    let mut board = position(&[107, 108, 0, 109, 2, 110, 4, 111], RuleSet::Caro);
    let (score, inc, _) = probe(&mut board, QSEARCH_MAX_PLY, -INF, INF, 1000, None);
    assert_eq!(score, -1000);
    let mut searched = inc.searched;
    searched.sort_unstable();
    assert_eq!(
        searched,
        [112, 113],
        "113 closes the far end without occupying 112"
    );
}

#[test]
fn qsearch_forced_extension_obeys_budget_and_restores_state() {
    let mut board = position(&[30, 0, 32, 1, 34, 2, 36, 3], RuleSet::Freestyle);
    let (_, inc, searcher) = probe(&mut board, QSEARCH_MAX_PLY, -INF, INF, 1000, Some(2));
    assert_eq!(searcher.nodes, 2);
    assert!(searcher.aborted && searcher.node_limit_hit());
    assert_eq!(inc.searched, [4]);
    assert!(inc.stack.is_empty());
}

#[test]
fn qsearch_quiet_stand_pat_and_cap_are_preserved() {
    for (qply, alpha, beta) in [(QSEARCH_MAX_PLY, -INF, INF), (0, 0, 20)] {
        let mut board = position(&[112, 113], RuleSet::Freestyle);
        let (score, inc, _) = probe(&mut board, qply, alpha, beta, 37, None);
        assert_eq!(score, 37);
        assert_eq!(inc.calls, 1);
        assert!(inc.searched.is_empty());
    }
}

#[test]
fn qsearch_replays_all_twelve_recorded_false_fail_highs() {
    let cases: serde_json::Value = serde_json::from_str(include_str!(
        "../tests/fixtures/qsearch_false_fail_high.json"
    ))
    .unwrap();
    let rows = cases.as_array().unwrap();
    assert_eq!(rows.len(), 12);
    for row in rows {
        let history: Vec<Move> = row["history"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_u64().unwrap() as Move)
            .collect();
        let mut board = position(&history, RuleSet::Freestyle);
        let alpha = row["alpha"].as_i64().unwrap() as i32;
        let beta = row["beta"].as_i64().unwrap() as i32;
        let raw = row["raw"].as_i64().unwrap() as i32;
        assert!(raw >= beta, "original static cutoff");
        let (score, inc, _) = probe(
            &mut board,
            row["qply"].as_u64().unwrap() as u32,
            alpha,
            beta,
            raw,
            None,
        );
        assert_eq!(score, -(WIN_SCORE - 7), "case {}", row["key"]);
        assert!(score < beta);
        assert_eq!(inc.calls, 0);
    }
}
