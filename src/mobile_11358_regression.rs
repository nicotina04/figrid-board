use super::*;

fn to_idx(row: usize, col: usize) -> Move {
    row * BOARD_SIZE + col
}

struct ConstantEval;

impl SearchEvalState for ConstantEval {
    fn push_move(&mut self, _: &Board, _: Move, _: bool) -> EvalStateStepProfile {
        EvalStateStepProfile::default()
    }
    fn pop_move(&mut self, _: bool) -> EvalStateStepProfile {
        EvalStateStepProfile::default()
    }
    fn eval(&mut self, board: &Board, _: bool) -> (i32, EvalStateStepProfile) {
        (if board.side_to_move == Stone::White { 100 } else { -10 }, EvalStateStepProfile::default())
    }
}

fn screenshot_board(plies: usize, rule: RuleSet, transform: usize) -> Board {
    let moves = "H3 H13 G8 H11 E6 H12 H10 I11 F7 D5 F5 D7 F6 H15 H14 F8 G6 D6 G4 I2 F3 F4 I6 H6 H5 E2 J7";
    let mut board = Board::new();
    board.set_rule_set(rule);
    for text in moves.split_whitespace().take(plies) {
        let col = (text.as_bytes()[0] - b'A') as usize;
        let row = 15 - text[1..].parse::<usize>().unwrap();
        let (mut row, mut col) = (row, col);
        if transform >= 4 { col = BOARD_SIZE - 1 - col; }
        for _ in 0..transform % 4 {
            (row, col) = (col, BOARD_SIZE - 1 - row);
        }
        let mv = to_idx(row, col);
        assert!(board.is_legal_move(mv), "{text}");
        board.make_move(mv);
    }
    board
}

#[test]
fn screenshot_11358_forced_loss_cannot_use_stand_pat() {
    let ordering = NnueWeights::zeros(crate::GOMOKU_NNUE_CONFIG);
    for rule in [RuleSet::Freestyle, RuleSet::Standard] {
      for transform in 0..8 {
        for (plies, distance) in [(23, 4), (25, 2)] {
            let mut board = screenshot_board(plies, rule, transform);
            let original = board.clone();
            let mut searcher = Searcher::new();
            let score = searcher.qsearch(&mut board, &ordering, &mut ConstantEval, 0, 0, -INF, INF);
            assert_eq!(score, -(WIN_SCORE - distance), "{rule:?}, D4={transform}, after ply {plies}");
            assert_eq!(board.history, original.history);
            assert_eq!(board.zobrist, original.zobrist);
            assert!(board.black == original.black && board.white == original.white);
        }
      }
    }
}

#[test]
fn forced_block_precedes_beta_cutoff_and_quiescence_limit() {
    let ordering = NnueWeights::zeros(crate::GOMOKU_NNUE_CONFIG);
    let mut board = Board::new();
    for (r, c) in [(7, 3), (7, 2), (7, 4), (0, 0), (7, 5), (0, 14), (7, 6)] {
        board.make_move(to_idx(r, c));
    }
    for (qply, beta) in [(0, 0), (0, INF), (QSEARCH_MAX_PLY, INF)] {
        let mut searcher = Searcher::new();
        let score = searcher.qsearch(&mut board, &ordering, &mut ConstantEval, qply, 0, -INF, beta);
        assert_eq!(score, 10, "qply={qply}, beta={beta}: the sole block must be played");
        assert_eq!(board.history.len(), 7);
    }
}

#[test]
fn immediate_win_precedes_opponent_fives_and_quiescence_limit() {
    let ordering = NnueWeights::zeros(crate::GOMOKU_NNUE_CONFIG);
    let mut board = Board::new();
    for (r, c) in [(7, 3), (5, 3), (7, 4), (5, 4), (7, 5), (5, 5), (7, 6), (5, 6)] {
        board.make_move(to_idx(r, c));
    }
    let mut searcher = Searcher::new();
    let score = searcher.qsearch(&mut board, &ordering, &mut ConstantEval, QSEARCH_MAX_PLY, 0, -INF, -200);
    assert_eq!(score, WIN_SCORE - 1);
}

#[test]
fn exact_five_does_not_promote_an_overline_to_a_win() {
    let ordering = NnueWeights::zeros(crate::GOMOKU_NNUE_CONFIG);
    for rule in [RuleSet::Freestyle, RuleSet::Standard] {
        let mut board = Board::new();
        board.set_rule_set(rule);
        for (r, c) in [(7, 3), (0, 0), (7, 4), (0, 3), (7, 5), (0, 6), (7, 7), (0, 9), (7, 8), (0, 12)] {
            board.make_move(to_idx(r, c));
        }
        // Completing the internal gap makes six. The two external cells do
        // not complete five because the existing internal gap remains.
        assert_eq!(classify_move_fast(&board, to_idx(7, 6), Stone::Black) == ThreatKind::Five,
                   rule == RuleSet::Freestyle);
        let mut searcher = Searcher::new();
        let score = searcher.qsearch(&mut board, &ordering, &mut ConstantEval, QSEARCH_MAX_PLY, 0, -INF, INF);
        assert_eq!(score, if rule == RuleSet::Freestyle { WIN_SCORE - 1 } else { -10 });
    }
}
