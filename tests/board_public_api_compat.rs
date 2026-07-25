use figrid_board::{BitBoard, Board, NUM_CELLS, RuleSet, Stone, to_idx};

#[test]
fn board_v0_8_1_exhaustive_struct_literal_remains_valid() {
    let initial_pattern_ids = Board::new().line_pattern_ids;
    // Compatibility guard: v0.8.1 exposed exactly these ten public fields
    // without `#[non_exhaustive]`. Adding any field to `Board` makes this
    // downstream-style literal stop compiling and therefore requires a
    // semver-major release.
    let mut board = Board {
        black: BitBoard::EMPTY,
        white: BitBoard::EMPTY,
        side_to_move: Stone::Black,
        move_count: 0,
        last_move: None,
        history: Vec::with_capacity(NUM_CELLS),
        zobrist: 0,
        line_pattern_ids: initial_pattern_ids,
        rule_set: RuleSet::Freestyle,
        exact5: false,
    };
    let expected_pattern_ids = board.line_pattern_ids.clone();
    let mv = to_idx(7, 7);

    board.make_move(mv);
    assert!(board.black.get(mv));
    assert!(!board.white.get(mv));
    assert_eq!(board.side_to_move, Stone::White);
    assert_eq!(board.move_count, 1);
    assert_eq!(board.last_move, Some(mv));
    assert_eq!(board.history, vec![mv]);
    assert_ne!(board.zobrist, 0);

    board.undo_move();
    assert!(board.black == BitBoard::EMPTY);
    assert!(board.white == BitBoard::EMPTY);
    assert_eq!(board.side_to_move, Stone::Black);
    assert_eq!(board.move_count, 0);
    assert_eq!(board.last_move, None);
    assert!(board.history.is_empty());
    assert_eq!(board.zobrist, 0);
    assert_eq!(board.line_pattern_ids, expected_pattern_ids);
    assert_eq!(board.rule_set, RuleSet::Freestyle);
    assert!(!board.exact5);
}
