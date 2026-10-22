use figrid_board::bitboard::{BitBoard, BitBoard15, BitBoard20};
use figrid_board::position::{Board, BoardGeometry};
use figrid_board::{Board15, Board20, Searcher15, Searcher20};

fn exercise_bits<B: BitBoard>() {
    let cells = B::SIZE * B::SIZE;
    let mut bits = B::EMPTY;
    for cell in 0..cells {
        assert!(!bits.get(cell));
        bits.set(cell);
    }
    assert_eq!(bits.count_ones(), cells as u32);
    assert_eq!(
        bits.iter_ones().collect::<Vec<_>>(),
        (0..cells).collect::<Vec<_>>()
    );
    let mut odd = B::EMPTY;
    for cell in (1..cells).step_by(2) {
        odd.set(cell);
    }
    assert_eq!(
        bits.and(&odd).iter_ones().collect::<Vec<_>>(),
        (1..cells).step_by(2).collect::<Vec<_>>()
    );
    let even = bits.and_not(&odd);
    assert!(!even.intersects(&odd));
    assert!(even.or(&odd) == bits);
    for cell in (0..cells).rev() {
        bits.clear(cell);
        assert!(!bits.get(cell));
    }
    assert!(bits.is_zero());
    assert!(bits.words().as_ref().iter().all(|&w| w == 0));
}

#[test]
fn both_bitboards_obey_the_same_generic_contract() {
    exercise_bits::<BitBoard15>();
    exercise_bits::<BitBoard20>();
    assert_eq!(std::mem::size_of::<BitBoard15>(), 32);
    assert_eq!(std::mem::size_of::<BitBoard20>(), 64);
}

struct PositionPair<B: BoardGeometry> {
    first: Board<B>,
    second: Board<B>,
}

#[test]
fn generic_positions_of_both_sizes_coexist_and_undo_exactly() {
    let mut small = PositionPair {
        first: Board::<BitBoard15>::new(),
        second: Board15::new(),
    };
    let mut large = PositionPair {
        first: Board::<BitBoard20>::new(),
        second: Board20::new(),
    };
    let small_hash = small.first.zobrist;
    let large_hash = large.first.zobrist;
    small.first.make_move(224);
    large.first.make_move(399);
    assert!(small.first.black.get(224));
    assert!(large.first.black.get(399));
    assert_eq!(small.second.move_count, 0);
    assert_eq!(large.second.move_count, 0);
    assert_eq!(small.first.line_pattern_ids.len(), 225);
    assert_eq!(large.first.line_pattern_ids.len(), 400);
    small.first.undo_move();
    large.first.undo_move();
    assert_eq!(small.first.zobrist, small_hash);
    assert_eq!(large.first.zobrist, large_hash);
    assert!(small.first.black.is_zero());
    assert!(large.first.black.is_zero());
    small.second.make_move(112);
    large.second.make_move(210);
    assert_eq!(small.first.move_count, 0);
    assert_eq!(large.first.move_count, 0);
    let _ = (Searcher15::new(), Searcher20::new());
}
