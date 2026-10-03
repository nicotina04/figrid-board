//! Reads boards (one per line: 225 chars of '.', 'X' = black, 'O' = white, row-major) and prints, per board, 225
//! chars: '!' where an empty cell is forbidden for black, '.' otherwise (occupied cells print as-is).
use figrid_board::board::{BitBoard, NUM_CELLS};
use figrid_board::renju::is_forbidden;
use std::io::{self, BufRead, Write};

fn main() {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let cells: Vec<char> = line.trim().chars().collect();
        assert_eq!(cells.len(), NUM_CELLS);
        let (mut b, mut w) = (BitBoard::EMPTY, BitBoard::EMPTY);
        for (i, ch) in cells.iter().enumerate() {
            match ch {
                'X' => b.set(i),
                'O' => w.set(i),
                _ => {}
            }
        }
        let s: String = (0..NUM_CELLS)
            .map(|i| match cells[i] {
                '.' => if is_forbidden(&b, &w, i) { '!' } else { '.' },
                c => c,
            })
            .collect();
        writeln!(out, "{s}").unwrap();
    }
}
