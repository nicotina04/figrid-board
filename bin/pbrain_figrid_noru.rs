//! Gomocup (Piskvork) protocol adapter for the noru-tactic NNUE engine.
//!
//! Ships as the `pbrain-figrid` binary (the original author's 0.3.1 Gomocup
//! submission was withdrawn on 2026-04-21). The pre-NNUE engine still ships
//! as `pbrain-figrid-legacy` for continuity with the 0.3.x series.

use std::io::{self, BufRead, Write};
use std::time::Duration;

use figrid_board::{to_idx, Board, Searcher, BOARD_SIZE, GOMOKU_NNUE_CONFIG};
use noru::network::NnueWeights;

const WEIGHTS_BYTES: &[u8] = include_bytes!("../models/gomoku_v14_broken_rapfi_wide.bin");

const MAX_DEPTH: u32 = 20;
const DEFAULT_TIMEOUT_MS: i64 = 30_000;
const DEFAULT_MATCH_MS: i64 = 1_000_000_000;
/// Headroom subtracted from the turn budget so the last node batch finishes
/// well before Piskvork's deadline. Without this, the 128-node deadline
/// check can overshoot by ~50 ms on NNUE-heavy positions.
const SAFETY_MARGIN_MS: i64 = 150;

struct ProtocolInfo {
    timeout_turn: i64,
    timeout_match: i64,
    time_left: i64,
    rule_exact5: bool,
    rule_continuous: bool,
    rule_renju: bool,
    rule_caro: bool,
}

impl ProtocolInfo {
    fn new() -> Self {
        Self {
            timeout_turn: DEFAULT_TIMEOUT_MS,
            timeout_match: DEFAULT_MATCH_MS,
            time_left: DEFAULT_MATCH_MS,
            rule_exact5: false,
            rule_continuous: false,
            rule_renju: false,
            rule_caro: false,
        }
    }

    fn rule_supported(&self) -> bool {
        !(self.rule_continuous || self.rule_renju || self.rule_caro || self.rule_exact5)
    }

    fn turn_budget(&self, move_count: usize) -> Duration {
        let sz = BOARD_SIZE as i64;
        let budget = if move_count <= 5 {
            self.timeout_turn
        } else {
            let per_match = self.timeout_match / (sz * sz / 2).max(1);
            let remaining_half = ((sz * sz - move_count as i64) / 2).max(1);
            let per_left = self.time_left.max(0) / remaining_half;
            self.timeout_turn.min(per_match).min(per_left)
        };
        Duration::from_millis((budget - SAFETY_MARGIN_MS).max(50) as u64)
    }

    fn update(&mut self, key: &str, val: &str) {
        let val = val.trim();
        match key {
            "timeout_turn" => {
                if let Ok(v) = val.parse() {
                    self.timeout_turn = v;
                }
            }
            "timeout_match" => {
                if let Ok(v) = val.parse() {
                    self.timeout_match = v;
                }
            }
            "time_left" => {
                if let Ok(v) = val.parse() {
                    self.time_left = v;
                }
            }
            "rule" => {
                if let Ok(b) = val.parse::<u8>() {
                    self.rule_exact5 = (b & 1) != 0;
                    self.rule_continuous = (b & 2) != 0;
                    self.rule_renju = (b & 4) != 0;
                    self.rule_caro = (b & 8) != 0;
                }
            }
            _ => {}
        }
    }
}

struct Engine {
    board: Board,
    weights: NnueWeights,
    searcher: Searcher,
    info: ProtocolInfo,
    started: bool,
}

impl Engine {
    fn new() -> Result<Self, String> {
        let weights = NnueWeights::load_from_bytes(WEIGHTS_BYTES, Some(GOMOKU_NNUE_CONFIG))
            .map_err(|e| format!("failed to load embedded weights: {e}"))?;
        Ok(Self {
            board: Board::new(),
            weights,
            searcher: Searcher::new(),
            info: ProtocolInfo::new(),
            started: false,
        })
    }

    fn reset_board(&mut self) {
        self.board = Board::new();
    }

    fn apply_opp_move(&mut self, x: u8, y: u8) -> Result<(), String> {
        let idx = xy_to_idx(x, y)?;
        if !self.board.is_empty(idx) {
            return Err(format!("cell ({x},{y}) already occupied"));
        }
        self.board.make_move(idx);
        Ok(())
    }

    fn choose_move(&mut self) -> Option<(u8, u8)> {
        let budget = self.info.turn_budget(self.board.move_count);
        let result = self
            .searcher
            .search(&mut self.board, &self.weights, MAX_DEPTH, Some(budget));
        let mv = result.best_move.or_else(|| {
            self.board.candidate_moves().first().copied()
        })?;
        if !self.board.is_empty(mv) {
            return None;
        }
        self.board.make_move(mv);
        Some(idx_to_xy(mv))
    }
}

fn xy_to_idx(x: u8, y: u8) -> Result<usize, String> {
    if (x as usize) >= BOARD_SIZE || (y as usize) >= BOARD_SIZE {
        return Err(format!("coord ({x},{y}) out of range"));
    }
    Ok(to_idx(y as usize, x as usize))
}

fn idx_to_xy(idx: usize) -> (u8, u8) {
    let row = idx / BOARD_SIZE;
    let col = idx % BOARD_SIZE;
    (col as u8, row as u8)
}

fn main() {
    let mut engine = match Engine::new() {
        Ok(e) => e,
        Err(e) => {
            println!("ERROR - {e}");
            std::process::exit(1);
        }
    };

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut reader = stdin.lock();
    let mut line = String::new();

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => std::process::exit(0),
            Ok(_) => {}
            Err(_) => std::process::exit(0),
        }

        let trimmed = line.trim_end_matches(['\n', '\r']);
        let mut split = trimmed.split_whitespace();
        let Some(raw_cmd) = split.next() else {
            continue;
        };
        let command = raw_cmd.to_uppercase();

        match command.as_str() {
            "START" => {
                let Some(sz_str) = split.next() else {
                    writeln!(stdout, "ERROR - missing board size").ok();
                    continue;
                };
                let Ok(sz) = sz_str.parse::<usize>() else {
                    writeln!(stdout, "ERROR - cannot parse board size").ok();
                    continue;
                };
                if sz != BOARD_SIZE {
                    writeln!(stdout, "ERROR - unsupported board size ({sz})").ok();
                    continue;
                }
                if !engine.info.rule_supported() {
                    writeln!(stdout, "ERROR - unsupported rule").ok();
                    continue;
                }
                engine.reset_board();
                engine.started = true;
                writeln!(stdout, "OK").ok();
            }
            "BEGIN" => {
                if !engine.started {
                    writeln!(stdout, "ERROR - engine not started").ok();
                    continue;
                }
                if let Some((x, y)) = engine.choose_move() {
                    writeln!(stdout, "{x},{y}").ok();
                } else {
                    writeln!(stdout, "ERROR - no legal move").ok();
                }
            }
            "TURN" => {
                if !engine.started {
                    writeln!(stdout, "ERROR - engine not started").ok();
                    continue;
                }
                let Some(payload) = split.next() else {
                    writeln!(stdout, "ERROR - missing coord").ok();
                    continue;
                };
                let mut parts = payload.split(',');
                let Some(x) = parts.next().and_then(|s| s.trim().parse::<u8>().ok()) else {
                    writeln!(stdout, "ERROR - bad coord").ok();
                    continue;
                };
                let Some(y) = parts.next().and_then(|s| s.trim().parse::<u8>().ok()) else {
                    writeln!(stdout, "ERROR - bad coord").ok();
                    continue;
                };
                if let Err(e) = engine.apply_opp_move(x, y) {
                    writeln!(stdout, "ERROR - {e}").ok();
                    continue;
                }
                if let Some((ox, oy)) = engine.choose_move() {
                    writeln!(stdout, "{ox},{oy}").ok();
                } else {
                    writeln!(stdout, "ERROR - no legal move").ok();
                }
            }
            "BOARD" => {
                if !engine.started {
                    writeln!(stdout, "ERROR - engine not started").ok();
                    continue;
                }
                let mut coords_own: Vec<(u8, u8)> = Vec::new();
                let mut coords_opp: Vec<(u8, u8)> = Vec::new();
                loop {
                    line.clear();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        std::process::exit(0);
                    }
                    let t = line.trim();
                    if t.to_uppercase() == "DONE" {
                        break;
                    }
                    let nums: Vec<u8> = t
                        .split(',')
                        .filter_map(|s| s.trim().parse::<u8>().ok())
                        .collect();
                    if nums.len() < 3 {
                        writeln!(stdout, "ERROR - expected input: 'x,y,player'").ok();
                        continue;
                    }
                    let (x, y, who) = (nums[0], nums[1], nums[2]);
                    match who {
                        1 => coords_own.push((x, y)),
                        2 => coords_opp.push((x, y)),
                        _ => {}
                    }
                }

                // own이 흑인지 백인지 결정.
                // 총 돌 수가 짝수면 다음은 흑 차례 → own=흑. 홀수면 own=백.
                let total = coords_own.len() + coords_opp.len();
                let own_is_white = total % 2 == 1;
                let (coords_b, coords_w) = if own_is_white {
                    (&coords_opp, &coords_own)
                } else {
                    (&coords_own, &coords_opp)
                };

                engine.reset_board();
                let zip_len = coords_b.len().min(coords_w.len());
                for i in 0..zip_len {
                    match xy_to_idx(coords_b[i].0, coords_b[i].1) {
                        Ok(idx) if engine.board.is_empty(idx) => engine.board.make_move(idx),
                        _ => {
                            writeln!(stdout, "ERROR - invalid BOARD state").ok();
                            engine.reset_board();
                            break;
                        }
                    }
                    match xy_to_idx(coords_w[i].0, coords_w[i].1) {
                        Ok(idx) if engine.board.is_empty(idx) => engine.board.make_move(idx),
                        _ => {
                            writeln!(stdout, "ERROR - invalid BOARD state").ok();
                            engine.reset_board();
                            break;
                        }
                    }
                }
                for i in zip_len..coords_b.len() {
                    let (bx, by) = coords_b[i];
                    match xy_to_idx(bx, by) {
                        Ok(idx) if engine.board.is_empty(idx) => engine.board.make_move(idx),
                        _ => {
                            writeln!(stdout, "ERROR - invalid BOARD state").ok();
                            engine.reset_board();
                            break;
                        }
                    }
                }

                if let Some((x, y)) = engine.choose_move() {
                    writeln!(stdout, "{x},{y}").ok();
                } else {
                    writeln!(stdout, "ERROR - no legal move").ok();
                }
            }
            "INFO" => {
                let Some(key) = split.next() else {
                    continue;
                };
                let Some(val) = split.next() else {
                    continue;
                };
                engine.info.update(key, val);
            }
            "END" => std::process::exit(0),
            "ABOUT" => {
                writeln!(
                    stdout,
                    "name=\"figrid\", version=\"0.6.1\", author=\"nicotina04 (successor to wuwbobo2021)\", country=\"KR\""
                )
                .ok();
            }
            _ => {
                writeln!(stdout, "UNKNOWN").ok();
            }
        }
        stdout.flush().ok();
    }
}
