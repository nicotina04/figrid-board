//! Gomocup (Piskvork) protocol adapter for the noru-tactic NNUE engine.
//!
//! Ships as the `pbrain-figrid` binary (the original author's 0.3.1 Gomocup
//! submission was withdrawn on 2026-04-21). The pre-NNUE engine still ships
//! as `pbrain-figrid-legacy` for continuity with the 0.3.x series.

use std::io::{self, BufRead, Write};
use std::time::Duration;

use figrid_board::{to_idx, Board, Searcher, BOARD_SIZE, GOMOKU_NNUE_CONFIG};
use noru::network::NnueWeights;

/// Source the v52 NNUE weights. Two modes:
///
/// 1. `embed-weights` cargo feature (Gomocup submission build) — gzip-compressed
///    weights are baked into the binary at compile time and decompressed once
///    on startup. Yields a single self-contained executable.
///
/// 2. Default (crates.io publish, dev builds) — weights are read from disk at
///    startup. Path resolution order:
///       a. `$FIGRID_WEIGHTS` env var if set
///       b. `./models/gomoku_v52_5stone_conv_93k.bin` relative to cwd
///       c. error out with a hint
#[cfg(feature = "embed-weights")]
fn load_weights_bytes() -> Result<Vec<u8>, String> {
    use flate2::read::GzDecoder;
    use std::io::Read;
    const COMPRESSED: &[u8] =
        include_bytes!("../models/gomoku_v52_5stone_conv_93k.bin.gz");
    let mut decoder = GzDecoder::new(COMPRESSED);
    let mut out = Vec::with_capacity(15_000_000);
    decoder
        .read_to_end(&mut out)
        .map_err(|e| format!("failed to decompress embedded weights: {e}"))?;
    Ok(out)
}

#[cfg(not(feature = "embed-weights"))]
fn load_weights_bytes() -> Result<Vec<u8>, String> {
    let path = std::env::var("FIGRID_WEIGHTS")
        .unwrap_or_else(|_| "models/gomoku_v52_5stone_conv_93k.bin".into());
    std::fs::read(&path).map_err(|e| {
        format!(
            "failed to read weights from `{path}`: {e}\n\
             hint: set $FIGRID_WEIGHTS or place the file at ./models/, \
             or rebuild with `--features embed-weights` for a self-contained binary"
        )
    })
}

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
        // Freestyle (all bits 0) and Standard (exact5 only) are supported.
        // Renju, Caro, and the obscure "continuous" overline-allowed rule
        // require dedicated training / search logic — left for future work.
        !(self.rule_continuous || self.rule_renju || self.rule_caro)
    }

    fn turn_budget(&self, move_count: usize) -> Duration {
        // No match budget announced (Piskvork running with `time_match=0`,
        // arena scripts that set only `timeout_turn`, etc.) — fall back to
        // the per-move cap. Anything below half of the sentinel default
        // means the controller actually told us a real budget.
        if self.timeout_match >= DEFAULT_MATCH_MS / 2 {
            return Duration::from_millis(
                (self.timeout_turn - SAFETY_MARGIN_MS).max(50) as u64,
            );
        }

        let time_left = self.time_left.max(0);
        if time_left <= 0 {
            // Out of time — respond instantly. Loses on time eventually but
            // never overshoots the controller's deadline.
            return Duration::from_millis(50);
        }

        // Real Gomocup games end well before the 225-cell board fills up;
        // 35 moves per side is a calibration-friendly midpoint between the
        // shortest decisive games (~25) and long late-mate fights (~50).
        // Old code divided the *whole match budget* by `15*15/2 = 112`, an
        // estimate that left ~70% of the time unused at game end.
        const EXPECTED_PER_SIDE: i64 = 35;
        let played_this_side = (move_count as i64 + 1) / 2;
        let remaining_this_side = (EXPECTED_PER_SIDE - played_this_side).max(5);
        let equal_share = time_left / remaining_this_side;

        // Phase-based multiplier. Spending more time in the tactical
        // midgame and less in the random opening / forced endgame matches
        // standard chess-engine practice and the diagnosis of figrid
        // losing in plies 8-25 (Phase A.1 white-loss analysis).
        // Multipliers stored in basis points / 100 to keep the math in i64.
        let phase_mul: i64 = match move_count {
            0..=5 => 30,    // opening — most engines waste time here
            6..=11 => 80,   // early — getting into tactics
            12..=24 => 150, // tactical peak — boost
            25..=34 => 100, // late midgame — equal share
            _ => 60,        // endgame — often forced
        };
        let phase_budget = (equal_share * phase_mul) / 100;

        // Hard caps:
        //   * `timeout_turn` is the controller-imposed per-move ceiling.
        //   * `time_left / 3` keeps a single move from blowing the budget;
        //     even the deepest tactical search rarely needs more than a
        //     third of remaining time.
        //   * 100 ms floor so abort logic still has a chance to fire.
        let safe_max = time_left / 3;
        let budget = phase_budget
            .min(self.timeout_turn)
            .min(safe_max)
            .max(100);

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
        let bytes = load_weights_bytes()?;
        let weights = NnueWeights::load_from_bytes(&bytes, Some(GOMOKU_NNUE_CONFIG))
            .map_err(|e| format!("failed to parse weights: {e}"))?;
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
                // Standard rule (rule=1): exactly-5 wins. Tell the board so
                // its check_win drops overlines from the win-set.
                engine.board.exact5 = engine.info.rule_exact5;
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
                    "name=\"figrid\", version=\"0.6.9\", author=\"nicotina04 (successor to wuwbobo2021)\", country=\"KR\""
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
