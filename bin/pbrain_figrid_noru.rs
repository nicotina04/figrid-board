//! Piskvork protocol adapter selecting a statically specialized engine at START.
use figrid_board::game::RuleSet;
use std::io::{self, BufRead, Write};
use std::time::Duration;
mod environment;
mod pbrain15;
mod pbrain20;
const DEFAULT_TIMEOUT_MS: i64 = 30_000;
const DEFAULT_MATCH_MS: i64 = 1_000_000_000;
const SAFETY_MARGIN_MS: i64 = 150;
#[derive(Clone)]
struct ProtocolInfo {
    timeout_turn: i64,
    timeout_match: i64,
    time_left: i64,
    rule_exact5: bool,
    rule_continuous: bool,
    rule_renju: bool,
    rule_caro: bool,
    /// figrid extension, not a Gomocup rule: bit 16 selects the Korean omok rule (`INFO rule 17`).
    rule_omok: bool,
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
            rule_omok: false,
        }
    }

    fn rule_set(&self) -> Option<RuleSet> {
        // Supported now: Freestyle (0), Standard exact-5 (1), Renju (4: black
        // exact five with forbidden double-four / double-three / overline,
        // white five or more) and Caro as Gomocup plays it (9 = caro|exact5:
        // exactly five, not blocked at both ends by stones), plus the figrid
        // extension rule 17 (Korean omok rule). Bare rule 8 (overline-wins
        // Caro) and continuous games are not modeled.
        if self.rule_continuous || (self.rule_renju && self.rule_caro) {
            return None;
        }
        // figrid extension: Korean omok rule (exactly five for both sides, double-three forbidden for both).
        if self.rule_omok {
            return if self.rule_exact5 && !self.rule_renju && !self.rule_caro {
                Some(RuleSet::Omok)
            } else {
                None
            };
        }
        if self.rule_renju {
            return Some(RuleSet::Renju);
        }
        if self.rule_caro {
            if self.rule_exact5 {
                Some(RuleSet::Caro)
            } else {
                None
            }
        } else if self.rule_exact5 {
            Some(RuleSet::Standard)
        } else {
            Some(RuleSet::Freestyle)
        }
    }

    fn rule_supported(&self) -> bool {
        self.rule_set().is_some()
    }

    fn turn_budget(&self, move_count: usize) -> Duration {
        // No match budget announced (Piskvork running with `time_match=0`,
        // arena scripts that set only `timeout_turn`, etc.) — fall back to
        // the per-move cap. Anything below half of the sentinel default
        // means the controller actually told us a real budget.
        if self.timeout_match >= DEFAULT_MATCH_MS / 2 {
            return Duration::from_millis((self.timeout_turn - SAFETY_MARGIN_MS).max(50) as u64);
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
        let budget = phase_budget.min(self.timeout_turn).min(safe_max).max(100);

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
                    self.rule_omok = (b & 16) != 0;
                }
            }
            _ => {}
        }
    }
}

enum RuntimeEngine {
    Small(Box<pbrain15::Engine>),
    Large(Box<pbrain20::Engine>),
}

macro_rules! dispatch {
    ($this:expr, $e:ident, $body:expr) => {
        match $this {
            RuntimeEngine::Small($e) => $body,
            RuntimeEngine::Large($e) => $body,
        }
    };
}

impl RuntimeEngine {
    fn new(size: usize, info: &ProtocolInfo) -> Result<Self, String> {
        let mut engine = match size {
            15 => Self::Small(Box::new(pbrain15::Engine::new()?)),
            20 => {
                #[cfg(not(feature = "codebook-eval"))]
                return Err("20x20 requires the codebook-eval feature".into());
                #[cfg(feature = "codebook-eval")]
                Self::Large(Box::new(pbrain20::Engine::new()?))
            }
            _ => return Err(format!("unsupported board size ({size})")),
        };
        engine.update_info(info);
        engine.reset_board(info.rule_set().unwrap_or(RuleSet::Freestyle));
        Ok(engine)
    }
    fn size(&self) -> usize {
        match self {
            Self::Small(_) => 15,
            Self::Large(_) => 20,
        }
    }
    fn update_info(&mut self, info: &ProtocolInfo) {
        dispatch!(self, e, {
            e.info = info.clone();
        });
    }
    fn reset_board(&mut self, rule: RuleSet) {
        dispatch!(self, e, {
            e.reset_board();
            e.board.set_rule_set(rule);
            e.started = true;
        });
    }
    fn choose_move(&mut self) -> Option<(u8, u8)> {
        dispatch!(self, e, e.choose_move())
    }
    fn apply_opp_move(&mut self, x: u8, y: u8) -> Result<(), String> {
        dispatch!(self, e, e.apply_opp_move(x, y))
    }
    fn play_at(&mut self, x: u8, y: u8) -> Result<(), String> {
        let size = self.size();
        if x as usize >= size || y as usize >= size {
            return Err("invalid BOARD state".into());
        }
        let idx = y as usize * size + x as usize;
        dispatch!(self, e, {
            if !e.board.is_empty(idx) {
                Err("invalid BOARD state".into())
            } else {
                e.board.make_move(idx);
                Ok(())
            }
        })
    }
    fn config_line(&self) -> &str {
        dispatch!(self, e, e.config_line.as_str())
    }
    fn no_move_error(&self) -> String {
        dispatch!(self, e, e.no_move_error().to_string())
    }
}

fn main() {
    let unknown = environment::unknown_engine_vars();
    if !unknown.is_empty() {
        for name in unknown {
            println!("ERROR unknown engine variable {name}");
        }
        io::stdout().flush().ok();
        std::process::exit(2);
    }
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut stdout = io::stdout().lock();
    macro_rules! reply { ($($arg:tt)*) => {{ writeln!(stdout,$($arg)*).ok(); stdout.flush().ok(); }}; }
    let mut info = ProtocolInfo::new();
    let mut engine: Option<RuntimeEngine> = None;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let mut fields = line.split_whitespace();
        let Some(command) = fields.next() else {
            continue;
        };
        match command.to_ascii_uppercase().as_str() {
            "START" => {
                let Some(raw) = fields.next() else {
                    reply!("ERROR - missing board size");
                    continue;
                };
                let Ok(size) = raw.parse::<usize>() else {
                    reply!("ERROR - cannot parse board size");
                    continue;
                };
                if !matches!(size, 15 | 20) {
                    reply!("ERROR - unsupported board size ({size})");
                    continue;
                }
                if !info.rule_supported() {
                    reply!("ERROR - unsupported rule");
                    continue;
                }
                if engine.as_ref().is_some_and(|e| e.size() == size) {
                    engine
                        .as_mut()
                        .unwrap()
                        .reset_board(info.rule_set().unwrap());
                } else {
                    match RuntimeEngine::new(size, &info) {
                        Ok(e) => engine = Some(e),
                        Err(err) => {
                            reply!("ERROR - {err}");
                            std::process::exit(1);
                        }
                    }
                }
                reply!("OK");
                reply!(
                    "MESSAGE config: board_size={size} {}",
                    engine.as_ref().unwrap().config_line()
                );
            }
            "INFO" => {
                let (Some(key), Some(value)) = (fields.next(), fields.next()) else {
                    continue;
                };
                info.update(key, value);
                if let Some(e) = engine.as_mut() {
                    e.update_info(&info);
                }
                if key == "rule" && !info.rule_supported() {
                    reply!("ERROR - unsupported rule {value}");
                }
            }
            "BEGIN" | "TURN" => {
                let Some(e) = engine.as_mut() else {
                    reply!("ERROR - engine not started");
                    continue;
                };
                if command.eq_ignore_ascii_case("TURN") {
                    let Some(raw) = fields.next() else {
                        reply!("ERROR - missing coord");
                        continue;
                    };
                    let Some((x, y)) = parse_coord(raw) else {
                        reply!("ERROR - bad coord");
                        continue;
                    };
                    if let Err(err) = e.apply_opp_move(x, y) {
                        reply!("ERROR - {err}");
                        continue;
                    }
                }
                if let Some((x, y)) = e.choose_move() {
                    reply!("{x},{y}");
                } else {
                    reply!("{}", e.no_move_error());
                }
            }
            "BOARD" => {
                let Some(e) = engine.as_mut() else {
                    reply!("ERROR - engine not started");
                    continue;
                };
                let mut own = Vec::new();
                let mut opp = Vec::new();
                let mut valid = true;
                loop {
                    line.clear();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        return;
                    }
                    if line.trim().eq_ignore_ascii_case("DONE") {
                        break;
                    }
                    let vals: Vec<_> = line.trim().split(',').collect();
                    if vals.len() != 3 {
                        valid = false;
                        continue;
                    }
                    match (
                        vals[0].trim().parse::<u8>(),
                        vals[1].trim().parse::<u8>(),
                        vals[2].trim().parse::<u8>(),
                    ) {
                        (Ok(x), Ok(y), Ok(1)) => own.push((x, y)),
                        (Ok(x), Ok(y), Ok(2)) => opp.push((x, y)),
                        (Ok(_), Ok(_), Ok(3)) => {}
                        _ => valid = false,
                    }
                }
                let (black, white) = if (own.len() + opp.len()) % 2 == 1 {
                    (&opp, &own)
                } else {
                    (&own, &opp)
                };
                if black.len() != white.len() && black.len() != white.len() + 1 {
                    valid = false;
                }
                e.reset_board(info.rule_set().unwrap_or(RuleSet::Freestyle));
                if valid {
                    for i in 0..black.len() {
                        let (x, y) = black[i];
                        if e.play_at(x, y).is_err() {
                            valid = false;
                            break;
                        }
                        if let Some(&(x, y)) = white.get(i) {
                            if e.play_at(x, y).is_err() {
                                valid = false;
                                break;
                            }
                        }
                    }
                }
                if !valid {
                    e.reset_board(info.rule_set().unwrap_or(RuleSet::Freestyle));
                    reply!("ERROR - invalid BOARD state");
                    continue;
                }
                if let Some((x, y)) = e.choose_move() {
                    reply!("{x},{y}");
                } else {
                    reply!("{}", e.no_move_error());
                }
            }
            "RESTART" => {
                let Some(e) = engine.as_mut() else {
                    reply!("ERROR - engine not started");
                    continue;
                };
                e.reset_board(info.rule_set().unwrap_or(RuleSet::Freestyle));
                reply!("OK");
            }
            "ABOUT" => reply!(
                "name=\"figrid\", version=\"{}\", author=\"Hogyung Choi\", country=\"KR\"",
                env!("CARGO_PKG_VERSION")
            ),
            "END" => break,
            _ => reply!("UNKNOWN"),
        }
    }
}

fn parse_coord(raw: &str) -> Option<(u8, u8)> {
    let (x, y) = raw.split_once(',')?;
    Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
}
