use figrid_board::vct::{
    has_immediate_five_fast_for_audit, has_immediate_five_reference_for_audit,
};
use figrid_board::{BOARD_SIZE, Board, Move, Stone, to_idx, to_rc};
use serde_json::{Value, json};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

struct Args {
    positions_jsonl: Vec<PathBuf>,
    games_jsonl: Option<PathBuf>,
    out_json: PathBuf,
    out_violations_jsonl: PathBuf,
    max_transitions: usize,
    max_violations: usize,
}

#[derive(Default)]
struct Stats {
    position_records: usize,
    position_boards: usize,
    game_records: usize,
    game_boards: usize,
    boolean_tests: u64,
    mismatches: u64,
    stored_violations: usize,
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let mut stats = Stats::default();
    let mut violations = File::create(&args.out_violations_jsonl).map_err(|e| {
        format!(
            "failed to create {}: {e}",
            args.out_violations_jsonl.display()
        )
    })?;

    for path in &args.positions_jsonl {
        sweep_position_jsonl(path, &args, &mut stats, &mut violations)?;
    }
    if let Some(path) = &args.games_jsonl {
        sweep_games_jsonl(path, &args, &mut stats, &mut violations)?;
    }

    let report = json!({
        "format": "rq578-immediate-five-check-v1",
        "positions_jsonl": args.positions_jsonl,
        "games_jsonl": args.games_jsonl,
        "max_transitions": args.max_transitions,
        "stats": {
            "position_records": stats.position_records,
            "position_boards": stats.position_boards,
            "game_records": stats.game_records,
            "game_boards": stats.game_boards,
            "boolean_tests": stats.boolean_tests,
            "mismatches": stats.mismatches,
            "stored_violations": stats.stored_violations,
        },
        "passed": stats.mismatches == 0,
    });
    let mut out = File::create(&args.out_json)
        .map_err(|e| format!("failed to create {}: {e}", args.out_json.display()))?;
    writeln!(out, "{}", serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| format!("failed to write {}: {e}", args.out_json.display()))?;

    println!(
        "rq578-immediate-five-check: position_boards={} game_boards={} boolean_tests={} mismatches={}",
        stats.position_boards, stats.game_boards, stats.boolean_tests, stats.mismatches
    );
    if stats.mismatches > 0 {
        return Err(format!(
            "immediate-five mismatch count: {} (see {})",
            stats.mismatches,
            args.out_violations_jsonl.display()
        ));
    }
    Ok(())
}

fn sweep_position_jsonl(
    path: &PathBuf,
    args: &Args,
    stats: &mut Stats,
    violations: &mut File,
) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|e| format!("failed to read {}:{}: {e}", path.display(), line_no + 1))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        stats.position_records += 1;
        let rec: Value = serde_json::from_str(line)
            .map_err(|e| format!("failed to parse {}:{}: {e}", path.display(), line_no + 1))?;
        let history = rec
            .get("position_history")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                format!(
                    "missing position_history in {}:{}",
                    path.display(),
                    line_no + 1
                )
            })?;
        let board = board_from_history(history)?;
        let label = json!({
            "source": "positions_jsonl",
            "path": path,
            "line": line_no + 1,
            "game_id": rec.get("game_id"),
            "ply": rec.get("ply"),
            "class": rec.get("class"),
        });
        sweep_board(&board, &label, stats, violations, args.max_violations)?;
        stats.position_boards += 1;
    }
    Ok(())
}

fn sweep_games_jsonl(
    path: &PathBuf,
    args: &Args,
    stats: &mut Stats,
    violations: &mut File,
) -> Result<(), String> {
    let games = load_trace_games(path, stats)?;
    if games.is_empty() {
        return Err(format!("no games loaded from {}", path.display()));
    }
    'passes: while stats.game_boards < args.max_transitions {
        for (game_idx, game) in games.iter().enumerate() {
            if stats.game_boards >= args.max_transitions {
                break 'passes;
            }
            let mut board = Board::new();
            for (ply, &mv) in game.moves.iter().enumerate() {
                if stats.game_boards >= args.max_transitions {
                    break 'passes;
                }
                if !board.is_empty(mv) {
                    break;
                }
                board.make_move(mv);
                let label = json!({
                    "source": "games_jsonl",
                    "path": path,
                    "line": game.line_no,
                    "pass_game_idx": game_idx,
                    "game_id": game.game_id,
                    "seed": game.seed,
                    "ply_after_move": ply + 1,
                });
                sweep_board(&board, &label, stats, violations, args.max_violations)?;
                stats.game_boards += 1;
            }
        }
    }
    Ok(())
}

fn sweep_board(
    board: &Board,
    label: &Value,
    stats: &mut Stats,
    violations: &mut File,
    max_violations: usize,
) -> Result<(), String> {
    for side in [Stone::Black, Stone::White] {
        for use_reach_mask in [false, true] {
            let reference = has_immediate_five_reference_for_audit(board, side, use_reach_mask);
            let fast = has_immediate_five_fast_for_audit(board, side, use_reach_mask);
            stats.boolean_tests += 1;
            if reference != fast {
                stats.mismatches += 1;
                if stats.stored_violations < max_violations {
                    stats.stored_violations += 1;
                    let rec = json!({
                        "label": label,
                        "side": stone_json(side),
                        "use_reach_mask": use_reach_mask,
                        "reference": reference,
                        "fast": fast,
                        "history_len": board.history.len(),
                        "history": history_json(board),
                    });
                    writeln!(violations, "{}", serde_json::to_string(&rec).unwrap())
                        .map_err(|e| format!("failed to write violation jsonl: {e}"))?;
                }
            }
        }
    }
    Ok(())
}

struct TraceGame {
    line_no: usize,
    game_id: Value,
    seed: Value,
    moves: Vec<Move>,
}

fn load_trace_games(path: &PathBuf, stats: &mut Stats) -> Result<Vec<TraceGame>, String> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let mut games = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|e| format!("failed to read {}:{}: {e}", path.display(), line_no + 1))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        stats.game_records += 1;
        let rec: Value = serde_json::from_str(line)
            .map_err(|e| format!("failed to parse {}:{}: {e}", path.display(), line_no + 1))?;
        let moves = rec
            .get("moves")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("missing moves array in {}:{}", path.display(), line_no + 1))?;
        let mut parsed = Vec::with_capacity(moves.len());
        for mv_json in moves {
            parsed.push(parse_move(mv_json)?);
        }
        games.push(TraceGame {
            line_no: line_no + 1,
            game_id: rec.get("game_id").cloned().unwrap_or(Value::Null),
            seed: rec.get("seed").cloned().unwrap_or(Value::Null),
            moves: parsed,
        });
    }
    Ok(games)
}

fn board_from_history(history: &[Value]) -> Result<Board, String> {
    let mut board = Board::new();
    for item in history {
        let side = parse_side(
            item.get("color")
                .and_then(Value::as_str)
                .ok_or("history item missing color")?,
        )?;
        if board.side_to_move != side {
            return Err(format!(
                "history side mismatch at move {}: board={:?} item={:?}",
                board.move_count, board.side_to_move, side
            ));
        }
        let mv = parse_move(item)?;
        if !board.is_empty(mv) {
            return Err(format!("duplicate move in history: {mv}"));
        }
        board.make_move(mv);
    }
    Ok(board)
}

fn parse_move(v: &Value) -> Result<Move, String> {
    let x = v.get("x").and_then(Value::as_u64).ok_or("move missing x")? as usize;
    let y = v.get("y").and_then(Value::as_u64).ok_or("move missing y")? as usize;
    if x >= BOARD_SIZE || y >= BOARD_SIZE {
        return Err(format!("move out of board: x={x} y={y}"));
    }
    Ok(to_idx(y, x))
}

fn parse_side(s: &str) -> Result<Stone, String> {
    match s {
        "B" | "Black" | "black" => Ok(Stone::Black),
        "W" | "White" | "white" => Ok(Stone::White),
        _ => Err(format!("bad side: {s}")),
    }
}

fn stone_json(side: Stone) -> &'static str {
    match side {
        Stone::Black => "B",
        Stone::White => "W",
    }
}

fn history_json(board: &Board) -> Value {
    Value::Array(
        board
            .history
            .iter()
            .enumerate()
            .map(|(i, &mv)| {
                let (row, col) = to_rc(mv);
                json!({
                    "x": col,
                    "y": row,
                    "color": if i % 2 == 0 { "B" } else { "W" },
                })
            })
            .collect(),
    )
}

fn parse_args() -> Result<Args, String> {
    let mut positions_jsonl = Vec::new();
    let mut games_jsonl = None;
    let mut out_json = None;
    let mut out_violations_jsonl = None;
    let mut max_transitions = 100_000usize;
    let mut max_violations = 20usize;

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--positions-jsonl" => positions_jsonl.push(PathBuf::from(next_arg(&mut it, &arg)?)),
            "--games-jsonl" => games_jsonl = Some(PathBuf::from(next_arg(&mut it, &arg)?)),
            "--out-json" => out_json = Some(PathBuf::from(next_arg(&mut it, &arg)?)),
            "--out-violations-jsonl" => {
                out_violations_jsonl = Some(PathBuf::from(next_arg(&mut it, &arg)?))
            }
            "--max-transitions" => {
                max_transitions = next_arg(&mut it, &arg)?
                    .parse()
                    .map_err(|e| format!("bad --max-transitions: {e}"))?
            }
            "--max-violations" => {
                max_violations = next_arg(&mut it, &arg)?
                    .parse()
                    .map_err(|e| format!("bad --max-violations: {e}"))?
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    if positions_jsonl.is_empty() && games_jsonl.is_none() {
        return Err("at least one --positions-jsonl or --games-jsonl is required".to_string());
    }

    Ok(Args {
        positions_jsonl,
        games_jsonl,
        out_json: out_json.ok_or("--out-json is required")?,
        out_violations_jsonl: out_violations_jsonl.ok_or("--out-violations-jsonl is required")?,
        max_transitions,
        max_violations,
    })
}

fn next_arg(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next()
        .ok_or_else(|| format!("{flag} requires an argument"))
}

fn print_help() {
    eprintln!(
        "Usage: rq578-immediate-five-check [--positions-jsonl FILE ...] [--games-jsonl FILE] --out-json FILE --out-violations-jsonl FILE [--max-transitions N] [--max-violations N]"
    );
}
