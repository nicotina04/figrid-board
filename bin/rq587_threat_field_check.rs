use figrid_board::threat_field::threat_field_transition_check_for_audit;
use figrid_board::{BOARD_SIZE, Move, to_idx};
use serde_json::{Value, json};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

struct Args {
    games_jsonl: PathBuf,
    out_json: PathBuf,
    max_transitions: usize,
}

#[derive(Default)]
struct Stats {
    game_records: usize,
    checked_games: usize,
    transitions: usize,
    undo_checks: usize,
}

struct TraceGame {
    line_no: usize,
    game_id: Value,
    seed: Value,
    moves: Vec<Move>,
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let games = load_games(&args.games_jsonl)?;
    if games.is_empty() {
        return Err(format!(
            "no games loaded from {}",
            args.games_jsonl.display()
        ));
    }

    let mut stats = Stats {
        game_records: games.len(),
        ..Stats::default()
    };
    let mut first_error = Value::Null;

    'passes: while stats.transitions < args.max_transitions {
        for game in &games {
            if stats.transitions >= args.max_transitions {
                break 'passes;
            }
            let remaining = args.max_transitions - stats.transitions;
            let take = game.moves.len().min(remaining);
            if take == 0 {
                continue;
            }
            match threat_field_transition_check_for_audit(&game.moves[..take]) {
                Ok((transitions, undos)) => {
                    stats.checked_games += 1;
                    stats.transitions += transitions;
                    stats.undo_checks += undos;
                }
                Err(error) => {
                    first_error = json!({
                        "line": game.line_no,
                        "game_id": game.game_id,
                        "seed": game.seed,
                        "prefix_len": take,
                        "error": error,
                    });
                    break 'passes;
                }
            }
        }
    }

    let passed = first_error.is_null();
    let report = json!({
        "format": "rq587-threat-field-check-v1",
        "games_jsonl": args.games_jsonl,
        "max_transitions": args.max_transitions,
        "stats": {
            "game_records": stats.game_records,
            "checked_games": stats.checked_games,
            "transitions": stats.transitions,
            "undo_checks": stats.undo_checks,
        },
        "first_error": first_error,
        "passed": passed,
    });
    let mut out = File::create(&args.out_json)
        .map_err(|e| format!("failed to create {}: {e}", args.out_json.display()))?;
    writeln!(out, "{}", serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| format!("failed to write {}: {e}", args.out_json.display()))?;

    println!(
        "rq587-threat-field-check: transitions={} undo_checks={} passed={}",
        stats.transitions, stats.undo_checks, passed
    );
    if passed {
        Ok(())
    } else {
        Err(format!("threat-field mismatch: {first_error}"))
    }
}

fn load_games(path: &PathBuf) -> Result<Vec<TraceGame>, String> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let mut games = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|e| format!("failed to read {}:{}: {e}", path.display(), line_no + 1))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
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

fn parse_move(v: &Value) -> Result<Move, String> {
    let x = v.get("x").and_then(Value::as_u64).ok_or("move missing x")? as usize;
    let y = v.get("y").and_then(Value::as_u64).ok_or("move missing y")? as usize;
    if x >= BOARD_SIZE || y >= BOARD_SIZE {
        return Err(format!("move out of board: ({x},{y})"));
    }
    Ok(to_idx(y, x))
}

fn parse_args() -> Result<Args, String> {
    let mut games_jsonl = None;
    let mut out_json = None;
    let mut max_transitions = 100_000usize;

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--games-jsonl" => games_jsonl = Some(PathBuf::from(next_arg(&mut it, &arg)?)),
            "--out-json" => out_json = Some(PathBuf::from(next_arg(&mut it, &arg)?)),
            "--max-transitions" => {
                max_transitions = next_arg(&mut it, &arg)?
                    .parse()
                    .map_err(|e| format!("invalid --max-transitions: {e}"))?
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown arg: {arg}")),
        }
    }

    Ok(Args {
        games_jsonl: games_jsonl.ok_or("missing --games-jsonl")?,
        out_json: out_json.ok_or("missing --out-json")?,
        max_transitions,
    })
}

fn next_arg(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} requires a value"))
}

fn print_help() {
    eprintln!(
        "usage: rq587-threat-field-check --games-jsonl games.jsonl --out-json out.json [--max-transitions 100000]"
    );
}
