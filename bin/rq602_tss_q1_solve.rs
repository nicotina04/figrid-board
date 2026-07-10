use figrid_board::{
    BOARD_SIZE, Board, Move, Q1CandidateAttempt, Q1TssConfig, Stone, search_q1_tss_root, to_idx,
    to_rc,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

struct Args {
    input: PathBuf,
    output: PathBuf,
    summary: PathBuf,
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let input = File::open(&args.input)
        .map_err(|e| format!("failed to open {}: {e}", args.input.display()))?;
    let mut output = File::create(&args.output)
        .map_err(|e| format!("failed to create {}: {e}", args.output.display()))?;
    let config = Q1TssConfig::default();
    let mut rows = 0usize;
    let mut errors = 0usize;
    let mut hits = 0usize;
    let mut short_rows = 0usize;
    let mut short_hits = 0usize;
    let mut labelled_first = 0usize;
    let mut stop_reasons = BTreeMap::<String, usize>::new();

    for (line_index, line) in BufReader::new(input).lines().enumerate() {
        let line = line.map_err(|e| format!("failed to read input: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        rows += 1;
        let record: Value = serde_json::from_str(&line)
            .map_err(|e| format!("invalid JSON at row {line_index}: {e}"))?;
        match solve_record(&record, &config) {
            Ok(solved) => {
                let hit = solved.get("hit").and_then(Value::as_bool) == Some(true);
                let mate_in = solved.get("mate_in").and_then(Value::as_i64);
                hits += usize::from(hit);
                if mate_in.is_some_and(|mate| mate <= 21) {
                    short_rows += 1;
                    short_hits += usize::from(hit);
                }
                labelled_first += usize::from(
                    solved.get("first_move_relation").and_then(Value::as_str) == Some("eq_rapfi"),
                );
                let reason = solved
                    .get("stop_reason")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                *stop_reasons.entry(reason).or_default() += 1;
                writeln!(output, "{}", serde_json::to_string(&solved).unwrap())
                    .map_err(|e| format!("failed to write output: {e}"))?;
            }
            Err(error) => {
                errors += 1;
                writeln!(
                    output,
                    "{}",
                    serde_json::to_string(&json!({
                        "format": "rq602-tss-q1-v1",
                        "error": error,
                        "input": record,
                    }))
                    .unwrap()
                )
                .map_err(|e| format!("failed to write error row: {e}"))?;
            }
        }
    }

    let summary = json!({
        "format": "rq602-tss-q1-summary-v1",
        "arm": "l3-q1-full-defense",
        "input": args.input,
        "output": args.output,
        "config": {
            "max_candidates": config.max_candidates,
            "quiet_plies": 1,
            "defenses": "all_legal_moves",
            "child_vct_arm": "rootk",
            "child_vct_depth": config.child_vct_depth,
            "global_budget_ms": config.time_budget.map(|budget| budget.as_millis()),
        },
        "rows": rows,
        "errors": errors,
        "hits": hits,
        "mate_le_21_rows": short_rows,
        "mate_le_21_hits": short_hits,
        "labelled_first_moves": labelled_first,
        "stop_reasons": stop_reasons,
    });
    let mut summary_file = File::create(&args.summary)
        .map_err(|e| format!("failed to create {}: {e}", args.summary.display()))?;
    writeln!(
        summary_file,
        "{}",
        serde_json::to_string_pretty(&summary).unwrap()
    )
    .map_err(|e| format!("failed to write summary: {e}"))?;
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    Ok(())
}

fn solve_record(record: &Value, config: &Q1TssConfig) -> Result<Value, String> {
    let mut board = board_from_history(
        record
            .get("position_history")
            .and_then(Value::as_array)
            .ok_or("missing position_history")?,
    )?;
    let side = parse_side(
        record
            .get("side_to_move")
            .and_then(Value::as_str)
            .ok_or("missing side_to_move")?,
    )?;
    if board.side_to_move != side {
        return Err("side-to-move mismatch".to_string());
    }
    let rapfi = parse_move(record.get("rapfi_move").ok_or("missing rapfi_move")?)?;
    let before = board.clone();
    let result = search_q1_tss_root(&mut board, config);
    if board.history != before.history
        || board.black != before.black
        || board.white != before.white
        || board.side_to_move != before.side_to_move
        || board.zobrist != before.zobrist
    {
        return Err("board round-trip mismatch".to_string());
    }
    let relation = match result.selected_move {
        Some(mv) if mv == rapfi => "eq_rapfi",
        Some(_) => "other_proved_move",
        None => "none",
    };
    Ok(json!({
        "format": "rq602-tss-q1-v1",
        "arm": record.get("arm"),
        "seed": record.get("seed"),
        "game_id": record.get("game_id"),
        "ply": record.get("ply"),
        "side_to_move": record.get("side_to_move"),
        "mate_in": record.get("mate_in"),
        "rapfi_move": move_json(rapfi),
        "hit": result.selected_move.is_some(),
        "selected_move": result.selected_move.map(move_json),
        "first_move_relation": relation,
        "candidate_count": result.candidate_count,
        "candidates_tested": result.candidates_tested,
        "child_nodes": result.child_nodes,
        "elapsed_ms": result.elapsed.as_millis(),
        "stop_reason": result.stop_reason.as_str(),
        "attempts": result.attempts.iter().map(attempt_json).collect::<Vec<_>>(),
    }))
}

fn attempt_json(attempt: &Q1CandidateAttempt) -> Value {
    json!({
        "move": move_json(attempt.mv),
        "forcing_gains": attempt.forcing_gains,
        "winning_gains": attempt.winning_gains,
        "defenses_total": attempt.defenses_total,
        "defenses_tested": attempt.defenses.len(),
        "complete": attempt.complete,
        "defenses": attempt.defenses.iter().map(|defense| json!({
            "move": move_json(defense.mv),
            "outcome": defense.outcome.as_str(),
            "child_nodes": defense.child_nodes,
            "child_first_move": defense.child_first_move.map(move_json),
        })).collect::<Vec<_>>(),
    })
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
            return Err("history side mismatch".to_string());
        }
        let mv = parse_move(item)?;
        if !board.is_empty(mv) {
            return Err("occupied history move".to_string());
        }
        board.make_move(mv);
    }
    Ok(board)
}

fn parse_move(value: &Value) -> Result<Move, String> {
    let x = value
        .get("x")
        .and_then(Value::as_u64)
        .ok_or("move missing x")? as usize;
    let y = value
        .get("y")
        .and_then(Value::as_u64)
        .ok_or("move missing y")? as usize;
    if x >= BOARD_SIZE || y >= BOARD_SIZE {
        return Err("move out of board".to_string());
    }
    Ok(to_idx(y, x))
}

fn parse_side(side: &str) -> Result<Stone, String> {
    match side {
        "B" | "black" | "Black" => Ok(Stone::Black),
        "W" | "white" | "White" => Ok(Stone::White),
        _ => Err(format!("unknown side: {side}")),
    }
}

fn move_json(mv: Move) -> Value {
    let (row, col) = to_rc(mv);
    json!({"x": col, "y": row})
}

fn parse_args() -> Result<Args, String> {
    let mut input = None;
    let mut output = None;
    let mut summary = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next().map(PathBuf::from),
            "--output" => output = args.next().map(PathBuf::from),
            "--summary" => summary = args.next().map(PathBuf::from),
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    Ok(Args {
        input: input.ok_or("missing --input")?,
        output: output.ok_or("missing --output")?,
        summary: summary.ok_or("missing --summary")?,
    })
}
