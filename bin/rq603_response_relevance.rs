use figrid_board::{
    BOARD_SIZE, Board, Move, QuietThreatConfig, ResponseRelevanceAudit, Stone,
    audit_quiet_response_relevance, to_idx, to_rc,
};
use serde_json::{Value, json};
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
    let mut rows = 0usize;
    let mut errors = 0usize;
    let mut total_f1 = 0usize;
    let mut total_f2 = 0usize;
    let mut causal_outside = 0usize;

    for (line_index, line) in BufReader::new(input).lines().enumerate() {
        let line = line.map_err(|e| format!("failed to read input: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        rows += 1;
        let record: Value = serde_json::from_str(&line)
            .map_err(|e| format!("invalid JSON at row {line_index}: {e}"))?;
        match audit_record(&record) {
            Ok(audited) => {
                total_f1 += audited["f1_width"].as_u64().unwrap_or(0) as usize;
                total_f2 += audited["f2_width"].as_u64().unwrap_or(0) as usize;
                causal_outside += audited["causal_outside_footprint"]
                    .as_array()
                    .map_or(0, Vec::len);
                writeln!(output, "{}", serde_json::to_string(&audited).unwrap())
                    .map_err(|e| format!("failed to write output: {e}"))?;
            }
            Err(error) => {
                errors += 1;
                writeln!(
                    output,
                    "{}",
                    serde_json::to_string(&json!({
                        "format": "rq603-response-relevance-v1",
                        "error": error,
                        "input": record,
                    }))
                    .unwrap()
                )
                .map_err(|e| format!("failed to write error: {e}"))?;
            }
        }
    }

    let summary = json!({
        "format": "rq603-response-relevance-summary-v1",
        "input": args.input,
        "output": args.output,
        "rows": rows,
        "errors": errors,
        "total_f1_replies": total_f1,
        "total_f2_replies": total_f2,
        "causal_outside_footprint": causal_outside,
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

fn audit_record(record: &Value) -> Result<Value, String> {
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
    let quiet_move = parse_move(record.get("rapfi_move").ok_or("missing rapfi_move")?)?;
    let before = board.clone();
    let audit =
        audit_quiet_response_relevance(&mut board, quiet_move, QuietThreatConfig::default())
            .map_err(str::to_string)?;
    if board.history != before.history
        || board.black != before.black
        || board.white != before.white
        || board.side_to_move != before.side_to_move
        || board.zobrist != before.zobrist
    {
        return Err("board round-trip mismatch".to_string());
    }
    Ok(audit_json(record, &audit))
}

fn audit_json(record: &Value, audit: &ResponseRelevanceAudit) -> Value {
    json!({
        "format": "rq603-response-relevance-v1",
        "arm": record.get("arm"),
        "seed": record.get("seed"),
        "game_id": record.get("game_id"),
        "ply": record.get("ply"),
        "side_to_move": record.get("side_to_move"),
        "mate_in": record.get("mate_in"),
        "quiet_move": move_json(audit.quiet_move),
        "forcing_gains": audit.forcing_gains,
        "winning_gains": audit.winning_gains,
        "gained_sources": moves_json(&audit.gained_sources),
        "gained_line_count": audit.gained_line_count,
        "legal_width": audit.legal_replies.len(),
        "immediate_replies": moves_json(&audit.immediate_replies),
        "defender_forcing_replies": moves_json(&audit.defender_forcing_replies),
        "footprint_replies": moves_json(&audit.footprint_replies),
        "causal_replies": moves_json(&audit.causal_replies),
        "f1_width": audit.f1_replies.len(),
        "f1_replies": moves_json(&audit.f1_replies),
        "f2_width": audit.f2_replies.len(),
        "f2_replies": moves_json(&audit.f2_replies),
        "causal_outside_footprint": moves_json(&audit.causal_outside_footprint),
    })
}

fn moves_json(moves: &[Move]) -> Vec<Value> {
    moves.iter().copied().map(move_json).collect()
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
