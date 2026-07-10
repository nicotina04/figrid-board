use figrid_board::{
    directional_aggregation_mismatches, generate_dependency_quiet_candidates, to_idx, to_rc, Board,
    DependencyQuietCandidate, Move, QuietThreatConfig, Stone, BOARD_SIZE,
};
use serde_json::{json, Value};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

struct Args {
    input: PathBuf,
    output: PathBuf,
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let input = File::open(&args.input)
        .map_err(|e| format!("failed to open {}: {e}", args.input.display()))?;
    let mut output = File::create(&args.output)
        .map_err(|e| format!("failed to create {}: {e}", args.output.display()))?;
    let mut scanned = 0usize;
    let mut emitted = 0usize;
    let mut directional_mismatches = 0usize;
    let mut subset_violations = 0usize;
    let config = QuietThreatConfig::default();

    for (source_index, line) in BufReader::new(input).lines().enumerate() {
        let line = line.map_err(|e| format!("failed to read input: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        scanned += 1;
        let rec: Value = serde_json::from_str(&line)
            .map_err(|e| format!("invalid JSON at source row {source_index}: {e}"))?;
        if rec.get("class").and_then(Value::as_str) != Some("tactical_missed_win") {
            continue;
        }

        let mut board = board_from_history(
            rec.get("position_history")
                .and_then(Value::as_array)
                .ok_or("missing position_history")?,
        )?;
        let side = parse_side(
            rec.get("side_to_move")
                .and_then(Value::as_str)
                .ok_or("missing side_to_move")?,
        )?;
        if board.side_to_move != side {
            return Err(format!("side mismatch at source row {source_index}"));
        }
        directional_mismatches += directional_aggregation_mismatches(&board, config);

        let rapfi = parse_move(rec.get("rapfi_move").ok_or("missing rapfi_move")?)?;
        let arms = generate_dependency_quiet_candidates(&mut board, config);
        subset_violations += arms
            .d2
            .iter()
            .filter(|candidate| {
                !arms
                    .d1
                    .iter()
                    .any(|d1_candidate| d1_candidate.mv == candidate.mv)
            })
            .count();

        let row = json!({
            "format": "rq598-tss-dependency-v1",
            "source_index": source_index,
            "arm": rec.get("arm"),
            "game_id": rec.get("game_id"),
            "ply": rec.get("ply"),
            "side_to_move": rec.get("side_to_move"),
            "rapfi_move": move_json(rapfi),
            "d1": arm_json(&arms.d1, rapfi),
            "d2": arm_json(&arms.d2, rapfi),
        });
        writeln!(output, "{}", serde_json::to_string(&row).unwrap())
            .map_err(|e| format!("failed to write {}: {e}", args.output.display()))?;
        emitted += 1;
    }

    println!(
        "rq598-tss-dependency-probe: scanned={scanned} emitted={emitted} \
         directional_mismatches={directional_mismatches} subset_violations={subset_violations}"
    );
    if directional_mismatches != 0 || subset_violations != 0 {
        return Err("RQ598 semantic gate failed".to_string());
    }
    Ok(())
}

fn arm_json(candidates: &[DependencyQuietCandidate], rapfi: Move) -> Value {
    let rank = candidates.iter().position(|c| c.mv == rapfi).map(|i| i + 1);
    json!({
        "count": candidates.len(),
        "rapfi_included": rank.is_some(),
        "rapfi_rank": rank,
        "candidates": candidates.iter().map(|c| json!({
            "move": move_json(c.mv),
            "forcing_gains": c.forcing_gains,
            "winning_gains": c.winning_gains,
            "dependency_links": c.dependency_links,
            "max_reused_support": c.max_reused_support,
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
            return Err(format!(
                "history side mismatch at move {}",
                board.move_count
            ));
        }
        let mv = parse_move(item)?;
        if !board.is_empty(mv) {
            return Err(format!("occupied history move {mv}"));
        }
        board.make_move(mv);
    }
    Ok(board)
}

fn parse_move(v: &Value) -> Result<Move, String> {
    let x = v.get("x").and_then(Value::as_u64).ok_or("move missing x")? as usize;
    let y = v.get("y").and_then(Value::as_u64).ok_or("move missing y")? as usize;
    if x >= BOARD_SIZE || y >= BOARD_SIZE {
        return Err(format!("move out of board: ({x},{y})"));
    }
    Ok(to_idx(y, x))
}

fn parse_side(s: &str) -> Result<Stone, String> {
    match s {
        "B" | "black" | "Black" => Ok(Stone::Black),
        "W" | "white" | "White" => Ok(Stone::White),
        _ => Err(format!("unknown side: {s}")),
    }
}

fn move_json(mv: Move) -> Value {
    let (row, col) = to_rc(mv);
    json!({"x": col, "y": row})
}

fn parse_args() -> Result<Args, String> {
    let mut input = None;
    let mut output = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next().map(PathBuf::from),
            "--output" => output = args.next().map(PathBuf::from),
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    Ok(Args {
        input: input.ok_or("missing --input")?,
        output: output.ok_or("missing --output")?,
    })
}
