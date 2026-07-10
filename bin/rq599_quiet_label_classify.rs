use figrid_board::pattern_table::WindowThreat;
use figrid_board::vct::ThreatKind;
use figrid_board::{
    classify_move_with_directions, to_idx, Board, Move, QuietThreatConfig, Stone, BOARD_SIZE,
};
use serde_json::{json, Map, Value};
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
    let config = QuietThreatConfig::default();
    let mut rows = 0usize;
    let mut evaluated = 0usize;
    let mut quiet = 0usize;

    for (line_index, line) in BufReader::new(input).lines().enumerate() {
        let line = line.map_err(|e| format!("failed to read input: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        rows += 1;
        let mut record: Value = serde_json::from_str(&line)
            .map_err(|e| format!("invalid JSON at row {line_index}: {e}"))?;
        let object = record
            .as_object_mut()
            .ok_or_else(|| format!("row {line_index} is not a JSON object"))?;
        if object.get("discovery_hit").and_then(Value::as_bool) != Some(true) {
            object.insert(
                "figrid_classification".to_string(),
                json!({"evaluated": false, "quiet": false}),
            );
            writeln!(output, "{}", serde_json::to_string(&record).unwrap())
                .map_err(|e| format!("failed to write output: {e}"))?;
            continue;
        }

        let board = board_from_history(
            object
                .get("position_history")
                .and_then(Value::as_array)
                .ok_or("missing position_history")?,
        )?;
        let side = parse_side(
            object
                .get("side_to_move")
                .and_then(Value::as_str)
                .ok_or("missing side_to_move")?,
        )?;
        if board.side_to_move != side {
            return Err(format!("side mismatch at row {line_index}"));
        }
        let mv = parse_move(
            object
                .get("rapfi")
                .and_then(Value::as_object)
                .and_then(|rapfi| rapfi.get("move"))
                .ok_or("missing rapfi.move")?,
        )?;
        if !board.is_empty(mv) {
            return Err(format!("occupied rapfi move at row {line_index}"));
        }
        let (aggregate, directions) = classify_move_with_directions(&board, mv, side, config);
        let is_quiet = aggregate == ThreatKind::None;
        evaluated += 1;
        quiet += usize::from(is_quiet);
        object.insert(
            "figrid_classification".to_string(),
            classification_json(aggregate, directions, is_quiet),
        );
        writeln!(output, "{}", serde_json::to_string(&record).unwrap())
            .map_err(|e| format!("failed to write output: {e}"))?;
    }

    println!("rq599-quiet-label-classify: rows={rows} evaluated={evaluated} quiet={quiet}");
    Ok(())
}

fn classification_json(aggregate: ThreatKind, directions: [WindowThreat; 4], quiet: bool) -> Value {
    let mut value = Map::new();
    value.insert("evaluated".to_string(), Value::Bool(true));
    value.insert("quiet".to_string(), Value::Bool(quiet));
    value.insert(
        "aggregate".to_string(),
        Value::String(threat_kind_name(aggregate).to_string()),
    );
    value.insert(
        "directions".to_string(),
        Value::Array(
            directions
                .into_iter()
                .map(|kind| Value::String(window_threat_name(kind).to_string()))
                .collect(),
        ),
    );
    Value::Object(value)
}

fn threat_kind_name(kind: ThreatKind) -> &'static str {
    match kind {
        ThreatKind::None => "None",
        ThreatKind::ClosedFour => "ClosedFour",
        ThreatKind::OpenThree => "OpenThree",
        ThreatKind::Five => "Five",
        ThreatKind::OpenFour => "OpenFour",
        ThreatKind::DoubleFour => "DoubleFour",
        ThreatKind::FourThree => "FourThree",
        ThreatKind::DoubleThree => "DoubleThree",
        ThreatKind::JumpThree => "JumpThree",
    }
}

fn window_threat_name(kind: WindowThreat) -> &'static str {
    match kind {
        WindowThreat::None => "None",
        WindowThreat::OpenTwo => "OpenTwo",
        WindowThreat::ClosedThree => "ClosedThree",
        WindowThreat::OpenThree => "OpenThree",
        WindowThreat::ClosedFour => "ClosedFour",
        WindowThreat::OpenFour => "OpenFour",
        WindowThreat::Five => "Five",
        WindowThreat::JumpThree => "JumpThree",
    }
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
        return Err(format!("move out of board: ({x},{y})"));
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
