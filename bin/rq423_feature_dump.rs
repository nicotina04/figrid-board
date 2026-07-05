use figrid_board::board::{Board, Move, Stone, to_idx};
use serde_json::{Value, json};
use std::env;
use std::path::PathBuf;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut input: Option<PathBuf> = None;
    let mut model: Option<String> = None;
    let mut limit: Option<usize> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = Some(PathBuf::from(args.next().ok_or("missing --input value")?)),
            "--model" => model = Some(args.next().ok_or("missing --model value")?),
            "--limit" => {
                limit = Some(
                    args.next()
                        .ok_or("missing --limit value")?
                        .parse()
                        .map_err(|e| format!("invalid --limit: {e}"))?,
                )
            }
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let input = input.ok_or("missing --input")?;
    let text = std::fs::read_to_string(&input).map_err(|e| format!("failed to read input: {e}"))?;
    let mut emitted = 0usize;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if limit.is_some_and(|max| emitted >= max) {
            break;
        }
        let row: Value =
            serde_json::from_str(line).map_err(|e| format!("invalid JSONL row: {e}"))?;
        let report = report_row(&row, model.as_deref())?;
        println!("{}", serde_json::to_string(&report).unwrap());
        emitted += 1;
    }
    Ok(())
}

fn print_help() {
    println!("usage: rq423-feature-dump --input ROWS.jsonl [--model MODEL.json] [--limit N]");
}

fn report_row(row: &Value, model_path: Option<&str>) -> Result<Value, String> {
    let mut board = Board::new();
    let position = row
        .get("position")
        .and_then(Value::as_array)
        .ok_or("missing position array")?;
    for mv in position {
        let color = parse_color(str_field(mv, "color")?)?;
        if board.side_to_move != color {
            return Err(format!(
                "position side mismatch: board={:?}, row={:?}",
                board.side_to_move, color
            ));
        }
        let idx = move_from_xy(num_field(mv, "x")? as usize, num_field(mv, "y")? as usize)?;
        board.make_move(idx);
    }
    let baseline = parse_move_key(str_field(row, "baseline_move")?)?;
    let test = parse_move_key(str_field(row, "test_move")?)?;
    if board.side_to_move != baseline.0 || board.side_to_move != test.0 {
        return Err(format!(
            "move side mismatch: stm={:?}, baseline={:?}, test={:?}",
            board.side_to_move, baseline.0, test.0
        ));
    }
    let rust_features =
        figrid_board::rq423_root_accept::debug_pair_features(&board, baseline.1, test.1);
    let expected = row
        .get("features")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .map(|v| v as f32)
                        .ok_or_else(|| "non-numeric feature".to_string())
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;

    let mut max_abs_diff = 0.0f32;
    let mut mismatch_count = 0usize;
    let mut first_mismatches = Vec::new();
    if let Some(expected) = expected.as_ref() {
        for (idx, (a, b)) in expected.iter().zip(&rust_features).enumerate() {
            let diff = (*a - *b).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
            if diff > 1.0e-4 {
                mismatch_count += 1;
                if first_mismatches.len() < 12 {
                    let name = row
                        .get("feature_names")
                        .and_then(Value::as_array)
                        .and_then(|names| names.get(idx))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    first_mismatches.push(json!({
                        "idx": idx,
                        "name": name,
                        "python": a,
                        "rust": b,
                        "abs_diff": diff,
                    }));
                }
            }
        }
        if expected.len() != rust_features.len() {
            mismatch_count += expected.len().abs_diff(rust_features.len());
        }
    }

    let mut python_probability = None;
    let mut rust_probability = None;
    let mut threshold = None;
    if let Some(model_path) = model_path {
        let (probability, th) =
            figrid_board::rq423_root_accept::debug_model_probability(model_path, &rust_features)?;
        rust_probability = Some(probability);
        threshold = Some(th);
        if let Some(expected) = expected.as_ref() {
            let (probability, _) =
                figrid_board::rq423_root_accept::debug_model_probability(model_path, expected)?;
            python_probability = Some(probability);
        }
    }

    Ok(json!({
        "source": row.get("source").cloned().unwrap_or(Value::Null),
        "seed": row.get("seed").cloned().unwrap_or(Value::Null),
        "outcome": row.get("outcome").cloned().unwrap_or(Value::Null),
        "ply": row.get("ply").cloned().unwrap_or(Value::Null),
        "baseline_move": row.get("baseline_move").cloned().unwrap_or(Value::Null),
        "test_move": row.get("test_move").cloned().unwrap_or(Value::Null),
        "feature_count": rust_features.len(),
        "expected_feature_count": expected.as_ref().map(Vec::len),
        "max_abs_diff": max_abs_diff,
        "mismatch_count": mismatch_count,
        "first_mismatches": first_mismatches,
        "python_probability": python_probability,
        "rust_probability": rust_probability,
        "threshold": threshold,
    }))
}

fn str_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string field {key}"))
}

fn num_field(value: &Value, key: &str) -> Result<i64, String> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("missing integer field {key}"))
}

fn parse_color(value: &str) -> Result<Stone, String> {
    match value {
        "B" | "black" | "Black" => Ok(Stone::Black),
        "W" | "white" | "White" => Ok(Stone::White),
        other => Err(format!("invalid color {other:?}")),
    }
}

fn parse_move_key(value: &str) -> Result<(Stone, Move), String> {
    let (color, xy) = value
        .split_once(':')
        .ok_or_else(|| format!("invalid move key {value:?}"))?;
    let (x, y) = xy
        .split_once(',')
        .ok_or_else(|| format!("invalid move xy {value:?}"))?;
    Ok((
        parse_color(color)?,
        move_from_xy(
            x.parse().map_err(|e| format!("invalid move x: {e}"))?,
            y.parse().map_err(|e| format!("invalid move y: {e}"))?,
        )?,
    ))
}

fn move_from_xy(x: usize, y: usize) -> Result<Move, String> {
    if x >= figrid_board::BOARD_SIZE || y >= figrid_board::BOARD_SIZE {
        return Err(format!("move out of bounds: {x},{y}"));
    }
    Ok(to_idx(y, x))
}
