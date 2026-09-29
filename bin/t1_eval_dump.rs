//! Bit-exact static-eval dump for the quantized codebook evaluator.
//!
//! Usage: `FIGRID_CODEBOOK_WEIGHTS=<model.json|model.ngcb> t1-eval-dump
//! --input games.jsonl --output dump.csv [--stride N]` (default stride 4).
//!
//! Replays every game of a JSONL file (one object per line with `game_id`
//! and `moves: [{"x":..,"y":..}, ...]`) through the incremental quantized
//! codebook state (push_move per move, full refresh once per game) and
//! prints `game_id,ply,value_bits` (value as 8 upper-case hex digits of the
//! f32 bit pattern) every `stride` plies. Two builds evaluate identically
//! on the sample iff their dumps are byte-identical.

use std::io::{BufRead, BufReader, Write};

use figrid_board::codebook_eval::{CodebookWeights, IncrementalQuantizedCodebookEval};
use figrid_board::{Board, to_idx};

fn main() -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut stride = 4usize;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--input" => input = it.next(),
            "--output" => output = it.next(),
            "--stride" => {
                stride = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or("bad --stride")?
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    let input = input.ok_or("--input required")?;
    let output = output.ok_or("--output required")?;

    let cb_path = std::env::var("FIGRID_CODEBOOK_WEIGHTS")
        .map_err(|_| "FIGRID_CODEBOOK_WEIGHTS required".to_string())?;
    let bytes = std::fs::read(&cb_path).map_err(|e| format!("read codebook: {e}"))?;
    let weights = CodebookWeights::from_bytes_auto(&bytes)
        .map_err(|e| format!("parse codebook: {e}"))?
        .quantize_i16_s32_s64();

    let file = std::fs::File::open(&input).map_err(|e| format!("open {input}: {e}"))?;
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&output).map_err(|e| format!("create {output}: {e}"))?,
    );
    let mut evals = 0u64;
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| format!("parse row: {e}"))?;
        let game_id = v["game_id"].as_u64().ok_or("missing game_id")?;
        let moves = v["moves"].as_array().ok_or("missing moves")?;
        let mut board = Board::new();
        let mut state = IncrementalQuantizedCodebookEval::new(&weights);
        state.refresh(&board, &weights);
        for (ply, m) in moves.iter().enumerate() {
            let x = m["x"].as_u64().ok_or("x")? as usize;
            let y = m["y"].as_u64().ok_or("y")? as usize;
            let idx = to_idx(y, x);
            if !board.is_empty(idx) {
                return Err(format!("g{game_id} ply {ply}: occupied"));
            }
            board.make_move(idx);
            state.push_move(&board, idx, &weights);
            if (ply + 1) % stride == 0 {
                let value = state.value(&board, &weights);
                writeln!(out, "{game_id},{},{:08X}", ply + 1, value.to_bits())
                    .map_err(|e| e.to_string())?;
                evals += 1;
            }
        }
    }
    out.flush().map_err(|e| e.to_string())?;
    eprintln!("evals: {evals}");
    Ok(())
}
