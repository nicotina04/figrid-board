//! Deterministic release replay. Run both revisions with identical arguments.
use figrid_board::{Board, GameResult, Searcher, GOMOKU_NNUE_CONFIG};
use figrid_board::codebook_eval::CodebookWeights;
use noru::network::NnueWeights;
use serde_json::{Value, json};
use std::{fs, io::{BufRead, BufReader, Read}, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let openings: Vec<Value> = BufReader::new(fs::File::open(&a[1])?).lines()
        .map(|s| serde_json::from_str(&s.unwrap()).unwrap()).collect();
    let mut weight_bytes = Vec::new();
    flate2::read::GzDecoder::new(&include_bytes!("../models/gomoku_v52_5stone_conv_93k.bin.gz")[..])
        .read_to_end(&mut weight_bytes)?;
    let weights = NnueWeights::load_from_bytes(&weight_bytes, Some(GOMOKU_NNUE_CONFIG))?;
    let codebook = CodebookWeights::from_json_bytes(include_bytes!("../models/gomoku_codebook_v1_swapclosed.json"))?
        .quantize_i16_s32_s64();
    let games = a.get(3).is_some_and(|x| x == "games");
    let mut rows = Vec::new();
    for frontier in [false, true] {
        for (i, o) in openings.iter().take(if games {6} else {30}).enumerate() {
            let mut board = Board::new();
            for m in o["history"].as_array().unwrap() {
                board.make_move(m["y"].as_u64().unwrap() as usize * 15 + m["x"].as_u64().unwrap() as usize);
            }
            loop {
                let mut searcher = Searcher::new();
                searcher.set_use_packed_line_windows(true);
                searcher.set_use_candidate_frontier(frontier);
                searcher.set_use_codebook_directional_delta(true);
                searcher.set_white_root_order_enabled(true)?;
                searcher.set_node_limit(Some(8192));
                let history = board.history.clone(); let hash = board.zobrist;
                let start = Instant::now();
                let result = searcher.search_codebook_eval_quantized(&mut board, &weights, &codebook, 64, None);
                let ms = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(board.history, history); assert_eq!(board.zobrist, hash);
                let mv = result.best_move.ok_or("no searched move")?; assert!(board.is_empty(mv));
                rows.push(json!({"frontier":frontier,"opening":i,"ply":board.move_count,
                    "move":mv,"score":result.score,"depth":result.depth,"nodes":result.nodes,"ms":ms}));
                board.make_move(mv);
                if !games || board.game_result() != GameResult::Ongoing {break}
            }
        }
    }
    fs::write(&a[2], serde_json::to_vec_pretty(&json!({"rows":rows}))?)?;
    Ok(())
}
