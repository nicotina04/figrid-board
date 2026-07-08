use figrid_board::{
    BOARD_SIZE, Board, Move, Stone, VctConfig, search_vct_audit_json, search_vct_with_stats,
    to_idx, to_rc,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
struct SolveConfig {
    depth: u32,
    budget_ms: u64,
}

struct Args {
    positions_jsonl: PathBuf,
    out_json: PathBuf,
    out_jsonl: PathBuf,
    configs: Vec<SolveConfig>,
    max_positions: usize,
    include_proof: bool,
    enable_jump_three: bool,
    enable_jump_three_attack_defense: bool,
    enable_jump_three_counter: bool,
    enable_jump_three_kind_scoped_defense: bool,
    jump_attack_max_or_levels: u32,
    enable_gap_four: bool,
    use_fast_classify: bool,
    use_threat_index: bool,
    node_budget: Option<u64>,
}

impl Args {
    fn jump_three_attack_defense(&self) -> bool {
        self.enable_jump_three || self.enable_jump_three_attack_defense
    }

    fn jump_three_counter(&self) -> bool {
        self.enable_jump_three || self.enable_jump_three_counter
    }

    fn jump_three_kind_scoped_defense(&self) -> bool {
        self.enable_jump_three_kind_scoped_defense
    }
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    if args.configs.is_empty() {
        return Err("at least one --configs entry is required".to_string());
    }

    let file = File::open(&args.positions_jsonl)
        .map_err(|e| format!("failed to open {}: {e}", args.positions_jsonl.display()))?;
    let mut out_jsonl = File::create(&args.out_jsonl)
        .map_err(|e| format!("failed to create {}: {e}", args.out_jsonl.display()))?;

    let mut records = 0usize;
    let mut usable = 0usize;
    let mut skipped = 0usize;
    let mut class_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut verdict_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut results = Vec::new();

    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| format!("failed to read jsonl: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        records += 1;
        if args.max_positions > 0 && usable >= args.max_positions {
            break;
        }
        let rec: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let solved = match solve_record(
            &rec,
            &args.configs,
            args.include_proof,
            args.enable_jump_three,
            args.enable_jump_three_attack_defense,
            args.enable_jump_three_counter,
            args.enable_jump_three_kind_scoped_defense,
            args.jump_attack_max_or_levels,
            args.enable_gap_four,
            args.use_fast_classify,
            args.use_threat_index,
            args.node_budget,
        ) {
            Ok(v) => v,
            Err(e) => {
                skipped += 1;
                json!({
                    "format": "rq547-vct-solve-v1",
                    "error": e,
                    "input": rec,
                })
            }
        };
        usable += 1;
        if let Some(cls) = solved.get("class").and_then(Value::as_str) {
            *class_counts.entry(cls.to_string()).or_default() += 1;
        }
        if let Some(verdict) = solved.get("verdict").and_then(Value::as_str) {
            *verdict_counts.entry(verdict.to_string()).or_default() += 1;
        }
        writeln!(out_jsonl, "{}", serde_json::to_string(&solved).unwrap())
            .map_err(|e| format!("failed to write {}: {e}", args.out_jsonl.display()))?;
        results.push(solved);
    }

    let report = json!({
        "format": "rq547-vct-solve-summary-v1",
        "positions_jsonl": args.positions_jsonl,
        "configs": args.configs.iter().map(|c| json!({"depth": c.depth, "budget_ms": c.budget_ms})).collect::<Vec<_>>(),
        "enable_jump_three": args.enable_jump_three,
        "jump_three_attack_defense": args.jump_three_attack_defense(),
        "jump_three_counter": args.jump_three_counter(),
        "jump_three_kind_scoped_defense": args.jump_three_kind_scoped_defense(),
        "jump_attack_max_or_levels": args.jump_attack_max_or_levels,
        "enable_gap_four": args.enable_gap_four,
        "use_fast_classify": args.use_fast_classify,
        "use_threat_index": args.use_threat_index,
        "node_budget": args.node_budget,
        "records_scanned": records,
        "usable": usable,
        "skipped": skipped,
        "class_counts": class_counts,
        "verdict_counts": verdict_counts,
        "results": results,
    });
    let mut out = File::create(&args.out_json)
        .map_err(|e| format!("failed to create {}: {e}", args.out_json.display()))?;
    writeln!(out, "{}", serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| format!("failed to write {}: {e}", args.out_json.display()))?;

    println!(
        "rq547-vct-solve: usable={} skipped={} verdicts={:?}",
        usable, skipped, verdict_counts
    );
    Ok(())
}

fn solve_record(
    rec: &Value,
    configs: &[SolveConfig],
    include_proof: bool,
    enable_jump_three: bool,
    enable_jump_three_attack_defense: bool,
    enable_jump_three_counter: bool,
    enable_jump_three_kind_scoped_defense: bool,
    jump_attack_max_or_levels: u32,
    enable_gap_four: bool,
    use_fast_classify: bool,
    use_threat_index: bool,
    node_budget: Option<u64>,
) -> Result<Value, String> {
    let class = rec
        .get("class")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let side = parse_side(
        rec.get("side_to_move")
            .and_then(Value::as_str)
            .ok_or("missing side_to_move")?,
    )?;
    let mut board = board_from_history(
        rec.get("position_history")
            .and_then(Value::as_array)
            .ok_or("missing position_history")?,
    )?;
    if board.side_to_move != side {
        return Err(format!(
            "side mismatch: board={:?} record={:?}",
            board.side_to_move, side
        ));
    }

    let pre_vct = run_sweep(
        &board,
        configs,
        include_proof,
        enable_jump_three,
        enable_jump_three_attack_defense,
        enable_jump_three_counter,
        enable_jump_three_kind_scoped_defense,
        jump_attack_max_or_levels,
        enable_gap_four,
        use_fast_classify,
        use_threat_index,
        node_budget,
    );
    let actual_move = parse_move(rec.get("actual_move").ok_or("missing actual_move")?)?;
    let after_actual_opp_vct = if board.is_empty(actual_move) {
        board.make_move(actual_move);
        let out = run_sweep(
            &board,
            configs,
            include_proof,
            enable_jump_three,
            enable_jump_three_attack_defense,
            enable_jump_three_counter,
            enable_jump_three_kind_scoped_defense,
            jump_attack_max_or_levels,
            enable_gap_four,
            use_fast_classify,
            use_threat_index,
            node_budget,
        );
        board.undo_move();
        out
    } else {
        json!({"error": "actual move is occupied"})
    };

    let verdict_source = if class == "tactical_mate_walkin" {
        "after_actual_opp_vct"
    } else {
        "pre_vct"
    };
    let selected = if verdict_source == "after_actual_opp_vct" {
        &after_actual_opp_vct
    } else {
        &pre_vct
    };
    let verdict = verdict_from_sweep(selected);
    let first_hit_relation =
        first_hit_relation(selected, rec.get("actual_move"), rec.get("rapfi_move"));

    Ok(json!({
        "format": "rq547-vct-solve-v1",
        "arm": rec.get("arm"),
        "source_path": rec.get("source_path"),
        "game_id": rec.get("game_id"),
        "ply": rec.get("ply"),
        "class": class,
        "dcp": rec.get("dcp"),
        "side_to_move": rec.get("side_to_move"),
        "actual_move": rec.get("actual_move"),
        "rapfi_move": rec.get("rapfi_move"),
        "verdict_source": verdict_source,
        "verdict": verdict,
        "first_hit_relation": first_hit_relation,
        "enable_jump_three": enable_jump_three,
        "jump_three_attack_defense": enable_jump_three || enable_jump_three_attack_defense,
        "jump_three_counter": enable_jump_three || enable_jump_three_counter,
        "jump_three_kind_scoped_defense": enable_jump_three_kind_scoped_defense,
        "jump_attack_max_or_levels": jump_attack_max_or_levels,
        "enable_gap_four": enable_gap_four,
        "use_fast_classify": use_fast_classify,
        "use_threat_index": use_threat_index,
        "node_budget": node_budget,
        "pre_vct": pre_vct,
        "after_actual_opp_vct": after_actual_opp_vct,
    }))
}

fn run_sweep(
    board: &Board,
    configs: &[SolveConfig],
    include_proof: bool,
    enable_jump_three: bool,
    enable_jump_three_attack_defense: bool,
    enable_jump_three_counter: bool,
    enable_jump_three_kind_scoped_defense: bool,
    jump_attack_max_or_levels: u32,
    enable_gap_four: bool,
    use_fast_classify: bool,
    use_threat_index: bool,
    node_budget: Option<u64>,
) -> Value {
    let mut attempts = Vec::new();
    let mut first_hit: Option<Value> = None;
    for (idx, cfg) in configs.iter().enumerate() {
        let mut b = board.clone();
        let cfg_obj = VctConfig {
            max_depth: cfg.depth,
            time_budget: if cfg.budget_ms == 0 {
                None
            } else {
                Some(Duration::from_millis(cfg.budget_ms))
            },
            node_budget,
            enable_jump_three,
            enable_jump_three_attack_defense,
            enable_jump_three_counter,
            enable_jump_three_kind_scoped_defense,
            jump_attack_max_or_levels,
            enable_gap_four,
            use_fast_classify,
            use_threat_index,
        };
        let started = Instant::now();
        let should_capture_proof = include_proof && first_hit.is_none();
        let (seq, proof, nodes, deadline_hits, node_budget_hits, termination_reason) =
            if should_capture_proof {
                let proof_json = search_vct_audit_json(&mut b, &cfg_obj);
                let seq = proof_json
                    .get("sequence")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|mv| parse_move(mv).ok())
                            .collect::<Vec<_>>()
                    });
                let nodes = proof_json.get("nodes").and_then(Value::as_u64).unwrap_or(0);
                let deadline_hits = proof_json
                    .get("deadline_hits")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let node_budget_hits = proof_json
                    .get("node_budget_hits")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let termination_reason = proof_json
                    .get("termination_reason")
                    .and_then(Value::as_str)
                    .unwrap_or(if seq.is_some() { "proved" } else { "exhausted" })
                    .to_string();
                let proof = if proof_json
                    .get("hit")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    Some(proof_json)
                } else {
                    None
                };
                (
                    seq,
                    proof,
                    nodes,
                    deadline_hits,
                    node_budget_hits,
                    termination_reason,
                )
            } else {
                let result = search_vct_with_stats(&mut b, &cfg_obj);
                let termination_reason = result.termination_reason().to_string();
                (
                    result.sequence,
                    None,
                    result.stats.nodes,
                    result.stats.deadline_hits,
                    result.stats.node_budget_hits,
                    termination_reason,
                )
            };
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let hit = seq.is_some();
        let seq_json = seq
            .as_ref()
            .map(|s| s.iter().map(|&mv| move_json(mv)).collect::<Vec<_>>());
        let mut attempt = json!({
            "idx": idx,
            "depth": cfg.depth,
            "budget_ms": cfg.budget_ms,
            "elapsed_ms": elapsed_ms,
            "nodes": nodes,
            "deadline_hits": deadline_hits,
            "node_budget_hits": node_budget_hits,
            "termination_reason": termination_reason,
            "hit": hit,
            "sequence_len": seq.as_ref().map(|s| s.len()).unwrap_or(0),
            "sequence": seq_json,
        });
        if let Some(proof) = proof {
            attempt["proof"] = proof;
        }
        if hit && first_hit.is_none() {
            first_hit = Some(attempt.clone());
        }
        attempts.push(attempt);
    }
    json!({
        "hit": first_hit.is_some(),
        "first_hit": first_hit,
        "attempts": attempts,
    })
}

fn verdict_from_sweep(sweep: &Value) -> &'static str {
    let Some(first) = sweep.get("first_hit") else {
        return "beyond_vct";
    };
    if first.is_null() {
        return "beyond_vct";
    }
    if first.get("idx").and_then(Value::as_u64) == Some(0) {
        "reachable_current_budget"
    } else {
        "budget_limited"
    }
}

fn first_hit_relation(
    sweep: &Value,
    actual: Option<&Value>,
    rapfi: Option<&Value>,
) -> &'static str {
    let Some(first) = sweep.get("first_hit") else {
        return "no_hit";
    };
    let Some(seq) = first.get("sequence").and_then(Value::as_array) else {
        return "no_hit";
    };
    let Some(first_mv) = seq.first() else {
        return "no_hit";
    };
    if same_json_move(Some(first_mv), actual) {
        "hit_first_eq_actual"
    } else if same_json_move(Some(first_mv), rapfi) {
        "hit_first_eq_rapfi"
    } else {
        "hit_first_other"
    }
}

fn same_json_move(a: Option<&Value>, b: Option<&Value>) -> bool {
    let (Some(a), Some(b)) = (a, b) else {
        return false;
    };
    a.get("x").and_then(Value::as_i64) == b.get("x").and_then(Value::as_i64)
        && a.get("y").and_then(Value::as_i64) == b.get("y").and_then(Value::as_i64)
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
            return Err(format!("occupied move in history: {mv}"));
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
    let mut positions_jsonl = None;
    let mut out_json = None;
    let mut out_jsonl = None;
    let mut configs = vec![
        SolveConfig {
            depth: 14,
            budget_ms: 250,
        },
        SolveConfig {
            depth: 14,
            budget_ms: 500,
        },
        SolveConfig {
            depth: 18,
            budget_ms: 1000,
        },
        SolveConfig {
            depth: 22,
            budget_ms: 2000,
        },
    ];
    let mut max_positions = 0usize;
    let mut include_proof = false;
    let mut enable_jump_three = env_flag("FIGRID_VCT_ENABLE_JUMP_THREE");
    let mut enable_jump_three_attack_defense =
        env_flag("FIGRID_VCT_ENABLE_JUMP_THREE_ATTACK_DEFENSE");
    let mut enable_jump_three_counter = env_flag("FIGRID_VCT_ENABLE_JUMP_THREE_COUNTER");
    let mut enable_jump_three_kind_scoped_defense =
        env_flag("FIGRID_VCT_ENABLE_JUMP_THREE_KIND_SCOPED_DEFENSE");
    let mut jump_attack_max_or_levels =
        env_u32("FIGRID_VCT_JUMP_ATTACK_MAX_OR_LEVELS").unwrap_or(u32::MAX);
    let mut enable_gap_four = env_flag("FIGRID_VCT_ENABLE_GAP_FOUR");
    let mut use_fast_classify = !env_flag("FIGRID_VCT_USE_SLOW_CLASSIFY");
    if env_flag("FIGRID_VCT_USE_FAST_CLASSIFY") {
        use_fast_classify = true;
    }
    let mut use_threat_index = env_flag("FIGRID_VCT_USE_THREAT_INDEX");
    let mut node_budget = env_u64("FIGRID_VCT_NODE_BUDGET");

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--positions-jsonl" => positions_jsonl = Some(PathBuf::from(next_arg(&mut it, &arg)?)),
            "--out-json" => out_json = Some(PathBuf::from(next_arg(&mut it, &arg)?)),
            "--out-jsonl" => out_jsonl = Some(PathBuf::from(next_arg(&mut it, &arg)?)),
            "--configs" => configs = parse_configs(&next_arg(&mut it, &arg)?)?,
            "--max-positions" => {
                max_positions = next_arg(&mut it, &arg)?
                    .parse()
                    .map_err(|e| format!("invalid --max-positions: {e}"))?
            }
            "--include-proof" => include_proof = true,
            "--enable-jump-three" => enable_jump_three = true,
            "--enable-jump-three-attack-defense" => enable_jump_three_attack_defense = true,
            "--enable-jump-three-counter" => enable_jump_three_counter = true,
            "--enable-jump-three-kind-scoped-defense" => {
                enable_jump_three_kind_scoped_defense = true
            }
            "--jump-attack-max-or-levels" => {
                jump_attack_max_or_levels = next_arg(&mut it, &arg)?
                    .parse()
                    .map_err(|e| format!("invalid --jump-attack-max-or-levels: {e}"))?
            }
            "--enable-gap-four" => enable_gap_four = true,
            "--use-fast-classify" => use_fast_classify = true,
            "--use-slow-classify" => use_fast_classify = false,
            "--use-threat-index" => use_threat_index = true,
            "--node-budget" => {
                node_budget = Some(
                    next_arg(&mut it, &arg)?
                        .parse()
                        .map_err(|e| format!("invalid --node-budget: {e}"))?,
                )
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown arg: {arg}")),
        }
    }
    Ok(Args {
        positions_jsonl: positions_jsonl.ok_or("missing --positions-jsonl")?,
        out_json: out_json.ok_or("missing --out-json")?,
        out_jsonl: out_jsonl.ok_or("missing --out-jsonl")?,
        configs,
        max_positions,
        include_proof,
        enable_jump_three,
        enable_jump_three_attack_defense,
        enable_jump_three_counter,
        enable_jump_three_kind_scoped_defense,
        jump_attack_max_or_levels,
        enable_gap_four,
        use_fast_classify,
        use_threat_index,
        node_budget,
    })
}

fn parse_configs(raw: &str) -> Result<Vec<SolveConfig>, String> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let Some((d, b)) = part.split_once(':') else {
            return Err(format!("bad config '{part}', expected depth:budget_ms"));
        };
        out.push(SolveConfig {
            depth: d
                .parse()
                .map_err(|e| format!("bad depth in config '{part}': {e}"))?,
            budget_ms: b
                .parse()
                .map_err(|e| format!("bad budget in config '{part}': {e}"))?,
        });
    }
    Ok(out)
}

fn next_arg(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} requires a value"))
}

fn env_flag(name: &str) -> bool {
    matches!(
        env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES") | Ok("on") | Ok("ON")
    )
}

fn env_u32(name: &str) -> Option<u32> {
    env::var(name).ok()?.parse().ok()
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.parse().ok()
}

fn print_help() {
    eprintln!(
        "Usage: rq547-vct-solve --positions-jsonl FILE --out-json FILE --out-jsonl FILE [--configs 14:250,14:500,18:1000,22:2000] [--max-positions N] [--include-proof] [--enable-jump-three] [--enable-jump-three-attack-defense] [--enable-jump-three-counter] [--enable-jump-three-kind-scoped-defense] [--jump-attack-max-or-levels K] [--enable-gap-four] [--use-fast-classify|--use-slow-classify] [--use-threat-index] [--node-budget N]"
    );
}
