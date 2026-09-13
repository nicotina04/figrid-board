//! The two remaining recorded-loss branches, verified without a model.
//!
//! Every fixture in a test is executed and printed before its final assertion,
//! so the unchanged engine produces reviewable negative-control observations.
//! A node-budget stop is inconclusive and never counts as a repaired proof.

use super::*;
use crate::board::GameResult;

const HISTORY_68: [Move; 31] = [
    112, 140, 127, 124, 125, 153, 170, 201, 184, 156, 172, 138, 168, 169, 137, 108, 92, 155, 157,
    142, 154, 123, 93, 185, 217, 141, 139, 143, 144, 183, 197,
];
const HISTORY_71: [Move; 20] = [
    112, 128, 143, 84, 54, 56, 28, 37, 82, 97, 111, 113, 127, 95, 157, 68, 142, 172, 159, 175,
];
const CLASSIFY_PATHS: [(bool, bool); 3] = [(false, false), (true, false), (true, true)];

struct Fixture {
    id: &'static str,
    history: &'static [Move],
    attack: Move,
    counter: Move,
}

fn fixtures() -> [Fixture; 2] {
    [
        Fixture {
            id: "a3_068_g1_p31",
            history: &HISTORY_68,
            attack: 110,
            counter: 202,
        },
        Fixture {
            id: "public_071_g0_p20",
            history: &HISTORY_71,
            attack: 141,
            counter: 98,
        },
    ]
}

fn config() -> VctConfig {
    VctConfig {
        max_depth: 14,
        time_budget: None,
        node_budget: Some(100_000),
        ..VctConfig::default()
    }
}

fn config_json() -> Value {
    let cfg = config();
    json!({
        "max_depth": cfg.max_depth, "selected_and_depth": 13,
        "after_counter_or_depth": 12, "time_budget_ms": null,
        "node_budget": cfg.node_budget,
        "enable_jump_three": cfg.enable_jump_three,
        "enable_jump_three_attack_defense": cfg.enable_jump_three_attack_defense,
        "enable_jump_three_counter": cfg.enable_jump_three_counter,
        "enable_jump_three_kind_scoped_defense": cfg.enable_jump_three_kind_scoped_defense,
        "jump_attack_max_or_levels": cfg.jump_attack_max_or_levels,
        "enable_gap_four": cfg.enable_gap_four,
        "gap_four_attack_max_or_levels": cfg.gap_four_attack_max_or_levels,
        "use_fast_classify": cfg.use_fast_classify,
        "use_threat_index": cfg.use_threat_index,
        "use_reach_mask": cfg.use_reach_mask,
        "use_fast_immediate_five": cfg.use_fast_immediate_five,
        "use_vct_scratch_buffers": cfg.use_vct_scratch_buffers,
        "profile": cfg.profile,
    })
}

fn replay(history: &[Move]) -> Board {
    let mut board = Board::new();
    for &mv in history {
        assert_eq!(board.game_result(), GameResult::Ongoing);
        assert!(
            board.is_legal_move(mv),
            "illegal recorded fixture move {mv}"
        );
        board.make_move(mv);
    }
    assert_eq!(board.game_result(), GameResult::Ongoing);
    board
}

fn transform(cell: Move, symmetry: usize) -> Move {
    let (mut row, mut col) = (cell / BOARD_SIZE, cell % BOARD_SIZE);
    if symmetry >= 4 {
        col = BOARD_SIZE - 1 - col;
    }
    for _ in 0..symmetry % 4 {
        (row, col) = (col, BOARD_SIZE - 1 - row);
    }
    row * BOARD_SIZE + col
}

fn restoration(actual: &Board, expected: &Board) -> Value {
    let fields = [
        (
            "black",
            actual.black.lo == expected.black.lo && actual.black.hi == expected.black.hi,
        ),
        (
            "white",
            actual.white.lo == expected.white.lo && actual.white.hi == expected.white.hi,
        ),
        ("side_to_move", actual.side_to_move == expected.side_to_move),
        ("move_count", actual.move_count == expected.move_count),
        ("last_move", actual.last_move == expected.last_move),
        ("history", actual.history == expected.history),
        ("zobrist", actual.zobrist == expected.zobrist),
        (
            "line_pattern_ids",
            actual.line_pattern_ids == expected.line_pattern_ids,
        ),
        ("rule_set", actual.rule_set == expected.rule_set),
        ("exact5", actual.exact5 == expected.exact5),
    ];
    let mut result = serde_json::Map::new();
    result.insert("all".into(), json!(fields.iter().all(|(_, good)| *good)));
    for (name, good) in fields {
        result.insert(name.into(), json!(good));
    }
    Value::Object(result)
}

fn stats_json(stats: &VctSearchStats) -> Value {
    json!({"nodes": stats.nodes, "deadline_hits": stats.deadline_hits,
           "node_budget_hits": stats.node_budget_hits, "hit_stop": stats.hit_stop()})
}

fn emit(fixture: &Fixture, test: &str, data: Value) {
    let board = replay(fixture.history);
    let mut output = json!({
        "schema": "noru.vct-followup-regression/v2", "test": test,
        "case_id": fixture.id, "history_before_attack": fixture.history,
        "attack": fixture.attack, "counter": fixture.counter,
        "attacker": stone_json(board.side_to_move), "config": config_json(),
        "model_calls": 0, "arena_games": 0, "training_runs": 0,
    });
    for (key, value) in data.as_object().unwrap() {
        output
            .as_object_mut()
            .unwrap()
            .insert(key.clone(), value.clone());
    }
    println!("VCT_FOLLOWUP_JSON {output}");
}

fn defenses_for_path(
    board: &Board,
    attack: Move,
    kind: ThreatKind,
    fast: bool,
    indexed: bool,
    use_reach: bool,
) -> Vec<Move> {
    let cfg = VctConfig {
        use_fast_classify: fast,
        use_threat_index: indexed,
        use_reach_mask: use_reach,
        ..config()
    };
    let index = indexed.then(|| VctThreatIndex::new(board));
    let reach = use_reach.then(|| reach_mask_for_side(board, board.side_to_move));
    find_defenses_with_counters(
        board,
        attack,
        kind,
        cfg.jump_three_flags(),
        index.as_ref(),
        reach.as_ref(),
        None,
        &mut VctScratch::default(),
        false,
    )
}

#[test]
fn both_recorded_counters_are_present_across_d4_and_classification_paths() {
    let mut failures = Vec::new();
    for fixture in fixtures() {
        let mut rows = Vec::new();
        for symmetry in 0..8 {
            let history: Vec<_> = fixture
                .history
                .iter()
                .map(|&m| transform(m, symmetry))
                .collect();
            let attack = transform(fixture.attack, symmetry);
            let counter = transform(fixture.counter, symmetry);
            let mut board = replay(&history);
            let attacker = board.side_to_move;
            let kind = classify_move_fast(&board, attack, attacker);
            board.make_move(attack);
            let saved = board.clone();
            let defender = board.side_to_move;
            let off =
                classify_move_fast_with_flags_for_audit(&board, counter, defender, false, false);
            let on =
                classify_move_fast_with_flags_for_audit(&board, counter, defender, false, true);
            let local = find_defenses(&board, attack, false, false);
            let reach_contains = reach_mask_for_side(&board, defender).get(counter);
            let mut reference: Option<Vec<Move>> = None;
            for (fast, indexed) in CLASSIFY_PATHS {
                for use_reach in [false, true] {
                    let defenses =
                        defenses_for_path(&board, attack, kind, fast, indexed, use_reach);
                    let mut sorted = defenses.clone();
                    sorted.sort_unstable();
                    let legal_unique = sorted.iter().all(|&mv| board.is_legal_move(mv))
                        && sorted.windows(2).all(|p| p[0] != p[1]);
                    let agrees = reference.as_ref().map_or(true, |r| r == &sorted);
                    if reference.is_none() {
                        reference = Some(sorted);
                    }
                    let restored = restoration(&board, &saved);
                    let contains = defenses.contains(&counter);
                    let good = contains
                        && legal_unique
                        && agrees
                        && reach_contains
                        && !local.contains(&counter)
                        && kind.is_forcing()
                        && !is_vct_terminal_win(kind)
                        && on.is_forcing()
                        && restored["all"] == true;
                    if !good {
                        failures.push(format!(
                            "{} symmetry={symmetry} fast={fast} index={indexed} reach={use_reach}",
                            fixture.id
                        ));
                    }
                    rows.push(json!({
                        "symmetry": symmetry, "fast": fast, "indexed": indexed, "reach": use_reach,
                        "history_before_attack": history, "attack": attack, "counter": counter,
                        "attack_kind": threat_json(kind), "counter_kind_gap_off": threat_json(off),
                        "counter_kind_gap_on": threat_json(on), "local_defenses": local,
                        "reach_contains_counter": reach_contains, "defenses": defenses,
                        "contains_counter": contains, "legal_unique": legal_unique,
                        "paths_agree": agrees, "restoration": restored, "passed": good,
                    }));
                }
            }
        }
        emit(
            &fixture,
            "counter_candidates",
            json!({"rows": rows, "vct_calls": 0}),
        );
    }
    assert!(
        failures.is_empty(),
        "recorded counter omissions or invariant failure: {failures:?}"
    );
}

fn selected_and_probe(fixture: &Fixture, audited: bool) -> Value {
    let cfg = config();
    let mut board = replay(fixture.history);
    let attacker = board.side_to_move;
    let kind = classify_move_fast(&board, fixture.attack, attacker);
    board.make_move(fixture.attack);
    let saved = board.clone();
    let mut sequence = vec![fixture.attack];
    let mut tt = TransTable::with_capacity(65_536);
    let mut stats = VctSearchStats::default();
    let mut scratch = VctScratch::default();
    let mut index = None;
    let mut audit = VctAuditLog::default();
    let proved = if audited {
        vct_and_audit(
            &mut board,
            attacker,
            kind,
            13,
            0,
            None,
            cfg.node_budget,
            cfg.jump_three_flags(),
            &mut index,
            &mut sequence,
            &mut tt,
            &mut audit,
            &mut stats,
            &mut scratch,
        )
    } else {
        vct_and(
            &mut board,
            attacker,
            kind,
            13,
            0,
            None,
            cfg.node_budget,
            cfg.jump_three_flags(),
            &mut None,
            &mut index,
            &mut sequence,
            &mut tt,
            &mut stats,
            &mut scratch,
        )
    };
    let expected_history = history_json(&saved);
    let selected_nodes: Vec<_> = audit
        .and_nodes
        .iter()
        .filter(|n| n.get("history") == Some(&expected_history))
        .cloned()
        .collect();
    json!({
        "proved": proved, "sequence": sequence, "stats": stats_json(&stats),
        "restoration": restoration(&board, &saved),
        "sequence_checkpoint_restored": sequence == vec![fixture.attack],
        "selected_and_node": selected_nodes.first(),
        "selected_and_node_count": selected_nodes.len(),
        "audited": audited,
    })
}

fn after_counter_or_probe(fixture: &Fixture) -> Value {
    let cfg = config();
    let mut board = replay(fixture.history);
    let attacker = board.side_to_move;
    board.make_move(fixture.attack);
    board.make_move(fixture.counter);
    let saved = board.clone();
    let mut sequence = Vec::new();
    let mut tt = TransTable::with_capacity(65_536);
    let mut stats = VctSearchStats::default();
    let proved = vct_or(
        &mut board,
        attacker,
        12,
        0,
        None,
        cfg.node_budget,
        cfg.jump_three_flags(),
        &mut None,
        &mut None,
        &mut sequence,
        &mut tt,
        &mut stats,
        &mut VctScratch::default(),
    );
    json!({
        "proved": proved, "sequence": sequence, "stats": stats_json(&stats),
        "restoration": restoration(&board, &saved),
        "sequence_checkpoint_restored": sequence.is_empty(),
    })
}

fn valid_refutation(probe: &Value) -> bool {
    probe["proved"] == false
        && probe["stats"]["hit_stop"] == false
        && probe["stats"]["nodes"].as_u64().unwrap_or(0) > 0
        && probe["restoration"]["all"] == true
        && probe["sequence_checkpoint_restored"] == true
}

#[test]
fn both_selected_attacks_and_actual_counter_positions_are_refuted_without_stops() {
    let mut failures = Vec::new();
    for fixture in fixtures() {
        let plain = selected_and_probe(&fixture, false);
        let audited = selected_and_probe(&fixture, true);
        let reply = after_counter_or_probe(&fixture);
        let passed = valid_refutation(&plain)
            && valid_refutation(&audited)
            && valid_refutation(&reply)
            && audited["selected_and_node_count"] == 1;
        if !passed {
            failures.push(fixture.id);
        }
        emit(
            &fixture,
            "selected_attack_and_reply",
            json!({
                "selected_and": plain, "selected_and_audit": audited,
                "after_counter_or": reply, "vct_calls": 3, "passed": passed,
            }),
        );
    }
    assert!(
        failures.is_empty(),
        "selected attacks not refuted or invariant/budget failure: {failures:?}"
    );
}

#[test]
fn public_root_search_does_not_return_the_certified_bad_attack() {
    let mut failures = Vec::new();
    for fixture in fixtures() {
        let mut board = replay(fixture.history);
        let saved = board.clone();
        let result = search_vct_with_stats(&mut board, &config());
        let bad = result.sequence.as_ref().and_then(|s| s.first()) == Some(&fixture.attack);
        let restored = restoration(&board, &saved);
        let passed = !bad && !result.stats.hit_stop() && restored["all"] == true;
        if !passed {
            failures.push(fixture.id);
        }
        emit(
            &fixture,
            "root_search",
            json!({
                "root": {
                    "sequence": result.sequence, "stats": stats_json(&result.stats),
                    "termination_reason": result.termination_reason(), "restoration": restored,
                    "returns_certified_bad_attack": bad,
                },
                "vct_calls": 1, "passed": passed,
            }),
        );
    }
    assert!(
        failures.is_empty(),
        "root returned refuted attack or hit a stop/invariant failure: {failures:?}"
    );
}

#[derive(Debug, PartialEq, Eq)]
struct SidecarSnapshot {
    synchronized: bool,
    packed_enabled: bool,
    frontier_enabled: bool,
    d4_enabled: bool,
    packed_windows: Vec<Option<u32>>,
    candidates: Vec<Move>,
}

fn sidecar_snapshot(board: &Board, state: &BoardSearchState) -> SidecarSnapshot {
    SidecarSnapshot {
        synchronized: state.is_synchronized(board),
        packed_enabled: state.packed_line_windows_enabled(),
        frontier_enabled: state.candidate_frontier_enabled(),
        d4_enabled: state.d4_hash_enabled(),
        packed_windows: (0..NUM_CELLS)
            .flat_map(|cell| (0..4).map(move |dir| state.packed_line_window(cell, dir)))
            .collect(),
        candidates: state.candidate_moves(board),
    }
}

#[test]
fn product_sidecar_entry_preserves_all_three_counter_refutations_and_caches() {
    const HISTORY_139: [Move; 27] = [
        112, 110, 140, 124, 152, 95, 64, 137, 83, 53, 105, 91, 126, 98, 154, 168, 151, 153, 82, 81,
        109, 123, 138, 169, 185, 139, 107,
    ];
    let [case_68, case_71] = fixtures();
    let cases = [
        Fixture {
            id: "a3_139_g1_p27",
            history: &HISTORY_139,
            attack: 167,
            counter: 106,
        },
        case_68,
        case_71,
    ];
    let cfg = config();
    let mut failures = Vec::new();
    for fixture in cases {
        let saved = replay(fixture.history);
        let mut plain_board = saved.clone();
        let plain = search_vct_with_stats(&mut plain_board, &cfg);
        let plain_restored = restoration(&plain_board, &saved)["all"] == true;
        let plain_passed = !plain.stats.hit_stop()
            && plain.stats.nodes > 0
            && plain_restored
            && plain.sequence.as_ref().and_then(|s| s.first()) != Some(&fixture.attack);
        let mut rows = Vec::new();
        // The pbrain enables both accelerators, but Searcher keeps the frontier
        // off during root VCT and enables it only for the subsequent main search.
        // Cover that product path plus callers retaining an enabled frontier.
        for frontier_enabled in [false, true] {
            let mut board = saved.clone();
            let mut state = BoardSearchState::new();
            state.set_packed_line_windows_enabled(&board, true);
            state.set_candidate_frontier_enabled(&board, frontier_enabled);
            let before = sidecar_snapshot(&board, &state);
            let initialized = before.synchronized
                && before.packed_enabled
                && before.frontier_enabled == frontier_enabled
                && !before.d4_enabled
                && before.packed_windows.iter().all(Option::is_some)
                && before.candidates == saved.candidate_moves();

            // The product wrapper discards stats. Check its exact internal path
            // with a deterministic node cap first: a capped None is not a pass.
            let measured = search_vct_with_stats_internal(&mut board, &cfg, Some(&mut state));
            let measured_board_restored = restoration(&board, &saved)["all"] == true;
            let measured_cache_restored = sidecar_snapshot(&board, &state) == before;
            let measured_passed = !measured.stats.hit_stop()
                && measured.stats.nodes > 0
                && measured.stats.nodes == plain.stats.nodes
                && measured.sequence == plain.sequence
                && measured_board_restored
                && measured_cache_restored;

            // Reuse the restored sidecar without a rebuild through the exact
            // entry called by Searcher. There is no clock deadline in either run.
            let product_sequence = search_vct_with_board_search_state(&mut board, &cfg, &mut state);
            let product_board_restored = restoration(&board, &saved)["all"] == true;
            let product_cache_restored = sidecar_snapshot(&board, &state) == before;
            let passed = plain_passed
                && initialized
                && measured_passed
                && product_sequence == measured.sequence
                && product_board_restored
                && product_cache_restored;
            if !passed {
                failures.push((fixture.id, frontier_enabled));
            }
            rows.push(json!({
                "packed_windows_enabled": true, "frontier_enabled": frontier_enabled,
                "product_root_default": !frontier_enabled, "initialized": initialized,
                "measured_sequence": measured.sequence, "measured_stats": stats_json(&measured.stats),
                "measured_board_restored": measured_board_restored,
                "measured_cache_restored": measured_cache_restored,
                "product_sequence": product_sequence, "product_board_restored": product_board_restored,
                "product_cache_restored": product_cache_restored, "passed": passed,
            }));
        }
        emit(
            &fixture,
            "product_sidecar_root",
            json!({
                "plain_sequence": plain.sequence, "plain_stats": stats_json(&plain.stats),
                "plain_board_restored": plain_restored, "rows": rows, "vct_calls": 5,
            }),
        );
    }
    assert!(
        failures.is_empty(),
        "sidecar counter/stop/restoration failure: {failures:?}"
    );
}
