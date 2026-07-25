#!/usr/bin/env python3
"""Analyze 0.8.2 release-stack same-binary A-B-B-A runs by game block."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(
    path: Path,
) -> tuple[
    dict[str, Any],
    dict[tuple[int, int], dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    seal = None
    positions: dict[tuple[int, int], dict[str, Any]] = {}
    games: dict[int, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        row = json.loads(raw)
        if row["kind"] == "seal":
            seal = row
        elif row["kind"] == "position":
            positions[(int(row["game_id"]), int(row["ply"]))] = row
        elif row["kind"] == "game":
            games[int(row["game_id"])] = row
    if seal is None:
        raise ValueError(f"{path}: missing seal")
    return seal, positions, games


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(q * len(ordered))))]


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", required=True)
    parser.add_argument("--b1", required=True)
    parser.add_argument("--b2", required=True)
    parser.add_argument("--a2", required=True)
    parser.add_argument("--output")
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--wall-gate", type=float, default=0.995)
    parser.add_argument("--timed", action="store_true")
    args = parser.parse_args()

    a1_seal, a1, a1_games = load(Path(args.a1))
    b1_seal, b1, b1_games = load(Path(args.b1))
    b2_seal, b2, b2_games = load(Path(args.b2))
    a2_seal, a2, a2_games = load(Path(args.a2))
    keys = set(a1)
    if not keys or not (keys == set(b1) == set(b2) == set(a2)):
        raise ValueError("ABBA position keys are empty or differ")
    game_keys = set(a1_games)
    if not game_keys or not (
        game_keys == set(b1_games) == set(b2_games) == set(a2_games)
    ):
        raise ValueError("ABBA game keys are empty or differ")
    if not b1_seal.get("candidate_frontier") or not b2_seal.get(
        "candidate_frontier"
    ):
        raise ValueError("B arms are not candidate-frontier")
    if a1_seal.get("candidate_frontier") or a2_seal.get("candidate_frontier"):
        raise ValueError("A arms unexpectedly enable candidate-frontier")
    policy = a1_seal.get("product_policy")
    if any(
        seal.get("product_policy") != policy
        for seal in (b1_seal, b2_seal, a2_seal)
    ):
        raise ValueError("product policy differs across arms")
    if any(
        not seal.get("packed_windows")
        for seal in (a1_seal, b1_seal, b2_seal, a2_seal)
    ):
        raise ValueError("all release-stack arms must enable packed windows")

    by_game: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "a_search_ns": 0.0,
            "b_search_ns": 0.0,
            "a_enable_ns": 0.0,
            "b_enable_ns": 0.0,
            "a_nodes": 0.0,
            "b_nodes": 0.0,
        }
    )
    decision_mismatches = 0
    result_mismatches = 0
    node_mismatches = 0
    a_depths: list[float] = []
    b_depths: list[float] = []

    for key in sorted(keys):
        game_id, _ = key
        ar1, br1, br2, ar2 = a1[key], b1[key], b2[key], a2[key]
        by_game[game_id]["a_search_ns"] += (
            ar1["elapsed_ns"] + ar2["elapsed_ns"]
        ) / 2.0
        by_game[game_id]["b_search_ns"] += (
            br1["elapsed_ns"] + br2["elapsed_ns"]
        ) / 2.0
        by_game[game_id]["a_nodes"] += (
            ar1["actual_main_nodes"]
            + ar1["actual_qsearch_nodes"]
            + ar2["actual_main_nodes"]
            + ar2["actual_qsearch_nodes"]
        ) / 2.0
        by_game[game_id]["b_nodes"] += (
            br1["actual_main_nodes"]
            + br1["actual_qsearch_nodes"]
            + br2["actual_main_nodes"]
            + br2["actual_qsearch_nodes"]
        ) / 2.0
        a_depths.append((ar1["searched_depth"] + ar2["searched_depth"]) / 2.0)
        b_depths.append((br1["searched_depth"] + br2["searched_depth"]) / 2.0)
        rows = (ar1, br1, br2, ar2)
        decision_mismatches += int(any(row["best_move"] != ar1["best_move"] for row in rows))
        result_mismatches += int(
            any(
                (row["best_move"], row["score"], row["searched_depth"])
                != (ar1["best_move"], ar1["score"], ar1["searched_depth"])
                for row in rows
            )
        )
        node_mismatches += int(
            any(
                (
                    row["completed_nodes"],
                    row["actual_main_nodes"],
                    row["actual_qsearch_nodes"],
                )
                != (
                    ar1["completed_nodes"],
                    ar1["actual_main_nodes"],
                    ar1["actual_qsearch_nodes"],
                )
                for row in rows
            )
        )

    for game_id in game_keys:
        by_game[game_id]["a_enable_ns"] = (
            a1_games[game_id]["enable_ns"] + a2_games[game_id]["enable_ns"]
        ) / 2.0
        by_game[game_id]["b_enable_ns"] = (
            b1_games[game_id]["enable_ns"] + b2_games[game_id]["enable_ns"]
        ) / 2.0

    games = sorted(by_game)
    a_search = sum(by_game[game]["a_search_ns"] for game in games)
    b_search = sum(by_game[game]["b_search_ns"] for game in games)
    a_total = a_search + sum(by_game[game]["a_enable_ns"] for game in games)
    b_total = b_search + sum(by_game[game]["b_enable_ns"] for game in games)
    a_nodes = sum(by_game[game]["a_nodes"] for game in games)
    b_nodes = sum(by_game[game]["b_nodes"] for game in games)
    nps_a = a_nodes / (a_search / 1_000_000_000.0)
    nps_b = b_nodes / (b_search / 1_000_000_000.0)

    rng = random.Random(0xDFA3)
    ratios = []
    for _ in range(args.bootstrap):
        sample = [rng.choice(games) for _ in games]
        sample_a = sum(
            by_game[game]["a_search_ns"] + by_game[game]["a_enable_ns"]
            for game in sample
        )
        sample_b = sum(
            by_game[game]["b_search_ns"] + by_game[game]["b_enable_ns"]
            for game in sample
        )
        ratios.append(sample_b / sample_a)
    upper = percentile(ratios, 0.95)
    wall_ratio = b_total / a_total
    exact = decision_mismatches == result_mismatches == node_mismatches == 0
    wall_gate_pass = wall_ratio <= args.wall_gate and upper < 1.0
    timed_gate_pass = (
        nps_b / nps_a > 1.0
        and b_nodes / a_nodes >= 1.0
        and median(b_depths) >= median(a_depths)
    )
    result = {
        "format": "dp-release-stack-abba-v1",
        "product_policy": policy,
        "games": len(games),
        "positions": len(keys),
        "search_wall_a_ns": a_search,
        "search_wall_b_ns": b_search,
        "product_total_a_ns": a_total,
        "product_total_b_ns": b_total,
        "wall_ratio_b_over_a": wall_ratio,
        "wall_saving": 1.0 - wall_ratio,
        "one_sided_95_bootstrap_upper": upper,
        "nodes_a": a_nodes,
        "nodes_b": b_nodes,
        "node_ratio_b_over_a": b_nodes / a_nodes,
        "actual_visited_nps_ratio_b_over_a": nps_b / nps_a,
        "searched_depth_p50_a": median(a_depths),
        "searched_depth_p50_b": median(b_depths),
        "searched_depth_mean_a": sum(a_depths) / len(a_depths),
        "searched_depth_mean_b": sum(b_depths) / len(b_depths),
        "decision_mismatches": decision_mismatches,
        "result_mismatches": result_mismatches,
        "node_mismatches": node_mismatches,
        "release_stack_gate": {
            "mode": "timed" if args.timed else "fixed-depth",
            "wall_threshold": args.wall_gate,
            "correctness_mismatch_zero": exact,
            "wall_ratio_lte_threshold": wall_ratio <= args.wall_gate,
            "one_sided_95_upper_lt_1": upper < 1.0,
            "nps_ratio_gt_1": nps_b / nps_a > 1.0,
            "node_ratio_gte_1": b_nodes / a_nodes >= 1.0,
            "median_depth_non_regression": median(b_depths) >= median(a_depths),
            "pass": timed_gate_pass if args.timed else exact and wall_gate_pass,
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
