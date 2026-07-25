"""Analyze same-binary CB-D1 A-B-B-A search-profile runs by game block."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


Key = tuple[str, str, int, str, int | None]


def load(path: Path) -> tuple[dict[str, Any] | None, dict[Key, dict[str, Any]]]:
    seal: dict[str, Any] | None = None
    rows: dict[Key, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        row = json.loads(raw)
        if row.get("kind") == "seal":
            if seal is not None:
                raise ValueError(f"{path}: duplicate seal")
            seal = row
            continue
        key = (
            json.dumps(row["game_id"], sort_keys=True, ensure_ascii=False),
            json.dumps(row.get("seed"), sort_keys=True, ensure_ascii=False),
            int(row["ply"]),
            json.dumps(row.get("actual_move"), sort_keys=True, ensure_ascii=False),
            int(row["root_zobrist"]) if "root_zobrist" in row else None,
        )
        if key in rows:
            raise ValueError(f"{path}: duplicate position {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"{path}: no positions")
    return seal, rows


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return ordered[index]


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
    parser.add_argument("--wall-gate", type=float, default=0.99)
    parser.add_argument("--timed", action="store_true")
    args = parser.parse_args()

    loaded = [
        load(Path(args.a1)),
        load(Path(args.b1)),
        load(Path(args.b2)),
        load(Path(args.a2)),
    ]
    seals = [item[0] for item in loaded]
    arms = [item[1] for item in loaded]
    if any(seal is not None for seal in seals):
        if any(seal is None for seal in seals):
            raise ValueError("ABBA seal presence differs")
        assert all(seal is not None for seal in seals)
        stable_seal_keys = (
            "format",
            "baseline_commit",
            "codebook_sha256",
            "input",
            "flat_weights",
            "depth",
            "time_ms",
            "root_vct_enabled",
            "rules",
            "eval",
            "white_root_order",
            "packed_windows",
            "candidate_frontier",
        )
        reference = seals[0]
        for arm_index, seal in enumerate(seals[1:], start=2):
            for field in stable_seal_keys:
                if seal[field] != reference[field]:
                    raise ValueError(
                        f"arm {arm_index}: seal field {field} differs: "
                        f"{seal[field]!r} != {reference[field]!r}"
                    )
        if seals[0]["directional_delta"] or seals[3]["directional_delta"]:
            raise ValueError("A seal unexpectedly enables CB-D1")
        if not seals[1]["directional_delta"] or not seals[2]["directional_delta"]:
            raise ValueError("B seal does not enable CB-D1")
    keys = set(arms[0])
    if any(set(arm) != keys for arm in arms[1:]):
        raise ValueError("ABBA position keys differ")

    a1, b1, b2, a2 = arms
    for key in keys:
        rows = [arm[key] for arm in arms]
        if any(row.get("eval") != "codebook-quant" for row in rows):
            raise ValueError(f"{key}: non-codebook arm")
        if a1[key].get("directional_delta") or a2[key].get("directional_delta"):
            raise ValueError(f"{key}: A arm unexpectedly enables CB-D1")
        if not b1[key].get("directional_delta") or not b2[key].get(
            "directional_delta"
        ):
            raise ValueError(f"{key}: B arm does not enable CB-D1")
        if seals[0] is not None:
            for row, seal in zip(rows, seals, strict=True):
                if row.get("requested_depth") != seal["depth"]:
                    raise ValueError(f"{key}: requested depth differs from seal")
                if row.get("time_ms") != seal["time_ms"]:
                    raise ValueError(f"{key}: time limit differs from seal")
                if row.get("root_vct_enabled") != seal["root_vct_enabled"]:
                    raise ValueError(f"{key}: root VCT state differs from seal")

    by_game: dict[str, dict[str, float]] = defaultdict(
        lambda: {"a_ns": 0.0, "b_ns": 0.0, "a_nodes": 0.0, "b_nodes": 0.0}
    )
    decision_mismatches = 0
    result_mismatches = 0
    node_mismatches = 0
    mismatch_examples: list[dict[str, Any]] = []
    a_depths: list[float] = []
    b_depths: list[float] = []

    def actual_nodes(row: dict[str, Any]) -> int:
        return int(row["shape"]["main_nodes"]) + int(row["shape"]["qsearch_nodes"])

    for key in sorted(keys):
        game = key[0]
        ar1, br1, br2, ar2 = a1[key], b1[key], b2[key], a2[key]
        by_game[game]["a_ns"] += (ar1["elapsed_ns"] + ar2["elapsed_ns"]) / 2.0
        by_game[game]["b_ns"] += (br1["elapsed_ns"] + br2["elapsed_ns"]) / 2.0
        by_game[game]["a_nodes"] += (
            actual_nodes(ar1) + actual_nodes(ar2)
        ) / 2.0
        by_game[game]["b_nodes"] += (
            actual_nodes(br1) + actual_nodes(br2)
        ) / 2.0
        a_depths.append((ar1["searched_depth"] + ar2["searched_depth"]) / 2.0)
        b_depths.append((br1["searched_depth"] + br2["searched_depth"]) / 2.0)

        rows = (ar1, br1, br2, ar2)
        decision_mismatches += int(
            any(row["best_move"] != ar1["best_move"] for row in rows)
        )
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
                    row["nodes"],
                    row["shape"]["main_nodes"],
                    row["shape"]["qsearch_nodes"],
                )
                != (
                    ar1["nodes"],
                    ar1["shape"]["main_nodes"],
                    ar1["shape"]["qsearch_nodes"],
                )
                for row in rows
            )
        )
        if any(
            (
                row["best_move"],
                row["score"],
                row["searched_depth"],
                row["nodes"],
                row["shape"]["main_nodes"],
                row["shape"]["qsearch_nodes"],
            )
            != (
                ar1["best_move"],
                ar1["score"],
                ar1["searched_depth"],
                ar1["nodes"],
                ar1["shape"]["main_nodes"],
                ar1["shape"]["qsearch_nodes"],
            )
            for row in rows
        ) and len(mismatch_examples) < 10:
            mismatch_examples.append(
                {
                    "key": {
                        "game_id": json.loads(game),
                        "seed": json.loads(key[1]),
                        "ply": key[2],
                        "actual_move": json.loads(key[3]),
                        "root_zobrist": key[4],
                    },
                    "arms": [
                        {
                            "best_move": row["best_move"],
                            "score": row["score"],
                            "depth": row["searched_depth"],
                            "nodes": row["nodes"],
                            "main": row["shape"]["main_nodes"],
                            "qsearch": row["shape"]["qsearch_nodes"],
                            "elapsed_ns": row["elapsed_ns"],
                        }
                        for row in rows
                    ],
                }
            )

    games = sorted(by_game)
    a_ns = sum(by_game[game]["a_ns"] for game in games)
    b_ns = sum(by_game[game]["b_ns"] for game in games)
    a_nodes = sum(by_game[game]["a_nodes"] for game in games)
    b_nodes = sum(by_game[game]["b_nodes"] for game in games)
    wall_ratio = b_ns / a_ns
    nps_ratio = (b_nodes / (b_ns / 1e9)) / (a_nodes / (a_ns / 1e9))

    rng = random.Random(0xCBD1)
    ratios = []
    for _ in range(args.bootstrap):
        sample = [rng.choice(games) for _ in games]
        sample_a = sum(by_game[game]["a_ns"] for game in sample)
        sample_b = sum(by_game[game]["b_ns"] for game in sample)
        ratios.append(sample_b / sample_a)
    upper = percentile(ratios, 0.95)

    exact = decision_mismatches == result_mismatches == node_mismatches == 0
    wall_pass = wall_ratio <= args.wall_gate and upper < 1.0
    timed_pass = (
        nps_ratio > 1.0
        and b_nodes / a_nodes >= 1.0
        and median(b_depths) >= median(a_depths)
    )
    result = {
        "format": "cb-d1-abba-v2" if seals[0] is not None else "cb-d1-abba-v1",
        "mode": "timed" if args.timed else "fixed-depth",
        "seal": seals[0],
        "games": len(games),
        "positions": len(keys),
        "wall_a_ns": a_ns,
        "wall_b_ns": b_ns,
        "wall_ratio_b_over_a": wall_ratio,
        "wall_saving": 1.0 - wall_ratio,
        "one_sided_95_bootstrap_upper": upper,
        "nodes_a": a_nodes,
        "nodes_b": b_nodes,
        "node_ratio_b_over_a": b_nodes / a_nodes,
        "actual_visited_nps_ratio_b_over_a": nps_ratio,
        "searched_depth_p50_a": median(a_depths),
        "searched_depth_p50_b": median(b_depths),
        "decision_mismatches": decision_mismatches,
        "result_mismatches": result_mismatches,
        "node_mismatches": node_mismatches,
        "mismatch_examples": mismatch_examples,
        "gate": {
            "wall_threshold": args.wall_gate,
            "correctness_mismatch_zero": exact,
            "wall_ratio_lte_threshold": wall_ratio <= args.wall_gate,
            "one_sided_95_upper_lt_1": upper < 1.0,
            "nps_ratio_gt_1": nps_ratio > 1.0,
            "node_ratio_gte_1": b_nodes / a_nodes >= 1.0,
            "median_depth_non_regression": median(b_depths) >= median(a_depths),
            "pass": timed_pass if args.timed else exact and wall_pass,
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
