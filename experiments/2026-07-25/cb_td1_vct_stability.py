"""Audit repeated product VCT-ON ABBA runs for deadline instability."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


Key = tuple[str, str, int, str, int | None]
COMMON_SEAL_FIELDS = (
    "format",
    "baseline_commit",
    "codebook_sha256",
    "input",
    "rules",
    "eval",
    "flat_weights",
    "depth",
    "time_ms",
    "root_vct_enabled",
    "white_root_order",
    "packed_windows",
    "candidate_frontier",
)


def load(path: Path) -> tuple[dict[str, Any], dict[Key, dict[str, Any]]]:
    seals: list[dict[str, Any]] = []
    rows: dict[Key, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        row = json.loads(raw)
        if row.get("kind") == "seal":
            seals.append(row)
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
    if len(seals) != 1:
        raise ValueError(f"{path}: expected exactly one seal, got {len(seals)}")
    seal = seals[0]
    if not seal.get("root_vct_enabled"):
        raise ValueError(f"{path}: root VCT is not enabled")
    if not rows:
        raise ValueError(f"{path}: no positions")
    return seal, rows


def validate_arms(
    name: str,
    loaded: list[tuple[dict[str, Any], dict[Key, dict[str, Any]]]],
) -> list[dict[Key, dict[str, Any]]]:
    seals = [seal for seal, _ in loaded]
    expected_token_delta = [False, True, True, False]
    for arm, (seal, expected_token) in enumerate(
        zip(seals, expected_token_delta, strict=True)
    ):
        missing = [field for field in COMMON_SEAL_FIELDS if field not in seal]
        if missing:
            raise ValueError(f"{name}: arm {arm} seal is missing {missing}")
        if seal["format"] != "cb-token-delta-ab-v1":
            raise ValueError(f"{name}: arm {arm} has unexpected seal format")
        if seal.get("directional_delta") is not True:
            raise ValueError(f"{name}: arm {arm} does not enable directional delta")
        if seal.get("token_delta") is not expected_token:
            raise ValueError(
                f"{name}: arm {arm} token_delta={seal.get('token_delta')!r}, "
                f"expected {expected_token}"
            )
        for field in COMMON_SEAL_FIELDS:
            if seal.get(field) != seals[0].get(field):
                raise ValueError(f"{name}: seal field {field!r} differs across arms")

    arms = [rows for _, rows in loaded]
    for arm, (rows, expected_token) in enumerate(
        zip(arms, expected_token_delta, strict=True)
    ):
        for key, row in rows.items():
            if row.get("root_vct_enabled") is not True:
                raise ValueError(f"{name}: arm {arm} row {key} has VCT disabled")
            if row.get("directional_delta") is not True:
                raise ValueError(
                    f"{name}: arm {arm} row {key} lacks directional delta"
                )
            if row.get("token_delta") is not expected_token:
                raise ValueError(
                    f"{name}: arm {arm} row {key} token_delta mismatch"
                )
    return arms


def signature(row: dict[str, Any]) -> tuple[Any, ...]:
    shape = row["shape"]
    return (
        json.dumps(row["best_move"], sort_keys=True),
        int(row["score"]),
        int(row["searched_depth"]),
        int(row["nodes"]),
        int(shape["main_nodes"]),
        int(shape["qsearch_nodes"]),
        int(shape["tt_probes"]),
        int(shape["tt_hits"]),
        int(shape["tt_cutoffs"]),
    )


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return ordered[index]


def key_json(key: Key) -> dict[str, Any]:
    return {
        "game_id": json.loads(key[0]),
        "seed": json.loads(key[1]),
        "ply": key[2],
        "actual_move": json.loads(key[3]),
        "root_zobrist": key[4],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        nargs=5,
        action="append",
        metavar=("NAME", "A1", "B1", "B2", "A2"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    args = parser.parse_args()

    groups = []
    reference_keys: set[Key] | None = None
    for name, a1_path, b1_path, b2_path, a2_path in args.group:
        loaded = [load(Path(path)) for path in (a1_path, b1_path, b2_path, a2_path)]
        arms = validate_arms(name, loaded)
        keys = set(arms[0])
        if any(set(arm) != keys for arm in arms[1:]):
            raise ValueError(f"{name}: ABBA position keys differ")
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError(f"{name}: position cohort differs")
        groups.append((name, arms))
    assert reference_keys is not None
    keys = reference_keys

    group_results = []
    direct_all: dict[Key, list[tuple[Any, ...]]] = defaultdict(list)
    journal_all: dict[Key, list[tuple[Any, ...]]] = defaultdict(list)
    for name, (a1, b1, b2, a2) in groups:
        direct_stable = 0
        journal_stable = 0
        decision_direct_stable = 0
        decision_journal_stable = 0
        unstable: set[Key] = set()
        stable_cross_divergence = 0
        for key in keys:
            a_signatures = [signature(a1[key]), signature(a2[key])]
            b_signatures = [signature(b1[key]), signature(b2[key])]
            direct_all[key].extend(a_signatures)
            journal_all[key].extend(b_signatures)
            a_stable = a_signatures[0] == a_signatures[1]
            b_stable = b_signatures[0] == b_signatures[1]
            direct_stable += int(a_stable)
            journal_stable += int(b_stable)
            decision_direct_stable += int(
                a_signatures[0][0] == a_signatures[1][0]
            )
            decision_journal_stable += int(
                b_signatures[0][0] == b_signatures[1][0]
            )
            if not a_stable or not b_stable:
                unstable.add(key)
            elif a_signatures[0] != b_signatures[0]:
                stable_cross_divergence += 1
        group_results.append(
            {
                "name": name,
                "positions": len(keys),
                "direct_repeat_stable": direct_stable,
                "journal_repeat_stable": journal_stable,
                "direct_decision_repeat_stable": decision_direct_stable,
                "journal_decision_repeat_stable": decision_journal_stable,
                "union_unstable": len(unstable),
                "stable_cross_variant_divergence": stable_cross_divergence,
            }
        )

    union_unstable = {
        key
        for key in keys
        if len(set(direct_all[key])) != 1 or len(set(journal_all[key])) != 1
    }
    stable_cross_divergence = sum(
        1
        for key in keys - union_unstable
        if direct_all[key][0] != journal_all[key][0]
    )

    by_game: dict[str, dict[str, float]] = defaultdict(
        lambda: {"a_ns": 0.0, "b_ns": 0.0}
    )
    group_wall = []
    for name, (a1, b1, b2, a2) in groups:
        a_ns = 0.0
        b_ns = 0.0
        for key in keys - union_unstable:
            a_value = (a1[key]["elapsed_ns"] + a2[key]["elapsed_ns"]) / 2.0
            b_value = (b1[key]["elapsed_ns"] + b2[key]["elapsed_ns"]) / 2.0
            a_ns += a_value
            b_ns += b_value
            by_game[key[0]]["a_ns"] += a_value
            by_game[key[0]]["b_ns"] += b_value
        group_wall.append(
            {
                "name": name,
                "a_ns": a_ns,
                "b_ns": b_ns,
                "ratio_b_over_a": b_ns / a_ns,
            }
        )

    a_ns = sum(value["a_ns"] for value in by_game.values())
    b_ns = sum(value["b_ns"] for value in by_game.values())
    games = sorted(by_game)
    rng = random.Random(0xCBD1_5A81)
    ratios = []
    for _ in range(args.bootstrap):
        sample = [rng.choice(games) for _ in games]
        sample_a = sum(by_game[game]["a_ns"] for game in sample)
        sample_b = sum(by_game[game]["b_ns"] for game in sample)
        ratios.append(sample_b / sample_a)
    upper = percentile(ratios, 0.95)

    unstable_details = []
    for key in sorted(union_unstable):
        unstable_details.append(
            {
                "position": key_json(key),
                "direct_unique_signatures": len(set(direct_all[key])),
                "journal_unique_signatures": len(set(journal_all[key])),
                "direct_unique_decisions": len({value[0] for value in direct_all[key]}),
                "journal_unique_decisions": len(
                    {value[0] for value in journal_all[key]}
                ),
            }
        )

    point = b_ns / a_ns
    result = {
        "format": "cb-token-delta-vct-stability-v1",
        "groups": group_results,
        "positions": len(keys),
        "measurements_per_position": len(groups) * 4,
        "direct_all_repeat_stable": sum(
            len(set(direct_all[key])) == 1 for key in keys
        ),
        "journal_all_repeat_stable": sum(
            len(set(journal_all[key])) == 1 for key in keys
        ),
        "union_unstable": len(union_unstable),
        "unstable_positions": unstable_details,
        "stable_positions": len(keys - union_unstable),
        "stable_cross_variant_divergence": stable_cross_divergence,
        "symmetric_censored_wall": {
            "groups": group_wall,
            "a_ns": a_ns,
            "b_ns": b_ns,
            "ratio_b_over_a": point,
            "one_sided_95_game_block_bootstrap_upper": upper,
        },
        "gate": {
            "stable_cross_variant_divergence_zero": stable_cross_divergence == 0,
            "wall_ratio_lte_1_005": point <= 1.005,
            "bootstrap_upper_lt_1_01": upper < 1.01,
            "pass": (
                stable_cross_divergence == 0
                and point <= 1.005
                and upper < 1.01
            ),
        },
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
