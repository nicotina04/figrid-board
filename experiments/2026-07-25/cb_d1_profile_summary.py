"""Aggregate paired or A-B-B-A RQ582 profile rows for CB-D1 or CB-TD1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = [
    "eval_state_dirty_list_ns",
    "eval_state_frame_write_ns",
    "eval_state_backup_ns",
    "eval_state_recompute_ns",
    "eval_state_aggregate_ns",
    "eval_state_restore_ns",
    "eval_state_forward_ns",
    "eval_state_push_pop_ns",
    "eval_ns",
    "make_undo_ns",
    "board_make_undo_ns",
    "total_ns",
]


def load(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise ValueError(f"{path}: no profile rows")
    keys = [position_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate profile position")
    return rows


def position_key(row: dict) -> tuple:
    return (
        row["game_id"],
        row.get("seed"),
        row["ply"],
        row["move_count"],
    )


def search_result(row: dict) -> tuple:
    return (
        row["best_move"],
        row["score"],
        row["searched_depth"],
        row["nodes"],
        row["actual_visited_nodes"],
        row["shape"]["main_nodes"],
        row["shape"]["qsearch_nodes"],
    )


def policy(row: dict) -> tuple:
    profile = row["profile"]
    return (
        row["eval"],
        row["depth"],
        row["time_ms_limit"],
        row["node_budget"],
        row["product_defaults"],
        row["directional_delta"],
        row["use_threat_field"],
        row["use_lazy_threat_field"],
        row["use_move_picker"],
        row["use_tail_threat_materialize"],
        row["stress_threat_field"],
        profile["enabled"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--a1")
    parser.add_argument("--b1")
    parser.add_argument("--b2")
    parser.add_argument("--a2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--token-delta", action="store_true")
    args = parser.parse_args()

    paired = args.a is not None or args.b is not None
    abba = any(value is not None for value in (args.a1, args.b1, args.b2, args.a2))
    if paired == abba:
        parser.error("provide exactly --a/--b or --a1/--b1/--b2/--a2")
    if paired:
        if args.a is None or args.b is None:
            parser.error("--a and --b must be provided together")
        arm_rows = [load(Path(args.a)), load(Path(args.b))]
        a_arms = [arm_rows[0]]
        b_arms = [arm_rows[1]]
        arm_names = ["a", "b"]
    else:
        if any(value is None for value in (args.a1, args.b1, args.b2, args.a2)):
            parser.error("--a1, --b1, --b2, and --a2 must all be provided")
        arm_rows = [
            load(Path(args.a1)),
            load(Path(args.b1)),
            load(Path(args.b2)),
            load(Path(args.a2)),
        ]
        a_arms = [arm_rows[0], arm_rows[3]]
        b_arms = [arm_rows[1], arm_rows[2]]
        arm_names = ["a1", "b1", "b2", "a2"]

    if any(len(rows) != len(arm_rows[0]) for rows in arm_rows[1:]):
        raise ValueError("profile arm lengths differ")
    for rows in arm_rows[1:]:
        if any(
            position_key(reference) != position_key(candidate)
            for reference, candidate in zip(arm_rows[0], rows)
        ):
            raise ValueError("profile position keys differ")
    reference_policy = policy(arm_rows[0][0])
    if reference_policy[:5] != ("codebook-quant", 4, None, None, True):
        raise ValueError(
            "profile gate requires codebook-quant, depth 4, no time/node limit, "
            "and product defaults"
        )
    for rows in arm_rows:
        if any(policy(row) != reference_policy for row in rows):
            raise ValueError("profile policy differs within or between arms")
        if any(not row["profile"]["enabled"] for row in rows):
            raise ValueError("search profiling was not enabled")
        if any(row["profile"]["root_vct_calls"] != 0 for row in rows):
            raise ValueError("profile gate requires root VCT OFF")
        if any(
            row["actual_visited_nodes"]
            != row["shape"]["main_nodes"] + row["shape"]["qsearch_nodes"]
            for row in rows
        ):
            raise ValueError("actual visited node accounting mismatch")

    if args.token_delta:
        if not all(
            row.get("directional_delta") for rows in arm_rows for row in rows
        ):
            raise ValueError("CB-TD1 profile requires CB-D1 in both arms")
        if any(row.get("token_delta") for rows in a_arms for row in rows):
            raise ValueError("profile A unexpectedly enables CB-TD1")
        if not all(row.get("token_delta") for rows in b_arms for row in rows):
            raise ValueError("profile B does not enable CB-TD1")

    result_mismatches = 0
    call_mismatches = 0
    mismatch_examples = []
    call_fields = [
        key
        for key in arm_rows[0][0]["profile"]
        if key.endswith("_calls")
    ]
    for rows in zip(*arm_rows):
        if any(search_result(row) != search_result(rows[0]) for row in rows[1:]):
            result_mismatches += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(
                    {
                        "position": position_key(rows[0]),
                        "arms": {
                            name: search_result(row)
                            for name, row in zip(arm_names, rows)
                        },
                    }
                )
        for field in call_fields:
            if any(
                row["profile"][field] != rows[0]["profile"][field]
                for row in rows[1:]
            ):
                call_mismatches += 1
                break

    a = {
        field: sum(row["profile"][field] for rows in a_arms for row in rows)
        for field in FIELDS
    }
    b = {
        field: sum(row["profile"][field] for rows in b_arms for row in rows)
        for field in FIELDS
    }
    wall_a = sum(row["elapsed_ns"] for rows in a_arms for row in rows)
    wall_b = sum(row["elapsed_ns"] for rows in b_arms for row in rows)
    bucket_ratios = {
        field: b[field] / a[field] if a[field] else None for field in FIELDS
    }
    hot_gate = (
        bucket_ratios["eval_ns"] <= 1.01
        and bucket_ratios["eval_state_push_pop_ns"] <= 1.01
    )
    result = {
        "format": (
            "cb-token-delta-profile-summary-v2"
            if args.token_delta
            else "cb-d1-profile-summary-v2"
        ),
        "design": "abba" if abba else "paired",
        "arm_order": arm_names,
        "positions": len(arm_rows[0]),
        "measurements": sum(len(rows) for rows in arm_rows),
        "wall_a_ns": wall_a,
        "wall_b_ns": wall_b,
        "wall_ratio_b_over_a": wall_b / wall_a,
        "repeatability": (
            {
                "a2_over_a1_wall": (
                    sum(row["elapsed_ns"] for row in arm_rows[3])
                    / sum(row["elapsed_ns"] for row in arm_rows[0])
                ),
                "b2_over_b1_wall": (
                    sum(row["elapsed_ns"] for row in arm_rows[2])
                    / sum(row["elapsed_ns"] for row in arm_rows[1])
                ),
            }
            if abba
            else None
        ),
        "result_mismatches": result_mismatches,
        "profile_call_mismatches": call_mismatches,
        "mismatch_examples": mismatch_examples,
        "buckets": {
            field: {
                "a_ns": a[field],
                "b_ns": b[field],
                "ratio_b_over_a": bucket_ratios[field],
                "a_wall_share": a[field] / wall_a,
                "b_wall_share": b[field] / wall_b,
            }
            for field in FIELDS
        },
        "gate": {
            "result_mismatch_zero": result_mismatches == 0,
            "profile_call_mismatch_zero": call_mismatches == 0,
            "eval_ratio_lte_1_01": bucket_ratios["eval_ns"] <= 1.01,
            "push_pop_ratio_lte_1_01": (
                bucket_ratios["eval_state_push_pop_ns"] <= 1.01
            ),
            "pass": result_mismatches == 0 and call_mismatches == 0 and hot_gate,
        },
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
