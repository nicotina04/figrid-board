"""Aggregate paired RQ582 profile rows for CB-D1 or CB-TD1."""

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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--token-delta", action="store_true")
    args = parser.parse_args()
    a_rows = load(Path(args.a))
    b_rows = load(Path(args.b))
    if len(a_rows) != len(b_rows):
        raise ValueError("profile arm lengths differ")
    if any(
        (a["game_id"], a["ply"]) != (b["game_id"], b["ply"])
        for a, b in zip(a_rows, b_rows)
    ):
        raise ValueError("profile position keys differ")
    if args.token_delta:
        if not all(row.get("directional_delta") for row in a_rows + b_rows):
            raise ValueError("CB-TD1 profile requires CB-D1 in both arms")
        if any(row.get("token_delta") for row in a_rows):
            raise ValueError("profile A unexpectedly enables CB-TD1")
        if not all(row.get("token_delta") for row in b_rows):
            raise ValueError("profile B does not enable CB-TD1")

    a = {field: sum(row["profile"][field] for row in a_rows) for field in FIELDS}
    b = {field: sum(row["profile"][field] for row in b_rows) for field in FIELDS}
    wall_a = sum(row["elapsed_ns"] for row in a_rows)
    wall_b = sum(row["elapsed_ns"] for row in b_rows)
    result = {
        "format": (
            "cb-token-delta-profile-summary-v1"
            if args.token_delta
            else "cb-d1-profile-summary-v1"
        ),
        "positions": len(a_rows),
        "wall_a_ns": wall_a,
        "wall_b_ns": wall_b,
        "wall_ratio_b_over_a": wall_b / wall_a,
        "buckets": {
            field: {
                "a_ns": a[field],
                "b_ns": b[field],
                "ratio_b_over_a": b[field] / a[field] if a[field] else None,
                "a_wall_share": a[field] / wall_a,
                "b_wall_share": b[field] / wall_b,
            }
            for field in FIELDS
        },
    }
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
