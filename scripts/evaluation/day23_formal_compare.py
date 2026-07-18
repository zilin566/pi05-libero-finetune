from __future__ import annotations

import csv
import json
import math
from pathlib import Path


OUTPUT_ROOT = Path("experiments/day23_eval/formal_compare")

POLICIES = {
    "official": {
        "root": OUTPUT_ROOT / "official",
        "filename": "ckptofficial_pi05_libero_task5_state{state}.jsonl",
    },
    "old_5k": {
        "root": OUTPUT_ROOT / "old_5k",
        "filename": "ckptold_5k_task5_state{state}.jsonl",
    },
    "second_round_13999": {
        "root": Path(
            "experiments/day23_eval/rollout_candidate_13999"
        ),
        "filename": "ckptday22_s13999_task5_state{state}.jsonl",
    },
}


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")

    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total

    centre = (
        proportion + z**2 / (2 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )

    return max(0.0, centre - margin), min(1.0, centre + margin)


def read_result(policy: str, state: int) -> dict:
    specification = POLICIES[policy]

    path = (
        specification["root"]
        / f"task5_state{state}"
        / specification["filename"].format(state=state)
    )

    if not path.exists():
        return {
            "status": "MISSING",
            "success": None,
            "eef_movement_l2": None,
            "path": str(path),
        }

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    episode_end = next(
        (
            record
            for record in reversed(records)
            if record.get("event") == "episode_end"
        ),
        None,
    )

    if episode_end is None:
        return {
            "status": "NO_EPISODE_END",
            "success": None,
            "eef_movement_l2": None,
            "path": str(path),
        }

    success = bool(episode_end.get("success", False))

    return {
        "status": "SUCCESS" if success else "FAIL",
        "success": success,
        "eef_movement_l2": episode_end.get("eef_movement_l2"),
        "path": str(path),
    }


def main() -> None:
    paired_rows = []
    all_results: dict[str, list[dict]] = {
        policy: [] for policy in POLICIES
    }

    for state in range(10):
        row = {"state": state}

        for policy in POLICIES:
            result = read_result(policy, state)
            all_results[policy].append(result)

            row[policy] = result["status"]
            row[f"{policy}_eef_movement_l2"] = result[
                "eef_movement_l2"
            ]

        paired_rows.append(row)

    paired_csv = OUTPUT_ROOT / "task5_paired_states.csv"

    with paired_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(paired_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(paired_rows)

    summaries = []

    for policy, results in all_results.items():
        completed = [
            result
            for result in results
            if result["success"] is not None
        ]

        successes = sum(
            int(result["success"]) for result in completed
        )
        total = len(completed)
        lower, upper = wilson_interval(successes, total)

        summaries.append({
            "policy": policy,
            "successes": successes,
            "total": total,
            "success_rate": successes / total if total else None,
            "wilson_95_lower": lower,
            "wilson_95_upper": upper,
        })

    summary_csv = OUTPUT_ROOT / "task5_policy_summary.csv"

    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summaries)

    result_json = OUTPUT_ROOT / "task5_formal_results.json"
    result_json.write_text(
        json.dumps(
            {
                "protocol": {
                    "suite": "libero_10",
                    "task_id": 5,
                    "initial_states": list(range(10)),
                    "environment_seed": 7,
                    "replan_steps": 5,
                    "max_policy_steps": 520,
                },
                "summaries": summaries,
                "paired_states": paired_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=== Formal Task 5 Results ===")
    for item in summaries:
        print(
            f"{item['policy']:22s} "
            f"{item['successes']}/{item['total']} "
            f"CI=[{item['wilson_95_lower']:.3f}, "
            f"{item['wilson_95_upper']:.3f}]"
        )

    print()
    print("=== Paired States ===")
    print("state,official,old_5k,second_round_13999")
    for row in paired_rows:
        print(
            f"{row['state']},"
            f"{row['official']},"
            f"{row['old_5k']},"
            f"{row['second_round_13999']}"
        )

    print()
    print("Summary CSV:", summary_csv)
    print("Paired CSV:", paired_csv)
    print("JSON:", result_json)


if __name__ == "__main__":
    main()
