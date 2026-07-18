from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("experiments/day18_action_l2")


def load(name: str):
    npz = np.load(ROOT / f"{name}_predictions.npz")
    metrics = json.loads(
        (ROOT / f"{name}_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    return npz, metrics


norm_data, norm_metrics = load("norm")
no_norm_data, no_norm_metrics = load("no_norm")

if not np.array_equal(
    norm_data["episode_ids"],
    no_norm_data["episode_ids"],
):
    raise RuntimeError("两组 episode 不一致")

if not np.array_equal(
    norm_data["frame_ids"],
    no_norm_data["frame_ids"],
):
    raise RuntimeError("两组 frame 不一致")

if not np.allclose(
    norm_data["targets"],
    no_norm_data["targets"],
):
    raise RuntimeError("两组 ground truth 不一致")

summary = {
    "clean_norm": norm_metrics,
    "clean_no_norm": no_norm_metrics,
}

(ROOT / "action_l2_comparison.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

headers = [
    "Metric",
    "Clean + Norm",
    "Clean + No Norm",
]

rows = [
    [
        "Raw-space Action L2",
        norm_metrics["raw_action_l2_mean"],
        no_norm_metrics["raw_action_l2_mean"],
    ],
    [
        "Raw-space RMSE",
        norm_metrics["raw_action_rmse"],
        no_norm_metrics["raw_action_rmse"],
    ],
    [
        "Raw-space MAE",
        norm_metrics["raw_action_mae"],
        no_norm_metrics["raw_action_mae"],
    ],
    [
        "Mean inference ms",
        norm_metrics["mean_inference_ms"],
        no_norm_metrics["mean_inference_ms"],
    ],
]

print(
    f"{headers[0]:<28}"
    f"{headers[1]:>20}"
    f"{headers[2]:>22}"
)

for metric, norm_value, no_norm_value in rows:
    print(
        f"{metric:<28}"
        f"{norm_value:>20.6f}"
        f"{no_norm_value:>22.6f}"
    )

print("\n===== Per-dimension RMSE =====")

for dimension, (norm_value, no_norm_value) in enumerate(
    zip(
        norm_metrics["per_dim_rmse"],
        no_norm_metrics["per_dim_rmse"],
        strict=True,
    )
):
    name = (
        "gripper"
        if dimension == 6
        else f"action_dim_{dimension}"
    )

    print(
        f"{name:<28}"
        f"{norm_value:>20.6f}"
        f"{no_norm_value:>22.6f}"
    )

markdown_lines = [
    "# Raw-Space Action Evaluation",
    "",
    "| Metric | Clean + Norm | Clean + No Norm |",
    "|---|---:|---:|",
]

for metric, norm_value, no_norm_value in rows:
    markdown_lines.append(
        f"| {metric} | {norm_value:.6f} | "
        f"{no_norm_value:.6f} |"
    )

markdown_lines.extend(
    [
        "",
        "## Per-Dimension RMSE",
        "",
        "| Dimension | Clean + Norm | Clean + No Norm |",
        "|---|---:|---:|",
    ]
)

for dimension, (norm_value, no_norm_value) in enumerate(
    zip(
        norm_metrics["per_dim_rmse"],
        no_norm_metrics["per_dim_rmse"],
        strict=True,
    )
):
    name = (
        "gripper"
        if dimension == 6
        else f"action_dim_{dimension}"
    )

    markdown_lines.append(
        f"| {name} | {norm_value:.6f} | "
        f"{no_norm_value:.6f} |"
    )

(ROOT / "action_l2_comparison.md").write_text(
    "\n".join(markdown_lines) + "\n",
    encoding="utf-8",
)

print("\nSaved:")
print(ROOT / "action_l2_comparison.json")
print(ROOT / "action_l2_comparison.md")
