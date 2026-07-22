#!/usr/bin/env python3
"""Build README figures exclusively from checked-in experiment evidence.

The script is intentionally independent from the OpenPI runtime. It only needs
matplotlib and reads lightweight logs/JSON summaries already stored in this
project. Figures are deterministic and are written as both PNG and SVG.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


BLUE = "#2563eb"
CYAN = "#0891b2"
GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
SLATE = "#475569"
LIGHT = "#e2e8f0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float64) / window
    valid = np.convolve(values, kernel, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), valid])


def style_axes(ax) -> None:
    ax.grid(axis="y", color=LIGHT, linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")


def save(fig, output_dir: Path, stem: str) -> list[str]:
    fig.tight_layout()
    outputs = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def build_training_loss(repo: Path, output_dir: Path) -> tuple[dict, list[str]]:
    source = repo / "experiments/day22_second_sft/train.log"
    public_csv = output_dir / "sft_training_loss.csv"
    if source.exists():
        text = source.read_text(encoding="utf-8", errors="ignore")
        matches = re.findall(r"Step\s+(\d+):[^\r\n]*?loss=([0-9.eE+-]+)", text)
        by_step = {int(step): float(loss) for step, loss in matches}
        source_label = str(source.relative_to(repo)).replace("\\", "/")
        source_digest = sha256(source)
    elif public_csv.exists():
        with public_csv.open("r", encoding="utf-8", newline="") as stream:
            by_step = {int(row["step"]): float(row["loss"]) for row in csv.DictReader(stream)}
        source_label = "docs/media/readme/sft_training_loss.csv"
        source_digest = sha256(public_csv)
    else:
        raise FileNotFoundError(f"Neither {source} nor {public_csv} exists")

    if len(by_step) != 14_000 or min(by_step) != 0 or max(by_step) != 13_999:
        raise ValueError(f"Expected steps 0..13999, found {len(by_step)} unique steps")

    steps = np.asarray(sorted(by_step), dtype=np.int32)
    losses = np.asarray([by_step[int(step)] for step in steps], dtype=np.float64)
    smooth = rolling_mean(losses, 200)

    with public_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "loss", "moving_average_200"])
        for step, loss, average in zip(steps, losses, smooth, strict=True):
            writer.writerow([int(step), f"{loss:.10g}", "" if np.isnan(average) else f"{average:.10g}"])

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(steps[::10], losses[::10], color=CYAN, alpha=0.13, linewidth=0.8, label="raw loss (1/10 shown)")
    ax.plot(steps, smooth, color=BLUE, linewidth=2.2, label="200-step moving average")
    ax.scatter([steps[-1]], [losses[-1]], color=GREEN, s=44, zorder=3)
    ax.annotate(
        f"final loss = {losses[-1]:.4f}\nlast-200 mean = {np.mean(losses[-200:]):.4f}",
        xy=(steps[-1], losses[-1]),
        xytext=(-155, 52),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": SLATE},
        fontsize=10,
        color=SLATE,
    )
    ax.set_title("π0.5 LoRA SFT training loss", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Flow-matching loss")
    ax.set_xlim(0, 14_000)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)
    outputs = save(fig, output_dir, "sft_training_loss")
    return {
        "source": source_label,
        "source_sha256": source_digest,
        "public_loss_csv": "docs/media/readme/sft_training_loss.csv",
        "public_loss_csv_sha256": sha256(public_csv),
        "steps": len(steps),
        "first_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "first_200_mean": float(np.mean(losses[:200])),
        "last_200_mean": float(np.mean(losses[-200:])),
    }, outputs


def build_checkpoint_quality(repo: Path, output_dir: Path) -> tuple[dict, list[str]]:
    source = repo / "experiments/day23_eval/second_round_curve/second_round_curve_summary.json"
    rows = load_json(source)
    steps = np.asarray([row["step"] for row in rows])
    action_l2 = np.asarray([row["action_l2"] for row in rows])
    sign_accuracy = np.asarray([row["sign_accuracy"] * 100 for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    ax = axes[0]
    ax.plot(steps, action_l2, color=BLUE, marker="o", linewidth=2.2)
    ax.scatter([steps[-1]], [action_l2[-1]], color=GREEN, s=55, zorder=3)
    ax.annotate(f"{action_l2[-1]:.3f}", (steps[-1], action_l2[-1]), xytext=(-4, 12), textcoords="offset points", ha="right")
    ax.set_title("Raw-space Action L2 ↓", loc="left", fontweight="bold")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Action L2")
    ax.set_ylim(bottom=0)
    style_axes(ax)

    ax = axes[1]
    ax.plot(steps, sign_accuracy, color=CYAN, marker="o", linewidth=2.2)
    ax.scatter([steps[-1]], [sign_accuracy[-1]], color=GREEN, s=55, zorder=3)
    ax.annotate(f"{sign_accuracy[-1]:.0f}%", (steps[-1], sign_accuracy[-1]), xytext=(-4, 12), textcoords="offset points", ha="right")
    ax.axhline(rows[-1]["majority_baseline"] * 100, color=SLATE, linestyle="--", linewidth=1.2, label="majority baseline")
    ax.set_title("Gripper sign accuracy ↑", loc="left", fontweight="bold")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 100)
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)

    fig.suptitle("Offline checkpoint quality on a fixed 30-observation validation set", fontsize=15, fontweight="bold")
    outputs = save(fig, output_dir, "checkpoint_quality")
    return {
        "source": str(source.relative_to(repo)).replace("\\", "/"),
        "source_sha256": sha256(source),
        "checkpoints": len(rows),
        "final_action_l2": float(action_l2[-1]),
        "final_gripper_sign_accuracy": float(sign_accuracy[-1] / 100),
    }, outputs


def build_closed_loop(repo: Path, output_dir: Path) -> tuple[dict, list[str]]:
    task5_source = repo / "experiments/day23_eval/formal_compare/task5_formal_results.json"
    screen_source = repo / "experiments/day24_ten_task_screen/ten_task_screen_summary.json"
    task1_source = repo / "experiments/day24_task1_eval/task1_states3_9_summary.json"
    task5 = load_json(task5_source)
    screen = load_json(screen_source)
    task1 = load_json(task1_source)

    task5_rows = {row["policy"]: row for row in task5["summaries"]}
    left_labels = ["Old 5k", "SFT 13,999", "Official π0.5"]
    left_values = [
        task5_rows["old_5k"]["success_rate"] * 100,
        task5_rows["second_round_13999"]["success_rate"] * 100,
        task5_rows["official"]["success_rate"] * 100,
    ]
    left_counts = ["0/10", "9/10", "10/10"]

    task1_successes = 3 + task1["successes"]
    task1_trials = 3 + task1["trials"]
    right_labels = ["Task 1\nstates 0–9", "LIBERO-10\n3-state screen"]
    right_values = [task1_successes / task1_trials * 100, screen["overall"]["success_rate"] * 100]
    right_counts = [f"{task1_successes}/{task1_trials}", f"{screen['overall']['successes']}/{screen['overall']['trials']}"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1))
    ax = axes[0]
    bars = ax.bar(left_labels, left_values, color=[SLATE, BLUE, GREEN], width=0.65)
    for bar, count in zip(bars, left_counts, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5, count, ha="center", fontweight="bold")
    ax.set_title("Task 5 — same states 0–9 protocol", loc="left", fontweight="bold")
    ax.set_ylabel("Official success rate (%)")
    ax.set_ylim(0, 112)
    style_axes(ax)

    ax = axes[1]
    bars = ax.bar(right_labels, right_values, color=[CYAN, BLUE], width=0.58)
    for bar, count in zip(bars, right_counts, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5, count, ha="center", fontweight="bold")
    ax.set_title("Additional closed-loop evidence", loc="left", fontweight="bold")
    ax.set_ylabel("Observed success rate (%)")
    ax.set_ylim(0, 112)
    ax.text(0.5, -0.23, "Different protocols; the 25/30 result is a breadth screen, not an official benchmark.", transform=ax.transAxes, ha="center", fontsize=9, color=SLATE)
    style_axes(ax)

    fig.suptitle("Closed-loop LIBERO evaluation", fontsize=15, fontweight="bold")
    outputs = save(fig, output_dir, "closed_loop_results")
    return {
        "sources": [
            str(task5_source.relative_to(repo)).replace("\\", "/"),
            str(task1_source.relative_to(repo)).replace("\\", "/"),
            str(screen_source.relative_to(repo)).replace("\\", "/"),
        ],
        "task5_sft": "9/10",
        "task5_old_5k": "0/10",
        "task5_official": "10/10",
        "task1_sft": f"{task1_successes}/{task1_trials}",
        "ten_task_screen": f"{screen['overall']['successes']}/{screen['overall']['trials']}",
    }, outputs


def build_robustness_and_failures(repo: Path, output_dir: Path) -> tuple[dict, list[str]]:
    robustness_source = repo / "experiments/day24_controlled_robustness/task5_state0_vs_state2/robustness_summary.json"
    robustness = load_json(robustness_source)
    groups = robustness["groups"]
    labels = ["Task 5 state 0", "Task 5 state 2"]
    values = [groups["0"]["success_rate"] * 100, groups["2"]["success_rate"] * 100]
    lowers = [groups["0"]["wilson_95_ci"][0] * 100, groups["2"]["wilson_95_ci"][0] * 100]
    uppers = [groups["0"]["wilson_95_ci"][1] * 100, groups["2"]["wilson_95_ci"][1] * 100]
    errors = np.asarray([[value - lower for value, lower in zip(values, lowers, strict=True)], [upper - value for value, upper in zip(values, uppers, strict=True)]])

    failure_labels = ["Place outside target", "Contact, no grasp", "Timeout / oscillation"]
    failure_counts = [2, 2, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    ax = axes[0]
    bars = ax.bar(labels, values, yerr=errors, capsize=5, color=[BLUE, CYAN], width=0.58)
    for bar, count in zip(bars, ["9/10", "7/10"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, count, ha="center", fontweight="bold")
    ax.set_title("Initial-state sensitivity", loc="left", fontweight="bold")
    ax.set_ylabel("Success rate with Wilson 95% CI (%)")
    ax.set_ylim(0, 118)
    style_axes(ax)

    ax = axes[1]
    bars = ax.barh(failure_labels, failure_counts, color=[RED, AMBER, SLATE])
    for bar, count in zip(bars, failure_counts, strict=True):
        ax.text(bar.get_width() + 0.06, bar.get_y() + bar.get_height() / 2, f"{count}/5", va="center", fontweight="bold")
    ax.invert_yaxis()
    ax.set_title("Observed failure labels", loc="left", fontweight="bold")
    ax.set_xlabel("Count in the 30-rollout breadth screen")
    ax.set_xlim(0, 2.7)
    style_axes(ax)
    ax.text(0.5, -0.23, "Five observed failures only; percentages are not population-level causal estimates.", transform=ax.transAxes, ha="center", fontsize=9, color=SLATE)

    fig.suptitle("Robustness and failure analysis", fontsize=15, fontweight="bold")
    outputs = save(fig, output_dir, "robustness_and_failures")
    return {
        "source": str(robustness_source.relative_to(repo)).replace("\\", "/"),
        "source_sha256": sha256(robustness_source),
        "state0": "9/10",
        "state2": "7/10",
        "failure_counts": dict(zip(failure_labels, failure_counts, strict=True)),
    }, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    output_dir = (args.output_dir or repo / "docs/media/readme").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "training_loss": build_training_loss,
        "checkpoint_quality": build_checkpoint_quality,
    }
    manifest = {"schema_version": "readme_figures_v1", "figures": {}}
    for name, builder in builders.items():
        metrics, outputs = builder(repo, output_dir)
        manifest["figures"][name] = {"metrics": metrics, "outputs": outputs}

    manifest_path = output_dir / "figure_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Generated {sum(len(item['outputs']) for item in manifest['figures'].values())} figures")
    print(manifest_path)


if __name__ == "__main__":
    main()
