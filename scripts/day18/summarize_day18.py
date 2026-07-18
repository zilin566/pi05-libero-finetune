from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("experiments")
OUTPUT = ROOT / "day18_summary"
OUTPUT.mkdir(parents=True, exist_ok=True)

LOGS = {
    "n10": ROOT / "day18_data_scale/logs/n10_s100.log",
    "n50_clean_norm": ROOT / "day18_data_scale/logs/n50_s100.log",
    "n100": ROOT / "day18_data_scale/logs/n100_s100.log",
    "n50_no_norm": ROOT / "day18_norm_ablation/logs/n50_nonorm_s100.log",
    "n50_dirty_norm": ROOT / "day18_dirty_ablation/logs/n50_dirty_s100.log",
}

ACTION_METRICS = {
    "clean_norm": ROOT / "day18_action_l2/norm_metrics.json",
    "clean_no_norm": ROOT / "day18_action_l2/no_norm_metrics.json",
    "dirty_norm": ROOT / "day18_action_l2/dirty_metrics.json",
}

STEP_PATTERN = re.compile(
    r"Step\s+(\d+):\s+"
    r"grad_norm=([0-9.eE+-]+),\s+"
    r"loss=([0-9.eE+-]+),\s+"
    r"param_norm=([0-9.eE+-]+)"
)


def parse_log(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"日志不存在：{path}")

    # 使用 step 作为 key，避免日志中有重复输出。
    rows_by_step: dict[int, dict] = {}

    for line in path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():
        match = STEP_PATTERN.search(line)

        if not match:
            continue

        step = int(match.group(1))

        rows_by_step[step] = {
            "step": step,
            "grad_norm": float(match.group(2)),
            "loss": float(match.group(3)),
            "param_norm": float(match.group(4)),
        }

    rows = [rows_by_step[key] for key in sorted(rows_by_step)]

    if not rows:
        raise RuntimeError(f"没有从日志中解析出 Step：{path}")

    return rows


def summarize(rows: list[dict]) -> dict:
    losses = np.asarray(
        [row["loss"] for row in rows],
        dtype=np.float64,
    )
    grads = np.asarray(
        [row["grad_norm"] for row in rows],
        dtype=np.float64,
    )

    last20 = losses[-20:]
    mean = float(losses.mean())
    std = float(losses.std())
    threshold = mean + 2 * std

    spike_rows = [
        row for row in rows
        if row["loss"] > threshold
    ]

    return {
        "steps": len(rows),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "minimum_loss": float(losses.min()),
        "maximum_loss": float(losses.max()),
        "overall_mean_loss": mean,
        "overall_loss_std": std,
        "last20_mean_loss": float(last20.mean()),
        "last20_loss_std": float(last20.std()),
        "max_grad_norm": float(grads.max()),
        "spike_threshold": threshold,
        "spike_count": len(spike_rows),
        "spike_steps": [
            {
                "step": row["step"],
                "loss": row["loss"],
            }
            for row in spike_rows
        ],
    }


def save_training_summary(
    rows_map: dict[str, list[dict]],
    summaries: dict[str, dict],
) -> None:
    json_path = OUTPUT / "training_summary.json"
    json_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fields = [
        "experiment",
        "steps",
        "initial_loss",
        "final_loss",
        "minimum_loss",
        "maximum_loss",
        "overall_mean_loss",
        "overall_loss_std",
        "last20_mean_loss",
        "last20_loss_std",
        "max_grad_norm",
        "spike_count",
    ]

    with (OUTPUT / "training_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for name, summary in summaries.items():
            writer.writerow({
                "experiment": name,
                **{
                    key: summary[key]
                    for key in fields
                    if key != "experiment"
                },
            })

    # 数据规模曲线
    plt.figure(figsize=(9, 5))

    for name in ["n10", "n50_clean_norm", "n100"]:
        rows = rows_map[name]
        plt.plot(
            [row["step"] for row in rows],
            [row["loss"] for row in rows],
            label=name,
        )

    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Data-Scale Ablation: n10 vs n50 vs n100")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT / "data_scale_loss.png",
        dpi=180,
    )
    plt.close()

    # Norm / Dirty 曲线
    plt.figure(figsize=(9, 5))

    for name in [
        "n50_clean_norm",
        "n50_no_norm",
        "n50_dirty_norm",
    ]:
        rows = rows_map[name]
        plt.plot(
            [row["step"] for row in rows],
            [row["loss"] for row in rows],
            label=name,
        )

    plt.xlabel("Training step")
    plt.ylabel("Training-space loss")
    plt.title("Normalization and Dirty-Data Training Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT / "norm_dirty_loss.png",
        dpi=180,
    )
    plt.close()


def load_action_metrics() -> dict[str, dict]:
    metrics = {}

    for name, path in ACTION_METRICS.items():
        if not path.exists():
            raise FileNotFoundError(f"指标文件不存在：{path}")

        metrics[name] = json.loads(
            path.read_text(encoding="utf-8")
        )

    return metrics


def save_action_summary(metrics: dict[str, dict]) -> None:
    fields = [
        "experiment",
        "raw_action_l2_mean",
        "raw_action_rmse",
        "raw_action_mae",
        "mean_inference_ms",
    ]

    with (OUTPUT / "action_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for name, values in metrics.items():
            writer.writerow({
                "experiment": name,
                **{
                    field: values[field]
                    for field in fields
                    if field != "experiment"
                },
            })

    (OUTPUT / "action_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    names = list(metrics)
    l2_values = [
        metrics[name]["raw_action_l2_mean"]
        for name in names
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(names, l2_values)
    plt.ylabel("Raw-space Action L2")
    plt.title("Raw-Action-Space Evaluation")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT / "action_l2_comparison.png",
        dpi=180,
    )
    plt.close()


def pct_change(new: float, old: float) -> float:
    return (new - old) / old * 100.0


def write_markdown(
    summaries: dict[str, dict],
    action_metrics: dict[str, dict],
) -> None:
    clean = summaries["n50_clean_norm"]
    no_norm = summaries["n50_no_norm"]
    dirty = summaries["n50_dirty_norm"]

    clean_action = action_metrics["clean_norm"]
    no_norm_action = action_metrics["clean_no_norm"]
    dirty_action = action_metrics["dirty_norm"]

    lines = [
        "# Day18 - Data Validation and Ablation Experiments",
        "",
        "## 1. 实验背景",
        "",
        "LIBERO 抽查数据中未发现需要剔除的真实硬错误，因此没有进行两组完全相同的 raw-vs-cleaned 训练。主实验调整为数据规模、Normalization 和合成有限动作异常三类消融。",
        "",
        "## 2. 数据规模消融",
        "",
        "| Dataset | Steps | Final loss | Mean loss | Last-20 mean | Last-20 std | Max grad norm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name in ["n10", "n50_clean_norm", "n100"]:
        s = summaries[name]
        lines.append(
            f"| {name} | {s['steps']} | "
            f"{s['final_loss']:.6f} | "
            f"{s['overall_mean_loss']:.6f} | "
            f"{s['last20_mean_loss']:.6f} | "
            f"{s['last20_loss_std']:.6f} | "
            f"{s['max_grad_norm']:.6f} |"
        )

    lines.extend([
        "",
        "该实验是在相同训练步数和计算预算下比较不同 episode 数量的训练动态，不能单独替代固定验证集或仿真成功率。",
        "",
        "## 3. Normalization 消融",
        "",
        "| Metric | Clean + Norm | Clean + No Norm |",
        "|---|---:|---:|",
        f"| Training mean loss | {clean['overall_mean_loss']:.6f} | {no_norm['overall_mean_loss']:.6f} |",
        f"| Last-20 loss | {clean['last20_mean_loss']:.6f} | {no_norm['last20_mean_loss']:.6f} |",
        f"| Max grad norm | {clean['max_grad_norm']:.6f} | {no_norm['max_grad_norm']:.6f} |",
        f"| Raw-space Action L2 | {clean_action['raw_action_l2_mean']:.6f} | {no_norm_action['raw_action_l2_mean']:.6f} |",
        f"| Raw-space RMSE | {clean_action['raw_action_rmse']:.6f} | {no_norm_action['raw_action_rmse']:.6f} |",
        f"| Raw-space MAE | {clean_action['raw_action_mae']:.6f} | {no_norm_action['raw_action_mae']:.6f} |",
        "",
        "无归一化模型具有更低的训练空间 loss，但两组训练目标尺度不同，训练 loss 不能直接横向比较。在统一的 LIBERO 原始动作空间中，Normalization 模型的 Action L2、RMSE 和 MAE 更低。",
        "",
        "### Per-Dimension RMSE",
        "",
        "| Dimension | Clean + Norm | Clean + No Norm |",
        "|---|---:|---:|",
    ])

    for dim, (norm_value, no_norm_value) in enumerate(
        zip(
            clean_action["per_dim_rmse"],
            no_norm_action["per_dim_rmse"],
            strict=True,
        )
    ):
        name = "gripper" if dim == 6 else f"action_dim_{dim}"
        lines.append(
            f"| {name} | {norm_value:.6f} | "
            f"{no_norm_value:.6f} |"
        )

    lines.extend([
        "",
        "## 4. Synthetic Dirty-Data Ablation",
        "",
        "| Metric | Clean + Norm | Dirty + Norm | Change |",
        "|---|---:|---:|---:|",
        (
            f"| Training mean loss | "
            f"{clean['overall_mean_loss']:.6f} | "
            f"{dirty['overall_mean_loss']:.6f} | "
            f"{pct_change(dirty['overall_mean_loss'], clean['overall_mean_loss']):+.2f}% |"
        ),
        (
            f"| Training loss std | "
            f"{clean['overall_loss_std']:.6f} | "
            f"{dirty['overall_loss_std']:.6f} | "
            f"{pct_change(dirty['overall_loss_std'], clean['overall_loss_std']):+.2f}% |"
        ),
        (
            f"| Maximum loss | "
            f"{clean['maximum_loss']:.6f} | "
            f"{dirty['maximum_loss']:.6f} | "
            f"{pct_change(dirty['maximum_loss'], clean['maximum_loss']):+.2f}% |"
        ),
        (
            f"| Raw-space Action L2 | "
            f"{clean_action['raw_action_l2_mean']:.6f} | "
            f"{dirty_action['raw_action_l2_mean']:.6f} | "
            f"{pct_change(dirty_action['raw_action_l2_mean'], clean_action['raw_action_l2_mean']):+.2f}% |"
        ),
        (
            f"| Raw-space RMSE | "
            f"{clean_action['raw_action_rmse']:.6f} | "
            f"{dirty_action['raw_action_rmse']:.6f} | "
            f"{pct_change(dirty_action['raw_action_rmse'], clean_action['raw_action_rmse']):+.2f}% |"
        ),
        (
            f"| Raw-space MAE | "
            f"{clean_action['raw_action_mae']:.6f} | "
            f"{dirty_action['raw_action_mae']:.6f} | "
            f"{pct_change(dirty_action['raw_action_mae'], clean_action['raw_action_mae']):+.2f}% |"
        ),
        "",
        "人工有限动作离群值使训练 loss 及其波动上升，同时降低了未见 episode 上的原始动作空间预测精度，验证了数据检查和异常动作清洗的必要性。",
        "",
        "## 5. 当前实验限制",
        "",
        "- 当前训练仅为 100 step，小规模结果主要反映训练动态。",
        "- Action L2 使用固定离线验证样本，不能完全代表机器人任务成功率。",
        "- 最终任务级结论需在后续 LIBERO rollout 中使用 success rate 验证。",
        "",
        "## 6. 生成文件",
        "",
        "- `data_scale_loss.png`",
        "- `norm_dirty_loss.png`",
        "- `action_l2_comparison.png`",
        "- `training_summary.csv`",
        "- `action_metrics.csv`",
    ])

    (OUTPUT / "ablation_table.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows_map = {
        name: parse_log(path)
        for name, path in LOGS.items()
    }

    summaries = {
        name: summarize(rows)
        for name, rows in rows_map.items()
    }

    save_training_summary(rows_map, summaries)

    action_metrics = load_action_metrics()
    save_action_summary(action_metrics)

    write_markdown(summaries, action_metrics)

    print("Day18 汇总完成：")
    print(OUTPUT.resolve())

    for path in sorted(OUTPUT.iterdir()):
        print(" -", path.name)


if __name__ == "__main__":
    main()
