from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("experiments/day23_eval/second_round_curve")
OUTPUT_CSV = ROOT / "second_round_curve_summary.csv"
OUTPUT_JSON = ROOT / "second_round_curve_summary.json"

VARIANTS = [
    ("day22_s1000", 1000),
    ("day22_s3000", 3000),
    ("day22_s5000", 5000),
    ("day22_s7000", 7000),
    ("day22_s10000", 10000),
    ("day22_s12000", 12000),
    ("day22_s13999", 13999),
]


def safe_div(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def mae(error: np.ndarray) -> float:
    return float(np.mean(np.abs(error)))


def load_variant(variant: str) -> tuple[dict, dict[str, np.ndarray]]:
    metrics_path = ROOT / f"{variant}_metrics.json"
    predictions_path = ROOT / f"{variant}_predictions.npz"

    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    with np.load(predictions_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}

    return metrics, arrays


def main() -> None:
    rows: list[dict] = []

    reference_episode_ids: np.ndarray | None = None
    reference_frame_ids: np.ndarray | None = None
    reference_targets: np.ndarray | None = None

    for variant, step in VARIANTS:
        metadata, arrays = load_variant(variant)

        predictions = np.asarray(arrays["predictions"], dtype=np.float64)
        targets = np.asarray(arrays["targets"], dtype=np.float64)
        episode_ids = np.asarray(arrays["episode_ids"])
        frame_ids = np.asarray(arrays["frame_ids"])

        if predictions.shape != targets.shape:
            raise ValueError(
                f"{variant}: prediction shape {predictions.shape} "
                f"!= target shape {targets.shape}"
            )

        if predictions.ndim != 3 or predictions.shape[-1] != 7:
            raise ValueError(
                f"{variant}: expected [N, H, 7], got {predictions.shape}"
            )

        if not np.isfinite(predictions).all():
            raise ValueError(f"{variant}: predictions contain NaN or Inf")

        if not np.isfinite(targets).all():
            raise ValueError(f"{variant}: targets contain NaN or Inf")

        if reference_episode_ids is None:
            reference_episode_ids = episode_ids
            reference_frame_ids = frame_ids
            reference_targets = targets
        else:
            if not np.array_equal(episode_ids, reference_episode_ids):
                raise ValueError(f"{variant}: episode manifest mismatch")

            if not np.array_equal(frame_ids, reference_frame_ids):
                raise ValueError(f"{variant}: frame manifest mismatch")

            if not np.allclose(targets, reference_targets, atol=0.0, rtol=0.0):
                raise ValueError(f"{variant}: target arrays mismatch")

        error = predictions - targets

        # 与原脚本一致：先对每个 action step 的 7 维误差求 L2，
        # 再对所有 sample 和 horizon step 求均值。
        action_l2_per_step = np.linalg.norm(error, axis=-1)

        arm_error = error[..., :6]
        translation_error = error[..., :3]
        rotation_error = error[..., 3:6]
        gripper_error = error[..., 6]

        target_gripper = targets[..., 6]
        prediction_gripper = predictions[..., 6]

        target_positive = target_gripper > 0
        target_negative = target_gripper < 0
        target_zero = target_gripper == 0

        prediction_positive = prediction_gripper > 0
        prediction_negative = ~prediction_positive

        true_positive = int(np.sum(prediction_positive & target_positive))
        true_negative = int(np.sum(prediction_negative & target_negative))
        false_positive = int(np.sum(prediction_positive & target_negative))
        false_negative = int(np.sum(prediction_negative & target_positive))

        positive_count = int(np.sum(target_positive))
        negative_count = int(np.sum(target_negative))
        zero_count = int(np.sum(target_zero))
        total_count = int(target_gripper.size)

        sign_correct = (
            (prediction_positive & target_positive)
            | (prediction_negative & target_negative)
        )

        sign_accuracy = safe_div(
            int(np.sum(sign_correct)),
            positive_count + negative_count,
        )

        positive_recall = safe_div(true_positive, true_positive + false_negative)
        negative_recall = safe_div(true_negative, true_negative + false_positive)

        majority_baseline = safe_div(
            max(positive_count, negative_count),
            positive_count + negative_count,
        )

        # 只统计每个 action chunk 内部发生的目标 sign 切换。
        # 不跨 sample、不跨 episode 比较，避免制造伪 transition。
        target_sign = target_positive
        transition_mask = target_sign[:, 1:] != target_sign[:, :-1]

        transition_target = target_sign[:, 1:]
        transition_prediction = prediction_positive[:, 1:]

        transition_correct = (
            transition_prediction == transition_target
        ) & transition_mask

        transition_count = int(np.sum(transition_mask))
        transition_possible = int(transition_mask.size)
        transition_ratio = safe_div(transition_count, transition_possible)
        transition_sign_accuracy = safe_div(
            int(np.sum(transition_correct)),
            transition_count,
        )

        row = {
            "variant": variant,
            "step": step,
            "num_samples": int(predictions.shape[0]),
            "horizon": int(predictions.shape[1]),
            "action_l2": float(action_l2_per_step.mean()),
            "action_l2_std": float(action_l2_per_step.std()),
            "rmse": rmse(error),
            "mae": mae(error),
            "arm_rmse": rmse(arm_error),
            "arm_mae": mae(arm_error),
            "translation_rmse": rmse(translation_error),
            "rotation_rmse": rmse(rotation_error),
            "gripper_rmse": rmse(gripper_error),
            "gripper_mae": mae(gripper_error),
            "sign_accuracy": sign_accuracy,
            "positive_recall": positive_recall,
            "negative_recall": negative_recall,
            "majority_baseline": majority_baseline,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "zero_count": zero_count,
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "transition_count": transition_count,
            "transition_possible": transition_possible,
            "transition_ratio": transition_ratio,
            "transition_sign_accuracy": transition_sign_accuracy,
            "mean_inference_ms": float(
                metadata.get("mean_inference_ms", float("nan"))
            ),
        }
        rows.append(row)

    fieldnames = list(rows[0].keys())

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_JSON.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Validation manifest check: PASS")
    print(
        "Episodes:",
        sorted(np.unique(reference_episode_ids).tolist()),
    )
    print("Samples:", len(reference_episode_ids))
    print("Frame IDs:", reference_frame_ids.tolist())

    print()
    header = (
        f"{'step':>6} "
        f"{'L2':>8} "
        f"{'RMSE':>8} "
        f"{'MAE':>8} "
        f"{'arm_R':>8} "
        f"{'grip_R':>8} "
        f"{'sign_acc':>9} "
        f"{'trans_acc':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        transition_text = (
            "NA"
            if np.isnan(row["transition_sign_accuracy"])
            else f"{row['transition_sign_accuracy']:.4f}"
        )

        print(
            f"{row['step']:6d} "
            f"{row['action_l2']:8.4f} "
            f"{row['rmse']:8.4f} "
            f"{row['mae']:8.4f} "
            f"{row['arm_rmse']:8.4f} "
            f"{row['gripper_rmse']:8.4f} "
            f"{row['sign_accuracy']:9.4f} "
            f"{transition_text:>10}"
        )

    first = rows[0]
    print()
    print("Validation gripper distribution:")
    print("  positive:", first["positive_count"])
    print("  negative:", first["negative_count"])
    print("  zero:", first["zero_count"])
    print("  majority baseline:", f"{first['majority_baseline']:.4f}")
    print("  within-chunk transitions:", first["transition_count"])
    print("  transition ratio:", f"{first['transition_ratio']:.4f}")

    print()
    print("CSV:", OUTPUT_CSV)
    print("JSON:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
