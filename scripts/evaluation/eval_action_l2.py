from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from openpi.policies import policy_config
from openpi.training import config as training_config


DATASET_ROOT = Path(
    "/root/autodl-tmp/cache/lerobot/"
    "physical-intelligence/libero"
)

VARIANTS = {
    "norm": {
        "config": "pi05_libero_day18_n50_s100",
        "checkpoint": (
            "checkpoints/pi05_libero_day18_n50_s100/"
            "day18_n50_s100/99"
        ),
    },
    "no_norm": {
        "config": "pi05_libero_day18_n50_nonorm_s100",
        "checkpoint": (
            "checkpoints/pi05_libero_day18_n50_nonorm_s100/"
            "day18_n50_nonorm_s100/99"
        ),
    },

    "dirty": {
        "config": "pi05_libero_day18_n50_dirty_s100",
        "checkpoint": (
            "checkpoints/pi05_libero_day18_n50_dirty_s100/"
            "day18_n50_dirty_s100/99"
        ),
    },
    "n100_s100": {
        "config": "pi05_libero_day18_n100_s100",
        "checkpoint": (
            "checkpoints/pi05_libero_day18_n100_s100/"
            "day18_n100_s100/99"
        ),
    },
    "n100_s500": {
        "config": "pi05_libero_day5_n100_s1000",
        "checkpoint": (
            "checkpoints/pi05_libero_day5_n100_s1000/"
            "day5_n100_s1000/499"
        ),
    },
    "n100_s1000": {
        "config": "pi05_libero_day5_n100_s1000",
        "checkpoint": (
            "checkpoints/pi05_libero_day5_n100_s1000/"
            "day5_n100_s1000/999"
        ),
    },
    "n100_s2000": {
        "config": "pi05_libero_day5_n100_s1000",
        "checkpoint": (
            "checkpoints/pi05_libero_day5_n100_s1000/"
            "day5_n100_s1000/2000"
        ),
    },
    "n100_s3000": {
        "config": "pi05_libero_day5_n100_s1000",
        "checkpoint": (
            "checkpoints/pi05_libero_day5_n100_s1000/"
            "day5_n100_s1000/3000"
        ),
    },
    "n100_s4000": {
        "config": "pi05_libero_day5_n100_s1000",
        "checkpoint": (
            "checkpoints/pi05_libero_day5_n100_s1000/"
            "day5_n100_s1000/4000"
        ),
    },
    "n100_s6000": {
        "config": "pi05_libero_day5_n100_s1000",
        "checkpoint": (
            "checkpoints/pi05_libero_day5_n100_s1000/"
            "day5_n100_s1000/6000"
        ),
    },
    "n100_s7000": {
        "config": "pi05_libero_day5_n100_s1000",
        "checkpoint": (
            "checkpoints/pi05_libero_day5_n100_s1000/"
            "day5_n100_s1000/7000"
        ),
    },
    "n100_s8000": {
        "config": "pi05_libero_day5_n100_s1000",
        "checkpoint": (
            "checkpoints/pi05_libero_day5_n100_s1000/"
            "day5_n100_s1000/8000"
        ),
    },
    "n100_s9000": {
        "config": "pi05_libero_day5_n100_s1000",
        "checkpoint": (
            "checkpoints/pi05_libero_day5_n100_s1000/"
            "day5_n100_s1000/9000"
        ),
    },
    "n100_s5000": {
        "config": "pi05_libero_day5_n100_s1000",
        "checkpoint": (
            "checkpoints/pi05_libero_day5_n100_s1000/"
            "day5_n100_s1000/5000"
        ),
    },
    "n100_s10000": {
        "config": "pi05_libero_day5_n100_s1000",
        "checkpoint": (
            "checkpoints/pi05_libero_day5_n100_s1000/"
            "day5_n100_s1000/9999"
        ),
    },

    "day22_s1000": {
        "config": "pi05_libero_day22_bs4_e2",
        "checkpoint": (
            "checkpoints/pi05_libero_day22_bs4_e2/"
            "day22_bs4_e2/1000"
        ),
    },
    "day22_s3000": {
        "config": "pi05_libero_day22_bs4_e2",
        "checkpoint": (
            "checkpoints/pi05_libero_day22_bs4_e2/"
            "day22_bs4_e2/3000"
        ),
    },
    "day22_s5000": {
        "config": "pi05_libero_day22_bs4_e2",
        "checkpoint": (
            "checkpoints/pi05_libero_day22_bs4_e2/"
            "day22_bs4_e2/5000"
        ),
    },
    "day22_s7000": {
        "config": "pi05_libero_day22_bs4_e2",
        "checkpoint": (
            "checkpoints/pi05_libero_day22_bs4_e2/"
            "day22_bs4_e2/7000"
        ),
    },
    "day22_s10000": {
        "config": "pi05_libero_day22_bs4_e2",
        "checkpoint": (
            "checkpoints/pi05_libero_day22_bs4_e2/"
            "day22_bs4_e2/10000"
        ),
    },
    "day22_s12000": {
        "config": "pi05_libero_day22_bs4_e2",
        "checkpoint": (
            "checkpoints/pi05_libero_day22_bs4_e2/"
            "day22_bs4_e2/12000"
        ),
    },
    "day22_s13999": {
        "config": "pi05_libero_day22_bs4_e2",
        "checkpoint": (
            "checkpoints/pi05_libero_day22_bs4_e2/"
            "day22_bs4_e2/13999"
        ),
    },

}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate π0.5 actions in raw LIBERO action space."
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        required=True,
    )
    parser.add_argument("--episode-start", type=int, default=50)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--samples-per-episode", type=int, default=3)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/day18_action_l2"),
    )
    return parser.parse_args()


def load_tasks() -> dict[int, str]:
    path = DATASET_ROOT / "meta/tasks.jsonl"
    tasks: dict[int, str] = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)

            task_index = item.get(
                "task_index",
                item.get("index"),
            )
            task_text = item.get(
                "task",
                item.get(
                    "task_text",
                    item.get("text"),
                ),
            )

            if task_index is not None and task_text is not None:
                tasks[int(task_index)] = str(task_text)

    if not tasks:
        raise RuntimeError("tasks.jsonl 中没有解析到任务文本")

    return tasks


def decode_image(value: object) -> np.ndarray:
    """Decode a LeRobot image struct into uint8 HWC RGB."""

    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"), dtype=np.uint8)

    if isinstance(value, dict):
        image_bytes = value.get("bytes")

        if image_bytes is not None:
            with Image.open(io.BytesIO(bytes(image_bytes))) as image:
                return np.asarray(
                    image.convert("RGB"),
                    dtype=np.uint8,
                )

        image_path = value.get("path")

        if image_path:
            path = Path(image_path)

            if not path.is_absolute():
                path = DATASET_ROOT / path

            with Image.open(path) as image:
                return np.asarray(
                    image.convert("RGB"),
                    dtype=np.uint8,
                )

    array = np.asarray(value)

    if array.ndim != 3:
        raise ValueError(
            f"无法解析图像，类型={type(value)}, shape={array.shape}"
        )

    if np.issubdtype(array.dtype, np.floating):
        if array.max() <= 1.0:
            array = array * 255.0

        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))

    return array


def episode_path(episode_index: int) -> Path:
    info = json.loads(
        (DATASET_ROOT / "meta/info.json").read_text(
            encoding="utf-8"
        )
    )

    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size

    relative = info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )

    path = DATASET_ROOT / relative

    if not path.exists():
        raise FileNotFoundError(path)

    return path


def select_frame_indices(
    num_frames: int,
    horizon: int,
    count: int,
) -> list[int]:
    max_start = num_frames - horizon

    if max_start < 0:
        return []

    indices = np.linspace(
        0,
        max_start,
        num=count,
        dtype=int,
    )

    return sorted(set(indices.tolist()))


def create_policy(
    variant: str,
):
    specification = VARIANTS[variant]

    config = training_config.get_config(
        specification["config"]
    )

    checkpoint = Path(
        specification["checkpoint"]
    ).resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    if variant != "no_norm":
        # 从 checkpoint/assets 加载训练时使用的 quantile norm stats。
        policy = policy_config.create_trained_policy(
            config,
            checkpoint,
        )
    else:
        # 空字典会让 Normalize / Unnormalize 都成为 no-op。
        # 不能传 None，因为 None 会触发从 checkpoint 加载 stats。
        policy = policy_config.create_trained_policy(
            config,
            checkpoint,
            norm_stats={},
        )

    return config, policy


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict:
    diff = predictions - targets

    action_l2_per_step = np.linalg.norm(
        diff,
        axis=-1,
    )

    per_dim_mae = np.mean(
        np.abs(diff),
        axis=(0, 1),
    )

    per_dim_rmse = np.sqrt(
        np.mean(diff**2, axis=(0, 1))
    )

    flat_predictions = predictions.reshape(
        -1,
        predictions.shape[-1],
    )

    flat_targets = targets.reshape(
        -1,
        targets.shape[-1],
    )

    return {
        "num_samples": int(predictions.shape[0]),
        "action_horizon": int(predictions.shape[1]),
        "action_dim": int(predictions.shape[2]),
        "raw_action_l2_mean": float(
            action_l2_per_step.mean()
        ),
        "raw_action_l2_std": float(
            action_l2_per_step.std()
        ),
        "raw_action_rmse": float(
            np.sqrt(np.mean(diff**2))
        ),
        "raw_action_mae": float(
            np.mean(np.abs(diff))
        ),
        "gripper_rmse": float(per_dim_rmse[-1]),
        "per_dim_mae": per_dim_mae.tolist(),
        "per_dim_rmse": per_dim_rmse.tolist(),
        "prediction_mean": (
            flat_predictions.mean(axis=0).tolist()
        ),
        "prediction_std": (
            flat_predictions.std(axis=0).tolist()
        ),
        "prediction_min": (
            flat_predictions.min(axis=0).tolist()
        ),
        "prediction_max": (
            flat_predictions.max(axis=0).tolist()
        ),
        "target_mean": (
            flat_targets.mean(axis=0).tolist()
        ),
        "target_std": (
            flat_targets.std(axis=0).tolist()
        ),
        "target_min": (
            flat_targets.min(axis=0).tolist()
        ),
        "target_max": (
            flat_targets.max(axis=0).tolist()
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks()
    config, policy = create_policy(args.variant)

    horizon = int(config.model.action_horizon)
    model_action_dim = int(config.model.action_dim)

    predictions = []
    targets = []
    episode_ids = []
    frame_ids = []
    inference_times = []

    sample_index = 0

    episode_range = range(
        args.episode_start,
        args.episode_start + args.num_episodes,
    )

    for episode_index in episode_range:
        path = episode_path(episode_index)
        table = pq.read_table(path)
        rows = table.to_pylist()

        frame_indices = select_frame_indices(
            num_frames=len(rows),
            horizon=horizon,
            count=args.samples_per_episode,
        )

        print(
            f"episode={episode_index}, "
            f"frames={len(rows)}, "
            f"samples={frame_indices}"
        )

        for frame_index in frame_indices:
            row = rows[frame_index]
            task_index = int(row["task_index"])

            if task_index not in tasks:
                raise KeyError(
                    f"task_index={task_index} 没有对应任务文本"
                )

            observation = {
                "observation/image": decode_image(
                    row["image"]
                ),
                "observation/wrist_image": decode_image(
                    row["wrist_image"]
                ),
                "observation/state": np.asarray(
                    row["state"],
                    dtype=np.float32,
                ),
                "prompt": tasks[task_index],
            }

            target_chunk = np.asarray(
                [
                    rows[index]["actions"]
                    for index in range(
                        frame_index,
                        frame_index + horizon,
                    )
                ],
                dtype=np.float32,
            )

            # 每个样本固定一份 noise；两次独立运行会生成完全相同的 noise。
            rng = np.random.default_rng(
                args.noise_seed + sample_index
            )

            noise = rng.standard_normal(
                size=(horizon, model_action_dim)
            ).astype(np.float32)

            result = policy.infer(
                observation,
                noise=noise,
            )

            predicted_chunk = np.asarray(
                result["actions"],
                dtype=np.float32,
            )

            if predicted_chunk.ndim != 2:
                raise ValueError(
                    "预测 action 维度异常："
                    f"{predicted_chunk.shape}"
                )

            predicted_chunk = predicted_chunk[
                :horizon,
                :7,
            ]

            if predicted_chunk.shape != target_chunk.shape:
                raise ValueError(
                    "预测与标签 shape 不一致："
                    f"pred={predicted_chunk.shape}, "
                    f"target={target_chunk.shape}"
                )

            predictions.append(predicted_chunk)
            targets.append(target_chunk)
            episode_ids.append(episode_index)
            frame_ids.append(frame_index)

            timing = result.get("policy_timing", {})
            inference_times.append(
                float(timing.get("infer_ms", np.nan))
            )

            print(
                f"  sample={sample_index:03d}, "
                f"frame={frame_index:03d}, "
                f"infer_ms={inference_times[-1]:.1f}"
            )

            sample_index += 1

    prediction_array = np.stack(predictions)
    target_array = np.stack(targets)

    metrics = compute_metrics(
        prediction_array,
        target_array,
    )

    metrics["variant"] = args.variant
    metrics["config"] = VARIANTS[args.variant]["config"]
    metrics["checkpoint"] = VARIANTS[args.variant][
        "checkpoint"
    ]
    metrics["episode_start"] = args.episode_start
    metrics["num_episodes"] = args.num_episodes
    metrics["samples_per_episode"] = (
        args.samples_per_episode
    )
    metrics["noise_seed"] = args.noise_seed
    metrics["mean_inference_ms"] = float(
        np.nanmean(inference_times)
    )

    output_npz = (
        args.output_dir
        / f"{args.variant}_predictions.npz"
    )

    np.savez_compressed(
        output_npz,
        predictions=prediction_array,
        targets=target_array,
        episode_ids=np.asarray(episode_ids),
        frame_ids=np.asarray(frame_ids),
        inference_ms=np.asarray(inference_times),
    )

    output_json = (
        args.output_dir
        / f"{args.variant}_metrics.json"
    )

    output_json.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n===== Evaluation result =====")
    print(json.dumps(metrics, indent=2))
    print("\npredictions:", output_npz)
    print("metrics:", output_json)


if __name__ == "__main__":
    main()
