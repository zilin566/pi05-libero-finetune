from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE = Path(
    "/root/autodl-tmp/cache/lerobot/"
    "physical-intelligence/libero_day18_n50"
)

TARGET = Path(
    "/root/autodl-tmp/cache/lerobot/"
    "physical-intelligence/libero_day18_n50_dirty_finite"
)

# 在 5 条轨迹中注入有限异常动作
DIRTY_EPISODES = [0, 10, 20, 30, 40]
CORRUPTION_RATIO = 0.10
INJECTED_ABS_VALUE = 3.0
SEED = 42


def replace_parquet(path: Path, table: pa.Table) -> None:
    """写入临时文件，再替换目标文件，避免修改源数据的硬链接。"""
    temp_path = path.with_name(path.stem + "_dirty_tmp.parquet")
    pq.write_table(table, temp_path)
    os.replace(temp_path, path)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"干净 n50 数据不存在：{SOURCE}")

    if TARGET.exists():
        raise FileExistsError(
            f"目标目录已存在：{TARGET}\n"
            f"确认可以重建后执行：rm -rf {TARGET}"
        )

    # 未修改文件使用硬链接，减少额外磁盘占用。
    # 被修改的 parquet 随后通过 os.replace 变为独立文件。
    shutil.copytree(
        SOURCE,
        TARGET,
        copy_function=os.link,
    )

    info_path = TARGET / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))

    data_pattern = info["data_path"]
    chunks_size = int(info.get("chunks_size", 1000))

    manifest = {
        "experiment": "synthetic_finite_action_corruption",
        "source_dataset": str(SOURCE),
        "target_dataset": str(TARGET),
        "corruption_ratio": CORRUPTION_RATIO,
        "injected_abs_value": INJECTED_ABS_VALUE,
        "seed": SEED,
        "episodes": [],
    }

    total_corrupted = 0

    for episode_index in DIRTY_EPISODES:
        episode_chunk = episode_index // chunks_size

        relative_path = Path(
            data_pattern.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
        )

        parquet_path = TARGET / relative_path

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"找不到 episode 文件：{parquet_path}"
            )

        table = pq.read_table(parquet_path)

        action_key = next(
            (
                key
                for key in ["actions", "action"]
                if key in table.column_names
            ),
            None,
        )

        if action_key is None:
            raise KeyError(
                f"{parquet_path} 中找不到 actions/action 字段，"
                f"当前字段：{table.column_names}"
            )

        actions = table[action_key].to_pylist()
        num_frames = len(actions)

        rng = np.random.default_rng(SEED + episode_index)

        corrupted_count = max(
            1,
            round(num_frames * CORRUPTION_RATIO),
        )

        selected_frames = sorted(
            rng.choice(
                num_frames,
                size=corrupted_count,
                replace=False,
            ).tolist()
        )

        original_examples = []
        dirty_examples = []

        for frame_index in selected_frames:
            action = np.asarray(
                actions[frame_index],
                dtype=np.float32,
            ).copy()

            if action.size == 0:
                raise ValueError(
                    f"episode {episode_index} frame {frame_index} action 为空"
                )

            if len(original_examples) < 3:
                original_examples.append(
                    {
                        "frame": frame_index,
                        "action_dim0": float(action[0]),
                    }
                )

            action[0] = (
                INJECTED_ABS_VALUE
                if action[0] >= 0
                else -INJECTED_ABS_VALUE
            )

            if len(dirty_examples) < 3:
                dirty_examples.append(
                    {
                        "frame": frame_index,
                        "action_dim0": float(action[0]),
                    }
                )

            actions[frame_index] = action.tolist()

        field_type = table.schema.field(action_key).type
        new_action_column = pa.array(actions, type=field_type)

        column_index = table.schema.get_field_index(action_key)
        dirty_table = table.set_column(
            column_index,
            action_key,
            new_action_column,
        )

        replace_parquet(parquet_path, dirty_table)

        total_corrupted += len(selected_frames)

        manifest["episodes"].append(
            {
                "episode_index": episode_index,
                "num_frames": num_frames,
                "action_key": action_key,
                "corrupted_count": len(selected_frames),
                "corrupted_frames": selected_frames,
                "corrupted_dimension": 0,
                "original_examples": original_examples,
                "dirty_examples": dirty_examples,
            }
        )

        print(
            f"episode={episode_index:03d} | "
            f"frames={num_frames} | "
            f"corrupted={len(selected_frames)}"
        )

    manifest["total_corrupted_frames"] = total_corrupted

    manifest_path = TARGET / "meta/corruption_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDirty dataset 创建完成")
    print("dataset:", TARGET)
    print("total corrupted frames:", total_corrupted)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
