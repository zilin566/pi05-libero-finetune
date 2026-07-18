from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


SOURCE = Path(
    "/root/autodl-tmp/cache/lerobot/"
    "physical-intelligence/libero"
)

OUTPUT_PARENT = SOURCE.parent
SIZES = [10, 50, 100]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_subset(size: int) -> None:
    target = OUTPUT_PARENT / f"libero_day18_n{size}"

    if target.exists():
        raise FileExistsError(
            f"{target} 已存在。为避免误覆盖，请先人工检查或删除。"
        )

    source_info_path = SOURCE / "meta" / "info.json"
    source_episodes_path = SOURCE / "meta" / "episodes.jsonl"

    info = json.loads(source_info_path.read_text(encoding="utf-8"))
    episodes = read_jsonl(source_episodes_path)

    if len(episodes) < size:
        raise ValueError(
            f"源数据只有 {len(episodes)} 条 episode，无法创建 n={size}"
        )

    subset = episodes[:size]

    target_meta = target / "meta"
    target_meta.mkdir(parents=True, exist_ok=True)

    # 保持同一份任务映射与统计量，确保消融只改变数据规模。
    for name in ["tasks.jsonl", "stats.json"]:
        src = SOURCE / "meta" / name
        if src.exists():
            shutil.copy2(src, target_meta / name)

    write_jsonl(target_meta / "episodes.jsonl", subset)

    subset_info = dict(info)
    subset_info["total_episodes"] = size
    subset_info["total_frames"] = sum(
        int(row.get("length", 0)) for row in subset
    )
    subset_info["splits"] = {"train": f"0:{size}"}

    chunks_size = int(subset_info.get("chunks_size", 1000))
    subset_info["total_chunks"] = (size + chunks_size - 1) // chunks_size

    (target_meta / "info.json").write_text(
        json.dumps(subset_info, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    data_pattern = info["data_path"]

    for row in subset:
        episode_index = int(row["episode_index"])
        episode_chunk = episode_index // chunks_size

        relative_path = Path(
            data_pattern.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
        )

        src_file = SOURCE / relative_path
        dst_file = target / relative_path

        if not src_file.exists():
            raise FileNotFoundError(f"缺少源文件：{src_file}")

        hardlink_or_copy(src_file, dst_file)

    print(
        f"完成 n={size}: "
        f"episodes={size}, "
        f"frames={subset_info['total_frames']}, "
        f"path={target}"
    )


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"完整数据集不存在：{SOURCE}")

    for size in SIZES:
        build_subset(size)


if __name__ == "__main__":
    main()
