from pathlib import Path
from datasets import load_dataset
import json
import numpy as np

ROOT = Path("/root/autodl-tmp/cache/lerobot/physical-intelligence/libero")
META_DIR = ROOT / "meta"

OUT_DIR = Path("logs/day8_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "sample_summary.json"
CROSS_TASK_LOG_PATH = Path("logs/day8_cross_episode_task_check.txt")

CHECK_EPISODE_IDS = [0, 1, 2, 10]


def load_tasks(tasks_path: Path):
    tasks = {}
    if not tasks_path.exists():
        print("tasks.jsonl not found:", tasks_path)
        return tasks

    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            task_index = item.get("task_index", item.get("index", None))
            task_text = item.get("task", item.get("task_text", item.get("text", None)))
            if task_index is not None:
                tasks[int(task_index)] = task_text
    return tasks


def to_shape(value):
    try:
        arr = np.asarray(value)
        return list(arr.shape)
    except Exception:
        return None


def brief_value(value, max_items=8):
    if value is None:
        return None
    try:
        arr = np.asarray(value)
        flat = arr.reshape(-1)
        return flat[:max_items].tolist()
    except Exception:
        return str(value)[:200]


def image_info(value):
    if value is None:
        return {"type": "None"}

    info = {"type": type(value).__name__}

    if hasattr(value, "size"):
        try:
            info["size"] = list(value.size)
        except Exception:
            pass

    if hasattr(value, "mode"):
        try:
            info["mode"] = value.mode
        except Exception:
            pass

    if isinstance(value, dict):
        info["dict_keys"] = list(value.keys())

    return info


def main():
    episode_files = sorted(ROOT.rglob("episode_*.parquet"))
    tasks = load_tasks(META_DIR / "tasks.jsonl")

    print("root:", ROOT)
    print("num_episode_files:", len(episode_files))
    print("num_tasks_loaded:", len(tasks))

    if len(episode_files) == 0:
        raise FileNotFoundError("No episode_*.parquet found.")

    summary = {
        "root": str(ROOT),
        "num_episode_files": len(episode_files),
        "num_tasks_loaded": len(tasks),
        "checked_episode_ids": [],
        "episodes": [],
    }

    cross_lines = []
    cross_lines.append(f"num_episode_files: {len(episode_files)}")
    cross_lines.append(f"num_tasks_loaded: {len(tasks)}")
    cross_lines.append("")

    for ep_id in CHECK_EPISODE_IDS:
        if ep_id >= len(episode_files):
            continue

        p = episode_files[ep_id]
        print("=" * 80)
        print("Loading:", p)

        ds = load_dataset("parquet", data_files=[str(p)], split="train")
        print("episode length:", len(ds))
        print("columns:", ds.column_names)

        if len(ds) == 0:
            continue

        sample_indices = sorted(set([0, len(ds) // 2, len(ds) - 1]))

        ep_report = {
            "episode_file": p.name,
            "episode_id_from_list": ep_id,
            "num_frames": len(ds),
            "columns": ds.column_names,
            "samples": [],
        }

        first = ds[0]
        task_index = first.get("task_index", None)
        task_text = tasks.get(int(task_index), None) if task_index is not None else None

        cross_lines.append(
            f"{p.name} -> episode_index={first.get('episode_index', ep_id)}, "
            f"task_index={task_index}, task_text={task_text}"
        )

        for idx in sample_indices:
            sample = ds[idx]
            state_key = "state" if "state" in sample else ("states" if "states" in sample else None)
            action_key = "actions" if "actions" in sample else ("action" if "action" in sample else None)

            item = {
                "sample_index_in_episode": idx,
                "episode_index": int(sample.get("episode_index", ep_id)),
                "frame_index": int(sample.get("frame_index", idx)),
                "global_index": int(sample.get("index", -1)) if sample.get("index", None) is not None else None,
                "task_index": int(sample.get("task_index", -1)) if sample.get("task_index", None) is not None else None,
                "task_text": tasks.get(int(sample.get("task_index", -1)), None)
                if sample.get("task_index", None) is not None
                else None,
                "state_key": state_key,
                "action_key": action_key,
                "image_info": image_info(sample.get("image", None)),
                "wrist_image_info": image_info(sample.get("wrist_image", None)),
            }

            if state_key is not None:
                item["state_shape"] = to_shape(sample[state_key])
                item["state_preview"] = brief_value(sample[state_key])

            if action_key is not None:
                item["action_shape"] = to_shape(sample[action_key])
                item["action_preview"] = brief_value(sample[action_key])

            ep_report["samples"].append(item)
            print(item)

        summary["checked_episode_ids"].append(ep_id)
        summary["episodes"].append(ep_report)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(CROSS_TASK_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(cross_lines) + "\n")

    print("=" * 80)
    print("Saved sample summary to:", SUMMARY_PATH)
    print("Saved cross episode task check to:", CROSS_TASK_LOG_PATH)


if __name__ == "__main__":
    main()
