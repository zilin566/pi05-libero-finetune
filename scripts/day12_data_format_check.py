from pathlib import Path
from datasets import load_dataset
import json
import numpy as np

root = Path("/root/autodl-tmp/cache/lerobot/physical-intelligence/libero")
meta_dir = root / "meta"

out_dir = Path("logs/day12_data_format_check")
out_dir.mkdir(parents=True, exist_ok=True)

summary_path = out_dir / "day12_data_format_check.json"

repo_id = "physical-intelligence/libero"
local_path = str(root)

episode_files = sorted(root.rglob("episode_*.parquet"))

check_episode_ids = [0, 1, 2, 10, 50, 100]
check_episode_ids = [i for i in check_episode_ids if i < len(episode_files)]


def load_json(path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"load_error": repr(e)}


def load_jsonl_preview(path, max_lines=5):
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            if line.strip():
                items.append(json.loads(line))
    return items


def load_tasks(path):
    tasks = {}
    if not path.exists():
        return tasks
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            task_index = item.get("task_index", item.get("index", None))
            task_text = item.get("task", item.get("task_text", item.get("text", None)))
            if task_index is not None:
                tasks[int(task_index)] = task_text
    return tasks


def to_1d_array(x):
    return np.asarray(x, dtype=np.float32).reshape(-1)


def image_info(x):
    if x is None:
        return {"type": "None"}

    info = {"type": type(x).__name__}

    if hasattr(x, "size"):
        try:
            info["size"] = list(x.size)
        except Exception:
            pass

    if hasattr(x, "mode"):
        try:
            info["mode"] = x.mode
        except Exception:
            pass

    if isinstance(x, dict):
        info["dict_keys"] = list(x.keys())

    return info


tasks = load_tasks(meta_dir / "tasks.jsonl")

info_json = load_json(meta_dir / "info.json")
stats_json = load_json(meta_dir / "stats.json")

meta_files = {
    "tasks.jsonl": (meta_dir / "tasks.jsonl").exists(),
    "episodes.jsonl": (meta_dir / "episodes.jsonl").exists(),
    "info.json": (meta_dir / "info.json").exists(),
    "stats.json": (meta_dir / "stats.json").exists(),
}

summary = {
    "repo_id": repo_id,
    "local_path": local_path,
    "num_episode_files": len(episode_files),
    "meta_dir": str(meta_dir),
    "meta_files": meta_files,
    "num_tasks_loaded": len(tasks),
    "tasks_preview": load_jsonl_preview(meta_dir / "tasks.jsonl", max_lines=5),
    "episodes_preview": load_jsonl_preview(meta_dir / "episodes.jsonl", max_lines=3),
    "info_json_keys": list(info_json.keys()) if isinstance(info_json, dict) else None,
    "stats_json_keys": list(stats_json.keys()) if isinstance(stats_json, dict) else None,
    "checked_episode_ids": check_episode_ids,
    "episodes": [],
    "training_format": {
        "vision_inputs": ["image", "wrist_image"],
        "state_input": "state",
        "language_input": "task_index -> meta/tasks.jsonl -> task_text",
        "action_target": "actions",
        "formula": "image + wrist_image + state + task_text -> actions",
    },
}

for ep_id in check_episode_ids:
    p = episode_files[ep_id]
    print("=" * 80)
    print("Loading:", p.name)

    ds = load_dataset("parquet", data_files=[str(p)], split="train")
    first = ds[0]

    columns = ds.column_names

    task_index = first.get("task_index", None)
    task_text = None
    if task_index is not None:
        task_text = tasks.get(int(task_index), None)

    state = to_1d_array(first["state"]) if "state" in first else None
    actions = to_1d_array(first["actions"]) if "actions" in first else None

    item = {
        "episode_file": p.name,
        "episode_index": int(first.get("episode_index", ep_id)),
        "num_frames": len(ds),
        "columns": columns,
        "has_image": "image" in columns,
        "has_wrist_image": "wrist_image" in columns,
        "has_state": "state" in columns,
        "has_actions": "actions" in columns,
        "has_task_index": "task_index" in columns,
        "task_index": int(task_index) if task_index is not None else None,
        "task_text": task_text,
        "task_mapped": task_text is not None,
        "image_info": image_info(first.get("image", None)),
        "wrist_image_info": image_info(first.get("wrist_image", None)),
        "state_dim": int(state.shape[0]) if state is not None else None,
        "action_dim": int(actions.shape[0]) if actions is not None else None,
        "state_example": state.tolist() if state is not None else None,
        "action_example": actions.tolist() if actions is not None else None,
    }

    summary["episodes"].append(item)

    print({
        "episode_file": item["episode_file"],
        "num_frames": item["num_frames"],
        "columns": item["columns"],
        "task_index": item["task_index"],
        "task_mapped": item["task_mapped"],
        "state_dim": item["state_dim"],
        "action_dim": item["action_dim"],
        "image_info": item["image_info"],
        "wrist_image_info": item["wrist_image_info"],
    })

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("=" * 80)
print("Saved Day12 data format check to:", summary_path)
print("repo_id:", repo_id)
print("local_path:", local_path)
print("num_episode_files:", len(episode_files))
print("num_tasks_loaded:", len(tasks))
print("checked_episode_ids:", check_episode_ids)
print("training_format:", summary["training_format"]["formula"])
