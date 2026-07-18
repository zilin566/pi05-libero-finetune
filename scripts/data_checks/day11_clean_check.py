from pathlib import Path
from datasets import load_dataset
import numpy as np
import json

root = Path("/root/autodl-tmp/cache/lerobot/physical-intelligence/libero")
meta_dir = root / "meta"

out_dir = Path("logs/day11_clean_check")
out_dir.mkdir(parents=True, exist_ok=True)

summary_path = out_dir / "clean_check_summary.json"

episode_files = sorted(root.rglob("episode_*.parquet"))

print("num_episode_files:", len(episode_files))

if len(episode_files) == 0:
    raise FileNotFoundError("No episode_*.parquet found.")

# 低配机器先做小样本，不要全量检查
check_episode_ids = [0, 1, 2, 10, 50, 100]
check_episode_ids = [i for i in check_episode_ids if i < len(episode_files)]

# 清洗规则阈值：先保守设置，只用于发现问题，不直接删除数据
MIN_EPISODE_LEN = 50
ZERO_ACTION_EPS = 1e-6
ALL_ZERO_RATIO_WARN = 0.95
ACTION_ABS_WARN = 1.05
STATE_ABS_WARN = 10.0

state_candidates = ["state", "states", "observation.state"]
action_candidates = ["actions", "action"]
main_image_candidates = ["image", "observation.image", "main_image"]
wrist_image_candidates = ["wrist_image", "observation.wrist_image"]


def first_existing_key(sample, candidates):
    for key in candidates:
        if key in sample:
            return key
    return None


def to_1d_array(value):
    arr = np.asarray(value, dtype=np.float32)
    return arr.reshape(-1)


def is_missing_image(value):
    if value is None:
        return True
    # datasets Image feature 有时返回 PIL Image；正常不为空即可
    # 有时也可能是 dict，如 {"path": ..., "bytes": ...}
    if isinstance(value, dict):
        if value.get("bytes", "exists") is None and value.get("path", "exists") is None:
            return True
    return False


def load_tasks(meta_dir):
    tasks_path = meta_dir / "tasks.jsonl"
    tasks = {}
    if not tasks_path.exists():
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


tasks = load_tasks(meta_dir)
print("num_tasks_loaded:", len(tasks))

episode_reports = []

total_frames = 0
total_missing_main_images = 0
total_missing_wrist_images = 0
total_nan_state_frames = 0
total_inf_state_frames = 0
total_nan_action_frames = 0
total_inf_action_frames = 0
total_zero_action_frames = 0
total_action_outlier_frames = 0
total_state_outlier_frames = 0

flagged_episodes = []

for ep_id in check_episode_ids:
    p = episode_files[ep_id]
    print("=" * 80)
    print("Loading:", p.name)

    ds = load_dataset("parquet", data_files=[str(p)], split="train")
    num_frames = len(ds)
    total_frames += num_frames

    print("episode length:", num_frames)
    print("columns:", ds.column_names)

    if num_frames == 0:
        report = {
            "episode_file": p.name,
            "episode_id_from_list": ep_id,
            "num_frames": 0,
            "flags": ["empty_episode"],
        }
        episode_reports.append(report)
        flagged_episodes.append(report)
        continue

    first_sample = ds[0]

    state_key = first_existing_key(first_sample, state_candidates)
    action_key = first_existing_key(first_sample, action_candidates)
    main_image_key = first_existing_key(first_sample, main_image_candidates)
    wrist_image_key = first_existing_key(first_sample, wrist_image_candidates)

    task_index = first_sample.get("task_index", None)
    task_text = None
    task_mapped = False

    if task_index is not None:
        try:
            task_index_int = int(task_index)
            task_text = tasks.get(task_index_int, None)
            task_mapped = task_text is not None
        except Exception:
            task_index_int = None
    else:
        task_index_int = None

    missing_main_images = 0
    missing_wrist_images = 0

    nan_state_frames = 0
    inf_state_frames = 0
    nan_action_frames = 0
    inf_action_frames = 0

    zero_action_frames = 0
    action_outlier_frames = 0
    state_outlier_frames = 0

    action_values = []
    state_values = []

    for i in range(num_frames):
        sample = ds[i]

        if main_image_key is None or is_missing_image(sample.get(main_image_key, None)):
            missing_main_images += 1

        if wrist_image_key is None or is_missing_image(sample.get(wrist_image_key, None)):
            missing_wrist_images += 1

        if state_key is not None:
            state = to_1d_array(sample[state_key])
            state_values.append(state)

            if np.isnan(state).any():
                nan_state_frames += 1

            if np.isinf(state).any():
                inf_state_frames += 1

            if np.abs(state).max() > STATE_ABS_WARN:
                state_outlier_frames += 1

        if action_key is not None:
            action = to_1d_array(sample[action_key])
            action_values.append(action)

            if np.isnan(action).any():
                nan_action_frames += 1

            if np.isinf(action).any():
                inf_action_frames += 1

            if np.linalg.norm(action) < ZERO_ACTION_EPS:
                zero_action_frames += 1

            if np.abs(action).max() > ACTION_ABS_WARN:
                action_outlier_frames += 1

    flags = []

    if num_frames < MIN_EPISODE_LEN:
        flags.append("short_episode")

    if state_key is None:
        flags.append("missing_state_key")

    if action_key is None:
        flags.append("missing_action_key")

    if main_image_key is None:
        flags.append("missing_main_image_key")

    if wrist_image_key is None:
        flags.append("missing_wrist_image_key")

    if missing_main_images > 0:
        flags.append("missing_main_image_frame")

    if missing_wrist_images > 0:
        flags.append("missing_wrist_image_frame")

    if nan_state_frames > 0:
        flags.append("nan_state")

    if inf_state_frames > 0:
        flags.append("inf_state")

    if nan_action_frames > 0:
        flags.append("nan_action")

    if inf_action_frames > 0:
        flags.append("inf_action")

    if action_outlier_frames > 0:
        flags.append("action_outlier")

    if state_outlier_frames > 0:
        flags.append("state_outlier")

    zero_action_ratio = zero_action_frames / max(num_frames, 1)
    if zero_action_ratio > ALL_ZERO_RATIO_WARN:
        flags.append("mostly_zero_action")

    if task_index is None:
        flags.append("missing_task_index")
    elif not task_mapped:
        flags.append("task_index_not_mapped")

    report = {
        "episode_file": p.name,
        "episode_id_from_list": ep_id,
        "episode_index": int(first_sample.get("episode_index", ep_id)),
        "num_frames": num_frames,
        "task_index": task_index_int,
        "task_text": task_text,
        "task_mapped": task_mapped,
        "state_key": state_key,
        "action_key": action_key,
        "main_image_key": main_image_key,
        "wrist_image_key": wrist_image_key,
        "missing_main_images": missing_main_images,
        "missing_wrist_images": missing_wrist_images,
        "nan_state_frames": nan_state_frames,
        "inf_state_frames": inf_state_frames,
        "nan_action_frames": nan_action_frames,
        "inf_action_frames": inf_action_frames,
        "zero_action_frames": zero_action_frames,
        "zero_action_ratio": zero_action_ratio,
        "action_outlier_frames": action_outlier_frames,
        "state_outlier_frames": state_outlier_frames,
        "flags": flags,
    }

    if len(action_values) > 0:
        action_arr = np.stack(action_values, axis=0)
        report["action_shape"] = list(action_arr.shape)
        report["action_min"] = action_arr.min(axis=0).tolist()
        report["action_max"] = action_arr.max(axis=0).tolist()
        report["action_std"] = action_arr.std(axis=0).tolist()

    if len(state_values) > 0:
        state_arr = np.stack(state_values, axis=0)
        report["state_shape"] = list(state_arr.shape)
        report["state_min"] = state_arr.min(axis=0).tolist()
        report["state_max"] = state_arr.max(axis=0).tolist()
        report["state_std"] = state_arr.std(axis=0).tolist()

    episode_reports.append(report)

    if len(flags) > 0:
        flagged_episodes.append(report)

    total_missing_main_images += missing_main_images
    total_missing_wrist_images += missing_wrist_images
    total_nan_state_frames += nan_state_frames
    total_inf_state_frames += inf_state_frames
    total_nan_action_frames += nan_action_frames
    total_inf_action_frames += inf_action_frames
    total_zero_action_frames += zero_action_frames
    total_action_outlier_frames += action_outlier_frames
    total_state_outlier_frames += state_outlier_frames

    print("report:", {
        "episode_file": report["episode_file"],
        "num_frames": report["num_frames"],
        "task_index": report["task_index"],
        "task_mapped": report["task_mapped"],
        "flags": report["flags"],
    })

summary = {
    "checked_episode_ids": check_episode_ids,
    "num_checked_episodes": len(check_episode_ids),
    "total_frames": total_frames,
    "thresholds": {
        "MIN_EPISODE_LEN": MIN_EPISODE_LEN,
        "ZERO_ACTION_EPS": ZERO_ACTION_EPS,
        "ALL_ZERO_RATIO_WARN": ALL_ZERO_RATIO_WARN,
        "ACTION_ABS_WARN": ACTION_ABS_WARN,
        "STATE_ABS_WARN": STATE_ABS_WARN,
    },
    "totals": {
        "missing_main_images": total_missing_main_images,
        "missing_wrist_images": total_missing_wrist_images,
        "nan_state_frames": total_nan_state_frames,
        "inf_state_frames": total_inf_state_frames,
        "nan_action_frames": total_nan_action_frames,
        "inf_action_frames": total_inf_action_frames,
        "zero_action_frames": total_zero_action_frames,
        "action_outlier_frames": total_action_outlier_frames,
        "state_outlier_frames": total_state_outlier_frames,
    },
    "num_flagged_episodes": len(flagged_episodes),
    "flagged_episodes": [
        {
            "episode_file": r["episode_file"],
            "episode_index": r.get("episode_index"),
            "flags": r["flags"],
        }
        for r in flagged_episodes
    ],
    "episodes": episode_reports,
}

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("=" * 80)
print("Saved clean check summary to:", summary_path)
print("num_checked_episodes:", summary["num_checked_episodes"])
print("total_frames:", summary["total_frames"])
print("num_flagged_episodes:", summary["num_flagged_episodes"])
print("totals:", summary["totals"])

if len(flagged_episodes) == 0:
    print("Clean check passed on sampled episodes: no flags found.")
else:
    print("Flagged episodes:")
    for item in summary["flagged_episodes"]:
        print(item)
