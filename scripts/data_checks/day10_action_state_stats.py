from pathlib import Path
from datasets import load_dataset
import numpy as np
import json

root = Path("/root/autodl-tmp/cache/lerobot/physical-intelligence/libero")

out_dir = Path("logs/day10_action_state_stats")
out_dir.mkdir(parents=True, exist_ok=True)

summary_path = out_dir / "action_state_stats.json"
plot_path = out_dir / "action_state_stats.png"

episode_files = sorted(root.rglob("episode_*.parquet"))

print("num_episode_files:", len(episode_files))

if len(episode_files) == 0:
    raise FileNotFoundError("No episode_*.parquet found.")

# 低配机器先抽小样本，不要全量统计
check_episode_ids = [0, 1, 2, 10, 50, 100]
check_episode_ids = [i for i in check_episode_ids if i < len(episode_files)]

state_candidates = ["states", "state", "observation.state"]
action_candidates = ["actions", "action"]


def first_existing_key(sample, candidates):
    for key in candidates:
        if key in sample:
            return key
    return None


def to_1d_array(value):
    arr = np.asarray(value, dtype=np.float32)
    return arr.reshape(-1)


all_states = []
all_actions = []
episode_summaries = []

state_key_used = None
action_key_used = None

for ep_id in check_episode_ids:
    p = episode_files[ep_id]
    print("=" * 80)
    print("Loading:", p.name)

    ds = load_dataset("parquet", data_files=[str(p)], split="train")
    print("episode length:", len(ds))
    print("columns:", ds.column_names)

    if len(ds) == 0:
        continue

    first_sample = ds[0]

    state_key = first_existing_key(first_sample, state_candidates)
    action_key = first_existing_key(first_sample, action_candidates)

    state_key_used = state_key_used or state_key
    action_key_used = action_key_used or action_key

    print("state_key:", state_key)
    print("action_key:", action_key)

    ep_states = []
    ep_actions = []

    for i in range(len(ds)):
        sample = ds[i]

        if state_key is not None:
            ep_states.append(to_1d_array(sample[state_key]))

        if action_key is not None:
            ep_actions.append(to_1d_array(sample[action_key]))

    ep_item = {
        "episode_file": p.name,
        "episode_index": int(first_sample.get("episode_index", ep_id)),
        "num_frames": len(ds),
        "state_key": state_key,
        "action_key": action_key,
    }

    if len(ep_states) > 0:
        ep_states = np.stack(ep_states, axis=0)
        all_states.append(ep_states)
        ep_item["state_shape"] = list(ep_states.shape)
        ep_item["state_dim"] = int(ep_states.shape[1])

    if len(ep_actions) > 0:
        ep_actions = np.stack(ep_actions, axis=0)
        all_actions.append(ep_actions)
        ep_item["action_shape"] = list(ep_actions.shape)
        ep_item["action_dim"] = int(ep_actions.shape[1])

    episode_summaries.append(ep_item)
    print(ep_item)

result = {
    "checked_episode_ids": check_episode_ids,
    "num_checked_episodes": len(check_episode_ids),
    "state_key_used": state_key_used,
    "action_key_used": action_key_used,
    "episodes": episode_summaries,
}


def calc_stats(arr):
    return {
        "shape": list(arr.shape),
        "dim": int(arr.shape[1]),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "global_min": float(arr.min()),
        "global_max": float(arr.max()),
        "global_mean": float(arr.mean()),
        "global_std": float(arr.std()),
    }


if len(all_states) > 0:
    states = np.concatenate(all_states, axis=0)
    result["states"] = calc_stats(states)
    print("states shape:", states.shape)

if len(all_actions) > 0:
    actions = np.concatenate(all_actions, axis=0)
    result["actions"] = calc_stats(actions)
    print("actions shape:", actions.shape)

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("=" * 80)
print("Saved stats to:", summary_path)

try:
    import matplotlib.pyplot as plt

    if "actions" in result:
        action_mean = np.array(result["actions"]["mean"])
        action_std = np.array(result["actions"]["std"])
        dims = np.arange(len(action_mean))

        plt.figure(figsize=(8, 4))
        plt.bar(dims, action_mean, yerr=action_std)
        plt.xlabel("Action dimension")
        plt.ylabel("Mean ± std")
        plt.title("LIBERO action statistics")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print("Saved plot to:", plot_path)
    else:
        print("No action stats found, skip plot.")

except Exception as e:
    print("Plot skipped due to error:", repr(e))
