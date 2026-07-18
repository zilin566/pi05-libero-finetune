from pathlib import Path
from datasets import load_dataset
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import json

root = Path("/root/autodl-tmp/cache/lerobot/physical-intelligence/libero")
meta_dir = root / "meta"

out_dir = Path("logs/day13_visualize")
out_dir.mkdir(parents=True, exist_ok=True)

summary_path = out_dir / "visualize_summary.json"

# 低配机器先做小样本可视化，不做全量
check_episode_ids = [0, 1, 2, 10, 50, 100]


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


def ensure_rgb(img):
    if img is None:
        return Image.new("RGB", (256, 256), color=(0, 0, 0))
    if hasattr(img, "convert"):
        return img.convert("RGB")
    return Image.fromarray(np.asarray(img)).convert("RGB")


def make_image_grid(ds, episode_index, task_index, task_text, save_path):
    frame_ids = sorted(set([0, len(ds) // 2, len(ds) - 1]))

    cell_w, cell_h = 256, 256
    label_h = 40
    cols = len(frame_ids)
    rows = 2

    canvas = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for c, frame_id in enumerate(frame_ids):
        sample = ds[frame_id]
        frame_index = int(sample.get("frame_index", frame_id))

        main_img = ensure_rgb(sample.get("image", None)).resize((cell_w, cell_h))
        wrist_img = ensure_rgb(sample.get("wrist_image", None)).resize((cell_w, cell_h))

        x = c * cell_w

        y_main = 0
        canvas.paste(main_img, (x, y_main + label_h))
        draw.text((x + 5, y_main + 5), f"main | frame {frame_index}", fill=(0, 0, 0))

        y_wrist = cell_h + label_h
        canvas.paste(wrist_img, (x, y_wrist + label_h))
        draw.text((x + 5, y_wrist + 5), f"wrist | frame {frame_index}", fill=(0, 0, 0))

    # 顶部任务文本如果太长，不直接写进图里，避免太挤；保存在 summary json
    canvas.save(save_path)


def plot_curves(arr, title, ylabel, save_path):
    arr = np.asarray(arr, dtype=np.float32)

    plt.figure(figsize=(10, 4))
    for d in range(arr.shape[1]):
        plt.plot(arr[:, d], label=f"dim {d}")

    plt.xlabel("Frame index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    episode_files = sorted(root.rglob("episode_*.parquet"))
    tasks = load_tasks(meta_dir / "tasks.jsonl")

    print("num_episode_files:", len(episode_files))
    print("num_tasks_loaded:", len(tasks))

    if len(episode_files) == 0:
        raise FileNotFoundError("No episode_*.parquet found.")

    valid_episode_ids = [i for i in check_episode_ids if i < len(episode_files)]

    summary = {
        "root": str(root),
        "num_episode_files": len(episode_files),
        "num_tasks_loaded": len(tasks),
        "checked_episode_ids": valid_episode_ids,
        "output_dir": str(out_dir),
        "episodes": [],
    }

    for ep_id in valid_episode_ids:
        p = episode_files[ep_id]

        print("=" * 80)
        print("Loading:", p.name)

        ds = load_dataset("parquet", data_files=[str(p)], split="train")
        num_frames = len(ds)

        if num_frames == 0:
            continue

        first = ds[0]
        episode_index = int(first.get("episode_index", ep_id))
        task_index = int(first.get("task_index", -1)) if first.get("task_index", None) is not None else None
        task_text = tasks.get(task_index, None) if task_index is not None else None

        states = []
        actions = []

        for i in range(num_frames):
            sample = ds[i]
            states.append(to_1d_array(sample["state"]))
            actions.append(to_1d_array(sample["actions"]))

        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)

        grid_path = out_dir / f"episode_{episode_index:06d}_main_wrist_grid.png"
        action_curve_path = out_dir / f"episode_{episode_index:06d}_action_curve.png"
        state_curve_path = out_dir / f"episode_{episode_index:06d}_state_curve.png"

        make_image_grid(ds, episode_index, task_index, task_text, grid_path)
        plot_curves(
            actions,
            title=f"Episode {episode_index} Action Curve | task_index={task_index}",
            ylabel="Action value",
            save_path=action_curve_path,
        )
        plot_curves(
            states,
            title=f"Episode {episode_index} State Curve | task_index={task_index}",
            ylabel="State value",
            save_path=state_curve_path,
        )

        ep_item = {
            "episode_file": p.name,
            "episode_index": episode_index,
            "num_frames": num_frames,
            "task_index": task_index,
            "task_text": task_text,
            "state_shape": list(states.shape),
            "action_shape": list(actions.shape),
            "image_grid_path": str(grid_path),
            "action_curve_path": str(action_curve_path),
            "state_curve_path": str(state_curve_path),
            "state_min": states.min(axis=0).tolist(),
            "state_max": states.max(axis=0).tolist(),
            "action_min": actions.min(axis=0).tolist(),
            "action_max": actions.max(axis=0).tolist(),
        }

        summary["episodes"].append(ep_item)

        print({
            "episode_file": p.name,
            "episode_index": episode_index,
            "num_frames": num_frames,
            "task_index": task_index,
            "task_text": task_text,
            "state_shape": list(states.shape),
            "action_shape": list(actions.shape),
            "saved": [
                str(grid_path),
                str(action_curve_path),
                str(state_curve_path),
            ],
        })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Saved visualize summary to:", summary_path)
    print("Saved visualizations to:", out_dir)


if __name__ == "__main__":
    main()
