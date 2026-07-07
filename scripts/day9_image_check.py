from pathlib import Path
from datasets import load_dataset
import json

ROOT = Path("/root/autodl-tmp/cache/lerobot/physical-intelligence/libero")
OUT_DIR = Path("logs/day9_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = Path("logs/day9_image_check_summary.json")

CHECK_EPISODE_IDS = [0, 1, 2, 10, 50, 100]


def image_info(img):
    if img is None:
        return {"type": "None", "missing": True}

    info = {"type": type(img).__name__, "missing": False}

    if hasattr(img, "size"):
        try:
            info["size"] = list(img.size)
        except Exception:
            pass

    if hasattr(img, "mode"):
        try:
            info["mode"] = img.mode
        except Exception:
            pass

    if isinstance(img, dict):
        info["dict_keys"] = list(img.keys())

    return info


def save_image(img, path: Path):
    if img is None:
        return False

    # datasets Image feature usually returns PIL.Image.Image
    if hasattr(img, "save"):
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return True

    # fallback for dict-style image objects
    if isinstance(img, dict) and img.get("bytes") is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(img["bytes"])
        return True

    return False


def main():
    episode_files = sorted(ROOT.rglob("episode_*.parquet"))
    print("num_episode_files:", len(episode_files))

    if len(episode_files) == 0:
        raise FileNotFoundError("No episode_*.parquet found.")

    summary = {
        "root": str(ROOT),
        "num_episode_files": len(episode_files),
        "checked_episode_ids": [],
        "saved_images_dir": str(OUT_DIR),
        "episodes": [],
    }

    for ep_id in CHECK_EPISODE_IDS:
        if ep_id >= len(episode_files):
            continue

        p = episode_files[ep_id]
        print("=" * 80)
        print("Loading:", p.name)

        ds = load_dataset("parquet", data_files=[str(p)], split="train")
        print("episode length:", len(ds))
        print("columns:", ds.column_names)

        if len(ds) == 0:
            continue

        image_key = "image" if "image" in ds.column_names else None
        wrist_key = "wrist_image" if "wrist_image" in ds.column_names else None

        sample_indices = sorted(set([0, len(ds) // 2, len(ds) - 1]))

        ep_report = {
            "episode_file": p.name,
            "episode_id_from_list": ep_id,
            "num_frames": len(ds),
            "main_image_key": image_key,
            "wrist_image_key": wrist_key,
            "frames": [],
        }

        for idx in sample_indices:
            sample = ds[idx]

            frame_index = int(sample.get("frame_index", idx))
            episode_index = int(sample.get("episode_index", ep_id))
            task_index = int(sample.get("task_index", -1)) if sample.get("task_index", None) is not None else None

            main_img = sample.get(image_key, None) if image_key else None
            wrist_img = sample.get(wrist_key, None) if wrist_key else None

            main_path = OUT_DIR / f"episode_{episode_index:06d}_frame_{frame_index:06d}_main.png"
            wrist_path = OUT_DIR / f"episode_{episode_index:06d}_frame_{frame_index:06d}_wrist.png"

            main_saved = save_image(main_img, main_path)
            wrist_saved = save_image(wrist_img, wrist_path)

            item = {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "task_index": task_index,
                "main_image_info": image_info(main_img),
                "wrist_image_info": image_info(wrist_img),
                "main_image_saved": main_saved,
                "wrist_image_saved": wrist_saved,
                "main_image_path": str(main_path) if main_saved else None,
                "wrist_image_path": str(wrist_path) if wrist_saved else None,
            }

            ep_report["frames"].append(item)
            print(item)

        summary["checked_episode_ids"].append(ep_id)
        summary["episodes"].append(ep_report)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Saved image check summary to:", SUMMARY_PATH)
    print("Saved images to:", OUT_DIR)


if __name__ == "__main__":
    main()
