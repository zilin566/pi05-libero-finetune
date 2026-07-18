from pathlib import Path
import re


CONFIG_PATH = Path("src/openpi/training/config.py")
SOURCE_NAME = "pi05_libero_day18_n50_s100"

NEW_CONFIGS = [
    ("pi05_libero_day18_n50_nonorm_smoke", 10, 5),
    ("pi05_libero_day18_n50_nonorm_s100", 100, 50),
]

EMPTY_ASSETS = (
    "/root/autodl-tmp/vla_work/openpi/"
    "assets/day18_no_norm_empty"
)


def extract_config(text: str, name: str) -> tuple[str, int]:
    marker = f'name="{name}",'
    marker_pos = text.find(marker)

    if marker_pos == -1:
        raise ValueError(f"找不到源配置：{name}")

    start = text.rfind("TrainConfig(", 0, marker_pos)
    if start == -1:
        raise ValueError("找不到 TrainConfig 起点")

    depth = 0

    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1

            if depth == 0:
                end = i + 1

                if end < len(text) and text[end] == ",":
                    end += 1

                return text[start:end], end

    raise ValueError("找不到 TrainConfig 终点")


def build_config(
    source: str,
    name: str,
    steps: int,
    save_interval: int,
) -> str:
    block = source.replace(
        f'name="{SOURCE_NAME}",',
        f'name="{name}",',
        1,
    )

    block = re.sub(
        r"num_train_steps=\d+,",
        f"num_train_steps={steps},",
        block,
        count=1,
    )

    block = re.sub(
        r"save_interval=\d+,",
        f"save_interval={save_interval},",
        block,
        count=1,
    )

    marker = "data=LeRobotLiberoDataConfig("

    replacement = (
        "data=LeRobotLiberoDataConfig(\n"
        "            assets=AssetsConfig(\n"
        f'                assets_dir="{EMPTY_ASSETS}",\n'
        '                asset_id="no_norm",\n'
        "            ),"
    )

    if marker not in block:
        raise ValueError("源配置中找不到 LeRobotLiberoDataConfig")

    block = block.replace(marker, replacement, 1)
    return block


def main() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    source, insert_pos = extract_config(text, SOURCE_NAME)

    blocks = []

    for name, steps, save_interval in NEW_CONFIGS:
        if f'name="{name}",' in text:
            print("已存在，跳过：", name)
            continue

        blocks.append(
            build_config(source, name, steps, save_interval)
        )

    if not blocks:
        print("没有新增配置")
        return

    insertion = "\n\n" + "\n\n".join(blocks)
    text = text[:insert_pos] + insertion + text[insert_pos:]

    CONFIG_PATH.write_text(text, encoding="utf-8")
    print("新增配置数量：", len(blocks))


if __name__ == "__main__":
    main()
