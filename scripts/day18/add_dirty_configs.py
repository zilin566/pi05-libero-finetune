from pathlib import Path
import re


CONFIG_PATH = Path("src/openpi/training/config.py")

SOURCE_CONFIG = "pi05_libero_day18_n50_s100"
SOURCE_REPO = "physical-intelligence/libero_day18_n50"
DIRTY_REPO = "physical-intelligence/libero_day18_n50_dirty_finite"

NEW_CONFIGS = [
    ("pi05_libero_day18_n50_dirty_smoke", 10, 5),
    ("pi05_libero_day18_n50_dirty_s100", 100, 50),
]


def extract_train_config(text: str, config_name: str) -> tuple[str, int]:
    marker = f'name="{config_name}",'
    marker_pos = text.find(marker)

    if marker_pos == -1:
        raise ValueError(f"找不到源配置：{config_name}")

    start = text.rfind("TrainConfig(", 0, marker_pos)

    if start == -1:
        raise ValueError("找不到 TrainConfig 起点")

    depth = 0

    for index in range(start, len(text)):
        char = text[index]

        if char == "(":
            depth += 1

        elif char == ")":
            depth -= 1

            if depth == 0:
                end = index + 1

                if end < len(text) and text[end] == ",":
                    end += 1

                return text[start:end], end

    raise ValueError("找不到 TrainConfig 终点")


def build_config(
    source_block: str,
    new_name: str,
    train_steps: int,
    save_interval: int,
) -> str:
    block = source_block.replace(
        f'name="{SOURCE_CONFIG}",',
        f'name="{new_name}",',
        1,
    )

    if f'repo_id="{SOURCE_REPO}",' not in block:
        raise ValueError("源配置中的 repo_id 与预期不一致")

    block = block.replace(
        f'repo_id="{SOURCE_REPO}",',
        f'repo_id="{DIRTY_REPO}",',
        1,
    )

    block = re.sub(
        r"num_train_steps=\d+,",
        f"num_train_steps={train_steps},",
        block,
        count=1,
    )

    block = re.sub(
        r"save_interval=\d+,",
        f"save_interval={save_interval},",
        block,
        count=1,
    )

    return block


def main() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    source_block, insert_pos = extract_train_config(
        text,
        SOURCE_CONFIG,
    )

    generated = []

    for name, steps, interval in NEW_CONFIGS:
        if f'name="{name}",' in text:
            print(f"配置已存在，跳过：{name}")
            continue

        generated.append(
            build_config(
                source_block,
                name,
                steps,
                interval,
            )
        )

    if not generated:
        print("没有新增配置")
        return

    insertion = "\n\n" + "\n\n".join(generated)
    text = text[:insert_pos] + insertion + text[insert_pos:]

    CONFIG_PATH.write_text(text, encoding="utf-8")
    print(f"新增配置数量：{len(generated)}")


if __name__ == "__main__":
    main()
