from pathlib import Path


CONFIG_PATH = Path("src/openpi/training/config.py")
SOURCE_NAME = "pi05_libero_debug_lora"


def extract_train_config(text: str, config_name: str) -> tuple[str, int]:
    marker = f'name="{config_name}",'
    marker_pos = text.find(marker)

    if marker_pos == -1:
        raise ValueError(f"找不到配置：{config_name}")

    start = text.rfind("TrainConfig(", 0, marker_pos)

    if start == -1:
        raise ValueError("无法定位 TrainConfig 起始位置")

    depth = 0
    started = False
    end = None

    for i in range(start, len(text)):
        char = text[i]

        if char == "(":
            depth += 1
            started = True
        elif char == ")":
            depth -= 1

            if started and depth == 0:
                end = i + 1

                if end < len(text) and text[end] == ",":
                    end += 1

                break

    if end is None:
        raise ValueError("无法定位 TrainConfig 结束位置")

    return text[start:end], end


def make_config(
    source_block: str,
    name: str,
    repo_id: str,
    train_steps: int,
    save_interval: int,
) -> str:
    block = source_block.replace(
        f'name="{SOURCE_NAME}",',
        f'name="{name}",',
        1,
    )

    block = block.replace(
        'repo_id="physical-intelligence/libero",',
        f'repo_id="{repo_id}",',
        1,
    )

    block = block.replace(
        "num_train_steps=10,",
        f"num_train_steps={train_steps},",
        1,
    )

    block = block.replace(
        "save_interval=5,",
        f"save_interval={save_interval},",
        1,
    )

    return block


def main() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    source_block, insert_pos = extract_train_config(text, SOURCE_NAME)

    generated = []

    for size in [10, 50, 100]:
        repo_id = f"physical-intelligence/libero_day18_n{size}"

        specifications = [
            (
                f"pi05_libero_day18_n{size}_smoke",
                10,
                5,
            ),
            (
                f"pi05_libero_day18_n{size}_s100",
                100,
                50,
            ),
        ]

        for name, steps, save_interval in specifications:
            if f'name="{name}",' in text:
                print(f"配置已存在，跳过：{name}")
                continue

            generated.append(
                make_config(
                    source_block=source_block,
                    name=name,
                    repo_id=repo_id,
                    train_steps=steps,
                    save_interval=save_interval,
                )
            )

    if not generated:
        print("没有需要新增的配置")
        return

    insertion = "\n\n" + "\n\n".join(generated)
    text = text[:insert_pos] + insertion + text[insert_pos:]

    CONFIG_PATH.write_text(text, encoding="utf-8")
    print(f"新增配置数量：{len(generated)}")


if __name__ == "__main__":
    main()
