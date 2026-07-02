# pi05-libero-finetune

基于 **openpi / π0.5** 的 LIBERO 数据理解、LeRobot 数据格式分析、训练配置阅读与后续微调实验项目。

本项目目标不是重新设计 VLA 模型，而是完成一个清晰的工程闭环：

LIBERO 数据理解
→ LeRobot 格式转换
→ π0.5 训练配置分析
→ 小规模微调
→ 仿真评测
→ 简历项目包装
```

## Project Goal

本项目用于复现和理解 π0.5 在 LIBERO 数据上的训练流程，重点关注：

- LIBERO 数据从哪里来
- LIBERO 如何转换成 LeRobot 格式
- openpi 中 `pi05_libero` 配置如何读取数据
- π0.5 训练时使用哪些字段
- 后续如何进行小规模微调与评测

---

## Current Progress

| Day | Topic | Status |
|---|---|---|
| Day1 | openpi 项目结构与官方流程理解 | Done |
| Day2 | LIBERO 数据流与 LeRobot 格式理解 | Done |
| Day3 | π0.5 训练配置与 norm stats 准备 | Todo |
| Day4+ | 小规模训练、日志记录与评测 | Todo |

---

## Data Pipeline

LIBERO 数据进入 π0.5 训练流程的核心链路：

```text
openvla/modified_libero_rlds
→ examples/libero/convert_libero_data_to_lerobot.py
→ physical-intelligence/libero
→ pi05_libero
→ scripts/train.py
```

说明：

| Item | Meaning |
|---|---|
| `openvla/modified_libero_rlds` | 原始 LIBERO / RLDS 数据 |
| `convert_libero_data_to_lerobot.py` | LIBERO 到 LeRobot 格式的转换脚本 |
| `physical-intelligence/libero` | 已转换好的 LeRobot 格式 LIBERO 数据集 |
| `pi05_libero` | openpi 中 π0.5 + LIBERO 的训练配置 |
| `train.py` | openpi 训练入口 |

---

## LeRobot Data Fields

转换后的 LeRobot 数据主要字段如下：

| Field | Meaning | Shape |
|---|---|---|
| `image` | 主视角图像 | `256 x 256 x 3` |
| `wrist_image` | 腕部相机图像 | `256 x 256 x 3` |
| `state` | 机器人状态 | `8` |
| `actions` | 机器人动作 | `7` |
| `task` | 语言指令 | string |

π0.5 的训练形式可以理解为：

```text
image + wrist_image + state + task → actions
```

也就是：

```text
视觉观察 + 机器人状态 + 语言指令 → 机器人动作
```

---

## pi05_libero Config

当前确认的 `pi05_libero` 关键信息：

| Config Item | Value |
|---|---|
| config name | `pi05_libero` |
| model | `pi0.Pi0Config(pi05=True)` |
| repo_id | `physical-intelligence/libero` |
| prompt_from_task | `True` |
| action_horizon | `10` |
| batch_size | `256` |
| num_train_steps | `30000` |

结论：

```text
pi05_libero 默认读取 physical-intelligence/libero，
即已经转换成 LeRobot 格式的 LIBERO 数据集。
```

---

## Repository Structure

```text
pi05-libero-finetune/
├── README.md
├── notes/
│   ├── day1_openpi_reading.md
│   └── day2_libero_data.md
├── scripts/
├── configs/
├── docs/
├── experiments/
├── logs/
├── assets/
└── .gitignore
```

---

## Notes

学习记录放在 `notes/` 目录下：

| File | Content |
|---|---|
| `day1_openpi_reading.md` | openpi 项目结构、官方 pipeline 理解 |
| `day2_libero_data.md` | LIBERO 数据流、LeRobot 字段、pi05_libero 数据来源 |

---

## Next Steps

后续计划：

- 阅读 `LeRobotLiberoDataConfig`
- 理解 `pi05_libero` 完整训练配置
- 尝试运行 `compute_norm_stats.py --help`
- 准备小规模 debug 训练命令
- 编写数据检查脚本
- 记录训练日志、loss 曲线和失败案例

---

## Project Positioning

本项目对标 VLA / 具身智能实习岗位中的以下要求：

- VLA 模型复现与微调
- 多模态机器人数据处理
- LeRobot / LIBERO 数据格式理解
- openpi / π0.5 训练流程理解
- 仿真评测与实验分析