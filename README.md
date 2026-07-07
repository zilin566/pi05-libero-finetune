# pi05-libero-finetune
基于 openpi / π0.5 的 LIBERO 数据理解、LeRobot 数据格式分析、训练配置阅读与后续微调实验项目。
本项目目标不是重新设计 VLA 模型，而是完成一个清晰的工程闭环：
LIBERO 数据理解 → LeRobot 数据格式分析 → openpi / pi05_libero 训练配置阅读 → 数据检查与可视化 → π0.5 小规模微调 → 仿真评测 → 简历项目包装
## 1. Project Goal
本项目用于复现和理解 π0.5 在 LIBERO 数据上的训练流程，重点关注：
1. LIBERO 数据从哪里来
2. LIBERO 如何以 LeRobot 格式被 openpi 读取
3. openpi 中 pi05_libero 配置如何对齐数据
4. π0.5 训练时使用哪些字段
5. 如何进行小规模 debug training / LoRA fine-tuning
6. 如何记录训练日志、数据检查结果和评测结果
本项目对标 VLA / 具身智能实习岗位中的以下能力：
VLA 模型复现与微调
多模态机器人数据处理
LeRobot / LIBERO 数据格式理解
openpi / π0.5 训练流程理解
仿真评测与实验分析
## 2. Data Pipeline
LIBERO 数据进入 π0.5 训练流程的核心链路为：
openvla/modified_libero_rlds → examples/libero/convert_libero_data_to_lerobot.py → physical-intelligence/libero → pi05_libero → scripts/train.py
说明：
| Component | Meaning |
|---|---|
| openvla/modified_libero_rlds | 原始 LIBERO / RLDS 数据 |
| convert_libero_data_to_lerobot.py | LIBERO 到 LeRobot 格式的转换脚本 |
| physical-intelligence/libero | 已转换好的 LeRobot 格式 LIBERO 数据集 |
| pi05_libero | openpi 中 π0.5 + LIBERO 的训练配置 |
| scripts/train.py | openpi 训练入口 |
当前项目没有重新生成新的 converted_dataset/ 目录，而是复用已有的本地 LeRobot 缓存：
/root/autodl-tmp/cache/lerobot/physical-intelligence/libero
该目录已经包含 episode parquet 和 meta 文件，因此可以视为当前可训练数据路径。
## 3. Dataset Source
当前使用的数据集 repo_id 为：
physical-intelligence/libero
本地缓存路径为：
/root/autodl-tmp/cache/lerobot/physical-intelligence/libero
当前检查结果：
num_episode_files = 1693
num_tasks_loaded  = 40
metadata 文件包括：
meta/tasks.jsonl
meta/episodes.jsonl
meta/info.json
meta/stats.json
## 4. Episode Parquet Format
每个 episode parquet 中包含以下字段：
image
wrist_image
state
actions
timestamp
frame_index
episode_index
index
task_index
核心训练字段如下：
| Field | Meaning |
|---|---|
| image | 主视角 RGB 图像 |
| wrist_image | 手腕相机 RGB 图像 |
| state | 机器人状态，8 维 |
| actions | 机器人动作标签，7 维 |
| task_index | 语言任务编号 |
| timestamp | 时间戳 |
| frame_index | 当前 episode 内的帧编号 |
| episode_index | 当前轨迹编号 |
| index | 全局样本编号 |
注意：
实际状态字段名是 state，不是 states。
实际动作字段名是 actions，不是 action。
## 5. Image / State / Action Format
图像字段检查结果：
image       = 256 × 256 RGB
wrist_image = 256 × 256 RGB
状态与动作字段检查结果：
state   = 8 dimensions
actions = 7 dimensions
Day10 抽查 6 个 episode，共 1908 帧，确认：
state shape   = [1908, 8]
actions shape = [1908, 7]
actions 数值范围：
actions global_min = -1.0
actions global_max = 1.0
这说明抽样数据中的 action 字段不是空动作、不是全 0，也没有明显异常极值。
## 6. Language Task / Prompt Mapping
episode parquet 中直接保存的是 task_index。
对应的语言任务文本保存在 meta/tasks.jsonl。
映射关系为：
task_index → meta/tasks.jsonl → task_text
在 openpi / π0.5 中，prompt 可以理解为模型接收的语言任务指令。
当前项目中：
task_text ≈ prompt
openpi 的 pi05_libero 配置中使用 prompt_from_task=True，这表示训练时的语言 prompt 会从 LIBERO 的 task 信息中自动构造。
## 7. Current Training Format
当前 VLA 训练输入输出形式可以总结为：
image + wrist_image + state + task_text → actions
其中：
image       = 主视角图像
wrist_image = 手腕相机图像
state       = 8 维机器人状态
task_text   = 由 task_index 映射得到的语言任务指令
actions     = 7 维机器人动作标签
模型学习的是：在当前视觉、机器人状态和语言任务条件下，预测下一步机器人动作。
## 8. pi05_libero Config
openpi 中与 LIBERO 对应的训练配置为：
| Config Item | Value |
|---|---|
| config name | pi05_libero |
| model | pi0.Pi0Config(pi05=True) |
| data config | LeRobotLiberoDataConfig |
| repo_id | physical-intelligence/libero |
| prompt_from_task | True |
| action_horizon | 10 |
| batch_size | 256 |
| num_train_steps | 30000 |
这说明 openpi 中的 pi05_libero 配置与当前本地 LIBERO / LeRobot 数据缓存是对齐的。
## 9. Data Check Progress
当前已经完成的数据检查流程：
Day8  Dataset parser
Day9  Image / wrist image check
Day10 Action / state statistics
Day11 Clean check
Day12 Data format alignment
### Day8: Dataset Parser
已确认：
1. 本地 LIBERO cache 可以读取
2. episode parquet 文件存在
3. task_index 可以通过 meta/tasks.jsonl 映射到 task_text
4. sample_index / frame_index / episode_index / task_index 含义已区分
### Day9: Image Check
已确认：
1. image 可以正常读取
2. wrist_image 可以正常读取
3. 图像格式为 256×256 RGB
4. 图像不是空图、黑图或乱码图
### Day10: Action / State Statistics
已确认：
state   = 8 维
actions = 7 维
actions 数值范围大致位于 [-1, 1]
### Day11: Clean Check
Day11 抽查 6 个 LIBERO episode，共 1908 帧。
未发现明显异常：
missing_main_images   = 0
missing_wrist_images  = 0
nan_state_frames      = 0
inf_state_frames      = 0
nan_action_frames     = 0
inf_action_frames     = 0
zero_action_frames    = 0
action_outlier_frames = 0
state_outlier_frames  = 0
num_flagged_episodes  = 0
### Day12: Data Format Alignment
已确认：
1. repo_id = physical-intelligence/libero
2. local_path = /root/autodl-tmp/cache/lerobot/physical-intelligence/libero
3. episode parquet 字段与训练输入对齐
4. pi05_libero 使用 LeRobotLiberoDataConfig
5. prompt_from_task=True
6. 当前训练格式为 image + wrist_image + state + task_text → actions
## 10. Training Debug Pipeline
Day6 已经成功将训练 debug 配置扩展到 100 step。
当前可行 debug 配置：
config = pi05_libero_debug_lora
batch_size = 1
max_token_len = 64
num_train_steps = 100
24GB 显存下当前可行方案：
LoRA + batch_size=1 + max_token_len=64
该阶段目标不是追求指标，而是确认：
1. 训练入口可以启动
2. pi05_libero 配置可以读取
3. 数据字段可以正常进入训练
4. loss 可以正常记录
5. 显存占用可控
## 11. Repository Structure
当前项目结构规划：
pi05-libero-finetune/
├── README.md
├── notes/
│   ├── day1.md
│   ├── day2_libero_data.md
│   ├── day8_sample_check.md
│   ├── day9_image_check.md
│   ├── day10_action_state_stats.md
│   ├── day11_clean_check.md
│   └── day12_data_format.md
├── scripts/
│   ├── day8_dataset_parser.py
│   ├── day9_image_check.py
│   ├── day10_action_state_stats.py
│   ├── day11_clean_check.py
│   └── day12_data_format_check.py
├── docs/
│   └── data_format.md
├── logs/
│   ├── day8_samples/
│   ├── day9_images/
│   ├── day10_action_state_stats/
│   ├── day11_clean_check/
│   └── day12_data_format_check/
├── experiments/
├── assets/
└── .gitignore
## 12. Notes
学习记录和每日检查结果放在 notes/。
脚本输出和中间日志放在 logs/。
项目说明文档放在 docs/。
实验结果、训练日志、评测结果后续放在 experiments/。
## 13. Next Steps
后续计划：
Day13: 数据验证与可视化展示（image / wrist_image grid、action curve、state curve、task-aligned samples）
Day14: VLA 数据集和模型速览（LIBERO / OXE / DROID / GO-1；RT-1 / RT-2 / OpenVLA / π0.5 / RDT / GR00T / π0.6）
Day15: π0.5 小规模微调（small batch、LoRA / SFT debug training、first_train_log.txt）
## 14. Project Positioning
本项目定位为：openpi / π0.5 + LIBERO 的数据流、训练配置、微调与评测闭环复现项目。
当前阶段重点不是重新设计模型，而是证明：
1. 数据能读
2. 图像能看
3. state / action 能统计
4. 异常样本能检查
5. 数据格式能解释
6. openpi 配置能对齐
7. 后续可以进入小规模微调




