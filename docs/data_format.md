# Data Format
## Overview
本项目使用 LIBERO (LeRobot format) 数据集进行 openpi / π0.5 的训练与实验。
当前使用的数据集：
repo_id = physical-intelligence/libero
本地缓存路径：
/root/autodl-tmp/cache/lerobot/physical-intelligence/libero
当前本地数据已经是 LeRobot 可训练格式，无需重新转换。
# Dataset Structure
数据集主要由两部分组成：
physical-intelligence/libero/
├── episode_*.parquet
└── meta/
    ├── tasks.jsonl
    ├── episodes.jsonl
    ├── info.json
    └── stats.json
其中：
| File | Description |
|------|-------------|
| episode_*.parquet | 每条机器人轨迹 |
| tasks.jsonl | 语言任务列表 |
| episodes.jsonl | episode 信息 |
| info.json | 数据集信息 |
| stats.json | 数据统计信息 |
# Episode Fields
每个 episode parquet 包含以下字段：
| Field | Description |
|--------|-------------|
| image | 主视角 RGB 图像 |
| wrist_image | 手腕相机 RGB 图像 |
| state | 机器人状态（8维） |
| actions | 机器人动作（7维） |
| task_index | 语言任务编号 |
| timestamp | 时间戳 |
| frame_index | 当前轨迹帧编号 |
| episode_index | 当前轨迹编号 |
| index | 全局样本编号 |
注意：
实际字段名是 state，不是 states。
实际字段名是 actions，不是 action。
# Image Format
检查结果：
image       : RGB 256 × 256
wrist_image : RGB 256 × 256
两个相机均可以正常读取。
# State / Action Format
机器人状态：
state = 8 dimensions
机器人动作：
actions = 7 dimensions
Day10 抽样统计结果：
state shape   = [1908, 8]
actions shape = [1908, 7]
actions 数值范围：
global_min = -1.0
global_max = 1.0
# Language Task
episode 中保存的是 task_index。
语言任务来自 meta/tasks.jsonl。
映射关系：
task_index → meta/tasks.jsonl → task_text
训练时：
task_text ≈ prompt
# Training Format
当前 π0.5 的训练输入输出形式：
image + wrist_image + state + task_text → actions
# openpi Configuration
对应配置：
config = pi05_libero
主要配置：
| Item | Value |
|------|-------|
| repo_id | physical-intelligence/libero |
| prompt_from_task | True |
| model | Pi0Config(pi05=True) |
说明：
prompt_from_task=True 表示训练时语言输入由 task_index → task_text → prompt 自动生成。
# Current Status
目前已经完成：
✓ Dataset parser
✓ Image check
✓ Action / State statistics
✓ Clean check
✓ Data format alignment
目前数据可以正常用于后续 π0.5 小规模微调。
# Next Step
下一阶段：
Day13 Dataset Visualization（image grid、wrist image grid、action curve、state curve、task visualization）
Day14 Dataset / Model Survey
Day15 π0.5 Small-scale Fine-tuning