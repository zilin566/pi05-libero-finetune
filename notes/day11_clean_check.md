# Day11：LIBERO 数据清洗检查
Day11 主要做 LIBERO / LeRobot 数据的 clean check。本次不训练模型，不使用 GPU，也不修改原始数据，只检查样本中是否存在明显异常，并为后续清洗规则、数据格式整理和 π0.5 小规模微调做准备。
Day11 不只是检查异常数值，而是检查训练样本是否存在图像缺失、NaN/inf、空 action、异常 state/action、过短 episode、task 映射失败等问题。
## 1. 今日目标
1. 检查 episode 是否过短
2. 检查 image / wrist_image 是否缺失
3. 检查 state 是否存在 NaN / inf
4. 检查 actions 是否存在 NaN / inf
5. 检查 action 是否为空动作或全 0
6. 检查 action / state 是否存在异常极值
7. 检查 task_index 是否能映射到 task_text
8. 保存 clean_check_summary.json
## 2. 数据来源
本次继续复用本地 LeRobot 缓存：
/root/autodl-tmp/cache/lerobot/physical-intelligence/libero
metadata 路径：
/root/autodl-tmp/cache/lerobot/physical-intelligence/libero/meta/tasks.jsonl
脚本检测到：
num_episode_files = 1693
num_tasks_loaded = 40
说明本地 LIBERO 数据和 task metadata 都可以正常读取。
## 3. 已执行脚本
scripts/day11_clean_check.py
运行命令：
uv run python scripts/day11_clean_check.py | tee logs/day11_clean_check_log.txt
输出文件：
logs/day11_clean_check_log.txt
logs/day11_clean_check/clean_check_summary.json
## 4. 抽查范围
本次先在低配无卡环境下做小样本检查，没有直接检查全量 1693 个 episode。
抽查 episode：
episode_000000
episode_000001
episode_000002
episode_000010
episode_000050
episode_000100
各 episode 帧数：
episode_000000: 214 frames
episode_000001: 284 frames
episode_000002: 345 frames
episode_000010: 383 frames
episode_000050: 290 frames
episode_000100: 392 frames
总计：
num_checked_episodes = 6
total_frames = 1908
## 5. 字段检查结果
每个抽查 episode 都包含以下字段：
image
wrist_image
state
actions
timestamp
frame_index
episode_index
index
task_index
其中训练相关核心字段为：
image       = 主视角图像
wrist_image = 手腕相机图像
state       = 机器人状态
actions     = 机器人动作标签
task_index  = 语言任务编号
task_index 可以通过 meta/tasks.jsonl 映射到对应的 task_text。
本次抽查中所有 episode 的 task_mapped 都为 True，说明语言任务编号可以正常映射到任务文本。
## 6. 清洗检查规则
本次检查的异常类型包括：
short_episode
missing_main_image_key
missing_wrist_image_key
missing_main_image_frame
missing_wrist_image_frame
missing_state_key
missing_action_key
nan_state
inf_state
nan_action
inf_action
mostly_zero_action
action_outlier
state_outlier
missing_task_index
task_index_not_mapped
使用的保守阈值：
MIN_EPISODE_LEN = 50
ZERO_ACTION_EPS = 1e-6
ALL_ZERO_RATIO_WARN = 0.95
ACTION_ABS_WARN = 1.05
STATE_ABS_WARN = 10.0
这些阈值目前只用于发现潜在问题，不直接删除或修改原始数据。
## 7. 检查结果
总体结果：
num_checked_episodes = 6
total_frames = 1908
num_flagged_episodes = 0
异常统计：
missing_main_images   = 0
missing_wrist_images  = 0
nan_state_frames      = 0
inf_state_frames      = 0
nan_action_frames     = 0
inf_action_frames     = 0
zero_action_frames    = 0
action_outlier_frames = 0
state_outlier_frames  = 0
flagged episodes：
flagged_episodes = []
每个抽查 episode 的 flags 均为空。
## 8. Day11 结论
Day11 抽查 6 个 LIBERO episode，共 1908 帧。
本次 clean check 结果：
1. image / wrist_image 字段均存在
2. state / actions 字段均存在
3. task_index 均能成功映射到 task_text
4. 未发现 main image 缺失
5. 未发现 wrist image 缺失
6. 未发现 state NaN / inf
7. 未发现 action NaN / inf
8. 未发现空 action 或全 0 action
9. 未发现 action 异常极值
10. 未发现 state 异常极值
11. 未发现过短 episode
因此，当前小样本 clean check 通过。
需要注意：本次结果只能说明抽样 episode 没有发现异常，不能严格代表全量 1693 个 episode 完全无问题。后续如果时间允许，可以扩大抽查范围或做全量轻量检查。
## 9. 当前数据流理解
结合 Day8、Day9、Day10、Day11，目前数据流可以理解为：
episode_*.parquet → frame-level samples → image / wrist_image / state / actions / task_index → meta/tasks.jsonl → task_text
训练输入输出形式：
image + wrist_image + state + task_text → actions
Day8 主要确认数据能读和 task 映射。
Day9 主要确认 image / wrist_image 能正常打开。
Day10 主要确认 state/action 维度和统计范围。
Day11 主要确认样本中是否存在明显异常。
## 10. 后续 TODO
1. Day12 整理 data_format.md，明确字段名和本地路径
2. 明确 openpi / pi05_libero 实际读取哪些字段
3. 后续可扩大 clean check 范围
4. Day13 做 image grid、action curve、state curve 可视化
5. Day15 前对比 compute_norm_stats 输出结果
6. 暂时不真正删除数据，先保留 clean check 作为训练前质量证明
## 11. 一句话总结
Day11 完成了 LIBERO 数据清洗检查：抽查 6 个 episode、1908 帧，未发现图像缺失、NaN/inf、空 action、异常动作、异常 state 或 task 映射失败，当前小样本数据质量检查通过。