# Day12：LIBERO 数据格式确认（openpi / LeRobot 对齐）
Day12 原计划是"格式转换：转成 openpi / LeRobot 可训练格式，保存 repo_id / local path，交付 converted_dataset"。但 Day8~Day11 已确认本地缓存本身就是 LeRobot 格式，且 openpi 的 pi05_libero 配置直接指向同一数据集，因此今天不重新转换数据，而是整理"当前数据已经是 openpi / LeRobot 可训练格式"的证据。本次不训练模型，不使用 GPU，不修改原始数据。
## 1. 今日目标
1. 确认本地数据已是 openpi / LeRobot 可训练格式，无需重新转换
2. 记录 repo_id 与 local path
3. 整理 episode parquet 字段清单
4. 确认 state / action 维度
5. 确认 image / wrist_image 格式
6. 确认 task_index → task_text 映射方式
7. 确认 openpi pi05_libero 配置与本地数据对齐
8. 说明 prompt_from_task=True 的含义
9. 保存 day12_data_format_check.json
10. README 增加 Data Format Check 小节
## 2. 计划调整说明
原计划交付物是 converted_dataset。实际情况是数据下载时就是 LeRobot 标准缓存格式，无需转换。因此 Day12 的交付物调整为：数据格式证明文档（本笔记）+ 格式检查 json + README 小节。这与原计划中"保存 repo_id / local path"的核心目标一致。
## 3. 数据路径与 repo_id
local_path = /root/autodl-tmp/cache/lerobot/physical-intelligence/libero
repo_id    = physical-intelligence/libero
metadata   = /root/autodl-tmp/cache/lerobot/physical-intelligence/libero/meta/tasks.jsonl
num_episode_files = 1693
num_tasks_loaded  = 40
## 4. 已执行脚本
scripts/day12_data_format_check.py
运行命令：
uv run python scripts/day12_data_format_check.py | tee logs/day12_data_format_check_log.txt
输出文件：
logs/day12_data_format_check_log.txt
logs/day12_data_format_check/day12_data_format_check.json
## 5. episode 字段清单
每个 episode parquet 包含以下字段：
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
## 6. state / action 维度（Day10 已验证）
state dim  = 8
action dim = 7
action 全局范围为 [-1, 1]，第 7 维范围 [-1, 1]、std 接近 1，推测对应 gripper 开合动作。
## 7. image / wrist_image 格式（Day9 已验证）
image       = 主视角 RGB 图像，256×256
wrist_image = 手腕相机 RGB 图像，256×256
两路图像均可从 parquet 正常解码打开，Day11 抽查未发现缺失帧。
## 8. task_index → task_text 映射（Day8 / Day11 已验证）
task_index 通过 meta/tasks.jsonl 映射到 task_text，共 40 个 task。Day11 抽查中所有 episode 的 task_mapped 均为 True。
## 9. openpi / pi05_libero 配置对齐
openpi 的 pi05_libero 配置使用：
repo_id = "physical-intelligence/libero"
prompt_from_task = True
该 repo_id 与本地缓存路径一一对应，说明 openpi 训练时可以直接读取当前本地缓存，不需要额外的格式转换步骤。
## 10. prompt_from_task=True 是什么
表示训练时的语言 prompt 不是数据集中单独存储的文本字段，而是由每帧的 task_index 经 meta/tasks.jsonl 映射得到的 task_text 自动生成。也就是说，tasks.jsonl 里的任务描述文本会作为语言指令输入模型。
## 11. 训练输入输出格式
image + wrist_image + state + task_text → actions
## 12. Day12 结论
1. 本地 LIBERO 数据已是 LeRobot 标准缓存格式，无需重新转换
2. repo_id 与 local path 已确认并记录
3. 字段清单、state/action 维度、图像格式均与 pi05_libero 训练预期一致
4. task_index 可正常映射 task_text，prompt_from_task=True 即依赖该映射
5. openpi pi05_libero 配置与本地数据对齐，具备直接训练的数据条件
Day12 数据格式确认通过。
## 13. 后续 TODO
1. docs/data_format.md 可在 Day13 或 Day14 补充（内容与本笔记接近，作为项目正式文档）
2. Day13 做 image grid、action curve、state curve 可视化
3. Day15 前运行 compute_norm_stats 并与 Day10 统计结果对比
4. 后续如扩大 clean check 范围，同步更新本文档
## 14. 一句话总结
Day12 确认本地 LIBERO / LeRobot 缓存（repo_id=physical-intelligence/libero）已经是 openpi pi05_libero 可直接训练的格式：字段、维度、图像、task 映射全部对齐，repo_id 与本地路径已固化记录，无需重新转换数据。