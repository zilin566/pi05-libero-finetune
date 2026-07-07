
Day10 主要检查 LIBERO / LeRobot 数据中的 state 和 action 字段。本次不训练模型，不使用 GPU，只做数据统计和维度验证。
## 1. 今日目标
1. 读取本地 episode parquet
2. 确认 state / action 字段名
3. 确认 state / action 维度
4. 统计 min / max / mean / std
5. 保存 action_state_stats.json
6. 生成 action_state_stats.png
## 2. 数据来源
本次继续复用本地 LeRobot 缓存：
/root/autodl-tmp/cache/lerobot/physical-intelligence/libero
抽查 episode：
episode_000000
episode_000001
episode_000002
episode_000010
episode_000050
episode_000100
共检查：6 个 episode，1908 帧。
## 3. 已执行脚本
scripts/day10_action_state_stats.py
运行命令：
uv run python scripts/day10_action_state_stats.py | tee logs/day10_action_state_stats_log.txt
输出文件：
logs/day10_action_state_stats_log.txt
logs/day10_action_state_stats/action_state_stats.json
logs/day10_action_state_stats/action_state_stats.png
## 4. 字段检查结果
实际字段名为：
state_key_used  = state
action_key_used = actions
注意：原计划里有时写成 states，但实际 parquet 中字段名是 state。后续 Day12 写 data_format.md 时应以实际字段名为准。
## 5. State 统计结果
state 总 shape：
states shape = [1908, 8]
state dim    = 8
全局统计：
global_min  = -3.1433
global_max  = 3.6714
global_mean = 0.2915
global_std  = 1.2193
结论：
state 是 8 维，符合 pi05_libero 训练数据预期。
state 数值范围正常，未发现明显乱码、空值或全 0 问题。
## 6. Action 统计结果
action 总 shape：
actions shape = [1908, 7]
action dim    = 7
各维 min：
[-0.9375, -0.8625, -0.8946, -0.1586, -0.2154, -0.3643, -1.0]
各维 max：
[0.8304, 0.8518, 0.9375, 0.1864, 0.2154, 0.3611, 1.0]
各维 mean：
[0.0219, 0.0369, -0.0671, 0.0049, 0.0014, -0.0039, -0.1205]
各维 std：
[0.2884, 0.3419, 0.3102, 0.0446, 0.0578, 0.1002, 0.9927]
全局统计：
global_min  = -1.0
global_max  = 1.0
global_mean = -0.0181
global_std  = 0.4336
结论：
actions 是 7 维，符合预期。
action 数值整体位于 [-1, 1]。
action std 不为 0，说明不是空动作或全零动作。
第 7 维范围为 [-1, 1]，std 接近 1，推测对应 gripper 开合动作。
## 7. 当前数据流理解
结合 Day8、Day9、Day10，目前数据流可以理解为：
episode_*.parquet → frame-level samples → image / wrist_image / state / actions / task_index → meta/tasks.jsonl → task_text
训练输入输出形式：
image + wrist_image + state + task_text → actions
## 8. Day10 结论
Day10 抽查 6 个 LIBERO episode，共 1908 帧。实际 state 字段名为 state，维度为 8；实际 action 字段名为 actions，维度为 7。state/action 维度符合 pi05_libero 训练数据预期，action 全局范围为 [-1, 1]，未发现明显异常极值。当前小样本统计未发现明显字段缺失、维度错误或动作全零问题。Day10 action/state 统计检查通过。
## 9. 后续 TODO
1. Day11 检查异常样本：NaN / inf / 空 action / 过短 episode
2. 检查 action 是否存在全 0 片段
3. 检查 image / wrist_image 是否缺失
4. 后续和 compute_norm_stats 的统计结果做对比
5. Day12 整理 data_format.md，明确字段名为 image / wrist_image / state / actions / task_index
## 10. 收尾验证
确认统计图已生成：
ls -lh logs/day10_action_state_stats/action_state_stats.png