Day8 总结：LIBERO 数据样本抽查（2026-07-06）

**今日做了什么**
今天不训练，专门补数据理解：复用本地旧缓存（/root/autodl-tmp/cache/lerobot/physical-intelligence/libero），读取 episode parquet，抽查 sample 字段，理清 task_index 与语言任务的映射，并保存 parser 输出和检查日志。

**核心结论**
1. 本地旧缓存可直接复用，无需重新下载数据集
2. 数据规模：1693 个 episode，40 个 language tasks
3. action 维度 = 7，与 pi05_libero 训练预期一致
4. task_index 通过 meta/tasks.jsonl 映射到完整语言任务文本
5. sample_index / frame_index / task_index 是三个不同概念：sample_index 是全局 sample 编号，frame_index 是 episode 内帧编号，task_index 是语言任务编号
6. 之前多个样本 task_index=0 只是因为都来自 episode 0，不是数据集只有一个任务（跨 episode 已确认：ep0→task0, ep1→task1, ep2→task2, ep10→task6）

episode_*.parquet  保存轨迹帧数据
tasks.jsonl        保存 task_index 到语言任务文本的映射
episodes.jsonl     保存 episode 级别信息
stats.json         保存数据统计信息

sample_index   = 当前抽查的全局 sample / frame 编号
episode_index  = 当前样本属于第几个 episode
frame_index    = 当前样本在该 episode 内的帧编号
task_index     = 当前 episode 对应的语言任务编号
task_text      = task_index 映射得到的语言任务文本
actions        = 当前帧对应的机器人动作
states         = 当前帧对应的机器人状态
**数据流理解**
episode_*.parquet
→ frame-level samples
→ image / wrist_image / states / actions / task_index
→ meta/tasks.jsonl
→ task_text
训练形式：image + wrist_image + state + task_text → actions

**今日产出**
scripts/day8_dataset_parser.py= 读取数据 + 解析字段 + 检查维度 + 保存摘要
作用：episode_*.parquet
↓
读取几个 sample / frame
↓
检查里面有哪些字段
↓
确认 episode_index / frame_index / task_index / actions / states / image 字段
↓
把结果保存成 sample_summary.json
day8_dataset_parser.py 是把原始轨迹读出来并整理成可检查的结构化信息；
可视化是下一步，在此基础上把 image / wrist_image / action 曲线保存成图片。
输出文件：
logs/day8_dataset_parser_log.txt
logs/day8_samples/sample_summary.json

logs/day8_cross_episode_task_check.txt：是 Day8 的任务映射检查日志，用来证明 episode → task_index → task_text 这条链路是正常的。

.parquet = 真正的数据文件，里面存轨迹、图像、动作、状态
.json / jsonl = 保存说明信息、检查结果、metadata
parser = 把数据读出来并整理成人能看懂的脚本

episode_000000.parquet
↓
day8_dataset_parser.py
↓
sample_summary.json

day8_dataset_parser.py              用于读取本地 parquet / arrow 缓存并抽查样本字段
sample_summary.json                 保存样本字段、类型、维度和预览信息
day8_dataset_parser_log.txt         保存 parser 运行日志
day8_cross_episode_task_check.txt   保存跨 episode 的 task_index / task_text 检查结果

**遗留 TODO**
确认 states 是否为 8 维；保存 image / wrist_image 样例图并检查分辨率；检查 episode 长度分布；统计 action / state 数值范围；为 Day9 图像检查和 Day10 action/state 统计做准备。
