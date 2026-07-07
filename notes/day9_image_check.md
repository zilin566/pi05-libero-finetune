# Day9：LIBERO 图像样本检查

Day9 主要完成了 LIBERO / LeRobot 数据中的图像字段检查。本次不训练模型、不使用 GPU，只验证数据集里的主视角图像和腕部相机图像是否能正常读取、保存和人工查看。

本次继续复用本地旧缓存：
/root/autodl-tmp/cache/lerobot/physical-intelligence/libero

已创建并运行图像检查脚本：
scripts/day9_image_check.py

运行命令：
uv run python scripts/day9_image_check.py | tee logs/day9_image_check_log.txt

输出文件：
- logs/day9_image_check_log.txt
- logs/day9_image_check_summary.json
- logs/day9_images/

本次跨多个 episode 抽查图像样本：
episode_000000、episode_000001、episode_000002、episode_000010、episode_000050、episode_000100

每个 episode 抽查三帧：first frame、middle frame、last frame
每帧保存两个视角：main image、wrist image
最终共保存 36 张图片（6 episodes × 3 frames × 2 views = 36 images）

图像字段检查结果：
- main_image_key = image
- wrist_image_key = wrist_image

保存出的图像格式：
- image size = 256 × 256
- image mode = RGB

已在 VSCode 中人工打开保存的图像，例如 episode_000002_frame_000000_wrist.png。
图像内容正常，可以看到桌面、杯子和机械臂局部，说明 wrist_image 不是空图、不是乱码图、不是黑图，图像保存和读取流程正常。

结合 Day8 和 Day9，目前 LIBERO / LeRobot 数据流可以理解为：
episode_*.parquet → frame-level samples → image / wrist_image / states / actions / task_index → meta/tasks.jsonl → task_text

训练输入输出形式可以概括为：
image + wrist_image + state + task_text → actions

其中：
- image：主视角图像
- wrist_image：腕部相机图像
- state：机器人状态
- task_text：语言任务指令
- actions：机器人动作

今日结论：
1. main image 可以正常读取
2. wrist image 可以正常读取
3. 两个图像视角均为 256 × 256 RGB
4. 图像可以按照 episode / frame / task 保存
5. 已保存 36 张图像样本
6. 已人工打开 wrist image，确认画面内容正常

Day9 的意义是确认 LIBERO 数据中的视觉输入部分可读取、可保存、可检查，后续可以用于训练、可视化和 README 展示。

Day10 继续做 action / state 统计：
1. 确认 states 是否为 8 维
2. 确认 actions 是否为 7 维
3. 统计 state / action 的 min、max、mean、std
4. 保存 action_state_stats.json
5. 画 action/state 统计图
6. 写 notes/day10_action_state_stats.md
