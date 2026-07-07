# Day2 LIBERO 数据准备记录

## 1. 今日目标

搞清楚 LIBERO 数据如何进入 π0.5 训练流程。
 LIBERO 数据是怎么从原始数据，变成 π0.5 训练能读取的数据格式的。
核心链路：

```text
openvla/modified_libero_rlds
→ convert_libero_data_to_lerobot.py
→ physical-intelligence/libero
→ pi05_libero
→ train.py


pi05_libero 是 openpi 中用于 π0.5 + LIBERO 的训练配置
它读取 physical-intelligence/libero
模型是 pi0.Pi0Config(pi05=True)
输入是 image + wrist_image + state + task
输出是 actions
action_horizon = 10
默认 batch_size = 256
默认训练 30000 steps


## 2. 数据转化脚本
脚本位置：
examples/libero/convert_libero_data_to_lerobot.py
作用
把原始 LIBERO / RLDS 数据转换成 LeRobot 格式
转换命令
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

leRobot数据字段
| Field       | 中文含义   | Shape         |
| ----------- | ------ | ------------- |
| image       | 主视角图像  | 256 x 256 x 3 |
| wrist_image | 腕部相机图像 | 256 x 256 x 3 |
| state       | 机器人状态  | 8             |
| actions     | 机器人动作  | 7             |
| task        | 语言指令   | string        |
image + wrist_image + state + task → actions

pi05_libero配置
| 配置项              | 含义                           |
| ---------------- | ---------------------------- |
| config name      | pi05_libero                  |
| model            | pi0.Pi0Config(pi05=True)     |
| repo_id          | physical-intelligence/libero |
| prompt_from_task | True                         |
| action_horizon   | 10                           |
| batch_size       | 256                          |
| num_train_steps  | 30000                        |

# 3. 这些专业词是什么意思？

| 术语 | 中文理解 |
|---|---|
| LIBERO | 一个机器人操作任务数据集 / benchmark |
| RLDS | 一种原始机器人轨迹数据格式 |
| LeRobot | 一种机器人数据集格式和工具，不是模型 |
| openpi | π 系列模型的代码库 |
| π0.5 / pi05 | 你要微调的 VLA 模型 |
| repo_id | HuggingFace 数据集或模型的名字 |
| dataset | 数据集 |
| episode | 一整条机器人任务轨迹 |
| step / frame | 轨迹中的某一个时间步 |
| observation | 当前看到的东西，比如图像和机器人状态 |
| image | 主相机图像 |
| wrist_image | 机械臂腕部相机图像 |
| state | 机器人自身状态 |
| action | 机器人要执行的动作 |
| task | 语言任务指令 |
| prompt_from_task | 从 task 字段读取语言指令 |
| action_horizon | 一次预测未来多少步动作 |
| batch_size | 一次训练读多少条数据 |
| checkpoint | 模型权重存档 |
| norm stats | 数据归一化统计量 |
| train config | 训练配置 |
| finetune | 微调 |
| LoRA | 低显存微调方法 |
| SFT | 监督微调 |
# 4. 今天你真正要理解的一句话

LeRobot 不是模型，而是机器人数据格式。
LIBERO 原始数据先转成 LeRobot 格式。
π0.5 训练时读取 image、wrist_image、state、task，然后预测 actions。
