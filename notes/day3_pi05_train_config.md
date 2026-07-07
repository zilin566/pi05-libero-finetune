# Day3 π0.5 训练配置记录

## 1. 今日目标

搞清楚 `pi05_libero` 这个训练配置是怎么控制 π0.5 训练流程的。

数据怎么进入训练
→ 模型怎么创建
→ checkpoint 怎么加载
→ loss 怎么计算
→ 参数怎么更新
```

核心链路：

```text
pi05_libero
→ LeRobotLiberoDataConfig
→ physical-intelligence/libero
→ Pi0Config(pi05=True)
→ train.py
→ compute_loss
→ optimizer update
```

## 2. pi05_libero 配置

| 配置项 | 含义 |
|---|---|
| config name | pi05_libero |
| model | pi0.Pi0Config(pi05=True) |
| repo_id | physical-intelligence/libero |
| prompt_from_task | True |
| action_horizon | 10 |
| batch_size | 256 |
| num_train_steps | 30000 |
| checkpoint | gs://openpi-assets/checkpoints/pi05_base/params |

当前理解：

```text
pi05_libero 是 openpi 中用于 π0.5 + LIBERO 的训练配置。
它读取 physical-intelligence/libero，
并从 pi05_base checkpoint 开始继续训练，不是从零训练。
```
## 4. LeRobotLiberoDataConfig 是什么？

`LeRobotLiberoDataConfig` 的作用：

```text
把 LeRobot 数据字段转换成 openpi 模型训练需要的字段。
```

字段映射关系：

```text
image        → observation/image
wrist_image  → observation/wrist_image
state        → observation/state
task         → prompt
actions      → actions
```

当前理解：

```text
physical-intelligence/libero 是已经转换好的 LeRobot 格式 LIBERO 数据集。
pi05_libero 通过 LeRobotLiberoDataConfig 读取这个数据集。
```

---

## 5. train.py 主流程

今天查看了：

```text
scripts/train.py
```

核心流程可以理解为：

```text
读取 TrainConfig
→ 创建模型
→ 加载 checkpoint
→ 创建 optimizer
→ 读取 batch
→ 计算 loss
→ 更新参数
```

---

## 6. init_train_state() 做什么？

`init_train_state()` 是训练开始前的初始化函数。

作用：

```text
创建模型
→ 加载 pi05_base checkpoint
→ 合并预训练参数
→ 初始化 optimizer
→ 创建 TrainState
```

当前理解：

```text
TrainState 保存训练过程中的模型参数、optimizer 状态、step 等信息。
pi05_libero 会先加载 pi05_base 权重，再继续在 LIBERO 上训练。
```

---

## 7. train_step() 做什么？

`train_step()` 是一次训练更新。

流程：

```text
读取 batch
→ 拆成 observation 和 actions
→ model.compute_loss(observation, actions)
→ 计算梯度
→ optimizer 更新参数
→ 返回 loss
```

当前理解：

```text
train_step 本质上是：
让模型根据 observation 预测 actions，
再用 loss 来更新模型参数。
```

---

## 8. compute_norm_stats.py 是什么？

今天运行了：

```bash
uv run scripts/compute_norm_stats.py --help
```

确认脚本可以启动。

主要参数：

| 参数 | 含义 |
|---|---|
| --config-name | 指定训练配置，例如 pi05_libero |
| --max-frames | 最多读取多少帧数据，用于 debug |

后续可能使用：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero --max-frames 100
```

作用：

```text
compute_norm_stats.py 用来计算 state 和 actions 的归一化统计量，
为后续训练做数据标准化准备。
```

---

## 9. 今天这些专业词是什么意思？

| 术语 | 中文理解 |
|---|---|
| TrainConfig | 训练配置 |
| Pi0Config | π0 / π0.5 模型配置 |
| pi05=True | 启用 π0.5 模型设置 |
| checkpoint | 模型权重存档 |
| weight_loader | 加载 checkpoint 的工具 |
| TrainState | 训练状态，保存参数和 optimizer 状态 |
| optimizer | 优化器，用来更新模型参数 |
| batch | 一次训练读入的一批数据 |
| observation | 模型输入，包括图像、状态、语言 |
| actions | 模型要预测的机器人动作 |
| loss | 预测动作和真实动作之间的误差 |
| gradient | 梯度，用来指导参数更新 |
| action_horizon | 一次预测未来多少步动作 |
| norm stats | 数据归一化统计量 |

---

## 10. 今日结论

1. `pi05_libero` 是 openpi 中用于 π0.5 在 LIBERO 上训练的配置。
2. 它读取的数据集是 `physical-intelligence/libero`。
3. 模型是 `pi0.Pi0Config(pi05=True)`。
4. 训练输入是 `image + wrist_image + state + task`。
5. 训练目标是 `actions`。
6. `action_horizon = 10`，表示一次预测未来 10 步动作。
7. 默认 `batch_size = 256`，默认训练 `30000` steps。
8. 训练会加载 `pi05_base` checkpoint，不是从零开始。
9. `train.py` 的核心流程是创建模型、加载权重、计算 loss、更新参数。
10. `compute_norm_stats.py` 后续需要用 `--config-name pi05_libero` 指定配置。

---

## 11. 今天真正要理解的一句话

```text
pi05_libero 定义了 π0.5 如何读取 LIBERO 数据、加载基础模型权重，
并通过 image、wrist_image、state、task 来预测未来 actions。
