
# Day1 项目启动记录

## 1. 今天的核心目标

今天不追求跑通模型，主要目标是确定项目方向、建立项目仓库、明确 π0.5 微调复现的整体流程。

## 2. 项目题目

基于 openpi 的 π0.5 在 LIBERO 仿真任务上的微调与评测复现。

## 3. 为什么选择这个项目

原因：

1. 目标明确，适合 VLA 实习简历展示；
2. 不需要真机，能在仿真环境中完成闭环；
3. 参考已有 VLA 微调复现项目，但主线更新为 π0.5；
4. 项目重点是数据、训练、推理、评测，而不是空泛读论文；
5. 能体现工程能力和复现能力。

## 4. 项目最小闭环

```text
openpi 环境
↓
LIBERO 数据
↓
数据格式转换
↓
π0.5 微调
↓
policy server 推理
↓
LIBERO 评测
↓
结果记录与失败分析

# Day1 openpi 阅读记录

## 今日完成

- AutoDL 数据盘安装 openpi 成功
- 找到 π0.5 + LIBERO 配置：`pi05_libero`
- 找到数据转换脚本
- 找到 LIBERO 仿真评测入口

## 环境信息

openpi 路径：

```text
/root/autodl-tmp/vla_work/openpi
```

Python 环境：

```text
/root/autodl-tmp/conda_envs/openpi_py311
```

验证结果：

```text
openpi import ok
```

## 核心入口

```text
π0.5 配置：
src/openpi/training/config.py

数据转换：
examples/libero/convert_libero_data_to_lerobot.py

仿真评测：
examples/libero/README.md
examples/libero/main.py
```

## pi05_libero 配置要点

```text
config name: pi05_libero
model: Pi0Config(pi05=True)
dataset: physical-intelligence/libero
checkpoint: pi05_base
action_horizon: 10
batch_size: 256
num_train_steps: 30000
```

第一次跑通时先改小：

```text
batch_size: 8 / 16 / 32
num_train_steps: 500 / 1000 / 2000
```

## 项目闭环

```text
LIBERO 数据
→ LeRobot 格式转换
→ π0.5 微调
→ policy server 推理
→ LIBERO 仿真评测
→ success rate / rollout 视频 / 失败分析
```

## 明天任务

准备 LIBERO 数据，并尝试运行数据转换脚本。
