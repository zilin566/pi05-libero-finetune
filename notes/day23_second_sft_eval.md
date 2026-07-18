# Day 23｜第二轮 SFT 评测与正式 Rollout

> 日期：2026-07-16  
> 项目：openpi / π0.5 + LIBERO LoRA 微调与评测  
>
> 今日核心结论：**第二轮训练显著优于旧训练配方。固定离线验证集上，Action L2 从旧 5k 的 1.4378 降到 0.4371，gripper sign accuracy 从 48% 提升到 89%；在完全相同的 Task 5 states 0-9 上，成功率从旧 5k 的 0/10 提升到 9/10，官方 baseline 为 10/10。额外双物体任务达到 3/3，因此不再继续训练到 6 epoch，转入 README、Demo、失败分析和投递阶段。**

---

## 目录

- [1. 第二轮训练配置](#1-第二轮训练配置)
- [2. 七点离线评测](#2-七点离线评测)
- [3. Task 5 正式三方对照](#3-task-5-正式三方对照)
- [4. Task 1 压力测试](#4-task-1-压力测试)
- [5. 失败案例](#5-失败案例)
- [6. 部署指标](#6-部署指标)
- [7. 最终结论](#7-最终结论)
- [8. 今日产出与下一步](#8-今日产出与下一步)

---

## 1. 第二轮训练配置

```text
config              = pi05_libero_day22_bs4_e2
initialization      = pi05_base
dataset             = physical-intelligence/libero_day18_n100
batch_size          = 4
train steps         = 14,000
final checkpoint    = 13,999
samples seen        ≈ 56,000
epoch coverage      ≈ 2.02
warmup_steps        = 500
learning rate       = 5e-5 → 5e-6 cosine decay
action_horizon      = 10
```

Checkpoint 完整性检查：

```text
_CHECKPOINT_METADATA   PASS
assets                 PASS
params                 PASS
train_state            PASS
```

单样本推理结果：

```text
prediction shape       = (1, 10, 7)
target shape           = (1, 10, 7)
NaN / Inf              = 无
```

---

## 2. 七点离线评测

### 2.1 固定评测条件

```text
episodes             = 100-109
samples per episode  = 3
observations         = 30 / checkpoint
noise seed           = 2026
```

所有 checkpoint 使用完全相同的 observation、frame、target 和 norm stats。

### 2.2 结果

| Step | Action L2 ↓ | RMSE ↓ | MAE ↓ | Arm RMSE ↓ | Gripper RMSE ↓ | Sign Accuracy ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 1k | 0.9403 | 0.4512 | 0.2077 | 0.2413 | 1.0371 | 72.67% |
| 3k | 0.7759 | 0.3935 | 0.1678 | 0.1990 | 0.9199 | 79.33% |
| 5k | 0.5999 | 0.3397 | 0.1265 | 0.1514 | 0.8186 | 83.00% |
| 7k | 0.5284 | 0.3079 | 0.1121 | 0.1429 | 0.7355 | 86.33% |
| 10k | 0.4792 | 0.2901 | 0.1034 | 0.1332 | 0.6948 | 88.00% |
| 12k | 0.4480 | 0.2818 | 0.0964 | 0.1244 | 0.6805 | 88.33% |
| **13,999** | **0.4371** | **0.2770** | **0.0947** | **0.1236** | **0.6673** | **89.00%** |

### 2.3 主要结论

从 1k 到 13,999：

```text
Action L2         下降约 53.5%
RMSE              下降约 38.6%
MAE               下降约 54.4%
Arm RMSE          下降约 48.8%
Gripper RMSE      下降约 35.7%
Sign Accuracy     提升 16.33 个百分点
```

与旧 5k 对比：

| 指标 | 旧 5k | 第二轮 13,999 |
|---|---:|---:|
| Action L2 | 1.4378 | **0.4371** |
| RMSE | 0.6070 | **0.2770** |
| MAE | 0.2978 | **0.0947** |
| Gripper RMSE | 1.4399 | **0.6673** |
| Sign Accuracy | 48.00% | **89.00%** |

结论：

> 第二轮模型在连续动作和 gripper 判别上都显著改善，而且 10k→12k→14k 尾部仍继续变好，没有出现旧训练的后期退化。

---

## 3. Task 5 正式三方对照

任务：

```text
pick up the book and place it in the back compartment of the caddy
```

统一条件：

```text
suite              = libero_10
task_id            = 5
initial states     = 0-9
environment seed   = 7
replan_steps       = 5
max_policy_steps   = 520
```

### 3.1 正式结果

| Policy | Successes / 10 | Success Rate | Wilson 95% CI |
|---|---:|---:|---:|
| Official π0.5 LIBERO | **10/10** | **100%** | [0.722, 1.000] |
| Old 5k | **0/10** | **0%** | [0.000, 0.278] |
| Second-round 13,999 | **9/10** | **90%** | [0.596, 0.982] |

### 3.2 逐 state 结果

```text
state 0：
official SUCCESS
old 5k FAIL
second-round FAIL

states 1-9：
official SUCCESS
old 5k FAIL
second-round SUCCESS
```

第二轮相对旧 5k：

```text
9 个 state：FAIL → SUCCESS
0 个 state：SUCCESS → FAIL
```

结论：

> 在相同任务、相同 states 0-9 和相同评测管线下，第二轮训练将成功率从 0/10 提升到 9/10，并接近官方 checkpoint 的 10/10。

---

## 4. Task 1 压力测试

任务：

```text
put both the cream cheese box and the butter in the basket
```

特点：

```text
双物体
连续抓取与放置
多阶段动作序列
```

结果：

| State | Result | Success Step | Mean Latency* | P95 |
|---:|---|---:|---:|---:|
| 0 | SUCCESS | 254 | 86.93 ms | 102.94 ms |
| 1 | SUCCESS | 364 | 94.51 ms | 96.55 ms |
| 2 | SUCCESS | 239 | 95.30 ms | 96.61 ms |

```text
Task 1 result      = 3/3
Wilson 95% CI      = [0.439, 1.000]
```

\* 已排除第一次 JAX 编译时间。

结论：

> 第二轮模型不仅学会了 Task 5 的单物体放置，也在额外双物体多阶段任务的三个状态上全部成功，表现出一定的跨任务能力。

注意：Task 1 的 3/3 只作为压力测试，不与 Task 5 的 9/10 合并。

---

## 5. 失败案例

### Task 5 State 0

日志结果：

```text
environment success   = False
reward                = 0
exception             = None
EEF movement L2       = 0.6948 m
```

视频中模型已经完成：

```text
接近 → 夹取 → 抬起 → 搬运 → 插入 → 释放
```

但环境未触发成功，可能原因：

```text
书没有完全进入 back compartment 的精确判定区域
书倾斜靠在格子边缘
物体中心或包围盒仍在 success predicate 边界外
```

失败标签：

```text
placed_in_caddy_but_target_predicate_not_satisfied
```

机械臂后期姿势异常的原因不是“没有返回原位”，而是：

```text
环境未返回 success
→ rollout 继续执行到 520 steps
→ 策略持续重规划
→ 出现放置后的动作漂移
```

---

## 6. 部署指标

```text
GPU                       = NVIDIA GeForce RTX 5090
total VRAM                = 32,607 MiB
Policy Server resident    = 24,680 MiB
steady-state latency      ≈ 90-100 ms / query
action chunk              = (10, 7)
replan_steps              = 5
```

第一次请求约 23-26 秒，主要来自 JAX 首次编译，不计入稳态推理延迟。

正确表述：

> 加载后的 Policy Server 驻留显存约 24.7 GB，稳态推理延迟约 90-100 ms/query。

---

## 7. 最终结论

Day 23 进入：

```text
分支 a：出现正式 success
```

最终结果：

```text
Task 5：
old 5k             = 0/10
second-round       = 9/10
official baseline  = 10/10

Task 1 stress：
second-round       = 3/3
```

因此：

```text
不继续训练到 6 epoch
不修改 gripper loss
不继续堆 checkpoint
不进入在线 RL
```

核心判断：

1. 第二轮离线曲线持续改善；
2. 闭环能力从“无任务导向”提升到稳定完成任务；
3. 成功率从 0/10 提升到 9/10；
4. 双物体压力任务达到 3/3；
5. 当前项目已经完成“训练—离线评测—闭环成功—正式对照”的完整闭环。

---

## 8. 今日产出与下一步

### 8.1 主要产出

```text
experiments/day23_eval/second_round_curve/
experiments/day23_eval/formal_compare/
experiments/day23_eval/rollout_candidate_13999/
experiments/day23_eval/stress_task1/
experiments/day23_eval/deployment/
experiments/day23_eval/snapshot/
```

新增脚本：

```text
scripts/evaluation/day23_summarize_curve.py
scripts/evaluation/day23_formal_compare.py
scripts/evaluation/rollout_smoke_day22.py
scripts/interview_practice/eval_action_l2.py
```

### 8.2 下一步

```text
1. 更新 README 的离线曲线和正式成功率表
2. 完成 eval_protocol.md
3. 完成 deployment_note.md
4. 整理 Task 5 state 0 失败案例
5. 制作 Task 5 成功 GIF 和 Task 1 双物体 Demo
6. 更新简历项目 bullet
7. 进入投递与面试准备
```

---

## 面试 30 秒版

我基于 openpi 的 π0.5 在 LIBERO 上完成了从数据检查、quantile normalization、LoRA 微调、固定离线评测到闭环 rollout 的完整管线。第一轮训练使用 batch size 1、恒定学习率，而且只覆盖约 0.36 epoch，在训练分布内 Task 5 上是 0/10。第二轮我改成 batch size 4、500-step warmup、cosine decay，并训练约 2 epoch。固定验证集上 Action L2 从 1.4378 降到 0.4371，gripper sign accuracy 从 48% 提升到 89%；相同 Task 5 states 0-9 下，成功率从 0/10 提升到 9/10，官方 baseline 为 10/10。额外双物体任务也达到 3/3，因此停止继续堆 epoch，转入失败分析、Demo 和项目交付。
