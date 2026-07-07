# Week1 Report：openpi / π0.5 + LIBERO 微调闭环

## 1. 本周目标

打通 openpi / π0.5 在 LIBERO 数据上的最小训练闭环。

主线：

```text
openpi 项目结构
→ LIBERO / LeRobot 数据流
→ pi05_libero 配置
→ norm stats
→ LoRA debug 配置
→ 10 step / 100 step 小规模训练
```

## 2. 已完成内容

```text
Day1：理解 openpi 项目结构和 π0.5 主线
Day2：理解 LIBERO → LeRobot → pi05_libero 数据流
Day3：理解 pi05_libero 和 train.py 主流程
Day4：完成 norm stats 计算
Day5：跑通 LoRA + max_token_len=64 的 10 step debug
Day6：扩展到 100 step 小规模训练并保存 checkpoint
Day7：整理 README、docs 和 week1_report
```

## 3. 核心实验结果

原始 pi05_libero：

```text
batch_size = 1
num_train_steps = 2
结果：OOM（init_train_state 阶段，24GB 显存不够）
```

LoRA debug（成功配置）：

```text
config = pi05_libero_debug_lora
batch_size = 1
max_token_len = 64
num_train_steps = 100
结果：成功
```

Day6 训练日志（每 10 步）：

```text
Step 0:  loss=0.0285
Step 10: loss=0.0938
Step 20: loss=0.0554
Step 30: loss=0.0829
Step 40: loss=0.1145
Step 50: loss=0.0649
Step 60: loss=0.0566
Step 70: loss=0.0936
Step 80: loss=0.1066
Step 90: loss=0.0761
```

## 4. 当前可行配置

```text
config = pi05_libero_debug_lora
batch_size = 1
model.max_token_len = 64
num_workers = 0
wandb_enabled = False
```

结论：π0.5 + LIBERO + LoRA 的小规模训练闭环已经跑通。

## 5. 主要踩坑

```text
1. openpi 路径一开始找错（真实路径在 vla_work 下）
2. Hugging Face 连接需要 hf-mirror 镜像
3. /autodl-pub/data 是只读目录，不能作为缓存
4. 原始 pi05_libero 在 24GB 显存下 OOM
5. LoRA 默认 token_len=200 仍然 OOM
6. 降到 max_token_len=64 后训练成功
7. checkpoint 占磁盘较大，需要及时清理旧实验
```

## 6. Day8 TODO

```text
1. 抽查 LIBERO episode
2. 检查 image / wrist_image / state / actions / task 是否对齐
3. 统计 state/action 维度和范围
4. 准备 sample_check.md
5. 为后续数据清洗和评测做准备
```