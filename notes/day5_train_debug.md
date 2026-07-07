# Day5 总结：无 GPU 白天准备 + LoRA Debug 配置

白天没有 GPU，不跑训练，只做训练前准备；晚上跑 π0.5 + LIBERO 最小训练实验，验证训练闭环。
## 完成事项
1. **路径确认**：openpi 真实路径为 `/root/autodl-tmp/vla_work/openpi`（原路径报错，用 find 找到）。
2. **环境变量**：沿用 Day4 配置（HF 镜像源、HF_HOME、HF_DATASETS_CACHE、HF_LEROBOT_HOME 等）。
3. **norm stats 检查**：Day4 生成的 `norm_stats.json`（4.0K）仍在，包含 state(8维)/actions(7维) 的 mean/std/q01/q99，训练时用于归一化。
4. **train.py 参数**：确认支持 `--batch-size`、`--num-train-steps`、`--log-interval`、`--save-interval`、`--num-workers`、`--no-wandb-enabled`、`--overwrite` 等覆盖参数。
5. **pi05_libero 配置理解**：官方 LIBERO 微调配置，从 pi05_base checkpoint 继续微调；默认 batch_size=256、steps=30000，太大不适合 debug。
6. **train.py 主流程理解**：读配置 → 建模型/optimizer → 加载 pi05_base → dataloader 读 batch → compute_loss → 反向传播更新 → 打印 loss/grad_norm/param_norm → 存 checkpoint。
7. **数据流理解**：LIBERO RLDS → 转换脚本 → LeRobot 格式（image / wrist_image / state / actions / task）→ physical-intelligence/libero → pi05_libero 训练。
8. **新增 LoRA debug 配置** `pi05_libero_debug_lora`：
   - paligemma / action expert 均用 LoRA 变体，只训 LoRA 参数
   - batch_size=1，steps=10，关闭 EMA 和 wandb
   - 语法检查通过（py_compile 无输出）
   - norm stats 已复制到 `assets/pi05_libero_debug_lora/`
   - `--help` 确认新配置注册成功

## 晚上计划
```bash
# 先确认 GPU
nvidia-smi

# 先跑原始配置最小测试（bs=1, steps=2）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run scripts/train.py pi05_libero \
  --exp-name=day5_pi05_libero_bs1_s2 \
  --batch-size=1 --num-train-steps=2 \
  --log-interval=1 --save-interval=1 --num-workers=0 \
  --no-wandb-enabled --overwrite

# OOM 则改跑 LoRA debug
uv run scripts/train.py pi05_libero_debug_lora \
  --exp-name=day5_lora_bs1_s10 --overwrite
```

观察点：dataloader 初始化、pi05_base 加载、loss / grad_norm / param_norm 打印、checkpoint 保存、是否 OOM、显存占用。

## 结论
训练所需的数据、norm stats、配置、脚本、LoRA debug 配置全部就绪。
晚上只要能看到 loss / grad_norm / param_norm 并成功保存 checkpoint，即说明训练闭环跑通。






&&&&&&&详细步骤：

## 一、今日定位

主线：确认路径 → 检查 norm stats → 理解 pi05_libero 配置 → 理解 train.py 流程 → 理解 LIBERO 数据格式 → 准备 LoRA debug 配置 → 晚上跑最小训练实验。

---

## 二、白天完成事项

### 1. openpi 路径确认
- 原路径 `/root/autodl-tmp/openpi` 报 `No such file or directory`
- 用 find 找到真实路径：

```bash
find /root -maxdepth 4 -type d -name "openpi" 2>/dev/null
# → /root/autodl-tmp/vla_work/openpi
```

- 确认关键文件存在：`src/openpi/training/config.py`、`scripts/train.py`、`examples/libero/README.md`
- 后续所有命令先 `cd /root/autodl-tmp/vla_work/openpi`

### 2. 环境变量（沿用 Day4）
```bash
unset LEROBOT_HOME
unset TRANSFORMERS_CACHE
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/cache/huggingface/datasets
export HF_LEROBOT_HOME=/root/autodl-tmp/cache/lerobot
export OMP_NUM_THREADS=4
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=120
```

要点：HF_ENDPOINT 走镜像避免超时；缓存统一放 autodl-tmp；`/autodl-pub/data` 只读不能作缓存；TRANSFORMERS_CACHE 已弃用，统一用 HF_HOME。

### 3. norm stats 检查
- 文件仍在：`assets/pi05_libero/physical-intelligence/libero/norm_stats.json`（4.0K）
- 内容包含 state（8维）和 actions（7维）各自的 mean / std / q01 / q99
- 作用：训练时模型不用原始 state/actions 数值，而是根据 norm stats 做归一化
- 检查结果已存日志：`logs/day5_norm_stats_files.txt`、`day5_norm_stats_size.txt`、`day5_norm_stats_content.txt`

### 4. train.py 参数确认
```bash
uv run scripts/train.py pi05_libero --help
```
确认支持命令行覆盖：`--exp-name`、`--batch-size`、`--num-workers`、`--num-train-steps`、`--log-interval`、`--save-interval`、`--overwrite`、`--no-wandb-enabled`。

默认 batch_size=256、num_train_steps=30000，太大不适合 debug，但可以全部用命令行覆盖成小规模。

### 5. pi05_libero 配置理解
核心配置：
- `model = Pi0Config(pi05=True)`，`action_horizon = 10`
- 数据集：`physical-intelligence/libero`
- checkpoint：`pi05_base`（不是从零训练，是加载后继续微调）
- `prompt_from_task = True`，`ema_decay = 0.999`

训练输入输出：`image + wrist_image + state + task → actions`

### 6. train.py 主流程理解
```text
读取 TrainConfig → 创建 optimizer 和模型 → 加载 pi05_base checkpoint
→ 创建 dataloader → 读取 LIBERO batch → 拆成 observation + actions
→ model.compute_loss → 反向传播得 grads → optimizer 更新参数
→ 打印 loss / grad_norm / param_norm → 保存 checkpoint
```

成功训练的标志输出：`Initialized data loader`、`Initialized train state`、`Step 0: loss=..., grad_norm=..., param_norm=...`

### 7. LIBERO 数据流理解
`convert_libero_data_to_lerobot.py` 把原始 LIBERO RLDS（来自 openvla/modified_libero_rlds）转换为 LeRobot 格式。

字段：image（主视角）、wrist_image（腕部相机）、state（8维）、actions（7维）、task（语言指令）。

```text
LIBERO RLDS → 转换脚本 → LeRobot 格式 → physical-intelligence/libero → pi05_libero → train.py
```

### 8. 新增 LoRA debug 配置 `pi05_libero_debug_lora`
**为什么需要**：原始配置 batch=256、30000 步、full fine-tuning 显存压力大；Day5 只需确认训练闭环，不动原始配置，另开一个 debug config。

**关键设置**：
- `paligemma_variant = gemma_2b_lora`、`action_expert_variant = gemma_300m_lora` → 只训练 LoRA 参数，降低显存
- `batch_size = 1`、`num_train_steps = 10`、`log_interval = 1`、`save_interval = 5`、`num_workers = 0`
- `ema_decay = None`（关 EMA）、`wandb_enabled = False`
- `weight_loader = pi05_base`、`freeze_filter = get_freeze_filter()`

**落地过程**：
- 容器无 nano，用 Python 脚本自动插入配置
- `python -m py_compile src/openpi/training/config.py` 无输出，语法通过
- 复制 norm stats（openpi 按 config name 查找 assets）：

```bash
mkdir -p assets/pi05_libero_debug_lora
cp -r assets/pi05_libero/physical-intelligence assets/pi05_libero_debug_lora/
```

- `uv run scripts/train.py pi05_libero_debug_lora --help` 能显示 usage，说明配置注册成功

---

## 三、晚上 GPU 实验计划
晚上有 GPU 后，开始做小规模训练 debug。

### 1. 原始 pi05_libero 测试

先跑原始配置的最小测试：

```text
config = pi05_libero
batch_size = 1
num_train_steps = 2
````

结果：

```text
原始 pi05_libero 在 init_train_state 阶段 OOM。

结论：

```text
原始 pi05_libero 接近 full fine-tuning，24GB 显存不够。
```

---

### 2. LoRA debug 测试

随后切换到低显存配置：

```text
config = pi05_libero_debug_lora
batch_size = 1
num_train_steps = 10
```

结果：

```text
LoRA 配置成功初始化 dataloader 和 train state，
但默认 max_token_len=200 时，在第一个 train_step 仍然 OOM。
```
报错关键词：

```text
RESOURCE_EXHAUSTED: Out of memory
trying to allocate 5.37GiB
```

结论：

```text
LoRA 比原始配置更省显存，但默认 token 长度仍然太大。

### 3. LoRA + token_len=64 成功

进一步降低 token 长度：

```text
config = pi05_libero_debug_lora
batch_size = 1
num_train_steps = 10
model.max_token_len = 64
```

结果：

```text
成功完成 10 step 小规模训练。
```

日志：

```text
Step 0: grad_norm=1.0725, loss=0.0285, param_norm=1803.7705
Step 1: grad_norm=1.3787, loss=0.0668, param_norm=1803.7705
Step 2: grad_norm=0.8123, loss=0.0560, param_norm=1803.7705
Step 3: grad_norm=1.3699, loss=0.0534, param_norm=1803.7705
Step 4: grad_norm=0.9282, loss=0.1113, param_norm=1803.7706
Step 5: grad_norm=0.9109, loss=0.0223, param_norm=1803.7706
Step 6: grad_norm=7.8328, loss=0.3435, param_norm=1803.7708
Step 7: grad_norm=0.9081, loss=0.0676, param_norm=1803.7708
Step 8: grad_norm=0.6124, loss=0.0518, param_norm=1803.7710
Step 9: grad_norm=1.3621, loss=0.0931, param_norm=1803.7712
```

观察：

```text
loss 能正常计算
grad_norm 能正常打印
param_norm 能正常打印
param_norm 从 1803.7705 变化到 1803.7712
说明参数发生了更新
```

---

### 4. 今日结论

Day5 晚上成功验证了训练闭环：

```text
pi05_base checkpoint
→ LIBERO batch
→ norm stats
→ LoRA train state
→ compute_loss
→ backward / update
→ loss / grad_norm / param_norm logging
→ checkpoint save
```

最终结论：

```text
Day5 成功完成 π0.5 + LIBERO + openpi 的小规模训练 debug。
原始 pi05_libero 在 24GB 显存下 OOM。
LoRA 默认 token_len=200 仍然 OOM。
LoRA + batch_size=1 + max_token_len=64 可以成功完成 10 step 训练。
```

---

### 5. Day6 下一步

```text
固定 pi05_libero_debug_lora 配置
继续使用 max_token_len=64
尝试 100~500 steps 小规模训练
记录 loss 曲线和显存峰值
检查 checkpoint 保存和恢复
整理 README 的 Training Debug Pipeline

