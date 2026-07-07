# Day4 norm stats 与数据读取记录

## 核心入口

```text
openpi/scripts/compute_norm_stats.py
```

## 今日目标

```text
确认 pi05_libero 能否真实读取 physical-intelligence/libero，
并完成 state/actions 的归一化统计量计算。
```

核心流程：

```text
physical-intelligence/libero
→ LeRobotDataset
→ pi05_libero
→ compute_norm_stats.py
→ assets/pi05_libero/physical-intelligence/libero
```

---

## 环境变量设置

```bash
unset LEROBOT_HOME

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/cache/huggingface/datasets
export HF_LEROBOT_HOME=/root/autodl-tmp/cache/lerobot
export OMP_NUM_THREADS=4
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=120
```

当前理解：

```text
HF_ENDPOINT 用于切换 HuggingFace 镜像。
HF_HOME 用于统一管理 HuggingFace 缓存。
HF_LEROBOT_HOME 用于保存 LeRobot 数据缓存。
LEROBOT_HOME 已弃用，不能继续使用。
```

---

## 成功命令

最终成功运行：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero --max-frames 1000 | tee logs/day4_norm_stats_1000.txt
```

---

## 运行结果

终端显示：

```text
Fetching data files: 100%
Resolving data files: 100%
Downloading data: 100%
Generating train split: 273465 examples
Loading dataset shards: 100%
Computing stats: 100%
Writing stats to: assets/pi05_libero/physical-intelligence/libero
```

当前结果：

```text
physical-intelligence/libero 数据集下载完成。
pi05_libero 可以正常读取 LeRobot 格式的 LIBERO 数据。
1000 frames 的 norm stats 计算完成。
归一化统计量已经写入 assets 目录。
```

---

## 数据集情况

本次下载内容：

```text
episode parquet 文件约 1699 个
train split 共 273465 examples
```

当前理解：

```text
episode 表示一条完整机器人任务轨迹。
frame/example 表示轨迹中的一个时间步样本。
```

`--max-frames=1000` 的含义：

```text
最多使用 1000 个 frame 计算 norm stats，
但不代表只下载 1000 个 frame。
LeRobot 构建数据集时仍可能下载大量 episode parquet 文件。
```

---

## norm stats 是什么？

```text
norm stats = normalization statistics
```

作用：

```text
统计 state/actions 的均值和标准差，
用于后续训练时做数据归一化。
```

归一化目的：

```text
让不同维度的数据处在相近数值尺度，
避免某些数值范围大的维度主导 loss，
提高训练稳定性。
```

---

## 今日遇到的问题

### 1. huggingface.co 连接超时

报错现象：

```text
Connection to huggingface.co timed out
```

解决方法：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

### 2. /autodl-pub/data 只读

报错现象：

```text
Read-only file system: /autodl-pub/data/cache
```

当前结论：

```text
/autodl-pub/data 虽然空间大，但不能写入。
缓存只能放在 /root/autodl-tmp。
```

---

### 3. TRANSFORMERS_CACHE 已弃用

warning：

```text
TRANSFORMERS_CACHE is deprecated. Use HF_HOME instead.
```

当前处理：

```text
后续统一使用 HF_HOME，不再依赖 TRANSFORMERS_CACHE。
这是 warning，不影响 norm stats。
```

---

### 4. LeRobot v2.0 warning

warning：

```text
physical-intelligence/libero is in 2.0 format
current version of LeRobot is backward-compatible with it
```

当前理解：

```text
数据集是 v2.0 格式，但当前 LeRobot 可以兼容读取。
不需要手动转换 v2.1。
```

---

### 5. max-frames=50 失败

报错：

```text
ValueError: Cannot compute statistics for less than 2 vectors.
```

原因理解：

```text
max-frames=50 太小，实际参与统计的有效向量不足。
```

解决：

```text
改为 max-frames=300 后成功。
最终完成 max-frames=1000。
```

---

## 今日结论

```text
Day4 已完成 π0.5 + LIBERO 的数据下载、数据加载和 norm stats 计算流程。
```

已经确认：

```text
1. physical-intelligence/libero 数据集可以成功下载。
2. pi05_libero 配置可以正常读取 LIBERO 数据。
3. compute_norm_stats.py 可以正常运行。
4. 1000 frames 的 state/actions 归一化统计量已经计算完成。
5. norm stats 已写入 assets/pi05_libero/physical-intelligence/libero。
```

---

## 当前项目进度

```text
Day1：openpi 项目结构与 π0.5 主线理解
Day2：LIBERO → LeRobot 数据流理解
Day3：pi05_libero 训练配置理解
Day4：LIBERO 数据读取与 norm stats 计算完成
```

---

## 下一步任务

```text
准备 Day5 小规模训练 debug。
```

Day5 目标：

```text
不是正式训练，
而是确认训练脚本能启动，
模型能加载 pi05_base checkpoint，
batch 能正常读取，
loss 能正常计算，
参数能更新几个 step。
```

需要重点关注：

```text
batch_size 怎么改小
num_train_steps 怎么改小
checkpoint 是否能自动加载
训练日志保存在哪里
GPU 显存是否够用
```

一句话总结：
```text
Day4 打通了 π0.5 + LIBERO 的数据读取和归一化统计流程，为后续小规模训练 debug 做好了数据标准化准备。
```