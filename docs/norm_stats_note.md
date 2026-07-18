# Norm Stats in openpi / π0.5

## 1. Norm Stats 是什么

Norm Stats 是从训练数据中计算出的各维度统计量,主要包含:

- mean:均值
- std:标准差
- q01:1% 分位数
- q99:99% 分位数

本项目中统计对象包括:

- state:机器人状态,8 维
- actions:机器人动作,7 维

## 2. 为什么需要归一化

机器人数据中不同维度的单位和数值范围可能不同。如果直接训练,数值范围大的维度可能主导 loss,导致优化不稳定。

标准化:

```text
x_norm = (x - mean) / std
```

反标准化:

```text
x = x_norm * std + mean
```

训练和推理必须使用同一套统计量。

## 3. 为什么只能使用 Train Split

Norm Stats 只能由 train split 计算。

如果将 validation 或 test 数据用于计算 mean/std,就会提前利用测试集的分布信息,造成 data leakage。

正确流程:

1. 只使用 train split 计算统计量
2. 训练阶段使用该统计量
3. validation/test 使用同一套统计量
4. 部署推理同样使用该统计量

## 4. openpi 计算入口

```bash
uv run python scripts/compute_norm_stats.py \
  --config-name pi05_libero_debug_lora \
  --max-frames 100
```

参数:

- `--config-name`:选择数据与模型配置
- `--max-frames`:限制最终参与统计的帧数

注意:`max-frames=100` 只限制实际统计帧数。Hugging Face Datasets 仍可能先生成完整 train split 和 Arrow 缓存。

## 5. 本次运行信息

- Dataset:physical-intelligence/libero
- 数据格式:LeRobot v2.0
- 数据文件:1693
- Train examples:273465
- Dataset shards:70
- 统计帧数:100

## 6. State 统计结果

State mean:

```text
[-0.03786254, 0.02911713, 0.72740686, 3.00025249,
 -0.13771303, -0.07687049, 0.02718850, -0.02797369]
```

State std:

```text
[0.11885727, 0.14698650, 0.40357450, 0.27613589,
 0.79980701, 0.24388961, 0.01384920, 0.01335375]
```

检查结果:

- shape = (8,)
- min std = 0.01335375
- near-zero indices = []

## 7. Actions 统计结果

Actions mean:

```text
[0.08159464, 0.13063928, -0.09071784, -0.00200393,
 0.00249571, -0.00158179, -0.07199997]
```

Actions std:

```text
[0.34045112, 0.35486531, 0.45766029, 0.03702425,
 0.05802936, 0.07507898, 0.99740463]
```

检查结果:

- shape = (7,)
- min std = 0.03702424
- near-zero indices = []

Actions 最后一维的 q01 接近 -1、q99 接近 1,可能对应夹爪开合类动作,但需要结合正式字段定义确认。

## 8. 100 帧统计的定位

100 帧统计主要用于:

- 验证计算流程
- 验证维度
- 检查数值异常
- 支持 debug 微调

正式扩大训练时,应优先使用更大规模的 train split 统计量,以减少抽样波动。

## 9. 结论

本次 Norm Stats 流程已完整跑通。State 和 actions 的维度与数据检查结果一致,且不存在标准差接近 0 的危险维度。