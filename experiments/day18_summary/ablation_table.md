# Day18 - Data Validation and Ablation Experiments

## 1. 实验背景

LIBERO 抽查数据中未发现需要剔除的真实硬错误，因此没有进行两组完全相同的 raw-vs-cleaned 训练。主实验调整为数据规模、Normalization 和合成有限动作异常三类消融。

## 2. 数据规模消融

| Dataset | Steps | Final loss | Mean loss | Last-20 mean | Last-20 std | Max grad norm |
|---|---:|---:|---:|---:|---:|---:|
| n10 | 100 | 0.113200 | 0.068609 | 0.056675 | 0.043880 | 6.490900 |
| n50_clean_norm | 100 | 0.077900 | 0.077000 | 0.079785 | 0.051891 | 9.670400 |
| n100 | 100 | 0.074000 | 0.069731 | 0.075780 | 0.060656 | 12.115500 |

该实验是在相同训练步数和计算预算下比较不同 episode 数量的训练动态，不能单独替代固定验证集或仿真成功率。

## 3. Normalization 消融

| Metric | Clean + Norm | Clean + No Norm |
|---|---:|---:|
| Training mean loss | 0.077000 | 0.049751 |
| Last-20 loss | 0.079785 | 0.049780 |
| Max grad norm | 9.670400 | 13.094100 |
| Raw-space Action L2 | 1.211887 | 1.337490 |
| Raw-space RMSE | 0.542507 | 0.563986 |
| Raw-space MAE | 0.265523 | 0.297710 |

无归一化模型具有更低的训练空间 loss，但两组训练目标尺度不同，训练 loss 不能直接横向比较。在统一的 LIBERO 原始动作空间中，Normalization 模型的 Action L2、RMSE 和 MAE 更低。

### Per-Dimension RMSE

| Dimension | Clean + Norm | Clean + No Norm |
|---|---:|---:|
| action_dim_0 | 0.331991 | 0.285278 |
| action_dim_1 | 0.382664 | 0.372442 |
| action_dim_2 | 0.344873 | 0.340215 |
| action_dim_3 | 0.042040 | 0.158933 |
| action_dim_4 | 0.080247 | 0.385504 |
| action_dim_5 | 0.062416 | 0.172121 |
| gripper | 1.293253 | 1.298932 |

## 4. Synthetic Dirty-Data Ablation

| Metric | Clean + Norm | Dirty + Norm | Change |
|---|---:|---:|---:|
| Training mean loss | 0.077000 | 0.093506 | +21.44% |
| Training loss std | 0.047225 | 0.079514 | +68.37% |
| Maximum loss | 0.234700 | 0.453000 | +93.01% |
| Raw-space Action L2 | 1.211887 | 1.308358 | +7.96% |
| Raw-space RMSE | 0.542507 | 0.570385 | +5.14% |
| Raw-space MAE | 0.265523 | 0.284890 | +7.29% |

人工有限动作离群值使训练 loss 及其波动上升，同时降低了未见 episode 上的原始动作空间预测精度，验证了数据检查和异常动作清洗的必要性。

## 5. 当前实验限制

- 当前训练仅为 100 step，小规模结果主要反映训练动态。
- Action L2 使用固定离线验证样本，不能完全代表机器人任务成功率。
- 最终任务级结论需在后续 LIBERO rollout 中使用 success rate 验证。

## 6. 生成文件

- `data_scale_loss.png`
- `norm_dirty_loss.png`
- `action_l2_comparison.png`
- `training_summary.csv`
- `action_metrics.csv`
