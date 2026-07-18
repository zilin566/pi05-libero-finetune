# Day 24｜正式评测扩展、失败分布与 bf16 部署记录

> 日期:2026-07-17
> 项目:openpi / π0.5 + LIBERO LoRA 微调与评测
>
> 今日核心结论:**step 13,999 的评测证据全面增强。Task 1 states 0-9 达到 10/10;LIBERO-10 每任务 3 初态覆盖筛查 25/30,失败按任务聚类(Task 8 为 0/3 的系统性弱点)。Day 23 的 Task 5 state 0 失败在 11 次后续重复中未复现;受控实验(state 0 vs 2 = 9/10 vs 7/10)与成功样本余量分析共同指向同一几何结论——模型的放置分布横跨 back region 最窄轴(x 半宽仅 27.75 mm)的 predicate 边界,4 条失败全部为 x 轴 containment miss,成功样本 x 余量最小仅 0.002 mm。Task 5 的 ~90% 本质上是该放置分布落在 x 边界内的概率。确定性审计证明 policy RNG 无法显式固定,轨迹级复现不成立但任务级成功稳定,因此重复 rollout 天然等价于策略噪声下的多次采样。bf16 部署:稳态均值 83.2 ms、P95 99.9 ms、观测峰值显存 24.11 GiB。**

---

## 目录

- [1. 今日目标与完成情况](#1-今日目标与完成情况)
- [2. 在线评测确定性审计](#2-在线评测确定性审计)
- [3. Task 5 State 0 成功判定与定量归因](#3-task-5-state-0-成功判定与定量归因)
- [4. Task 1 正式补测](#4-task-1-正式补测)
- [5. LIBERO-10 十任务覆盖筛查](#5-libero-10-十任务覆盖筛查)
- [6. 失败分布](#6-失败分布)
- [7. Task 5 受控初态鲁棒性实验](#7-task-5-受控初态鲁棒性实验)
- [8. Task 5 证据汇总与报告口径](#8-task-5-证据汇总与报告口径)
- [9. bf16 部署工程指标](#9-bf16-部署工程指标)
- [10. 最终结论](#10-最终结论)
- [11. 今日产出与下一步](#11-今日产出与下一步)
- [面试 30 秒版](#面试-30-秒版)

---

## 1. 今日目标与完成情况

原计划(顺延版 Day 24):bf16 工程记录、latency/VRAM、deployment_note。

实际完成:

```text
P0  rollout seed / RNG / 非确定性审计
P1  Task 5 state 0 success predicate 定量分析(BDDL + 源码级)
P2  Task 1 states 3-9 补测 → 完整 10/10
P4  LIBERO-10 十任务三状态覆盖筛查(25/30)
P6  失败轨迹分类与代表视频
    受控 initial-state 鲁棒性实验(state 0 vs 2)
    bf16 latency / VRAM 工程记录
```

> Day 24 原定工程记录完成,同时提前吸收计划 Day 25 的正式评测扩展与失败分布主体。计划遗留:指令同义改写实验(可选,后置);多 ENV_SEED 复评被第 2 节的审计结论替代——policy RNG 每次 rollout 本就不同,重复运行即等价于策略噪声下的重复采样。

---

## 2. 在线评测确定性审计

### 2.1 三类状态必须区分

```text
INITIAL_STATE_ID = LIBERO 预定义初始状态编号(不是随机种子)
ENV_SEED         = NumPy / LIBERO 环境 seed(不是 policy seed)
Policy RNG       = Policy Server 内部 JAX RNG;
                   新服务器默认从 jax.random.key(0) 开始,每次 inference 后推进;
                   serve_policy.py 未暴露显式 seed 参数
```

### 2.2 同一服务器重复(不重启)

固定 Task 5 / state 1 / ENV_SEED 7 / ckpt 13,999,同一进程连跑两次:两次均成功,但轨迹不同、成功步不同、最终 EEF 相差约 5.96 mm。原因:服务器内 RNG 持续推进。

### 2.3 每次重启服务器重复

两次 fresh server、相同条件:

```text
前 10 个 wait action     完全一致
first divergent index    = 10(第一个 policy action 即分叉)
max absolute delta       = 0.001953
action L2 delta          = 0.002626
两条轨迹均成功,终止步差 1 步
```

### 2.4 结论

> 固定 initial state、ENV_SEED 并重置默认 Policy RNG,仍不能逐动作复现;残余差异来自初始 observation 微差或 JAX/GPU 数值非确定性。因此:
>
> ```text
> trajectory-level reproducibility = 不成立
> task-level repeated success      = 稳定(本组 state 1:4/4)
> ```
>
> **方法论推论:每次 rollout 都在独立采样策略噪声,"重复运行 N 次"即策略随机性下的 N 次采样——成功率的不确定度评估不需要额外的 seed 扫描,只需要足够的重复次数。**

证据:`experiments/day24_robustness/determinism_check.json`

---

## 3. Task 5 State 0 成功判定与定量归因

任务:pick up the book and place it in the back compartment of the caddy

### 3.1 正式 BDDL goal 与实际判定

```text
In(black_book_1, desk_caddy_1_back_contain_region)
```

`In` = check_contact AND check_contain,但 containment site 的 check_contact 恒为 True,因此**实际成功条件 = 书的 body 中心点落在 back_contain_region 的世界坐标 AABB 内**(z 下界额外放宽 0.01 m)。

predicate **不检查**:整书包围盒、姿态角、夹爪是否释放、书与 caddy 是否真实接触。

区域几何(desk_caddy.xml):

```text
back region local half-size = [x: 0.02775, y: 0.06216, z: 0.06046] m
→ x 是三轴中最窄的,半宽仅 27.75 mm
```

### 3.2 Day 23 失败未复现

本日重新运行 state 0:首条诊断即成功(x/y/z margin = 18.9 / 39.3 / 5.6 mm),继而 10 次重复全部成功——**Day 24 follow-up 合计 11/11**。

10 次成功样本的最近边界余量:

| Axis | Min | Mean | Max |
|---|---:|---:|---:|
| x | **0.002 mm** | 12.084 mm | 22.365 mm |
| y | 26.423 mm | 47.120 mm | 61.406 mm |
| z | **0.412 mm** | 4.303 mm | 14.764 mm |

run 8 的 x 余量 0.002 mm、run 10 的 z 余量 0.412 mm——成功轨迹经常擦着 predicate 边界过线。

### 3.3 归因结论(与第 7 节合并的最终版)

Day 23 原始失败轨迹未记录 book position 与 target bounds,无法恢复当次越界轴;当时的标签修订为:

```text
一级现象:place_outside_target
二级诊断:predicate_boundary_miss / near_boundary_placement_variability
```

结合第 7 节受控实验(4/4 失败均为 x 轴 miss)与本节成功样本余量(x 均值 12 mm,最小 0.002 mm,而 x 半宽仅 27.75 mm):

> **几何级归因:模型在 Task 5 的放置终点分布横跨了最窄轴(x)的 predicate 边界。成功与失败不是两种行为模式,而是同一放置分布落在 x 边界两侧的采样结果;Task 5 的 ~90% 成功率即该分布的界内概率。** Day 23 的失败与 Day 24 的 11/11 并不矛盾——它们是同一分布的不同批采样。

证据:`experiments/day24_state0_analysis/`(attribution.md、repeat_summary.json)、`scripts/evaluation/rollout_predicate_day24.py`

---

## 4. Task 1 正式补测

任务:put both the cream cheese box and the butter in the basket(双物体、多阶段)

| Initial states | Success |
|---|---:|
| 0-2(Day 23) | 3/3 |
| 3-9(Day 24) | 7/7 |
| **0-9 合计** | **10/10** |

```text
Success rate = 100%,Wilson 95% CI = [0.722, 1.000]
states 3-9:policy queries 46-51,executed steps 227-251,EEF L2 0.258-0.305 m
```

> 双物体多阶段任务 10 初态全成,模型能力不局限于 Task 5。

证据:`experiments/day24_task1_eval/task1_states3_9_summary.json`

---

## 5. LIBERO-10 十任务覆盖筛查

### 5.1 协议

ckpt 13,999;task 0-9 × states 0-2 = 30 episodes;ENV_SEED 7;replan 5;max 520。定义为**覆盖筛查**,不是正式 LIBERO benchmark。

### 5.2 结果

| Task | Result | Task description |
|---:|---:|---|
| 0 | 3/3 | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 3/3 | put both the cream cheese box and the butter in the basket |
| 2 | 3/3 | turn on the stove and put the moka pot on it |
| 3 | 3/3 | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | 3/3 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | 2/3 | pick up the book and place it in the back compartment of the caddy |
| 6 | 3/3 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 3/3 | put both the alphabet soup and the cream cheese box in the basket |
| 8 | **0/3** | put both moka pots on the stove |
| 9 | 2/3 | put the yellow and white mug in the microwave and close it |

```text
successes = 25/30,observed rate = 83.3%
Wilson 95% CI = [0.664, 0.927]
```

> **CI 保留说明:该区间把 30 条视为独立样本;实际失败明显按任务聚类(Task 8 独占 3/5),任务级相关会使真实不确定度大于名义 CI。引用时表述为"25/30 覆盖筛查",不表述为"83.3% ± CI 的 benchmark 成功率"。**

### 5.3 结论

7 个任务 3/3;Task 5 偶发末端放置失败;Task 9 微波炉放置/关门阶段失败;**Task 8 为系统性弱点**。

证据:`experiments/day24_ten_task_screen/`

---

## 6. 失败分布

五条失败均运行满 520 policy steps(timeout 是终止方式,不是根因)。

### 6.1 统一失败标签

| Task / State | 到达阶段 | 一级可观察标签 | 二级诊断 |
|---|---|---|---|
| T5 / S2 | 搬运与放置 | place_outside_target | book_wedged_at_compartment_edge |
| T8 / S0 | 第一壶完成 | contact_no_grasp | second_object_grasp_failure |
| T8 / S1 | 第一壶完成 | timeout_or_oscillation | repeated_second_object_regrasp |
| T8 / S2 | 第一物抓取 | contact_no_grasp | first_object_grasp_failure |
| T9 / S1 | 微波炉放置 | place_outside_target | premature_release_at_microwave_threshold |

一级分布:place_outside_target 2(40%)/ contact_no_grasp 2(40%)/ timeout_or_oscillation 1(20%)。

> 样本仅 5 条,只报告"观察到的失败分布",不写成普遍因果占比。

### 6.2 Task 8 任务级弱点

三个 state 全败但非零知识:S0/S1 均完成第一壶后卡在第二壶(抓取失败/反复重抓),S2 第一壶即失败。总结为 `multi_object_sequential_manipulation_failure`:第二物体重定位与抓取失败、部分完成后的恢复能力弱、长时序子任务切换不稳定。

> 待查(已列入下一步):Task 8 在 n100 中的示范条数——若恰为最少(任务示范数范围 7-13),则"示范数 vs 成功率"的相关性本身是一条结论。

证据:`experiments/day24_ten_task_screen/failure_*` 

---

## 7. Task 5 受控初态鲁棒性实验

### 7.1 设计

唯一变量 = INITIAL_STATE_ID(state 0 vs 2);固定 ckpt/ENV_SEED/replan/max steps/同一持续 server;AB-BA 交替;每组 10 条。

> 限制:policy RNG 无法显式固定,故为生产式随机 rollout 下的 initial-state robustness,非严格配对实验(依据第 2 节,这等价于策略噪声下的独立采样)。

### 7.2 成功率

| Initial state | Success | Rate | Wilson 95% CI |
|---:|---:|---:|---:|
| 0 | 9/10 | 90% | [0.596, 0.982] |
| 2 | 7/10 | 70% | [0.397, 0.892] |

> 观察差 20 个百分点,但 CI 明显重叠——state 2 观察成功率更低,当前样本量不足以声称统计显著。

### 7.3 失败几何

| State | x margin | y margin | z margin | Failed axes |
|---:|---:|---:|---:|---|
| 2 | -112.101 mm | +59.474 | +50.350 | x |
| 2 | -93.254 mm | +26.122 | +44.077 | x |
| 0 | -91.880 mm | +54.294 | +63.743 | x |
| 2 | -3.148 mm | -50.503 | +5.897 | x, y |

```text
4/4 失败:x 轴 containment miss(1 条并发 y 轴)
0/4 失败:z 轴
```

### 7.4 归因

一级 place_outside_target;二级 x_axis_containment_miss(1 条 x_y)。

> state 0 与 state 2 是**同一 predicate 级失败机制**在不同初态下的不同发生频率;结果不支持此前"z 方向插入深度不足"的解释。与第 3 节合并后的最终几何结论见 3.3。

证据:`experiments/day24_controlled_robustness/task5_state0_vs_state2/`

---

## 8. Task 5 证据汇总与报告口径

Task 5 现有四批重叠评测,**必须统一引用口径,防止数字混用**:

| 批次 | 协议 | 结果 | 用途与口径 |
|---|---|---:|---|
| Day 23 正式评测 | states 0-9 各 1 条,预登记协议 | **9/10** | **README / 简历主表的 headline 数字**(预登记、未选择性重跑) |
| Day 24 state 0 复测 | state 0 × 11 条(1 诊断 + 10 重复) | 11/11 | 失败分析附注:"Day 23 的 state 0 失败在 11 次重复中未复现" |
| Day 24 受控实验 | state 0 / state 2 各 10 条 | 9/10 / 7/10 | 鲁棒性小节专用,不与主表合并 |
| Day 24 审计重复 | state 1 × 4 条 | 4/4 | 确定性审计专用 |

聚合参考(仅内部使用,不对外作为 headline):全部 45 条 Task 5 rollout 共 40 成功,≈ 89%——与 Day 23 主表的 90% 一致,交叉印证了 headline 的稳健性。

> 口径规则:对外主数字永远引用预登记的 Day 23 正式表(9/10 + CI);Day 24 各批作为归因与鲁棒性证据分节引用,并注明各自协议。禁止挑最好的一批(如 11/11)替换主表。

---

## 9. bf16 部署工程指标

### 9.1 配置确认

```text
ckpt 13,999;Pi0Config;dtype bfloat16;policy batch 1;action_horizon 10
内部 action dim 32 ←(padding)← LIBERO [H,7] →(unpad)→ 返回 [10,7];replan 5
```

### 9.2 Fresh-server latency(35 queries:1 冷启动 + 34 稳态)

| Metric | Result |
|---|---:|
| Cold-start first query | 24,850.5 ms(JAX tracing/JIT/首次初始化,不计入稳态) |
| Steady mean | **83.246 ms** |
| Steady P50 | 81.183 ms |
| Steady P95 | **99.901 ms** |

### 9.3 VRAM(nvidia-smi,约 200 ms 采样)

| Metric | Result |
|---|---:|
| Loaded-idle resident | 24,617 MiB(24.04 GiB) |
| Observed sampled peak | 24,693 MiB(24.11 GiB) |
| Peak above resident | 76 MiB |

表述为"observed sampled peak",非 CUDA allocator 瞬时峰值。

### 9.4 工程结论

> RTX 5090 / bf16 / batch 1:预热后均值 83.2 ms、P95 99.9 ms;驻留 24.04 GiB、观测峰值 24.11 GiB。主要部署瓶颈是 fresh-server 冷启动 ≈ 24.85 s——长驻 server 应在时延敏感请求前完成 warm-up。本 latency 为 policy-query 口径,不含真实系统的传感、通信、安全控制与执行器延迟。

证据:`experiments/day24_deployment/`

---

## 10. 最终结论

step 13,999 的闭环证据链:

```text
Task 5 正式(预登记)        = 9/10(Day 24 补充 25 条重复,聚合 ≈89% 交叉印证)
Task 1 正式 states 0-9      = 10/10
LIBERO-10 三状态覆盖筛查    = 25/30(失败按任务聚类,Task 8 系统性)
Task 5 失败几何             = 放置分布横跨最窄轴(x, 半宽 27.75 mm)的边界
确定性                      = 轨迹级不可复现,任务级稳定
bf16 稳态                   = mean 83.2 ms / P95 99.9 ms / peak 24.11 GiB
```

核心判断:

1. 第二轮模型具备稳定闭环任务能力,且跨任务(7 任务 3/3);
2. Task 5 失败的本质是放置分布 vs 最窄轴边界的几何问题,非抓取或接口问题;
3. Task 8 多物体顺序操作是当前唯一系统性弱点;
4. 重复 rollout 即策略噪声采样,成功率评估靠重复次数而非 seed 扫描;
5. 维持决策:不加 epoch、不改 gripper loss、不堆 checkpoint、暂不 RL;
6. 主线闭合:数据检查 → normalization → LoRA SFT → 离线评测 → 正式成功率 → 跨任务覆盖 → 失败分布 → 受控鲁棒性 → 部署记录。

---

## 11. 今日产出与下一步

### 11.1 新增脚本

```text
scripts/evaluation/rollout_predicate_day24.py
```

### 11.2 实验目录

```text
experiments/day24_robustness/            experiments/day24_state0_analysis/
experiments/day24_task1_eval/            experiments/day24_ten_task_screen/
experiments/day24_controlled_robustness/ experiments/day24_deployment/
```

### 11.3 视频归档

完整视频 `video/`;GitHub 代表视频 `docs/media/day24/`(建议 3 成功 + 3 典型失败)。

### 11.4 下一步

```text
1. 每任务示范数 vs 成功率相关性检查(task_distribution.csv 现成,5 分钟)
   ——重点确认 Task 8 的示范条数
2. README 正式结果表更新,严格按第 8 节报告口径
3. eval_protocol.md 定稿(补入确定性审计结论与重复采样原则)
4. Task 5 / Task 1 Demo GIF;成功与失败代表视频说明
5. 失败分布与部署指标写入项目讲稿;简历 bullet 更新
6. (可选,后置)指令同义改写鲁棒性实验
7. 进入 Day 25/26 文档、Demo 与交付阶段;不再需要 GPU、不再增加普通 rollout
```

---

## 面试 30 秒版

我在完成 π0.5 第二轮 LoRA 微调后,补齐了闭环评测、失败诊断和部署记录。Task 1 十个 predefined initial states 达到 10/10;LIBERO-10 每任务 3 初态覆盖筛查 25/30,失败按任务聚类,Task 8 的多物体顺序操作是唯一系统性弱点。针对 Task 5,我读了 LIBERO 的 BDDL 和 predicate 源码,确认成功只取决于书的 body 中心是否落入 back region 的 AABB——而该区域 x 半宽只有 27.75 mm,是三轴最窄的。受控初态实验里 4 条失败全部是 x 轴越界,成功样本的 x 余量最小只有 0.002 mm,所以结论是:模型的放置分布横跨了最窄轴的判定边界,90% 成功率就是这个分布的界内概率,Day 23 的失败和后来 11/11 的重复成功是同一分布的不同采样。我还做了确定性审计:policy RNG 无法显式固定,轨迹级复现不成立但任务级成功稳定,所以成功率评估靠重复采样而不是 seed 扫描。部署上 bf16 稳态均值 83.2 ms、P95 99.9 ms、峰值显存 24.11 GiB,冷启动 24.85 秒是唯一工程瓶颈。
