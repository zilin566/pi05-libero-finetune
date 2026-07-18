# VLA Dataset Survey
本文档整理 VLA / 具身智能领域的代表性数据集，并说明本项目所用 LIBERO 数据在生态中的位置。整理时间：Day14。
## Overview
| Dataset | 年份 | 类型 | 规模 | 机器人 | 特点 |
|---|---|---|---|---|---|
| LIBERO | 2023 | 仿真 | 130 任务 / 4 套件 | Franka Panda（仿真） | 终身学习 benchmark，语言条件操作，评测标准统一 |
| OXE (Open X-Embodiment) | 2023 | 真机聚合 | 100 万+ 轨迹 / 22 种本体 | 多种 | 60+ 数据集聚合，RLDS 格式，跨本体迁移 |
| DROID | 2024 | 真机 | 约 7.6 万轨迹 / 350+ 小时 / 564 场景 | Franka Panda | 13 个机构统一平台分布式采集，多视角 |
| AgiBot World | 2024-2025 | 真机 | 100 万+ 轨迹 / 100 台机器人 | 智元 G1 双臂 | 长程任务、双臂灵巧操作，GO-1 训练数据 |
## LIBERO
发布于 2023 年（NeurIPS 2023），基于 robosuite / MuJoCo 的仿真终身学习 benchmark，机器人为 Franka Panda 单臂。
任务结构分为 4 个套件：
LIBERO-Spatial：10 个任务，考察空间关系理解（同样的物体、不同摆放）
LIBERO-Object：10 个任务，考察物体类型泛化（同样的布局、不同物体）
LIBERO-Goal：10 个任务，考察任务目标泛化（同样的物体布局、不同目标）
LIBERO-100：100 个任务，拆分为 LIBERO-90（预训练）和 LIBERO-10 / LIBERO-Long（长程评测）
与本项目数据的对应关系：
本项目使用 physical-intelligence/libero，包含 Spatial / Object / Goal / 10 四个套件，4 × 10 = 40 个任务，对应本地 meta/tasks.jsonl 中 num_tasks_loaded = 40。
原始每任务约 50 条演示，四套件理论上约 2000 条轨迹；本地实际 1693 个 episode，推测为上游 openvla/modified_libero_rlds 转换时过滤了失败重放和无效样本（待查转换脚本确认）。
数据字段：image / wrist_image 为 256×256 RGB，state 8 维，actions 7 维（6 维末端位姿增量 + 1 维 gripper），与本项目 Day10-Day13 检查结果一致。
定位：VLA 微调实验的标准入门 benchmark，仿真评测成本低、任务定义清晰、社区对比结果多。
## OXE (Open X-Embodiment)
发布于 2023 年 10 月，Google DeepMind 联合 20+ 机构发起，将 60 多个已有机器人数据集统一为 RLDS 格式聚合。
规模：100 万+ 真机轨迹，覆盖 22 种机器人本体、500+ 技能。
核心结论：跨本体混合数据训练的 RT-1-X / RT-2-X 在多数本体上优于单一本体数据训练的模型，证明跨本体正迁移存在。
意义：使机器人大数据预训练成为可行路线，OpenVLA（约 97 万轨迹）、π0 等模型的预训练数据大量来自 OXE。
与本项目的关系：本项目 LIBERO 数据上游 openvla/modified_libero_rlds 即 RLDS 格式，属于 OXE 生态标准格式。
## DROID
发布于 2024 年，大规模真机操作数据集：约 7.6 万轨迹、350+ 小时、564 个场景、86 类任务，13 个机构用统一 Franka 硬件平台分布式采集，Oculus VR 遥操作。
每条轨迹包含 2 个外部视角 + 1 个腕部视角，场景覆盖办公室、厨房、实验室等真实环境。
特点：相比 OXE 的聚合旧数据，DROID 是统一硬件、统一协议采集的新数据，场景多样性和标注一致性更好。
定位：真机单臂操作预训练/微调的高质量数据源，openpi 提供 DROID 相关训练配置。
## AgiBot World / GO-1
AgiBot World 是智元机器人发布的大规模真机数据集：100 万+ 轨迹，来自 100 台同构双臂机器人，覆盖家居、零售、工业、餐饮、办公等 100+ 真实场景，包含大量长程任务和灵巧手操作。
GO-1（Genie Operator-1，2025）是基于该数据训练的 VLA 基座模型，采用 ViLLA 框架：VLM + 隐式动作规划器（Latent Planner）+ 动作专家（Action Expert），通过隐式动作建模利用无动作标注的视频数据。
意义：代表国内真机数据 + 基座模型路线，统一本体采集，规模对标 OXE。
## 数据演进主线
单一本体自采（RT-1 时代）→ 跨本体聚合（OXE）→ 统一协议高质量真机（DROID / AgiBot World）→ 部署经验数据（π*0.6 RECAP）。
## 本项目数据链路
LIBERO 仿真 benchmark → openvla/modified_libero_rlds（RLDS）→ physical-intelligence/libero（LeRobot）→ pi05_libero 训练配置。