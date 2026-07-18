# VLA Model Survey (Update)
本文档整理 VLA 领域代表性模型的架构、动作表示与演进关系，并说明本项目所用 π0.5 的位置。整理时间：Day14，为 docs/vla_model_survey.md 的更新版。
## Overview
| Model | 机构 / 年份 | 骨干 | 动作表示 | 核心思想 |
|---|---|---|---|---|
| RT-1 | Google / 2022.12 | EfficientNet + Transformer（约 35M） | 离散 token（256 bins） | 大规模模仿学习，13 万条演示 |
| RT-2 | Google DeepMind / 2023.07 | PaLI-X 55B / PaLM-E 12B | 动作当文本 token | VLM 联合微调，网络知识迁移，首提 VLA |
| OpenVLA | Stanford 等 / 2024.06 | Prismatic-7B（Llama2 + DINOv2 + SigLIP） | 离散动作 token | 开源 7B VLA，OXE 97 万轨迹，LoRA 微调 |
| π0 | Physical Intelligence / 2024.10 | PaliGemma 3B + 300M 动作专家 | flow matching 连续动作块 | 高频（50Hz）连续控制，跨本体预训练 |
| π0.5 | Physical Intelligence / 2025.04 | 同 π0 架构改进 | FAST 离散 token + flow matching | 异构共训 + 分层推理，开放世界泛化 |
| RDT-1B | 清华 / 2024.10 | 1.2B Diffusion Transformer | 扩散去噪动作块 | 双臂基座，统一 128 维动作空间 |
| GR00T N1 | NVIDIA / 2025.03 | Eagle-2 VLM + DiT 动作头 | flow matching | 人形基座，双系统慢思考+快控制 |
| π0.6 / π*0.6 | Physical Intelligence / 2025.11 | π0.5 升级 | 同 π 系 | RECAP 真机强化学习 |
## RT-1
Google 2022 年 12 月发布，机器人 Transformer 路线的起点。
架构：EfficientNet-B3 提取图像特征（FiLM 语言条件化）→ TokenLearner 压缩 → decoder-only Transformer 输出动作。
动作表示：每维离散为 256 bins，作为 token 预测。
数据：Everyday Robots 平台 17 个月采集约 13 万条演示、700+ 任务。
意义：证明大数据 + Transformer + 模仿学习在真机操作上可行，3Hz 实时控制，已知任务成功率约 97%。
局限：语言理解与泛化有限，离散化损失动作精度。
## RT-2
Google DeepMind 2023 年 7 月发布，正式提出 VLA（Vision-Language-Action）概念。
核心思想：将预训练 VLM（PaLI-X 55B / PaLM-E 12B）在网络数据 + 机器人数据上联合微调，动作直接表示为文本数字 token。
关键收获：网络知识迁移到机器人控制，出现语义推理能力（如"拿起灭绝的动物"→ 抓恐龙玩偶），未见场景表现较 RT-1 约翻倍。
局限：55B 模型推理昂贵，闭源，控制频率低。
## OpenVLA
2024 年 6 月发布，第一个有广泛影响力的全开源 7B VLA。
架构：Prismatic-7B——视觉 DINOv2 + SigLIP 双编码器融合，语言 Llama-2 7B；动作离散 256 bins 占用词表低频 token。
数据：OXE 约 97 万条真机轨迹。
结果：比 RT-2-X（55B）绝对成功率高约 16.5%，参数量约 1/7；支持 LoRA 微调与量化推理。
与本项目关系：本项目 LIBERO 数据上游 openvla/modified_libero_rlds 来自 OpenVLA 团队；OpenVLA 在 LIBERO 四套件的微调结果（平均约 76%，Long 套件偏低）是 π0.5 对比的 baseline。
## π0
Physical Intelligence 2024 年 10 月发布。
架构：PaliGemma 3B VLM 骨干 + 约 300M 动作专家（action expert），注意力连接。
动作表示：flow matching 直接生成连续动作块（action chunk，horizon 50），控制频率可达 50Hz，避开离散化损失。
数据：1 万+ 小时自采多平台真机数据 + OXE。
## π0.5
Physical Intelligence 2025 年 4 月发布，核心改进是开放世界泛化，两个关键设计：
一、异构数据共训（co-training）：机器人数据之外混入网络多模态数据（VQA、检测、caption）、口头指令、子任务标注，保留语义理解能力。
二、分层推理：同一模型先以文本形式预测高层语义子任务，再生成底层连续动作；训练结合 FAST 离散动作 token 与 flow matching 两种目标。
效果：能在未见过的真实家庭完成清理厨房、整理卧室等长程任务；在 LIBERO 四套件成功率普遍 90%+，明显高于 OpenVLA 微调结果。
本项目使用的 pi05_libero 即 openpi 官方的 π0.5 + LIBERO 微调配置。
## RDT-1B
清华 2024 年 10 月发布，当时最大的扩散式机器人基座模型（1.2B）。
架构：视觉 SigLIP、语言 T5-XXL 编码，主体 Diffusion Transformer，扩散去噪生成动作块。
关键设计：统一可解释动作空间——将不同机器人动作映射到物理含义统一的 128 维空间，支持在 46 个异构数据集（约百万级轨迹）上预训练。
特点：主打双臂协同操作，在 ALOHA 平台微调评测；代表扩散生成路线，与 π 系 flow matching 同属连续动作生成阵营。
## GR00T N1
NVIDIA 2025 年 3 月发布的开源人形机器人基座模型。
架构为显式双系统：System 2 = Eagle-2 VLM（约 2B），负责场景理解与指令解析，低频运行（约 10Hz）；System 1 = Diffusion Transformer 动作头，flow matching 生成高频连续动作（可达 100Hz 级）。
数据金字塔：底层网络视频（世界知识）、中层合成数据（仿真生成）、顶层真机演示（人形遥操作）。
意义：将慢思考 + 快控制做成显式双模块，面向人形全身控制，后续有 N1.5 等迭代。
## π0.6 / π*0.6
Physical Intelligence 2025 年 11 月发布。π0.6 为 π0.5 基座升级；重点是 π*0.6 引入的 RECAP 方法（RL with Experience and Corrections via Advantage-conditioned Policies）。
核心思想：从真机部署经验中学习——收集自主执行数据与人工纠正数据，训练价值函数估计 advantage，用 advantage 条件化策略，使模型学会区分好坏行为。
效果：在做浓缩咖啡、叠衣物、组装纸箱等长程任务上，较纯模仿基线吞吐量约翻倍、失败率约减半，可长时间连续运行。
意义：代表 VLA 从模仿学习为主走向真机强化学习闭环。
## 技术演进主线
第一步 RT-1：大数据 + Transformer 模仿学习可行，动作离散 token。
第二步 RT-2 / OpenVLA：预训练 VLM 知识迁移，VLA 范式确立并开源化。
第三步 π0 / RDT / GR00T：动作表示升级为连续生成（flow matching / diffusion），高频控制与动作块预测，出现分层结构。
第四步 π0.5 / π0.6：开放世界泛化（异构共训 + 分层推理）→ 真机 RL 自我改进（RECAP）。
## 本项目位置
本项目 = 用第三/四代 VLA 模型（π0.5）在第一代标准 benchmark（LIBERO）上跑通"数据检查 → 微调 → 评测"的完整工程闭环。
注：文中 OpenVLA 约 76%、π0.5 90%+ 等数字为社区公开结果的大致水平，正式引用前请核对论文原文。