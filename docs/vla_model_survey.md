# VLA 模型调研（简版）

目的：梳理主流 VLA 模型的输入 / 输出 / 动作表示 / 应用，明确当前项目主线。

| 模型      | 输入                                 | 输出                  | 动作表示方式                         | 主要应用                | 和当前项目关系                      |
| ------- | ---------------------------------- | ------------------- | ------------------------------ | ------------------- | ---------------------------- |
| RT-1    | 图像 + 语言 + 状态                       | 机器人动作               | 离散动作 token                     | 多任务机器人模仿学习          | VLA 早期代表，作为背景了解              |
| RT-2    | 图像 + 语言                            | 语言 token + 动作 token | 动作 token 化                     | 将 VLM 语义知识迁移到机器人控制  | 理解 VLM 到 VLA 的关键模型           |
| OpenVLA | 图像 + 语言 + 状态                       | 机器人动作               | VLA 动作输出                       | 开源 VLA 复现、微调、评测     | 参考其 LIBERO 项目组织方式            |
| π0      | 图像 + 语言 + 状态                       | 连续动作                | flow matching action           | 通用机器人策略             | π0.5 的前身                     |
| π0.5    | image + wrist_image + state + task | actions             | 连续动作序列 / action chunk          | LIBERO 微调、openpi 主线 | 当前项目核心模型                     |
| π0.6    | 多模态观测 + prompt + 状态                | 子任务 + 动作            | 分层动作生成                         | 更复杂真实任务             | π0.5 后续演进方向                  |
| π0.7    | 多模态 context + 指令                   | 可控机器人动作             | steerable action generation    | 多策略、长程任务、上下文控制      | 前沿了解，不是当前主线                  |
| GR00T   | 语言 + 图像 + 人形机器人观测                  | 人形机器人动作             | humanoid continuous control    | 人形机器人技能学习           | 与当前 LIBERO manipulation 方向不同 |
| RDT     | 多模态观测 + 语言 + 状态                    | 连续机器人动作             | diffusion / transformer policy | 双臂、连续控制、多峰动作生成      | 动作生成路线参考                     |

## RT-1 / RT-2（Google）

- 输入：第三视角图像 + 语言指令
- 输出：离散化动作 token（每维切成 256 个 bin，自回归输出）
- 动作表示：离散 token
- 主要应用：Google 真机办公室/厨房操作任务；RT-2 用 VLM（PaLI-X / PaLM-E）直接输出动作 token，证明网络知识能迁移到机器人控制
- 与本项目关系：VLA 开山工作，仅调研，不复现

## OpenVLA

- 输入：单视角图像 + 语言指令
- 输出：7 维离散动作 token
- 动作表示：离散 token（Llama 2 7B backbone + DINOv2/SigLIP 视觉编码器）
- 主要应用：在 Open X-Embodiment 上训练，支持 LoRA 微调，LIBERO 是常用评测基准
- 与本项目关系：本项目最初的参考对象；LIBERO 数据源（openvla/modified_libero_rlds）来自 OpenVLA 生态，但训练栈已换成 openpi

## π0 / π0.5（Physical Intelligence）

- 输入：多视角图像（主视角 + 腕部）+ 机器人状态 + 语言指令
- 输出：连续动作块（action chunk，horizon = 10+）
- 动作表示：连续动作 + flow matching，由 action expert 生成，不做离散化
- π0：PaliGemma VLM + 300M action expert
- π0.5：在 π0 基础上强化开放世界泛化（异构数据 co-training、支持离散状态输入等）
- π0.6 / π0.7：后续迭代版本，细节以官方发布为准
- 主要应用：真实机器人操作、开放家庭环境任务
- 与本项目关系：**当前项目主线模型**，用 openpi 官方代码在 LIBERO 上做 LoRA 微调

## GR00T（NVIDIA）

- 输入：图像 + 语言 + 机器人状态
- 输出：连续动作
- 动作表示：双系统架构，System 2（VLM 理解）+ System 1（diffusion/flow 动作头）
- 主要应用：人形机器人基础模型（GR00T N1 / N1.5）
- 与本项目关系：仅调研；"VLM + 动作专家"的架构思路与 π0 系列同源

## RDT（清华 RDT-1B）

- 输入：多视角图像 + 语言 + 本体感知
- 输出：连续动作
- 动作表示：Diffusion Transformer（1B 参数）
- 主要应用：双臂操作任务
- 与本项目关系：仅调研，代表 diffusion 路线的动作生成

## 总结

```text
离散 token 路线：RT-1 / RT-2 / OpenVLA
连续动作路线：π0 / π0.5（flow matching）、GR00T、RDT（diffusion）

当前项目真正使用的是 openpi / π0.5。
OpenVLA、RT、GR00T、RDT 是调研对象，不是当前复现主线。
```