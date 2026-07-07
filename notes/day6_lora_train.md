# Day6：LoRA 100-step 小规模训练

## 1. 今日目标

Day6 的目标是在 Day5 跑通 10 step 的基础上，继续验证 LoRA 配置能否稳定训练更长步数。

使用配置：

```text
config = pi05_libero_debug_lora
batch_size = 1
model.max_token_len = 64
num_train_steps = 100
log_interval = 10
save_interval = 100
num_workers = 0
wandb_enabled = False

---

## 2. 训练命令

Day6 使用的训练命令：

```bash
uv run scripts/train.py pi05_libero_debug_lora \
  --exp-name=day6_lora_bs1_s100_tok64 \
  --model.max-token-len=64 \
  --num-train-steps=100 \
  --log-interval=10 \
  --save-interval=100 \
  --num-workers=0 \
  --no-wandb-enabled \
  --overwrite \
  2>&1 | tee logs/day6_lora_bs1_s100_tok64.txt

Day6 成功完成 100 step 小规模训练。
Step 0: grad_norm=1.0725, loss=0.0285, param_norm=1803.7705
Step 10: grad_norm=1.7093, loss=0.0938, param_norm=1803.7708
Step 20: grad_norm=0.7571, loss=0.0554, param_norm=1803.7731
Step 30: grad_norm=0.8927, loss=0.0829, param_norm=1803.7760
Step 40: grad_norm=1.3080, loss=0.1145, param_norm=1803.7786
Step 50: grad_norm=0.6856, loss=0.0649, param_norm=1803.7806
Step 60: grad_norm=0.9716, loss=0.0566, param_norm=1803.7832
Step 70: grad_norm=1.5809, loss=0.0936, param_norm=1803.7855
Step 80: grad_norm=2.1796, loss=0.1066, param_norm=1803.7875
Step 90: grad_norm=0.8393, loss=0.0761, param_norm=1803.7899

保存路径 checkpoints/pi05_libero_debug_lora/day6_lora_bs1_s100_tok64

LIBERO batch
→ norm stats
→ pi05_base checkpoint
→ LoRA train state
→ compute_loss
→ backward / update
→ logging
→ checkpoint save

loss 能正常计算
grad_norm 能正常打印
param_norm 持续变化
参数发生更新
训练没有 OOM
checkpoint 保存成功


| 模型          | 一句话理解                                         | 主要应用                                |
| ----------- | --------------------------------------------- | ----------------------------------- |
| **RT-1**    | 早期机器人 Transformer，用图像和语言直接预测动作                | 多任务机器人模仿学习                          |
| **RT-2**    | 把视觉语言模型的语义知识迁移到机器人动作                          | 需要语义理解的机器人控制                        |
| **OpenVLA** | 开源 VLA 模型，适合复现和微调                             | LIBERO / OXE / 机器人策略微调              |
| **π0**      | flow-based VLA，用 VLM + action expert 生成连续动作   | 通用机器人操作策略                           |
| **π0.5**    | π0 的升级版，更强调开放环境泛化                             | 你当前 openpi / LIBERO 项目主线            |
| **π0.6**    | π0.5 的增强版，更强 backbone 和 prompt / conditioning | 更复杂真实任务、后续 RL / experience learning |
| **π0.7**    | 可 steerable 的通用机器人模型，用多模态上下文控制“怎么做”           | 多策略、多机器人、复杂长程任务                     |
| **GR00T**   | NVIDIA 面向人形机器人的 VLA foundation model          | humanoid manipulation / 人形机器人技能     |
| **RDT**     | diffusion transformer 机器人策略模型                 | 双臂、连续动作、多模态动作分布                     |

当前项目实际主线：

```text
openpi / π0.5
→ physical-intelligence/libero
→ norm stats
→ pi05_libero_debug_lora
→ LoRA + max_token_len=64
→ 小规模训练日志与 checkpoint