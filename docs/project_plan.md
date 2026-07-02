
# π0.5-LIBERO 项目路线图

## 阶段一：项目启动

目标：

- 建立 GitHub 项目；
- 写 README 初版；
- 明确项目路线；
- 明确最终简历包装方式。

产出：

- README.md
- docs/project_plan.md
- notes/day1_project_start.md

## 阶段二：openpi 环境搭建

目标：

- 克隆 openpi 仓库；
- 配置 Python / Conda 环境；
- 安装依赖；
- 确认 π0.5 相关代码和配置位置。

产出：

- 环境安装记录；
- 报错与解决方案记录；
- openpi 仓库结构笔记。

## 阶段三：LIBERO 数据准备

目标：

- 理解 LIBERO 数据格式；
- 明确 observation、action、language instruction 的存储方式；
- 整理训练所需数据字段；
- 完成数据格式转换。

产出：

- 数据字段说明；
- 数据转换脚本；
- 数据样例可视化。

## 阶段四：π0.5 微调

目标：

- 找到 π0.5 训练配置；
- 修改数据路径和训练参数；
- 小规模跑通训练；
- 记录训练 loss 和 action error。

产出：

- 训练配置文件；
- 训练日志；
- loss 曲线；
- 显存占用记录。

## 阶段五：仿真评测

目标：

- 启动 policy server；
- 接入 LIBERO 仿真环境；
- 运行若干测试任务；
- 统计 success rate 和失败案例。

产出：

- 评测结果表；
- 成功/失败案例截图；
- failure analysis。

## 阶段六：项目包装

目标：

- 整理 README；
- 整理结果图；
- 写简历项目描述；
- 准备面试讲解稿。

最终产出：

- GitHub repo；
- 项目 README；
- 训练结果；
- 评测结果；
- 简历项目描述。