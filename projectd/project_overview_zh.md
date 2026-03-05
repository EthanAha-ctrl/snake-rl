# CoC (Circle of Confusion)：基于视觉模拟的强化学习猜数字项目

这是一份关于 `snake-rl/projectd` 项目的快速指南，旨在帮助您快速了解该项目的背景、结构和核心逻辑。

## 1. 项目背景与目标

本项目训练了一个基于**强化学习（SAC，Soft Actor-Critic）**的智能体（Agent），目标是在闭区间 `[0, 1]` 内“猜出”一个目标数值（`ground_truth`）。

**核心创新点：视觉观测模拟 (Vision Simulation)**
不同于传统的强化学习环境直接提供精确的数值误差，本游戏中的环境观测值（Observations）是模拟人类视觉“模糊度”（Circle of Confusion / 弥散圆）生成的：
- 系统不直接告诉 Agent 差值是多少，而是根据差值（`guess - ground_truth`）大小，模拟出某种程度的“视觉模糊特征”。
- 这些特征取自一个预训练的 HRNet 模型（数据来源于 `reference/project7`）。系统在运行时查阅/插值这些提取好的特征张量（Tensor），计算出Soft Expectation（感知模糊半径或锐度），将其作为 Agent 可以观测到的信息。
- **[最新更新] 背景相对散景模拟**：底层视觉特征的生成（`reference/project7`）最近从“绝对独立的背景深度”切换为了基于目标的“前景对齐+相对背景散景（Delta Depth）”逻辑。这意味着环境现在模拟的光学模糊更贴近真实物理镜头（即：环境以正确对焦物体为基准，背景模糊度是相对的）。

## 2. 核心文件及模块解析

### 2.1 环境模块 (Environment)
基于 OpenAI Gym 接口封装（见 `my_gym.py`），提供连续动作空间和视觉模拟观测：
- **`coc_env.py`**：基础环境。
  - **动作 (Action)**：`[Guess, Trigger]`。Agent给出连续的猜测值调整 `Guess` 和决定是否提交的离散动作 `Trigger`（>0.5视为提交）。
  - **观测 (Observation)**：`abs(vision_radius/10.0 - ground_truth)`，其中半径基于 Tensor 数据库的插值动态计算获取。
- **`coc_sharpness_env.py`**：扩展环境。
  - 观测空间更丰富，除了预期的模糊半径外还包含了 **Sharpness（锐度）** 特征 (`[Sharpness, Expected Radius]`)。

### 2.2 强化学习算法模块
系统使用了 Soft Actor-Critic (SAC) 以及一些扩展：
- **`sac_trainer.py`**：
  - **Actor网络 (TanhGaussianPolicy)**：双头设计。同时输出连续的猜测调整量（Sigmoid激活）和离散的触发概率（Gumbel Softmax）。
  - **Critic网络**：使用了双 Q 网络架构（Double Q-Learning）以减轻过估计问题。
  - **Behaviour Cloning (BC)**：代码中包含了 `update_bc` 逻辑，支持旁路 Critic 和 SAC ，直接基于提取好的专家数据/规则进行行为克隆训练。
- **`history_stacker.py`**：实现了环境历史观测帧的堆叠融合，赋予 Agent 时序感知能力。
- **`transformer_encoder.py`**：提供了一个基于 Transformer 的特征编码模块，可用于提取历史观测 / 高维视觉特征中的复杂序列关系。

### 2.3 入口与工具脚本
- **`train.py`**：主训练脚本，用于实例化环境、Trainer并启动 SAC 或混合训练流程。
- **`evaluate.py`**：用于加载已保存权重（如目录下众多的 `sac_coc_best*.pth` 文件）验证 Agent 性能的过程。
- **`visualize_sharpness.py`**：用于分析和可视化环境中给出的“锐度（Sharpness）”与实际误差之间的映射关系的工具。

## 3. 依赖的数据
项目在运行和测试时强制依赖外部提取的特征张量数据，请确保以下路径下有对应数据（否则 `LMDB` 报错）：
- **元数据**：`reference/project7/data/coc_meta.pkl`
- **Tensor流**：`reference/project7/data/coc_tensor_10x15x20.lmdb`

## 4. 如何运行（快速上手）

若要启动训练：
```bash
python train.py
```
若要评估一个现存的好模型（如 `sac_coc_best.pth`）：
```bash
python evaluate.py
```

*注：项目依赖 `torch`, `lmdb`, `tensorboard` 等包，由于它使用真实的卷积/Transformer网络特征流以及离散+连续混合的动作空间控制，训练时建议配置GPU环境。*
