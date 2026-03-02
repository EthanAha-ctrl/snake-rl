# CoC: Vision-Based Guessing Game

这是一个基于强化学习 (SAC) 的 Agent，目标是在[0, 1]范围内猜测一个数。
观测值 (Observations) 不是直接给出的数值差，而是通过模拟人类视觉误差生成的。具体来说，我们使用了一个预训练的 HRNet (来自 `reference/project7`) 提取的特征 Tensor，并根据 Soft Expectation 计算出感知到的模糊半径。

## 核心机制

### 环境 (`coc_env.py`)
- **目标**: 猜出 `ground_truth` (0-1)。
- **动作**: `[Guess, Trigger]`。
    - `Guess`: 调整当前的猜测值。
    - `Trigger`: 提交猜测 (当 > 0.5)。
- **观测模拟 (Vision Simulation)**:
    1.  Agent 给出一个猜测值 `guess`。
    2.  将其映射到 [0, 10] 范围。
    3.  从 `reference/project7/data/coc_tensor_10x15x20.lmdb` 中读取相邻 Label 的 HRNet 特征张量。
    4.  对张量进行线性插值混合。
    5.  计算混合张量的 **Top-2 Soft Expectation**，得到 `vision_radius`。
    6.  最终观测值为 `abs(vision_radius/10.0 - ground_truth)`。

### 算法 (`sac_trainer.py`)
- 使用 **Soft Actor-Critic (SAC)** 算法。
- **Actor 网络**: 双头输出，分别输出连续的猜测值 (Sigmoid) 和离散的触发概率 (Gumbel Softmax)。
- **History Stacker**: (`history_stacker.py`) 堆叠历史观测帧作为输入。

## 文件结构

- `coc_env.py`: 核心环境逻辑，集成了 LMDB Tensor 读取。
- `sac_trainer.py`: SAC 算法实现 (Actor, Critic, ReplayBuffer)。
- `train.py`: 训练入口脚本。
- `evaluate.py`: 评估脚本。
- `history_stacker.py`: 观测历史堆叠工具。
- `my_gym.py`: 简化的 Gym 接口。
- `reference/project7`: 视觉模型 (HRNet) 的原始项目及数据来源。

## 数据依赖

本项目依赖于以下数据文件 (位于 `reference/project7/data/`):
- `coc_tensor_10x15x20.lmdb`: HRNet 提取的特征数据库。
- `coc_meta.pkl`: 对应的元数据。

## 运行

训练模型:

```bash
python train.py
```
