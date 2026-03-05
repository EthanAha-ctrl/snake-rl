# HRNet 模糊度特征提取 (原项目7)

本项目包含用于强化学习环境的 HRNet 视觉特征提取流程。此目录已经过精简，移除了原有的训练和评估脚本，仅保留与强化学习环境部署和数据生成相关的核心推理逻辑。

## 核心流程：张量数据库生成

我们将原始图像的 HRNet 特征（10x15x20 张量）提取并导出到独立的 LMDB 数据库中供强化学习环境使用。

- **脚本**：`generate_tensor_db.py`
  - 核心光学模糊合成算法见 `blur_ops.py`。
  - 在运行时合成具有前/背景相对散景的对焦栈图像。
  - 运行定义在 `model.py` 中的 HRNet-W18 推理。
  - 将输出张量 `[10, 15, 20]`（通道数, 高度, 宽度）保存至 `data/coc_tensor_10x15x20.lmdb`。
  - **注意**：分辨率自然下采样了 32 倍（480/32=15, 640/32=20）。

## 关键脚本

| 脚本 | 描述 |
| :--- | :--- |
| `model.py` | MIL-PatchCNN 网络架构定义。 |
| `blur_ops.py` | 光学模糊、弥散圆 (CoC) 模拟以及前背景相对景深合成算法。 |
| `1_preprocessing.py` | 生成单平面模糊图像数据集。 |
| `2_visualize_dataset.py` | 用于检查预处理数据集及清晰度标签。 |
| `3_train.py` | 训练 MIL-PatchCNN 网络。 |
| `generate_tensor_db.py` | (可选参考) 生成强化学习环境所需的特征张量数据库。 |

## 使用方法

### 生成张量数据库
```bash
python 1_preprocessing.py
python 3_train.py
```
