# HRNet 模糊分类项目

本项目使用 HRNet-W18 模型来对图像的模糊半径（0-9）进行分类。

## 最近更新

### 1. 标签范围修复 (0-9)
- **问题**: 训练脚本 `4_train.py` 之前使用的是 1-10 的标签范围并减 1，这导致在数据集更新为直接使用 0-9 时出现了错误。
- **修复**: 更新了 `4_train.py`，现在直接使用 0-9 的标签而无需修改。

### 2. 评估元数据格式
- **更新**: `5_evaluate.py` 脚本已更新，可以处理新的元数据格式，该格式包含 3 个元素：`(key, label, sharpness_map)`。

### 3. Tensor 数据库生成 (新)
我们添加了一个新的处理流，将 HRNet 的特征（10x15x20 张量）导出到一个单独的 LMDB 数据库中以供未来使用。

- **脚本**: `3_generate_tensor_db.py`
  - 从 `coc_train.lmdb` 加载图像。
  - 运行 HRNet 推理。
  - 将输出的张量 `[10, 15, 20]` (通道数, 高度, 宽度) 保存到 `coc_tensor_10x15x20.lmdb`。
  - **注意**: 分辨率被自然下采样了 32 倍 (480/32=15, 640/32=20)。

- **验证**: `7_verify_tensor_performance.py`
  - 读取生成的张量。
  - 应用 **Top-k Soft Expectation (Top-k 软期望)**:
    1. 计算 10 个类别的 Softmax 概率。
    2. 选择 Top-3 概率（将其余屏蔽为 0）。
    3. 重新归一化概率使其总和为 1。
    4. 计算加权的期望半径。
  - 验证生成的特征是否与真实标签匹配。
  - 输出每个标签的详细误差直方图。

## 关键脚本

| 脚本 | 描述 |
| :--- | :--- |
| `2_preprocessing.py` | 从源图像生成训练数据集 (LMDB)。 |
| `4_train.py` | 在数据集上训练 HRNet-W18 模型。 |
| `5_evaluate.py` | 在测试集上运行评估循环。 |
| `6_evaluate_single_image.py` | 在单张图像上可视化模型预测（带有裁剪图）。 |
| `3_generate_tensor_db.py` | **[新]** 生成特征 Tensor 数据库。 |
| `7_verify_tensor_performance.py` | **[新]** 验证 Tensor 数据库质量。 |

## 使用方法

### 生成 Tensor 数据库
```bash
python 3_generate_tensor_db.py
```

### 验证 Tensor 数据库
```bash
python 7_verify_tensor_performance.py
```
这会打印出每个标签（0-9）的详细误差分析表，显示准确率和误差分布。
