# 视场相关模糊 (Field-Dependent Blur) 实施指南

本指南详细记录了在使用模拟/占位的 Circular SFR (Spatial Frequency Response) 点扩散函数 (PSF) 构建**视场相关模糊处理管线 (Field-Dependent Blur Pipeline)** 时所采取的步骤。

## 实施内容

1.  **创建了 `sfr_processor.py`**:
    -   实现了一个基础框架 `multiframe_average(image_paths)`。当未来有真实数据时，此函数将用于对多张带有噪点的原始图像进行平均降噪。
    -   实现了 `get_mock_psf_grid(radius, grid_h, grid_w)` 来动态生成一个符合物理规律的 PSF 网格。
    -   这些模拟的 PSF 主要是用来复现真实镜头的像差：
        -   网格中心的 PSF 会保持相对锐利，或者带有一点点均匀的轻微高斯模糊（即便是 `radius == 0` 的情况，也会施加基础模糊以模拟镜头固有的不完美）。
        -   网格边缘的 PSF 会改变其协方差矩阵来进行拉伸，以此模拟径向和切向的像散 (Astigmatism) 及彗差 (Coma)，从而形成细长的核 (Kernel)。

2.  **更新了 `blur_ops.py`**:
    -   移除了原本均匀的圆形 CPU/GPU 卷积核 (`cv2.circle`)。
    -   新增了 `field_dependent_convolution(img_tensor, psf_grid)`，该函数能正确执行空间可变的模糊：
        -   它会提取网格尺寸（比如 3x3），并使用网格中的**每一个** PSF 对整张图像分别进行 2D 卷积。
        -   随后应用**空间融合 (Spatial Blending)**（基于目标区域的双线性插值权重）来无缝合并这些卷积结果，防止图像上出现生硬的网格拼接边界。
    -   更新了 `core_blur` 以便从 `sfr_processor.py` 获取这些空间网格，并应用新的卷积方法。

3.  **数据集管线兼容性验证**:
    -   由于 `core_blur` 的输入和输出接口保持不变（仍然接收 `a_list`/`b_list` 并返回模糊后的批次图像和清晰度图），因此 `preprocessing.py` 可以零成本无缝调用这套全新的物理引擎。
    -   同理，`generate_tensor_db.py` 也能直接摄取这些新生成的图像，无需进行任何代码层面的修改。

## 验证结果

我们编写并运行了测试脚本 `test_field_dependent_blur.py`:
-   它成功生成了一个 480x640 的合成网格图像。
-   它针对模糊半径 0、1 和 2 分别运行了 `core_blur`。
-   输出结果正确生成了 3 张融合后的图像。在这些图像中，你可以看到镜头像差是如何基于空间位置平滑地改变网格线条的。

*这套处理管线目前已经完全就绪。一旦你拍摄好真实的 Circular SFR Chart 数据，随时可以将其接入系统！*
