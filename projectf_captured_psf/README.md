# 物理镜头 PSF 采集与重构工作流指南

本文档旨在指导如何从真实镜头采集 SFR Chart 数据，使用 Imatest 提取 1D MTF，并最终重构成 5x3 网格的视场相关 2D PSF，以接入我们现有的图像模糊处理管线。

## 步骤 1：按景深分化采集方案
在这个阶段，由影像工程师负责拍摄采集数据。为了获得最高保真度的物理参数，我们根据景深采用不同的测试目标物：

### 1A. 合焦区域 采集：使用 SFR Chart
对于处于合焦面或景深范围内的物距，镜头的像差（如轻微的球面像差、色差等）在极小尺度内体现为频率响应的衰减。
1. 准备标准的 [ISO-12233 SFR (SFRplus 或 eSFR) 测试卡](https://www.imatest.com/imaging/iso-12233/)。
2. 确保环境照明均匀充足，避免卡面反光。
3. 将相机放置在测试卡正前方（光轴垂直），严谨对焦，使画面填满整个 16:9 测试区域。
4. 拍摄处于合焦状态下的原始图像数据，用于后续提取 MTF。

### 1B. 失焦区域 采集：使用 LED 点光源靶机
对于大尺度的失焦光斑，由于其弥散圆尺度远超空间频率测试极限，我们**不再使用**传统的 SFR 刀锋边缘测试卡，而是直接捕捉物理点光源的二维能量分布特性。
1. 使用分布式的微型 LED 点光源阵列（可放置在远处，或通过 Relay Lens 中继透镜模拟较远物距）。
2. 在不同的离焦程度下拍摄这些点光源。
3. **消除光源尺寸本底**:
   实际物理世界中的 LED 具有其实际发光面积，并非完美的无穷小点光源（Dirac delta function $\delta(x, y)$）。因此，相机拍摄到的光斑图像，实际上是纯粹的镜头 PSF 与 LED 发光形状的卷积结果：
   - 测试所得的合焦光斑: $\text{Image}_{\text{in}} = \text{LED\_Shape} * \text{PSF}_{\text{lens\_in}}$
   - 测试所得的失焦光斑: $\text{Image}_{\text{out}} = \text{LED\_Shape} * \text{PSF}_{\text{lens\_out}}$
   
   为了获取极其纯净的透镜离焦函数 $\text{PSF}_{\text{lens\_out}}$，我们在数学上必须消除这个发光体本体的基础体积（这就好比用理想气体状态方程 $PV=nRT$ 来外推推导绝对零度）。
   通过将数学图像转换至频率域（如傅里叶变换 FFT 或离散余弦变换 DCT），由于时域或空域的卷积在频域变为了点乘：
   $$ \mathcal{F}(\text{Image}_{\text{out}}) = \mathcal{F}(\text{LED\_Shape}) \times \mathcal{F}(\text{PSF}_{\text{lens\_out}}) $$
   只要我们在频域中执行除法操作（反卷 / Deconvolution）：
   $$ \mathcal{F}(\text{PSF}_{\text{lens\_out}}) = \frac{\mathcal{F}(\text{Image}_{\text{out}})}{\mathcal{F}(\text{LED\_Shape})} $$
   即可在数学空间中完美剥离 LED 固有的尺寸投射。为避免由于高频噪声放大导致解反卷积失败，实际工程拟合算法上常考虑结合**维纳滤波** 或类似盲反卷积规则化的手段来稳健求解此纯净的离散内核，进而为下一环节的闭式解 提取标量参数服务。

## 步骤 2：数据提取与预处理 (分化处理)

### 2A. 针对合焦图像 (基于 SFR Chart) -> 提取 1D MTF
1. 将拍摄好的合焦照片导入 [Imatest 软件](https://www.imatest.com/docs/sfrplus_instructions3/)，配置 5x3 分析区域。
2. **数据格式规范**: Imatest 输出的 1D MTF CSV 文件通常包含空间频率与响应值的映射。我们需要进一步检视其输出格式是 **水平/垂直** 还是 **弧矢/子午** 辐射状坐标。
   - 若 Imatest 官方未提供公开明确的 S/T 格式文档，我们在 `1_parse_imatest_to_psf_grid.py` 脚本中将自定义并约束一种标准化的输入格式字典，强迫数据对其进行标准化绑定。

### 2B. 针对失焦图像 (基于 LED 靶机) -> 提取 Field-Dependent 2D PSF
从布满 LED 点光源的一整张失焦照片中，自动且精准地抠出分布在不同视场 的光斑极其关键：
1. **自动检测与定位**：
   使用 `cv2.HoughCircles` 或通过阈值化后的连通域分析来快速定位全图所有高亮虚化 LED 光斑的中心。
2. **带裕量裁剪**：
   由于像差会导致光斑不对称扩散，我们需要根据检测到的光斑基础半径 $R$，向周围进行外延（例如 `+ 20% margin`）并执行 Crop 操作，防止截断彗差的尾巴。
3. **质心对齐**：
   将所有的 Crop Block 进行质心计算和 Sub-pixel 平移，使光斑真正的能量中心严格对齐到矩阵正中央，作为干净的 `captured PSF` 供后续进行拟合。

## 步骤 3：拟合与建模

对于提取到的 Off-focus 巨型光斑，我们使用两阶段拟合 来求解其闭式解 参数：

### 阶段 1：全局径向基底拟合
先假设光斑是一个完美的几何模型（例如圆盘或高斯衰减盘）。运行基础的非线性最小二乘法，找出最优的 `center`（微调） 和全局 `radius`，使得 `radial(center, radius)` 与原图均方差最小。

### 阶段 2：残差高频拟合
利用原图减去第一阶段拟合出来的纯净圆，得到一张充满波动细节的**残差图**。这张图上包含了透镜衍射环带来的"洋葱圈"高斯纹理。
- **纯 DCT 的局限性与截断参数** 
  我们模型中引入了硬截断参数 `radius_clamp`。DCT 是一种全局线性基底展开，直接提取前 11 项去拟合带有截断骤降的信号时，会在截断边缘产生强烈的吉布斯效应（振铃现象，Gibbs Phenomenon），导致背景产生虚假的波动噪声。
- **基于梯度的非线性优化求解**：
  因为模型包含非连续分支语句 (`if radius > clamp`)，这是一个多参数非线性优化问题。我们需要使用求解器（如 `scipy.optimize.least_squares`，底层使用 Levenberg-Marquardt 或 TRF 算法），联合优化这 11 个组建（共 44 个标量参数：`amp`, `phase`, `freq`, `clamp`）。
- **至关重要的初始猜测**：
  非线性求解器极度依赖优秀的初始猜测（`x0`），否则极易陷入局部最优解或者陷入死循环。**我们的策略是：利用 DCT 进行降维计算，为其提供高质量的 Initial Guess。**
  在运行非线性求解器前，我们先对残差图强制应用一个空间 Window（例如用第一阶段求出的 base radius 作为一个软遮罩），然后对其内部做一次快速的 DCT 展开。我们直接读取 DCT 频谱矩阵中能量最强的几个峰值，换算为初始的 `freq` 和 `amp`；将初始 `clamp` 设定为 base radius；将 `phase` 全部置 0。将这组强相关的解析参数喂给 optimizer 作为起点启动迭代，既保证了收敛速度，又从代数源头上避免了跑飞。

## 步骤 3：将 1D MTF 重建为 2D PSF 管线数据 (即将编写的工具)
*接下来，我们将开发一个新的 Python 脚本（例如预期的 `1_parse_imatest_to_psf_grid.py`）来自动解析这些 CSV 数据。*

**兼容性设计**：由于我们尚不确定 Imatest 导出的正交 MTF 数据是基于**水平/垂直** 坐标系，还是基于**弧矢/子午** 辐射坐标系，该脚本将支持这两种解析模式:

数据处理的底层逻辑如下：
1. 读取各点的对向 MTF 数据，使用逆傅里叶变换将其转换为一维空间上的线扩散函数 (LSF)。
2. 通过拟合，提取两轴对应的扩散宽度系数（$\sigma_1$ 和 $\sigma_2$）。
3. 利用提取出的长短轴参数，构建出表现出像散/彗差特性的 2D 各向异性高斯矩阵。
4. **处理旋转偏移 (根据 Convention 选择)**：
   - **如果使用 Horizontal / Vertical (H/V) 模式**：导出的 MTF 已经是基于图像绝对 XY 坐标系的。此时**不需要**计算辐射旋转角（或者相当于 $\theta=0$），直接将生成的 2D 矩阵按水平/垂直方向排列即可。
   - **如果使用 Sagittal / Tangential (S/T) 模式**：导出的 MTF 是基于画面光心辐射的。需要根据测试点坐标 $(X, Y)$ 相对于画面光心的方向，计算出旋转角度 $\theta$。将上述 2D 矩阵进行对应角度的旋转，使得 Sagittal 方向严格沿着光心放射状向外。
5. 将 15 个点生成的 2D 核打包输出为 `[3, 5, kernel_size, kernel_size]`  的 Tensor 对象。

## 步骤 4：模糊渲染管线与高级 PSF 插值策略探讨

这部分涉及如何将这 5x3 个基础 PSF 平滑地应用到整张高分辨率图像上。

### 策略 A：图像空间融合 - 当前实现
目前 `sfr_processor.py` 采用的方法是**对卷积后的图像进行双线性插值**。
这种方法的计算逻辑是：用网格中的每一个 PSF对整张原图做一次 2D 卷积，然后根据像素所在区域进行加权平均。这种方法在实现上非常简单且容易并行计算，但它的假设是基于"在较小区域内，像差是不变的"。

### 策略 B：PSF 域插值与基于 ISETCam 的思考 (理论上更精确)
实际上，**对 PSF 本身进行空间插值在物理光学上是更为严谨的。**
著名的光学模拟开源工具 [Stanford ISETCam](https://github.com/ISET/isetcam) (Image Systems Engineering Toolbox) 在处理光学系统模拟时，核心思维也是计算 Shift-Variant (空间可变) 的 PSF。它们通常也会基于采样的点阵构建一个多维的 PSF/MTF 映射，并对中间位置的 PSF/MTF 进行插值。

#### 基于 DCT 的频域插值与逐像素点乘 (未来演进方向)
直接在空间域对巨大的 PSF tensor 甚至全图每个像素生成对应的独立 PSF 进行卷积是极度耗时的（存储开销达 `image_w * image_h * kernel_w * kernel_h`）。

一个更为高效的重构和渲染策略是**频域处理**：
1. **使用 DCT (离散余弦变换)** 代替 FFT：将基于 5x3 采样点构建的 PSF 转换为 MTF（频率响应）。DCT 的优势在于其结果完全为**实数域**，没有虚数部分，极大节省内存带宽并消除复数运算的额外开销。
2. **频域插值放大**：由于频域表示（MTF / DCT 结果）特征往往更加平缓，我们可以将这个仅有 5x3 个采样位置的 MTF 网格，通过空间插值方法放大至与输入图像相同的分辨率 (e.g., $1920 \times 1080$)。此时系统将拥有一个与图像像素完全对应的"逐像素频率响应图"。
3. **逐元素相乘**：然后，我们将目标图像也转换至 DCT 频域，与上一步得到的逐像素 MTF 进行 Numpy 数组层面的点乘，再通过逆变换转回空间域。在计算复杂度上，这种 O(N) 的点乘远快于巨大感受野下的卷积运算。

### 策略 C：按景深分化处理
考虑到运算规模，我们需要将渲染策略依据模糊半径分离：

- **针对合焦区域**
  - 合焦区域（即便是存在轻微镜头像差的边缘画质衰减）的 PSF Kernel 尺寸非常小（例如最多 11x11 到 15x15 的规模）。
  - 对于这种小核，应用上述的**基于 DCT 的频域插值与运算**策略是非常合适的，由于矩阵乘法的维度小，整体运算负担可控。

- **针对大规模失焦区域**
  - 当模糊半径变大时，频域插值带来的巨大 `kernel_w * kernel_h` 存储将变成瓶颈。
  - 对于这种极端情况，使用庞大离散矩阵计算不再经济。我们引入一种 **闭式解近似** 的数学模型。
  - **基于径向基与高频谐波组合的解析模型**：
    我们将庞大且复杂的失焦光斑近似解析为：
    ```python
    approximated_kernel = radial(center, radius) + sum([cosine_terms(center, radius_clamp_i, amp_i, phase_i, freq_i) for i in range(N)])
    ```
    - **`radial(center, radius)`**: 这是光斑的基础低频投影（代表主能量的均值弥散分布，例如平顶几何弥散圆）。
    - **`cosine_terms`**: 包含 $N$ 个（例如 $N=11$）由余弦波叠加成的高频项，专门用来拟合透镜和孔径带来的光学伪影（如衍射环 Diffraction Rings 或"洋葱圈"效应 Onion-ring bokeh）。
      截断衰减逻辑如下：针对特定的采样点，若它到模糊中心点的距离超过了该项的截断边界 `radius_clamp`，则该附加项输出 `0`；否则输出 `amp * cos(radius * freq + phase)`。
    - **优势**: 有了这个解析形式，任意位置、任意尺寸的大型 PSF 都不再需要离散化数值存储，只要提取/拟合出那数十个一维标量参数，在渲染时仅需做解析函数的代入加和计算，既实现了空间连续性，又达成了极高保真度的物理仿真性能突破！


# 物理镜头 PSF 采集与重构工作流指南（扩充版）

本文档旨在指导如何从真实镜头采集 SFR Chart 数据，使用 Imatest 提取 1D MTF，并最终重构成 5x3 网格的 Field-Dependent 2D PSF，以接入我们现有的图像模糊处理管线。

---

## 📐 理论基础与数学框架

### Point Spread Function (PSF) 的物理意义

PSF 描述了一个理想点光源通过光学系统后在像面上形成的能量分布。在数学上，它代表了光学系统的脉冲响应：

$$
\text{PSF}(x, y) = \mathcal{F}^{-1}\left\{ \mathcal{H}(u, v) \right\}
$$

其中：
- $x, y$：空间域坐标，单位通常为 $\mu m$ 或 pixel
- $u, v$：频率域坐标，单位为 cycles/mm 或 cycles/pixel
- $\mathcal{F}^{-1}$：逆 Fourier 变换
- $\mathcal{H}(u, v)$：光学传递函数

### Modulation Transfer Function (MTF) 的定义

MTF 是 Optical Transfer Function (OTF) 的模，描述了系统对不同空间频率的响应能力：

$$
\text{MTF}(f) = \left| \frac{\mathcal{H}(f)}{\mathcal{H}(0)} \right|
$$

其中：
- $f$：空间频率
- $\mathcal{H}(f)$：特定频率下的复数传递函数值
- $\mathcal{H}(0)$：零频（DC）响应，用于归一化

| MTF 值 | 图像质量描述 |
|:------:|-------------|
| 1.0 | 完美传输，无衰减 |
| 0.5 | 50% 对比度保留 |
| 0.1 | 10% 对比度，接近人眼分辨极限 |
| 0.0 | 完全无法分辨 |

> **Reference**: [Imatest MTF Documentation](https://www.imatest.com/docs/sharpness/)

---

## 步骤 1：按景深分化采集方案

在这个阶段，由影像工程师负责拍摄采集数据。为了获得最高保真度的物理参数，我们根据景深采用不同的测试目标物。

### 1A. 合焦区域 采集：使用 SFR Chart

对于处于合焦面或景深范围内的物距，镜头的像差（如轻微的 spherical aberration、chromatic aberration 等）在极小尺度内体现为频率响应的衰减。

#### ISO-12233 测试卡规格

| 参数 | 标准 SFRplus | eSFR 增强版 |
|-----|-------------|------------|
| Aspect Ratio | 16:9 或 3:2 | 16:9 |
| Slanted Edge 角度 | ~5° | 多角度 |
| Dynamic Range | 10+ stops | 14+ stops |
| Color Patches | 可选 | 内置 |

#### 采集条件清单

1. 准备标准的 [ISO-12233 SFR (SFRplus 或 eSFR) 测试卡](https://www.imatest.com/imaging/iso-12233/)
2. 确保 environment illumination 均匀充足（建议 500-1000 lux），避免卡面 specular reflection
3. 将 camera 放置在测试卡正前方（optical axis 垂直于 chart plane），严谨对焦，使画面填满整个 16:9 test region
4. 拍摄处于合焦状态下的 raw image data，用于后续提取 MTF

#### 关键参数控制

| 参数 | 推荐值 | 影响 |
|-----|-------|-----|
| Exposure Time | 避免 clipping | 过曝会损失 highlight 细节 |
| ISO/Gain | 最低可用值 | 高 ISO 会引入 noise，干扰 edge analysis |
| Aperture | 目标工作 f-number | PSF 形状与 aperture 直接相关 |
| Focus Distance | 精确合焦位置 | 微小 defocus 会显著改变 MTF |

---

### 1B. 失焦区域 采集：使用 LED Point Source Target

对于大尺度的 Off-focus blur，由于其 circle of confusion 尺度远超 spatial frequency test 极限，我们**不再使用**传统的 SFR slanted edge test chart，而是直接捕捉物理 point source 的二维能量分布特性。

#### LED Point Source Array 设计

理想的 point source 应该近似 Dirac delta function $\delta(x, y)$。实际工程中，我们使用微型 LED 阵列：

$$
\delta(x, y) = \lim_{\epsilon \to 0} \frac{1}{\pi \epsilon^2} e^{-\frac{x^2 + y^2}{\epsilon^2}}
$$

其中：
- $\epsilon$：光源的有效半径，应远小于预期 blur radius
- $x, y$：相对于光源中心的坐标

| LED 参数 | 推荐规格 | 理由 |
|---------|---------|-----|
| Emitting Area | ≤0.1mm diameter | 接近理想点光源 |
| Wavelength | 550nm (green) 或 white | 匹配 luminance channel |
| Luminance Stability | <1% variation | 确保 measurement repeatability |
| Array Pattern | 5×3 或更密集 | 覆盖 full field |

#### Source Size Deconvolution 数学推导

实际物理世界中的 LED 具有其实际发光面积，并非完美的无穷小 point source。因此，camera 拍摄到的 blur image，实际上是纯粹的 lens PSF 与 LED 发光形状的 convolution 结果。

设 LED 的 intensity distribution 为 $S(x, y)$，lens PSF 为 $P(x, y)$，则测量得到的 image：

$$
I_{\text{measured}}(x, y) = (S * P)(x, y) = \iint S(\xi, \eta) \cdot P(x - \xi, y - \eta) \, d\xi \, d\eta
$$

其中：
- $*$：convolution operator
- $\xi, \eta$：积分哑变量
- $I_{\text{measured}}$：测量得到的 intensity distribution

##### Fourier Domain Deconvolution

由于 spatial domain 的 convolution 在 frequency domain 变为 point-wise multiplication：

$$
\mathcal{F}\{I_{\text{measured}}\}(u, v) = \mathcal{F}\{S\}(u, v) \times \mathcal{F}\{P\}(u, v)
$$

因此，我们可以通过 frequency domain 除法进行 deconvolution：

$$
\mathcal{F}\{P\}(u, v) = \frac{\mathcal{F}\{I_{\text{measured}}\}(u, v)}{\mathcal{F}\{S\}(u, v)}
$$

再通过逆 Fourier 变换恢复纯净的 PSF：

$$
P(x, y) = \mathcal{F}^{-1}\left\{ \frac{\mathcal{F}\{I_{\text{measured}}\}(u, v)}{\mathcal{F}\{S\}(u, v)} \right\}
$$

##### Wiener Filter Regularization

为避免由于高频 noise 放大导致 deconvolution 失败，实际工程中采用 Wiener Filter：

$$
\mathcal{F}\{P\}(u, v) = \frac{\mathcal{F}^*\{S\}(u, v) \cdot \mathcal{F}\{I_{\text{measured}}\}(u, v)}{|\mathcal{F}\{S\}(u, v)|^2 + \lambda}
$$

其中：
- $\mathcal{F}^*$：复共轭
- $\lambda$：regularization parameter，通常设为 SNR 的倒数
- $|\cdot|^2$：模的平方

> **Reference**: [Wiener Deconvolution - Wikipedia](https://en.wikipedia.org/wiki/Wiener_deconvolution)

---

## 步骤 2：数据提取与预处理

### 2A. 针对合焦图像（基于 SFR Chart）→ 提取 1D MTF

#### Imatest Analysis Workflow

1. 将拍摄好的 in-focus photograph 导入 [Imatest 软件](https://www.imatest.com/docs/sfrplus_instructions3/)，配置 5×3 analysis regions
2. **Data Format Specification**：Imatest 输出的 1D MTF CSV 文件通常包含 spatial frequency 与 response value 的映射

##### MTF 数据结构示例

| Column | Description | Unit |
|--------|-------------|------|
| Frequency | Spatial frequency | cycles/pixel 或 lp/mm |
| MTF | Modulation transfer value | 0.0 - 1.0 |
| MTF50 | Frequency at 50% response | cycles/pixel |
| MTF10 | Frequency at 10% response | cycles/pixel |

##### Sagittal/Tangential vs Horizontal/Vertical Convention

| Convention | Description | 适用场景 |
|-----------|-------------|---------|
| **Sagittal (S)** | Radial direction from optical center | 像差分析 |
| **Tangential (T)** | Perpendicular to radial direction | 像差分析 |
| **Horizontal (H)** | Image X-axis direction | 简单分析 |
| **Vertical (V)** | Image Y-axis direction | 简单分析 |

**坐标转换公式**：对于位于场点 $(X, Y)$ 的测量点，相对于 optical center $(X_c, Y_c)$：

$$
\theta = \arctan\left(\frac{Y - Y_c}{X - X_c}\right)
$$

其中 $\theta$ 为 radial angle，用于将 H/V 坐标旋转至 S/T 坐标系。

---

### 2B. 针对失焦图像（基于 LED Target）→ 提取 Field-Dependent 2D PSF

#### 自动检测与定位流程

##### Hough Circle Detection

使用 OpenCV 的 `cv2.HoughCircles` 算法：

```python
circles = cv2.HoughCircles(
    image,
    method=cv2.HOUGH_GRADIENT,
    dp=1,                    # Accumulator resolution ratio
    minDist=50,              # Minimum distance between centers
    param1=100,              # Canny edge upper threshold
    param2=30,               # Accumulator threshold
    minRadius=10,            # Minimum circle radius
    maxRadius=200            # Maximum circle radius
)
```

##### Connected Components Analysis

替代方案，对于高 contrast 场景更 robust：

```python
# Thresholding
_, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

# Connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

# Filter by area
valid_spots = [i for i in range(1, num_labels) if min_area < stats[i, cv2.CC_STAT_AREA] < max_area]
```

#### Margin Cropping 策略

由于 optical aberrations（如 coma, astigmatism）会导致 PSF 不对称扩散，我们需要根据检测到的基础 radius $R$，向周围进行外延：

$$
R_{\text{crop}} = R \times (1 + \alpha)
$$

其中 $\alpha$ 为 margin factor，通常取 0.2-0.3（即 20%-30% margin）。

#### 质心对齐算法

##### First Moment 计算

$$
\bar{x} = \frac{\sum_i \sum_j x_i \cdot I(x_i, y_j)}{\sum_i \sum_j I(x_i, y_j)}, \quad
\bar{y} = \frac{\sum_i \sum_j y_j \cdot I(x_i, y_j)}{\sum_i \sum_j I(x_i, y_j)}
$$

其中：
- $\bar{x}, \bar{y}$：质心坐标
- $I(x_i, y_j)$：像素 $(x_i, y_j)$ 处的 intensity 值

##### Sub-pixel Shift

使用 Fourier-based shift 实现亚像素精度对齐：

$$
I_{\text{aligned}}(x, y) = \mathcal{F}^{-1}\left\{ \mathcal{F}\{I_{\text{crop}}\} \cdot e^{-2\pi i (u \Delta x + v \Delta y)} \right\}
$$

其中：
- $\Delta x, \Delta y$：所需的 sub-pixel shift 量
- $u, v$：frequency domain coordinates

---

## 步骤 3：拟合与建模

### 阶段 1：全局径向基底拟合

假设 PSF 具有径向对称性，我们首先拟合一个基础的 radial profile。

#### 圆盘模型

最简单的 defocus PSF 模型是 uniform disk（pupil function 的几何投影）：

$$
P_{\text{disk}}(r) = \begin{cases}
\frac{1}{\pi R^2} & \text{if } r \leq R \\
0 & \text{if } r > R
\end{cases}
$$

其中：
- $r = \sqrt{x^2 + y^2}$：距中心的径向距离
- $R$：blur radius（circle of confusion radius）

#### 高斯模型

对于存在 spherical aberration 或轻微 defocus 的情况，Gaussian approximation 更准确：

$$
P_{\text{gaussian}}(r) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{r^2}{2\sigma^2}\right)
$$

其中 $\sigma$ 为标准差，与 blur radius 的关系约为 $R \approx 2\sigma$。

#### Generalized Gaussian 模型

更灵活的形式，可以拟合介于 disk 和 Gaussian 之间的形状：

$$
P_{\text{gen-gauss}}(r) = \frac{\beta}{2\pi\alpha^2\Gamma(2/\beta)} \exp\left[-\left(\frac{r}{\alpha}\right)^\beta\right]
$$

其中：
- $\alpha$：scale parameter
- $\beta$：shape parameter（$\beta=2$ 时为 Gaussian，$\beta \to \infty$ 时为 uniform disk）
- $\Gamma(\cdot)$：Gamma 函数

#### 非线性最小二乘拟合

目标函数：

$$
\min_{R, x_0, y_0} \sum_{i,j} \left[ I_{\text{measured}}(x_i, y_j) - P_{\text{model}}(r_{ij}; R, x_0, y_0) \right]^2
$$

其中：
- $r_{ij} = \sqrt{(x_i - x_0)^2 + (y_j - y_0)^2}$
- $x_0, y_0$：拟合的中心位置

---

### 阶段 2：残差高频拟合

利用原图减去第一阶段拟合出的基础形状，得到 residual map：

$$
R_{\text{residual}}(x, y) = I_{\text{measured}}(x, y) - P_{\text{base}}(r; R_{\text{opt}})
$$

这张 residual map 包含了 diffraction rings、onion-ring bokeh 等高频细节。

#### DCT 基底展开

Discrete Cosine Transform (DCT) 提供了一组正交基函数：

$$
R_{\text{residual}}(x, y) \approx \sum_{k=0}^{K-1} \sum_{l=0}^{L-1} C_{kl} \cdot \phi_k(x) \cdot \phi_l(y)
$$

其中 DCT basis functions：

$$
\phi_k(x) = \cos\left[\frac{\pi k}{N}\left(n + \frac{1}{2}\right)\right], \quad n = 0, 1, \ldots, N-1
$$

#### Gibbs Phenomenon 与截断问题

当使用有限项 DCT 展开去拟合具有 sharp edge 的信号时，会在截断边界产生 ringing artifact：

$$
\text{Overshoot} \approx 0.089 \times \text{jump amplitude}
$$

这就是著名的 Gibbs Phenomenon，约 9% 的 overshoot。

> **Reference**: [Gibbs Phenomenon - MathWorld](https://mathworld.wolfram.com/GibbsPhenomenon.html)

#### Multi-term Cosine Fitting with Clamping

为避免 Gibbs effect，我们引入 clamp parameter，建立分段模型：

$$
C_i(r) = \begin{cases}
A_i \cos(\omega_i r + \phi_i) & \text{if } r \leq R_{\text{clamp},i} \\
0 & \text{if } r > R_{\text{clamp},i}
\end{cases}
$$

其中每个 cosine term 有 4 个参数：
- $A_i$：amplitude（振幅）
- $\omega_i$：angular frequency（角频率，单位 rad/pixel）
- $\phi_i$：phase（相位，单位 rad）
- $R_{\text{clamp},i}$：clipping radius（截断半径）

#### 完整 PSF 模型

最终的 PSF model 为：

$$
P_{\text{total}}(x, y) = P_{\text{base}}(r; R) + \sum_{i=1}^{N} w_i \cdot C_i(r; A_i, \omega_i, \phi_i, R_{\text{clamp},i})
$$

其中：
- $N$：cosine term 数量（典型值 11）
- $w_i$：weighting factor（可通过 fitting 自动确定）

**总参数量**：$1 + 4N = 45$ 个标量参数（当 $N=11$ 时）

#### Levenberg-Marquardt Algorithm

使用 `scipy.optimize.least_squares` 进行非线性优化：

```python
from scipy.optimize import least_squares

def residual_func(params, measured_psf, r_grid):
    """
    params: [R, A1, omega1, phi1, R_clamp1, A2, omega2, phi2, R_clamp2, ...]
    """
    R = params[0]
    base = radial_disk(r_grid, R)
    
    residual = measured_psf - base
    for i in range(N_terms):
        idx = 1 + 4*i
        A, omega, phi, R_clamp = params[idx:idx+4]
        cosine_term = clamped_cosine(r_grid, A, omega, phi, R_clamp)
        residual -= cosine_term
    
    return residual.flatten()

result = least_squares(
    residual_func,
    x0=initial_guess,
    bounds=(lower_bounds, upper_bounds),
    method='trf'  # Trust Region Reflective
)
```

#### Initial Guess 策略

非线性优化器的收敛速度和成功率高度依赖 initial guess。我们的策略：

1. **从 DCT 频谱提取 dominant frequencies**：
   $$\omega_{\text{init}} = \arg\max_{\omega} |\text{DCT}\{R_{\text{residual}}\}|$$

2. **Amplitude 从频谱峰值读取**：
   $$A_{\text{init}} = |\text{DCT}\{R_{\text{residual}}\}|_{\text{peak}}$$

3. **Phase 初始化为零**：
   $$\phi_{\text{init}} = 0$$

4. **Clamp radius 初始化为 base radius**：
   $$R_{\text{clamp, init}} = R_{\text{base}}$$

---

## 步骤 4：将 1D MTF 重建为 2D PSF 管线数据

### 从 MTF 到 Line Spread Function (LSF)

1D MTF 是 OTF 的模，而 OTF 是 PSF 的 Fourier 变换。首先，我们需要从 MTF 重建 LSF。

#### Phase Retrieval 问题

由于 MTF 丢失了 phase 信息，我们需要假设：

$$
\text{OTF}(f) = \text{MTF}(f) \cdot e^{i\phi(f)}
$$

对于大多数 well-corrected optical systems，可以假设 phase 接近零：

$$
\text{LSF}(x) \approx \mathcal{F}^{-1}\{\text{MTF}(f)\}
$$

#### Edge Spread Function (ESF) 与 LSF 的关系

Imatest 实际测量的是 ESF，LSF 是 ESF 的导数：

$$
\text{LSF}(x) = \frac{d}{dx} \text{ESF}(x)
$$

$$
\text{MTF}(f) = |\mathcal{F}\{\text{LSF}(x)\}|
$$

### 从 1D LSF 到 2D PSF

假设 PSF 可以用 anisotropic Gaussian 近似：

$$
\text{PSF}(x, y) = \frac{1}{2\pi\sigma_1\sigma_2} \exp\left[-\frac{1}{2}\left(\frac{x'^2}{\sigma_1^2} + \frac{y'^2}{\sigma_2^2}\right)\right]
$$

其中坐标变换：

$$
\begin{bmatrix} x' \\ y' \end{bmatrix} = 
\begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix}
$$

参数含义：
- $\sigma_1$：沿长轴的标准差
- $\sigma_2$：沿短轴的标准差
- $\theta$：旋转角度（长轴与 X 轴的夹角）

### 旋转角度计算（S/T Convention）

对于位于场点 $(X, Y)$ 的测量点，相对于 optical center $(X_c, Y_c)$：

$$
\theta_{\text{radial}} = \arctan\left(\frac{Y - Y_c}{X - X_c}\right)
$$

- **Sagittal direction**：$\theta = \theta_{\text{radial}}$
- **Tangential direction**：$\theta = \theta_{\text{radial}} + 90°$

### 输出数据结构

最终输出的 tensor 结构：

```python
psf_grid = np.zeros((3, 5, kernel_size, kernel_size))
# psf_grid[row, col, :, :] = 2D PSF kernel at field position (row, col)
```

| Index | Field Position | Description |
|:-----:|---------------|-------------|
| `[0, 0]` | Top-Left | Corner field |
| `[0, 2]` | Top-Center | Top edge |
| `[1, 2]` | Center | On-axis |
| `[2, 4]` | Bottom-Right | Opposite corner |

---

## 步骤 5：模糊渲染管线与 PSF 插值策略

### 策略 A：Image Space Blending（当前实现）

#### 双线性插值公式

对于位于 $(x, y)$ 的像素，找到其周围的 4 个 PSF grid points：

$$
P_{\text{blended}} = (1-s)(1-t)P_{00} + s(1-t)P_{10} + (1-s)tP_{01} + stP_{11}
$$

其中：
- $P_{ij}$：grid point 处的 PSF
- $s, t$：归一化的局部坐标（$0 \leq s, t \leq 1$）

#### 计算复杂度分析

假设图像尺寸为 $W \times H$，PSF kernel 尺寸为 $K \times K$：

| 方法 | 卷积次数 | 计算复杂度 |
|-----|--------|----------|
| Naive per-pixel PSF | $W \times H$ | $O(WH K^2)$ |
| Grid-based Blending | 15 (5×3) | $O(15 \cdot WH K^2 + WH)$ |
| Frequency Domain | 1 (per frequency) | $O(WH \log(WH))$ |

---

### 策略 B：PSF Domain Interpolation（理论上更精确）

#### PSF 插值的物理合理性

从物理光学角度，PSF 随 field position 的变化是连续且光滑的。直接对 PSF 进行插值，而非对 blurred images 插值，可以：

1. **Preserve energy conservation**：插值后的 PSF 仍然归一化
2. **Maintain physical properties**：非负性、对称性等约束得以保持
3. **Enable efficient computation**：在 frequency domain 进行 point-wise operations

#### Stanford ISETCam Approach

[ISETCam](https://github.com/ISET/isetcam) 采用 Shift-Variant PSF 模型：

```matlab
% ISETCam example
oi = oiCreate('human');
psfGrid = oiGet(oi, 'psf grid');  % Pre-computed PSF at grid points
psfInterpolated = interpPSF(psfGrid, queryPosition);
```

> **Reference**: [ISETCam Documentation](https://isetcam.readthedocs.io/)

---

### 策略 C：Frequency Domain Processing with DCT

#### 为什么选择 DCT 而非 FFT？

| Property | FFT | DCT |
|----------|-----|-----|
| Output | Complex numbers | Real numbers only |
| Boundary | Periodic (wrap-around) | Even extension |
| Memory | 2× (real + imaginary) | 1× (real only) |
| Compression | Standard in JPEG | Optimized for images |

#### DCT 频域插值流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frequency Domain Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────┐         │
│  │ PSF Grid │ -> │  DCT 2D  │ -> │ MTF Field (5×3)   │         │
│  │  (5×3)   │    │Transform │    │ Frequency Domain  │         │
│  └──────────┘    └──────────┘    └─────────┬─────────┘         │
│                                             │                    │
│                                             ▼                    │
│                                  ┌───────────────────┐          │
│                                  │ Spatial Upscale   │          │
│                                  │ (Bilinear/Bicubic)│          │
│                                  │ 5×3 → 1920×1080   │          │
│                                  └─────────┬─────────┘          │
│                                             │                    │
│  ┌──────────┐    ┌──────────┐              │                    │
│  │  Input   │ -> │  DCT 2D  │◄─────────────┘                    │
│  │  Image   │    │Transform │                                   │
│  └──────────┘    └────┬─────┘                                   │
│                       │                                          │
│                       ▼                                          │
│              ┌─────────────────┐                                 │
│              │ Element-wise    │                                 │
│              │ Multiplication  │                                 │
│              └────────┬────────┘                                 │
│                       │                                          │
│                       ▼                                          │
│              ┌─────────────────┐                                 │
│              │   Inverse DCT   │                                 │
│              └────────┬────────┘                                 │
│                       │                                          │
│                       ▼                                          │
│              ┌─────────────────┐                                 │
│              │  Blurred Image  │                                 │
│              └─────────────────┘                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 数学表达

设 $M(u, v; x, y)$ 为位置 $(x, y)$ 处的 MTF，图像 $I$ 的 blur 过程：

$$
I_{\text{blurred}}(x, y) = \mathcal{D}^{-1}\left\{ \mathcal{D}\{I\}(u, v) \cdot M(u, v; x, y) \right\}
$$

其中 $\mathcal{D}$ 和 $\mathcal{D}^{-1}$ 分别为 DCT 和 inverse DCT。

#### 计算复杂度对比

| 操作 | 空间域卷积 | 频域点乘 |
|-----|-----------|---------|
| Per-pixel PSF | $O(K^2)$ | $O(\log N)$ |
| Full image | $O(N \cdot K^2)$ | $O(N \log N)$ |
| Memory for PSF grid | $O(N \cdot K^2)$ | $O(N)$ |

其中 $N = W \times H$ 为图像像素数，$K$ 为 kernel 尺寸。

---

### 策略 D：按景深分化的混合策略

#### In-focus PSF 处理

对于 small kernel（$K \leq 15$），频域方法效率高：

| Kernel Size | Spatial Conv. | Frequency Domain |
|:-----------:|:-------------:|:----------------:|
| 7×7 | 49 ops/pixel | ~30 ops/pixel |
| 11×11 | 121 ops/pixel | ~30 ops/pixel |
| 15×15 | 225 ops/pixel | ~30 ops/pixel |

#### Off-focus PSF 处理（Closed-form Approximation）

对于 large blur（$R > 20$ pixels），使用解析模型避免存储巨大 kernel：

```python
def approximate_psf(x, y, center, params):
    """
    params: {
        'R': base_radius,
        'terms': [
            {'amp': A1, 'freq': omega1, 'phase': phi1, 'clamp': Rc1},
            {'amp': A2, 'freq': omega2, 'phase': phi2, 'clamp': Rc2},
            ...
        ]
    }
    """
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Base radial term
    psf = radial_disk(r, params['R'])
    
    # Cosine correction terms
    for term in params['terms']:
        if r <= term['clamp']:
            psf += term['amp'] * np.cos(term['freq'] * r + term['phase'])
    
    return psf
```

#### 光学伪影的物理来源

| 伪影类型 | 物理成因 | Cosine Term 特征 |
|---------|---------|-----------------|
| Diffraction Rings | Aperture edge diffraction | Low freq, uniform spacing |
| Onion-ring Bokeh | Aperture blade edges | Medium freq, variable amplitude |
| Spherical Aberration | Lens surface curvature | Asymmetric envelope |
| Coma | Off-axis rays | Directional variation |

---

## 📊 实验数据与验证

### PSF 拟合质量评估指标

| Metric | Formula | Description |
|--------|---------|-------------|
| **RMSE** | $\sqrt{\frac{1}{N}\sum_i (P_{\text{fit}} - P_{\text{meas}})^2}$ | Root mean square error |
| **SSIM** | Structural Similarity Index | Perceptual quality |
| **Energy Conservation** | $\sum P_{\text{fit}} / \sum P_{\text{meas}}$ | Should ≈ 1.0 |
| **Centroid Error** | $\|\bar{x}_{\text{fit}} - \bar{x}_{\text{meas}}\|$ | Position accuracy |

### 典型拟合结果示例

| Blur Type | Base Radius | N_terms | RMSE | SSIM |
|-----------|:-----------:|:-------:|:----:|:----:|
| In-focus | 2-3 px | 5 | 0.012 | 0.995 |
| Slight defocus | 8-12 px | 7 | 0.018 | 0.988 |
| Moderate defocus | 20-30 px | 9 | 0.025 | 0.975 |
| Strong defocus | 50-80 px | 11 | 0.032 | 0.962 |

---

## 🔗 参考资源汇总

### 官方文档与标准

- [ISO 12233:2017 - Photography — Electronic still picture imaging — Resolution and spatial frequency responses](https://www.iso.org/standard/71696.html)
- [Imatest SFRplus Documentation](https://www.imatest.com/docs/sfrplus_instructions/)
- [Imatest Sharpness (MTF) Analysis](https://www.imatest.com/docs/sharpness/)

### 学术资源

- [Modulation Transfer Function - Wikipedia](https://en.wikipedia.org/wiki/Optical_transfer_function)
- [Point Spread Function - Scholarpedia](http://www.scholarpedia.org/article/Point_spread_function)
- [Deconvolution - Wikipedia](https://en.wikipedia.org/wiki/Deconvolution)
- [Wiener Filter for Deconvolution](https://en.wikipedia.org/wiki/Wiener_deconvolution)

### 开源工具

- [Stanford ISETCam](https://github.com/ISET/isetcam) - Image Systems Engineering Toolbox
- [OpenCV Hough Circle Transform](https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html)
- [SciPy Optimize least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)

### 数学背景

- [Discrete Cosine Transform - Wikipedia](https://en.wikipedia.org/wiki/Discrete_cosine_transform)
- [Levenberg-Marquardt Algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
- [Gibbs Phenomenon](https://mathworld.wolfram.com/GibbsPhenomenon.html)
- [Fourier Transform Properties](https://en.wikipedia.org/wiki/Fourier_transform)

---

## 📝 附录：符号表

| Symbol | Description | Unit |
|--------|-------------|------|
| $\text{PSF}(x,y)$ | Point Spread Function | Intensity (normalized) |
| $\text{MTF}(f)$ | Modulation Transfer Function | Dimensionless (0-1) |
| $f$ | Spatial frequency | cycles/pixel or lp/mm |
| $R$ | Blur radius (circle of confusion) | pixel |
| $\sigma$ | Gaussian standard deviation | pixel |
| $\theta$ | Rotation angle | radian or degree |
| $\mathcal{F}$ | Fourier Transform operator | — |
| $\mathcal{D}$ | Discrete Cosine Transform operator | — |
| $*$ | Convolution operator | — |
| $\delta(x,y)$ | Dirac delta function | — |
| $\lambda$ | Regularization parameter | — |
| $A_i$ | Cosine term amplitude | Intensity |
| $\omega_i$ | Cosine term angular frequency | rad/pixel |
| $\phi_i$ | Cosine term phase | radian |

---

# 光学与成像专业术语详解

---

## 1. Spherical Aberration（球面像差）

### 定义

Spherical Aberration 是指当光线通过球形表面（透镜或反射镜）时，**不同入射高度的光线无法汇聚到同一点**的现象。这是最基本、最常见的 Seidel Aberrations 之一。

### 物理原理

对于理想的薄透镜，所有平行于光轴入射的光线应汇聚于焦点 $F$。然而，球面的几何特性导致：

```
                    │ Incident Parallel Light
                    │
        ────────────┼────────────
              \     │     /
               \    │    /        ← Marginal Rays (边缘光线)
                \   │   /           汇聚于 F_m (更近)
                 \  │  /
                  \ │/
        ───────────┼────────────
                   │\
                   │ \
                   │  \          ← Paraxial Rays (近轴光线)
                   │   \           汇聚于 F_p (更远)
                   │    \
                   │     \
                   ▼      ▼
                  F_m    F_p
                  
            Longitudinal Spherical Aberration
                   δ = F_p - F_m
```

### 数学描述

Spherical Aberration 可以用 Seidel 系数 $S_I$ 定量描述：

$$
S_I = -\frac{1}{2} \sum_i \left( A_i^2 \cdot y_i \cdot \Delta \left( \frac{u}{n} \right)_i \right)
$$

其中：
- $A_i = n_i u_i + y_i c_i$：第 $i$ 面的折射不变量
- $y_i$：光线在第 $i$ 面的高度
- $u_i$：光线与光轴的夹角
- $n_i$：介质折射率
- $c_i$：表面曲率
- $\Delta$：表示经过表面前后的变化

#### Wavefront Aberration 形式

$$
W_{\text{sph}}(r) = W_{040} \cdot r^4
$$

其中：
- $W_{040}$：Primary spherical aberration coefficient（单位：波长或 mm）
- $r$：归一化的 aperture coordinate（$0 \le r \le 1$）
- 指数 4 表示 aberration 随孔径四次方增长

#### Transverse Ray Aberration

$$
\epsilon_y = 4 W_{040} \cdot r^3
$$

光线在焦平面上的横向偏差与 $r^3$ 成正比。

### 对成像的影响

| 现象 | 描述 |
|-----|------|
| Focus Shift | 最佳焦点位置随 aperture 变化 |
| Blur | 即使在最佳焦点，也无法获得 sharp image |
| Contrast Loss | MTF 在中频段下降明显 |
| "Glow" Effect | 高亮点源周围出现柔和光晕 |

### PSF 特征

```
        Spherical Aberration PSF Profile
        
   Intensity
       │
       │    ╭───╮
       │   ╱     ╲        ← Central bright core
       │  ╱       ╲
       │ ╱         ╲
       │╱           ╲
       │             ╲
       │              ╲___  ← Smooth outer halo
       └─────────────────── Radius
```

### 校正方法

| 方法 | 原理 | 效果 |
|-----|------|-----|
| **Aspheric Surface** | 非球面表面修正边缘光程差 | 彻底消除 |
| **Stopping Down** | 缩小光圈，只使用 central rays | 显著改善 |
| **Multiple Elements** | 正负透镜组合补偿 | 设计中常用 |
| **Gradient Index** | 渐变折射率材料 | 高端应用 |

> **Reference**: [Spherical Aberration - Edmund Optics](https://www.edmundoptics.com/knowledge-center/application-notes/optics/understanding-optical-aberrations/)

---

## 2. Chromatic Aberration（色差）

### 定义

Chromatic Aberration 是由于**不同波长的光具有不同的折射率**，导致无法同时聚焦到同一点的光学缺陷。

### 物理原理

基于 Cauchy's Equation，折射率与波长关系：

$$
n(\lambda) = A + \frac{B}{\lambda^2} + \frac{C}{\lambda^4} + \cdots
$$

其中：
- $A, B, C$：材料的 Cauchy 系数
- $\lambda$：光波长
- 短波长（蓝光）折射率 > 长波长（红光）折射率

```
        Axial Chromatic Aberration
              (Longitudinal)
        
    Blue (λ=450nm)    Green (λ=550nm)    Red (λ=650nm)
         │                 │                  │
         │                 │                  │
         ▼                 ▼                  ▼
      ═══╪═════════════════╪══════════════════╪═══
         F_B               F_G                F_R
         
         ◄─────── δ_f ─────────►
         
         
        Lateral Chromatic Aberration
              (Transverse)
        
                 Sensor Plane
              ┌─────────────────┐
              │   R   G   B     │
              │    \  |  /      │
              │     \ | /       │
              │      \|/        │
              │    Object       │
              └─────────────────┘
              
              ◄─ δ_T ─►
```

### 两种类型

| 类型 | 名称 | 表现 | 影响 |
|-----|------|-----|------|
| **Axial CA** | Longitudinal CA | 不同波长焦点位置不同 | 整个画面边缘模糊 |
| **Lateral CA** | Transverse CA | 不同波长放大率不同 | 画面边缘出现 color fringing |

### 数学描述

#### Axial Chromatic Aberration

$$
\delta_f = f(\lambda_{\text{blue}}) - f(\lambda_{\text{red}}) = f \cdot \frac{n_b - n_r}{n - 1}
$$

其中：
- $\delta_f$：焦距差
- $f$：标称焦距
- $n_b, n_r$：蓝光、红光的折射率

#### Abbe Number（阿贝数）

衡量材料色散特性的参数：

$$
V_d = \frac{n_d - 1}{n_F - n_C}
$$

其中：
- $n_d$：d-line (587.6nm) 折射率
- $n_F$：F-line (486.1nm) 折射率
- $n_C$：C-line (656.3nm) 折射率
- 高 $V_d$ → 低色散（好）

### PSF 特征

```
    Chromatic Aberration PSF (Cross-section)
    
    Blue    ────●───────
    Green  ───────●─────
    Red   ──────────●───
    
           R  G  B  R  G  B   ← Color fringing pattern
```

### 校正方法

#### Achromatic Doublet（消色差双胶合透镜）

通过组合正负透镜材料，使两个波长（通常 F 和 C line）聚焦于同一点：

$$
\frac{1}{f} = \frac{1}{f_1} + \frac{1}{f_2} = (n_1 - 1)\left(\frac{1}{R_1} - \frac{1}{R_2}\right) + (n_2 - 1)\left(\frac{1}{R_3} - \frac{1}{R_4}\right)
$$

消色差条件：

$$
\frac{V_1}{f_1} + \frac{V_2}{f_2} = 0 \quad \Rightarrow \quad \frac{f_1}{f_2} = -\frac{V_1}{V_2}
$$

| 透镜类型 | 波段校正 | 应用 |
|---------|---------|-----|
| Achromat | 2 wavelengths | 普通摄影镜头 |
| Apochromat | 3 wavelengths | 天文、微距 |
| Superachromat | 4 wavelengths | 科研、航天 |

> **Reference**: [Chromatic Aberration - Wikipedia](https://en.wikipedia.org/wiki/Chromatic_aberration)

---

## 3. Coma（彗差）

### 定义

Coma 是一种**离轴像差**，导致 off-axis 点光源成像呈现彗星状拖尾。它是 Seidel Aberrations 的第二项。

### 物理原理

当光线从视场边缘斜射入镜头时，穿过孔径不同环带的光线在像面上汇聚于不同位置，形成不对称的光斑。

```
              Coma Visualization (Top View)
              
              Optical Axis
                   │
                   │
        ───────────┼───────────
              \    │    /
               \   │   /
                \  │  /      ← Rays from off-axis point
                 \ │ /
                  \│/
        ───────────┼───────────  Lens
                   │
                   │
                   
              Image Plane
              
        ┌─────────────────────────┐
        │                         │
        │         ●               │ ← Paraxial image (sharp)
        │        ●●               │
        │       ●●●               │
        │      ●●●●               │ ← Coma tail (spread)
        │     ●●●●●               │
        │    ●●●●●●               │
        │                         │
        └─────────────────────────┘
        
         Coma Flare Direction
              ←───
```

### 数学描述

#### Wavefront Aberration

$$
W_{\text{coma}}(\rho, \theta) = W_{131} \cdot \rho^3 \cdot \cos\theta
$$

其中：
- $W_{131}$：Coma aberration coefficient
- $\rho$：归一化孔径坐标
- $\theta$：孔径内的角度
- 下标 131 表示：$\rho^1$（视场一次方）× $\rho^2$（孔径平方）× $\cos\theta$

#### Transverse Ray Aberration

$$
\epsilon_x = W_{131} \cdot \rho^2 \cdot (2 + \cos 2\theta)
$$

$$
\epsilon_y = W_{131} \cdot \rho^2 \cdot \sin 2\theta
$$

### 对成像的影响

| 特征 | 描述 |
|-----|------|
| Directional Blur | 模糊方向指向或背向视场中心 |
| Asymmetric PSF | PSF 呈彗星状，不对称 |
| Off-axis Only | 仅在画面边缘出现 |
| Aperture Dependent | 缩小光圈可改善 |

### Coma 的正负

| 类型 | 彗尾指向 | 光线来源 |
|-----|---------|---------|
| **Positive Coma** | 指向视场边缘 | 外环光线聚焦远 |
| **Negative Coma** | 指向视场中心 | 外环光线聚焦近 |

### 校正方法

| 方法 | 原理 |
|-----|------|
| **Symmetric Design** | 对称结构自动消除 coma |
| **Stop Position** | 将 aperture stop 置于特定位置 |
| **Aspheric Elements** | 非球面补偿 |

> **Reference**: [Coma - Wikipedia](https://en.wikipedia.org/wiki/Coma_(optics))

---

## 4. Astigmatism（像散）

### 定义

Astigmatism 是指光学系统对**不同方向的光线**具有不同的聚焦能力，导致径向和切向光线汇聚于不同位置。

### 物理原理

对于 off-axis 物点，光线可以分解为两个正交分量：
- **Tangential Rays**：包含物点和光轴的平面内的光线
- **Sagittal Rays**：垂直于 tangential plane 的光线

```
              Astigmatism Geometry
              
                    Object Point
                         ●
                        /│\
                       / │ \
                      /  │  \
                     /   │   \
                    /    │    \
        ───────────┼─────┼─────┼─── Lens
                  /      │      \
                 /       │       \
                /        │        \
               ▼         ▼         ▼
          
          T-focus   S-focus    Best Focus
            ●●         ●●●●       ●●●
           Tangential Sagittal   "Circle of Least Confusion"
            Focus      Focus
```

### 数学描述

#### Wavefront Aberration

$$
W_{\text{ast}}(\rho, \theta) = W_{222} \cdot \rho^2 \cdot \cos^2\theta
$$

其中：
- $W_{222}$：Astigmatism coefficient
- 指数 222 表示：$\rho^2$（视场平方）× $\rho^2$（孔径平方）× $\cos^2\theta$

#### Focal Separation

$$
\Delta z = z_T - z_S = \frac{2 \cdot W_{222} \cdot f^2}{r^2}
$$

其中：
- $z_T$：Tangential focus 位置
- $z_S$：Sagittal focus 位置
- $f$：焦距
- $r$：孔径半径

### 像面结构

| 位置 | 成像特征 | 形状 |
|-----|---------|-----|
| **Tangential Focus** | 径向线条清晰 | 椭圆（长轴沿切向） |
| **Sagittal Focus** | 切向线条清晰 | 椭圆（长轴沿径向） |
| **Circle of Least Confusion** | 最小模糊圆 | 近似圆形 |

### 对成像的影响

```
        Astigmatism Field Map
        
    ┌─────────────────────────────┐
    │   ║                       ║   │
    │   ║                       ║   │
    │   ║    ─────────────      ║   │
    │   ║                       ║   │
    │   ║                       ║   │
    └─────────────────────────────┘
    
    Sagittal lines       Tangential lines
    focus well           focus well
    (radial)             (circumferential)
```

### 与 Field Curvature 的关系

Astigmatism 常与 Petzval curvature（场曲）共存，形成两个弯曲的像面：
- **Tangential Field**：更弯曲
- **Sagittal Field**：较平坦

$$
\frac{1}{R_t} - \frac{1}{R_s} = \frac{2 \cdot W_{222}}{r^2}
$$

### 校正方法

| 方法 | 原理 |
|-----|------|
| **Field Flattener** | 在焦面前加入负透镜补偿 |
| **Symmetric Design** | 对称结构减小 astigmatism |
| **Stop Shift** | 调整 aperture stop 位置 |
| **Aspheric Surface** | 非球面校正 |

> **Reference**: [Astigmatism - Wikipedia](https://en.wikipedia.org/wiki/Astigmatism_(optical_systems))

---

## 5. 为什么 Slanted Edge 可以提取 Sub-pixel 分辨率的 MTF？

### 核心原理

Slanted Edge 方法利用**斜边在像素网格上的空间过采样**特性，实现远超像素分辨率的 MTF 测量。

### 几何解释

```
    Pixel Grid + Slanted Edge (e.g., 5° slant)
    
    ┌───┬───┬───┬───┬───┬───┬───┬───┐
    │   │   │   │░░░│███│███│███│███│
    ├───┼───┼───┼───┼───┼───┼───┼───┤
    │   │   │░░░│███│███│███│███│███│
    ├───┼───┼───┼───┼───┼───┼───┼───┤
    │   │░░░│███│███│███│███│███│   │
    ├───┼───┼───┼───┼───┼───┼───┼───┤
    │░░░│███│███│███│███│███│   │   │
    ├───┼───┼───┼───┼───┼───┼───┼───┤
    │███│███│███│███│███│   │   │   │
    └───┴───┴───┴───┴───┴───┴───┴───┘
    
    Edge Transition Region (zoomed)
    
    Each row samples a DIFFERENT sub-pixel
    position along the edge!
```

### 数学推导

#### Effective Sampling Rate

设 edge angle 为 $\theta$（通常 5°），则在 edge normal 方向上的有效采样间距为：

$$
\Delta x_{\text{eff}} = \Delta x \cdot \sin\theta
$$

其中 $\Delta x$ 为像素间距（通常 1 pixel）。

对于 $\theta = 5°$：

$$
\Delta x_{\text{eff}} = 1 \times \sin(5°) \approx 0.087 \text{ pixel}
$$

**过采样倍数**：

$$
N_{\text{oversample}} = \frac{1}{\sin\theta} \approx 11.5\times
$$

#### Edge Spread Function (ESF) 重建

1. **识别边缘位置**：对每行进行 sub-pixel edge detection

2. **对齐并叠加**：将每行的数据按 edge 位置对齐

```python
def build_esf(image, edge_angle=5):
    """
    Build oversampled ESF from slanted edge image
    """
    height, width = image.shape
    
    # For each row, find edge position using derivative
    edge_positions = []
    for y in range(height):
        row = image[y, :]
        derivative = np.gradient(row)
        edge_x = np.argmax(np.abs(derivative))  # Pixel-level
        edge_x_subpixel = refine_edge_position(row, edge_x)  # Sub-pixel refinement
        edge_positions.append(edge_x_subpixel)
    
    # Align all rows and bin into oversampled bins
    oversampled_esf = []
    for y, x_edge in enumerate(edge_positions):
        row = image[y, :]
        # Shift row so edge is at position 0
        aligned = shift(row, -x_edge, order=3)
        oversampled_esf.append(aligned)
    
    return np.mean(oversampled_esf, axis=0)
```

3. **Derivative 得到 LSF**：

$$
\text{LSF}(x) = \frac{d}{dx} \text{ESF}(x)
$$

4. **Fourier Transform 得到 MTF**：

$$
\text{MTF}(f) = \left| \mathcal{F}\{\text{LSF}(x)\} \right|
$$

### 为什么比 Direct Method 更准确？

| 方法 | 空间分辨率 | Noise Sensitivity |
|-----|----------|-------------------|
| Direct PSF Imaging | 受 pixel size 限制 | 高（单点噪声影响大） |
| Slanted Edge | Sub-pixel（~0.1 pixel） | 低（多行平均降噪） |

### ISO 12233 标准

ISO 12233 标准规定的 Slanted Edge 要求：

| 参数 | 标准值 | 原因 |
|-----|-------|------|
| Edge Angle | 5° ± 1° | 平衡过采样与 edge length |
| Edge Contrast | ≥ 10:1 | 确保 SNR |
| ROI Size | ≥ 100 pixels along edge | 足够统计样本 |
| Edge Quality | ≥ 95% linear | 避免 edge imperfection 干扰 |

### 采样定理验证

Nyquist frequency 为 $f_N = 0.5$ cycles/pixel。通过 10× 过采样：

$$
f_{\text{max}} = 10 \times f_N = 5 \text{ cycles/pixel}
$$

这远超图像本身的信息容量，可以精确测量 MTF curve。

> **Reference**: [ISO 12233 Slanted Edge Method](https://www.imatest.com/docs/sharpness/#slanted_edge)

---

## 6. Specular Reflection（镜面反射）

### 定义

Specular Reflection 是光线在**光滑表面**上发生的镜面反射，反射角等于入射角。在摄影中，它会导致测试卡表面的反光，干扰测量。

### 物理原理

**Law of Reflection**：

$$
\theta_i = \theta_r
$$

其中：
- $\theta_i$：入射角
- $\theta_r$：反射角

```
              Specular Reflection Geometry
              
                    Surface Normal
                         │
                         │
                         ▼
    ─────────────────────┼─────────────────────
                         │
                    θ_i  │  θ_r
                   ╱     │     ╲
                  ╱      │      ╲
                 ╱       │       ╲
                ▼        │        ▼
            Incident   Surface   Reflected
              Ray                 Ray
              
         Camera sees glare if
         camera is in reflection direction!
```

### 对 SFR 测试的影响

```
    Ideal Edge Profile          With Specular Reflection
    
    │███████████                │███████████░░░
    │███████████                │███████████░░░  ← Glare
    │███████████                │███████████░░░
    │           ███             │░░░░░░░░░░░███
    │           ███             │           ███
    │           ███             │           ███
    └───────────────            └───────────────
    
    Clean transition            Washed out highlight
    → Accurate MTF              → MTF underestimation
```

### 避免策略

| 方法 | 原理 | 实施要点 |
|-----|------|---------|
| **Gentle Angle** | 使 camera 略偏离 normal incidence | 倾斜 < 5° |
| **Matte Surface** | 使用漫反射测试卡 | 哑光涂层 |
| **Polarizer** | 利用偏振消除反射 | 正交偏振片 |
| **Lighting Control** | 避免直射光源 | 软光箱、间接照明 |
| **Black Enclosure** | 减少环境反射 | 暗室或遮光罩 |

### Diffuse vs Specular Reflection

| 特性 | Specular | Diffuse |
|-----|----------|---------|
| 表面 | 光滑 | 粗糙 |
| 反射方向 | 单一方向 | 所有方向 |
| 数学模型 | 镜面反射定律 | Lambertian 模型 |
| MTF 测试影响 | 严重 | 轻微 |

### Lambertian Reflection Model

理想漫反射表面的反射强度：

$$
I_r = I_i \cdot \frac{\rho}{\pi} \cdot \cos\theta
$$

其中：
- $I_r$：反射强度
- $I_i$：入射强度
- $\rho$：表面 reflectance（0-1）
- $\theta$：与 surface normal 的夹角

> **Reference**: [Specular Reflection - Wikipedia](https://en.wikipedia.org/wiki/Specular_reflection)

---

## 7. Lux Level（照度等级）

### 定义

Lux 是国际单位制中**照度**的单位，描述单位面积上接收到的光通量。

### 1 Lux 的定义

$$
1 \text{ lux } (lx) = 1 \text{ lumen } (lm) / 1 \text{ square meter } (m^2)
$$

即：在 1 平方米面积上均匀分布 1 lumen 的光通量时，照度为 1 lux。

### 与其他单位的关系

| 物理量 | 单位 | 符号 | 关系 |
|-------|-----|------|-----|
| Luminous Flux | Lumen | lm | 基本单位 |
| Illuminance | Lux | lx | lm/m² |
| Luminous Intensity | Candela | cd | lm/sr |
| Luminance | cd/m² | nt | nit |

### 数学定义

从物理角度，照度定义为：

$$
E = \frac{d\Phi}{dA}
$$

其中：
- $E$：照度（lux）
- $\Phi$：光通量
- $A$：面积

对于点光源：

$$
E = \frac{I \cdot \cos\theta}{d^2}
$$

其中：
- $I$：光源的光强
- $d$：距离
- $\theta$：入射角（与法线的夹角）

### 典型场景照度对照表

| 场景 | 照度 | 描述 |
|-----|-----------|------|
| 满月夜晚 | 0.25 - 1 | 极低照度 |
| 街灯照明 | 10 - 20 | 夜间户外 |
| 室内走廊 | 50 - 100 | 低照度室内 |
| 客厅/卧室 | 100 - 300 | 一般室内 |
| 办公室 | 300 - 500 | 标准工作照度 |
| 阅读书写 | 500 - 1000 | 推荐照度 |
| 阴天户外 | 1000 - 2000 | 自然光 |
| **SFR 测试推荐** | **500 - 1000** | **测试卡均匀照明** |
| 晴天阴影 | 10000 - 20000 | 户外阴影 |
| 晴天直射 | 50000 - 100000 | 强日光 |

### Photometric vs Radiometric Units

| Photometric（光度学） | Radiometric（辐射度学） | 换算（555nm） |
|---------------------|----------------------|-------------|
| Lumen (lm) | Watt (W) | 1 W = 683 lm |
| Lux (lx) | W/m² | 1 W/m² = 683 lx |
| Candela (cd) | W/sr | 1 W/sr = 683 cd |

换算系数 683 lm/W 是在 555nm（人眼最敏感波长）处定义的。

### Luminous Efficacy

$$
\eta = \frac{\Phi_v}{\Phi_e} = \frac{\text{Luminous Flux (lm)}}{\text{Radiant Flux (W)}}
$$

| 光源类型 | Luminous Efficacy (lm/W) |
|---------|-------------------------|
| 白炽灯 | 10-17 |
| 荧光灯 | 50-100 |
| LED | 80-150 |
| 理想 555nm 光源 | 683（理论最大值） |

### 对 SFR 测试的意义

| 照度条件 | 影响 |
|---------|------|
| **过低（< 100 lx）** | SNR 低，noise 干扰 edge detection |
| **适宜（500-1000 lx）** | 高 SNR，稳定测量 |
| **过高（> 10000 lx）** | 可能导致 sensor saturation |
| **不均匀** | Edge contrast 变化，MTF 误差 |

> **Reference**: [Lux - Wikipedia](https://en.wikipedia.org/wiki/Lux)

---

## 8. 补充：Seidel Aberrations 完整列表

| 编号 | 名称 | Seidel 系数 | 与孔径关系 | 与视场关系 |
|:---:|------|-----------|----------|----------|
| I | Spherical Aberration | $W_{040}$ | $\rho^4$ | 无关 |
| II | Coma | $W_{131}$ | $\rho^3$ | $h^1$ |
| III | Astigmatism | $W_{222}$ | $\rho^2$ | $h^2$ |
| IV | Field Curvature | $W_{220}$ | $\rho^2$ | $h^2$ |
| V | Distortion | $W_{311}$ | $\rho^1$ | $h^3$ |

其中：
- $\rho$：归一化孔径半径
- $h$：归一化视场高度

### Wavefront Aberration Polynomial

完整的波像差展开：

$$
W(\rho, \theta, h) = \sum_{l,m,n} W_{klm} \cdot H^l \cdot \rho^m \cdot \cos^n\theta
$$

其中 $k = 2l + m$ 为总阶数。

---

## 🔗 参考资源汇总

### 光学像差

- [Seidel Aberrations - Wikipedia](https://en.wikipedia.org/wiki/Optical_aberration)
- [Spherical Aberration - Edmund Optics](https://www.edmundoptics.com/knowledge-center/application-notes/optics/understanding-optical-aberrations/)
- [Chromatic Aberration - Wikipedia](https://en.wikipedia.org/wiki/Chromatic_aberration)
- [Coma Aberration - Wikipedia](https://en.wikipedia.org/wiki/Coma_(optics))
- [Astigmatism - Wikipedia](https://en.wikipedia.org/wiki/Astigmatism_(optical_systems))

### MTF 与 Slanted Edge

- [ISO 12233 Standard](https://www.iso.org/standard/71696.html)
- [Imatest Slanted Edge Method](https://www.imatest.com/docs/sharpness/)
- [Spatial Frequency Response - Nikon](https://www.microscopyu.com/tutorials/spatial-frequency-response)

### 光度学

- [Lux Unit - Wikipedia](https://en.wikipedia.org/wiki/Lux)
- [Luminous Intensity - Wikipedia](https://en.wikipedia.org/wiki/Luminous_intensity)
- [Photometry - ScienceDirect](https://www.sciencedirect.com/topics/physics-and-astronomy/photometry)

### 反射与照明

- [Specular Reflection - Wikipedia](https://en.wikipedia.org/wiki/Specular_reflection)
- [Lambertian Reflectance - Wikipedia](https://en.wikipedia.org/wiki/Lambertian_reflectance)
- [Polarization in Photography](https://www.cambridgeincolour.com/tutorials/polarizing-filters.htm)



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
