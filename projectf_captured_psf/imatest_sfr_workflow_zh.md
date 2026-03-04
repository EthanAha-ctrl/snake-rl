# 物理镜头 PSF 采集与重构工作流指南

本文档旨在指导如何从真实镜头采集 SFR Chart 数据，使用 Imatest 提取 1D MTF，并最终重构成 5x3 网格的视场相关 (Field-Dependent) 2D PSF，以接入我们现有的图像模糊处理管线。

## 步骤 1：按景深分化采集方案 (In-focus vs. Off-focus)
在这个阶段，由影像工程师负责拍摄采集数据。为了获得最高保真度的物理参数，我们根据景深采用不同的测试目标物：

### 1A. 合焦区域 (In-focus) 采集：使用 SFR Chart
对于处于合焦面或景深范围内的物距，镜头的像差（如轻微的球面像差、色差等）在极小尺度内体现为频率响应的衰减。
1. 准备标准的 [ISO-12233 SFR (SFRplus 或 eSFR) 测试卡](https://www.imatest.com/imaging/iso-12233/)。
2. 确保环境照明均匀充足，避免卡面反光。
3. 将相机放置在测试卡正前方（光轴垂直），严谨对焦，使画面填满整个 16:9 测试区域。
4. 拍摄处于合焦状态下的原始图像数据，用于后续提取 MTF。

### 1B. 失焦区域 (Off-focus) 采集：使用 LED 点光源靶机
对于大尺度的失焦光斑（Off-focus PSF），由于其弥散圆尺度远超空间频率测试极限，我们**不再使用**传统的 SFR 刀锋边缘测试卡，而是直接捕捉物理点光源的二维能量分布特性。
1. 使用分布式的微型 LED 点光源阵列（可放置在远处，或通过 Relay Lens 中继透镜模拟较远物距）。
2. 在不同的离焦程度下拍摄这些点光源。
3. **消除光源尺寸本底 (Source Size Deconvolution / Extrapolation)**:
   实际物理世界中的 LED 具有其实际发光面积，并非完美的无穷小点光源（Dirac delta function $\delta(x, y)$）。因此，相机拍摄到的光斑图像，实际上是纯粹的镜头 PSF 与 LED 发光形状的卷积结果：
   - 测试所得的合焦光斑: $Image_{\text{in}} = LED\_Shape * PSF_{\text{lens\_in}}$
   - 测试所得的失焦光斑: $Image_{\text{out}} = LED\_Shape * PSF_{\text{lens\_out}}$
   
   为了获取极其纯净的透镜离焦函数 $PSF_{\text{lens\_out}}$，我们在数学上必须消除这个发光体本体的基础体积（这就好比用理想气体状态方程 $PV=nRT$ 来外推推导绝对零度）。
   通过将数学图像转换至频率域（如傅里叶变换 FFT 或离散余弦变换 DCT），由于时域或空域的卷积在频域变为了点乘：
   $$ \mathcal{F}(Image_{\text{out}}) = \mathcal{F}(LED\_Shape) \times \mathcal{F}(PSF_{\text{lens\_out}}) $$
   只要我们在频域中执行除法操作（反卷 / Deconvolution）：
   $$ \mathcal{F}(PSF_{\text{lens\_out}}) = \frac{\mathcal{F}(Image_{\text{out}})}{\mathcal{F}(LED\_Shape)} $$
   即可在数学空间中完美剥夺 LED 固有的尺寸投射。为避免由于高频噪声放大导致解反卷积失败，实际工程拟合算法上常考虑结合**维纳滤波 (Wiener Filter)** 或类似盲反卷积规则化的手段来稳健求解此纯净的离散内核，进而为下一环节的闭式解 (Closed-form formulas) 提取标量参数服务。

## 步骤 2：数据提取与预处理 (分化处理)

### 2A. 针对合焦图像 (基于 SFR Chart) -> 提取 1D MTF
1. 将拍摄好的合焦照片导入 [Imatest 软件](https://www.imatest.com/docs/sfrplus_instructions3/)，配置 5x3 分析区域。
2. **数据格式规范**: Imatest 输出的 1D MTF CSV 文件通常包含空间频率与响应值的映射。我们需要进一步检视其输出格式是 **水平/垂直 (H/V)** 还是 **弧矢/子午 (S/T)** 辐射状坐标。
   - 若 Imatest 官方未提供公开明确的 S/T 格式文档，我们在 `1_parse_imatest_to_psf_grid.py` 脚本中将自定义并约束一种标准化的输入格式字典，强迫数据对其进行标准化绑定。

### 2B. 针对失焦图像 (基于 LED 靶机) -> 提取 Field-Dependent 2D PSF
从布满 LED 点光源的一整张失焦照片中，自动且精准地抠出分布在不同视场 (Field) 的光斑极其关键：
1. **自动检测与定位 (OpenCV Circle Detection)**：
   使用 `cv2.HoughCircles` 或通过阈值化后的连通域分析（Connected Components）来快速定位全图所有高亮虚化 LED 光斑的中心。
2. **带裕量裁剪 (Margin Cropping)**：
   由于像差会导致光斑不对称扩散，我们需要根据检测到的光斑基础半径 $R$，向周围进行外延（例如 `+ 20% margin`）并执行 Crop 操作，防止截断彗差的尾巴。
3. **质心对齐 (Center Alignment)**：
   将所有的 Crop Block 进行质心计算和 Sub-pixel 平移，使光斑真正的能量中心严格对齐到矩阵正中央，作为干净的 `captured PSF` 供后续进行拟合。

## 步骤 3：拟合与建模 (Fitting Framework)

对于提取到的 Off-focus 巨型光斑，我们使用两阶段拟合 (Two-Stage Fitting) 来求解其闭式解 (Closed-form approximation) 参数：

### 阶段 1：全局径向基底拟合 (Base Radial Fitting)
先假设光斑是一个完美的几何模型（例如圆盘或高斯衰减盘）。运行基础的非线性最小二乘法，找出最优的 `center`（微调） 和全局 `radius`，使得 `radial(center, radius)` 与原图均方差最小。

### 阶段 2：残差高频拟合 (Residual Multi-term Cosine Fitting)
利用原图减去第一阶段拟合出来的纯净圆，得到一张充满波动细节的**残差图 (Residual Map)**。这张图上包含了透镜衍射环带来的“洋葱圈”高斯纹理。
- **纯 DCT 的局限性与截断参数 (Clamp)** 
  我们模型中引入了硬截断参数 `radius_clamp`。DCT 是一种全局线性基底展开，直接提取前 11 项去拟合带有截断骤降的信号时，会在截断边缘产生强烈的吉布斯效应（振铃现象，Gibbs Phenomenon），导致背景产生虚假的波动噪声。
- **基于梯度的非线性优化求解 (Non-linear Optimization Solver)**：
  因为模型包含非连续分支语句 (`if radius > clamp`)，这是一个多参数非线性优化问题。我们需要使用求解器（如 `scipy.optimize.least_squares`，底层使用 Levenberg-Marquardt 或 TRF 算法），联合优化这 11 个组建（共 44 个标量参数：`amp`, `phase`, `freq`, `clamp`）。
- **至关重要的初始猜测 (Initial Guess)**：
  非线性求解器极度依赖优秀的初始猜测（`x0`），否则极易陷入局部最优解或者陷入死循环。**我们的策略是：利用 DCT 进行降维计算，为其提供高质量的 Initial Guess。**
  在运行非线性求解器前，我们先对残差图强制应用一个空间 Window（例如用第一阶段求出的 base radius 作为一个软遮罩），然后对其内部做一次快速的 DCT 展开。我们直接读取 DCT 频谱矩阵中能量最强的几个峰值，换算为初始的 `freq` 和 `amp`；将初始 `clamp` 设定为 base radius；将 `phase` 全部置 0。将这组强相关的解析参数喂给 optimizer 作为起点启动迭代，既保证了收敛速度，又从代数源头上避免了跑飞 (divergence)。

## 步骤 3：将 1D MTF 重建为 2D PSF 管线数据 (即将编写的工具)
*接下来，我们将开发一个新的 Python 脚本（例如预期的 `1_parse_imatest_to_psf_grid.py`）来自动解析这些 CSV 数据。*

**兼容性设计**：由于我们尚不确定 Imatest 导出的正交 MTF 数据是基于**水平/垂直 (Horizontal/Vertical)** 坐标系，还是基于**弧矢/子午 (Sagittal/Tangential)** 辐射坐标系，该脚本将支持这两种解析模式 (Convention):

数据处理的底层逻辑如下：
1. 读取各点的对向 MTF 数据，使用逆傅里叶变换将其转换为一维空间上的线扩散函数 (LSF)。
2. 通过拟合，提取两轴对应的扩散宽度系数（$\sigma_1$ 和 $\sigma_2$）。
3. 利用提取出的长短轴参数，构建出表现出像散/彗差特性的 2D 各向异性高斯矩阵（2D Anisotropic Gaussian Kernel）。
4. **处理旋转偏移 (根据 Convention 选择)**：
   - **如果使用 Horizontal / Vertical (H/V) 模式**：导出的 MTF 已经是基于图像绝对 XY 坐标系的。此时**不需要**计算辐射旋转角（或者相当于 $\theta=0$），直接将生成的 2D 矩阵按水平/垂直方向排列即可。
   - **如果使用 Sagittal / Tangential (S/T) 模式**：导出的 MTF 是基于画面光心辐射的。需要根据测试点坐标 $(X, Y)$ 相对于画面光心的方向，计算出旋转角度 $\theta$。将上述 2D 矩阵进行对应角度的旋转，使得 Sagittal 方向严格沿着光心放射状向外。
5. 将 15 个点生成的 2D 核打包输出为 `[3, 5, kernel_size, kernel_size]`  的 Tensor 对象。

## 步骤 4：模糊渲染管线与高级 PSF 插值策略探讨

这部分涉及如何将这 5x3 个基础 PSF 平滑地应用到整张高分辨率图像上。

### 策略 A：图像空间融合 (Spatial Blending) - 当前实现
目前 `sfr_processor.py` 采用的方法是**对卷积后的图像进行双线性插值**。
这种方法的计算逻辑是：用网格中的每一个 PSF对整张原图做一次 2D 卷积，然后根据像素所在区域进行加权平均。这种方法在实现上非常简单且容易并行计算，但它的假设是基于“在较小区域内，像差是不变的”。

### 策略 B：PSF 域插值与基于 ISETCam 的思考 (理论上更精确)
实际上，**对 PSF 本身进行空间插值在物理光学上是更为严谨的。**
著名的光学模拟开源工具 [Stanford ISETCam](https://github.com/ISET/isetcam) (Image Systems Engineering Toolbox) 在处理光学系统模拟（Optics Module）时，核心思维也是计算 Shift-Variant (空间可变) 的 PSF。它们通常也会基于采样的点阵构建一个多维的 PSF/MTF 映射，并对中间位置的 PSF/MTF 进行插值。

#### 基于 DCT 的频域插值与逐像素点乘 (未来演进方向)
直接在空间域对巨大的 PSF tensor 甚至全图每个像素生成对应的独立 PSF 进行卷积是极度耗时的（存储开销达 `image_w * image_h * kernel_w * kernel_h`）。

一个更为高效的重构和渲染策略是**频域处理 (Frequency Domain Processing)**：
1. **使用 DCT (离散余弦变换)** 代替 FFT：将基于 5x3 采样点构建的 PSF 转换为 MTF（频率响应）。DCT 的优势在于其结果完全为**实数域**（Real Numbers），没有虚数部分，极大节省内存带宽并消除复数运算的额外开销。
2. **频域插值放大 (Spatial Upscaling)**：由于频域表示（MTF / DCT 结果）特征往往更加平缓，我们可以将这个仅有 5x3 个采样位置的 MTF 网格，通过空间插值方法放大至与输入图像相同的分辨率 (e.g., $1920 \times 1080$)。此时系统将拥有一个与图像像素完全对应的“逐像素频率响应图”。
3. **逐元素相乘 (Element-wise Multiplication)**：然后，我们将目标图像也转换至 DCT 频域，与上一步得到的逐像素 MTF 进行 Numpy 数组层面的点乘（Point-to-point array multiplication），再通过逆变换转回空间域。在计算复杂度上，这种 O(N) 的点乘远快于巨大感受野下的卷积运算。

### 策略 C：按景深分化处理 (In-Focus vs. Off-Focus)
考虑到运算规模，我们需要将渲染策略依据模糊半径分离：

- **针对合焦区域 (In-focus PSF)**
  - 合焦区域（即便是存在轻微镜头像差的边缘画质衰减）的 PSF Kernel 尺寸非常小（例如最多 11x11 到 15x15 的规模）。
  - 对于这种小核，应用上述的**基于 DCT 的频域插值与运算**策略是非常合适的，由于矩阵乘法的维度小，整体运算负担可控。

- **针对大规模失焦区域 (Off-focus PSF)**
  - 当模糊半径变大时，频域插值带来的巨大 `kernel_w * kernel_h` 存储将变成瓶颈。
  - 对于这种极端情况，使用庞大离散矩阵计算不再经济。我们引入一种 **闭式解近似 (Closed-form approximation)** 的数学模型。
  - **基于径向基与高频谐波组合的解析模型**：
    我们将庞大且复杂的失焦光斑近似解析为：
    ```python
    approximated_kernel = radial(center, radius) + sum([cosine_terms(center, radius_clamp_i, amp_i, phase_i, freq_i) for i in range(N)])
    ```
    - **`radial(center, radius)`**: 这是光斑的基础低频投影（代表主能量的均值弥散分布，例如平顶几何弥散圆）。
    - **`cosine_terms`**: 包含 $N$ 个（例如 $N=11$）由余弦波叠加成的高频项，专门用来拟合透镜和孔径带来的光学伪影（如衍射环 Diffraction Rings 或“洋葱圈”效应 Onion-ring bokeh）。
      截断衰减逻辑如下：针对特定的采样点，若它到模糊中心点的距离（radius）超过了该项的截断边界 `radius_clamp`，则该附加项输出 `0`；否则输出 `amp * cos(radius * freq + phase)`。
    - **优势**: 有了这个解析形式，任意位置、任意尺寸的大型 PSF 都不再需要离散化数值存储，只要提取/拟合出那数十个一维标量参数，在渲染时仅需做解析函数的代入加和计算，既实现了空间连续性，又达成了极高保真度的物理仿真性能突破！
