# 部署与测试全链路指南

本项目已从头重构了整个强化学习环境和推理引擎到 C++，使之能够完全且彻底地脱离 Python 原生运行并部署。以下是全部工作的成果与执行指南。

## 1. 数据准备 (Python)
所有的 C++ 图像处理将直接依赖预处理产生的原始 PNG 数据。运行下面的脚本即可将所有的模糊图片和相关信息打包成超大但不失紧凑的 BSON 二进制档案。
```bash
python convert_lmdb_to_bson.py
```
> **注意**: 如果找不到 `bson` 模块，请先通过 `pip install pymongo` 进行安装（内含 C 扩展加速版的 bson 库）。运行后会在目录产生 `coc_images.bson` 文件供 C++ 客户端直读。

## 2. 模型串联导出 (Python)
原庞大的 Actor-Encoder 网路被完全解耦，分别导出成标准的中间表示 `*.onnx`，彻底甩掉了所有不必要的训练冗余组件。
请依次平滑执行：
```bash
python export_hrnet.py
python export_transformer_and_actor.py
```
> **提示**: 生成 `hrnet.onnx`、`transformer.onnx`、`sac_actor.onnx` 三个模型。

如需获得 ARM 上表现最巅峰的 INT 8 动态量化，请再执行：
```bash
python quantize_models.py
```
> 将在同级目录并列产出 `*_int8.onnx`。

## 3. C++ 高级仿真环境及推理

在 `c_plus_plus_src` 目录下，我们实现了一个伟大的壮举：**纯 C++ 端完全一比一复刻了强化学习环境！**
- **`CoCEnv.cpp`**: 自带极高精度的 BSON 读取器、基于 `#include "stb_image.h"` 的无感 PNG 解码器，自带特征交叉图片混合插值阵列。它甚至在内部私有化了 `Ort::Session *hrnet` 对图像直给 ONNX 模拟出 Vision Tensor。
- **`HistoryStacker.cpp`**: 用 `std::deque` 动态维护的变长定维的 Observation/Action 堆叠器。
- **`inference.cpp`**: 串联三个 `*_int8.onnx` 模型，拉起 C++ 环境流全闭环测试。

### 编译步骤
你需要安装 `cmake`，并且系统要能找到 `nlohmann_json` (Header-only) 或对应的 ONNX Runtimes 头文件和动态链接库。然后使用极简方法挂起：

```bash
cd c_plus_plus_src
mkdir build && cd build
cmake ..
make
./inference
```
接下来应该就会看到 `CoCEnv` 在加载完所有的 BSON 之后，在控制台丝滑输出每 Step 的 Reward、Guess 概率和动作了！将其移植到 Raspberry Pi 即是大功告成。
