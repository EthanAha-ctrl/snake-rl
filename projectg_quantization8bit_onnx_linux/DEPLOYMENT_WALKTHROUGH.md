# Snake-RL C++ ONNX 端到端推理部署指南 (AArch64)

本文档面向需要在 **Apple Silicon (M系列芯片) Mac 本地虚拟机 (Fedora Linux AArch64本机虚拟化)** 等桌面级 ARM64 环境下**复现并编译** Snake-RL 强化学习脱机推理环境（基于 ONNX Runtime C++）的开发同事。

## 1. 核心目录与物料清单
请确保你已经将本项目的 `projectg_quantization8bit_onnx_linux` 目录克隆到目标板子中。主要需要的资产分为两部分：

1. **核心模型与数据 (项目根目录):**
   - `hrnet_int8.onnx` (11.6MB) - 量化版视觉骨干网
   - `transformer_int8.onnx` (546KB) - 量化版时空特征栈
   - `sac_actor.onnx` (6.3KB) - 极速 FP32 决策网络
   - `coc_images.bson` (1.5GB+) - BSON 形式打包的图像测试集

2. **C++ 源工程 (`c_plus_plus_src`):**
   - `inference.cpp` / `CoCEnv.cpp` / `HistoryStacker.cpp` 等模拟器源码
   - `CMakeLists.txt` - 已配置自动拉包的构建脚本

## 2. 编译前置准备

根据我们之前跑通的 `onnx-hello` 范例，你需要**手动向 `c_plus_plus_src` 目录补充两个外部依赖模块**，剩下的 CMake 会自动处理：

1. **ONNX Runtime (C/C++ SDK)**:
   - 从微软官网或包管理器获取适配 Linux AArch64 的 `onnxruntime` 二进制压缩包。
   - 解压后放入 `c_plus_plus_src` 下，并重命名为 `onnxruntime`。
   - 内部结构必须长这样：
     ```text
     c_plus_plus_src/
     └── onnxruntime/
         ├── include/     <-- 包含 onnxruntime_cxx_api.h 等头文件
         └── lib/         <-- 包含 libonnxruntime.so 等动态库
     ```

2. **`stb_image.h` (轻量级图像解码器)**:
   - 你可以直接从原先测试成功的 `reference/onnx-hello` 文件夹中将单头文件 `stb_image.h` 拷贝到 `c_plus_plus_src` 的同一级代码目录中。

*注：处理 BSON 需要的第三方解析库 `nlohmann_json` 已经被写在 CMake 中配置为了自动从 GitHub 临时同步，无需提前准备（要求编译时机器联网）。*

## 3. 标准编译流程 

在终端切换至 `c_plus_plus_src` 目录下，并执行外置 CMake 构建：

```bash
mkdir build
cd build
cmake ..
make -j4
```

## 4. 运行推理与测试
构建成功后，直接执行编译出的二进制文件：

```bash
./inference
```

该程序会在启动时做以下几件事：
1. 装载父目录 (`../`) 下生成的 三个 `*.onnx` 模型到跨线程 Session。
2. 读取并缓存在内存里的 `coc_images.bson` 测试数据集。
3. 进入 `CoCEnv` RL 模拟沙盘，由 `stb_image` 动态解码 png、由 C++ 的 `HistoryStacker` 并发压栈帧进行多维张量填充。
4. 依次打通 `HRNet` -> `Transformer` -> `SAC Actor` 的 ONNX 图，并在标准输出流呈现环境反馈：
   ```text
    Guess: 0.XY | Trigger: 0 | Reward: [..., ...]
   ```

若出现 `Fatal Error: Cannot open BSON file / ONNX load failed` 请随时检查上面所列出的相对文件路径 (`../xxx.onnx`)。
