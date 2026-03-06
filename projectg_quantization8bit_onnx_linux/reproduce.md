# 编译复现指南 (Snake-RL C++ Inference)

本文档记录了在 Linux 环境下（如 Fedora x86_64 虚拟机 / 无 `sudo` 权限的普通用户态）成功编译本预测引擎所需的完整修复步骤与依赖拉取命令。因为初始环境遇到了一些网络和依赖缺失的问题，以下是修正后的标准操作流程：

## 1. 安装最新版 CMake
由于系统自带的 CMake 版本可能过低或者未安装，且当前用户无 `sudo` 权限，建议通过 `pip` 安装用户隔离区的 CMake：

```bash
python3 -m pip install --user cmake
export PATH=~/.local/bin:$PATH
```

## 2. 下载必要资产与预编译库

请先切换至 C++ 源码目录：
```bash
cd c_plus_plus_src
```

### 2.1 获取图像解码库 `stb_image.h`
```bash
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -O stb_image.h
```

### 2.2 获取 ONNX Runtime
*注：请根据您的系统架构选择。我们在测试环境中检测到系统内核为 `x86_64`，因此使用了 x64 位包。若您在真正的 AArch64 (ARM64) 系统上，请解除对应命令的注释。*

```bash
# 对于 x86_64 (Intel/AMD) 架构：
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar xzf onnxruntime-linux-x64-1.18.0.tgz
mv onnxruntime-linux-x64-1.18.0 onnxruntime

# 对于 AArch64 (Apple M 系列虚拟机的 Linux ARM64) 架构：
# wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-aarch64-1.18.0.tgz
# tar xzf onnxruntime-linux-aarch64-1.18.0.tgz
# mv onnxruntime-linux-aarch64-1.18.0 onnxruntime
```

## 3. 代码与配置修复补丁

### 3.1 补充缺失的宏定义 (Fix undefined reference to `stbi_load_from_memory`)
在执行 Make 时可能会报找不到 `stbi_*` 相关函数的链接错误，需在 `CoCEnv.cpp` 的头文件引用前声明宏：

修改前 (`CoCEnv.cpp` 第 9-11 行附近):
```cpp
// Dependency assumes nlohmann/json for BSON parsing and stb_image.h for PNG
#include "stb_image.h"
```

修改后:
```cpp
// Dependency assumes nlohmann/json for BSON parsing and stb_image.h for PNG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
```

### 3.2 优化 CMake 网络依赖的抓取 (加速 `nlohmann_json` 开发包)
原有 CMakeLists 中直接从 Git 仓库 clone 可能非常缓慢且容易导致超期，建议将 `CMakeLists.txt` 中的 Git 下载方式变更为更快的 HTTP 断点直链包：

修改前 (`CMakeLists.txt` 第 9-12 行附近):
```cmake
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
```

修改后:
```cmake
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
```

## 4. 执行编译
由于依赖了从包体拉取的较新 FetchContent，建议向老版本 CMake 声明向下兼容策略，完成最终的外部构建：

```bash
mkdir -p build && cd build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make -j4
```

## 5. 开始推理测试
编译生成的执行文件为 `./inference`，此时调用它将会验证模型依赖关系。
```bash
./inference
```
*(注意：运行前务必将所需的三个 *.onnx 模型及 coc_images.bson 准备在项目根目录下)*
