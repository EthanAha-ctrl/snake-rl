cmake_minimum_required(VERSION 3.14)
project(SnakeRL_Inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Include nlohmann_json via FetchContent
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

# 2. Setup ONNX Runtime (align with reference/onnx-hello structure)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib)

# 3. Executable and Libraries
add_executable(inference
    inference.cpp
    CoCEnv.cpp
    HistoryStacker.cpp
)

target_link_libraries(inference PRIVATE onnxruntime nlohmann_json::nlohmann_json)
set_target_properties(inference PROPERTIES INSTALL_RPATH "$ORIGIN/onnxruntime/lib")
