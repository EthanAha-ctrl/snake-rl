cmake_minimum_required(VERSION 3.14)
project(SnakeRL_Inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

# Fix for CMake < 3.5 removed support in FetchContent
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# 1. Include nlohmann_json via FetchContent
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# 2. Setup ONNX Runtime (align with reference/onnx-hello structure)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib)

# 3. Executable and Libraries
add_executable(inference
    inference.cpp
    coc_env.cpp
    history_stacker.cpp
)

target_link_libraries(inference PRIVATE onnxruntime nlohmann_json::nlohmann_json)
set_target_properties(inference PROPERTIES INSTALL_RPATH "$ORIGIN/onnxruntime/lib")
