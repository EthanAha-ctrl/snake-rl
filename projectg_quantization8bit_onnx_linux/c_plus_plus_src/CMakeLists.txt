cmake_minimum_required(VERSION 3.14)
project(SnakeRL_Inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies:
# 1. nlohmann_json
# 2. onnxruntime

# Users must provide onnxruntime path or install it globally
find_package(OnnxRuntime)
find_package(nlohmann_json)

# For stb_image, it's a header-only library. The user should put stb_image.h in this directory.

add_executable(inference
    inference.cpp
    CoCEnv.cpp
    HistoryStacker.cpp
)

# Replace with the actual namespace/target names according to local setup
if(TARGET OnnxRuntime::onnxruntime OR TARGET onnxruntime)
    target_link_libraries(inference PRIVATE onnxruntime)
else()
    message(WARNING "onnxruntime not found via find_package. Adjust link libraries manually.")
    target_link_libraries(inference PRIVATE onnxruntime)
endif()

if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(inference PRIVATE nlohmann_json::nlohmann_json)
else()
    message(WARNING "nlohmann_json not found via find_package. Make sure it is in include paths.")
endif()
