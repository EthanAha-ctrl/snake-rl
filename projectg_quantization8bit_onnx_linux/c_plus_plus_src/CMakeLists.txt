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

# 3. Shared Library focus_engine
add_library(focus_engine SHARED engine.cpp)

# Attempt to find a static ONNX Runtime library as requested
find_library(ONNXRUNTIME_STATIC NAMES libonnxruntime.a onnxruntime.a PATHS ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib NO_DEFAULT_PATH)

if(ONNXRUNTIME_STATIC)
    message(STATUS "Found static ONNX Runtime: ${ONNXRUNTIME_STATIC}")
    # Link entire static archive to ensure all symbols are brought into the shared library
    target_link_libraries(focus_engine PRIVATE -Wl,--whole-archive ${ONNXRUNTIME_STATIC} -Wl,--no-whole-archive)
else()
    message(WARNING "Static ONNX Runtime (.a) not found. Falling back to dynamic linking (libonnxruntime.so).")
    target_link_libraries(focus_engine PRIVATE onnxruntime)
endif()

# 4. Test Executable
add_executable(test_inference
    test_inference.cpp
    coc_env.cpp
    history_stacker.cpp
)

target_link_libraries(test_inference PRIVATE focus_engine nlohmann_json::nlohmann_json)

# Set RPATH so the executable can find libfocus_engine.so in the same directory, 
# and optionally libonnxruntime.so if static linking failed.
set_target_properties(test_inference PROPERTIES INSTALL_RPATH "$ORIGIN:$ORIGIN/onnxruntime/lib")
