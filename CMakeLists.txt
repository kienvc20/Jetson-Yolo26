cmake_minimum_required(VERSION 3.10)

project(yolo26_seg_jetson_nano LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(Y26_BUILD_PYTHON "Build the pybind11 Python module" ON)
option(Y26_BUILD_APP "Build the image inference CLI" ON)

# JetPack 4.6 on the original Jetson Nano ships CUDA 10.2. FindCUDA is used
# intentionally because the stock Ubuntu 18.04 CMake is older.
find_package(CUDA 10.2 REQUIRED)

find_path(
    TENSORRT_INCLUDE_DIR
    NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/local/TensorRT/include
)

find_library(
    TENSORRT_LIBRARY
    nvinfer
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/local/TensorRT/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT was not found. Install JetPack/TensorRT or set TENSORRT_INCLUDE_DIR and TENSORRT_LIBRARY.")
endif()

list(APPEND CUDA_NVCC_FLAGS
    "-gencode=arch=compute_53,code=sm_53"
    "--use_fast_math"
)

cuda_add_library(
    yolo26seg_core
    src/trt_engine.cpp
    src/yolo26_seg.cpp
    cuda/preprocess.cu
)

target_include_directories(
    yolo26seg_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(
    yolo26seg_core
    PUBLIC ${TENSORRT_LIBRARY} ${CUDA_CUDART_LIBRARY}
)

if(Y26_BUILD_APP)
    find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)
    add_executable(yolo26seg_cli apps/infer_image.cpp)
    target_include_directories(yolo26seg_cli PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(yolo26seg_cli PRIVATE yolo26seg_core ${OpenCV_LIBS})
endif()

if(Y26_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(yolo26seg_runtime python/bindings.cpp)
    target_link_libraries(yolo26seg_runtime PRIVATE yolo26seg_core)
endif()
