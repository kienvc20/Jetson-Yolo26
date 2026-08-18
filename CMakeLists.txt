cmake_minimum_required(VERSION 3.10)

project(yolo26_seg_jetson_nano LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(Y26_BUILD_PYTHON "Build the pybind11 Python module" ON)
option(Y26_BUILD_APP "Build the image inference CLI" ON)
option(Y26_BUILD_BASLER "Build the dynamic Basler camera pipeline" OFF)

# JetPack 4.6 on the original Jetson Nano ships CUDA 10.2. FindCUDA is used
# intentionally because the stock Ubuntu 18.04 CMake is older.
find_package(CUDA 10.2 REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/local/TensorRT/include)
find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/local/TensorRT/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT was not found. Install JetPack/TensorRT or set TENSORRT_INCLUDE_DIR and TENSORRT_LIBRARY.")
endif()

list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_53,code=sm_53" "--use_fast_math")

cuda_add_library(yolo26seg_core
    src/trt_engine.cpp
    src/yolo26_seg.cpp
    src/dynamic_pipeline.cpp
    cuda/preprocess.cu)

target_include_directories(yolo26seg_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
target_link_libraries(yolo26seg_core PUBLIC ${TENSORRT_LIBRARY} ${CUDA_CUDART_LIBRARY})

if(Y26_BUILD_APP)
    find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)
    add_executable(yolo26seg_cli apps/infer_image.cpp)
    target_include_directories(yolo26seg_cli PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(yolo26seg_cli PRIVATE yolo26seg_core ${OpenCV_LIBS})
endif()

if(Y26_BUILD_BASLER)
    find_path(PYLON_INCLUDE_DIR pylon/PylonIncludes.h HINTS /opt/pylon/include /usr/include)
    find_library(PYLON_BASE_LIBRARY pylonbase HINTS /opt/pylon/lib /opt/pylon/lib64 /usr/lib)
    find_library(PYLON_UTILITY_LIBRARY pylonutility HINTS /opt/pylon/lib /opt/pylon/lib64 /usr/lib)
    if(NOT PYLON_INCLUDE_DIR OR NOT PYLON_BASE_LIBRARY)
        message(FATAL_ERROR "Basler pylon SDK not found. Set PYLON_INCLUDE_DIR/PYLON_BASE_LIBRARY or install pylon.")
    endif()
    add_executable(basler_dynamic_seg apps/basler_dynamic_seg.cpp)
    target_include_directories(basler_dynamic_seg PRIVATE ${PYLON_INCLUDE_DIR})
    target_link_libraries(basler_dynamic_seg PRIVATE yolo26seg_core ${PYLON_BASE_LIBRARY} ${PYLON_UTILITY_LIBRARY})
endif()

if(Y26_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(yolo26seg_runtime python/bindings.cpp)
    target_link_libraries(yolo26seg_runtime PRIVATE yolo26seg_core)
endif()
