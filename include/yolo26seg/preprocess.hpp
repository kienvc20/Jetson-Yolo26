#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace y26 {

struct LetterboxTransform {
    float scale{1.0F};
    float padX{0.0F};
    float padY{0.0F};
    int resizedWidth{0};
    int resizedHeight{0};
};

LetterboxTransform computeLetterbox(
    int sourceWidth,
    int sourceHeight,
    int destinationWidth,
    int destinationHeight);

void launchPreprocess(
    const std::uint8_t* sourceBgr,
    int sourceWidth,
    int sourceHeight,
    int sourceStride,
    void* destinationNchw,
    int destinationWidth,
    int destinationHeight,
    nvinfer1::DataType destinationType,
    cudaStream_t stream);

// Preprocess one image directly into slot `batchIndex` of an already allocated
// [N, 3, H, W] TensorRT input tensor.
void launchPreprocessBatchSlot(
    const std::uint8_t* sourceBgr,
    int sourceWidth,
    int sourceHeight,
    int sourceStride,
    void* destinationNchwBatch,
    int batchIndex,
    int destinationWidth,
    int destinationHeight,
    nvinfer1::DataType destinationType,
    cudaStream_t stream);

}  // namespace y26
