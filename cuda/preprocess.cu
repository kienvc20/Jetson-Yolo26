#include "yolo26seg/preprocess.hpp"

#include "yolo26seg/cuda_utils.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace y26 {
namespace {

template <typename Output>
__device__ Output convertOutput(float value);

template <>
__device__ float convertOutput<float>(float value) {
    return value;
}

template <>
__device__ __half convertOutput<__half>(float value) {
    return __float2half(value);
}

__device__ float sampleChannel(
    const std::uint8_t* source,
    int width,
    int height,
    int stride,
    float x,
    float y,
    int bgrChannel) {
    x = fminf(fmaxf(x, 0.0F), static_cast<float>(width - 1));
    y = fminf(fmaxf(y, 0.0F), static_cast<float>(height - 1));

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);

    const float p00 = source[y0 * stride + x0 * 3 + bgrChannel];
    const float p01 = source[y0 * stride + x1 * 3 + bgrChannel];
    const float p10 = source[y1 * stride + x0 * 3 + bgrChannel];
    const float p11 = source[y1 * stride + x1 * 3 + bgrChannel];

    const float top = p00 + (p01 - p00) * dx;
    const float bottom = p10 + (p11 - p10) * dx;
    return top + (bottom - top) * dy;
}

template <typename Output>
__global__ void preprocessKernel(
    const std::uint8_t* source,
    int sourceWidth,
    int sourceHeight,
    int sourceStride,
    Output* destination,
    int destinationWidth,
    int destinationHeight,
    float scale,
    float padX,
    float padY,
    int resizedWidth,
    int resizedHeight) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= destinationWidth || y >= destinationHeight) {
        return;
    }

    const int plane = destinationWidth * destinationHeight;
    const int offset = y * destinationWidth + x;
    const bool isPadding =
        x < static_cast<int>(padX) ||
        y < static_cast<int>(padY) ||
        x >= static_cast<int>(padX) + resizedWidth ||
        y >= static_cast<int>(padY) + resizedHeight;

    float red = 114.0F;
    float green = 114.0F;
    float blue = 114.0F;

    if (!isPadding) {
        const float sourceX =
            (static_cast<float>(x) - padX + 0.5F) / scale - 0.5F;
        const float sourceY =
            (static_cast<float>(y) - padY + 0.5F) / scale - 0.5F;

        blue = sampleChannel(
            source, sourceWidth, sourceHeight, sourceStride, sourceX, sourceY, 0);
        green = sampleChannel(
            source, sourceWidth, sourceHeight, sourceStride, sourceX, sourceY, 1);
        red = sampleChannel(
            source, sourceWidth, sourceHeight, sourceStride, sourceX, sourceY, 2);
    }

    destination[offset] = convertOutput<Output>(red / 255.0F);
    destination[plane + offset] = convertOutput<Output>(green / 255.0F);
    destination[2 * plane + offset] = convertOutput<Output>(blue / 255.0F);
}

}  // namespace

LetterboxTransform computeLetterbox(
    int sourceWidth,
    int sourceHeight,
    int destinationWidth,
    int destinationHeight) {
    if (sourceWidth <= 0 || sourceHeight <= 0 || destinationWidth <= 0 ||
        destinationHeight <= 0) {
        throw std::invalid_argument("Letterbox dimensions must be positive");
    }

    LetterboxTransform transform;
    transform.scale = std::min(
        static_cast<float>(destinationWidth) / sourceWidth,
        static_cast<float>(destinationHeight) / sourceHeight);
    transform.resizedWidth =
        static_cast<int>(std::round(sourceWidth * transform.scale));
    transform.resizedHeight =
        static_cast<int>(std::round(sourceHeight * transform.scale));
    transform.padX =
        static_cast<float>((destinationWidth - transform.resizedWidth) / 2);
    transform.padY =
        static_cast<float>((destinationHeight - transform.resizedHeight) / 2);
    return transform;
}

void launchPreprocess(
    const std::uint8_t* sourceBgr,
    int sourceWidth,
    int sourceHeight,
    int sourceStride,
    void* destinationNchw,
    int destinationWidth,
    int destinationHeight,
    nvinfer1::DataType destinationType,
    cudaStream_t stream) {
    const LetterboxTransform transform = computeLetterbox(
        sourceWidth,
        sourceHeight,
        destinationWidth,
        destinationHeight);

    const dim3 block(16, 16);
    const dim3 grid(
        (destinationWidth + block.x - 1) / block.x,
        (destinationHeight + block.y - 1) / block.y);

    if (destinationType == nvinfer1::DataType::kFLOAT) {
        preprocessKernel<<<grid, block, 0, stream>>>(
            sourceBgr,
            sourceWidth,
            sourceHeight,
            sourceStride,
            static_cast<float*>(destinationNchw),
            destinationWidth,
            destinationHeight,
            transform.scale,
            transform.padX,
            transform.padY,
            transform.resizedWidth,
            transform.resizedHeight);
    } else if (destinationType == nvinfer1::DataType::kHALF) {
        preprocessKernel<<<grid, block, 0, stream>>>(
            sourceBgr,
            sourceWidth,
            sourceHeight,
            sourceStride,
            static_cast<__half*>(destinationNchw),
            destinationWidth,
            destinationHeight,
            transform.scale,
            transform.padX,
            transform.padY,
            transform.resizedWidth,
            transform.resizedHeight);
    } else {
        throw std::runtime_error(
            "CUDA preprocessing supports only FP32 and FP16 model inputs");
    }

    Y26_CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace y26
