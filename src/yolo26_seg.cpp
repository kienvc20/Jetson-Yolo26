#include "yolo26seg/yolo26_seg.hpp"

#include "yolo26seg/cuda_utils.hpp"

#include <algorithm>
#include <stdexcept>

namespace y26 {

Yolo26Seg::Yolo26Seg(
    const std::string& enginePath,
    int inputWidth,
    int inputHeight,
    const std::string& inputName)
    : engine_(enginePath),
      inputName_(inputName),
      inputWidth_(inputWidth),
      inputHeight_(inputHeight) {
    if (inputWidth_ <= 0 || inputHeight_ <= 0) {
        throw std::invalid_argument("Model input width and height must be positive");
    }

    engine_.setInputShape(
        inputName_,
        nvinfer1::Dims4{1, 3, inputHeight_, inputWidth_});
    engine_.allocateBindings();

    Y26_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    Y26_CUDA_CHECK(cudaEventCreate(&preprocessStart_));
    Y26_CUDA_CHECK(cudaEventCreate(&preprocessEnd_));
    Y26_CUDA_CHECK(cudaEventCreate(&inferenceEnd_));
}

Yolo26Seg::~Yolo26Seg() {
    if (sourceDevice_ != nullptr) {
        cudaFree(sourceDevice_);
    }
    if (inferenceEnd_ != nullptr) {
        cudaEventDestroy(inferenceEnd_);
    }
    if (preprocessEnd_ != nullptr) {
        cudaEventDestroy(preprocessEnd_);
    }
    if (preprocessStart_ != nullptr) {
        cudaEventDestroy(preprocessStart_);
    }
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

InferenceStats Yolo26Seg::infer(
    const std::uint8_t* sourceBgr,
    int sourceWidth,
    int sourceHeight,
    int sourceStride) {
    if (sourceBgr == nullptr || sourceWidth <= 0 || sourceHeight <= 0 ||
        sourceStride < sourceWidth * 3) {
        throw std::invalid_argument("Invalid BGR source image");
    }

    const std::size_t sourceBytes =
        static_cast<std::size_t>(sourceStride) * sourceHeight;
    ensureSourceCapacity(sourceBytes);
    lastTransform_ = computeLetterbox(
        sourceWidth,
        sourceHeight,
        inputWidth_,
        inputHeight_);

    const int inputIndex = engine_.bindingIndex(inputName_);
    const BindingInfo& inputBinding = engine_.bindings().at(inputIndex);

    Y26_CUDA_CHECK(cudaEventRecord(preprocessStart_, stream_));
    Y26_CUDA_CHECK(cudaMemcpyAsync(
        sourceDevice_,
        sourceBgr,
        sourceBytes,
        cudaMemcpyHostToDevice,
        stream_));

    launchPreprocess(
        sourceDevice_,
        sourceWidth,
        sourceHeight,
        sourceStride,
        engine_.deviceBuffer(inputIndex),
        inputWidth_,
        inputHeight_,
        inputBinding.dataType,
        stream_);

    Y26_CUDA_CHECK(cudaEventRecord(preprocessEnd_, stream_));

    if (!engine_.enqueue(stream_)) {
        throw std::runtime_error("TensorRT enqueueV2 failed");
    }

    Y26_CUDA_CHECK(cudaEventRecord(inferenceEnd_, stream_));
    Y26_CUDA_CHECK(cudaEventSynchronize(inferenceEnd_));

    InferenceStats stats;
    Y26_CUDA_CHECK(cudaEventElapsedTime(
        &stats.preprocessMilliseconds,
        preprocessStart_,
        preprocessEnd_));
    Y26_CUDA_CHECK(cudaEventElapsedTime(
        &stats.inferenceMilliseconds,
        preprocessEnd_,
        inferenceEnd_));
    return stats;
}

std::vector<float> Yolo26Seg::downloadFloatOutput(
    const std::string& bindingName) {
    const int index = engine_.bindingIndex(bindingName);
    const BindingInfo& binding = engine_.bindings().at(index);
    if (binding.isInput) {
        throw std::invalid_argument("Requested binding is an input: " + bindingName);
    }
    if (binding.dataType != nvinfer1::DataType::kFLOAT) {
        throw std::invalid_argument(
            "downloadFloatOutput requires an FP32 output; " + bindingName +
            " is " + dataTypeName(binding.dataType));
    }

    std::vector<float> host(binding.bytes / sizeof(float));
    Y26_CUDA_CHECK(cudaMemcpyAsync(
        host.data(),
        engine_.deviceBuffer(index),
        binding.bytes,
        cudaMemcpyDeviceToHost,
        stream_));
    Y26_CUDA_CHECK(cudaStreamSynchronize(stream_));
    return host;
}

void Yolo26Seg::ensureSourceCapacity(std::size_t requiredBytes) {
    if (requiredBytes <= sourceCapacity_) {
        return;
    }

    if (sourceDevice_ != nullptr) {
        Y26_CUDA_CHECK(cudaFree(sourceDevice_));
        sourceDevice_ = nullptr;
        sourceCapacity_ = 0;
    }

    Y26_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&sourceDevice_),
        requiredBytes));
    sourceCapacity_ = requiredBytes;
}

}  // namespace y26
