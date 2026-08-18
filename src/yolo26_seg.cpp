#include "yolo26seg/yolo26_seg.hpp"

#include "yolo26seg/cuda_utils.hpp"

#include <stdexcept>

namespace y26 {

Yolo26Seg::Yolo26Seg(
    const std::string& enginePath,
    int inputWidth,
    int inputHeight,
    const std::string& inputName,
    int batchSize)
    : engine_(enginePath),
      inputName_(inputName),
      inputWidth_(inputWidth),
      inputHeight_(inputHeight),
      batchSize_(batchSize) {
    if (inputWidth_ <= 0 || inputHeight_ <= 0 || batchSize_ <= 0) {
        throw std::invalid_argument("Model input dimensions and batch size must be positive");
    }

    engine_.setInputShape(
        inputName_,
        nvinfer1::Dims4{batchSize_, 3, inputHeight_, inputWidth_});
    engine_.allocateBindings();

    Y26_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    Y26_CUDA_CHECK(cudaEventCreate(&preprocessStart_));
    Y26_CUDA_CHECK(cudaEventCreate(&preprocessEnd_));
    Y26_CUDA_CHECK(cudaEventCreate(&inferenceEnd_));
}

Yolo26Seg::~Yolo26Seg() {
    if (sourceDevice_ != nullptr) cudaFree(sourceDevice_);
    if (inferenceEnd_ != nullptr) cudaEventDestroy(inferenceEnd_);
    if (preprocessEnd_ != nullptr) cudaEventDestroy(preprocessEnd_);
    if (preprocessStart_ != nullptr) cudaEventDestroy(preprocessStart_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

InferenceStats Yolo26Seg::infer(
    const std::uint8_t* sourceBgr,
    int sourceWidth,
    int sourceHeight,
    int sourceStride) {
    if (batchSize_ != 1) {
        throw std::logic_error("infer() is only valid when batchSize == 1; use inferBatch()");
    }
    return inferBatch({BatchImageView{sourceBgr, sourceWidth, sourceHeight, sourceStride}});
}

InferenceStats Yolo26Seg::inferBatch(const std::vector<BatchImageView>& images) {
    if (static_cast<int>(images.size()) != batchSize_) {
        throw std::invalid_argument("inferBatch image count must equal configured fixed batch size");
    }

    std::size_t totalSourceBytes = 0;
    for (const BatchImageView& image : images) {
        if (image.bgr == nullptr || image.width <= 0 || image.height <= 0 ||
            image.stride < image.width * 3) {
            throw std::invalid_argument("Invalid BGR image in fixed batch");
        }
        totalSourceBytes += static_cast<std::size_t>(image.stride) * image.height;
    }
    ensureSourceCapacity(totalSourceBytes);

    const int inputIndex = engine_.bindingIndex(inputName_);
    const BindingInfo& inputBinding = engine_.bindings().at(inputIndex);

    Y26_CUDA_CHECK(cudaEventRecord(preprocessStart_, stream_));

    std::size_t sourceOffset = 0;
    for (int batchIndex = 0; batchIndex < batchSize_; ++batchIndex) {
        const BatchImageView& image = images[static_cast<std::size_t>(batchIndex)];
        const std::size_t bytes = static_cast<std::size_t>(image.stride) * image.height;
        std::uint8_t* deviceSource = sourceDevice_ + sourceOffset;

        Y26_CUDA_CHECK(cudaMemcpyAsync(
            deviceSource,
            image.bgr,
            bytes,
            cudaMemcpyHostToDevice,
            stream_));

        launchPreprocessBatchSlot(
            deviceSource,
            image.width,
            image.height,
            image.stride,
            engine_.deviceBuffer(inputIndex),
            batchIndex,
            inputWidth_,
            inputHeight_,
            inputBinding.dataType,
            stream_);

        sourceOffset += bytes;
    }

    Y26_CUDA_CHECK(cudaEventRecord(preprocessEnd_, stream_));

    if (!engine_.enqueue(stream_)) {
        throw std::runtime_error("TensorRT enqueueV2 failed");
    }

    Y26_CUDA_CHECK(cudaEventRecord(inferenceEnd_, stream_));
    Y26_CUDA_CHECK(cudaEventSynchronize(inferenceEnd_));

    InferenceStats stats;
    Y26_CUDA_CHECK(cudaEventElapsedTime(
        &stats.preprocessMilliseconds, preprocessStart_, preprocessEnd_));
    Y26_CUDA_CHECK(cudaEventElapsedTime(
        &stats.inferenceMilliseconds, preprocessEnd_, inferenceEnd_));
    return stats;
}

std::vector<float> Yolo26Seg::downloadFloatOutput(const std::string& bindingName) {
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
        host.data(), engine_.deviceBuffer(index), binding.bytes,
        cudaMemcpyDeviceToHost, stream_));
    Y26_CUDA_CHECK(cudaStreamSynchronize(stream_));
    return host;
}

void Yolo26Seg::ensureSourceCapacity(std::size_t requiredBytes) {
    if (requiredBytes <= sourceCapacity_) return;

    if (sourceDevice_ != nullptr) {
        Y26_CUDA_CHECK(cudaFree(sourceDevice_));
        sourceDevice_ = nullptr;
        sourceCapacity_ = 0;
    }

    Y26_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&sourceDevice_), requiredBytes));
    sourceCapacity_ = requiredBytes;
}

} // namespace y26
