#pragma once

#include "yolo26seg/preprocess.hpp"
#include "yolo26seg/trt_engine.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace y26 {

struct InferenceStats {
    float preprocessMilliseconds{0.0F};
    float inferenceMilliseconds{0.0F};
};

struct BatchImageView {
    const std::uint8_t* bgr{nullptr};
    int width{0};
    int height{0};
    int stride{0};
};

class Yolo26Seg {
public:
    Yolo26Seg(
        const std::string& enginePath,
        int inputWidth,
        int inputHeight,
        const std::string& inputName = "images",
        int batchSize = 1);

    ~Yolo26Seg();

    Yolo26Seg(const Yolo26Seg&) = delete;
    Yolo26Seg& operator=(const Yolo26Seg&) = delete;

    InferenceStats infer(
        const std::uint8_t* sourceBgr,
        int sourceWidth,
        int sourceHeight,
        int sourceStride);

    InferenceStats inferBatch(const std::vector<BatchImageView>& images);

    std::vector<float> downloadFloatOutput(const std::string& bindingName);
    const std::vector<BindingInfo>& bindings() const noexcept { return engine_.bindings(); }
    LetterboxTransform lastTransform() const noexcept { return lastTransform_; }
    int batchSize() const noexcept { return batchSize_; }

private:
    void ensureSourceCapacity(std::size_t requiredBytes);

    TrtEngine engine_;
    std::string inputName_;
    int inputWidth_;
    int inputHeight_;
    int batchSize_{1};
    cudaStream_t stream_{nullptr};
    cudaEvent_t preprocessStart_{nullptr};
    cudaEvent_t preprocessEnd_{nullptr};
    cudaEvent_t inferenceEnd_{nullptr};
    std::uint8_t* sourceDevice_{nullptr};
    std::size_t sourceCapacity_{0};
    LetterboxTransform lastTransform_{};
};

} // namespace y26
