#pragma once

#include "yolo26seg/yolo26_seg.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace y26 {

struct CameraFrame {
    std::string cameraId;
    std::uint64_t sequence{0};
    std::uint64_t timestampNs{0};
    const std::uint8_t* bgr{nullptr};
    int width{0};
    int height{0};
    int stride{0};
    std::shared_ptr<void> owner;
};

struct SegmentationResult {
    std::string cameraId;
    std::uint64_t sequence{0};
    float preprocessMs{0.0F};
    float inferenceMs{0.0F};
    float bestScore{0.0F};
    int bestIndex{-1};
    std::size_t outputValues{0};
};

class DynamicSegPipeline {
public:
    DynamicSegPipeline(
        const std::string& enginePath,
        int inputWidth,
        int inputHeight,
        const std::string& inputName,
        const std::string& primaryOutputName,
        const std::vector<std::string>& configuredCameraIds);

    // frames must be in the same fixed order as configuredCameraIds.
    // Exactly one fresh frame per configured camera is required.
    std::vector<SegmentationResult> processFixedBatch(
        const std::vector<CameraFrame>& frames);

    int batchSize() const noexcept {
        return static_cast<int>(cameraIds_.size());
    }

private:
    static void basicPostprocess(
        const float* begin,
        const float* end,
        SegmentationResult& result);

    std::vector<std::string> cameraIds_;
    Yolo26Seg model_;
    std::string primaryOutputName_;
};

} // namespace y26
