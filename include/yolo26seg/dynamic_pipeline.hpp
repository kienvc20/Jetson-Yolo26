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
    std::shared_ptr<void> owner; // keeps the Pylon grab result alive
};

struct SegmentationResult {
    std::string cameraId;
    std::uint64_t sequence{0};
    float preprocessMs{0.0F};
    float inferenceMs{0.0F};
    float bestScore{0.0F};
    int bestClass{-1};
    std::size_t outputValues{0};
};

class DynamicSegPipeline {
public:
    DynamicSegPipeline(
        const std::string& enginePath,
        int inputWidth,
        int inputHeight,
        const std::string& inputName,
        const std::string& primaryOutputName);

    std::vector<SegmentationResult> process(
        const std::vector<CameraFrame>& readyFrames);

private:
    SegmentationResult processOne(const CameraFrame& frame);
    static void basicPostprocess(
        const std::vector<float>& output,
        SegmentationResult& result);

    Yolo26Seg model_;
    std::string primaryOutputName_;
};

} // namespace y26
