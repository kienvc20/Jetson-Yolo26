#include "yolo26seg/dynamic_pipeline.hpp"

#include <algorithm>
#include <stdexcept>

namespace y26 {

DynamicSegPipeline::DynamicSegPipeline(
    const std::string& enginePath,
    int inputWidth,
    int inputHeight,
    const std::string& inputName,
    const std::string& primaryOutputName,
    const std::vector<std::string>& configuredCameraIds)
    : cameraIds_(configuredCameraIds),
      model_(enginePath, inputWidth, inputHeight, inputName,
             static_cast<int>(configuredCameraIds.size())),
      primaryOutputName_(primaryOutputName) {
    if (cameraIds_.empty()) {
        throw std::invalid_argument("At least one configured camera is required");
    }
}

std::vector<SegmentationResult> DynamicSegPipeline::processFixedBatch(
    const std::vector<CameraFrame>& frames) {
    if (frames.size() != cameraIds_.size()) {
        throw std::invalid_argument("Fixed batch must contain exactly one frame per configured camera");
    }

    std::vector<BatchImageView> images;
    images.reserve(frames.size());
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const CameraFrame& frame = frames[i];
        if (frame.cameraId != cameraIds_[i]) {
            throw std::invalid_argument("Camera frame order does not match configured batch slots");
        }
        images.push_back(BatchImageView{
            frame.bgr, frame.width, frame.height, frame.stride});
    }

    const InferenceStats stats = model_.inferBatch(images);
    const std::vector<float> output = model_.downloadFloatOutput(primaryOutputName_);

    if (output.size() % frames.size() != 0) {
        throw std::runtime_error("Primary output cannot be evenly split across fixed camera batch");
    }

    const std::size_t valuesPerImage = output.size() / frames.size();
    std::vector<SegmentationResult> results;
    results.reserve(frames.size());

    for (std::size_t i = 0; i < frames.size(); ++i) {
        SegmentationResult result;
        result.cameraId = frames[i].cameraId;
        result.sequence = frames[i].sequence;
        result.preprocessMs = stats.preprocessMilliseconds;
        result.inferenceMs = stats.inferenceMilliseconds;
        result.outputValues = valuesPerImage;

        const float* begin = output.data() + i * valuesPerImage;
        basicPostprocess(begin, begin + valuesPerImage, result);
        results.push_back(result);
    }
    return results;
}

void DynamicSegPipeline::basicPostprocess(
    const float* begin,
    const float* end,
    SegmentationResult& result) {
    if (begin == end) return;
    const float* it = std::max_element(begin, end);
    result.bestScore = *it;
    result.bestIndex = static_cast<int>(it - begin);
}

} // namespace y26
