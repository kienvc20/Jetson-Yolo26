#include "yolo26seg/dynamic_pipeline.hpp"

#include <algorithm>
#include <stdexcept>

namespace y26 {

DynamicSegPipeline::DynamicSegPipeline(
    const std::string& enginePath,
    int inputWidth,
    int inputHeight,
    const std::string& inputName,
    const std::string& primaryOutputName)
    : model_(enginePath, inputWidth, inputHeight, inputName),
      primaryOutputName_(primaryOutputName) {}

std::vector<SegmentationResult> DynamicSegPipeline::process(
    const std::vector<CameraFrame>& readyFrames) {
    std::vector<SegmentationResult> results;
    results.reserve(readyFrames.size());

    // Dynamic camera set: the caller passes only cameras that produced a fresh
    // frame in this scheduling cycle. A slow/disconnected camera does not block
    // the others. The current TensorRT runtime is batch-1, so frames share one
    // model instance and are scheduled sequentially. This boundary is designed
    // to be replaced by true dynamic TensorRT batching once the engine is built
    // with a dynamic N optimization profile.
    for (const CameraFrame& frame : readyFrames) {
        if (frame.bgr == nullptr) {
            continue;
        }
        results.push_back(processOne(frame));
    }
    return results;
}

SegmentationResult DynamicSegPipeline::processOne(const CameraFrame& frame) {
    if (frame.width <= 0 || frame.height <= 0 || frame.stride < frame.width * 3) {
        throw std::invalid_argument("Invalid camera frame");
    }

    const InferenceStats stats = model_.infer(
        frame.bgr, frame.width, frame.height, frame.stride);

    const std::vector<float> output =
        model_.downloadFloatOutput(primaryOutputName_);

    SegmentationResult result;
    result.cameraId = frame.cameraId;
    result.sequence = frame.sequence;
    result.preprocessMs = stats.preprocessMilliseconds;
    result.inferenceMs = stats.inferenceMilliseconds;
    result.outputValues = output.size();
    basicPostprocess(output, result);
    return result;
}

void DynamicSegPipeline::basicPostprocess(
    const std::vector<float>& output,
    SegmentationResult& result) {
    // Intentionally model-agnostic smoke-test postprocess. YOLO26-seg output
    // layout must be inspected before implementing box/NMS/prototype masks.
    // For now report the largest finite-ish score-like value and its index.
    if (output.empty()) {
        return;
    }

    const auto it = std::max_element(output.begin(), output.end());
    result.bestScore = *it;
    result.bestClass = static_cast<int>(std::distance(output.begin(), it));
}

} // namespace y26
