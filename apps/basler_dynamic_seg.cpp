#include "yolo26seg/dynamic_pipeline.hpp"

#include <pylon/PylonIncludes.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint64_t nowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::vector<std::string> loadCameraIds(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Could not open camera config: " + path);

    std::vector<std::string> ids;
    std::string line;
    while (std::getline(file, line)) {
        const std::size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        const std::size_t last = line.find_last_not_of(" \t\r\n");
        ids.push_back(line.substr(first, last - first + 1));
    }
    if (ids.empty()) throw std::runtime_error("Camera config contains no serial numbers");
    return ids;
}

struct CameraSlot {
    std::string id;
    std::unique_ptr<Pylon::CInstantCamera> camera;
    std::uint64_t sequence{0};
    y26::CameraFrame latest;
    bool fresh{false};
};

std::vector<CameraSlot> openConfigured(const std::vector<std::string>& configuredIds) {
    Pylon::DeviceInfoList_t devices;
    Pylon::CTlFactory::GetInstance().EnumerateDevices(devices);

    std::vector<CameraSlot> slots;
    slots.reserve(configuredIds.size());

    for (const std::string& requestedId : configuredIds) {
        const Pylon::CDeviceInfo* match = nullptr;
        for (const auto& info : devices) {
            if (requestedId == info.GetSerialNumber().c_str()) {
                match = &info;
                break;
            }
        }
        if (match == nullptr) {
            throw std::runtime_error("Configured Basler camera not found: " + requestedId);
        }

        CameraSlot slot;
        slot.id = requestedId;
        slot.camera.reset(new Pylon::CInstantCamera(
            Pylon::CTlFactory::GetInstance().CreateDevice(*match)));
        slot.camera->Open();
        slot.camera->StartGrabbing(
            Pylon::GrabStrategy_LatestImageOnly,
            Pylon::GrabLoop_ProvidedByUser);
        std::cout << "[camera] slot=" << slots.size()
                  << " serial=" << slot.id << " online\n";
        slots.push_back(std::move(slot));
    }
    return slots;
}

bool allFresh(const std::vector<CameraSlot>& cameras) {
    for (const CameraSlot& slot : cameras) {
        if (!slot.fresh) return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "usage: basler_dynamic_seg ENGINE W H INPUT OUTPUT CAMERA_CONFIG\n";
        return 2;
    }

    Pylon::PylonInitialize();
    try {
        const std::vector<std::string> cameraIds = loadCameraIds(argv[6]);
        std::cout << "[batch] configured batch_size=" << cameraIds.size() << '\n';

        y26::DynamicSegPipeline pipeline(
            argv[1], std::stoi(argv[2]), std::stoi(argv[3]),
            argv[4], argv[5], cameraIds);

        auto cameras = openConfigured(cameraIds);

        while (true) {
            for (CameraSlot& slot : cameras) {
                if (!slot.camera->IsGrabbing()) {
                    throw std::runtime_error("Configured camera stopped grabbing: " + slot.id);
                }

                Pylon::CGrabResultPtr grab;
                if (!slot.camera->RetrieveResult(
                        0, grab, Pylon::TimeoutHandling_Return) ||
                    !grab || !grab->GrabSucceeded()) {
                    continue;
                }

                auto owner = std::make_shared<Pylon::CGrabResultPtr>(grab);
                y26::CameraFrame frame;
                frame.cameraId = slot.id;
                frame.sequence = ++slot.sequence;
                frame.timestampNs = nowNs();
                frame.bgr = static_cast<const std::uint8_t*>(grab->GetBuffer());
                frame.width = static_cast<int>(grab->GetWidth());
                frame.height = static_cast<int>(grab->GetHeight());
                frame.stride = frame.width * 3; // sample contract: tightly packed BGR8
                frame.owner = owner;

                // LatestImageOnly + replacement means a fast camera does not build
                // an unbounded queue while waiting for the slowest configured slot.
                slot.latest = std::move(frame);
                slot.fresh = true;
            }

            if (!allFresh(cameras)) continue;

            std::vector<y26::CameraFrame> batch;
            batch.reserve(cameras.size());
            for (CameraSlot& slot : cameras) {
                batch.push_back(slot.latest); // config order == TensorRT batch order
            }

            const auto results = pipeline.processFixedBatch(batch);
            for (const auto& r : results) {
                std::cout << "camera=" << r.cameraId
                          << " seq=" << r.sequence
                          << " batch_pre=" << r.preprocessMs << "ms"
                          << " batch_infer=" << r.inferenceMs << "ms"
                          << " output_values=" << r.outputValues
                          << " max=" << r.bestScore
                          << " best_index=" << r.bestIndex << '\n';
            }

            // Require a new frame from every configured camera before next batch.
            for (CameraSlot& slot : cameras) {
                slot.fresh = false;
                slot.latest = y26::CameraFrame{};
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        Pylon::PylonTerminate();
        return 1;
    }

    Pylon::PylonTerminate();
    return 0;
}
