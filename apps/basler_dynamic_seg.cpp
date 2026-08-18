#include "yolo26seg/dynamic_pipeline.hpp"

#include <pylon/PylonIncludes.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::uint64_t nowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct CameraSlot {
    std::string id;
    std::unique_ptr<Pylon::CInstantCamera> camera;
    std::uint64_t sequence{0};
};

std::vector<CameraSlot> discoverAndOpen() {
    Pylon::DeviceInfoList_t devices;
    Pylon::CTlFactory::GetInstance().EnumerateDevices(devices);

    std::vector<CameraSlot> slots;
    slots.reserve(devices.size());
    for (const auto& info : devices) {
        CameraSlot slot;
        slot.id = info.GetSerialNumber().c_str();
        slot.camera.reset(new Pylon::CInstantCamera(
            Pylon::CTlFactory::GetInstance().CreateDevice(info)));
        slot.camera->Open();
        slot.camera->StartGrabbing(
            Pylon::GrabStrategy_LatestImageOnly,
            Pylon::GrabLoop_ProvidedByUser);
        std::cout << "[camera] online serial=" << slot.id << '\n';
        slots.push_back(std::move(slot));
    }
    return slots;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: basler_dynamic_seg ENGINE W H INPUT OUTPUT\n";
        return 2;
    }

    Pylon::PylonInitialize();
    try {
        y26::DynamicSegPipeline pipeline(
            argv[1], std::stoi(argv[2]), std::stoi(argv[3]), argv[4], argv[5]);

        auto cameras = discoverAndOpen();
        if (cameras.empty()) {
            throw std::runtime_error("No Basler cameras found");
        }

        while (true) {
            std::vector<y26::CameraFrame> ready;
            ready.reserve(cameras.size());

            for (CameraSlot& slot : cameras) {
                if (!slot.camera->IsGrabbing()) {
                    continue;
                }

                Pylon::CGrabResultPtr grab;
                if (!slot.camera->RetrieveResult(
                        0, grab, Pylon::TimeoutHandling_Return) ||
                    !grab || !grab->GrabSucceeded()) {
                    continue; // this camera is simply absent from this cycle
                }

                // The shared owner retains a CGrabResultPtr. GetBuffer() itself
                // does not copy; the Pylon buffer remains valid while owner lives.
                auto owner = std::make_shared<Pylon::CGrabResultPtr>(grab);

                y26::CameraFrame frame;
                frame.cameraId = slot.id;
                frame.sequence = ++slot.sequence;
                frame.timestampNs = nowNs();
                frame.bgr = static_cast<const std::uint8_t*>(grab->GetBuffer());
                frame.width = static_cast<int>(grab->GetWidth());
                frame.height = static_cast<int>(grab->GetHeight());
                frame.stride = frame.width * 3; // requires camera PixelFormat=BGR8
                frame.owner = owner;
                ready.push_back(std::move(frame));
            }

            if (ready.empty()) {
                continue;
            }

            const auto results = pipeline.process(ready);
            for (const auto& r : results) {
                std::cout << "camera=" << r.cameraId
                          << " seq=" << r.sequence
                          << " pre=" << r.preprocessMs << "ms"
                          << " infer=" << r.inferenceMs << "ms"
                          << " output_values=" << r.outputValues
                          << " max=" << r.bestScore << '\n';
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
