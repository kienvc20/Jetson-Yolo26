#include "yolo26seg/trt_engine.hpp"
#include "yolo26seg/yolo26_seg.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void printUsage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " ENGINE IMAGE INPUT_WIDTH INPUT_HEIGHT [INPUT_NAME]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        const std::string enginePath = argv[1];
        const std::string imagePath = argv[2];
        const int inputWidth = std::stoi(argv[3]);
        const int inputHeight = std::stoi(argv[4]);
        const std::string inputName = argc == 6 ? argv[5] : "images";

        const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Could not read image: " + imagePath);
        }

        y26::Yolo26Seg model(
            enginePath,
            inputWidth,
            inputHeight,
            inputName);

        std::cout << "Bindings:\n";
        for (const y26::BindingInfo& binding : model.bindings()) {
            std::cout
                << "  [" << binding.index << "] "
                << (binding.isInput ? "input  " : "output ")
                << binding.name << " "
                << y26::dataTypeName(binding.dataType) << " "
                << y26::dimensionsToString(binding.dimensions) << " "
                << binding.bytes << " bytes\n";
        }

        constexpr int warmups = 5;
        for (int iteration = 0; iteration < warmups; ++iteration) {
            model.infer(
                image.data,
                image.cols,
                image.rows,
                static_cast<int>(image.step));
        }

        constexpr int runs = 20;
        float preprocessingTotal = 0.0F;
        float inferenceTotal = 0.0F;
        for (int iteration = 0; iteration < runs; ++iteration) {
            const y26::InferenceStats stats = model.infer(
                image.data,
                image.cols,
                image.rows,
                static_cast<int>(image.step));
            preprocessingTotal += stats.preprocessMilliseconds;
            inferenceTotal += stats.inferenceMilliseconds;
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "Average CUDA preprocess: "
                  << preprocessingTotal / runs << " ms\n"
                  << "Average TensorRT inference: "
                  << inferenceTotal / runs << " ms\n";

        const y26::LetterboxTransform transform = model.lastTransform();
        std::cout << "Letterbox: scale=" << transform.scale
                  << " padX=" << transform.padX
                  << " padY=" << transform.padY << '\n';

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
