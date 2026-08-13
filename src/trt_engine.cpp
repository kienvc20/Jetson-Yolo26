#include "yolo26seg/trt_engine.hpp"

#include "yolo26seg/cuda_utils.hpp"

#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace y26 {
namespace {

std::vector<char> readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Could not open TensorRT engine: " + path);
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + path);
    }

    file.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    if (!file.read(bytes.data(), size)) {
        throw std::runtime_error("Could not read TensorRT engine: " + path);
    }
    return bytes;
}

std::size_t elementSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            throw std::runtime_error("Unsupported TensorRT binding data type");
    }
}

std::size_t volume(const nvinfer1::Dims& dimensions) {
    std::size_t result = 1;
    for (int index = 0; index < dimensions.nbDims; ++index) {
        if (dimensions.d[index] < 0) {
            throw std::runtime_error(
                "Cannot allocate a binding with unresolved dynamic dimensions: " +
                dimensionsToString(dimensions));
        }
        result *= static_cast<std::size_t>(dimensions.d[index]);
    }
    return result;
}

}  // namespace

void TrtLogger::log(Severity severity, const char* message) noexcept {
    if (severity <= threshold_) {
        std::cerr << "[TensorRT] " << message << '\n';
    }
}

TrtEngine::TrtEngine(const std::string& enginePath) {
    const std::vector<char> bytes = readBinaryFile(enginePath);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size(), nullptr));
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    deviceBindings_.resize(static_cast<std::size_t>(engine_->getNbBindings()), nullptr);
    refreshBindingInfo();
}

TrtEngine::~TrtEngine() {
    releaseBindings();
}

int TrtEngine::bindingIndex(const std::string& name) const {
    const int index = engine_->getBindingIndex(name.c_str());
    if (index < 0) {
        throw std::runtime_error("TensorRT binding not found: " + name);
    }
    return index;
}

void TrtEngine::setInputShape(
    const std::string& name,
    const nvinfer1::Dims& dimensions) {
    const int index = bindingIndex(name);
    if (!engine_->bindingIsInput(index)) {
        throw std::runtime_error("Binding is not an input: " + name);
    }
    const nvinfer1::Dims engineDimensions = engine_->getBindingDimensions(index);
    bool hasWildcard = false;
    for (int axis = 0; axis < engineDimensions.nbDims; ++axis) {
        hasWildcard = hasWildcard || engineDimensions.d[axis] < 0;
    }

    if (hasWildcard) {
        if (!context_->setBindingDimensions(index, dimensions)) {
            throw std::runtime_error(
                "TensorRT rejected input dimensions for " + name + ": " +
                dimensionsToString(dimensions));
        }
    } else {
        bool matches = engineDimensions.nbDims == dimensions.nbDims;
        for (int axis = 0; matches && axis < engineDimensions.nbDims; ++axis) {
            matches = engineDimensions.d[axis] == dimensions.d[axis];
        }
        if (!matches) {
            throw std::runtime_error(
                "Requested input shape " + dimensionsToString(dimensions) +
                " does not match static engine shape " +
                dimensionsToString(engineDimensions));
        }
    }
    refreshBindingInfo();
}

void TrtEngine::allocateBindings() {
    if (!context_->allInputDimensionsSpecified()) {
        throw std::runtime_error("Not all TensorRT input dimensions are specified");
    }

    releaseBindings();
    refreshBindingInfo();

    for (BindingInfo& binding : bindings_) {
        binding.bytes = volume(binding.dimensions) * elementSize(binding.dataType);
        if (binding.bytes > 0) {
            Y26_CUDA_CHECK(cudaMalloc(&deviceBindings_[binding.index], binding.bytes));
        }
    }
}

bool TrtEngine::enqueue(cudaStream_t stream) {
    return context_->enqueueV2(deviceBindings_.data(), stream, nullptr);
}

void* TrtEngine::deviceBuffer(int index) {
    if (index < 0 || index >= static_cast<int>(deviceBindings_.size())) {
        throw std::out_of_range("TensorRT binding index is out of range");
    }
    return deviceBindings_[static_cast<std::size_t>(index)];
}

void* TrtEngine::deviceBuffer(const std::string& name) {
    return deviceBuffer(bindingIndex(name));
}

void TrtEngine::releaseBindings() noexcept {
    for (void*& pointer : deviceBindings_) {
        if (pointer != nullptr) {
            cudaFree(pointer);
            pointer = nullptr;
        }
    }
}

void TrtEngine::refreshBindingInfo() {
    bindings_.clear();
    bindings_.reserve(static_cast<std::size_t>(engine_->getNbBindings()));

    for (int index = 0; index < engine_->getNbBindings(); ++index) {
        BindingInfo info;
        info.index = index;
        info.name = engine_->getBindingName(index);
        info.isInput = engine_->bindingIsInput(index);
        info.dataType = engine_->getBindingDataType(index);
        info.dimensions = context_->getBindingDimensions(index);
        bindings_.push_back(info);
    }
}

std::string dimensionsToString(const nvinfer1::Dims& dimensions) {
    std::ostringstream stream;
    stream << '[';
    for (int index = 0; index < dimensions.nbDims; ++index) {
        if (index != 0) {
            stream << 'x';
        }
        stream << dimensions.d[index];
    }
    stream << ']';
    return stream.str();
}

const char* dataTypeName(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return "FP32";
        case nvinfer1::DataType::kHALF:
            return "FP16";
        case nvinfer1::DataType::kINT8:
            return "INT8";
        case nvinfer1::DataType::kINT32:
            return "INT32";
        case nvinfer1::DataType::kBOOL:
            return "BOOL";
        default:
            return "UNKNOWN";
    }
}

}  // namespace y26
