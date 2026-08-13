#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace y26 {

class TrtLogger final : public nvinfer1::ILogger {
public:
    explicit TrtLogger(Severity threshold = Severity::kWARNING)
        : threshold_(threshold) {}

    void log(Severity severity, const char* message) noexcept override;

private:
    Severity threshold_;
};

template <typename T>
struct TrtDestroy {
    void operator()(T* object) const noexcept {
        if (object != nullptr) {
            object->destroy();
        }
    }
};

struct BindingInfo {
    int index{-1};
    std::string name;
    bool isInput{false};
    nvinfer1::DataType dataType{nvinfer1::DataType::kFLOAT};
    nvinfer1::Dims dimensions{};
    std::size_t bytes{0};
};

class TrtEngine {
public:
    explicit TrtEngine(const std::string& enginePath);
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    int bindingIndex(const std::string& name) const;
    void setInputShape(const std::string& name, const nvinfer1::Dims& dimensions);
    void allocateBindings();
    bool enqueue(cudaStream_t stream);

    void* deviceBuffer(int index);
    void* deviceBuffer(const std::string& name);
    const std::vector<BindingInfo>& bindings() const noexcept { return bindings_; }

private:
    void releaseBindings() noexcept;
    void refreshBindingInfo();

    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy<nvinfer1::IExecutionContext>> context_;
    std::vector<void*> deviceBindings_;
    std::vector<BindingInfo> bindings_;
};

std::string dimensionsToString(const nvinfer1::Dims& dimensions);
const char* dataTypeName(nvinfer1::DataType type);

}  // namespace y26
