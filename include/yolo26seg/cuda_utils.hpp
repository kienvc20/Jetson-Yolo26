#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

#define Y26_CUDA_CHECK(call)                                                    \
    do {                                                                        \
        const cudaError_t y26_cuda_error = (call);                              \
        if (y26_cuda_error != cudaSuccess) {                                    \
            std::ostringstream y26_cuda_message;                               \
            y26_cuda_message << "CUDA error at " << __FILE__ << ':' << __LINE__ \
                             << ": " << cudaGetErrorString(y26_cuda_error);     \
            throw std::runtime_error(y26_cuda_message.str());                   \
        }                                                                       \
    } while (false)
