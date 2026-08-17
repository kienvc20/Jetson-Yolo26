#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ':' << __LINE__ << '\n';        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Input layout:  HWC uint8 RGB  [H][W][3]
// Output layout: CHW float32     [3][H][W]
//
// Each CUDA thread handles one pixel. This is intentionally simple: no resize,
// no letterbox and no batching yet, so the memory-layout change is easy to see.
__global__ void rgb_u8_to_chw_f32(
    const std::uint8_t* input,
    float* output,
    int width,
    int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int pixel = y * width + x;
    const int src = pixel * 3;
    const int plane = width * height;

    // RGB uint8 [0,255] -> float [0,1], while changing HWC -> CHW.
    output[0 * plane + pixel] = input[src + 0] / 255.0f;
    output[1 * plane + pixel] = input[src + 1] / 255.0f;
    output[2 * plane + pixel] = input[src + 2] / 255.0f;
}

int main()
{
    // Tiny 2x2 image so we can inspect every value by eye.
    constexpr int width = 2;
    constexpr int height = 2;
    constexpr int channels = 3;

    // Imagine this pointer came from Basler grabResult->GetBuffer().
    // Here a vector is used only to make the example runnable without a camera.
    std::vector<std::uint8_t> host_input = {
        255,   0,   0,   // red
          0, 255,   0,   // green
          0,   0, 255,   // blue
        255, 255, 255    // white
    };

    const std::size_t input_bytes = width * height * channels * sizeof(std::uint8_t);
    const std::size_t output_count = width * height * channels;
    const std::size_t output_bytes = output_count * sizeof(float);

    std::uint8_t* device_input = nullptr;
    float* device_output = nullptr;

    CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&device_output, output_bytes));

    // This IS a copy: host RAM -> GPU device memory.
    CUDA_CHECK(cudaMemcpy(
        device_input,
        host_input.data(),
        input_bytes,
        cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    rgb_u8_to_chw_f32<<<grid, block>>>(
        device_input,
        device_output,
        width,
        height);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host_output(output_count);

    // Copy back only for demonstration/verification. A real TensorRT pipeline
    // would normally keep device_output on the GPU and pass it downstream.
    CUDA_CHECK(cudaMemcpy(
        host_output.data(),
        device_output,
        output_bytes,
        cudaMemcpyDeviceToHost));

    const int plane = width * height;
    const char names[3] = {'R', 'G', 'B'};

    for (int c = 0; c < channels; ++c) {
        std::cout << names[c] << ": ";
        for (int i = 0; i < plane; ++i) {
            std::cout << host_output[c * plane + i] << ' ';
        }
        std::cout << '\n';
    }

    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_input));

    return 0;
}
