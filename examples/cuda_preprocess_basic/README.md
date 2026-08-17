# Basic CUDA Preprocess

A deliberately small CUDA C++ example for learning the path from a camera-style host buffer to a GPU tensor.

## What it does

The example starts with a tiny `2 x 2` RGB `uint8` image in host RAM, copies it to CUDA device memory, then runs one CUDA kernel that:

1. maps one GPU thread to one pixel,
2. normalizes `uint8 [0,255]` to `float32 [0,1]`, and
3. converts image layout from HWC to CHW.

Conceptually:

```text
Basler GetBuffer()                 this example
       |                               |
       v                               v
host uint8 RGB pointer          host_input.data()
       |                               |
       +---------- H2D copy -----------+
                       |
                       v
                 CUDA device RAM
                       |
                       v
             rgb_u8_to_chw_f32
                       |
                       v
              float32 CHW tensor
                       |
                       v
                 TensorRT later
```

`GetBuffer()` itself does not copy a Basler frame. The host-to-device transfer is the copy that occurs when the frame is uploaded to a discrete GPU.

## Build

Requires the CUDA Toolkit and a CUDA-capable NVIDIA GPU.

```bash
mkdir build
cd build
cmake ..
cmake --build . -j
./cuda_preprocess_basic
```

Expected output is approximately:

```text
R: 1 0 0 1
G: 0 1 0 1
B: 0 0 1 1
```

## Read the code in this order

Start at `main()`. Follow `host_input` -> `cudaMalloc` -> `cudaMemcpy` -> kernel launch. Then read `rgb_u8_to_chw_f32()` and focus on how `threadIdx`/`blockIdx` select a pixel and how the output index changes HWC into CHW.

## Next exercises

Do these one at a time rather than hiding them behind a framework:

1. Replace the fake `host_input` with a real Basler `grabResult->GetBuffer()` pointer.
2. Change synchronous `cudaMemcpy` to pinned memory + `cudaMemcpyAsync`.
3. Add bilinear resize.
4. Fuse resize + normalization + HWC-to-CHW in one kernel.
5. Add a batch dimension for multiple cameras.
6. Write FP16 output directly for a TensorRT FP16 engine.

The example intentionally does not use OpenCV, GStreamer, DeepStream or TensorRT yet. Its purpose is to make the CPU pointer -> GPU memory -> CUDA kernel -> tensor path explicit before adding those layers.
