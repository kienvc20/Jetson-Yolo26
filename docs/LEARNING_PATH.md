# Learning path

This repository is meant to be read and extended in small, testable steps. Do
not start by writing a custom TensorRT plugin. First understand where time and
memory are spent in the ordinary runtime.

## Lesson 1: Understand the complete data path

Read these files first:

1. `include/yolo26seg/yolo26_seg.hpp`
2. `src/yolo26_seg.cpp`
3. `apps/infer_image.cpp`

Trace one image through the program:

```text
cv::Mat on host
  -> persistent CUDA source buffer
  -> fused CUDA preprocessing
  -> TensorRT input binding
  -> enqueueV2
  -> TensorRT output bindings
```

Questions to answer:

- Why is the model object constructed once rather than once per frame?
- Which allocations happen once, and which operations happen every frame?
- Why does `infer()` synchronize only after recording the inference event?
- Why is `enqueueV2()` asynchronous with respect to the CPU?

Exercise: add total end-to-end CPU timing around `model.infer()` and compare it
with the two CUDA event measurements.

## Lesson 2: Learn TensorRT 8.2 ownership

Read:

1. `include/yolo26seg/trt_engine.hpp`
2. `src/trt_engine.cpp`

The important objects are:

- `IRuntime`: deserializes an engine plan
- `ICudaEngine`: immutable optimized network
- `IExecutionContext`: dimensions and per-execution state
- binding array: device pointers ordered by binding index

The original Nano uses TensorRT 8.2, so this code deliberately uses binding
indices and `enqueueV2()`. New tutorials using `setTensorAddress()` and
`enqueueV3()` target newer TensorRT releases and cannot be copied unchanged.

Exercise: print each binding name, input/output role, data type, shape and byte
size. Then explain why a dynamic `-1` dimension must be resolved before memory
allocation.

## Lesson 3: Study fused preprocessing

Read:

1. `include/yolo26seg/preprocess.hpp`
2. `cuda/preprocess.cu`

One kernel performs:

- letterbox geometry
- bilinear sampling
- BGR to RGB
- uint8 to float/half
- division by 255
- HWC to NCHW

This is useful fusion: it removes intermediate resized images, channel-swapped
images and normalized tensors.

Exercise: implement a CPU reference in Python and compare the maximum absolute
error of the CUDA tensor. Pay special attention to odd letterbox padding and
half-pixel interpolation coordinates.

## Lesson 4: Understand the Python boundary

Read:

1. `python/bindings.cpp`
2. `examples/python_infer.py`

Python provides control and integration. C++/CUDA owns hot operations. Notice
that the binding releases the Python GIL while inference runs.

Exercise: run two Python worker threads with separate `Yolo26Seg` instances.
Measure whether they overlap and explain the memory cost of two TensorRT
execution contexts.

## Lesson 5: Record the real YOLO26-seg outputs

Before writing a decoder, build the engine on the Nano and save its binding
table. Determine for every output:

- name
- shape
- data type
- whether boxes are already decoded
- where mask coefficients are located
- prototype tensor layout
- whether NMS is inside or outside the graph

Do not infer these properties from an older YOLOv8 tutorial. Export choices can
change the output contract.

Exercise: run the same image in PyTorch and TensorRT, download raw outputs for
debugging, and compare selected values before implementing postprocessing.

## Lesson 6: Detection decoding

Implement a result structure containing:

```cpp
struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int classId;
};
```

Then implement:

1. confidence filtering
2. class extraction
3. reverse-letterbox coordinate mapping
4. clipping to the source image
5. NMS only if the exported output requires it

Start with a C++ decoder for correctness. Move it to CUDA only after profiling
shows it matters.

## Lesson 7: Instance-mask reconstruction

Typical prototype-based segmentation reconstructs masks using:

```text
selected mask coefficients x shared prototypes -> mask logits
sigmoid -> crop -> resize -> threshold
```

Use cuBLAS for the matrix multiplication first. A custom fused CUDA kernel is a
later optimization, not the first implementation.

Keep logits and masks on the GPU. Return compact polygons or RLE to Python when
possible instead of copying every full-resolution mask.

## Lesson 8: Profile before optimizing

Measure separately:

- camera acquisition
- host-to-device copy
- preprocessing
- TensorRT inference
- detection decode
- mask reconstruction
- device-to-host result copy
- rendering or network transmission

Only fuse or rewrite the stage that dominates real end-to-end latency. On the
original Nano, memory pressure and unnecessary copies can be more damaging than
small amounts of C++ control overhead.

## Suggested checkpoints

Create one commit after each of these outcomes:

1. engine loads and bindings print correctly
2. preprocessing matches the Python reference
3. raw TensorRT output matches PyTorch
4. boxes and classes match Ultralytics
5. masks match Ultralytics
6. Python receives compact results
7. live camera pipeline runs with bounded memory
8. benchmark report records FPS, median, P95 and memory use
