# Architecture

## Boundary between Python and C++

Python owns configuration, business rules, logging and network integrations.
C++/CUDA owns every per-frame, per-pixel or GPU-sensitive operation.

```text
Python application
    |
    | infer(HWC uint8 frame)
    v
pybind11 boundary
    |
    v
C++ Yolo26Seg
    +-- persistent CUDA buffers
    +-- CUDA fused preprocess
    +-- TensorRT 8.2 enqueueV2
    +-- CUDA detection decode       [next milestone]
    +-- CUDA mask reconstruction    [next milestone]
    |
    v
compact detections + polygons/RLE
```

For the first working version, Python/OpenCV may own camera capture. In the
production version, C++ GStreamer should own capture so frames can stay in
NVMM/CUDA memory.

## Why one model

`yolo26n-seg` already produces detection classes, boxes and instance masks from
one backbone. Running FastSAM beside it would duplicate feature extraction and
put unnecessary pressure on the Nano's 4 GB shared memory.

## Milestones

### M1: Runtime skeleton

- deserialize engine
- inspect bindings
- set fixed input dimensions
- allocate persistent buffers
- execute and time inference

### M2: Correct preprocessing

- letterbox matching Ultralytics geometry
- BGR to RGB
- uint8 to FP32/FP16
- normalization and HWC to CHW
- numeric comparison with Python preprocessing

### M3: Output contract

Record the actual output binding names, dimensions, types and semantics for the
chosen YOLO26-seg export. Store a small golden-output fixture without committing
proprietary model weights.

### M4: Detection decode

- confidence filtering
- class extraction
- reverse letterbox coordinates
- confirm whether the selected export is end-to-end or needs NMS

### M5: Mask decode

- coefficient/prototype matrix multiplication with cuBLAS
- sigmoid
- crop in model coordinates
- resize to source resolution
- threshold or polygon/RLE encoding

### M6: Camera pipeline

- bounded queue
- GStreamer/NVMM capture
- one inference worker
- drop stale frames
- optional overlay without host round trip

## TensorRT version constraint

JetPack 4.6 ships TensorRT 8.2, so the runtime uses:

- binding indices
- `setBindingDimensions()`
- `enqueueV2()`
- `IPluginV2DynamicExt` if a custom plugin becomes necessary

Do not paste TensorRT 10/11 `setTensorAddress()` or `enqueueV3()` code into this
branch.
