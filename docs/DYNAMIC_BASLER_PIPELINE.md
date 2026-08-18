# Dynamic Basler -> CUDA -> TensorRT segmentation pipeline

This branch adds the first end-to-end camera skeleton without hiding the important ownership boundaries.

## Current data flow

```text
Basler cameras discovered at startup
        |
        | RetrieveResult(0, Return)
        v
CGrabResultPtr -- GetBuffer() --> Pylon host buffer (no GetBuffer copy)
        |                            |
        | shared owner              | cudaMemcpyAsync
        |                            v
        +----------------------> CUDA device buffer
                                     |
                                     v
                         fused CUDA preprocess
                         BGR8 -> letterbox -> RGB
                         -> normalize -> NCHW FP16/FP32
                                     |
                                     v
                              TensorRT model
                                     |
                                     v
                         basic CPU postprocess
```

## Dynamic camera behavior

Every scheduling cycle polls every active camera with a zero timeout. Only cameras with a fresh successful grab are placed in `readyFrames`. A slow camera therefore does not block a fast camera.

This is *dynamic camera scheduling*, not yet true TensorRT dynamic batching. The repository's current engine contract is batch 1. `DynamicSegPipeline` intentionally isolates the batching boundary so the next step can replace the per-frame loop with a `[N,3,H,W]` batch once the engine is exported/built with a dynamic `N` optimization profile.

## Pixel format

The sample expects the Basler camera to deliver `BGR8`. `frame.stride = width * 3` is therefore deliberate and must be replaced with the actual stride/padding if the selected camera format differs. Production code should configure/check `PixelFormat` explicitly rather than assuming it.

## Lifetime

Never queue only `grab->GetBuffer()`. The pointer belongs to a Pylon grab buffer. `CameraFrame::owner` holds a `shared_ptr<CGrabResultPtr>` so the grab result remains referenced until GPU upload/inference has consumed the host data.

## Build

Install CUDA/TensorRT and Basler pylon, then:

```bash
cmake -S . -B build \
  -DY26_BUILD_APP=OFF \
  -DY26_BUILD_PYTHON=OFF \
  -DY26_BUILD_BASLER=ON
cmake --build build -j2
```

Run:

```bash
./build/basler_dynamic_seg \
  models/yolo26n-seg.engine \
  512 512 images output0
```

Use the actual input/output binding names reported by your engine.

## Postprocess status

The current postprocess is intentionally only a smoke test: it downloads one FP32 output and reports its maximum value/index. It is not pretending that an unknown output index is a YOLO class.

Before implementing real YOLO26 segmentation postprocess, inspect the exact engine outputs and lock down:

- detection tensor layout
- class score layout
- mask coefficient layout
- prototype mask tensor layout
- whether NMS is inside or outside the engine

Then implement decode -> confidence filtering -> NMS -> mask coefficient x prototypes -> crop/resize/threshold.

## Next optimization milestone

1. Configure BGR8 and actual stride explicitly through pylon.
2. Use registered/pinned host buffers where the pylon transport path permits it.
3. Build TensorRT engine with dynamic batch `N=1..max_cameras`.
4. Preallocate one maximum-size input tensor.
5. Launch CUDA preprocessing into batch offsets.
6. Set TensorRT binding dimensions to current ready-frame count.
7. Enqueue one inference for the whole ready batch.
8. Move YOLO decode and mask reconstruction to CUDA.
9. Add bounded latest-frame queues/hotplug rediscovery if runtime camera attach/detach is required.
