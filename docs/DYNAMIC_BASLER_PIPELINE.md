# Config-driven Basler -> CUDA -> TensorRT segmentation pipeline

The camera config defines the TensorRT batch size and the permanent mapping between camera serial numbers and batch slots.

## Camera config

Example `configs/basler_cameras.example.txt`:

```text
40123456
40123457
40123458
40123459
```

This means:

```text
batch_size = 4
batch[0] = camera 40123456
batch[1] = camera 40123457
batch[2] = camera 40123458
batch[3] = camera 40123459
```

The runtime requires all configured cameras to be present at startup. The order in the config is preserved all the way through preprocessing, TensorRT and postprocess.

## Data flow

```text
configured Cam0 -> latest frame -> batch slot 0 --\
configured Cam1 -> latest frame -> batch slot 1 ---\
configured Cam2 -> latest frame -> batch slot 2 ----> async H2D
configured Cam3 -> latest frame -> batch slot 3 ---/       |
                                                         GPU
                                                          |
                                  CUDA preprocess into [N,3,H,W]
                                  letterbox + BGR->RGB
                                  normalize + HWC->CHW
                                                          |
                                                          v
                                               one TensorRT enqueueV2()
                                                          |
                                                          v
                                                batched model outputs
                                                          |
                                                          v
                                             split result by batch slot
```

`GetBuffer()` does not copy. `CameraFrame::owner` retains the `CGrabResultPtr` until the batch upload/inference path has consumed the host data.

## Scheduling policy

Each configured camera owns exactly one slot. Capture uses `GrabStrategy_LatestImageOnly`; while waiting for the slowest camera, a faster camera replaces its cached slot with its newest frame instead of building an unbounded queue.

A batch is submitted only when every configured slot has received a fresh frame. After inference all slots are marked non-fresh, so the next batch requires one new frame from every camera.

This deliberately makes throughput bounded by the slowest configured camera. A timeout/reuse/invalid-slot policy can be added later if lower latency is more important than one-new-frame-per-camera semantics.

## Engine batch must match config

The ONNX/TensorRT engine must be exported with the same static batch size as the number of configured cameras. For four cameras:

```bash
python scripts/export_onnx.py \
  --weights yolo26n-seg.pt \
  --imgsz 512 \
  --batch 4 \
  --output models/yolo26n-seg-b4.onnx

./scripts/build_engine.sh \
  models/yolo26n-seg-b4.onnx \
  models/yolo26n-seg-b4.engine
```

At runtime `Yolo26Seg` requests `[4,3,512,512]`. A static batch-1 engine will fail early instead of silently running four sequential inferences.

## Build

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
  models/yolo26n-seg-b4.engine \
  512 512 images output0 \
  configs/basler_cameras.txt
```

Use the actual input/output binding names reported by the engine.

## GPU preprocessing

Each host frame is copied asynchronously into a shared CUDA staging allocation. `launchPreprocessBatchSlot()` writes directly into its destination NCHW batch offset:

```text
slot 0 -> input + 0 * (3*H*W)
slot 1 -> input + 1 * (3*H*W)
...
```

There is no intermediate per-camera normalized tensor and only one TensorRT enqueue for the full configured batch.

## Pixel format

The sample currently assumes tightly packed `BGR8`, therefore `stride = width * 3`. Production code should explicitly configure/check Basler `PixelFormat` and actual row stride/padding.

## Postprocess status

The current postprocess remains intentionally model-agnostic. It downloads the configured FP32 primary output, divides the first/batch dimension into equal per-camera slices, and reports a maximum value/index for each slot.

It does not pretend to implement YOLO26 segmentation semantics before the exact engine output contract is known. Real postprocess still needs detection decode, confidence filtering, NMS, mask coefficients, prototype reconstruction, crop/resize and thresholding.

## Next optimization steps

1. Configure/check BGR8 and stride through pylon.
2. Register/use pinned host capture buffers where the transport path permits it.
3. Give camera uploads separate CUDA streams if profiling shows H2D serialization matters.
4. Move YOLO decode/NMS/mask reconstruction to CUDA.
5. Add timestamp skew metrics between batch slots.
6. Add a configurable missing-camera deadline policy if waiting for every fresh frame is too costly.
