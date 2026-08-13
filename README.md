# YOLO26-seg on Jetson Nano

A practice-first C++/CUDA/TensorRT runtime for deploying one `yolo26n-seg`
model on the original NVIDIA Jetson Nano 4 GB.

The target is deliberately JetPack 4.6 / CUDA 10.2 / TensorRT 8.2. The code
therefore uses TensorRT's binding API and `enqueueV2()`, rather than newer
TensorRT 10/11 APIs.

## Current milestone

Implemented:

- TensorRT 8.2 engine deserialization and binding inspection
- persistent input/output GPU allocations
- fixed-shape batch-1 inference with `enqueueV2()`
- fused CUDA letterbox, BGR-to-RGB, normalization and HWC-to-CHW preprocessing
- C++ image CLI
- optional pybind11 wrapper
- PyTorch-to-ONNX export and on-device TensorRT build scripts

Next:

- lock the exact YOLO26-seg output contract from an exported engine
- CUDA box and class decoding
- mask coefficient x prototype reconstruction
- crop, resize and threshold masks on GPU
- compact polygon/RLE results for Python
- GStreamer/NVMM camera input

The mask decoder is intentionally not guessed. Run the inspector on the actual
engine first and record its output names, types and dimensions.

## Target

- Original Jetson Nano 4 GB, compute capability 5.3
- JetPack 4.6.x
- CUDA 10.2
- TensorRT 8.2.x
- static input, batch 1
- FP16 engine

For an Orin Nano, use a separate branch because its JetPack, CUDA, TensorRT and
GPU architecture are different.

## 1. Export ONNX on a development PC

Do not install the current training stack on the Nano just to export the model.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U ultralytics onnx onnxsim

python scripts/export_onnx.py \
  --weights yolo26n-seg.pt \
  --imgsz 512 \
  --opset 13 \
  --output models/yolo26n-seg-512.onnx
```

If TensorRT 8.2 rejects an operator, keep the complete parser error. Do not add
a plugin until the graph and its unsupported node are known.

## 2. Build the engine on the Jetson Nano

TensorRT plan files are device- and runtime-specific. Copy the ONNX file to the
Nano and build there:

```bash
./scripts/build_engine.sh \
  models/yolo26n-seg-512.onnx \
  models/yolo26n-seg-512-fp16.engine
```

The script uses `trtexec --fp16 --workspace=1024`. Close memory-heavy programs
before building the engine.

## 3. Build the C++ runtime on Nano

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev python3-dev pybind11-dev

cmake -S . -B build \
  -DY26_BUILD_APP=ON \
  -DY26_BUILD_PYTHON=ON

cmake --build build -j2
```

Use `-j2`, not all CPU cores, to avoid unnecessary memory pressure.

## 4. Inspect and benchmark one image

```bash
./build/yolo26seg_cli \
  models/yolo26n-seg-512-fp16.engine \
  testdata/image.jpg \
  512 512 images
```

The final argument is the input binding name. Change it if engine inspection
reports a different name.

## 5. Python wrapper

```bash
PYTHONPATH=build python3 examples/python_infer.py \
  --engine models/yolo26n-seg-512-fp16.engine \
  --image testdata/image.jpg
```

Python is the control plane. The C++/CUDA library owns preprocessing, TensorRT
execution and eventually mask decoding. Avoid returning raw full-resolution
masks through Python unless they are truly needed there.

## Performance rules

- Allocate CUDA buffers once; never allocate per frame.
- Use one fixed input shape first.
- Keep preprocessing and mask reconstruction on the GPU.
- Return boxes, class IDs and compact polygons/RLE to Python.
- Build and benchmark the TensorRT engine on the target Nano.
- Measure end-to-end latency, not only TensorRT execution.
- Prefer dropping stale camera frames over creating an unbounded queue.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the component boundary and
planned milestones.
