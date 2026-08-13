# YOLO26-seg on Jetson Nano

A practice-first C++/CUDA/TensorRT runtime for deploying one `yolo26n-seg`
model on the original NVIDIA Jetson Nano 4 GB.

The target is deliberately JetPack 4.6 / CUDA 10.2 / TensorRT 8.2. The code
therefore uses TensorRT's binding API and `enqueueV2()`, rather than newer
TensorRT 10/11 APIs.

Repository initialization is in progress.