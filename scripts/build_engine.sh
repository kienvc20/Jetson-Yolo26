#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 MODEL.onnx MODEL.engine" >&2
    exit 2
fi

onnx_path=$1
engine_path=$2
trtexec_path=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

if [[ ! -f "$onnx_path" ]]; then
    echo "ONNX model not found: $onnx_path" >&2
    exit 1
fi

if [[ ! -x "$trtexec_path" ]]; then
    echo "trtexec not found or not executable: $trtexec_path" >&2
    exit 1
fi

mkdir -p "$(dirname "$engine_path")"

"$trtexec_path" \
    --onnx="$onnx_path" \
    --saveEngine="$engine_path" \
    --fp16 \
    --workspace=1024 \
    --verbose

echo "Built TensorRT engine: $engine_path"
