#!/usr/bin/env bash
# Convert an ONNX model to a TensorRT engine via trtexec.
# Requires TensorRT (trtexec) installed on the target machine.
#
# Usage: bash scripts/export_tensorrt.sh <model.onnx> <model.engine>
set -euo pipefail

ONNX="${1:?usage: export_tensorrt.sh <model.onnx> <model.engine>}"
ENGINE="${2:?usage: export_tensorrt.sh <model.onnx> <model.engine>}"

trtexec \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    --minShapes=input:1x3x128x128 \
    --optShapes=input:8x3x128x128 \
    --maxShapes=input:16x3x128x128
