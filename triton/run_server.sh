#!/usr/bin/env bash
# Build the KaoAnime Triton image and serve the ensemble on onnxruntime (CPU).
#
# Steps: stage the chosen ONNX into the model repository, build the image, then
# run Triton. Pick the model to serve with MODEL_ONNX (a file under the model dir,
# e.g. MODEL_ONNX=NOT.onnx). The HTTP / gRPC / metrics ports default to
# 8000 / 8001 / 8002 and can be overridden, e.g. `HTTP_PORT=9000 ...`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_TAG="${TRITON_TAG:-26.05-py3}"
IMAGE="${IMAGE:-kaoanime-triton}"
MODEL_ONNX="${MODEL_ONNX:?set MODEL_ONNX to the ONNX file to serve, e.g. MODEL_ONNX=NOT.onnx}"
HTTP_PORT="${HTTP_PORT:-8000}"
GRPC_PORT="${GRPC_PORT:-8001}"
METRICS_PORT="${METRICS_PORT:-8002}"

uv run python "${REPO_ROOT}/triton/setup_model_repository.py" --onnx "${MODEL_ONNX}"

docker build \
  --build-arg "TRITON_TAG=${TRITON_TAG}" \
  -t "${IMAGE}" \
  "${REPO_ROOT}/triton"

docker run --rm \
  -p "${HTTP_PORT}:${HTTP_PORT}" \
  -p "${GRPC_PORT}:${GRPC_PORT}" \
  -p "${METRICS_PORT}:${METRICS_PORT}" \
  -v "${REPO_ROOT}/triton/model_repository:/models" \
  "${IMAGE}" \
  tritonserver --model-repository=/models \
    --http-port="${HTTP_PORT}" \
    --grpc-port="${GRPC_PORT}" \
    --metrics-port="${METRICS_PORT}"
