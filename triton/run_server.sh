#!/usr/bin/env bash
# Build the KaoAnime Triton image and serve the ensemble on onnxruntime (CPU).
#
# Steps: populate the model repository (download + copy ONNX), build the image,
# then run Triton. The HTTP / gRPC / metrics ports default to 8000 / 8001 / 8002
# and can be overridden at launch, e.g. `HTTP_PORT=9000 bash triton/run_server.sh`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_TAG="${TRITON_TAG:-26.05-py3}"
IMAGE="${IMAGE:-kaoanime-triton}"
HTTP_PORT="${HTTP_PORT:-8000}"
GRPC_PORT="${GRPC_PORT:-8001}"
METRICS_PORT="${METRICS_PORT:-8002}"

uv run python "${REPO_ROOT}/triton/setup_model_repository.py"

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
