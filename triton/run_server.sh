#!/usr/bin/env bash
# Build the KaoAnime Triton image and serve the ensemble on onnxruntime (CPU).
#
# Steps: populate the model repository (download + copy ONNX), build the image,
# then run Triton with HTTP:8000 / gRPC:8001 / metrics:8002.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_TAG="${TRITON_TAG:-26.05-py3}"
IMAGE="${IMAGE:-kaoanime-triton}"

uv run python "${REPO_ROOT}/triton/setup_model_repository.py"

docker build \
  --build-arg "TRITON_TAG=${TRITON_TAG}" \
  -t "${IMAGE}" \
  "${REPO_ROOT}/triton"

docker run --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "${REPO_ROOT}/triton/model_repository:/models" \
  "${IMAGE}" \
  tritonserver --model-repository=/models
