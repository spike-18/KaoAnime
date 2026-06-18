# Triton Inference Server for KaoAnime.
#
# The full Triton image ships the onnxruntime backend (CPU execution is used here)
# and the Python backend; we only add Pillow for the pre/post-process Python
# backends. Use a full "-py3" tag, NOT "-py3-min" (the minimal image has no
# backends). Override the base tag with --build-arg TRITON_TAG=<tag>.
ARG TRITON_TAG=26.05-py3
FROM nvcr.io/nvidia/tritonserver:${TRITON_TAG}

RUN pip install --no-cache-dir pillow
