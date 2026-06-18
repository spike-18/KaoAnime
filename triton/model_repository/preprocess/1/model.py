"""Triton Python backend: decode an image and produce the transport-map input.

Mirrors ``scripts/infer_onnx.preprocess_image``: resize to 128x128, scale to
[0, 1], then normalise to [-1, 1] and convert HWC -> CHW. Input is the raw bytes
of an encoded image (JPEG/PNG/...); output feeds the ONNX transport map.
"""

import io

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image

IMAGE_SIZE = 128


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "IMAGE").as_numpy()
            batch = []
            for element in raw.reshape(-1):
                img = Image.open(io.BytesIO(bytes(element))).convert("RGB")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                arr = (arr - 0.5) / 0.5
                batch.append(arr.transpose(2, 0, 1))
            out = pb_utils.Tensor("input", np.stack(batch).astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
