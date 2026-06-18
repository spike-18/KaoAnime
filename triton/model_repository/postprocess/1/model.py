"""Triton Python backend: turn the transport-map output into an RGB image.

Mirrors ``scripts/infer_onnx.postprocess``: clip to [-1, 1], rescale to [0, 255],
convert CHW -> HWC and cast to uint8. Output is a ready-to-save RGB image array.
"""

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            arr = pb_utils.get_input_tensor_by_name(request, "output").as_numpy()
            arr = np.clip(arr, -1.0, 1.0)
            arr = (arr + 1.0) / 2.0
            arr = (arr.transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)
            out = pb_utils.Tensor("OUTPUT_IMAGE", arr)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
