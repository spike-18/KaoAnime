#!/usr/bin/env python
"""Export the NOT transport map (T) to ONNX, verifying parity with PyTorch.

Example:
    uv run python scripts/export_onnx.py --checkpoint checkpoints/not_ep10.ckpt \\
        --out models/export/model.onnx
"""

from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import torch

from kaoanime.config import Config
from kaoanime.model_not import NOTModel


def _verify_parity(
    transport: torch.nn.Module, onnx_path: str, image_size: int, tol: float = 1e-3
) -> None:
    import onnxruntime as ort

    sample = torch.randn(2, 3, image_size, image_size)
    with torch.no_grad():
        torch_out = transport(sample).numpy()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": sample.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    if max_diff > tol:
        raise ValueError(
            f"ONNX parity check failed: max abs diff {max_diff:.2e} > {tol:.0e}"
        )


def export_module(
    transport: torch.nn.Module, out: str, image_size: int = 128, opset: int = 17
) -> str:
    """Export a transport module to ONNX (dynamic batch) and verify parity."""
    transport.eval()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        transport,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # legacy exporter; the dynamo path needs the onnxscript dep
    )
    _verify_parity(transport, str(out_path), image_size)
    return str(out_path)


def export(
    checkpoint: str,
    out: str = "models/export/model.onnx",
    t_filters: int = 48,
    image_size: int = 128,
    opset: int = 17,
) -> str:
    """Load a NOT checkpoint and export its transport map T to ONNX."""
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    cfg = Config()
    cfg.not_.t_filters = t_filters
    model = NOTModel.load_from_checkpoint(
        str(ckpt), cfg=cfg, map_location="cpu", strict=False
    )
    model.eval()
    path = export_module(model.T, out, image_size=image_size, opset=opset)
    print(f"Exported ONNX to {path}")
    return path


if __name__ == "__main__":
    fire.Fire(export)
