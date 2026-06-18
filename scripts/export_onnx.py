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
from torch.export import Dim

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


def export_module(transport: torch.nn.Module, out: str, image_size: int = 128) -> str:
    """Export a transport module to ONNX (dynamic batch) and verify parity."""
    transport.eval()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        transport,
        (dummy,),
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"x": {0: Dim("batch")}},  # UNetGenerator.forward(self, x)
        dynamo=True,
    )
    _verify_parity(transport, str(out_path), image_size)
    return str(out_path)


def _load_transport(checkpoint: str, t_filters: int, t_norm: str) -> torch.nn.Module:
    """Build NOTModel from cfg, load the checkpoint, and verify T loaded fully.

    Loads non-strictly (the checkpoint also carries `f`/`fid` we ignore) but fails
    loudly if any transport (`T.*`) weight is missing — that means the checkpoint
    architecture does not match (e.g. wrong `t_norm` or `t_filters`).
    """
    cfg = Config()
    cfg.not_.t_filters = t_filters
    cfg.not_.t_norm = t_norm
    model = NOTModel(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    result = model.load_state_dict(state, strict=False)
    missing_t = [key for key in result.missing_keys if key.startswith("T.")]
    if missing_t:
        raise ValueError(
            f"Checkpoint does not match the model: {len(missing_t)} transport (T.*) "
            f"weights missing, e.g. {missing_t[:3]}. Try a different --t_norm "
            f"(instance/batch) or --t_filters to match how the checkpoint was trained."
        )
    model.eval()
    return model.T


def export(
    checkpoint: str,
    out: str = "models/export/model.onnx",
    t_filters: int = 48,
    t_norm: str = "batch",
    image_size: int = 128,
) -> str:
    """Load a NOT checkpoint and export its transport map T to ONNX."""
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    transport = _load_transport(str(ckpt), t_filters, t_norm)
    path = export_module(transport, out, image_size=image_size)
    print(f"Exported ONNX to {path}")
    return path


if __name__ == "__main__":
    fire.Fire(export)
