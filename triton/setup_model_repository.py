#!/usr/bin/env python
"""Stage an exported ONNX transport map into the Triton model repository.

Copies the chosen ONNX graph to ``transport/1/model.onnx`` (the filename Triton's
onnxruntime backend loads) and its external-weights file alongside under the name
embedded in the graph (``<onnx>.data``), so onnxruntime resolves them. You pick
which model to serve — download the weights first
(``scripts/download_model.py``) or export your own (``scripts/export_onnx.py``).

Example:
    uv run python triton/setup_model_repository.py --onnx NOT-cuda.onnx
"""

from __future__ import annotations

import shutil
from pathlib import Path

import fire

from kaoanime.config import Config

_REPO = Path(__file__).resolve().parent / "model_repository"


def setup(onnx: str) -> None:
    """Stage *onnx* (a file under cfg.eval.model_dir) as transport/1/model.onnx."""
    cfg = Config()
    src_dir = Path(cfg.eval.model_dir)
    onnx_path = src_dir / onnx
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}. Download the weights first "
            f"(scripts/download_model.py) or export one (scripts/export_onnx.py), "
            f"then pass --onnx <name> for a file under {src_dir}."
        )

    # The ONNX graph references its external weights by name (<onnx>.data); copy
    # that file under the same name so onnxruntime/Triton can resolve it.
    data_name = f"{onnx}.data"
    src_data = src_dir / data_name
    if not src_data.exists():
        raise FileNotFoundError(
            f"External-weights file not found: {src_data} (referenced by {onnx})."
        )

    dst = _REPO / "transport" / "1"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx_path, dst / "model.onnx")
    shutil.copyfile(src_data, dst / data_name)
    # Drop any stale weights file left over from a previously staged model.
    for old in dst.glob("*.onnx.data"):
        if old.name != data_name:
            old.unlink()
    print(f"Model repository ready at {dst} (model.onnx from {onnx})")


if __name__ == "__main__":
    fire.Fire(setup)
