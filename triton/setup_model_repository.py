#!/usr/bin/env python
"""Populate the Triton model repository with an exported ONNX transport map.

Copies the chosen ONNX graph and its external-weights file into ``transport/1``.
The graph is renamed to ``model.onnx`` (Triton's default) while the weights keep
the name embedded in the graph (``<onnx>.data``), so onnxruntime resolves them next
to the model file. The source ONNX is resolved under the model bundle directory
(``cfg.eval.model_dir``); if it is missing, the published bundle is fetched via
``ensure_model``.

Example:
    uv run python triton/setup_model_repository.py                  # model.onnx
    uv run python triton/setup_model_repository.py --onnx other.onnx
"""

from __future__ import annotations

import shutil
from pathlib import Path

import fire

from kaoanime.config import Config
from kaoanime.model_store import ensure_model

_REPO = Path(__file__).resolve().parent / "model_repository"


def setup(onnx: str = "model.onnx") -> None:
    cfg = Config()
    src_dir = Path(cfg.eval.model_dir)
    onnx_path = src_dir / onnx
    if not onnx_path.exists():
        # Fall back to downloading the published (canonically named) bundle.
        ensure_model(cfg)
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}. Export it first via "
            f"scripts/export_onnx.py, or pass --onnx <name> for a file under "
            f"{src_dir}."
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
    print(f"Model repository ready at {dst} (from {onnx})")


if __name__ == "__main__":
    fire.Fire(setup)
