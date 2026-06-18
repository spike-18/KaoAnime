#!/usr/bin/env python
"""Populate the Triton model repository with the ONNX transport map.

Downloads the model bundle if missing (``ensure_model``) and copies the exported
ONNX graph + its external weights into ``transport/1``. The ONNX file is renamed to
``model.onnx`` (Triton's default) while the weights keep the name embedded in the
graph (``last.onnx.data``), so onnxruntime resolves them next to the model file.

Example:
    uv run python triton/setup_model_repository.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import fire

from kaoanime.config import Config
from kaoanime.model_store import ONNX_DATA_NAME, ONNX_NAME, ensure_model

_REPO = Path(__file__).resolve().parent / "model_repository"


def setup() -> None:
    cfg = Config()
    ensure_model(cfg)
    src = Path(cfg.eval.model_dir)
    dst = _REPO / "transport" / "1"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / ONNX_NAME, dst / "model.onnx")
    shutil.copyfile(src / ONNX_DATA_NAME, dst / ONNX_DATA_NAME)
    print(f"Model repository ready at {dst}")


if __name__ == "__main__":
    fire.Fire(setup)
