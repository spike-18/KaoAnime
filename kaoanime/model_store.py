from __future__ import annotations

from pathlib import Path

# Canonical artifact names expected by inference and serving. The exported ONNX
# graph embeds a *relative* reference to its external-weights file by name
# ("last.onnx.data", the name it was exported under), so the published Google
# Drive bundle must already use these canonical names — otherwise onnxruntime/
# Triton cannot resolve the weights file.
ONNX_NAME = "last.onnx"
ONNX_DATA_NAME = "last.onnx.data"
CKPT_NAME = "last.ckpt"


def download_model(gdrive_id: str, dest: str = "models/export") -> None:
    """Download the published model bundle from a public Google Drive folder.

    The folder holds the transport map exported to ONNX (graph + external weights)
    plus the Lightning checkpoint, already named with the canonical scheme
    (last.onnx / last.onnx.data / last.ckpt) so the ONNX external weights resolve
    and downstream code finds predictable paths. Files are fetched into *dest*.
    """
    import gdown

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(
        id=gdrive_id, output=str(dest_path), quiet=False, use_cookies=False
    )

    missing = [
        name
        for name in (ONNX_NAME, ONNX_DATA_NAME, CKPT_NAME)
        if not (dest_path / name).exists()
    ]
    if missing:
        raise RuntimeError(
            f"Model download incomplete, missing {missing} in {dest_path}. The "
            f"Google Drive bundle must use the canonical names ({ONNX_NAME} / "
            f"{ONNX_DATA_NAME} / {CKPT_NAME}); Drive may also be rate-limiting "
            "large files, or a file is not shared as 'Anyone with the link'. "
            "Retry later or pull via 'dvc pull -r models'."
        )


def ensure_model(cfg) -> None:
    """Download the model bundle if any required artifact is missing."""
    dest = Path(cfg.eval.model_dir)
    required = [dest / ONNX_NAME, dest / ONNX_DATA_NAME, dest / CKPT_NAME]
    if all(path.exists() for path in required):
        return
    if not cfg.eval.model_gdrive_id:
        raise ValueError(
            "eval.model_gdrive_id is required to download the model bundle; set it "
            "to the public Google Drive folder id of the published model."
        )
    download_model(cfg.eval.model_gdrive_id, dest=str(dest))
