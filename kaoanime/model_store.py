from __future__ import annotations

from pathlib import Path

# Canonical artifact names expected by inference and serving. The exported ONNX
# graph embeds a *relative* reference to its external-weights file named
# "last.onnx.data" (the name it was exported under). Published Google Drive copies
# may carry different names (e.g. NOT.onnx), so downloads are normalised back to
# this scheme — otherwise onnxruntime/Triton cannot resolve the weights file.
ONNX_NAME = "last.onnx"
ONNX_DATA_NAME = "last.onnx.data"
CKPT_NAME = "last.ckpt"


def _normalise_one(dest: Path, suffix: str, target: str) -> None:
    """Rename the single file in *dest* ending with *suffix* to *target*."""
    matches = [
        path
        for path in dest.iterdir()
        if path.is_file() and path.name.endswith(suffix) and path.name != target
    ]
    if not matches:
        return
    if len(matches) > 1:
        raise ValueError(
            f"Expected one '*{suffix}' file in {dest}, found "
            f"{[path.name for path in matches]}"
        )
    matches[0].rename(dest / target)


def download_model(gdrive_id: str, dest: str = "models/export") -> None:
    """Download the published model bundle from a public Google Drive folder.

    The folder holds the transport map exported to ONNX (graph + external weights)
    plus the Lightning checkpoint. Files are fetched into *dest* and renamed to the
    canonical scheme (last.onnx / last.onnx.data / last.ckpt) so the ONNX external
    weights resolve and downstream code finds predictable paths.
    """
    import gdown

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(
        id=gdrive_id, output=str(dest_path), quiet=False, use_cookies=False
    )
    # Normalise data file first so the bare ".onnx" match below is unambiguous.
    _normalise_one(dest_path, ".onnx.data", ONNX_DATA_NAME)
    _normalise_one(dest_path, ".onnx", ONNX_NAME)
    _normalise_one(dest_path, ".ckpt", CKPT_NAME)

    missing = [
        name
        for name in (ONNX_NAME, ONNX_DATA_NAME, CKPT_NAME)
        if not (dest_path / name).exists()
    ]
    if missing:
        raise RuntimeError(
            f"Model download incomplete, missing {missing} in {dest_path}. Google "
            "Drive may be rate-limiting large files, or a file is not shared as "
            "'Anyone with the link'. Retry later or pull via 'dvc pull -r models'."
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
