from __future__ import annotations

from pathlib import Path


def download_model(gdrive_id: str, dest: str = "models/export") -> None:
    """Download a published model bundle from a public Google Drive folder.

    Fetches every file in the folder as-is into *dest* (ONNX graph + external
    weights + checkpoint, possibly several variants). Names are kept exactly as
    published — inference, serving and the export scripts each take an explicit
    model path, so no renaming or canonical-name assumptions are made here.
    """
    import gdown

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(
        id=gdrive_id, output=str(dest_path), quiet=False, use_cookies=False
    )
