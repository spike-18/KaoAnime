#!/usr/bin/env python
"""Download the published model bundle from Google Drive into the model dir.

Fetches the weights as-is (no renaming). Inference, Triton and the export scripts
then take an explicit model path, so you choose which variant to use.

Example:
    uv run python scripts/download_model.py                       # config defaults
    uv run python scripts/download_model.py --dest models/export
    uv run python scripts/download_model.py --gdrive_id <id> --dest <dir>
"""

from __future__ import annotations

import fire

from kaoanime.config import Config
from kaoanime.model_store import download_model


def main(gdrive_id: str = "", dest: str = "") -> None:
    cfg = Config()
    download_model(
        gdrive_id or cfg.eval.model_gdrive_id, dest=dest or cfg.eval.model_dir
    )


if __name__ == "__main__":
    fire.Fire(main)
