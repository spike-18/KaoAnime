from __future__ import annotations

from kaoanime.data.download import download_data
from kaoanime.data.layout import required_data_dirs


def ensure_data(cfg) -> None:
    """Download the dataset if any required split directory is missing."""
    missing = [d for d in required_data_dirs(cfg) if not d.is_dir()]
    if missing:
        download_data(cfg.data.variant)
