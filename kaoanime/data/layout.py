from __future__ import annotations

from pathlib import Path


def required_data_dirs(cfg) -> list[Path]:
    """The four split directories that must exist before training/inference."""
    data = cfg.data
    return [Path(data.root_a), Path(data.root_b), Path(data.test_a), Path(data.test_b)]
