from pathlib import Path

from omegaconf import OmegaConf

from kaoanime.config import Config
from kaoanime.data import ensure_data


def _cfg(tmp_path, variant="demo"):
    cfg = OmegaConf.structured(Config)
    cfg.data.variant = variant
    base = tmp_path / variant
    cfg.data.root_a = str(base / "trainA")
    cfg.data.root_b = str(base / "trainB")
    cfg.data.test_a = str(base / "testA")
    cfg.data.test_b = str(base / "testB")
    return cfg


def test_downloads_when_dirs_missing(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    called = {}
    monkeypatch.setattr(
        "kaoanime.data.ensure.download_data",
        lambda variant, **kw: called.setdefault("variant", variant),
    )
    ensure_data(cfg)
    assert called["variant"] == "demo"


def test_no_download_when_all_dirs_present(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    for key in ("root_a", "root_b", "test_a", "test_b"):
        Path(getattr(cfg.data, key)).mkdir(parents=True)
    called = {}
    monkeypatch.setattr(
        "kaoanime.data.ensure.download_data",
        lambda variant, **kw: called.setdefault("variant", variant),
    )
    ensure_data(cfg)
    assert called == {}
