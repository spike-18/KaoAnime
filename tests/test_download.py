import pytest

from kaoanime.data import download_data


def test_demo_calls_dvc_pull(monkeypatch):
    calls = {}

    def fake_pull(remote, targets):
        calls["remote"] = remote
        calls["targets"] = targets

    monkeypatch.setattr("kaoanime.data.download._dvc_pull", fake_pull)
    download_data("demo")
    assert calls["remote"] == "data"


def test_full_invokes_kaggle_and_gdown(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "kaoanime.data.download._download_celeba", lambda dest: seen.append("celeba")
    )
    monkeypatch.setattr(
        "kaoanime.data.download._download_anime", lambda dest: seen.append("anime")
    )
    monkeypatch.setattr(
        "kaoanime.data.download._layout_full", lambda dest: seen.append("layout")
    )
    download_data("full", dest=str(tmp_path))
    assert seen == ["celeba", "anime", "layout"]


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        download_data("nope")
