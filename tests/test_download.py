import pytest

from kaoanime.data import download_data


def test_demo_downloads_zip(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        "kaoanime.data.download._download_demo_zip",
        lambda gdrive_id, dest: seen.update(id=gdrive_id, dest=str(dest)),
    )
    download_data("demo", dest=str(tmp_path), demo_gdrive_id="ABC123")
    assert seen["id"] == "ABC123"


def test_demo_requires_gdrive_id():
    with pytest.raises(ValueError):
        download_data("demo", demo_gdrive_id="")


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
