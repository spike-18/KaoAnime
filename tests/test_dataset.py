# tests/test_dataset.py
from pathlib import Path
from PIL import Image
from kaoanime.utils.dataset import UnpairedImageDataset
from kaoanime.utils.transforms import get_transforms


def _make_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for i in range(count):
        Image.new("RGB", (64, 64), color=(i * 10, 0, 0)).save(directory / f"{i}.jpg")


def test_len_is_max_of_both_domains(tmp_path):
    _make_images(tmp_path / "A", 5)
    _make_images(tmp_path / "B", 3)
    ds = UnpairedImageDataset(
        tmp_path / "A", tmp_path / "B", get_transforms("test", 64)
    )
    assert len(ds) == 5


def test_getitem_returns_a_and_b_tensors(tmp_path):
    _make_images(tmp_path / "A", 3)
    _make_images(tmp_path / "B", 3)
    ds = UnpairedImageDataset(
        tmp_path / "A", tmp_path / "B", get_transforms("test", 64)
    )
    item = ds[0]
    assert set(item.keys()) == {"A", "B"}
    assert item["A"].shape == (3, 64, 64)
    assert item["B"].shape == (3, 64, 64)


def test_b_wraps_around_when_a_is_larger(tmp_path):
    _make_images(tmp_path / "A", 5)
    _make_images(tmp_path / "B", 3)
    ds = UnpairedImageDataset(
        tmp_path / "A", tmp_path / "B", get_transforms("test", 64)
    )
    # index 4: A[4], B[4 % 3 = 1] — both must be valid
    item = ds[4]
    assert item["B"].shape == (3, 64, 64)


def test_accepts_string_paths(tmp_path):
    _make_images(tmp_path / "A", 2)
    _make_images(tmp_path / "B", 2)
    ds = UnpairedImageDataset(
        str(tmp_path / "A"), str(tmp_path / "B"), get_transforms("test", 64)
    )
    assert len(ds) == 2


def test_a_wraps_around_when_b_is_larger(tmp_path):
    _make_images(tmp_path / "A", 3)
    _make_images(tmp_path / "B", 5)
    ds = UnpairedImageDataset(
        tmp_path / "A", tmp_path / "B", get_transforms("test", 64)
    )
    assert len(ds) == 5
    # index 4: A[4 % 3 = 1], B[4] — both must be valid
    item = ds[4]
    assert item["A"].shape == (3, 64, 64)
    assert item["B"].shape == (3, 64, 64)
