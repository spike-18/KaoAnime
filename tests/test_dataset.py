# tests/test_dataset.py
import csv
from pathlib import Path

import pytest
from PIL import Image as PILImage

from kaoanime.utils.dataset import (
    UnpairedImageDataset,
    _anime_crop,
    _find_celeba_attr_csv,
    _load_female_ids,
)


def _make_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for i in range(count):
        PILImage.new("RGB", (64, 64), color=(i * 10, 0, 0)).save(directory / f"{i}.jpg")
    if directory.name == "A":
        _write_csv(
            directory.parent / "list_attr_celeba.csv",
            ["image_id", "Male"],
            [[f"{i}.jpg", "-1"] for i in range(count)],
        )


def test_len_is_max_of_both_domains(tmp_path):
    _make_images(tmp_path / "A", 5)
    _make_images(tmp_path / "B", 3)
    ds = UnpairedImageDataset(
        tmp_path / "A", tmp_path / "B", image_size=64, train=False
    )
    assert len(ds) == 5


def test_getitem_returns_a_and_b_tensors(tmp_path):
    _make_images(tmp_path / "A", 3)
    _make_images(tmp_path / "B", 3)
    ds = UnpairedImageDataset(
        tmp_path / "A", tmp_path / "B", image_size=64, train=False
    )
    item = ds[0]
    assert set(item.keys()) == {"A", "B"}
    assert item["A"].shape == (3, 64, 64)
    assert item["B"].shape == (3, 64, 64)


def test_b_wraps_around_when_a_is_larger(tmp_path):
    _make_images(tmp_path / "A", 5)
    _make_images(tmp_path / "B", 3)
    ds = UnpairedImageDataset(
        tmp_path / "A", tmp_path / "B", image_size=64, train=False
    )
    # index 4: A[4], B[4 % 3 = 1] — both must be valid
    item = ds[4]
    assert item["B"].shape == (3, 64, 64)


def test_accepts_string_paths(tmp_path):
    _make_images(tmp_path / "A", 2)
    _make_images(tmp_path / "B", 2)
    ds = UnpairedImageDataset(
        str(tmp_path / "A"), str(tmp_path / "B"), image_size=64, train=False
    )
    assert len(ds) == 2


def test_a_wraps_around_when_b_is_larger(tmp_path):
    _make_images(tmp_path / "A", 3)
    _make_images(tmp_path / "B", 5)
    ds = UnpairedImageDataset(
        tmp_path / "A", tmp_path / "B", image_size=64, train=False
    )
    assert len(ds) == 5
    # index 4: A[4 % 3 = 1], B[4] — both must be valid
    item = ds[4]
    assert item["A"].shape == (3, 64, 64)
    assert item["B"].shape == (3, 64, 64)


def test_anime_crop_output_size():
    img = PILImage.new("RGB", (400, 300))
    result = _anime_crop(img, offset_x=0, offset_y=0)
    assert result.size == (256, 256), f"Expected (256, 256), got {result.size}"


def test_anime_crop_zero_offset_is_centred():
    # At 512×512, centre crop (offset=0) runs x:[128,384], y:[128,384].
    # Pixel (0,0) of the crop should equal pixel (128,128) of the 512 image.
    img = PILImage.new("RGB", (512, 512))
    pixels = img.load()
    for y in range(512):
        for x in range(512):
            pixels[x, y] = (x % 256, y % 256, 0)
    result = _anime_crop(img, offset_x=0, offset_y=0)
    r, g, _ = result.getpixel((0, 0))
    assert r == 128 and g == 128, f"Expected (128,128,_), got ({r},{g},_)"


def test_anime_crop_negative_offset_y_shifts_up():
    # offset_y=-7 → crop centre at y=249; top-left of crop at y=121
    img = PILImage.new("RGB", (512, 512))
    pixels = img.load()
    for y in range(512):
        for x in range(512):
            pixels[x, y] = (x % 256, y % 256, 0)
    result = _anime_crop(img, offset_x=0, offset_y=-7)
    r, g, _ = result.getpixel((0, 0))
    # top-left of crop: x=128, y=256-7-128=121
    assert r == 128 and g == 121, f"Expected (128,121,_), got ({r},{g},_)"


def test_anime_crop_works_on_non_square_input():
    img = PILImage.new("RGB", (800, 600))
    result = _anime_crop(img, offset_x=0, offset_y=-7)
    assert result.size == (256, 256)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def test_load_female_ids_selects_only_male_minus_one(tmp_path):
    csv_path = tmp_path / "list_attr_celeba.csv"
    # 'Male' is NOT the first attribute column — proves we key by header name.
    _write_csv(
        csv_path,
        ["image_id", "Attractive", "Male", "Young"],
        [
            ["000001.jpg", "1", "-1", "1"],  # female -> kept
            ["000002.jpg", "-1", "1", "1"],  # male   -> excluded
            ["000003.jpg", "1", "-1", "-1"],  # female -> kept
        ],
    )
    assert _load_female_ids(csv_path) == {"000001.jpg", "000003.jpg"}


def test_load_female_ids_raises_without_male_column(tmp_path):
    csv_path = tmp_path / "list_attr_celeba.csv"
    _write_csv(csv_path, ["image_id", "Young"], [["000001.jpg", "1"]])
    with pytest.raises(ValueError, match="Male"):
        _load_female_ids(csv_path)


def test_find_attr_csv_walks_up_parents(tmp_path):
    # root_a is two levels below where the CSV lives — mirrors the real
    # layout: .../CelebA/list_attr_celeba.csv vs .../CelebA/img/img.
    (tmp_path / "list_attr_celeba.csv").write_text("image_id,Male\n")
    root_a = tmp_path / "img_align_celeba" / "img_align_celeba"
    root_a.mkdir(parents=True)
    found = _find_celeba_attr_csv(root_a)
    assert found == tmp_path / "list_attr_celeba.csv"


def test_find_attr_csv_returns_none_when_absent(tmp_path):
    root_a = tmp_path / "A"
    root_a.mkdir()
    assert _find_celeba_attr_csv(root_a) is None


def test_find_attr_csv_accepts_string_path(tmp_path):
    (tmp_path / "list_attr_celeba.csv").write_text("image_id,Male\n")
    root_a = tmp_path / "A"
    root_a.mkdir()
    assert _find_celeba_attr_csv(str(root_a)) == tmp_path / "list_attr_celeba.csv"


def _make_celeba(
    tmp_path: Path, a_ids: list[str], b_count: int, female: set[str] | None = None
) -> tuple[Path, Path]:
    """Create A/ with a_ids, B/ with b_count images, and an attribute CSV
    at tmp_path. Every id in `female` (default: all a_ids) gets Male=-1."""
    female = set(a_ids) if female is None else female
    a_dir = tmp_path / "A"
    a_dir.mkdir()
    for name in a_ids:
        PILImage.new("RGB", (64, 64)).save(a_dir / name)
    b_dir = tmp_path / "B"
    b_dir.mkdir()
    for i in range(b_count):
        PILImage.new("RGB", (64, 64)).save(b_dir / f"{i}.jpg")
    rows = [[n, "-1" if n in female else "1"] for n in a_ids]
    _write_csv(tmp_path / "list_attr_celeba.csv", ["image_id", "Male"], rows)
    return a_dir, b_dir


def test_dataset_keeps_only_female_root_a(tmp_path):
    a_dir, b_dir = _make_celeba(
        tmp_path,
        ["f0.jpg", "m0.jpg", "f1.jpg"],
        b_count=2,
        female={"f0.jpg", "f1.jpg"},
    )
    ds = UnpairedImageDataset(a_dir, b_dir, image_size=64, train=False)
    names = sorted(p.name for p in ds._files_a)
    assert names == ["f0.jpg", "f1.jpg"]


def test_dataset_does_not_filter_extra_roots_a(tmp_path):
    a_dir, b_dir = _make_celeba(tmp_path, ["f0.jpg"], b_count=1, female={"f0.jpg"})
    extra = tmp_path / "ffhq"
    extra.mkdir()
    PILImage.new("RGB", (64, 64)).save(extra / "ffhq0.jpg")  # not in CSV
    ds = UnpairedImageDataset(
        a_dir, b_dir, image_size=64, train=False, extra_roots_a=[extra]
    )
    names = sorted(p.name for p in ds._files_a)
    assert names == ["f0.jpg", "ffhq0.jpg"]


def test_dataset_raises_when_attr_csv_missing(tmp_path):
    a_dir = tmp_path / "A"
    a_dir.mkdir()
    PILImage.new("RGB", (64, 64)).save(a_dir / "0.jpg")
    b_dir = tmp_path / "B"
    b_dir.mkdir()
    PILImage.new("RGB", (64, 64)).save(b_dir / "0.jpg")
    with pytest.raises(FileNotFoundError, match="list_attr_celeba.csv"):
        UnpairedImageDataset(a_dir, b_dir, image_size=64, train=False)


def test_dataset_raises_when_no_female_match(tmp_path):
    # CSV present but every id is male -> filter yields zero -> must fail loud
    a_dir, b_dir = _make_celeba(tmp_path, ["m0.jpg", "m1.jpg"], b_count=2, female=set())
    with pytest.raises(ValueError, match="matched 0"):
        UnpairedImageDataset(a_dir, b_dir, image_size=64, train=False)
