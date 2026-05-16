# tests/test_dataset.py
from pathlib import Path
from PIL import Image as PILImage
from kaoanime.utils.dataset import UnpairedImageDataset, _anime_crop


def _make_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for i in range(count):
        PILImage.new("RGB", (64, 64), color=(i * 10, 0, 0)).save(directory / f"{i}.jpg")


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
