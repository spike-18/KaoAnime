# tests/test_image.py
import torch
from PIL import Image
from kaoanime.utils.image import load_image, save_image


def test_load_image_returns_rgb_pil(tmp_path):
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_path)
    img = load_image(img_path)
    assert img.mode == "RGB"
    assert img.size == (64, 64)


def test_load_image_accepts_string_path(tmp_path):
    img_path = tmp_path / "test.png"
    Image.new("RGB", (32, 32)).save(img_path)
    img = load_image(str(img_path))
    assert isinstance(img, Image.Image)


def test_save_image_creates_parent_dirs(tmp_path):
    out_path = tmp_path / "subdir" / "nested" / "out.png"
    tensor = torch.zeros(3, 32, 32)
    save_image(tensor, out_path)
    assert out_path.exists()


def test_save_image_white_tensor_produces_white_pixels(tmp_path):
    out_path = tmp_path / "out.png"
    tensor = torch.ones(3, 16, 16)  # all 1.0 in [-1,1] = white
    save_image(tensor, out_path)
    img = Image.open(out_path).convert("RGB")
    pixels = list(img.getdata())
    assert all(v == 255 for px in pixels for v in px)


def test_save_image_black_tensor_produces_black_pixels(tmp_path):
    out_path = tmp_path / "out.png"
    tensor = torch.full((3, 16, 16), -1.0)  # all -1.0 in [-1,1] = black
    save_image(tensor, out_path)
    img = Image.open(out_path).convert("RGB")
    pixels = list(img.getdata())
    assert all(v == 0 for px in pixels for v in px)
