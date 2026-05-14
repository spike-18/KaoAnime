from pathlib import Path

import torch
from PIL import Image

from kaoanime.config import Config
from kaoanime.inference import run_inference
from kaoanime.model_cyclegan import KaoAnimeModel


def _make_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for i in range(count):
        Image.new("RGB", (128, 128), color=(i * 30, 100, 200)).save(directory / f"{i}.jpg")


def _cpu_model() -> KaoAnimeModel:
    model = KaoAnimeModel(Config())
    model.eval()
    return model


def test_infer_single_image(tmp_path):
    src = tmp_path / "input.jpg"
    Image.new("RGB", (128, 128)).save(src)
    out_dir = tmp_path / "out"

    run_inference(_cpu_model(), src, out_dir, image_size=128, direction="a2b", device="cpu")

    assert (out_dir / "input.jpg").exists()


def test_infer_directory(tmp_path):
    src_dir = tmp_path / "inputs"
    _make_images(src_dir, 3)
    out_dir = tmp_path / "out"

    run_inference(_cpu_model(), src_dir, out_dir, image_size=128, direction="a2b", device="cpu")

    assert len(list(out_dir.glob("*.jpg"))) == 3


def test_infer_b2a_direction(tmp_path):
    src = tmp_path / "anime.jpg"
    Image.new("RGB", (128, 128)).save(src)
    out_dir = tmp_path / "out"

    run_inference(_cpu_model(), src, out_dir, image_size=128, direction="b2a", device="cpu")

    assert (out_dir / "anime.jpg").exists()


def test_infer_output_filenames_match_input(tmp_path):
    src_dir = tmp_path / "inputs"
    src_dir.mkdir()
    Image.new("RGB", (64, 64)).save(src_dir / "face_001.png")
    Image.new("RGB", (64, 64)).save(src_dir / "face_002.png")
    out_dir = tmp_path / "out"

    run_inference(_cpu_model(), src_dir, out_dir, image_size=128, direction="a2b", device="cpu")

    assert (out_dir / "face_001.png").exists()
    assert (out_dir / "face_002.png").exists()


def test_infer_invalid_direction_raises(tmp_path):
    src = tmp_path / "input.jpg"
    Image.new("RGB", (128, 128)).save(src)
    out_dir = tmp_path / "out"

    import pytest
    with pytest.raises(ValueError, match="direction must be"):
        run_inference(_cpu_model(), src, out_dir, image_size=128, direction="A2B", device="cpu")


def test_infer_align_false_does_not_crash(tmp_path):
    """align=False should work identically to the old signature."""
    src = tmp_path / "input.jpg"
    Image.new("RGB", (128, 128)).save(src)
    out_dir = tmp_path / "out"
    run_inference(_cpu_model(), src, out_dir, image_size=128,
                  direction="a2b", device="cpu", align=False)
    assert (out_dir / "input.jpg").exists()


def test_infer_align_true_solid_image_still_produces_output(tmp_path):
    """align=True on a faceless image falls back to centre-crop (no crash, output written)."""
    src = tmp_path / "solid.jpg"
    Image.new("RGB", (128, 128), color=(128, 128, 128)).save(src)
    out_dir = tmp_path / "out"
    run_inference(_cpu_model(), src, out_dir, image_size=128,
                  direction="a2b", device="cpu", align=True)
    assert (out_dir / "solid.jpg").exists()
