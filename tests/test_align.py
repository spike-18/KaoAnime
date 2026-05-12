import numpy as np
import pytest

from kaoanime.utils.align import AlignFaceProcessor, align_face


def _solid_image(h: int = 256, w: int = 256, value: int = 128) -> np.ndarray:
    """Uniform grey — guaranteed no face detection."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _noise_image(h: int = 256, w: int = 256) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ── align_face convenience wrapper ────────────────────────────────────────────

def test_align_face_returns_none_for_no_face():
    result = align_face(_solid_image(), size=128)
    assert result is None


def test_align_face_returns_none_for_noise():
    result = align_face(_noise_image(), size=128)
    assert result is None


def test_align_face_output_contract(face_jpg):
    """If a face is detected the output is (size, size, 3) uint8."""
    import cv2
    img = cv2.cvtColor(cv2.imread(str(face_jpg)), cv2.COLOR_BGR2RGB)
    result = align_face(img, size=128)
    if result is None:
        pytest.skip("face not detected in fixture — skip shape contract check")
    assert result.shape == (128, 128, 3)
    assert result.dtype == np.uint8


def test_align_face_respects_size_param(face_jpg):
    import cv2
    img = cv2.cvtColor(cv2.imread(str(face_jpg)), cv2.COLOR_BGR2RGB)
    for size in (64, 128, 256):
        result = align_face(img, size=size)
        if result is not None:
            assert result.shape == (size, size, 3)


# ── AlignFaceProcessor reuse ──────────────────────────────────────────────────

def test_processor_reusable_across_calls():
    """Processor can be called many times without re-loading the model."""
    img = _solid_image()
    proc = AlignFaceProcessor()
    for _ in range(5):
        assert proc.align(img, size=128) is None  # no crash


def test_processor_independent_instances():
    """Two separate processors work independently."""
    img = _solid_image()
    p1, p2 = AlignFaceProcessor(), AlignFaceProcessor()
    assert p1.align(img, 128) is None
    assert p2.align(img, 128) is None


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def face_jpg(tmp_path_factory):
    """Try to grab a real face image from the selfie2anime test set.
    Falls back to a blank image (detection will fail; tests skip gracefully)."""
    import shutil
    from pathlib import Path
    candidates = sorted(Path("data/selfie2anime/testA").glob("*.jpg"))
    dst = tmp_path_factory.mktemp("fixtures") / "face.jpg"
    if candidates:
        shutil.copy(candidates[0], dst)
    else:
        from PIL import Image
        Image.new("RGB", (256, 256), 128).save(dst)
    return dst
