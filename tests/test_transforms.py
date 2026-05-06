from PIL import Image
from kaoanime.utils.transforms import get_transforms


def _dummy_image(size: int = 200) -> Image.Image:
    return Image.new("RGB", (size, size), color=(128, 64, 32))


def test_train_transforms_output_shape():
    t = get_transforms("train", image_size=128)
    result = t(_dummy_image())
    assert result.shape == (3, 128, 128)


def test_test_transforms_output_shape():
    t = get_transforms("test", image_size=128)
    result = t(_dummy_image())
    assert result.shape == (3, 128, 128)


def test_default_image_size_is_128():
    t = get_transforms("test")
    result = t(_dummy_image())
    assert result.shape == (3, 128, 128)


def test_transforms_output_in_neg1_to_1_range():
    t = get_transforms("test", image_size=64)
    result = t(_dummy_image())
    assert result.min() >= -1.0
    assert result.max() <= 1.0


def test_train_transforms_resize_before_crop():
    # Input smaller than image_size+30 should still produce correct output shape
    t = get_transforms("train", image_size=64)
    result = t(_dummy_image(size=50))
    assert result.shape == (3, 64, 64)
