# tests/test_utils_exports.py
from kaoanime import utils


def test_all_public_symbols_exported():
    assert hasattr(utils, "load_image")
    assert hasattr(utils, "save_image")
    assert hasattr(utils, "get_transforms")
    assert hasattr(utils, "UnpairedImageDataset")
    assert hasattr(utils, "create_dataloader")
