from omegaconf import OmegaConf

from kaoanime.config import Config
from kaoanime.data import required_data_dirs


def test_demo_variant_paths_resolve():
    cfg = OmegaConf.structured(Config)
    assert cfg.data.variant == "demo"
    assert cfg.data.root_a == "data/demo/trainA"
    assert cfg.data.root_b == "data/demo/trainB"
    assert cfg.data.test_a == "data/demo/testA"
    assert cfg.data.test_b == "data/demo/testB"


def test_full_variant_paths_resolve():
    cfg = OmegaConf.structured(Config)
    cfg.data.variant = "full"
    assert cfg.data.root_a == "data/full/trainA"
    assert cfg.data.test_b == "data/full/testB"


def test_required_data_dirs_lists_four_split_dirs():
    cfg = OmegaConf.structured(Config)
    dirs = required_data_dirs(cfg)
    assert [str(d) for d in dirs] == [
        "data/demo/trainA",
        "data/demo/trainB",
        "data/demo/testA",
        "data/demo/testB",
    ]
