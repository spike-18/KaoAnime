# kaoanime/config.py — unified config entry point
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from kaoanime.config_cyclegan import CycleGANModelConfig, CycleGANTrainConfig
from kaoanime.config_not import NOTConfig


@dataclass
class DataConfig:
    root_a: str = "/beta/home/madorskii/datasets/CelebA/img_align_celeba/img_align_celeba"
    root_b: str = "/beta/home/madorskii/datasets/alignedanimefaces/safebooru_jpeg"
    test_a: str = "/beta/home/madorskii/datasets/CelebA/test"
    test_b: str = "/beta/home/madorskii/datasets/alignedanimefaces/test"
    extra_roots_a: list[str] = field(default_factory=list)
    extra_roots_b: list[str] = field(default_factory=list)
    batch_size: int = 64
    image_size: int = 128
    num_workers: int = 8
    pin_memory: bool = True
    align_a: bool = False
    # Fixed crop transform for domain B (anime). Fractions of image size —
    # resolution-independent, so these values work unchanged at 256×256.
    anime_scale  : float = 1.20
    anime_shift_x: float = 0.00
    anime_shift_y: float = -0.02


@dataclass
class EvalConfig:
    checkpoint: str = ""
    output_dir: str = "outputs/eval"
    input: str = ""
    direction: str = "a2b"
    align: bool = False


@dataclass
class Config:
    model_type: str = "cyclegan"   # "cyclegan" or "not"
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # CycleGAN-specific (ignored when model_type="not")
    train: CycleGANTrainConfig = field(default_factory=CycleGANTrainConfig)
    model: CycleGANModelConfig = field(default_factory=CycleGANModelConfig)
    # NOT-specific (ignored when model_type="cyclegan")
    not_: NOTConfig = field(default_factory=NOTConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
