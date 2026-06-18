# kaoanime/config.py — unified config entry point
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from kaoanime.config_cyclegan import CycleGANConfig, CycleGANTrainConfig
from kaoanime.config_not import NOTConfig


@dataclass
class DataConfig:
    variant: str = "demo"  # "demo" (DVC-tracked sample) or "full" (downloaded)
    # Public Google Drive file id of demo.zip, fetched via gdown for the demo variant.
    demo_gdrive_id: str = "13VsEPFfbg1sIRloZDcNjJRyo2gh29VD2"
    root_a: str = "data/${data.variant}/trainA"
    root_b: str = "data/${data.variant}/trainB"
    test_a: str = "data/${data.variant}/testA"
    test_b: str = "data/${data.variant}/testB"
    extra_roots_a: list[str] = field(default_factory=list)
    extra_roots_b: list[str] = field(default_factory=list)
    batch_size: int = 64
    image_size: int = 128
    num_workers: int = 12
    pin_memory: bool = True
    align_a: bool = False
    # Fixed crop transform for domain B (anime). Pixel offsets from center (256, 256).
    anime_offset_x: int = 0
    anime_offset_y: int = -7


@dataclass
class EvalConfig:
    checkpoint: str = ""
    output_dir: str = "outputs/eval"
    input: str = ""
    direction: str = "a2b"
    align: bool = False


@dataclass
class Config:
    model_type: str = "not"  # "cyclegan" or "not"
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # CycleGAN-specific (ignored when model_type="not")
    train: CycleGANTrainConfig = field(default_factory=CycleGANTrainConfig)
    model: CycleGANConfig = field(default_factory=CycleGANConfig)
    # NOT-specific (ignored when model_type="cyclegan")
    not_: NOTConfig = field(default_factory=NOTConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
