# kaoanime/config.py
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    root_a: str = "data/selfie2anime/trainA"
    root_b: str = "data/selfie2anime/trainB"
    test_a: str = "data/selfie2anime/testA"
    test_b: str = "data/selfie2anime/testB"
    extra_roots_a: list[str] = field(default_factory=lambda: ["data/flickrfaceshq/resized"])
    extra_roots_b: list[str] = field(default_factory=lambda: ["data/animefacedataset/images"])
    batch_size: int = 4
    image_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainConfig:
    max_epochs: int = 50
    lr: float = 2e-4
    lr_decay_start_epoch: int = 30
    beta1: float = 0.5
    precision: str = "16-mixed"
    log_every_n_steps: int = 20
    mlflow_tracking_uri: str = "http://10.0.111.233:9999"
    log_image_every_n_steps: int = 300
    fid_every_n_steps: int = 2000
    fid_num_images: int = 512


@dataclass
class EvalConfig:
    checkpoint: str = ""
    output_dir: str = "outputs/eval"
    input: str = ""
    direction: str = "a2b"


@dataclass
class ModelConfig:
    num_filters: int = 128
    num_residual_blocks: int = 9
    generator: str = "resnet"       # "resnet" or "unet"
    discriminator: str = "patch"    # "patch" or "resnet"
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
