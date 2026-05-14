# kaoanime/config.py
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    root_a: str = "/beta/home/madorskii/datasets/CelebA/img_align_celeba/img_align_celeba"
    root_b: str = "/beta/home/madorskii/datasets/alignedanimefaces/safebooru_jpeg"
    test_a: str = "/beta/home/madorskii/datasets/CelebA/test"
    test_b: str = "/beta/home/madorskii/datasets/alignedanimefaces/test"
    extra_roots_a: list[str] = field(default_factory=list)
    extra_roots_b: list[str] = field(default_factory=list)
    batch_size: int = 8
    image_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    align_a: bool = False
    # Fixed crop transform for domain B (anime). Fractions of image size —
    # resolution-independent, so these values work unchanged at 256×256.
    anime_scale  : float = 1.20
    anime_shift_x: float = 0.00
    anime_shift_y: float = -0.02


@dataclass
class TrainConfig:
    max_epochs: int = 100
    lr: float = 2e-4
    lr_decay_start_epoch: int = 30
    beta1: float = 0.5
    precision: str = "16-mixed"
    log_every_n_steps: int = 50
    mlflow_tracking_uri: str = "http://10.0.111.233:9999"
    gen_steps: int = 1
    disc_steps: int = 1
    log_image_every_n_steps: int = 5000
    fid_every_n_steps: int = 2000
    fid_num_images: int = 512


@dataclass
class EvalConfig:
    checkpoint: str = ""
    output_dir: str = "outputs/eval"
    input: str = ""
    direction: str = "a2b"
    align: bool = False


@dataclass
class ModelConfig:
    num_filters: int = 64
    num_residual_blocks: int = 9
    generator: str = "unet"       # "resnet" or "unet"
    discriminator: str = "resnet"    # "patch" or "resnet"
    lambda_cycle: float = 10.0
    lambda_identity: float = 10.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
