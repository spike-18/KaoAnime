# kaoanime/config_cyclegan.py
from dataclasses import dataclass


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
class ModelConfig:
    num_filters: int = 64
    num_residual_blocks: int = 9
    generator: str = "unet"        # "resnet" or "unet"
    discriminator: str = "resnet"  # "patch" or "resnet"
    lambda_cycle: float = 10.0
    lambda_identity: float = 1.0
