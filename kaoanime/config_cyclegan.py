# kaoanime/config_cyclegan.py
from dataclasses import dataclass


@dataclass
class CycleGANTrainConfig:
    max_epochs: int = 100
    lr: float = 1e-3
    lr_decay_start_epoch: int = 30
    alpha: float = 0.993
    precision: str = "16-mixed"
    log_every_n_steps: int = 50
    mlflow_tracking_uri: str = "http://127.0.0.1:8080"
    gen_steps: int = 1
    disc_steps: int = 5
    log_image_every_n_steps: int = 5000
    fid_every_n_steps: int = 2000
    fid_num_images: int = 512
    beta1: float = 0.5
    resume_from_checkpoint: str = ""


@dataclass
class CycleGANConfig:
    num_filters: int = 48  # matches NOT t_filters default
    num_residual_blocks: int = 9
    generator: str = "unet"  # "resnet" or "unet"
    discriminator: str = "resnet"  # "patch" or "resnet"
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
