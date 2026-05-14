# kaoanime/config_not.py
from dataclasses import dataclass


@dataclass
class NOTConfig:
    t_iters: int = 10              # T inner-loop updates per f update (matches reference)
    t_lr: float = 1e-4             # transport map (UNet) learning rate
    f_lr: float = 1e-4             # potential (NOTPotential) learning rate
    t_filters: int = 48            # UNet num_filters (reference: base_factor=48)
    f_filters: int = 64            # NOTPotential num_filters
    max_steps: int = 100_001
    precision: str = "16-mixed"
    log_every_n_steps: int = 50
    mlflow_tracking_uri: str = "http://10.0.111.233:9999"
    log_image_every_n_steps: int = 1000
    fid_every_n_steps: int = 5000
    fid_num_images: int = 512
