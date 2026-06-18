# kaoanime/config_not.py
from dataclasses import dataclass


@dataclass
class NOTConfig:
    t_iters: int = (
        10  # T inner-loop updates per f update (Korotin et al. reference uses 10)
    )
    t_lr: float = 1e-4  # transport map (UNet) learning rate
    f_lr: float = 1e-4  # potential (NOTPotential) learning rate
    t_filters: int = 64  # UNet num_filters (reference: base_factor=48)
    f_filters: int = 64  # NOTPotential num_filters
    max_steps: int = 150001
    precision: str = "bf16-mixed"
    log_every_n_steps: int = 50
    mlflow_tracking_uri: str = "http://127.0.0.1:8080"
    log_image_every_n_steps: int = 500
    fid_every_n_steps: int = 500
    fid_num_images: int = 1024
    resume_from_checkpoint: str = ""
    t_grad_clip: float = 100.0
    beta1: float = 0.9  # Adam beta1 (Korotin reference default)
    beta2: float = 0.999  # Adam beta2 (Korotin reference default)
    lr_step_size: int = 75000  # StepLR decay interval in training steps
    lr_gamma: float = 0.5  # StepLR multiplicative decay factor
