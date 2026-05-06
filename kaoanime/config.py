# kaoanime/config.py
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DataConfig:
    root_a: str = MISSING
    root_b: str = MISSING
    test_a: str = MISSING
    test_b: str = MISSING
    batch_size: int = 1
    image_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainConfig:
    max_epochs: int = 200
    lr: float = 2e-4
    precision: str = "16-mixed"
    log_every_n_steps: int = 10


@dataclass
class EvalConfig:
    checkpoint: str = ""
    output_dir: str = "outputs/eval"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
