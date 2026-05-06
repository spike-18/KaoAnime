# tests/test_model.py
import pytest
import torch
from kaoanime.config import Config
from kaoanime.model import DummyGenerator, KaoAnimeModel


def test_dummy_generator_preserves_shape():
    gen = DummyGenerator()
    x = torch.randn(1, 3, 64, 64)
    assert gen(x).shape == x.shape


def test_kaoanime_model_training_step_returns_scalar():
    model = KaoAnimeModel(Config())
    batch = {"A": torch.randn(1, 3, 64, 64), "B": torch.randn(1, 3, 64, 64)}
    loss = model.training_step(batch, 0)
    assert loss.ndim == 0


def test_config_defaults():
    cfg = Config()
    assert cfg.train.precision == "16-mixed"
    assert cfg.train.max_epochs == 200
    assert cfg.train.lr == pytest.approx(2e-4)
    assert cfg.data.batch_size == 1
    assert cfg.data.image_size == 128
    assert cfg.eval.output_dir == "outputs/eval"
