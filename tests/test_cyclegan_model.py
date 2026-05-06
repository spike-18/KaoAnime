from __future__ import annotations

import torch

from kaoanime.models import PatchDiscriminator, ResNetGenerator


def test_resnet_generator_output_shape() -> None:
    model = ResNetGenerator()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.shape == x.shape


def test_resnet_generator_output_range() -> None:
    model = ResNetGenerator()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.min() >= -1.0
    assert out.max() <= 1.0


def test_resnet_generator_custom_blocks() -> None:
    model = ResNetGenerator(num_residual_blocks=6)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.shape == x.shape


def test_patch_discriminator_output_shape() -> None:
    model = PatchDiscriminator()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.shape == (1, 1, 14, 14)


def test_patch_discriminator_batch() -> None:
    model = PatchDiscriminator()
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 14, 14)
