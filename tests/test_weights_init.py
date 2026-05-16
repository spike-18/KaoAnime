import pytest
import torch
import torch.nn as nn

from kaoanime.models.weights_init import init_weights


def test_normal_init_sets_weight_std():
    m = nn.Conv2d(8, 16, 3, padding=1)
    init_weights(m, "normal", gain=0.02)
    std = m.weight.data.std().item()
    assert 0.005 < std < 0.04, f"Expected std≈0.02, got {std:.4f}"


def test_normal_init_zeros_bias():
    m = nn.Conv2d(8, 16, 3, padding=1)  # bias=True by default
    init_weights(m, "normal", gain=0.02)
    assert m.bias.data.abs().max().item() == 0.0


def test_normal_init_no_bias_layer():
    m = nn.Conv2d(8, 16, 3, padding=1, bias=False)
    init_weights(m, "normal", gain=0.02)  # must not crash
    std = m.weight.data.std().item()
    assert 0.005 < std < 0.04


def test_kaiming_init_sets_weight_std():
    # Conv2d(64, 128, 3): fan_in = 64*3*3 = 576
    # kaiming_normal (a=0.2, fan_in) → std = sqrt(2 / (576*(1+0.04))) ≈ 0.057
    m = nn.Conv2d(64, 128, 3, padding=1, bias=False)
    init_weights(m, "kaiming")
    std = m.weight.data.std().item()
    assert 0.03 < std < 0.15, f"Expected kaiming std ~0.057, got {std:.4f}"


def test_normal_init_mean_near_zero():
    # Normal(0, 0.02) should produce weights with mean very close to 0
    m = nn.Conv2d(64, 128, 3, padding=1, bias=False)
    init_weights(m, "normal", gain=0.02)
    mean = m.weight.data.mean().item()
    assert abs(mean) < 0.005, f"Expected mean≈0, got {mean:.4f}"


def test_spectral_norm_layer_init():
    m = nn.utils.spectral_norm(nn.Conv2d(8, 16, 3, padding=1))
    init_weights(m, "normal", gain=0.02)
    std = m.weight_orig.data.std().item()
    assert 0.005 < std < 0.04, f"weight_orig std should be ≈0.02, got {std:.4f}"


def test_unknown_init_type_raises():
    m = nn.Conv2d(8, 16, 3, padding=1)
    with pytest.raises(ValueError, match="Unknown init_type"):
        init_weights(m, "xavier")


def test_init_weights_applies_to_sequential():
    seq = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1, bias=False),
        nn.Conv2d(16, 8, 3, padding=1, bias=False),
    )
    init_weights(seq, "normal", gain=0.02)
    for layer in seq:
        std = layer.weight.data.std().item()
        assert 0.005 < std < 0.04


from kaoanime.models import (
    NOTPotential,
    PatchDiscriminator,
    ResNetDiscriminator,
    ResNetGenerator,
    UNetGenerator,
)


def _conv_weight_std(model: nn.Module) -> float:
    """Concatenate all Conv2d/ConvTranspose2d weights and return their std."""
    weights = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
            weights.append(w.data.flatten())
    assert weights, "No conv layers found"
    return torch.cat(weights).std().item()


def test_unet_generator_normal_init():
    g = UNetGenerator(num_filters=16)
    std = _conv_weight_std(g)
    assert std < 0.04, f"UNetGenerator std should be ≈0.02, got {std:.4f}"


def test_resnet_generator_normal_init():
    g = ResNetGenerator(num_filters=16, num_residual_blocks=2)
    std = _conv_weight_std(g)
    assert std < 0.04, f"ResNetGenerator std should be ≈0.02, got {std:.4f}"


def test_patch_discriminator_normal_init():
    d = PatchDiscriminator(num_filters=16)
    std = _conv_weight_std(d)
    assert std < 0.04, f"PatchDiscriminator std should be ≈0.02, got {std:.4f}"


def test_resnet_discriminator_kaiming_init():
    d = ResNetDiscriminator(num_filters=16)
    std = _conv_weight_std(d)
    assert std > 0.04, f"ResNetDiscriminator kaiming std should be >0.04, got {std:.4f}"


def test_not_potential_kaiming_init():
    f = NOTPotential(num_filters=16)
    std = _conv_weight_std(f)
    assert std > 0.04, f"NOTPotential kaiming std should be >0.04, got {std:.4f}"
