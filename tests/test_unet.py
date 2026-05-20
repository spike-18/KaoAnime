import torch
import torch.nn as nn

from kaoanime.models import UNetGenerator


def test_unet_uses_batchnorm():
    model = UNetGenerator(num_filters=8)
    has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    has_in = any(isinstance(m, nn.InstanceNorm2d) for m in model.modules())
    assert has_bn, "UNetGenerator must use BatchNorm2d (Korotin reference)"
    assert not has_in, "UNetGenerator must not use InstanceNorm2d"


def test_unet_output_shape_and_range():
    model = UNetGenerator(num_filters=8)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3, 32, 32), f"Expected (2,3,32,32), got {out.shape}"
    assert out.min().item() >= -1.0, "Output below -1 (Tanh broken?)"
    assert out.max().item() <= 1.0, "Output above 1 (Tanh broken?)"
