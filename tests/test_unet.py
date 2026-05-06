import torch

from kaoanime.models.unet import UNetGenerator


def test_unet_output_shape_matches_input():
    gen = UNetGenerator()
    x = torch.randn(1, 3, 128, 128)
    assert gen(x).shape == (1, 3, 128, 128)


def test_unet_output_in_tanh_range():
    gen = UNetGenerator()
    out = gen(torch.randn(1, 3, 128, 128))
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_unet_smaller_num_down():
    gen = UNetGenerator(num_filters=32, num_down=4)
    assert gen(torch.randn(1, 3, 64, 64)).shape == (1, 3, 64, 64)


def test_unet_parameter_count_scales_with_filters():
    small = UNetGenerator(num_filters=32)
    large = UNetGenerator(num_filters=64)
    assert sum(p.numel() for p in small.parameters()) < sum(
        p.numel() for p in large.parameters()
    )
