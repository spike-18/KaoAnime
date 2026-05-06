import torch
import pytest

from kaoanime.losses import CycleGANLoss


def _make_inputs(seed: int = 0):
    """Return a dict of random tensors matching expected shapes."""
    g = torch.Generator().manual_seed(seed)

    def img():
        return torch.rand(1, 3, 128, 128, generator=g)

    def disc():
        return torch.rand(1, 1, 14, 14, generator=g)

    return {
        "real_a": img(),
        "real_b": img(),
        "fake_a": img(),
        "fake_b": img(),
        "rec_a": img(),
        "rec_b": img(),
        "idt_a": img(),
        "idt_b": img(),
        "disc_fake_a": disc(),
        "disc_fake_b": disc(),
        "disc_real_a": disc(),
        "disc_real_b": disc(),
    }


def test_generator_loss_is_scalar():
    loss_fn = CycleGANLoss()
    t = _make_inputs()
    result = loss_fn.generator(
        t["real_a"],
        t["real_b"],
        t["fake_a"],
        t["fake_b"],
        t["rec_a"],
        t["rec_b"],
        t["idt_a"],
        t["idt_b"],
        t["disc_fake_a"],
        t["disc_fake_b"],
    )
    assert result.ndim == 0, f"Expected 0-dim scalar, got shape {result.shape}"


def test_discriminator_loss_is_scalar():
    loss_fn = CycleGANLoss()
    t = _make_inputs()
    result = loss_fn.discriminator(
        t["disc_real_a"],
        t["disc_fake_a"],
        t["disc_real_b"],
        t["disc_fake_b"],
    )
    assert result.ndim == 0, f"Expected 0-dim scalar, got shape {result.shape}"


def test_lambda_cycle_scales_loss():
    """Increasing lambda_cycle must increase generator loss when cycle loss > 0."""
    t = _make_inputs(seed=42)
    # Use identical rec and real so cycle loss isolates lambda effect cleanly,
    # but keep idt different from real so identity loss is non-zero and stable.
    # Actually we keep all tensors random — GAN term is fixed by disc outputs,
    # cycle and identity terms vary. We compare two lambdas with same data.
    loss_fn_low = CycleGANLoss(lambda_cycle=10.0, lambda_identity=5.0)
    loss_fn_high = CycleGANLoss(lambda_cycle=20.0, lambda_identity=5.0)

    kwargs = dict(
        real_a=t["real_a"],
        real_b=t["real_b"],
        fake_a=t["fake_a"],
        fake_b=t["fake_b"],
        rec_a=t["rec_a"],
        rec_b=t["rec_b"],
        idt_a=t["idt_a"],
        idt_b=t["idt_b"],
        disc_fake_a=t["disc_fake_a"],
        disc_fake_b=t["disc_fake_b"],
    )

    loss_low = loss_fn_low.generator(**kwargs)
    loss_high = loss_fn_high.generator(**kwargs)

    assert loss_high > loss_low, (
        f"Expected higher lambda_cycle to give higher loss, "
        f"but got low={loss_low.item():.4f} >= high={loss_high.item():.4f}"
    )


def test_perfect_cycle_gives_zero_cycle_loss():
    """When rec_a == real_a and rec_b == real_b, cycle loss must be zero."""
    t = _make_inputs(seed=7)
    # Override rec tensors with exact copies of real tensors
    real_a = t["real_a"]
    real_b = t["real_b"]
    rec_a = real_a.clone()
    rec_b = real_b.clone()

    loss_fn = CycleGANLoss(lambda_cycle=10.0, lambda_identity=0.0)

    # Compute cycle loss component directly
    cycle_loss = loss_fn.cycle(rec_a, real_a) + loss_fn.cycle(rec_b, real_b)

    assert cycle_loss.item() == pytest.approx(0.0, abs=1e-6), (
        f"Expected zero cycle loss with identical tensors, got {cycle_loss.item()}"
    )
