import torch
import torch.nn as nn
from torch import Tensor


class CycleGANLoss(nn.Module):
    """CycleGAN combined loss (LSGAN variant).

    Args:
        lambda_cycle: Weight for cycle-consistency loss.
        lambda_identity: Weight for identity loss.
    """

    def __init__(
        self, lambda_cycle: float = 10.0, lambda_identity: float = 5.0
    ) -> None:
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.gan = nn.MSELoss()
        self.cycle = nn.L1Loss()
        self.identity = nn.L1Loss()

    def generator(
        self,
        real_a: Tensor,
        real_b: Tensor,
        fake_a: Tensor,
        fake_b: Tensor,
        rec_a: Tensor,
        rec_b: Tensor,
        idt_a: Tensor,
        idt_b: Tensor,
        disc_fake_a: Tensor,
        disc_fake_b: Tensor,
    ) -> Tensor:
        """Compute combined generator loss.

        Args:
            real_a: Real images from domain A.
            real_b: Real images from domain B.
            fake_a: G_BA(real_b) — fake A images.
            fake_b: G_AB(real_a) — fake B images.
            rec_a: G_BA(fake_b) — reconstructed A images.
            rec_b: G_AB(fake_a) — reconstructed B images.
            idt_a: G_BA(real_a) — identity mapping for A.
            idt_b: G_AB(real_b) — identity mapping for B.
            disc_fake_a: Discriminator output for fake_a.
            disc_fake_b: Discriminator output for fake_b.

        Returns:
            Scalar tensor with the total generator loss.
        """
        loss_gan = self.gan(disc_fake_a, torch.ones_like(disc_fake_a)) + self.gan(
            disc_fake_b, torch.ones_like(disc_fake_b)
        )

        loss_cycle = self.lambda_cycle * (
            self.cycle(rec_a, real_a) + self.cycle(rec_b, real_b)
        )

        loss_identity = self.lambda_identity * (
            self.identity(idt_a, real_b) + self.identity(idt_b, real_a)
        )

        return loss_gan + loss_cycle + loss_identity

    def discriminator(
        self,
        disc_real_a: Tensor,
        disc_fake_a: Tensor,
        disc_real_b: Tensor,
        disc_fake_b: Tensor,
    ) -> Tensor:
        """Compute combined discriminator loss (LSGAN, real=1, fake=0, scaled by 0.5).

        Args:
            disc_real_a: Discriminator output for real A images.
            disc_fake_a: Discriminator output for fake A images.
            disc_real_b: Discriminator output for real B images.
            disc_fake_b: Discriminator output for fake B images.

        Returns:
            Scalar tensor with the total discriminator loss.
        """
        loss_d_a = 0.5 * (
            self.gan(disc_real_a, torch.ones_like(disc_real_a))
            + self.gan(disc_fake_a, torch.zeros_like(disc_fake_a))
        )
        loss_d_b = 0.5 * (
            self.gan(disc_real_b, torch.ones_like(disc_real_b))
            + self.gan(disc_fake_b, torch.zeros_like(disc_fake_b))
        )
        return loss_d_a + loss_d_b
