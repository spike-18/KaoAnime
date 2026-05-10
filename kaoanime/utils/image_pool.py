import random
import torch


class ImagePool:
    """Stores a rolling buffer of generated images for discriminator training.

    Implements the 50-image history pool from Shrivastava et al. used in
    the original CycleGAN paper to prevent discriminator oscillation.
    """

    def __init__(self, pool_size: int = 50) -> None:
        self.pool_size = pool_size
        self._pool: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Return a batch for discriminator training, mixing old and new images.

        Each image in the batch is independently either returned as-is (50%)
        or swapped with a random pool entry (50%). The pool is filled/updated
        with incoming images throughout.

        Args:
            images: Freshly generated fake images, shape (B, C, H, W).

        Returns:
            Tensor of same shape as input, detached, safe to pass to discriminator.
        """
        if self.pool_size == 0:
            return images.detach()
        result: list[torch.Tensor] = []
        for img in images.detach():
            img = img.unsqueeze(0)
            if len(self._pool) < self.pool_size:
                self._pool.append(img)
                result.append(img)
            elif random.random() > 0.5:
                idx = random.randint(0, self.pool_size - 1)
                swapped = self._pool[idx].clone()
                self._pool[idx] = img
                result.append(swapped)
            else:
                result.append(img)
        return torch.cat(result, dim=0)
