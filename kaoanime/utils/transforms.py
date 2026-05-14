from typing import Literal

import torch
from torchvision.transforms import v2


def get_transforms(mode: Literal["train", "test"], image_size: int = 128) -> v2.Compose:
    normalize = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    shared = [
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ]
    if mode == "train":
        shared.insert(1, v2.RandomHorizontalFlip())
    return v2.Compose(shared)
