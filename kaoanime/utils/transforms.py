from typing import Literal

import torch
from torchvision.transforms import v2


def get_transforms(mode: Literal["train", "test"], image_size: int = 128) -> v2.Compose:
    normalize = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if mode == "train":
        return v2.Compose(
            [
                v2.Resize(image_size + 30),
                v2.RandomCrop(image_size),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                normalize,
            ]
        )
    return v2.Compose(
        [
            v2.Resize(image_size),
            v2.CenterCrop(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
