from typing import Literal

from torchvision import transforms


def get_transforms(
    mode: Literal["train", "test"], image_size: int = 128
) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize(image_size + 30),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
