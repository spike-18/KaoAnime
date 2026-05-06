# kaoanime/utils/image.py
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(tensor: Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = tensor.detach().cpu().clamp(-1.0, 1.0)
    img = ((img + 1.0) / 2.0 * 255.0).to(torch.uint8)
    img = img.permute(1, 2, 0).numpy()
    Image.fromarray(img, mode="RGB").save(path)
