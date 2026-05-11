from __future__ import annotations

from pathlib import Path

import torch

from kaoanime.utils import get_transforms, load_image, save_image

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run_inference(
    model,
    input_path: str | Path,
    output_dir: str | Path,
    image_size: int = 128,
    direction: str = "a2b",
    device: str = "cuda",
) -> list[Path]:
    """Translate images from domain A→B (or B→A) and save to output_dir.

    Args:
        model: Loaded KaoAnimeModel instance.
        input_path: Path to a single image file or a directory of images.
        output_dir: Directory where translated images are written.
        image_size: Size used for center-crop resize preprocessing.
        direction: "a2b" uses g_ab (selfie→anime); "b2a" uses g_ba (anime→selfie).
        device: Torch device string, e.g. "cuda" or "cpu".

    Returns:
        List of output file paths that were written.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transforms("test", image_size=image_size)
    generator = model.g_ab if direction == "a2b" else model.g_ba

    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)

    written: list[Path] = []
    with torch.no_grad():
        for p in paths:
            img = load_image(p)
            tensor = transform(img).unsqueeze(0).to(device)
            out = generator(tensor.float())[0]
            save_image(out, output_dir / p.name)
            written.append(output_dir / p.name)

    return written
