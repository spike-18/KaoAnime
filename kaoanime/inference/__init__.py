from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from kaoanime.utils import get_transforms, load_image, save_image
from kaoanime.utils.align import AlignFaceProcessor

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run_inference(
    model,
    input_path: str | Path,
    output_dir: str | Path,
    image_size: int = 128,
    direction: str = "a2b",
    device: str = "cuda",
    align: bool = False,
) -> list[Path]:
    """Translate images from domain A→B (or B→A) and save to output_dir.

    Args:
        model: Loaded KaoAnimeModel or NOTModel instance.
        input_path: Path to a single image file or a directory of images.
        output_dir: Directory where translated images are written.
        image_size: Size used for center-crop resize preprocessing.
        direction: "a2b" uses g_ab/T (selfie→anime); "b2a" uses g_ba (anime→selfie,
                   CycleGAN only — NOT model is unidirectional).
        device: Torch device string, e.g. "cuda" or "cpu".
        align: Apply MediaPipe landmark alignment before the transform pipeline.
               Images where no face is detected fall back to standard centre-crop.

    Returns:
        List of output file paths that were written.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transforms("test", image_size=image_size)
    if direction not in {"a2b", "b2a"}:
        raise ValueError(f"direction must be 'a2b' or 'b2a', got {direction!r}")
    if direction == "a2b":
        generator = model.g_ab if hasattr(model, "g_ab") else model.T
    else:
        if not hasattr(model, "g_ba"):
            raise ValueError(
                "b2a direction is not supported for NOT model (unidirectional transport)"
            )
        generator = model.g_ba

    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS
        )

    if not paths:
        raise FileNotFoundError(
            f"No images found in {input_path!r} (supported: {_IMAGE_EXTS})"
        )

    aligner = AlignFaceProcessor() if align else None

    written: list[Path] = []
    with torch.no_grad():
        for p in paths:
            img = load_image(p)
            if aligner is not None:
                arr = np.array(img)
                aligned_arr = aligner.align(arr, size=image_size)
                if aligned_arr is not None:
                    img = Image.fromarray(aligned_arr)
                # face not detected — fall back to standard centre-crop
            tensor = transform(img).unsqueeze(0).to(device)
            out = generator(tensor.float())[0]
            save_image(out, output_dir / p.name)
            written.append(output_dir / p.name)

    if aligner is not None:
        aligner.close()

    return written
