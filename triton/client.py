#!/usr/bin/env python
"""Demo / test client for the KaoAnime Triton ensemble (selfie -> anime).

Accepts one or more image files and/or directories, sends the raw bytes to the
``kaoanime`` ensemble (preprocess -> transport -> postprocess) over HTTP, and
saves the anime-domain results to an output directory. Depends only on
``tritonclient[http]``, ``numpy`` and ``pillow`` — no torch.

Examples:
    # one file
    uv run python triton/client.py examples/selfie_1.jpg

    # several files
    uv run python triton/client.py examples/selfie_1.jpg examples/selfie_2.jpg

    # a whole directory, custom output dir
    uv run python triton/client.py examples --output_dir outputs/examples
"""

from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import tritonclient.http as httpclient
from PIL import Image

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_MODEL = "kaoanime"


def _collect_images(inputs: tuple[str, ...]) -> list[Path]:
    """Expand the given files/directories into a sorted list of image paths."""
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(
                sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
            )
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"No such file or directory: {item}")
    if not paths:
        raise FileNotFoundError(f"No images found in {list(inputs)}")
    return paths


def _infer_one(client: httpclient.InferenceServerClient, data: bytes) -> np.ndarray:
    """Run one image (raw encoded bytes) through the ensemble; return HWC uint8."""
    inp = httpclient.InferInput("IMAGE", [1, 1], "BYTES")
    inp.set_data_from_numpy(np.array([[data]], dtype=object))
    out = httpclient.InferRequestedOutput("OUTPUT_IMAGE")
    response = client.infer(_MODEL, inputs=[inp], outputs=[out])
    return response.as_numpy("OUTPUT_IMAGE")[0]


def main(
    *inputs: str,
    output_dir: str = "outputs/examples",
    url: str = "localhost:8000",
) -> None:
    if not inputs:
        raise ValueError(
            "Pass one or more image files or a directory, e.g.\n"
            "  uv run python triton/client.py examples/selfie_1.jpg\n"
            "  uv run python triton/client.py examples"
        )
    paths = _collect_images(inputs)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = httpclient.InferenceServerClient(url=url)
    if not client.is_model_ready(_MODEL):
        raise RuntimeError(
            f"Model {_MODEL!r} is not ready on {url}. Is the server up "
            "(bash triton/run_server.sh)?"
        )

    for image_path in paths:
        result = _infer_one(client, image_path.read_bytes())
        dst = out_dir / image_path.name
        Image.fromarray(result).save(dst)
        print(f"  {image_path} -> {dst}")
    print(f"Saved {len(paths)} image(s) to {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
