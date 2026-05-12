#!/usr/bin/env python
"""Offline batch face alignment — normalise a dataset directory to canonical crops.

Usage examples:
    # Real faces (CelebA) — MediaPipe landmark alignment
    uv run python scripts/align_dataset.py \\
        --input  /beta/home/madorskii/datasets/CelebA/img_align_celeba/img_align_celeba \\
        --output data/celeba_aligned \\
        --mode   real --size 128 --workers 8

    # Pre-aligned anime faces — centre-crop normalisation (no detection required)
    uv run python scripts/align_dataset.py \\
        --input  /beta/home/madorskii/datasets/alignedanimefaces/safebooru_jpeg \\
        --output data/anime_aligned \\
        --mode   center-crop --size 128 --workers 8
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── worker state (one per process) ────────────────────────────────────────────

_processor = None


def _init_real_worker() -> None:
    global _processor
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kaoanime.utils.align import AlignFaceProcessor
    _processor = AlignFaceProcessor()


# ── per-image functions ───────────────────────────────────────────────────────

def _process_real(args: tuple[Path, Path, int]) -> str:
    src, dst, size = args
    raw = cv2.imread(str(src))
    if raw is None:
        return f"SKIP {src.name} (unreadable)"
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    aligned = _processor.align(img, size)
    if aligned is None:
        return f"SKIP {src.name} (no face detected)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    return f"OK   {src.name}"


def _process_center(args: tuple[Path, Path, int]) -> str:
    src, dst, size = args
    raw = cv2.imread(str(src))
    if raw is None:
        return f"SKIP {src.name} (unreadable)"
    h, w = raw.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = raw[y0:y0 + s, x0:x0 + s]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return f"OK   {src.name}"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align faces in a dataset directory to a canonical crop."
    )
    parser.add_argument("--input",   required=True, type=Path,
                        help="Source image directory (searched recursively)")
    parser.add_argument("--output",  required=True, type=Path,
                        help="Destination directory (mirrors source structure)")
    parser.add_argument("--mode",    default="real",
                        choices=["real", "center-crop"],
                        help="'real': MediaPipe landmark alignment (CelebA, user photos). "
                             "'center-crop': fixed square crop (pre-aligned anime datasets).")
    parser.add_argument("--size",    default=128,   type=int,
                        help="Output image size in pixels (default 128)")
    parser.add_argument("--workers", default=4,     type=int,
                        help="Parallel worker processes (default 4)")
    args = parser.parse_args()

    paths = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    if not paths:
        print(f"No images found in {args.input}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(paths):,} images.  mode={args.mode}  size={args.size}  "
          f"workers={args.workers}")

    tasks = [
        (src, args.output / src.relative_to(args.input), args.size)
        for src in paths
    ]

    if args.mode == "real":
        fn, init = _process_real, _init_real_worker
    else:
        fn, init = _process_center, None

    ok = skip = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init) as pool:
        futs = {pool.submit(fn, t): t for t in tasks}
        for n, fut in enumerate(as_completed(futs), 1):
            msg = fut.result()
            if msg.startswith("OK"):
                ok += 1
            else:
                skip += 1
                print(f"  {msg}", file=sys.stderr)
            if n % 5_000 == 0 or n == len(tasks):
                print(f"  {n:,}/{len(tasks):,}  aligned={ok:,}  skipped={skip:,}")

    print(f"\nDone.  Aligned: {ok:,}   Skipped: {skip:,}")
    if skip and args.mode == "real":
        print("Skipped images had no detectable face and are NOT in the output directory.")


if __name__ == "__main__":
    main()
