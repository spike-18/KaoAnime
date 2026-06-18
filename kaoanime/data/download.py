from __future__ import annotations

import zipfile
from pathlib import Path


def _download_demo_zip(gdrive_id: str, dest: Path) -> None:
    """Download the public demo.zip from Google Drive (no auth) and unzip it.

    The archive contains a top-level ``demo/`` directory, so extracting into
    ``dest`` yields ``dest/demo/{trainA,trainB,testA,testB}``.
    """
    import gdown

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "demo.zip"
    gdown.download(id=gdrive_id, output=str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    zip_path.unlink()


def _download_anime(dest: Path) -> None:
    """Download reitanaka/alignedanimefaces via the Kaggle API into dest."""
    import kaggle

    dest.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()  # raises a clear error if ~/.kaggle/kaggle.json is missing
    kaggle.api.dataset_download_files(
        "reitanaka/alignedanimefaces", path=str(dest), unzip=True
    )


def _download_celeba(dest: Path) -> None:
    """Download CelebA aligned images + attribute CSV from the CUHK Google Drive."""
    import gdown

    dest.mkdir(parents=True, exist_ok=True)
    # CelebA aligned&cropped folder (img_align_celeba + annotations) on Google Drive.
    gdown.download_folder(
        id="0B7EVK8r0v71pWEZsZE9oNnFzTm8",
        output=str(dest),
        quiet=False,
        use_cookies=False,
    )


def _layout_full(dest: Path) -> None:
    """Arrange downloaded CelebA/anime into dest/full/{trainA,...} via the CLI helper."""
    from scripts.prepare_dataset import full as _full

    _full(
        src_a=str(dest / "img_align_celeba"),
        src_b=str(dest / "safebooru_jpeg"),
        attr_csv=str(dest / "list_attr_celeba.csv"),
        out=str(dest / "full"),
    )


def download_data(variant: str, dest: str = "data", demo_gdrive_id: str = "") -> None:
    """Fetch the dataset for the given variant.

    demo -> download the public demo.zip from Google Drive (gdown, no auth).
    full -> download CelebA (gdown) + AlignedAnimeFaces (kaggle), then lay out.
    """
    if variant == "demo":
        if not demo_gdrive_id:
            raise ValueError(
                "demo_gdrive_id is required for the demo variant; set "
                "cfg.data.demo_gdrive_id to the public demo.zip Google Drive file id."
            )
        _download_demo_zip(demo_gdrive_id, Path(dest))
        return
    if variant == "full":
        dest_path = Path(dest)
        _download_celeba(dest_path)
        _download_anime(dest_path)
        _layout_full(dest_path)
        return
    raise ValueError(f"Unknown variant {variant!r}; expected 'demo' or 'full'")
