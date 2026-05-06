from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from kaoanime.utils.dataloader import create_dataloader
from kaoanime.utils.dataset import UnpairedImageDataset


def _make_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for i in range(count):
        Image.new("RGB", (64, 64)).save(directory / f"{i}.jpg")


def test_create_dataloader_returns_dataloader():
    ds = TensorDataset(torch.zeros(10, 3, 64, 64))
    dl = create_dataloader(
        ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False
    )
    assert isinstance(dl, DataLoader)


def test_create_dataloader_respects_batch_size(tmp_path):
    _make_images(tmp_path / "A", 4)
    _make_images(tmp_path / "B", 4)
    ds = UnpairedImageDataset(
        tmp_path / "A", tmp_path / "B", image_size=64, train=False
    )
    dl = create_dataloader(
        ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False
    )
    batch = next(iter(dl))
    assert batch["A"].shape == (2, 3, 64, 64)
    assert batch["B"].shape == (2, 3, 64, 64)


def test_create_dataloader_drop_last(tmp_path):
    # 3 samples, batch_size=2 -> drop_last means only 1 batch of 2
    _make_images(tmp_path / "A", 3)
    _make_images(tmp_path / "B", 3)
    ds = UnpairedImageDataset(
        tmp_path / "A", tmp_path / "B", image_size=64, train=False
    )
    dl = create_dataloader(
        ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False
    )
    batches = list(dl)
    assert len(batches) == 1
    assert batches[0]["A"].shape[0] == 2
