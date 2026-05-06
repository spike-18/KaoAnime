# Utils Design — KaoAnime

## Context

Greenfield project. No source code exists yet. All utility code lives in `kaoanime/utils/`; the `data/` directory is reserved for image files tracked by DVC.

## Module Layout

```
kaoanime/
  utils/
    __init__.py       # re-exports all public symbols
    image.py          # load_image, save_image
    transforms.py     # get_transforms
    dataset.py        # UnpairedImageDataset
    dataloader.py     # create_dataloader
```

## `image.py`

```python
def load_image(path: str | Path) -> Image.Image
def save_image(tensor: Tensor, path: str | Path) -> None
```

- `load_image` opens the file as RGB PIL Image. Callers apply transforms.
- `save_image` expects a float tensor in `[-1, 1]` (CycleGAN output range), denormalizes to `[0, 255]`, and saves via PIL. Creates parent directories if needed.

## `transforms.py`

```python
def get_transforms(mode: Literal["train", "test"], image_size: int = 128) -> transforms.Compose
```

- **train**: resize to `image_size + 30`, random crop to `image_size`, random horizontal flip, `ToTensor`, normalize to `[-1, 1]`.
- **test**: resize to `image_size`, center crop to `image_size`, `ToTensor`, normalize to `[-1, 1]`.

## `dataset.py`

```python
class UnpairedImageDataset(Dataset):
    def __init__(self, root_a: str | Path, root_b: str | Path, transform: Callable)
    def __getitem__(self, index: int) -> dict[str, Tensor]  # {"A": ..., "B": ...}
    def __len__(self) -> int  # max(len_a, len_b)
```

- Reads all image files from two directories independently.
- B is sampled as `index % len_b` (unpaired by design).
- `__len__` is `max(len_a, len_b)` so the larger domain is fully covered per epoch.

## `dataloader.py`

```python
def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader
```

Thin wrapper around `torch.utils.data.DataLoader` with `drop_last=True`.

## `__init__.py`

Re-exports: `load_image`, `save_image`, `get_transforms`, `UnpairedImageDataset`, `create_dataloader`.
