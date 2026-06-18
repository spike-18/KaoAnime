# Women-Only CelebA Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restrict CelebA domain A to female images (`Male == -1`) for both training and evaluation, unconditionally, by filtering inside `UnpairedImageDataset`.

**Architecture:** Two pure helpers in `kaoanime/utils/dataset.py` — one locates `list_attr_celeba.csv` by walking up from `root_a`, one parses it into a set of female `image_id`s. `UnpairedImageDataset.__init__` collects `root_a` and `extra_roots_a` separately, keeps only female ids from `root_a`, leaves `extra_roots_a` and domain B untouched, and raises `FileNotFoundError` if the CSV cannot be found. Because `train.py` and `eval.py` both pass the CelebA dir as the dataset's first positional arg, no entry-point changes are needed.

**Tech Stack:** Python 3, stdlib `csv`, PyTorch `Dataset`, pytest. All Python invoked via `uv run python` / `uv run pytest` (project convention — never bare `python`).

**Spec:** `docs/superpowers/specs/2026-05-17-women-only-celeba-design.md`

---

## File Structure

| File                        | Responsibility            | Change                                                                                                                                                       |
| --------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `kaoanime/utils/dataset.py` | Dataset + file collection | Add `import csv`, two module-level constants, `_find_celeba_attr_csv`, `_load_female_ids`; rework `UnpairedImageDataset.__init__` file collection            |
| `tests/test_dataset.py`     | Dataset unit tests        | Add tests for the two helpers + filtering behavior + fail-loud; add a CSV-writer helper and use it in the 5 existing tests that build `UnpairedImageDataset` |
| `tests/test_dataloader.py`  | DataLoader unit tests     | Add the same CSV-writer helper and use it in the 2 existing tests that build `UnpairedImageDataset`                                                          |

No changes to `train.py`, `eval.py`, `config.py`, or domain B handling.

---

### Task 1: `_load_female_ids` — parse the attribute CSV

**Files:**

- Modify: `kaoanime/utils/dataset.py` (add `import csv` and the function)
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_dataset.py`:

```python
import csv
import pytest
from kaoanime.utils.dataset import _load_female_ids


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def test_load_female_ids_selects_only_male_minus_one(tmp_path):
    csv_path = tmp_path / "list_attr_celeba.csv"
    # 'Male' is NOT the first attribute column — proves we key by header name.
    _write_csv(
        csv_path,
        ["image_id", "Attractive", "Male", "Young"],
        [
            ["000001.jpg", "1", "-1", "1"],   # female -> kept
            ["000002.jpg", "-1", "1", "1"],   # male   -> excluded
            ["000003.jpg", "1", "-1", "-1"],  # female -> kept
        ],
    )
    assert _load_female_ids(csv_path) == {"000001.jpg", "000003.jpg"}


def test_load_female_ids_raises_without_male_column(tmp_path):
    csv_path = tmp_path / "list_attr_celeba.csv"
    _write_csv(csv_path, ["image_id", "Young"], [["000001.jpg", "1"]])
    with pytest.raises(ValueError, match="Male"):
        _load_female_ids(csv_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dataset.py::test_load_female_ids_selects_only_male_minus_one tests/test_dataset.py::test_load_female_ids_raises_without_male_column -v`
Expected: FAIL with `ImportError: cannot import name '_load_female_ids'`

- [ ] **Step 3: Implement the function**

In `kaoanime/utils/dataset.py`, add `import csv` to the imports block (top of file, after `from pathlib import Path`):

```python
import csv
```

Then add this function immediately after the `_IMAGE_EXTENSIONS` constant (around line 13, before `_worker_processor`):

```python
def _load_female_ids(csv_path: Path) -> set[str]:
    """Return CelebA image_ids whose 'Male' attribute is -1 (female)."""
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "Male" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} has no 'Male' column; columns={reader.fieldnames}"
            )
        id_field = reader.fieldnames[0]
        return {row[id_field] for row in reader if row["Male"] == "-1"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py::test_load_female_ids_selects_only_male_minus_one tests/test_dataset.py::test_load_female_ids_raises_without_male_column -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add kaoanime/utils/dataset.py tests/test_dataset.py
git commit -m "feat: add _load_female_ids CelebA attribute parser"
```

---

### Task 2: `_find_celeba_attr_csv` — locate the CSV by walking up

**Files:**

- Modify: `kaoanime/utils/dataset.py` (add two constants + the function)
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_dataset.py`:

```python
from kaoanime.utils.dataset import _find_celeba_attr_csv


def test_find_attr_csv_walks_up_parents(tmp_path):
    # root_a is two levels below where the CSV lives — mirrors the real
    # layout: .../CelebA/list_attr_celeba.csv vs .../CelebA/img/img.
    (tmp_path / "list_attr_celeba.csv").write_text("image_id,Male\n")
    root_a = tmp_path / "img_align_celeba" / "img_align_celeba"
    root_a.mkdir(parents=True)
    found = _find_celeba_attr_csv(root_a)
    assert found == tmp_path / "list_attr_celeba.csv"


def test_find_attr_csv_returns_none_when_absent(tmp_path):
    root_a = tmp_path / "A"
    root_a.mkdir()
    assert _find_celeba_attr_csv(root_a) is None


def test_find_attr_csv_accepts_string_path(tmp_path):
    (tmp_path / "list_attr_celeba.csv").write_text("image_id,Male\n")
    root_a = tmp_path / "A"
    root_a.mkdir()
    assert _find_celeba_attr_csv(str(root_a)) == tmp_path / "list_attr_celeba.csv"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dataset.py -k find_attr_csv -v`
Expected: FAIL with `ImportError: cannot import name '_find_celeba_attr_csv'`

- [ ] **Step 3: Implement the function**

In `kaoanime/utils/dataset.py`, add these two constants directly below the `_IMAGE_EXTENSIONS` line:

```python
_ATTR_CSV_NAME = "list_attr_celeba.csv"
_ATTR_CSV_SEARCH_DEPTH = 5
```

Then add this function directly below `_load_female_ids`:

```python
def _find_celeba_attr_csv(root_a: str | Path) -> Path | None:
    """Search root_a and up to _ATTR_CSV_SEARCH_DEPTH parents for the CSV."""
    start = Path(root_a)
    for d in [start, *list(start.parents)[:_ATTR_CSV_SEARCH_DEPTH]]:
        candidate = d / _ATTR_CSV_NAME
        if candidate.is_file():
            return candidate
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -k find_attr_csv -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add kaoanime/utils/dataset.py tests/test_dataset.py
git commit -m "feat: add _find_celeba_attr_csv path walk-up locator"
```

---

### Task 3: Filter `root_a` in `UnpairedImageDataset.__init__`

This task changes `__init__` to make filtering mandatory. That immediately
breaks the 5 existing `test_dataset.py` tests and the 2 existing
`test_dataloader.py` tests that build `UnpairedImageDataset` without a CSV, so
this same task repairs them. The suite must be green before the commit.

**Files:**

- Modify: `kaoanime/utils/dataset.py` (`UnpairedImageDataset.__init__`)
- Test: `tests/test_dataset.py` (new behavior tests + repair 5 existing)
- Test: `tests/test_dataloader.py` (repair 2 existing)

- [ ] **Step 1: Write the failing new-behavior tests**

Add to the end of `tests/test_dataset.py`:

```python
def _make_celeba(tmp_path: Path, a_ids: list[str], b_count: int,
                  female: set[str] | None = None) -> tuple[Path, Path]:
    """Create A/ with a_ids, B/ with b_count images, and an attribute CSV
    at tmp_path. Every id in `female` (default: all a_ids) gets Male=-1."""
    female = set(a_ids) if female is None else female
    a_dir = tmp_path / "A"
    a_dir.mkdir()
    for name in a_ids:
        PILImage.new("RGB", (64, 64)).save(a_dir / name)
    b_dir = tmp_path / "B"
    b_dir.mkdir()
    for i in range(b_count):
        PILImage.new("RGB", (64, 64)).save(b_dir / f"{i}.jpg")
    rows = [[n, "-1" if n in female else "1"] for n in a_ids]
    _write_csv(tmp_path / "list_attr_celeba.csv", ["image_id", "Male"], rows)
    return a_dir, b_dir


def test_dataset_keeps_only_female_root_a(tmp_path):
    a_dir, b_dir = _make_celeba(
        tmp_path,
        ["f0.jpg", "m0.jpg", "f1.jpg"],
        b_count=2,
        female={"f0.jpg", "f1.jpg"},
    )
    ds = UnpairedImageDataset(a_dir, b_dir, image_size=64, train=False)
    names = sorted(p.name for p in ds._files_a)
    assert names == ["f0.jpg", "f1.jpg"]


def test_dataset_does_not_filter_extra_roots_a(tmp_path):
    a_dir, b_dir = _make_celeba(
        tmp_path, ["f0.jpg"], b_count=1, female={"f0.jpg"}
    )
    extra = tmp_path / "ffhq"
    extra.mkdir()
    PILImage.new("RGB", (64, 64)).save(extra / "ffhq0.jpg")  # not in CSV
    ds = UnpairedImageDataset(
        a_dir, b_dir, image_size=64, train=False, extra_roots_a=[extra]
    )
    names = sorted(p.name for p in ds._files_a)
    assert names == ["f0.jpg", "ffhq0.jpg"]


def test_dataset_raises_when_attr_csv_missing(tmp_path):
    a_dir = tmp_path / "A"
    a_dir.mkdir()
    PILImage.new("RGB", (64, 64)).save(a_dir / "0.jpg")
    b_dir = tmp_path / "B"
    b_dir.mkdir()
    PILImage.new("RGB", (64, 64)).save(b_dir / "0.jpg")
    with pytest.raises(FileNotFoundError, match="list_attr_celeba.csv"):
        UnpairedImageDataset(a_dir, b_dir, image_size=64, train=False)
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `uv run pytest tests/test_dataset.py -k "keeps_only_female or does_not_filter_extra or raises_when_attr_csv_missing" -v`
Expected: FAIL — `test_dataset_keeps_only_female_root_a` and
`test_dataset_does_not_filter_extra_roots_a` fail (no filtering yet), and
`test_dataset_raises_when_attr_csv_missing` fails (no exception raised).

- [ ] **Step 3: Implement the `__init__` change**

In `kaoanime/utils/dataset.py`, find the current first two lines of
`UnpairedImageDataset.__init__`:

```python
        self._files_a = _collect_files([root_a] + list(extra_roots_a or []))
        self._files_b = _collect_files([root_b] + list(extra_roots_b or []))
```

Replace those two lines with:

```python
        attr_csv = _find_celeba_attr_csv(root_a)
        if attr_csv is None:
            raise FileNotFoundError(
                f"{_ATTR_CSV_NAME} not found within {_ATTR_CSV_SEARCH_DEPTH} "
                f"parent levels of root_a={root_a!r}. Women-only filtering is "
                f"mandatory; see docs/superpowers/specs/"
                f"2026-05-17-women-only-celeba-design.md."
            )
        female_ids = _load_female_ids(attr_csv)
        root_a_files = [
            p for p in _collect_files([root_a]) if p.name in female_ids
        ]
        extra_a_files = _collect_files(list(extra_roots_a or []))
        self._files_a = sorted(root_a_files + extra_a_files)
        self._files_b = _collect_files([root_b] + list(extra_roots_b or []))
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -k "keeps_only_female or does_not_filter_extra or raises_when_attr_csv_missing" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full suite to observe the expected regressions**

Run: `uv run pytest tests/test_dataset.py tests/test_dataloader.py -v`
Expected: the 5 pre-existing `test_dataset.py` tests that call
`UnpairedImageDataset` (`test_len_is_max_of_both_domains`,
`test_getitem_returns_a_and_b_tensors`, `test_b_wraps_around_when_a_is_larger`,
`test_accepts_string_paths`, `test_a_wraps_around_when_b_is_larger`) and the 2
`test_dataloader.py` tests (`test_create_dataloader_respects_batch_size`,
`test_create_dataloader_drop_last`) now FAIL with `FileNotFoundError:
list_attr_celeba.csv ...`. This is expected — filtering is now mandatory.

- [ ] **Step 6: Repair the existing `test_dataset.py` tests**

In `tests/test_dataset.py`, replace the existing `_make_images` helper:

```python
def _make_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for i in range(count):
        PILImage.new("RGB", (64, 64), color=(i * 10, 0, 0)).save(directory / f"{i}.jpg")
```

with a version that also writes an attribute CSV (one parent level above the
A directory) marking every generated id female, so the existing assertions
about counts/wrap-around still hold:

```python
def _make_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for i in range(count):
        PILImage.new("RGB", (64, 64), color=(i * 10, 0, 0)).save(directory / f"{i}.jpg")
    if directory.name == "A":
        _write_csv(
            directory.parent / "list_attr_celeba.csv",
            ["image_id", "Male"],
            [[f"{i}.jpg", "-1"] for i in range(count)],
        )
```

(`_write_csv` was added in Task 1 Step 1 and is module-level in this file.)

- [ ] **Step 7: Repair the existing `test_dataloader.py` tests**

In `tests/test_dataloader.py`, add `import csv` at the top (after the existing
`from pathlib import Path`) and replace the existing `_make_images` helper:

```python
def _make_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for i in range(count):
        Image.new("RGB", (64, 64)).save(directory / f"{i}.jpg")
```

with:

```python
def _make_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for i in range(count):
        Image.new("RGB", (64, 64)).save(directory / f"{i}.jpg")
    if directory.name == "A":
        with open(directory.parent / "list_attr_celeba.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image_id", "Male"])
            w.writerows([[f"{i}.jpg", "-1"] for i in range(count)])
```

- [ ] **Step 8: Run the full suite to verify it is green**

Run: `uv run pytest tests/test_dataset.py tests/test_dataloader.py -v`
Expected: PASS — all tests pass (the repaired 7 plus the new behavior tests).

- [ ] **Step 9: Commit**

```bash
git add kaoanime/utils/dataset.py tests/test_dataset.py tests/test_dataloader.py
git commit -m "feat: filter CelebA domain A to female images (Male == -1)"
```

---

### Task 4: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass, no errors.

- [ ] **Step 2: Lint and format**

Run: `uvx ruff check kaoanime/utils/dataset.py tests/test_dataset.py tests/test_dataloader.py && uvx ruff format --check kaoanime/utils/dataset.py tests/test_dataset.py tests/test_dataloader.py`
Expected: no lint errors; format check passes. If `ruff format --check`
reports changes, run `uvx ruff format <files>`, re-run the suite (Step 1),
and amend nothing — make a new commit:

```bash
git add -u && git commit -m "style: ruff format women-only filtering changes"
```

- [ ] **Step 3: Real-data sanity check (manual, optional)**

If the real CelebA dataset is present, confirm filtering actually reduces the
domain-A count:

```bash
uv run python -c "
from kaoanime.utils.dataset import UnpairedImageDataset
ds = UnpairedImageDataset(
    '/beta/home/madorskii/datasets/CelebA/img_align_celeba/img_align_celeba',
    '/beta/home/madorskii/datasets/alignedanimefaces/safebooru_jpeg',
    image_size=128, train=True)
print('domain A files after filter:', len(ds._files_a))
"
```

Expected: a count well below the unfiltered 201,599 (CelebA is ~58% female,
so roughly ~115k–120k), printed with no exception.

---

## Self-Review

**1. Spec coverage:**

- Filter in dataset from attribute CSV → Tasks 1–3.
- Hardcoded/unconditional (no config field) → Task 3 `__init__` (no config touched).
- `_find_celeba_attr_csv` walk-up → Task 2.
- `_load_female_ids` keyed by `"Male"` header name → Task 1.
- `extra_roots_a` unfiltered, domain B untouched → Task 3 Step 3 + test `test_dataset_does_not_filter_extra_roots_a`.
- Fail loudly when CSV missing (`FileNotFoundError`) → Task 3 Step 3 + test `test_dataset_raises_when_attr_csv_missing`.
- No `"Male"` column → `ValueError` → Task 1 + test `test_load_female_ids_raises_without_male_column`.
- Files absent from CSV excluded (conservative) → Task 3 Step 3 list comprehension (`p.name in female_ids`) + test `test_dataset_keeps_only_female_root_a` (`m0.jpg` excluded).
- Auto-applies to train and eval (no entry-point change) → File Structure note; `train.py`/`eval.py` untouched because filtering keys on the dataset's `root_a`.
- All Python via `uv run` → every run/command uses `uv run` / `uvx`.
  No gaps.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code step shows complete code; every run step shows the exact command and expected output.

**3. Type/name consistency:** `_load_female_ids(csv_path: Path) -> set[str]`, `_find_celeba_attr_csv(root_a: str|Path) -> Path|None`, constants `_ATTR_CSV_NAME` / `_ATTR_CSV_SEARCH_DEPTH`, helpers `_write_csv` / `_make_celeba` — all names used identically across Tasks 1–4. `_write_csv` is defined in Task 1 Step 1 before Task 3 reuses it. `import csv` added to `dataset.py` (Task 1) and `tests/test_dataloader.py` (Task 3 Step 7); `tests/test_dataset.py` imports `csv` in Task 1 Step 1. Consistent.
