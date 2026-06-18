# Women-Only CelebA Filtering — Design

**Date:** 2026-05-17
**Status:** Approved (pending spec review)

## Goal

Train and evaluate the selfie→anime models using only female CelebA images
(domain A). The CelebA real-face distribution is mixed-gender; restricting
domain A to women narrows the input distribution and is expected to help the
translation models converge.

## Scope

- **In scope:** Filtering domain A (CelebA) to female images for both training
  (`root_a`) and evaluation/inference (`test_a`).
- **Out of scope:** Domain B (anime) is never filtered. `extra_roots_a`
  (e.g. FFHQ) have no CelebA attribute file and are never filtered. A
  pre-existing bug in `eval.py:15` (passes `transform` where the
  `UnpairedImageDataset` signature expects `image_size`) is unrelated to this
  work and is **not** addressed here.

## Decisions (from brainstorming)

1. **Filter scope:** Training **and** eval/test, so the domain-A input
   distribution is consistent everywhere.
2. **Config control:** **Hardcoded women-only.** No config field; filtering is
   unconditional. Training on the full set again would require a code change
   (acceptable per the user).
3. **Approach:** Filter inside `UnpairedImageDataset` by parsing
   `list_attr_celeba.csv` at construction time. Single source of truth,
   auto-applies to eval, no manual preprocessing step.

## Architecture

All logic lives in `kaoanime/utils/dataset.py`. Filtering is keyed on whichever
directory is passed as the dataset's `root_a`. Because `train.py` passes
`cfg.data.root_a` and `eval.py` passes `cfg.data.test_a` as the first
positional argument to `UnpairedImageDataset`, filtering inside the dataset
covers both paths with **no changes to `train.py` or `eval.py`**.

### Components (all in `kaoanime/utils/dataset.py`)

1. **`_find_celeba_attr_csv(root_a: str | Path) -> Path | None`**
   Walks from `root_a` upward (the directory itself plus up to 5 parent
   levels) looking for a file named `list_attr_celeba.csv`. Returns the first
   match as a `Path`, or `None` if none is found within the search depth.

2. **`_load_female_ids(csv_path: Path) -> set[str]`**
   Parses the CSV with the stdlib `csv.DictReader`. Returns the set of
   `image_id` values whose `"Male"` column equals `"-1"`. Keys off the
   `"Male"` header **by name**, not a hardcoded column index, so it is robust
   to CelebA's column ordering. If the CSV has no `"Male"` column, raise a
   `ValueError` with a clear message.

3. **`UnpairedImageDataset.__init__` change**
   - Collect `root_a` files and `extra_roots_a` files **separately** (instead
     of the current single merged `_collect_files([root_a] + extra_roots_a)`).
   - Locate the attribute CSV via `_find_celeba_attr_csv(root_a)`:
     - **Found:** build the female-id set via `_load_female_ids`; keep only
       `root_a` files whose `Path.name` is in that set. `extra_roots_a` files
       pass through unfiltered.
     - **Not found:** raise `FileNotFoundError` with a clear message naming
       `root_a` and the filename searched for. A mandatory filter that
       silently degraded to "all images" would defeat the convergence goal, so
       it fails loudly rather than training on the wrong distribution.
   - Concatenate `[filtered root_a files] + [extra_roots_a files]` and sort for
     deterministic ordering. `_files_b` construction is unchanged.

Domain B (`root_b`, `extra_roots_b`) is untouched.

## Data Flow

| Caller        | First positional arg → dataset `root_a` | Result            |
| ------------- | --------------------------------------- | ----------------- |
| `train.py:27` | `cfg.data.root_a` (CelebA train dir)    | Filtered to women |
| `eval.py:15`  | `cfg.data.test_a` (CelebA `test/` dir)  | Filtered to women |

Both CelebA directories use image-id filenames (e.g. `000088.jpg`,
`202570.jpg`) that the single `list_attr_celeba.csv` covers, so one source of
truth serves both. The CSV (~202k rows) is parsed once in the main process
during dataset construction, before DataLoader workers fork — sub-second,
stdlib only, no per-worker cost.

## Error Handling

- **CSV not found** walking up from `root_a` → `FileNotFoundError` with a
  message naming the searched filename and `root_a`.
- **CSV present but no `"Male"` column** → `ValueError` with a clear message.
- **A `root_a` file is absent from the CSV** → treated as not-female and
  excluded. Conservative: keeps the training distribution clean rather than
  admitting unlabeled images.

## Testing

Unit tests in `tests/test_dataset.py` using a temporary directory tree with a
tiny synthetic `list_attr_celeba.csv` and a few dummy image files:

1. `_load_female_ids` returns exactly the ids whose `"Male" == "-1"`, ignoring
   column order.
2. `_find_celeba_attr_csv` finds the CSV when it sits several parent levels
   above `root_a`, and returns `None` when it is absent.
3. `UnpairedImageDataset` keeps only female ids from `root_a` while leaving
   `extra_roots_a` files unfiltered, and never filters `root_b`.
4. Constructing `UnpairedImageDataset` with a `root_a` that has no reachable
   `list_attr_celeba.csv` raises `FileNotFoundError`.

All Python invoked via `uv run python` / `uv run pytest` per project
convention.
