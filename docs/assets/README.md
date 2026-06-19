# README assets

Image files referenced by the top-level `README.md`. They are intentionally not
committed yet — add them here to fill the placeholders:

- `hero.png` — a before→after banner shown under the title (a selfie → anime strip,
  ~720 px wide).
- `anime_1.jpg`, `anime_2.jpg`, `anime_3.jpg` — anime outputs for the
  `examples/selfie_{1,2,3}.jpg` inputs, shown in the **Examples** table.

Generate the anime outputs from the bundled selfies, e.g. via the Triton client:

```bash
uv run python triton/client.py examples --output_dir docs/assets
```

then rename/crop as needed to match the names above.
