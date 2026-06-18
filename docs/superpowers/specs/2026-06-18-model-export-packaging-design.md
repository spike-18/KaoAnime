# Model Export & Production Packaging — Design

**Date:** 2026-06-18
**Status:** Approved (pending spec review)

## Goal

Package the trained NOT model for production per the course "Model production
packaging" requirement:

- Export the transport map to **ONNX** (a standalone script).
- Provide a **TensorRT** conversion **script** (no engine built in this repo).
- Ship a **dependency-light (torch-free) preprocessing/postprocessing** module and
  an **onnxruntime-based inference** entry point.
- Document **Production preparation** (steps + delivery bundle) and **Infer** in the
  README.

## Context & Constraints

- The only network needed for inference is the NOT transport map `T`
  (`UNetGenerator`): input `(N, 3, 128, 128)` in `[-1, 1]`, output the same shape
  in `[-1, 1]` (tanh). The potential `f` is training-only.
- **Alignment cannot go into ONNX.** It uses MediaPipe Face Landmarker — a separate
  non-PyTorch model (`face_landmarker.task`) — which cannot be traced into a torch
  graph. So preprocessing is shipped as code + asset, not converted to ONNX.
- Existing preprocessing is torch-coupled: `utils/transforms.py` (torchvision) and
  `utils/image.py` (torch tensors). `utils/align.py` is **torch-free**
  (cv2 / mediapipe / numpy) and is reused as-is.
- Environment: GPU (A100) + CUDA 12.8 present; `onnx`/`onnxruntime` not installed;
  TensorRT (`tensorrt`/`trtexec`) not installed — so TensorRT is script-only.
- TensorRT uses FP16 (no INT8), so no calibration/example data is needed in DVC.
- Exported artifacts are uploaded to the `models` Google Drive folder **manually**
  by the maintainer (DVC `models` remote needs OAuth, deliberately avoided).

## Decisions

1. **ONNX export = separate script** `scripts/export_onnx.py` (fire CLI). Exports
   only `model.T` with a dynamic batch axis; verifies parity against PyTorch via
   onnxruntime.
2. **TensorRT = shell script** `scripts/export_tensorrt.sh` wrapping
   `trtexec --onnx=… --fp16 --saveEngine=…`. Documented; not executed here;
   `tensorrt` is **not** a project dependency.
3. **Torch-free production path** in a new `kaoanime/serving/` package:
   numpy/cv2/PIL preprocessing (reusing `AlignFaceProcessor` for optional
   alignment) + an onnxruntime runner. No torch/lightning imports.
4. **Production inference entry point** `infer_onnx.py` at repo root (public API),
   using `kaoanime.serving` + onnxruntime only.
5. Existing `infer.py` (torch/Lightning, checkpoint-based) stays unchanged for
   development use.

## Architecture

### New dependencies

`onnx`, `onnxruntime` (runtime). `tensorrt` is intentionally not added.

### `scripts/export_onnx.py` (fire CLI)

`export(checkpoint, out, t_filters=48, image_size=128, opset=17)`:

- Load `NOTModel.load_from_checkpoint(checkpoint, cfg=…, strict=False)` on CPU,
  `model.eval()`, take `model.T`.
- `torch.onnx.export` with `dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}`,
  input names `["input"]`, output `["output"]`, dummy `(1, 3, image_size, image_size)`.
- Parity check: run PyTorch `T` and an `onnxruntime.InferenceSession` on the same
  random input; assert max abs diff < 1e-3. Raise if it fails.
- Write to `out` (default `models/export/model.onnx`).

### `scripts/export_tensorrt.sh`

Thin wrapper (positional args: onnx path, engine path):

```bash
trtexec --onnx="$1" --fp16 --saveEngine="$2" \
    --minShapes=input:1x3x128x128 --optShapes=input:8x3x128x128 --maxShapes=input:16x3x128x128
```

Documented in README; requires TensorRT installed on the target machine.

### `kaoanime/serving/` (torch-free)

- `preprocess.py`
  - `preprocess_image(path, image_size=128, align=False) -> np.ndarray` →
    `(1, 3, image_size, image_size)` float32 in `[-1, 1]`. Steps: load (PIL/cv2),
    optional `AlignFaceProcessor` align (fallback to center-crop on no-face),
    resize to `image_size`, scale to `[-1, 1]`, HWC→CHW, add batch dim. Must match
    `transforms.get_transforms("test")` numerically.
  - `postprocess(array) -> np.ndarray` → `(H, W, 3)` uint8 from `(1|.,3,H,W)` in
    `[-1, 1]`.
- `onnx_runner.py`
  - `OnnxModel(onnx_path)` wrapping `onnxruntime.InferenceSession`; `run(array)`
    returns the output array.

No module here imports torch. `kaoanime/__init__.py` must stay import-light (verify
it does not pull torch) so `import kaoanime.serving…` is torch-free.

### `infer_onnx.py` (repo root, public API)

fire CLI `main(onnx, input, output_dir, image_size=128, align=False)`:
iterate input file/dir → `preprocess_image` → `OnnxModel.run` → `postprocess` →
save. Depends only on `kaoanime.serving`, onnxruntime, numpy, cv2/PIL, mediapipe.

### Delivery bundle (README "Production preparation")

`model.onnx` + `models/face_landmarker.task` + `kaoanime/serving/` +
runtime deps `onnxruntime, opencv-python, mediapipe, numpy, pillow`. Optionally a
TensorRT `.engine` built on the target via `export_tensorrt.sh`.

## Data Flow

| Step   | Command                                                                         |
| ------ | ------------------------------------------------------------------------------- |
| Export | `uv run python scripts/export_onnx.py --checkpoint <ckpt> --out model.onnx`     |
| TRT    | `bash scripts/export_tensorrt.sh model.onnx model.engine` (target machine)      |
| Serve  | `uv run python infer_onnx.py --onnx model.onnx --input <path> --output-dir out` |

## Error Handling

- Missing checkpoint / onnx file → clear `FileNotFoundError`.
- ONNX parity check failure → raise with the measured max diff.
- No face detected during alignment → fall back to center crop (same as the torch path).

## Testing

- `export_onnx`: on a tiny `NOTModel` (small `t_filters`) export to a temp path and
  assert the file exists and onnxruntime output matches PyTorch within tolerance.
- `serving.preprocess`: `preprocess_image` output shape `(1,3,128,128)`, dtype
  float32, range within `[-1, 1]`. Parity against `transforms.get_transforms("test")`
  on a sample image is **approximate** (PIL/numpy resize ≠ torchvision resize
  bit-for-bit): assert mean abs diff < ~2e-2, not exact equality.
- `serving.postprocess`: round-trips `[-1,1]` → uint8 correctly (shape/range).
- `OnnxModel`: runs a trivially exported identity-ish ONNX and returns expected shape.
- All external-free (no network); torch used only to build the parity reference.

## Out of Scope

- Building/verifying a TensorRT engine here (script + docs only).
- Inference server (Triton / MLflow Serving) — separate requirement.
- Changes to `infer.py` or the training pipeline.
