import pytest
import onnxruntime as ort
import torch

from kaoanime.models import UNetGenerator
from scripts.export_onnx import export_module


def test_export_module_writes_onnx_and_passes_parity(tmp_path):
    transport = UNetGenerator(num_filters=8)
    out = tmp_path / "m.onnx"
    # export_module runs an internal parity check and raises on mismatch
    export_module(transport, str(out), image_size=128)
    assert out.exists()


def test_exported_onnx_supports_dynamic_batch(tmp_path):
    transport = UNetGenerator(num_filters=8)
    out = tmp_path / "m.onnx"
    export_module(transport, str(out), image_size=128)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for batch in (1, 4):
        result = sess.run(None, {"input": torch.randn(batch, 3, 128, 128).numpy()})[0]
        assert result.shape == (batch, 3, 128, 128)


def test_export_guards_against_norm_mismatch(tmp_path):
    from kaoanime.config import Config
    from kaoanime.model_not import NOTModel
    from scripts.export_onnx import export

    cfg = Config()
    cfg.not_.t_filters = 8
    cfg.not_.t_norm = "instance"
    ckpt = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": NOTModel(cfg).state_dict()}, ckpt)

    # matching norm exports fine
    out_ok = tmp_path / "ok.onnx"
    export(str(ckpt), out=str(out_ok), t_filters=8, t_norm="instance")
    assert out_ok.exists()

    # mismatched norm fails loudly (missing T.* keys)
    with pytest.raises(ValueError, match="transport"):
        export(str(ckpt), out=str(tmp_path / "bad.onnx"), t_filters=8, t_norm="batch")
