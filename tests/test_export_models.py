
from scripts.export_models import stage_models


def test_stage_copies_checkpoints_and_returns_paths(tmp_path):
    ckpt1 = tmp_path / "a.ckpt"
    ckpt2 = tmp_path / "b.ckpt"
    ckpt1.write_bytes(b"x")
    ckpt2.write_bytes(b"y")
    out = tmp_path / "models" / "export"
    staged = stage_models([str(ckpt1), str(ckpt2)], out_dir=str(out))
    assert [p.name for p in staged] == ["a.pt", "b.pt"]
    assert all(p.exists() for p in staged)


def test_stage_missing_checkpoint_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        stage_models([str(tmp_path / "nope.ckpt")], out_dir=str(tmp_path / "out"))
