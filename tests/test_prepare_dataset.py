import csv
from pathlib import Path

import pytest

from scripts.prepare_dataset import build_layout


def _make_source(tmp_path: Path) -> tuple[Path, Path, Path]:
    src_a = tmp_path / "celeba"
    src_b = tmp_path / "anime"
    src_a.mkdir()
    src_b.mkdir()
    # 6 CelebA images: even ids female (Male=-1), odd ids male (Male=1)
    rows = []
    for i in range(6):
        name = f"{i:06d}.jpg"
        (src_a / name).write_bytes(b"a")
        rows.append({"image_id": name, "Male": "-1" if i % 2 == 0 else "1"})
    for i in range(6):
        (src_b / f"anime_{i}.jpg").write_bytes(b"b")
    csv_path = tmp_path / "list_attr_celeba.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "Male"])
        writer.writeheader()
        writer.writerows(rows)
    return src_a, src_b, csv_path


def test_demo_layout_counts_and_women_only(tmp_path):
    src_a, src_b, csv_path = _make_source(tmp_path)
    out = tmp_path / "out"
    build_layout(
        src_a=str(src_a),
        src_b=str(src_b),
        attr_csv=str(csv_path),
        out=str(out),
        n_train_a=2,
        n_train_b=2,
        n_test_a=1,
        n_test_b=1,
        seed=0,
    )
    train_a = sorted(p.name for p in (out / "trainA").iterdir())
    test_a = sorted(p.name for p in (out / "testA").iterdir())
    # only even (female) ids selected, train and test disjoint
    assert len(train_a) == 2 and len(test_a) == 1
    selected_a = set(train_a) | set(test_a)
    assert all(int(n[:6]) % 2 == 0 for n in selected_a)
    assert len(selected_a) == 3  # no overlap between train and test
    assert len(list((out / "trainB").iterdir())) == 2
    assert len(list((out / "testB").iterdir())) == 1
    assert (out / "list_attr_celeba.csv").exists()


def test_demo_layout_is_deterministic(tmp_path):
    src_a, src_b, csv_path = _make_source(tmp_path)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    for out in (out1, out2):
        build_layout(
            src_a=str(src_a),
            src_b=str(src_b),
            attr_csv=str(csv_path),
            out=str(out),
            n_train_a=2,
            n_train_b=2,
            n_test_a=1,
            n_test_b=1,
            seed=0,
        )
    assert sorted(p.name for p in (out1 / "trainA").iterdir()) == sorted(
        p.name for p in (out2 / "trainA").iterdir()
    )


def test_missing_source_raises(tmp_path):
    _, src_b, csv_path = _make_source(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_layout(
            src_a=str(tmp_path / "nope"),
            src_b=str(src_b),
            attr_csv=str(csv_path),
            out=str(tmp_path / "out"),
            n_train_a=1,
            n_train_b=1,
            n_test_a=1,
            n_test_b=1,
            seed=0,
        )
