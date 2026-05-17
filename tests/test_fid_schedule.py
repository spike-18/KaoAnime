import math

from kaoanime.utils import fid_should_accumulate, fid_should_compute


def test_disabled_when_n_fid_non_positive():
    assert fid_should_accumulate(5, 0, 4) is False
    assert fid_should_compute(5, 0) is False
    assert fid_should_accumulate(5, -1, 4) is False


def test_accumulation_window_ends_at_compute_boundary():
    n_fid, num_batches = 1000, 4
    # Idle for the bulk of the window — NOT the first batches.
    for step in (1, 2, 500, 996):
        assert fid_should_accumulate(step, n_fid, num_batches) is False, step
    # Accumulate only in the last `num_batches` steps ending at the boundary.
    for step in (997, 998, 999, 1000):
        assert fid_should_accumulate(step, n_fid, num_batches) is True, step
    # Next window: idle again right after the boundary.
    assert fid_should_accumulate(1001, n_fid, num_batches) is False
    for step in (1997, 1998, 1999, 2000):
        assert fid_should_accumulate(step, n_fid, num_batches) is True, step


def test_compute_fires_exactly_on_boundary():
    assert fid_should_compute(1000, 1000) is True
    assert fid_should_compute(2000, 1000) is True
    assert fid_should_compute(999, 1000) is False
    assert fid_should_compute(1001, 1000) is False


def test_exactly_num_batches_accumulations_per_window():
    n_fid, num_batches = 1000, 4
    hits = [s for s in range(1, n_fid + 1) if fid_should_accumulate(s, n_fid, num_batches)]
    assert hits == [997, 998, 999, 1000]
    assert len(hits) == num_batches


def test_num_batches_at_least_covers_required_images():
    # 512 images, batch 128 -> 4 batches needed; window must yield >= that.
    num_batches = math.ceil(512 / 128)
    hits = [s for s in range(1, 2001) if fid_should_accumulate(s, 2000, num_batches)]
    # Two windows (compute at 2000 only within range start), 4 per window boundary.
    assert hits == [1997, 1998, 1999, 2000]


def test_num_batches_ge_window_accumulates_every_step():
    # Degenerate: need more batches than the window length -> accumulate always.
    for step in range(1, 21):
        assert fid_should_accumulate(step, 10, 50) is True
