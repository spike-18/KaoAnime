"""FID accumulation/compute scheduling.

The naive approach (accumulate as soon as the buffer is empty, then idle until
the next compute) makes the reported FID reflect the generator as it was at the
*start* of each window — lagging real quality by a full ``fid_every_n_steps``.

These helpers instead place the accumulation window at the *end* of each window,
so ``fid.compute()`` always measures the current generator.

``step`` is the 1-based count of training batches processed so far.
"""

from __future__ import annotations


def fid_should_accumulate(step: int, n_fid: int, num_batches: int) -> bool:
    """True if FID images should be accumulated at this step.

    Accumulation happens only in the final ``num_batches`` steps that end
    exactly on a compute boundary (``step % n_fid == 0``).
    """
    if n_fid <= 0:
        return False
    pos = step % n_fid
    return pos == 0 or pos > n_fid - num_batches


def fid_should_compute(step: int, n_fid: int) -> bool:
    """True if ``fid.compute()`` should run at this step."""
    return n_fid > 0 and step % n_fid == 0
