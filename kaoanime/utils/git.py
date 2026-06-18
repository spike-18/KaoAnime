from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_git_commit() -> str:
    """Return the current git commit SHA, or "unknown" if it cannot be resolved.

    Resolved against the repository root so it works regardless of the process
    working directory (e.g. Hydra changes cwd to the run output dir).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()
