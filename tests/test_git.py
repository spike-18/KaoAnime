import re

from kaoanime.utils.git import get_git_commit


def test_get_git_commit_returns_sha_or_unknown():
    sha = get_git_commit()
    # Inside this repo it is a 40-char hex SHA; the fallback is "unknown".
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha)
