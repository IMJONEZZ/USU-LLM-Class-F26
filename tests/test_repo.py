import subprocess

from tests.helpers import (
    ROOT,
)


# ---- M2b follow-ups from the real-data sample ----
def _git_ignored(path):
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", path], cwd=ROOT, check=False
        ).returncode
        == 0
    )


def test_data_dir_is_ignored():
    assert _git_ignored("data/train.jsonl")


def test_assets_dir_is_ignored():
    # decision: no dataset is committed, including the hand-written ones
    assert _git_ignored("assets/test_set.jsonl")
    assert _git_ignored("assets/brainteaser.jsonl")
