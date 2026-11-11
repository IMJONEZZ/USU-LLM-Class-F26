import hashlib
from pathlib import Path

import pytest

from src import modal_server

VENDORED = Path(__file__).parent.parent / "third_party" / "instruction_following_eval"


def test_kv_cache_counts_keys_and_values_for_every_layer_head_and_token():
    # 2 (K and V) * 28 layers * 8 kv heads * 128 head dim * 2 bytes * 1000 tokens
    mib = modal_server.kv_cache_mib(layers=28, kv_heads=8, head_dim=128, tokens=1000)
    assert mib == pytest.approx(109.375)


def test_vendored_checker_files_match_recorded_hashes():
    # Google's files must stay byte-for-byte unedited; any change goes in src/checks.py instead
    for line in (VENDORED / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        assert hashlib.sha256((VENDORED / name).read_bytes()).hexdigest() == expected, (
            name
        )
