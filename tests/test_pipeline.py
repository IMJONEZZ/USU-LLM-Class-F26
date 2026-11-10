import pytest

from src import config, modal_server


def test_kv_cache_counts_keys_and_values_for_every_layer_head_and_token():
    # 2 (K and V) * 28 layers * 8 kv heads * 128 head dim * 2 bytes * 1000 tokens
    mib = modal_server.kv_cache_mib(layers=28, kv_heads=8, head_dim=128, tokens=1000)
    assert mib == pytest.approx(109.375)


def test_split_fractions_sum_to_one():
    # the second sequential split derives its size from these, so a drifted fraction would leave rows unassigned
    total = config.TRAIN_FRACTION + config.VALIDATION_FRACTION + config.TEST_FRACTION
    assert total == pytest.approx(1)
