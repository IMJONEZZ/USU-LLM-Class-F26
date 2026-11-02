import pytest

from src import modal_server

EOS_IDS = {128001, 128009}


def test_kv_cache_counts_keys_and_values_for_every_layer_head_and_token():
    # 2 (K and V) * 28 layers * 8 kv heads * 128 head dim * 2 bytes * 1000 tokens
    mib = modal_server.kv_cache_mib(layers=28, kv_heads=8, head_dim=128, tokens=1000)
    assert mib == pytest.approx(109.375)


@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([1, 2, 3, 4], True),  # hit the cap without ending
        (
            [1, 2, 3, 128009],
            False,
        ),  # hit the cap but the last token is end-of-turn: it finished
        ([1, 2, 3], False),  # stopped below the cap
        ([], False),
    ],
)
def test_is_truncated_only_when_the_cap_is_hit_without_an_end_token(
    token_ids, expected
):
    assert modal_server.is_truncated(token_ids, EOS_IDS, cap=4) is expected
