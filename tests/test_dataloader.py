from src.dataloader import _chunk_token_ids, create_dataloader_v1, create_dataloader_v2


def test_chunk_count_matches_formula():
    max_length = 4
    stride = 2
    token_ids = list(range(20))  # 20 fake tokens, values don't matter here

    input_chunks, target_chunks = _chunk_token_ids(token_ids, max_length, stride)

    expected_count = (len(token_ids) - max_length) // stride
    assert len(input_chunks) == expected_count
    assert len(target_chunks) == expected_count


def test_chunk_shapes_match_max_length():
    max_length = 4
    stride = 2
    token_ids = list(range(20))

    input_chunks, target_chunks = _chunk_token_ids(token_ids, max_length, stride)

    assert all(len(chunk) == max_length for chunk in input_chunks)
    assert all(len(chunk) == max_length for chunk in target_chunks)


def test_create_dataloader_v1_returns_batches_with_expected_shape():
    text = "The quick brown fox jumps over the lazy dog many times in a row"

    dataloader = create_dataloader_v1(
        text, batch_size=2, max_length=4, stride=2, shuffle=False, drop_last=True
    )
    input_batch, target_batch = next(iter(dataloader))

    assert input_batch.shape == (2, 4)
    assert target_batch.shape == (2, 4)


def test_create_dataloader_v2_returns_batches_with_max_length_columns():
    lines = ["The quick brown fox jumps over the lazy dog"] * 50

    dataloader = create_dataloader_v2(lines, max_length=4, stride=2)
    input_batch, target_batch = next(iter(dataloader))

    assert input_batch.shape[1] == 4
    assert target_batch.shape[1] == 4
    assert input_batch.shape[0] == target_batch.shape[0]


from src.dataloader import _group_lines_by_char_count


def test_group_lines_by_char_count_groups_until_threshold_reached():
    lines = ["aaa", "bbb", "ccc", "ddd"]

    blocks = _group_lines_by_char_count(lines, target_chars=5)

    assert blocks == ["aaa\nbbb", "ccc\nddd"]


def test_group_lines_by_char_count_keeps_leftover_partial_block():
    lines = ["aaa", "bbb", "ccc"]

    blocks = _group_lines_by_char_count(lines, target_chars=5)

    assert blocks == ["aaa\nbbb", "ccc"]


def test_create_dataloader_v2_handles_short_lines_without_crashing():
    lines = ["Hi.", "No.", "Yes.", "Okay."] * 150

    dataloader = create_dataloader_v2(lines)
    input_batch, target_batch = next(iter(dataloader))

    assert input_batch.shape[1] == 256
    assert target_batch.shape[1] == 256
    assert input_batch.shape[0] == target_batch.shape[0]
    assert input_batch.shape[0] > 0
