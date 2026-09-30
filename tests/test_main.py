import sys
from unittest.mock import MagicMock, patch

from src.main import BIT_WIDTHS, MODEL_CHECKPOINTS, main


def test_main_calls_generate_for_every_model_and_bit_width_combo():
    fake_torch = MagicMock()

    with (
        patch("src.main.generate") as mock_generate,
        patch.dict(sys.modules, {"torch": fake_torch}),
    ):
        mock_generate.local.return_value = True
        main.local()

    expected_calls = [
        (model_name, bit_width)
        for model_name in MODEL_CHECKPOINTS
        for bit_width in BIT_WIDTHS
    ]
    actual_calls = [call.args for call in mock_generate.local.call_args_list]
    assert actual_calls == expected_calls

    assert fake_torch.cuda.empty_cache.call_count == len(expected_calls)


def test_main_reports_no_fit_for_failed_attempts(capsys):
    fake_torch = MagicMock()
    failing_combo = (MODEL_CHECKPOINTS[-1], BIT_WIDTHS[-1])

    def fake_local(model_name, bit_width):
        return (model_name, bit_width) != failing_combo

    with (
        patch("src.main.generate") as mock_generate,
        patch.dict(sys.modules, {"torch": fake_torch}),
    ):
        mock_generate.local.side_effect = fake_local
        main.local()

    captured = capsys.readouterr()
    failing_name, failing_bits = failing_combo
    assert f"{failing_name} ({failing_bits}): NO FIT" in captured.out
