import hashlib
from pathlib import Path

import pytest

from src import config, estimate, modal_server

VENDORED = Path(__file__).parent.parent / "third_party" / "instruction_following_eval"


def test_kv_cache_counts_keys_and_values_for_every_layer_head_and_token():
    # 2 (K and V) * 28 layers * 8 kv heads * 128 head dim * 2 bytes * 1000 tokens
    mib = modal_server.kv_cache_mib(layers=28, kv_heads=8, head_dim=128, tokens=1000)
    assert mib == pytest.approx(109.375)


def test_vllm_command_takes_model_dtype_and_length_from_config():
    command = modal_server.vllm_command()
    assert command[:3] == ["vllm", "serve", config.MODEL_NAME]
    assert command[command.index("--served-model-name") + 1] == config.SERVED_NAME
    assert command[command.index("--dtype") + 1] == config.DTYPE
    assert command[command.index("--max-model-len") + 1] == str(config.MAX_MODEL_LEN)
    assert command[command.index("--generation-config") + 1] == "vllm"


def test_estimate_sums_calls_and_dollars_by_model():
    gemini_row = {
        "model": "gemini",
        "calls": 2,
        "input_tokens": 1000,
        "output_tokens": 500,
        "basis": "assumed",
    }
    student_row = {"model": "student", "calls": 10, "seconds": 100, "basis": "assumed"}
    totals = estimate.estimate([student_row, gemini_row, gemini_row])
    per_call = (
        1000 * config.GEMINI_INPUT_PER_MILLION + 500 * config.GEMINI_OUTPUT_PER_MILLION
    ) / 1e6
    assert totals["student"]["calls"] == 10
    assert totals["student"]["dollars"] == pytest.approx(100 * config.T4_PER_SECOND)
    assert totals["gemini"]["calls"] == 4
    assert totals["gemini"]["dollars"] == pytest.approx(4 * per_call)


def test_vendored_checker_files_match_recorded_hashes():
    # Google's files must stay byte-for-byte unedited; any change goes in src/checks.py instead
    for line in (VENDORED / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        assert hashlib.sha256((VENDORED / name).read_bytes()).hexdigest() == expected, (
            name
        )
