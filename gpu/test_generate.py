"""Check result selection and failure reporting without downloading models."""

import pytest
from generate import largest_success, report


def test_largest_success_ignores_larger_failures():
    results = [
        {"model_id": "large", "parameters": 14_000_000_000, "status": "oom"},
        {"model_id": "small", "parameters": 500_000_000, "status": "success"},
        {"model_id": "medium", "parameters": 3_000_000_000, "status": "success"},
    ]
    assert largest_success(results) == "medium"


def test_no_success_is_explicit():
    with pytest.raises(ValueError, match="No successful"):
        largest_success([{"model_id": "failed", "parameters": 10, "status": "error"}])


def test_report_keeps_download_errors_distinct_from_oom():
    text = report(
        [
            {
                "model_id": "download",
                "parameters": None,
                "status": "error",
                "error": "offline",
            },
            {
                "model_id": "large",
                "parameters": 14_000_000_000,
                "status": "oom",
                "error": "CUDA OOM",
            },
        ]
    )
    assert "| download | unknown | error | offline |" in text
    assert "| large | 14.000 | oom | CUDA OOM |" in text
