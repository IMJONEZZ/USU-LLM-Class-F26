from unittest.mock import patch

from src.gpu_check import check_gpu


def test_check_gpu():
    with (
        patch("src.gpu_check.subprocess.run") as mock_run,
        patch("src.gpu_check.time.sleep") as mock_sleep,
    ):
        check_gpu.local()

    mock_run.assert_called_once_with(["nvidia-smi"], check=True)
    mock_sleep.assert_called_once_with(30)
