import sys
from types import SimpleNamespace

from src.gpu_test import test_gpu as modal_gpu_function


def test_gpu_reports_cuda_information(monkeypatch, capsys):
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda: "Test GPU",
        memory_allocated=lambda: 1234,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    modal_gpu_function.local()

    assert capsys.readouterr().out.splitlines() == [
        "CUDA available: True",
        "GPU: Test GPU",
        "1234",
    ]
