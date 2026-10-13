import pytest

from tests.helpers import (
    ROOT,
)

# ---- the Modal container runs Python 3.11; the files it imports must parse there ----
_UNPARENTHESIZED_EXCEPT = r"^\s*except\s+[^(:\n]+,[^:\n]+:\s*$"


@pytest.mark.parametrize(
    "name",
    [
        "src/config.py",
        "src/data.py",
        "src/eval.py",
        "src/train.py",
        "scripts/modal_app.py",
    ],
)
def test_container_imported_files_avoid_syntax_newer_than_python_3_11(name):
    import re

    source = (ROOT / name).read_text(encoding="utf-8")
    offenders = [
        line for line in source.splitlines() if re.match(_UNPARENTHESIZED_EXCEPT, line)
    ]
    assert offenders == []
