import pandas as pd

import src.main
from src.main import main


def test_main_selects_resumes_from_saved_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "resumes.csv"
    pd.DataFrame(
        {
            "ID": range(9),
            "Resume_str": ["plain resume text"] * 9,
            "Category": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        }
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(src.main, "DATA_PATH", csv_path)

    selected_resumes = main()

    assert len(selected_resumes) == 9
