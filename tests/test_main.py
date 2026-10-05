import pandas as pd

import src.main
from src.main import main


def test_main_splits_sampled_resumes_from_saved_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "resumes.csv"
    pd.DataFrame(
        {
            "ID": range(30),
            "Resume_str": ["plain resume text"] * 30,
            "Category": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        }
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(src.main, "DATA_PATH", csv_path)

    train, validation, test = main()

    assert (len(train), len(validation), len(test)) == (21, 6, 3)
