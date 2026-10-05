import pandas as pd

import src.main
from src.main import main


def test_main_cleans_resumes_from_saved_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "resumes.csv"
    pd.DataFrame(
        {
            "ID": [1, 2],
            "Resume_str": ["  Chef\xa0jane@example.com  ", "Cook (555) 123-4567"],
            "Category": ["CHEF", "CHEF"],
        }
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(src.main, "DATA_PATH", csv_path)

    cleaned_resumes = main()

    assert cleaned_resumes == ["Chef [EMAIL]", "Cook [PHONE]"]
