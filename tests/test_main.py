from src.main import run_demo


def test_main_runs_without_errors():
    preprocessed = ["Luke", ",", "I", "am", "your", "father", "."]
    run_demo(preprocessed, vocab_size=50)
