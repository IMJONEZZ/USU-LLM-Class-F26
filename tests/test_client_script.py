from pathlib import Path

from src.client_script import call_api, transform_for_api
from src.schemas import MCQ, Question

GOOD_QUESTION = Question(
    question=MCQ(
        stem="What does HTTP stand for?",
        key="HyperText Transfer Protocol",
        distractors=[
            "High Transfer Text Protocol",
            "Hyper Tool Transfer Process",
            "Host Text Transfer Protocol",
        ],
    ),
    topic="web apis",
)
FAKE_DATA_PATH = Path(__file__).parent.parent / "fake_data.jsonl"


def test_transform_reads_correctly():
    history = transform_for_api(FAKE_DATA_PATH)
    assert (
        history[0].response
        == "Path segments are plain strings in the URL, so only scalar types can be parsed from them"
    )


def test_call_api_returns_int():
    several_questions = [GOOD_QUESTION for _ in range(3)]
    answer = call_api(questions=several_questions, history_path=FAKE_DATA_PATH)
    assert isinstance(answer, int)
