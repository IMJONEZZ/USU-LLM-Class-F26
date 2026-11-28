import json
import os

import requests

from src.schemas import MCQ, HistoryRecord, Inputs, Question

GATEWAY_URL = "https://madisonbsperry--vllm-inference-get-idx.us-east.modal.run"


def transform_for_api(history_path: str) -> list[HistoryRecord]:
    with open(history_path, "r") as h:
        history = []
        for line in h:
            record = json.loads(line)

            key = record["key"]
            answer = record["answer"]
            distractors = [
                record["options"][option]
                for option in record["options"]
                if option != key
            ]

            skipped = record["status"] != "answered"
            line_history = HistoryRecord(
                question=MCQ(
                    stem=record["question"],
                    key=record["options"][key],
                    distractors=distractors,
                ),
                topic=record["concept"],
                response=record["options"][answer],
                confidence=record["confidence"],
                correct=record["correct"],
                skipped=skipped,
                date=record["ts"],
            )
            history.append(line_history)

    return history


def call_api(questions: list[Question], history_path: str):

    history = transform_for_api(history_path)

    question_input = Inputs(candidate_questions=questions, history=history)

    headers = {
        "Modal-Key": os.environ["MODAL_ID"],
        "Modal-Secret": os.environ["MODAL_SECRET"],
    }

    response = requests.post(
        url=GATEWAY_URL,
        json=question_input.model_dump(mode="json"),
        headers=headers,
        timeout=300,
    )
    response.raise_for_status()

    response_idx = response.json()["best_question"]
    return response_idx
