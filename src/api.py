from random import randint

from fastapi import FastAPI

from schemas import Inputs

app = FastAPI()


@app.post("/question_selection")
def get_idx(inputs: Inputs) -> dict:
    # from prompts import create_prompts
    # from modal_server import send_to_modal
    # from stats import select_best
    # check_api_key()
    # prompts = create_prompts(inputs= inputs)
    # probs = send_to_modal(prompts = prompts)
    # random_int = select_best(probs = probs)

    random_int = randint(0, len(inputs.candidate_questions) - 1)

    return {"best_question": random_int}
