from concurrent.futures import ThreadPoolExecutor

import requests

API_URL = "https://loganmons01--example-vllm-inference-server.us-east.modal.direct/v1/completions"
MODEL_NAME = "A6-llama-3.2-1b-lora"

QUESTIONS = [
    "If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?",
    "Is a pound of feathers or a British pound heavier?",
    "A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on?",
    "What happens if you crack your knuckles a lot?",
    "If there is a shark in the pool of my basement, is it safe to go upstairs?",
    "How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world?",
    "Who is the current President of the United States?",
    "Was Talos alive?",
    "How many Ls are in the word parallel?",
    "What is the riddle of the sphinx, and what are all possible answers satisfying all conditions?",
]


def ask(question: str) -> str:
    response = requests.post(
        API_URL,
        json={
            "model": MODEL_NAME,
            "prompt": f"Question:\n{question}\nAnswer:\n",
            "max_tokens": 512,
            "temperature": 0,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"].strip()


def main() -> None:
    # ThreadPoolExecutor runs these blocking requests concurrently and keeps
    # results in the same order as QUESTIONS.
    with ThreadPoolExecutor(max_workers=len(QUESTIONS)) as executor:
        answers = list(executor.map(ask, QUESTIONS))

    for index, (question, answer) in enumerate(zip(QUESTIONS, answers), start=1):
        print(f"{index}. Question: {question}\n   Answer: {answer}\n")


if __name__ == "__main__":
    main()
