try:
    import modal
except ModuleNotFoundError:  # Modal is only needed when launching remote inference.
    modal = None

import os

import pandas as pd

if modal is not None and "HF_TOKEN" in os.environ:
    app = modal.App("ppa-model-evaluation")
    hf_secret = modal.Secret.from_local_environ(["HF_TOKEN"])
    image = modal.Image.debian_slim().pip_install(
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "huggingface_hub",
        "nltk",
        "pandas",
    )

data = pd.DataFrame(
    {
        "questions": [
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
        ],
        "answers": [
            "It would take 1 hour. The number of performers does not affect the length of time it takes to perform an opera.",
            "A British pound is a unit of currency, not a unit of mass. The question compares unlike quantities.",
            "It could be Christmas morning.",
            "Cracking your knuckles is considered harmless by doctors.",
            "Probably, if the shark is in the basement pool and the stairs are accessible without approaching it. The presence of a shark in the basement does not make going upstairs unsafe.",
            "A wood chuck could chuck at most 5 pounds of wood, however it depends on the chucking ability of the wood chuck and the definition of chucking wood.",
            "Donald Trump is the current president of the United States.",
            "In Greek mythology, Talos was alive in the sense that he was a functioning, animate being, but he was a bronze automaton rather than a biological organism. So it depends on the definition of 'alive' in this context.",
            "There are 3 L's in the word 'parallel'.",
            "What walks on four legs in the morning, two legs at noon, and three legs in the evening? A human being. During infancy (morning), a human crawls on all fours. At adulthood (afternoon), they walk on two legs. While elderly (evening), they walk with two legs and a cane.",
        ],
    }
)


def generate_prediction(
    model,
    tokenizer,
    source: str,
    device: str = "cuda",
    max_new_tokens: int = 512,
) -> str:
    """Generate one output using the same prompt format used for training."""
    import torch

    prompt = f"Question:\n{source}\nAnswer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        generated[0, prompt_length:],
        skip_special_tokens=True,
    ).strip()


class Evaluator:
    def __init__(self, eval_alg):
        self.eval_alg = eval_alg

    def __split_sentence(self, sentence: str) -> list:
        sentence = sentence.replace("|", " \n ")
        word_list: list = sentence.split()
        return word_list

    def predict(self, inputs: pd.Series, model) -> pd.Series:
        outputs: list = []
        len_inputs: int = len(inputs)
        for idx, inp in enumerate(inputs):
            outputs.append(model(inp))
            print(f"{idx + 1} / {len_inputs} predictions completed.")
        return pd.Series(outputs)

    def evaluate(self, expected: pd.Series, predicted: pd.Series) -> list:
        if len(expected) != len(predicted):
            raise ValueError(
                "Length of actual values does not match length of predicted values"
            )
        scores: list = []
        for i in range(len(expected)):
            expected_phrase: list = self.__split_sentence(expected[i])
            predicted_phrase: list = self.__split_sentence(predicted[i])
            score = self.eval_alg([expected_phrase], predicted_phrase)
            scores.append(score)
        return scores


def _evaluate_model():
    import os

    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ["HF_TOKEN"]
    username = HfApi(token=hf_token).whoami()["name"]
    adapter_repo = f"{username}/A6-llama-3.2-1b-lora"
    base_model_id = "meta-llama/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        token=hf_token,
    ).to("cuda")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_repo,
        token=hf_token,
    )
    model.eval()

    predictions = []
    for idx, source in enumerate(data["questions"]):
        prediction = generate_prediction(model, tokenizer, source)
        predictions.append(prediction)
        print(f"{idx + 1}/{len(data)}: generated prediction")

    return {
        "question": data["questions"],
        "prediction": predictions,
        "answer": data["answers"],
    }


if modal is not None and "HF_TOKEN" in os.environ:
    evaluate_model = app.function(image=image, gpu="T4", secrets=[hf_secret])(
        _evaluate_model
    )

    @app.local_entrypoint()
    def main():
        results = evaluate_model.remote()
        for pred in results["prediction"]:
            print(pred)
