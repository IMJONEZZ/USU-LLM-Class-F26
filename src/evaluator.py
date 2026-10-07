try:
    import modal
except ModuleNotFoundError:  # Modal is only needed when launching remote inference.
    modal = None

import os

import nltk.translate.bleu_score as bleu
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
        "input": [
            "While holding the pin, decentralize the power kit then rotate the lever 90 degrees.",
            "Lock the position in place.",
            "Proceed to load the excavator after completing Step 5.4.1.",
            "Verify or sign the previous step.",
            "IF the lever is in the locked position, THEN operate the load as previously specified.",
            "Remove the 8000 lb tool as directed by the engineer, ensuring safety for the next operator.",
            "Do not forget to connect the screws when operating the following steps:",
            "It is important to ensure that the safety trainings are completed in order to proceed.",
            "Leave the machine arm resting on the floor then raise the arm once you are instructed by the procedural lead.",
            "Follow the instructions and ensure safety precautions are kept simulaneously.",
            "IF operators are unable to do the testing, THEN assign a testing phase during the next procedure.",
            "You will want to grade performance since it is mandatory to qualify for completion.",
            "Do not observe the tube directly, observe from a distance (at least 20 feet).",
            "if the project is completed, then include the signatures from every single engineer.",
            "Try not to forget locking the door after use.",
            "You must hold the tool sturdily. Immediately after, operate the crane with caution.",
            "Lift, observe, and replace the coil.",
            "Mark the tool and casing for operational purposes.",
            "Stop the rotation of the bearing once the speed hits 1200 RPMs (or after 5 minutes).",
            "Exit the building, leaving the hazardous material behind.",
        ],
        "expected": [
            "**DECENTRALIZE** the power kit while holding the pin.|**ROTATE** the lever 90 degrees.",
            "**LOCK** the position in place.",
            "**LOAD** the excavator after completing Step 5.4.1.",
            "**VERIFY** or **SIGN** the previous step.",
            "IF the lever is in the locked position, THEN operate the load as previously specified.",
            "**REMOVE** the 8000 lb tool as directed by the engineer, ensuring safety for the next operator.",
            "**CONNECT** the screws when operating the following steps:",
            "**ENSURE** the safety trainings are completed before proceeding.",
            "**LEAVE** the machine arm on the floor.|**RAISE** the arm once instructed by the procedural lead.",
            "**FOLLOW** the instructions and **ENSURE** safety precautions are kept simultaneously.",
            "IF operators are unable to test, THEN assign a testing phase during the next procedure.",
            "**GRADE** performance to qualify for completion.",
            "**OBSERVE** from a distance (at least 20 feet).",
            "IF the project is completed, THEN include the signatures from each engineer.",
            "**LOCK** the door after use.",
            "**HOLD** the tool sturdily.|**OPERATE** the crane with caution immediately after.",
            "**LIFT** the coil.|**OBSERVE** the coil.|**REPLACE** the coil.",
            "**MARK** the tool and casing for operational purposes.",
            "**STOP** the rotation of the bearing once the speed hits 1200 RPMs (or after 5 minutes).",
            "**EXIT** the building.|**LEAVE** the hazardous material behind.",
        ],
        "predicted": [
            "**HOLD** the pin while decentralizing the power kit.|**ROTATE** the lever 90 degrees.",
            "**LOCK** the position in place.",
            "**LOAD** the excavator after completing Step 5.4.1.",
            "**VERIFY** or **SIGN** the previous step.",
            "IF the lever is in the locked position, THEN **OPERATE** the load as previously specified.",
            "**REMOVE** the 8000 lb tool as directed by the engineer.|**ENSURE** safety for the next operator.",
            "Do not forget to connect the screws when operating the following steps:",
            "**ENSURE** that the safety trainings are completed.",
            "Leave the machine arm resting on the floor.|**RAISE** the arm once you are instructed by the procedural lead.",
            "**FOLLOW** the instructions.|**ENSURE** safety precautions are kept simultaneously.",
            "**ASSIGN** a testing phase during the next procedure if operators are unable to do the testing.",
            "**GRADE** performance since it is mandatory to qualify for completion.",
            "Do not observe the tube directly.|**OBSERVE** from a distance (at least 20 feet).",
            "if the project is completed, then include the signatures from every single engineer.",
            "Try to remember locking the door after use.",
            "**HOLD** the tool sturdily.|**OPERATE** the crane with caution.",
            "**LIFT** the coil. **OBSERVE** the coil. **REPLACE** the coil.",
            "**MARK** the tool and casing for operational purposes.",
            "**STOP** the rotation of the bearing once the speed hits 1200 RPMs (or after 5 minutes).",
            "**EXIT** the building.|**LEAVE** the hazardous material behind.",
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

    prompt = f"Convert to PPA:\n{source}\nOutput:\n"
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
    adapter_repo = f"{username}/ppa-llama-3.2-1b-lora"
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
    for idx, source in enumerate(data["input"]):
        prediction = generate_prediction(model, tokenizer, source)
        predictions.append(prediction)
        print(f"{idx + 1}/{len(data)}: generated prediction")

    expected = pd.Series(data["expected"])
    predicted = pd.Series(predictions)
    evaluator = Evaluator(bleu.sentence_bleu)
    scores = evaluator.evaluate(expected, predicted)
    average_score = sum(scores) / len(scores)

    for idx, score in enumerate(scores):
        print(f"Expected: {expected[idx]}\nPredicted: {predicted[idx]}\nBLEU: {score}")
    print(f"Average BLEU: {average_score}")

    return {
        "predictions": predictions,
        "scores": scores,
        "average_bleu": average_score,
    }


if modal is not None:
    evaluate_model = app.function(image=image, gpu="T4", secrets=[hf_secret])(
        _evaluate_model
    )

    @app.local_entrypoint()
    def main():
        results = evaluate_model.remote()
        print(f"Average BLEU: {results['average_bleu']}")
