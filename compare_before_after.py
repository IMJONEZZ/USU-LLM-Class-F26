import os

import dspy

from src.prompt_optimization import (
    SpeakerAttribution,
    extract_labeled_lines,
    split_dataset,
)

lm = dspy.LM(
    "groq/openai/gpt-oss-20b",
    api_key=os.environ["GROQ_API_KEY"],
    max_tokens=1000,
    reasoning_effort="low",
)
dspy.configure(lm=lm)

lines = extract_labeled_lines()
train, dev, test = split_dataset(lines)

baseline_predictor = dspy.Predict(SpeakerAttribution)
optimized_predictor = dspy.Predict(SpeakerAttribution)
optimized_predictor.load("optimized_prompt.json")

challenging_examples = [test[1], test[4]]

for example in challenging_examples:
    print(f"Line: {example['text']}")
    print(f"Actual speaker: {example['speaker']}")

    baseline_result = baseline_predictor(dialogue_line=example["text"])
    print(f"Before optimization: {baseline_result.speaker}")

    optimized_result = optimized_predictor(dialogue_line=example["text"])
    print(f"After optimization:  {optimized_result.speaker}")
    print("=" * 60)
