---
size_categories: n<1K
tags:
- synthetic
- distilabel
- rlaif
---

<p align="left">
  <a href="https://github.com/argilla-io/distilabel">
    <img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>
  </a>
</p>

# Dataset Card for synthetic_trick_questions

This dataset has been created with [distilabel](https://distilabel.argilla.io/).



## Dataset Summary

This dataset contains a `pipeline.yaml` which can be used to reproduce the pipeline that generated it in distilabel using the `distilabel` CLI:

```console
distilabel pipeline run --config "https://huggingface.co/datasets/synthetic_trick_questions/raw/main/pipeline.yaml"
```

or explore the configuration:

```console
distilabel pipeline info --config "https://huggingface.co/datasets/synthetic_trick_questions/raw/main/pipeline.yaml"
```

## Dataset structure

The examples have the following structure per configuration:


<details><summary> Configuration: default </summary><hr>

```json
{
    "category": "scaling_rate_trap",
    "distilabel_metadata": {
        "raw_input_text_generation_0": [
            {
                "content": "Write one short trick question similar in spirit to: \u0027If it takes 1 hour for 60 people to play an opera, how long for 600 people?\u0027 (a scenario where naive proportional scaling gives a wrong answer, because the real-world event has a fixed duration regardless of headcount). Then provide a correct, step-by-step reasoning answer. Do NOT reuse the opera example. Format exactly as \u0027Question: ...\u0027 then \u0027Answer: ...\u0027.",
                "role": "user"
            }
        ],
        "raw_output_text_generation_0": "Question: If 4 groups of 1500 students each attend a concert and meet every week, what is the average time required for 1000 students to participate?\n\nAnswer: \n\nTo solve this problem, we need to consider the total number of interactions between the students in all 4 groups.\n\nEach group meets once per week, so each student will appear in that meeting 7 times.\n\nSince there are 1500 students in each group, let\u0027s denote the total attendance of each group as Gx7 (where x represents the number of weeks).\nThen we multiply by (4) since there are 4 such groups \nSo 7x * 4 = Total Attendance = 2800.\n\n\n\nNow we divide Total Attendance by NumberOf Students which is 1000",
        "statistics_text_generation_0": {
            "input_tokens": 92,
            "output_tokens": 156
        }
    },
    "generation": "Question: If 4 groups of 1500 students each attend a concert and meet every week, what is the average time required for 1000 students to participate?\n\nAnswer: \n\nTo solve this problem, we need to consider the total number of interactions between the students in all 4 groups.\n\nEach group meets once per week, so each student will appear in that meeting 7 times.\n\nSince there are 1500 students in each group, let\u0027s denote the total attendance of each group as Gx7 (where x represents the number of weeks).\nThen we multiply by (4) since there are 4 such groups \nSo 7x * 4 = Total Attendance = 2800.\n\n\n\nNow we divide Total Attendance by NumberOf Students which is 1000",
    "instruction": "Write one short trick question similar in spirit to: \u0027If it takes 1 hour for 60 people to play an opera, how long for 600 people?\u0027 (a scenario where naive proportional scaling gives a wrong answer, because the real-world event has a fixed duration regardless of headcount). Then provide a correct, step-by-step reasoning answer. Do NOT reuse the opera example. Format exactly as \u0027Question: ...\u0027 then \u0027Answer: ...\u0027.",
    "model_name": "unsloth/Llama-3.2-1B-Instruct"
}
```

This subset can be loaded as:

```python
from datasets import load_dataset

ds = load_dataset("synthetic_trick_questions", "default")
```

Or simply as it follows, since there's only one configuration and is named `default`:

```python
from datasets import load_dataset

ds = load_dataset("synthetic_trick_questions")
```


</details>
