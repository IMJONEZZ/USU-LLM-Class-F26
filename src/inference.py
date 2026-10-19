__generated_with = "0.25.1"

# %%
from time import perf_counter

import requests

# uvx --link-mode=copy marimo export script marimo_notebooks/inference.py -o src/inference.py


# %%
def get_riddle_answer(riddle):
    started_at = perf_counter()
    response = requests.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        json={
            "model": "riddles",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Solve the user's riddle. Respond with only the name of the answer "
                        "in 1 to 3 words. Omit introductory text, explanations, and quotation marks."
                    ),
                },
                {"role": "user", "content": riddle},
            ],
            "temperature": 0,
            "max_tokens": 16,
        },
        timeout=120,
    )
    elapsed_seconds = perf_counter() - started_at
    response.raise_for_status()
    print(f"---------Riddle: {riddle}--------")
    print(f"Answer: {response.json()['choices'][0]['message']['content']}")
    print(f"Response time: {elapsed_seconds:.2f} seconds")


# %%
get_riddle_answer(
    "What walks on four legs in the morning, two legs at noon, and three legs in the evening?"
)
_riddles = [
    (
        (
            "What has roots as nobody sees,\n"
            "Is taller than trees,\n"
            "Up, up it goes,\n"
            "And yet never grows?"
        ),
        "A mountain",
    ),
    (
        (
            "Thirty white horses on a red hill,\n"
            "First they champ,\n"
            "Then they stamp,\n"
            "Then they stand still."
        ),
        "Teeth",
    ),
    (
        "Voiceless it cries,\nWingless flutters,\nToothless bites,\nMouthless mutters.",
        "Wind",
    ),
    (
        (
            "An eye in a blue face\n"
            "Saw an eye in a green face.\n"
            '"That eye is like to this eye"\n'
            "Said the first eye,\n"
            '"But in low place,\n'
            'Not in high place."'
        ),
        "The sun",
    ),
    (
        (
            "It cannot be seen, cannot be felt,\n"
            "Cannot be heard, cannot be smelt.\n"
            "It lies behind stars and under hills,\n"
            "And empty holes it fills.\n"
            "It comes first and follows after,\n"
            "Ends life, kills laughter."
        ),
        "Dark",
    ),
    (
        "A box without hinges, key, or lid,\nYet golden treasure inside is hid.",
        "An egg",
    ),
    (
        (
            "Alive without breath,\n"
            "As cold as death;\n"
            "Never thirsty, ever drinking,\n"
            "All in mail never clinking."
        ),
        "Fish",
    ),
    (
        (
            "No-legs lay on one-leg,\n"
            "two-legs sat near on three-legs,\n"
            "four-legs got some."
        ),
        "Fish on a table, man on a stool, cat gets the scraps",
    ),
    (
        (
            "This thing all things devours:\n"
            "Birds, beasts, trees, flowers;\n"
            "Gnaws iron, bites steel;\n"
            "Grinds hard stones to meal;\n"
            "Slays king, ruins town,\n"
            "And beats high mountain down."
        ),
        "Time",
    ),
]
for _riddle, _expected_answer in _riddles:
    print()
    get_riddle_answer(_riddle)
    print(f"Expected answer: {_expected_answer}")
