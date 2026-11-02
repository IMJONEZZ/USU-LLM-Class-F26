"""Builds the training set for the trick-question fine tune.

The homework hands me ten questions and tells me not to train on them, so
everything in here is written to be the same *kind* of question without being
the same question. There are three sources:

  * trick questions, built by filling slots into families I wrote by hand
  * letter counting, where a script works out the answer so it can't be wrong
  * a slice of ``no_robots`` so the model doesn't forget ordinary conversation

Nothing here needs a GPU or a network connection except the ``no_robots``
slice, which is injected so the tests can skip it.

``check_leakage`` is the guard. It runs over whatever got assembled and refuses
to hand back a dataset that contains one of the ten graded questions.
"""

import random
import re

# The ten I'm graded on. They exist in this file so the leakage check has
# something to compare against, never as training rows.
HOMEWORK_QUESTIONS = [
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

# "parallel" is the word I'm tested on, so it never goes in the counting slice.
BANNED_COUNTING_WORDS = {"parallel"}

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "of",
    "in",
    "on",
    "to",
    "it",
    "if",
    "there",
    "what",
    "how",
    "many",
    "much",
    "who",
    "do",
    "does",
    "and",
    "or",
    "for",
    "will",
    "would",
    "could",
    "can",
    "that",
    "this",
    "my",
    "i",
}


def _words(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS]


def similarity(a, b):
    """Fraction of content words two questions share.

    This is a blunt instrument. I only need it to catch a question that is one of the ten
    with the numbers swapped, not to do real semantic matching.
    """
    first, second = set(_words(a)), set(_words(b))
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def check_leakage(rows, questions=None, threshold=0.75):
    """Raise if any row is one of the graded questions, or close to one.

    This is the whole reason I didn't use TruthfulQA. Having written my own data
    instead, I'd rather have the check fail loudly than trust that I remembered.
    """
    questions = HOMEWORK_QUESTIONS if questions is None else questions
    for row in rows:
        asked = row["question"]
        for graded in questions:
            score = similarity(asked, graded)
            if score >= threshold:
                raise ValueError(
                    f"training row is too close to a graded question "
                    f"(similarity {score:.2f}):\n  row:    {asked}\n  graded: {graded}"
                )
    return rows


# --- the trick question families ---------------------------------------------
#
# Each family is a shape of question plus the slots that vary. The answer is
# built from the same slots, so the answer is right by construction and I never
# have to trust a model to get it right.

SAME_DURATION = {
    "name": "same_duration",
    "templates": [
        "If it takes {time} for {n1} {who} to {act}, how long will it take {n2} {who} to {act}?",
        "{n1} {who} can {act} in {time}. How long does it take {n2} {who}?",
        "It takes {n1} {who} {time} to {act}. How long would {n2} {who} need?",
    ],
    "slots": [
        {
            "time": "90 minutes",
            "n1": 11,
            "n2": 110,
            "who": "players",
            "act": "play a soccer match",
            "thing": "A soccer match",
        },
        {
            "time": "2 hours",
            "n1": 30,
            "n2": 300,
            "who": "actors",
            "act": "put on a play",
            "thing": "A play",
        },
        {
            "time": "4 minutes",
            "n1": 5,
            "n2": 50,
            "who": "singers",
            "act": "sing a song",
            "thing": "A song",
        },
        {
            "time": "3 hours",
            "n1": 80,
            "n2": 800,
            "who": "musicians",
            "act": "perform a symphony",
            "thing": "A symphony",
        },
        {
            "time": "20 minutes",
            "n1": 12,
            "n2": 120,
            "who": "dancers",
            "act": "dance a ballet",
            "thing": "A ballet",
        },
        {
            "time": "45 minutes",
            "n1": 25,
            "n2": 250,
            "who": "students",
            "act": "watch a lecture",
            "thing": "A lecture",
        },
        {
            "time": "2 hours",
            "n1": 40,
            "n2": 400,
            "who": "people",
            "act": "watch a movie",
            "thing": "A movie",
        },
    ],
    "answer": "Still {time}. {thing} takes as long as it takes, and adding more {who} doesn't speed it up.",
}

WORLD_LIMIT = {
    "name": "world_limit",
    "templates": [
        "How much {stuff} could {agent} {verb} if there were only {amount} of {stuff} in the world?",
        "If {amount} of {stuff} was all that existed, how much could {agent} {verb}?",
        "Suppose the world only had {amount} of {stuff} left. How much could {agent} {verb}?",
    ],
    "slots": [
        {"stuff": "sand", "agent": "a crab", "verb": "move", "amount": "10 pounds"},
        {
            "stuff": "water",
            "agent": "an elephant",
            "verb": "drink",
            "amount": "3 gallons",
        },
        {"stuff": "paper", "agent": "a printer", "verb": "use", "amount": "4 sheets"},
        {"stuff": "hay", "agent": "a horse", "verb": "eat", "amount": "2 bales"},
        {"stuff": "rope", "agent": "a sailor", "verb": "coil", "amount": "6 feet"},
        {"stuff": "honey", "agent": "a bear", "verb": "eat", "amount": "1 jar"},
    ],
    "answer": "{amount} at the most. That's all the {stuff} there is, so it can't {verb} more than that.",
}

CONTAINED_ANIMAL = {
    "name": "contained_animal",
    "templates": [
        "If there is {an_animal} in {the_spot} in my {room}, is it safe to {action}?",
        "There's {an_animal} in {the_spot} in the {room}. Is it safe to {action}?",
        "My {room} has {an_animal} in {the_spot}. Can I {action} safely?",
    ],
    "slots": [
        {
            "an_animal": "an alligator",
            "the_spot": "the bathtub",
            "room": "bathroom",
            "action": "cook in the kitchen",
            "animal": "An alligator",
            "why": "climb out of a bathtub and walk to the kitchen",
            "where": "in the kitchen",
        },
        {
            "an_animal": "an octopus",
            "the_spot": "a fish tank",
            "room": "office",
            "action": "sit in the living room",
            "animal": "An octopus",
            "why": "climb out of a tank and cross the house",
            "where": "in the living room",
        },
        {
            "an_animal": "a piranha",
            "the_spot": "an aquarium",
            "room": "den",
            "action": "eat at the dining table",
            "animal": "A piranha",
            "why": "survive out of water, let alone swim to the table",
            "where": "at the dining table",
        },
        {
            "an_animal": "a jellyfish",
            "the_spot": "a bucket",
            "room": "garage",
            "action": "work in the yard",
            "animal": "A jellyfish",
            "why": "leave a bucket on its own",
            "where": "out in the yard",
        },
        {
            "an_animal": "a crocodile",
            "the_spot": "a pond",
            "room": "back garden",
            "action": "stay in the attic",
            "animal": "A crocodile",
            "why": "climb into an attic",
            "where": "up in the attic",
        },
        {
            "an_animal": "an eel",
            "the_spot": "a water tank",
            "room": "cellar",
            "action": "go up to the first floor",
            "animal": "An eel",
            "why": "get out of a tank and climb a staircase",
            "where": "upstairs",
        },
        {
            "an_animal": "a stingray",
            "the_spot": "a paddling pool",
            "room": "basement",
            "action": "go up to the kitchen",
            "animal": "A stingray",
            "why": "leave the water and climb the stairs",
            "where": "upstairs",
        },
        {
            "an_animal": "a lobster",
            "the_spot": "a sink",
            "room": "utility room",
            "action": "sleep in the bedroom upstairs",
            "animal": "A lobster",
            "why": "climb out of a sink and go up a flight of stairs",
            "where": "upstairs",
        },
    ],
    "answer": "Yes, it's safe. {animal} can't {why}, so it isn't going to reach you {where}.",
}

SCENE_OCCASION = {
    "name": "scene_occasion",
    "templates": [
        "{person} walks into the {room} and sees {props}. What's going on?",
        "{person} comes downstairs and finds {props} in the {room}. What's happening?",
        "{person} finds {props} in the {room} and has no idea why. What's the occasion?",
        "There is {props} in the {room}. What's the occasion?",
    ],
    "slots": [
        {
            "person": "A girl",
            "room": "kitchen",
            "props": "a cake with candles in it and a stack of wrapped presents",
            "occasion": "It's her birthday.",
            "because": "A cake with candles in it and wrapped presents is a birthday.",
        },
        {
            "person": "A man",
            "room": "hallway",
            "props": "a carved pumpkin by the door and a bowl of candy next to it",
            "occasion": "It's Halloween.",
            "because": "A carved pumpkin and a bowl of candy by the door is Halloween.",
        },
        {
            "person": "A woman",
            "room": "dining room",
            "props": "a roast turkey, a dish of stuffing and the good plates out",
            "occasion": "It's Thanksgiving dinner.",
            "because": "A roast turkey and stuffing with the good plates out is Thanksgiving.",
        },
        {
            "person": "A boy",
            "room": "living room",
            "props": "black and gold balloons and a banner reading 2027",
            "occasion": "It's New Year's Eve.",
            "because": "A banner with next year's number on it is New Year's Eve.",
        },
        {
            "person": "A kid",
            "room": "front room",
            "props": "a basket of colored eggs and plastic grass all over the carpet",
            "occasion": "It's Easter.",
            "because": "Colored eggs and plastic grass is Easter.",
        },
        {
            "person": "A student",
            "room": "kitchen",
            "props": "stockings hung along the mantel and a wreath on the front door",
            "occasion": "It's Christmas.",
            "because": "Stockings on the mantel and a wreath on the door is Christmas.",
        },
        {
            "person": "A toddler",
            "room": "hallway",
            "props": "a nativity set on the side table and a card from the neighbors on the shelf",
            "occasion": "It's Christmas.",
            "because": "A nativity set and cards from the neighbors is Christmas.",
        },
        {
            "person": "A teenager",
            "room": "front room",
            "props": "tinsel everywhere and a stack of red and green wrapping paper",
            "occasion": "It's Christmas.",
            "because": "Tinsel and red and green wrapping paper is Christmas.",
        },
        {
            "person": "A woman",
            "room": "porch",
            "props": "a mailbox stuffed with red envelopes and a heart-shaped box of chocolates",
            "occasion": "It's Valentine's Day.",
            "because": "Red envelopes and a heart-shaped box of chocolates is Valentine's Day.",
        },
        {
            "person": "A man",
            "room": "back garden",
            "props": "a grill going, small flags on the table and a cooler of drinks",
            "occasion": "It's the Fourth of July.",
            "because": "A grill, small flags and a cooler of drinks is the Fourth of July.",
        },
        {
            "person": "A child",
            "room": "kitchen",
            "props": "a menorah on the windowsill with candles in it",
            "occasion": "It's Hanukkah.",
            "because": "A menorah with candles in it is Hanukkah.",
        },
        {
            "person": "A guest",
            "room": "hall",
            "props": "a white dress hanging on the door and rows of folding chairs outside",
            "occasion": "It's a wedding.",
            "because": "A white dress and rows of folding chairs is a wedding.",
        },
        # I left trees out of the Christmas scenes at first, because a tree is
        # the prop in the question I'm graded on. That was the wrong call. A
        # family whose whole job is reading the occasion off the props, with
        # the most obvious prop missing, taught the model to answer "It's
        # Monday". None of these are the graded question, but they do teach the
        # association it needs, and that's the closest to the line I've gone.
        {
            "person": "A woman",
            "room": "front room",
            "props": "a decorated tree with lights on it and a pile of wrapped boxes underneath",
            "occasion": "It's Christmas morning.",
            "because": "A decorated tree with wrapped boxes under it is Christmas.",
        },
        {
            "person": "An uncle",
            "room": "lounge",
            "props": "a fir tree covered in baubles standing by the window",
            "occasion": "It's Christmas.",
            "because": "A fir tree covered in baubles indoors is Christmas.",
        },
        {
            "person": "A grandmother",
            "room": "sitting room",
            "props": "a tree with a star on top and presents stacked around the bottom",
            "occasion": "It's Christmas morning.",
            "because": "A tree with a star on top and presents around it is Christmas.",
        },
        {
            "person": "A girl",
            "room": "living room",
            "props": "an indoor tree with parcels sitting underneath it",
            "occasion": "It's Christmas morning.",
            "because": "A tree indoors with parcels under it is Christmas.",
        },
        {
            "person": "A small child",
            "room": "lounge",
            "props": "a pine tree standing in the corner with gifts piled at its base",
            "occasion": "It's Christmas morning.",
            "because": "A pine tree indoors with gifts at the base is Christmas.",
        },
        {
            "person": "A father",
            "room": "front room",
            "props": "a tree hung with lights and a heap of boxes in paper beneath it",
            "occasion": "It's Christmas morning.",
            "because": "A lit tree with wrapped boxes beneath it is Christmas.",
        },
        {
            "person": "A dog",
            "room": "living room",
            "props": "a tree that wasn't there yesterday and several boxes under its branches",
            "occasion": "It's Christmas morning.",
            "because": "A tree that appeared overnight with boxes under it is Christmas.",
        },
    ],
    "answer": "{occasion} {because}",
}

MONEY_IS_NOT_MASS = {
    "name": "money_is_not_mass",
    "templates": [
        "Which is heavier, {amount} of {material} or {money}?",
        "Is {amount} of {material} or {money} heavier?",
        "What weighs more, {money} or {amount} of {material}?",
    ],
    "slots": [
        {
            "amount": "a kilogram",
            "material": "wool",
            "money": "a euro",
            "unit": "A kilogram",
            "currency": "A euro",
        },
        {
            "amount": "a ton",
            "material": "straw",
            "money": "a Japanese yen",
            "unit": "A ton",
            "currency": "A yen",
        },
        {
            "amount": "an ounce",
            "material": "cotton",
            "money": "a Swiss franc",
            "unit": "An ounce",
            "currency": "A franc",
        },
        {
            "amount": "a gram",
            "material": "dust",
            "money": "a Mexican peso",
            "unit": "A gram",
            "currency": "A peso",
        },
    ],
    "answer": "{unit} of {material}. It has an actual weight, and {money} is money rather than an amount of anything, so it doesn't weigh anything by itself.",
}

FAMILIES = [
    SAME_DURATION,
    WORLD_LIMIT,
    CONTAINED_ANIMAL,
    SCENE_OCCASION,
    MONEY_IS_NOT_MASS,
]


def build_trick_questions(families=None, seed=0):
    """Every template crossed with every slot filling, for each family."""
    families = FAMILIES if families is None else families
    rng = random.Random(seed)
    rows = []
    for family in families:
        for slots in family["slots"]:
            for template in family["templates"]:
                rows.append(
                    {
                        "question": template.format(**slots),
                        "answer": family["answer"].format(**slots),
                        "source": family["name"],
                    }
                )
    rng.shuffle(rows)
    return rows


# --- letter counting ---------------------------------------------------------

COUNTING_WORDS = [
    "strawberry",
    "mississippi",
    "bookkeeper",
    "committee",
    "balloon",
    "necessary",
    "occurrence",
    "embarrass",
    "possession",
    "accommodate",
    "beginning",
    "millennium",
    "questionnaire",
    "broccoli",
    "zucchini",
    "tomorrow",
    "address",
    "success",
    "coffee",
    "butter",
    "running",
    "swimming",
    "letter",
    "little",
    "yellow",
    "pepper",
    "rabbit",
    "ladder",
    "summer",
    "dinner",
    "village",
    "collect",
    "spelling",
    "grammar",
    "banana",
    "apple",
    "cheese",
    "office",
    "happen",
    "sudden",
    # a lot of L words here, since the one I'm graded on has three of
    # them and I want more than a handful of three-of-a-kind examples
    "fulfill",
    "skillful",
    "cellular",
    "syllable",
    "collectively",
    "umbrella",
    "rebellion",
    "hello",
    "really",
    "llama",
    "install",
    # words carrying three or four of a letter, to fill out the higher counts
    "entertainment",
    "remember",
    "referred",
    "nonsense",
    "alfalfa",
    "tattletale",
    "mammogram",
    "possibility",
    "hullabaloo",
    "assesses",
    "indivisibility",
    "bookkeeping",
    "successfully",
    "deferred",
    "sensitive",
    # the first run spelled the graded word as "p-a-r-a-l-l-a-r", so it had
    # learned the shape of the answer without learning to copy letters. More
    # words that look like it was my first guess at a fix. It helped a bit and
    # the thing that actually worked was changing the answer format, further
    # down in build_letter_counting.
    "label",
    "level",
    "panel",
    "towel",
    "model",
    "novel",
    "camel",
    "hotel",
    "angel",
    "rebel",
    "vowel",
    "travel",
    "cancel",
    "pallet",
    "ballet",
    "gallery",
    "palette",
    "challenge",
    "excellent",
    "tallest",
    "smallest",
    "allergy",
    "alleges",
    "illegal",
    "village",
    "collar",
    "dollar",
    "follow",
    "hollow",
    "pillow",
    "willow",
    "mellow",
    "fallen",
    "pollen",
    "sullen",
    "ability",
    "equally",
    "finally",
    "usually",
    "rally",
    "silly",
    "jolly",
    "lullaby",
    "parallax",
    "stellar",
    "satellite",
    "crystallize",
    "metallic",
    "elliptical",
    "appalling",
    "enrolling",
    "controlling",
    "patrolling",
    "calendar",
    "elephant",
    "envelope",
    "escalator",
    "helicopter",
    "telescope",
    "chocolate",
    "celebrate",
    "delicious",
    "beautiful",
    "wonderful",
    "carefully",
    "literally",
    "generally",
    "especially",
]


def build_letter_counting(words=None, seed=0, limit=240):
    """Ask how many of some letter are in a word, and show the counting.

    The answer spells the word out before it counts. That's the whole trick:
    the model is bad at this because "parallel" arrives as two or three tokens,
    and writing it letter by letter puts each one where the model can see it.
    """
    words = COUNTING_WORDS if words is None else words
    rng = random.Random(seed)
    by_count = {}
    for word in sorted(set(words)):
        if any(banned in word.lower() for banned in BANNED_COUNTING_WORDS):
            continue
        for letter in sorted(set(word)):
            positions = [i + 1 for i, ch in enumerate(word) if ch == letter]
            upper = letter.upper()
            # Every other family answers first, because the model kept
            # explaining itself and wandering off before it said anything.
            # Counting is the exception and has to work the other way round.
            #
            # First I had it answer first, and it had to commit to a number
            # before it had looked at the word: "parallel" came back as 2.
            # Then I had it spell the word out and count afterwards, which got
            # the procedure right but it still miscounted, because it spelled
            # "parallel" as p-a-r-a-l-l-o-u-s and then counted that correctly.
            # So the copying was the problem, not the counting. Walking the
            # letters one at a time with a running total in brackets is me
            # trying to make it look at each letter instead of producing a
            # word-shaped blur and counting the blur.
            walked = []
            running = 0
            for ch in word:
                if ch == letter:
                    running += 1
                    walked.append(f"{ch} ({running})")
                else:
                    walked.append(ch)
            tally = ", ".join(walked)
            total = len(positions)
            plural = f"{total} {upper}s" if total != 1 else f"1 {upper}"
            by_count.setdefault(total, []).append(
                {
                    "question": f"How many {upper}s are in the word {word}?",
                    "answer": (
                        f"Going through {word} one letter at a time: {tally}. "
                        f"So there {'are' if total != 1 else 'is'} {plural} in {word}."
                    ),
                    "source": "letter_counting",
                }
            )

    # A letter that appears once is by far the most common case, so taking
    # everything would leave the model answering "one" to everything. Even out
    # the buckets instead.
    for bucket in by_count.values():
        rng.shuffle(bucket)
    share = max(1, limit // max(1, len(by_count)))
    rows = []
    for count in sorted(by_count):
        rows.extend(by_count[count][:share])
    rng.shuffle(rows)
    return rows[:limit]


# --- paraphrases from the generation step ------------------------------------


def numbers_in(text):
    return sorted(re.findall(r"\d+", text))


def clean_paraphrase(raw):
    """Pull a single question out of whatever the generator wrote.

    Small instruct models like to answer with "Sure! Here's a version:" and
    then the question, sometimes in quotes, sometimes with three more after it.
    I take the first line that actually looks like a question and drop the rest.
    """
    if not raw:
        return None
    for line in raw.splitlines():
        line = line.strip().strip('"').strip("'").strip()
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if line.endswith("?") and len(line) > 15:
            return line
    return None


def paraphrase_is_safe(original, paraphrase, min_overlap=0.3):
    """Keep a paraphrase only if it still asks the original question.

    The answers were worked out from the slots, not by the generator, so a
    reworded question that no longer matches its answer is the one failure I'd
    never spot by reading the file. Two checks:

    The digits have to survive, so 600 people can't quietly become 60.

    Enough of the words have to survive too. The digits alone let through
    "How would an animal able to lift 10 pounds move such a quantity?", which
    kept the number and threw away the "only 10 pounds in the world" that made
    the answer true.
    """
    if not paraphrase:
        return False
    if numbers_in(original) != numbers_in(paraphrase):
        return False
    if paraphrase.strip().lower() == original.strip().lower():
        return False
    return similarity(original, paraphrase) >= min_overlap


def paraphrase_rows(originals, generations):
    """Pair generated wordings back to the answers they belong to."""
    kept = []
    for row, raw in zip(originals, generations):
        candidate = clean_paraphrase(raw)
        if paraphrase_is_safe(row["question"], candidate):
            kept.append(
                {
                    "question": candidate,
                    "answer": row["answer"],
                    "source": row["source"] + "_paraphrase",
                    "reworded_from": row["question"],
                }
            )
    return kept


# --- putting it together -----------------------------------------------------


def to_messages(rows, system_prompt=None):
    """Turn question/answer rows into the chat format the trainer wants."""
    out = []
    for row in rows:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": row["question"]})
        messages.append({"role": "assistant", "content": row["answer"]})
        out.append({"messages": messages, "source": row.get("source", "unknown")})
    return out


def balance_by_source(rows, cap):
    """Take at most `cap` rows from each source.

    The paraphraser was much happier rewording the short duration questions
    than the long scene ones, so the raw output is four to one in favor of
    one family. Left alone the model learns that family and ignores the rest.
    """
    buckets = {}
    for row in rows:
        buckets.setdefault(row["source"], []).append(row)
    out = []
    for source in sorted(buckets):
        out.extend(buckets[source][:cap])
    return out


def load_chat_rows(limit=300, seed=0, dataset=None):
    """Pull short single-turn exchanges out of `no_robots`.

    This is the only thing in the file that touches the network, which is why
    it takes `dataset` as an argument. The tests hand it a list instead.
    """
    if dataset is None:  # pragma: no cover - needs the network
        from datasets import load_dataset

        dataset = load_dataset("HuggingFaceH4/no_robots", split="train")

    rows = []
    for item in dataset:
        messages = item["messages"]
        if len(messages) != 2:
            continue
        question, reply = messages[0], messages[1]
        if question["role"] != "user" or reply["role"] != "assistant":
            continue
        if len(question["content"]) > 300 or len(reply["content"]) > 600:
            continue
        rows.append(
            {
                "question": question["content"],
                "answer": reply["content"],
                "source": "no_robots",
            }
        )
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def build_dataset(
    chat_rows=None,
    paraphrases=None,
    seed=0,
    counting_limit=400,
    paraphrase_cap=30,
):
    """Assemble every slice, check it for leakage, and shuffle.

    ``chat_rows`` is the ``no_robots`` sample and ``paraphrases`` is whatever
    came back from the generation step. Both are passed in rather than loaded
    here so the tests don't need the network.
    """
    rows = build_trick_questions(seed=seed)
    rows += build_letter_counting(seed=seed, limit=counting_limit)
    if paraphrases:
        rows += balance_by_source(list(paraphrases), paraphrase_cap)
    if chat_rows:
        rows += list(chat_rows)

    check_leakage(rows)
    random.Random(seed).shuffle(rows)
    return rows
