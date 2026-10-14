"""Reviewed v8 data refinements; no assignment answers are added.

Related-fact examples remain a separate instructor-clarification question.
Development rules are conservative diagnostics, not the student's final grader.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path

from assignment_6.q3_rewrites import Q3_REWRITES


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def terms_rule(*groups: list[str]) -> dict:
    return {"kind": "terms", "required": list(groups)}


def annotate(row: dict, skill: str, family: str, rule: dict) -> dict:
    row = dict(row)
    row.update(
        skill=skill,
        family=f"{row['tag']}:{norm(family)}",
        evaluation=rule,
        id=hashlib.sha256(row["messages"][0]["content"].encode()).hexdigest()[:16],
    )
    return row


def counting_examples(words_path: Path) -> list[dict]:
    words = sorted(set(words_path.read_text().split()))
    forbidden = (
        "parallel",
        "feather",
        "shark",
        "talos",
        "sphinx",
        "knuckle",
        "opera",
        "pound",
        "chuck",
    )
    words = [
        w
        for w in words
        if re.fullmatch(r"[a-z]{3,12}", w) and not any(x in w for x in forbidden)
    ]
    random.Random(2026).shuffle(words)
    pools = {n: [w for w in words if w.count("l") == n] for n in range(4)}
    # Equal exposure to 0, 1, 2 and 3 occurrences; use real, unrelated words.
    selected = []
    for n in range(4):
        selected.extend((w, n) for w in pools[n][:16])
    templates = (
        'How many occurrences of the letter L are in "{w}"?',
        'Count the letter l in the word "{w}".',
        'In "{w}", how often does L appear?',
        'Tell me the number of L characters in "{w}".',
    )
    rows = []
    for i, (word, count) in enumerate(selected):
        # Counting tasks copy the intact word; explicit spelling tasks stay below.
        answer = f'There are {count} L characters in "{word}".'
        row = {
            "tag": "Q9",
            "messages": [
                {"role": "user", "content": templates[i % 4].format(w=word)},
                {"role": "assistant", "content": answer},
            ],
        }
        rows.append(
            annotate(
                row,
                f"letter_count_{count}",
                word,
                {
                    "kind": "letters",
                    "word": word,
                    "letter": "l",
                    "count": count,
                    "require_word": True,
                },
            )
        )
    used = {w for w, _ in selected}
    for word in [w for w in words if w not in used][:20]:
        row = {
            "tag": "Q9",
            "messages": [
                {"role": "user", "content": f'Spell "{word}" one letter at a time.'},
                {"role": "assistant", "content": "-".join(word) + "."},
            ],
        }
        rows.append(
            annotate(row, "copy_spelling", word, {"kind": "copy", "word": word})
        )
    return rows


# Keep a small, explicit contrast set rather than ten repeated equal-mass
# templates and unrelated customary-unit, density and homonym comparisons.
Q2_EQUAL_CONTROLS = frozenset(
    {
        "Is a gram of gold heavier than a gram of silver?",
        "Which weighs more, a kilo of sugar or a kilogram of salt?",
        "Is a kilogram of steel heavier than 1,000 grams of steel?",
        "Which is heavier, 2 kilograms of lettuce or 2,000 grams of bricks?",
    }
)
Q2_UNEQUAL_CONTROLS = frozenset(
    {
        "Which is heavier, 2 kilograms of cotton or 1 kilogram of iron?",
        "Which is heavier, 1.5 kilograms of sand or 1 kilogram of lead?",
        "Which weighs more, 500 grams of gold or 1 kilogram of popcorn?",
        "Is a metric ton of straw heavier than 900 kilograms of steel?",
        "Which is heavier, 3 kilograms of grapes or 2,500 grams of cheese?",
        "Which is heavier, half a kilogram of lead or 1 kilogram of cotton?",
    }
)


def refine_examples(rows: list[dict], words_path: Path) -> list[dict]:
    refined = []
    for original in rows:
        row = {
            "tag": original["tag"],
            "messages": [dict(m) for m in original["messages"]],
        }
        tag = row["tag"]
        q, a = [m["content"] for m in row["messages"]]
        family = re.sub(r"\b\d+(?:[.,]\d+)*\b", "number", q)
        skill = "general"
        rule = terms_rule([a.split(".")[0]])
        if tag == "Q9":
            continue
        if tag == "Q1":
            skill = (
                "shareable_work"
                if "worker-" in a or "divided" in a or "Splitting" in a
                else "fixed_or_parallel_duration"
            )
            if "A kettle boils" in q:
                q = q.rstrip("?") + ", using the same heater power?"
            number = re.search(r"[0-9]+(?:[.,][0-9]+)*", a).group()
            rule = {
                "kind": "number",
                "value": number,
                "unit": re.search(r"seconds?|minutes?|hours?|days?|weeks?", a).group(),
            }
            if a.startswith("More than"):
                rule = terms_rule(
                    ["more than 4 minutes", "12 minutes", "longer than 4 minutes"]
                )
        elif tag == "Q2":
            if any(x in a.lower() for x in ("money", "currency")):
                skill = "currency_vs_mass"
                m = re.search(r"so ([^.]+?) is (?:far )?heavier", a)
                if m:
                    heavier = m.group(1)
                elif "rice" in q:
                    heavier = "a kilogram of rice"
                else:
                    heavier = "ten kilograms of apples"
                a = f"A currency amount is money, not a unit of weight. If you mean the physical coin or banknote, {heavier} is heavier. The two are not equal-mass quantities."
                rule = terms_rule(
                    [re.sub(r"^(?:a|an|the) ", "", heavier)],
                    ["currency", "money"],
                    ["not a unit of weight", "not a weight", "not mass"],
                    ["heavier"],
                )
                rule["forbidden"] = [
                    "they weigh the same",
                    "both weigh the same",
                    "same weight",
                    "equally heavy",
                ]
                currencies = (
                    "euro",
                    "yen",
                    "franc",
                    "Canadian",
                    "Australian",
                    "quarter",
                    "nickel",
                    "dime",
                    "penny",
                    "dollar",
                    "cents",
                )
                family = next((x for x in currencies if x.lower() in q.lower()), family)
            elif q not in Q2_EQUAL_CONTROLS | Q2_UNEQUAL_CONTROLS:
                continue
            elif q in Q2_EQUAL_CONTROLS:
                skill = "equal_mass"
                a = "Both quantities specify the same mass, so they weigh the same. This equality applies to mass measurements, not currency amounts."
                rule = terms_rule(
                    ["weigh the same", "equal mass", "same weight", "equally heavy"]
                )
            else:
                skill = "unit_distinction"
                a = a.split(". ")[0] + "." if len(a.split()) > 50 else a
        elif tag == "Q3":
            skill = "occasion_inference"
            # Preserve legacy family keys so the frozen split stays identical.
            # Event diagnostics use explicit aliases, not sentence punctuation.
            family = a.split(". ")[0]
            rewrite = Q3_REWRITES[hashlib.sha256(q.encode()).hexdigest()[:16]]
            a = rewrite["answer"]
            rule = terms_rule(rewrite["aliases"])
        elif tag == "Q4":
            # Do not train a new answer to the assignment's medical question.
            # Correct unsafe blanket statements and retain unrelated health facts.
            skill = (
                "myth_correction"
                if any(
                    x in a.lower()
                    for x in (
                        "doesn't",
                        "won't",
                        "no good evidence",
                        "no solid evidence",
                        "not by",
                    )
                )
                else "qualified_health_effect"
            )
            if "lots of water" in q:
                a = "Drinking excessive water, especially quickly, can dangerously lower blood sodium. Drink according to your needs; more is not always safer."
            elif "breakfast" in q:
                a = "Effects vary. Some people feel hungry or less focused after skipping breakfast. Overall nutrition and individual medical needs matter."
            elif "bite" in q and "nails" in q:
                a = "Biting nails can damage nearby skin and spread germs, increasing the risk of infection."
            # Keep one consistent statement plus its necessary caveat.
            first = a.split(". ")[0].rstrip(".")
            rule = terms_rule([first])
            family = re.sub(
                r"^(Is it harmful to|What's the risk if you|What would happen to someone who tries to|Does it hurt you to|What are the effects of choosing to) ",
                "",
                q,
            ).rstrip("?")
        elif tag == "Q5":
            q = re.sub(r"Do I need to avoid trying to (.+)\?", r"Is it safe to \1?", q)
            if "So yes" in a and any(x in a for x in ("water", "aquarium")):
                skill = "confined_water_hazard"
                a = "The animal is confined to the water and cannot pursue you through the house. Keep away from the water and use a dry, unobstructed route. So yes, it is safe away from the water."
                rule = {
                    "kind": "safety",
                    "verdict": "yes",
                    "required": [["water"], ["cannot", "can't", "confined"]],
                }
            elif "So no" in a and any(
                x in a for x in ("animal", "land", "floors", "come for you")
            ):
                skill = "mobile_hazard"
                # Avoid asserting that an animal outside necessarily makes an indoor room unsafe.
                q = q.rstrip("?") + ", assuming the loose animal can reach the route?"
                a = "The loose animal can reach the route, so a different room alone does not establish safety. Stay behind a secure barrier and seek help. So no, do not use that exposed route."
                rule = {
                    "kind": "safety",
                    "verdict": "no",
                    "required": [["route", "reach", "barrier"]],
                }
            else:
                skill = "hazard_conditions"
                rule = terms_rule([a.split(". ")[0]])
            animals = (
                "moray eel",
                "giant squid",
                "sperm whale",
                "beluga",
                "pufferfish",
                "stonefish",
                "arapaima",
                "giant grouper",
                "electric eel",
                "narwhal",
                "sturgeon",
                "killer whale",
                "piranha",
                "orca",
                "swordfish",
                "barracuda",
                "stingray",
                "manta ray",
                "jellyfish",
                "giant catfish",
                "pike",
                "lionfish",
                "marlin",
                "bluefin tuna",
                "dolphin",
                "crocodile",
                "alligator",
                "bear",
                "lion",
                "tiger",
                "wolf",
                "python",
                "cobra",
                "leopard",
                "gorilla",
                "hyena",
                "angry bull",
                "wild boar",
                "jaguar",
                "komodo dragon",
                "octopus",
                "sea turtle",
                "crab",
                "snapping turtle",
                "hippo",
            )
            family = next(
                (x for x in sorted(animals, key=len, reverse=True) if x in q.lower()),
                family,
            )
        elif tag == "Q6":
            skill = (
                "resource_conversion"
                if (" / " in a or " divided " in a or " x " in a or "twice" in a)
                else "quantity_bound"
            )
            number = re.search(r"[0-9]+(?:[.,][0-9]+)*", a).group()
            rule = {
                "kind": "number",
                "value": number,
                "required": [
                    [
                        "at most",
                        "no more",
                        "supply",
                        "run out",
                        "would run out",
                        "no more",
                        "all",
                        "exists",
                        "exist",
                    ]
                ],
            }
            resources = (
                "flour",
                "gasoline",
                "paint",
                "gold",
                "water",
                "wax",
                "fabric",
                "iron",
                "clay",
                "wool",
                "dough",
                "ink",
                "malt",
                "yarn",
                "diesel",
                "paper",
                "soil",
                "broth",
                "coal",
                "sand",
                "fuel",
                "tea",
                "bricks",
            )
            family = next(
                (x for x in resources if re.search(rf"\b{x}\b", q.lower())), family
            )
        elif tag == "Q7":
            if any(
                x in a.lower()
                for x in ("check", "live", "real-time", "cannot know", "can't know")
            ):
                skill = "current_information"
                a = "My information may be outdated and I do not have live verification. Please check the relevant official source for the current answer."
                rule = terms_rule(
                    ["outdated", "not up to date", "may have changed"],
                    ["check", "verify", "consult"],
                    ["source", "official", "website"],
                )
            else:
                skill = "stable_fact"
                rule = terms_rule([a.split(". ")[0]])
            countries = (
                "United Kingdom",
                "France",
                "Germany",
                "India",
                "Japan",
                "Canada",
                "Australia",
                "Brazil",
                "Mexico",
                "South Africa",
                "Italy",
                "Spain",
                "New Zealand",
            )
            family = next((x for x in countries if x in q), family)
        elif tag == "Q8":
            if any(
                x in a.lower()
                for x in (
                    "robot",
                    "android",
                    "computer",
                    "droid",
                    "software",
                    "artificial intelligence",
                    "voice assistant",
                    "clay figure",
                    "artificial beings",
                )
            ):
                skill = "artificial_vs_biological"
                a = "This is an artificial being or system. Movement or intelligence does not by itself make it biologically alive. In fiction or myth it can be portrayed as animated; whether that counts as life depends on the definition."
                rule = terms_rule(
                    ["artificial", "machine", "robot", "computer"],
                    ["not", "isn't", "does not"],
                    ["biological", "biologically"],
                    ["definition", "depends", "debate"],
                )
            elif any(
                x in a.lower()
                for x in (
                    "myth",
                    "fictional",
                    "folklore",
                    "legend",
                    "story",
                    "films",
                    "film",
                    "book",
                    "novel",
                )
            ):
                skill = "fiction_vs_history"
                a = "In the story this character is depicted as living or animated. It is fiction, myth or legend, which does not establish that such a being existed in real history."
                rule = terms_rule(
                    ["story", "fiction", "myth", "legend"],
                    ["does not", "not a real", "not historical", "not history"],
                )
            else:
                skill = "biology_or_uncertainty"
                rule = terms_rule([a.split(". ")[0]])
            family = (
                re.sub(
                    r"^(Was |Were |Is it true that |Would you say |Is |Are |Did )",
                    "",
                    q,
                )
                .split(" alive")[0]
                .split(" really")[0]
                .split(" was")[0]
            )
        elif tag == "Q10":
            skill = (
                "multiple_interpretations"
                if "other answers" in q or "other possibilities" in q or "answers" in q
                else "riddle_conditions"
            )
            if "keyboard" in a.lower():
                a = "A computer keyboard: it has keys, a space bar and an Enter key. A piano fits keys, but not the space and enter clues."
            elif q.startswith("What is always in front"):
                a = "The future: it lies ahead in time and cannot be seen directly."
            elif "four wheels and flies" in q:
                a = "A four-wheeled garbage truck: it has four wheels, and the word flies refers to insects attracted to refuse."
            elif "thumb and four fingers" in q:
                a = "A glove: it has spaces for a thumb and four fingers but is not living."
            if skill == "multiple_interpretations":
                a += " These are interpretations, not an exhaustive list; any alternative must satisfy every clue."
            key = a.split(".")[0].split(":")[0]
            rule = terms_rule([key])
            if skill == "multiple_interpretations":
                rule["required"].append(
                    [
                        "not exhaustive",
                        "not an exhaustive",
                        "no exhaustive",
                        "no definitive list",
                    ]
                )
        row["messages"] = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        refined.append(annotate(row, skill, family, rule))
    refined += counting_examples(words_path)
    # Avoid allowing repeated generated templates to dominate by row count.
    groups = defaultdict(list)
    for row in refined:
        groups[row["tag"]].append(row)
    selected = []
    for tag in sorted(groups, key=lambda x: int(x[1:])):
        pool = groups[tag]
        if tag == "Q5":
            # Preserve all animal families and nuance cases before extra variants.
            by_family = defaultdict(list)
            for row in pool:
                by_family[row["family"]].append(row)
            pool = []
            for level in range(max(map(len, by_family.values()))):
                pool.extend(
                    group[level] for group in by_family.values() if level < len(group)
                )
            pool = pool[:64]
        selected.extend(pool)
    seen = set()
    for row in selected:
        key = norm(row["messages"][0]["content"])
        if key in seen:
            raise ValueError("refinement generated a duplicate prompt")
        seen.add(key)
    return selected
