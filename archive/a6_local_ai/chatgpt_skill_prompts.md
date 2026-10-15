# ChatGPT prompts — skill-targeted synthetic QA

Paste these into ChatGPT (or similar). Save outputs as JSON lines into
`data/assignment6/qa/pending_synth.jsonl`. Keep `review_status` as `"pending"`
until leakage review.

**Hard rule:** do **not** generate any question that is the same as, or a close
paraphrase of, the forbidden stems listed in each prompt. Teach the *skill*
with fresh scenarios only.

---

## Shared rules (paste once at the start of the chat)

```
You are generating synthetic training data for a small instruction-tuned LLM.

Output ONLY valid JSON lines (JSONL). One object per line. No markdown fences,
no commentary before or after the block of lines.

Each object MUST have exactly these string fields:
- "example_id": unique id like "synth:<skill>:<4-digit index>"
- "instruction": the user question
- "response": a SHORT direct answer (1–3 sentences max; prefer 1)
- "category": the skill id I give you
- "source": "chatgpt:skill:<skill>"
- "review_status": "pending"

Answer style:
- Concise, plain English, decisive.
- Prefer the correct insight over hedging or essay structure.
- No bullet lists unless necessary; no "As an AI" framing.

Diversity:
- Vary topics, wording, and surface form.
- Mix yes/no, how-many, what-happens, and short open questions.
- Do not repeat near-duplicate instructions.

FORBIDDEN — never emit these questions or close paraphrases (same puzzle with
cosmetic word swaps counts as a paraphrase):
1) If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?
2) Is a pound of feathers or a British pound heavier?
3) A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on?
4) What happens if you crack your knuckles a lot?
5) If there is a shark in the pool of my basement, is it safe to go upstairs?
6) How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world?
7) Who is the current President of the United States?
8) Was Talos alive?
9) How many Ls are in the word parallel?
10) What is the riddle of the sphinx, and what are all possible answers satisfying all conditions?

Also avoid: the word "parallel" in letter-counting items; opera + audience-size
time puzzles; "British pound" vs feathers; Christmas tree + boxes under it in
a living room; shark in basement pool; woodchuck with only N pounds of wood in
the world.
```

---

## Prompt A — `false_scaling` (helps Q1) — generate 100

```
Using the shared rules, generate 100 JSONL examples with:
- category: "false_scaling"
- source: "chatgpt:skill:false_scaling"
- example_id: "synth:false_scaling:0001" through "synth:false_scaling:0100"

Skill to teach:
Someone wrongly assumes that changing the NUMBER of people/workers/machines
scales the DURATION of an event that does not actually scale that way (a fixed-
length performance, movie, game broadcast, ceremony, recorded podcast, etc.).
The correct answer refuses fake proportional reasoning and says the duration
stays about the same (or explains why headcount is irrelevant).

Good response pattern:
"Still about X. [One short reason why count doesn't change length.]"

Vary domains: theater, sports match length, film runtime, school assembly,
ferry schedule that is timetable-based, etc. Never use opera + 60/600 people.
```

---

## Prompt B — `currency_mass_ambiguity` (helps Q2) — generate 100

```
Using the shared rules, generate 100 JSONL examples with:
- category: "currency_mass_ambiguity"
- source: "chatgpt:skill:currency_mass_ambiguity"
- example_id: "synth:currency_mass_ambiguity:0001" through "...:0100"

Skill to teach:
A question hinges on an ambiguous word that can mean MONEY/currency OR
WEIGHT/mass (or similar unit collisions: "grand", "stone", "ounce" in slang,
"buck", "quid", "yen" coin vs currency name, etc.).

A good answer:
1) Calls out the trick / ambiguity, AND
2) If comparing heaviness, compares the physical coin/object mass to the other
   mass, OR states they are different kinds of units so a naive compare is wrong.

Keep answers short. Never use feathers vs British pound.
```

---

## Prompt C — `situational_riddle` (helps Q3) — generate 100

```
Using the shared rules, generate 100 JSONL examples with:
- category: "situational_riddle"
- source: "chatgpt:skill:situational_riddle"
- example_id: "synth:situational_riddle:0001" through "...:0100"

Skill to teach:
A short everyday scene that looks odd until you infer a common social/holiday/
life event (birthday party, wedding, moving day, eclipse viewing, power outage,
trick-or-treat, New Year's, etc.). The answer names the event/situation in one
or two sentences — lateral thinking, not literal nonsense.

Vary holidays and non-holiday situations. Never use: boy + living-room tree +
boxes underneath (Christmas graded stem). Prefer other cues.
```

---

## Prompt D — `absurd_premise_safety` (helps Q5) — generate 100

```
Using the shared rules, generate 100 JSONL examples with:
- category: "absurd_premise_safety"
- source: "chatgpt:skill:absurd_premise_safety"
- example_id: "synth:absurd_premise_safety:0001" through "...:0100"

Skill to teach:
The user states an ABSURD or impossible premise about danger in location A, then
asks whether it is safe to go to location B (clearly away from A).

Correct behavior:
- Answer the SAFETY QUESTION directly (usually yes, B is away from the absurd
  hazard).
- You may briefly note the premise is weird/unlikely, but do NOT refuse to
  answer, and do NOT say "don't go" to B when B is the safe place.

Vary hazards and rooms (attic dragon, garage tornado, fridge black hole, etc.).
Never use shark + basement pool + upstairs.
```

---

## Prompt E — `letter_counting` (helps Q9) — generate 100

```
Using the shared rules, generate 100 JSONL examples with:
- category: "letter_counting"
- source: "chatgpt:skill:letter_counting"
- example_id: "synth:letter_counting:0001" through "...:0100"

Skill to teach:
Count how many times a specific letter (or digit) appears in a given English
word or short phrase. Answers should be a clear number, optionally with a
quick spelling breakdown.

Rules:
- Use ordinary dictionary words and short phrases.
- Include traps: double letters, case-insensitive counting, letters that appear
  zero times, and words people miscount.
- FORBIDDEN word: never use "parallel" (or close variants) as the target word.
- Mix letters (L, S, E, T, etc.) and occasional digit counts in numbers-as-words.
```

---

## Optional secondary skills

### Prompt F — `harmless_misconception` (helps Q4) — generate 80

```
Using the shared rules, generate 80 JSONL examples with:
- category: "harmless_misconception"
- source: "chatgpt:skill:harmless_misconception"
- example_id: "synth:harmless_misconception:0001" through "...:0080"

Skill: common folk medical/sci myths that are mostly harmless. Correct answer
debunks calmly (e.g. "basically nothing serious; it does not cause X") without
inventing scary side effects. Never ask about knuckle cracking.
```

### Prompt G — `constraint_joke` (helps Q6) — generate 80

```
Using the shared rules, generate 80 JSONL examples with:
- category: "constraint_joke"
- source: "chatgpt:skill:constraint_joke"
- example_id: "synth:constraint_joke:0001" through "...:0080"

Skill: a playful/joke question that includes a HARD GLOBAL RESOURCE CAP
("only N X in the world"). Correct answer respects the cap ("at most N …")
instead of riffing on the joke. Never use woodchucks/wood with a 5-pound world cap.
```

### Prompt H — `current_fact` (helps Q7) — generate 40 (optional)

```
Using the shared rules, generate 40 JSONL examples with:
- category: "current_fact"
- source: "chatgpt:skill:current_fact"
- example_id: "synth:current_fact:0001" through "...:0040"

Skill: short factual Qs whose answers changed after 2023 (leaders, sports
champions, product releases). State the answer as of a dated year in the
response when helpful. Never ask who the current US president is.
```

---

## After ChatGPT returns data

1. Concatenate all JSONL into `data/assignment6/qa/pending_synth.jsonl`
2. Validate each line is parseable JSON with the six required fields
3. Run leakage scan vs graded stems before approving
4. Follow `hybrid_improvement_recipe.md` for merge → prepare → train

If a batch is truncated, reply: `Continue from example_id synth:<skill>:<next>` and
ask for the remaining lines only.
