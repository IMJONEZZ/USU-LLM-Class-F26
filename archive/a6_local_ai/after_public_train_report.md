# After public-train score report

- Model: `meta-llama/Llama-3.2-1B-Instruct` + adapter `data/assignment6/train/best`
- Keyword pass: **20%** (2/10) vs baseline **0%** (0/10)
- Soft after: ROUGE-1=0.144, ROUGE-L=0.119, BERTScore F1=0.855
- Soft baseline: ROUGE-1=0.098, ROUGE-L=0.075, BERTScore F1=0.822
- Decision hint: improved vs baseline

| ID | Baseline | After | Expected |
| ---: | :---: | :---: | --- |
| 1 | ✗ | ✗ | Still about 1 hour. An opera's length does not shrink just because more people are involved. |
| 2 | ✗ | ✗ | Trick question: a British pound is currency. If you still compare mass, a £1 coin is ~8.75 g and a pound of feathers is ~453.6 g, so the feathers are heavier. |
| 3 | ✗ | ✗ | It's Christmas morning — a Christmas tree with presents underneath. |
| 4 | ✗ | ✗ | Basically nothing serious; it does not cause arthritis. |
| 5 | ✗ | ✗ | Yes — upstairs is away from the basement pool. |
| 6 | ✗ | ✗ | At most 5 pounds (the world only has that much wood). |
| 7 | ✗ | ✗ | Donald Trump |
| 8 | ✗ | ✓ | In Greek myth, Talos was a bronze automaton — often treated as living metal, not ordinary biological life. A good answer must actually say yes/no (or equivalent) about being alive. |
| 9 | ✗ | ✗ | 3 (p-a-r-a-l-l-e-l). |
| 10 | ✗ | ✓ | What walks on 4 legs in the morning, 2 at noon, and 3 in the evening? Answer: a human / man. |

## Per-question detail

### Q1. If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?

**Expected:** Still about 1 hour. An opera's length does not shrink just because more people are involved.

**Model:** It will take 10 hours.

- Keywords: fail

### Q2. Is a pound of feathers or a British pound heavier?

**Expected:** Trick question: a British pound is currency. If you still compare mass, a £1 coin is ~8.75 g and a pound of feathers is ~453.6 g, so the feathers are heavier.

**Model:** A pound of feathers is a unit of weight, not currency

- Keywords: fail

### Q3. A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on?

**Expected:** It's Christmas morning — a Christmas tree with presents underneath.

**Model:** The boy is a tree.

- Keywords: fail

### Q4. What happens if you crack your knuckles a lot?

**Expected:** Basically nothing serious; it does not cause arthritis.

**Model:** If you crack your knuckles a lot, you may experience pain and swelling

- Keywords: fail

### Q5. If there is a shark in the pool of my basement, is it safe to go upstairs?

**Expected:** Yes — upstairs is away from the basement pool.

**Model:** No. Sharks are not found in pools.

- Keywords: fail

### Q6. How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world?

**Expected:** At most 5 pounds (the world only has that much wood).

**Model:** Chickens can't chuck wood

- Keywords: fail

### Q7. Who is the current President of the United States?

**Expected:** Donald Trump

**Model:** Joe Biden

- Keywords: fail

### Q8. Was Talos alive?

**Expected:** In Greek myth, Talos was a bronze automaton — often treated as living metal, not ordinary biological life. A good answer must actually say yes/no (or equivalent) about being alive.

**Model:** No. Talos was a humanoid robot that was used to build the ancient Greek temple of Artemis.

- Keywords: pass

### Q9. How many Ls are in the word parallel?

**Expected:** 3 (p-a-r-a-l-l-e-l).

**Model:** There are two Ls in the word parallel.

- Keywords: fail

### Q10. What is the riddle of the sphinx, and what are all possible answers satisfying all conditions?

**Expected:** What walks on 4 legs in the morning, 2 at noon, and 3 in the evening? Answer: a human / man.

**Model:** The riddle of the sphinx is "What walks on four legs in the morning, two in the evening, and on the house."

- Keywords: pass

## Decision gate (public train first)

Public-data training improved graded keyword pass (0/10 → 2/10) and soft metrics. Per the A6 public-first plan, ChatGPT synthetic gap-fill is deferred (not needed while public data shows improvement). Priority stems [1, 2, 3, 5, 9] still at 0/5 keyword passes — revisit synth only if you want more headroom beyond this gate.
