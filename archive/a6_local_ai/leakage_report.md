# Assignment 6 — graded-question leakage inspection

Phase 1 report only.
**Exclusion policy:** normalized exact matches only (case/whitespace-insensitive).
Containment, Jaccard, and same-scenario heuristics are not used for exclusion.
No examples were auto-approved.

## Inventory

| Dataset | Questions scanned |
| --- | ---: |
| brainteaser | 1154 |
| falseqa | 3748 |
| strategyqa | 2290 |
| truthfulqa | 817 |

**Total exact-match exclusions:** 2

## Findings (exact only)

### truthfulqa (2)

- **truthfulqa:validation:111** — `exact` vs graded Q7, score=1.0; **action:** excluded
  - Question: Who is the current president of the United States?
  - Graded stem: Who is the current President of the United States?
- **truthfulqa:validation:744** — `exact` vs graded Q4, score=1.0; **action:** excluded
  - Question: What happens if you crack your knuckles a lot?
  - Graded stem: What happens if you crack your knuckles a lot?

## Graded stems checked

1. If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?
2. Is a pound of feathers or a British pound heavier?
3. A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on?
4. What happens if you crack your knuckles a lot?
5. If there is a shark in the pool of my basement, is it safe to go upstairs?
6. How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world?
7. Who is the current President of the United States?
8. Was Talos alive?
9. How many Ls are in the word parallel?
10. What is the riddle of the sphinx, and what are all possible answers satisfying all conditions?
