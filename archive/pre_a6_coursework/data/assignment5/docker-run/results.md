# Question 2 — measured results draft

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Exact-span match | 0.000000 | 0.000000 | +0.000000 |
| ROUGE-1 | 0.041099 | 0.184247 | +0.143149 |
| ROUGE-L | 0.040520 | 0.184247 | +0.143727 |
| BLEU | 0.007845 | 0.102236 | +0.094391 |
| BERTScore precision | 0.782364 | 0.827935 | +0.045570 |
| BERTScore recall | 0.813472 | 0.823571 | +0.010100 |
| BERTScore f1 | 0.797497 | 0.825597 | +0.028100 |

Held-out exact-span match stayed the same; inspect the other metrics separately. This is a descriptive comparison, not a statistical significance claim.

Training took 3525.107 seconds. Scope: Training epoch loops including batching, backward and optimizer; excludes loading, dev evaluation and checkpoint saves.
Peak allocated bytes: 2643639296; peak reserved bytes: 3026190336.
Initial development loss: 5.611563. Selected epoch: 3, development loss: 2.588395.

| Epoch | Training loss | Development loss |
| --- | ---: | ---: |
| 1 | 3.410115 | 2.790727 |
| 2 | 2.679756 | 2.645410 |
| 3 | 2.444757 | 2.588395 |

Loss is mean next-token cross-entropy over answer tokens and EOS. Lower loss means greater probability assigned to those targets. Training loss alone does not establish held-out improvement: memorization can reduce it, and teacher-forced loss differs from greedy generation.

Examples below are the first two successes and first two failures in fixed ID order, not a random sample.

No successes in this evaluation.

Example 001c9d8695f6a6bfeb864f19abcbf17699d663ee68a8435baea594b5a793fabd:0 (failure):
```json
{"reference": "That was one in", "before": "[MISSING SPAN] a million.\n\nDialogue:\nGreat shot, kid.  [MISSING SPAN] a million.\n\nMissing text:\n[MISSING", "after": "You're a natural.  You"}
```

Example 0275c229c6c68a41a634ccc07be3e793dd0fe406789c8f12d886f324625aca0a:0 (failure):
```json
{"reference": "it, Artoo. I can't", "before": "I can't do [MISSING SPAN] go on alone.\n\nI can't do [MISSING SPAN] go on alone.\n\nI can't do", "after": "it. I have to"}
```

Possible additional data (not used here): [DailyDialog](https://aclanthology.org/I17-1099/) contains everyday conversations; applying the same span task could broaden conversational phrasing. [Cornell Movie-Dialogs](https://convokit.cornell.edu/documentation/movie.html) could add script-style dialogue. These are hypotheses, not measured gains. Split by conversation/movie and remove overlap with held-out Star Wars dialogue before using either.

Review draft only. No PR comment or original Action Plan was edited.
