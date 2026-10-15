# Assignment 1: BPE working notes

These notes collect your stated motivation and a proposed learning sequence. They are preparation for the assignment's PR comment, not a completed submission comment.

## Your stated reasoning, with wording polished

> I chose BPE because its use in language models influenced my choice. Reading Chapter 4.1 helped me understand how important tokenization is to later stages of the model pipeline. I want to reduce the chance of tokenization limitations causing problems in later work.
>
> I work in AI research, so learning the implementation is part of my objective. I am willing to take on extra complexity because understanding it is valuable to me.

BPE is an established method used in current models. Its adoption is a reason to study it; improvements in this assignment should be supported by comparisons with our existing tokenizer. The assignment's [Hugging Face resource](https://huggingface.co/docs/transformers/tokenizer_summary) describes BPE and its byte-level variant.

## Working direction

Explore a small byte-level BPE implementation. Bytes provide the initial vocabulary; learned merges combine frequent adjacent sequences into additional tokens. Encoding new text uses the learned merge order. Decoding reconstructs the bytes before decoding the complete byte sequence as UTF-8.

The original SimpleTokenizer provides the comparison baseline. We can retain it while experimenting so that its behavior and existing tests remain available for comparison.

## Proposed milestones

1. **Establish the assignment branch and initial PR.** Branch from `talha_shahzad_0`, as the PDF requests. Prepare the plan before publishing it as the first PR comment, and leave that comment unchanged afterward. It must address implementation, testing, and final review as well as your reasons and alternatives. Check CI at the start.
2. **Work through one merge on paper.** Start with `ababab`. Understand byte IDs, adjacent-pair counts, and replacement before adding file loading or training loops.
3. **Build the byte representation.** Check conversion from text to bytes and back on ASCII, accented text, emojis, and whitespace. Reconstruct all bytes before UTF-8 decoding; an individual token may hold only part of a multi-byte character.
4. **Learn merges on a tiny fixture.** Count adjacent pairs, select a pair, replace non-overlapping occurrences, and record the new token and rule. Discuss how to break equal-frequency ties and when to stop before implementing those decisions.
5. **Encode new text using saved rules.** Keep training separate from encoding. Test that encoding neither changes the learned vocabulary nor learns new merges from the input.
6. **Use the Star Wars dialogue.** Apply the working implementation to the same dialogue fields used for Assignment 0. Compare results on fresh examples as well as training text. Learn the merge rules from the training data only.
7. **Verify and review.** Run pytest and Ruff, keep coverage at or above the configured 80%, push changes, and confirm CI. Write the final reflection in your own words, using the observed results.

## A first example to understand

ASCII `a` and `b` have byte values 97 and 98:

```text
Text:          ababab
Initial IDs:   [97, 98, 97, 98, 97, 98]

Adjacent-pair counts:
(97, 98): 3
(98, 97): 2
```

The most frequent pair is `(97, 98)`. If the example assigns that merged token ID 256:

```text
New token:     256 represents the bytes for "ab"
After merge:   [256, 256, 256]
Decoded text:  ababab
```

The sequence is shorter and still represents the same text. This is one training step, not a complete tokenizer. The exact ID is illustrative; the implementation needs a consistent assignment rule.

## Evidence to gather

| Behavior | Proposed check |
| --- | --- |
| Exact reconstruction | Encode and decode empty text, spaces, tabs, newlines, accents, and emojis. |
| Unfamiliar input | Recover fresh text that the word-level baseline replaces with `<\|unk\|>`. |
| Correct training | Check expected pair counts and a known merge on a tiny corpus. |
| Overlapping pairs | Use an example such as `aaaa` to check replacement behavior. |
| Stable results | Train twice with the same settings and compare learned rules. |
| Fixed encoding rules | Confirm that encoding fresh text does not modify trained state. |
| Useful merges | Compare token counts before and after applying learned merges to the same text. |
| Comparison with the baseline | Report text recovery and token counts, including cases where the baseline loses information. |

A lower token count alone does not establish better tokenization: the original tokenizer can collapse an entire unfamiliar word into one unknown token. Compare information preserved as well as sequence length.

## Choices to record before finishing the PR plan

Your motivation and willingness to accept complexity are recorded above. As you work through the small example, record the alternatives you actually consider and your own reasons for selecting or rejecting them. Implementation details still to discuss include training limits, equal-frequency ties, special-token behavior, and whether to keep the new implementation in a separate file.

The PDF asks for a researched initial plan and a later reflection. Keep the initial reasoning intact, and explain later changes in the reflection.

## Project verification commands

```bash
uv run ruff format .
uv run ruff check .
uv run pytest --cov-report=term-missing
```
