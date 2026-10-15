# Assignment 3: evaluator and tests explained

Personal study notes for [src/evaluator.py](../src/evaluator.py) and
[tests/test_evaluator.py](../tests/test_evaluator.py). The actual test filename is
`test_evaluator.py`, even if I informally call it “tests evaluate.” These notes
explain the implementation; they are not an Action Plan or a PR submission.

## 1. What this assignment actually does

The program asks a pretrained model to reconstruct a missing part of dialogue.
It keeps the original tokens as the answer key, hides the chosen span, gets model
predictions, and compares those predictions with the answer key.

The model does not learn from this run. There is no optimizer, training loop,
backpropagation, or weight update.

Three pieces from the assignments have different responsibilities:

| Component | Responsibility | Used directly by this evaluator? |
| --- | --- | --- |
| Custom BPE tokenizer | Learn merges and convert text into my own token IDs | No |
| Earlier data loader | Build next-token inputs and shifted targets | No |
| BERT evaluator | Prepare masked targets and measure a pretrained BERT model | Yes |

The reusable piece is the raw dialogue dataset. BERT must receive IDs from its
matching tokenizer: the same integer can mean different tokens in different
vocabularies. My earlier data loader also prepares a different prediction task.

There are **two pretrained model roles** in this program:

| Model | Role |
| --- | --- |
| `google-bert/bert-base-uncased` | Predict the missing WordPiece tokens |
| `roberta-base` | Supply representations used by the BERTScore metric |

BERTScore does not run the reconstruction again. It scores the already produced
text against the reference text using its own encoder.

## 2. How data moves through the program

```text
JSON dialogue records
        |
        v
load_dialogues -> Dialogue objects
        |
        v
load_or_create_splits -> saved example-to-partition assignments
        |
        +---- training lines ------> baseline_token
        |
        +---- selected dev/test lines
                         |
                         v
                  prepare_examples
                         |
                         +---- skipped IDs and reasons
                         |
                         v
                   MaskedExample objects
                         |
                         v
                    predict_spans
                         |
                         v
                    frozen token predictions
                         |
                         +---- strict_metrics: accuracy and exact-span match
                         |
                         +---- decode spans -> ROUGE, BLEU, BERTScore
                         |
                         v
               JSON report and printed score summary
```

“Frozen predictions” means the same prediction lists are reused for every
metric. Each inference batch runs once; the program does not rerun BERT for each
metric. RoBERTa still performs its separate computation inside BERTScore.

## 3. The small helpers and data objects

### `_integer(name, value, minimum=0)`

This checks arguments such as seed, batch size, token ID, and mask position.
It rejects floats, strings, negative values where inappropriate, and booleans.

Why reject booleans explicitly? In Python, `isinstance(True, int)` is true. A
simple integer check could accidentally accept `True` as a batch size of 1.
The helper prevents that ambiguity. Calling it with `minimum=1` also rejects 0.

### `_digest(value)`

This converts a value to JSON, encodes it as UTF-8, and calculates a SHA-256 hash.
Dictionary keys are sorted during serialization. The result is used as a stable
identifier or ordering key; it is not a model prediction.

The program uses this helper for source identities, the dataset fingerprint,
partition ordering, and mask selection. Unlike relying on process-specific
Python string hashes, this rule can be reproduced across program launches.

### `Dialogue`

A frozen dataclass containing:

| Field | Meaning |
| --- | --- |
| `example_id` | Stable identity for the source record |
| `character` | Speaker name, retained as source information |
| `text` | Dialogue utterance supplied to the tokenizer |
| `group` | Optional scene/conversation identity |

The speaker name contributes to the source identity but is not part of the
model input. `frozen=True` prevents ordinary reassignment of fields.

### `MaskedExample`

A frozen dataclass containing an example ready for inference:

| Field | Meaning |
| --- | --- |
| `example_id` | Links the prepared example to its source |
| `text` | Original utterance, retained for inspection |
| `input_ids` | Complete token sequence with selected positions masked |
| `mask_positions` | Indices of target positions in that sequence |
| `reference_ids` | Original token IDs at the target positions |
| `word_start` | Zero-based index of the first selected lexical word |
| `char_start`, `char_end` | Character boundaries of the selected span |

Token sequences are tuples. The combination of tuples and a frozen dataclass
helps prevent accidentally modifying the answer key while preparing batches.
`char_end` is exclusive, following normal Python slicing conventions.

## 4. Reading the corpus and identifying records

`load_dialogues(path)` expects a nonempty JSON array of objects such as:

```json
[{"Character": "NARRATOR", "Line": "one two three four five six seven eight"}]
```

It verifies that `Character` and `Line` exist and are strings. An empty string
can be a valid source field; whether an utterance is usable for this task is
checked later. Invalid JSON and missing files produce their normal exceptions.

If a supported grouping field exists, the loader requires a usable value on
every record. It looks for `Conversation`, `conversation`, `Scene`, then `scene`.
If episode information exists, that information qualifies the group identifier:
scene 1 in one film should not automatically become scene 1 in another film.

The ID hashes the speaker, utterance, and group. A repeated identical record
receives an occurrence suffix such as `:0` or `:1`. That retains each source row
without giving two rows the same ID. Reordering distinct records does not change
their identities. Identical duplicates remain indistinguishable apart from the
occurrence suffix; they are not removed by this implementation.

`_validate_dialogues` performs checks for helpers receiving `Dialogue` objects:
there must be examples, IDs must be nonempty and unique, and text must be a string.

## 5. Train, development, test, and the persisted split

`load_or_create_splits(dialogues, path, seed=0)` returns a manifest: a JSON-compatible
dictionary recording which example belongs to which partition.

| Partition | Used for |
| --- | --- |
| `train` | Counting token frequencies for the simple baseline |
| `dev` | Debugging and experimenting with evaluation settings |
| `test` | Producing final reported measurements |

The `train` name does **not** mean BERT is being trained. BERT's weights remain
unchanged. Also, `eval` means evaluation generally; it is not a fourth partition.

### Creating a new manifest

The splitting unit is a whole dialogue group when grouping exists, otherwise
an individual line. Units are sorted by a hash of the seed and unit identity.
The resulting order is divided approximately 80/10/10 into train/dev/test.

The code requires at least three units and reserves at least one for each
partition. Small datasets therefore cannot always have the exact requested
percentages. With grouped data, percentages apply to groups, not necessarily
to the number of dialogue lines.

Our actual 2,523-line corpus has no grouping fields. Its saved partition sizes
are 2,018 training lines, 252 development lines, and 253 test lines.

The manifest stores its schema version, dataset fingerprint, splitting seed,
unit type, and every example's membership. The file is created with mode `"x"`,
which refuses to overwrite an already existing file.

### Reusing a manifest

The existing mapping is authoritative. The program verifies that the fingerprint,
IDs, partition labels, grouping, and required metadata are consistent. It also
checks that groups do not cross boundaries and that every partition is nonempty.
It raises an error on incompatible data instead of quietly making a new split.

Changing the global seed on a later run changes masking but does not repartition
an existing manifest. The report records the stored split seed separately from
the current masking seed.

A line-level partition does not guarantee that related lines or repeated dialogue
are independent. Its purpose here is to follow the stated fallback consistently.

## 6. Preparing one masked example, step by step

`prepare_examples` handles the conversion from source dialogue to model inputs.

### A. Find lexical words and BERT tokens

The `WORD` regex treats letters/digits as words and allows internal apostrophes
and several hyphen forms. This keeps `don't`, `don’t`, and `C-3PO` together as
lexical words. Standalone punctuation does not count as a word.

A lexical word is not the same as a WordPiece token. A name or contraction can
produce several WordPieces. The tokenizer returns both IDs and character offsets,
which connect the raw-text word boundaries to token positions.

The program requests a fast tokenizer because it needs those offsets. It uses
`truncation=False`: it measures the entire encoded input before deciding whether
the example is too long.

### B. Apply eligibility rules

The default maximum input length is 256, including `[CLS]` and `[SEP]`.
An overlength example is excluded rather than truncated.

The word requirement is:

```text
2 visible words + 4 target words + 2 visible words = at least 8 lexical words
```

For eight words, only one span meets that rule. For longer utterances there can
be multiple candidate spans. The loop starts at word index 2. Its final valid
start is `number_of_words - 6`; `range(2, len(words) - 5)` expresses this with
Python's exclusive upper bound.

A candidate's character span starts at the first target word and ends after the
fourth target word. Punctuation between those endpoints is included. Tokens
crossing a boundary, special tokens, and unknown-token IDs make that candidate
ineligible. Another valid candidate can still be selected from the same line.

If no candidate survives, the line is excluded as `no_maskable_span`.
The other reasons are `overlength` and `insufficient_words`. Excluded IDs and
reason counts are retained, rather than silently disappearing from the report.

### C. Select one candidate reproducibly

The choice is computed as:

```python
int(_digest([seed, row.example_id]), 16) % len(candidates)
```

The hexadecimal hash becomes an integer. Taking its remainder modulo the number
of valid candidates gives an index into that list.

The same record, seed, tokenizer, and masking rules produce the same choice.
It does not depend on the order of other records, their filtering, or batch size.
A different seed can select a different span but does not have to do so for every
example, especially when only one candidate exists.

### D. Preserve the answer before changing the input

The original IDs at the selected positions are copied into `reference_ids`.
Only then does the program replace those positions in a separate input list with
the tokenizer's mask ID. That prevents the reference from becoming `[MASK]` too.

Here is the simple eight-word example using the **test fixture's vocabulary**,
not the real pretrained BERT vocabulary:

```text
Text: one two three four five six seven eight

Original IDs:  [2, 5, 6, 7, 8, 9, 10, 11, 12, 3]
Masked IDs:    [2, 5, 6, 4, 4, 4,  4, 11, 12, 3]
Target indices:         3  4  5   6
Reference IDs: [7, 8, 9, 10]
```

In that fixture, 2 is `[CLS]`, 3 is `[SEP]`, and 4 is `[MASK]`.
The four masked words are `three four five six`. BERT still sees two words on
either side. The stored character boundaries are 8 and 27.

For a more difficult example:

```text
one two don't C-3PO, three four five six.
        ^^^^^^^^^^^^^^^^^^^^^^
        selected span: don't C-3PO, three four
```

Every WordPiece of the four selected lexical words is masked, along with the
internal comma. The final period remains visible because it is outside the span.
The target can therefore contain more than four token positions.

## 7. Running batched inference

`predict_spans` validates prepared examples, moves the model to the chosen device,
and performs inference on batches without changing their order.

### Evaluation mode and inference mode

`model.eval()` changes training-dependent behavior such as dropout. It does not
choose a dataset split, compute a score, or train the model.

`torch.inference_mode()` disables gradient tracking and related autograd work.
The two calls do different jobs, so the code uses both.

### Padding and attention masks

Batches can contain different input lengths. The tokenizer pads shorter sequences
on the right and provides an attention mask:

```text
Real input positions: 1
Added padding:         0
```

Padding lets the batch fit a rectangular tensor. The attention mask tells the
model which positions are real inputs. Right padding also preserves the stored
target indices. A final smaller batch is still processed.

### Logits and argmax

The output shape is:

```text
(batch size, padded sequence length, vocabulary size)
```

For every input position, the model produces a score for every vocabulary token.
These raw scores are called logits. The evaluator checks that they align with
the input and that the selected target logits are finite.

For one example, it extracts only `mask_positions`, then applies `argmax` over
the vocabulary dimension. This selects the largest-scoring token ID. Computing
softmax first is unnecessary when only the highest-scoring token is wanted.

All masks are predicted simultaneously. A prediction at one position cannot use
the newly predicted replacement at another position in the same pass. It sees
the other mask inputs. This can produce repetition or inconsistent phrases.

Predicted special tokens are not filtered. If `[SEP]` wins, it stays in the
prediction and counts as an error against a normal reference token.

The number of masks already reveals the original span's WordPiece count. This
is reconstruction with a known token count, not unrestricted sentence generation.

## 8. `strict_metrics`: measuring exact token reconstruction

This function receives lists of token-ID lists, not decoded text. For example:

```python
predictions = [[1], [2, 9, 9]]
references  = [[1], [2, 3, 4]]
```

The first example has one correct position out of one. The second has one
correct position out of three. Across the dataset:

```text
correct_tokens = 1 + 1 = 2
masked_tokens  = 1 + 3 = 4
token_accuracy = 2 / 4 = 0.5

exact_spans      = 1
example_count   = 2
exact_span_match = 1 / 2 = 0.5
```

Micro-averaging means adding the token counts before dividing. Averaging each
example's accuracy would instead give `(1 + 1/3) / 2`, approximately `0.667`.
That would give a one-token span the same weight as a three-token span. Our
metric gives every masked token position the same weight.

Exact-span match gives every example equal weight, but requires all its token
IDs to match. One wrong ID makes that example incorrect. The implementation's
`exact += matches == len(reference)` works because Python adds `True` as 1 and
`False` as 0.

Why compare IDs? Decoding can normalize spacing or join subwords. A text-only
comparison could conceal a difference in the actual reconstruction. The strict
metrics compare exactly what the model predicted at each masked position.

The function rejects empty collections, unequal example counts, empty spans,
unequal span lengths, and invalid token IDs. It reports counts alongside ratios
so the arithmetic can be checked later.

## 9. `baseline_token`: giving BERT a simple point of comparison

The baseline learns one token from the training partition:

1. Tokenize the training dialogue text with the same BERT tokenizer.
2. Count every non-special token ID.
3. Select the most frequent ID.
4. If frequencies tie, select the smaller ID.

The sorting expression is:

```python
min(counts, key=lambda token: (-counts[token], token))
```

The negative frequency makes a larger count come first; the token ID resolves
ties deterministically. Punctuation is allowed because it is a normal part of
BERT's vocabulary. In the recorded experiment, the selected token was `.`.

For a reference span with five tokens, the baseline predicts five copies of this
one token. It receives the same reference lengths as BERT and is scored with the
same strict metrics. It never uses test-token frequencies to choose its token.

This is a deliberately weak context-free baseline. Beating it shows improvement
over always predicting the most frequent training token; it does not establish
that BERT is a strong reconstruction model. No BERT weights are trained here.

## 10. `text_metrics`: comparing the decoded missing spans

This function wraps existing Hugging Face metric implementations. Our code
controls their inputs, configuration, aggregation, and reporting; it does not
reimplement ROUGE, BLEU, or BERTScore.

All three receive only the missing-span prediction and missing-span reference.
The visible surrounding dialogue is excluded from scoring. Including that
unchanged context would reward the model for text it did not reconstruct.

| Metric | What it contributes | Configuration in this evaluator |
| --- | --- | --- |
| ROUGE-1 | Unigram overlap between prediction and reference | Mean per-example F1; no stemming |
| ROUGE-L | Overlap based on the longest common subsequence | Mean per-example F1; no stemming |
| BLEU | Corpus-level n-gram precision with a brevity penalty | Maximum order 2; smoothing enabled |
| BERTScore precision | Similarity of predicted tokens to reference tokens | `roberta-base`, no IDF weighting, no baseline rescaling |
| BERTScore recall | Similarity of reference tokens to predicted tokens | Same encoder and settings |
| BERTScore F1 | Combines precision and recall for semantic comparison | Mean per-example F1; primary semantic score |

### ROUGE aggregation

`use_aggregator=False` asks the library for individual example scores. Our code
then takes their arithmetic mean:

```python
rouge = {name: sum(values) / len(values) for name, values in rouge.items()}
```

This avoids the library's stochastic bootstrap aggregation. A zero is still a
valid score and must remain in the average. For `[0.0, 1.0]`, the mean is `0.5`.

### BLEU reference shape

The BLEU wrapper expects a list of possible references for each prediction.
Although our dataset supplies only one reference per example, the required
shape is still nested:

```python
references = ["three four", "five six"]
bleu_references = [["three four"], ["five six"]]
```

BLEU is computed once over the complete prediction collection. We do not
average sentence-level BLEU scores. Order 2 uses unigrams and bigrams, which is
more suitable for these short spans than requiring four-token matches.
Smoothing prevents missing n-gram matches from automatically forcing the
entire score to zero. It does not make an incorrect phrase correct.

### Why BERTScore uses RoBERTa

There are two separate model roles:

- `google-bert/bert-base-uncased` fills the masked positions.
- `roberta-base` encodes predicted and reference text for BERTScore.

BERTScore compares contextual token representations. Its precision and recall
use token similarities rather than exact BERT token-ID matches. This can assign
partial credit to a plausible alternative, but it can also assign a relatively
high value to a phrase that fails to recover the required original wording.

The library returns precision, recall, and F1 arrays. We average each array
separately. The reported F1 is the mean of individual F1 scores, not a newly
computed harmonic mean of the aggregate precision and recall.

The report also saves the metric's `hashcode` as `bertscore_hash`. This records
configuration information that helps identify how BERTScore was calculated.
An unrescaled F1 of `0.82` should not be read as “82% of the answer is correct.”

### Input validation and the editor warnings

Before loading metrics, the function rejects empty collections, mismatched
counts, non-string values, and empty or whitespace-only strings.

Hugging Face metric computation can return `None` on a non-main distributed
process. The `if ... is None` checks provide a clear runtime error and narrow the
possible type before our code indexes the result. This explains the earlier
editor warnings on `bertscore[key]`: the possible `None` result needed handling.
That particular warning did not, by itself, establish a wrong Python path.

`metric_loader` is an optional injectable function. Normal execution uses
`evaluate.load`; tests pass controlled replacements. This lets tests verify our
wrapper's behavior without downloading the external metrics or encoder.

## 11. `evaluate_examples`: connecting the pieces

This function orchestrates scoring for already prepared examples:

```text
choose the baseline token from training
          ↓
predict all BERT spans once, using batches
          ↓
collect original reference IDs
          ↓
build baseline predictions with matching lengths
          ↓
create readable per-example records
          ↓
calculate strict metrics for BERT and baseline
          ↓
optionally calculate text metrics for BERT
```

“Once” means one call to `predict_spans` for this evaluation. That helper still
makes one model forward pass per batch. Each metric then uses the same frozen
prediction collection, so metric comparisons do not involve new model answers.

`_decode` uses `skip_special_tokens=False` and
`clean_up_tokenization_spaces=False`. A predicted special token remains visible
and is not silently removed from the answer.

Each saved example includes its original text, input IDs, mask positions,
reference IDs, character boundaries, prediction IDs, and readable prediction
and reference. `asdict(example)` converts the dataclass into a dictionary before
these extra fields are added.

The readable `masked_text` field inserts `[MISSING SPAN]` into the original
text. That is a report display marker. The actual model input uses one BERT
`[MASK]` token ID at each selected token position.

The returned report has `bert`, `baseline`, and `examples` fields, plus
`text_metrics` when enabled. The baseline receives strict metrics only in this
implementation.

## 12. `main` and `_versions`: running and recording an experiment

`main(argv=None)` supports both shell execution and direct calls from tests.
With no supplied argument list, `argparse` reads the command line. A test can
instead call `main(["some-corpus.json", ...])` directly.

| Argument | Default | Purpose |
| --- | --- | --- |
| Positional corpus path | `SW_EpisodeIV_VI.json` | Raw dialogue JSON |
| `--splits` | `data/evaluation/splits.json` | Persistent membership manifest |
| `--output` | `data/evaluation/results.json` | JSON experiment report |
| `--split` | `test` | Choose `dev` or `test` |
| `--sample-size` | Unset | Limit a development run only |
| `--seed` | `0` | Deterministic splitting and masking inputs |
| `--max-length` | `256` | Complete input length, including special tokens |
| `--batch-size` | `8` | Number of examples processed together |
| `--device` | `cpu` | PyTorch execution device |
| `--revision` | `main` | Requested BERT model/tokenizer revision |
| `--strict-only` | Disabled | Skip text metrics for debugging |

`dev` means the development partition used while building or debugging the
pipeline. `test` means the held-out partition used for the final experiment.
“Evaluation” describes the act of measuring performance and can refer to a run
on either partition; it is not an additional fourth partition.

A development subset is selected after eligibility filtering. Because prepared
examples are sorted by ID, taking `examples[:sample_size]` produces a fixed
prefix. Final test runs reject `--sample-size` and use all eligible test rows.
The preparation statistics describe the selected partition before this optional
development sampling; `sample_count` records the actual number evaluated.

The command also prevents the output path from overwriting the corpus or split
manifest. It rejects a maximum length above the model's position-embedding
capacity, even if a particular batch happens to be shorter.

The report records model and tokenizer IDs, requested and resolved BERT
revision, dataset fingerprint, split membership information, seeds, batch size,
length limit, device, exclusions, masking rules, metric settings, and package
versions. `_versions` skips a package if its distribution metadata is missing,
but continues collecting the remaining versions.

Saving uses `allow_nan=False` so invalid non-finite values cannot silently become
nonstandard JSON numbers. Parent directories are created as needed. The output
file is replaced when the same output path is used again; specify a different
path to retain multiple runs. The evaluator generates JSON, not a Markdown
results document.

Examples, run from the repository root:

```bash
# Small development run, strict metrics only.
uv run python src/evaluator.py --split dev --sample-size 8 --strict-only

# Complete eligible test set, all implemented metrics.
uv run python src/evaluator.py --split test --output data/evaluation/results.json
```

A normal full run needs the pretrained model, tokenizer, metric dependencies,
and BERTScore encoder to be available. Their initial acquisition can require
network access. The controlled test suite described below avoids that need.

## 13. How to read `tests/test_evaluator.py`

The tests ask whether the evaluator behaves correctly, even when the model's
answers are poor. A correct evaluator must report wrong predictions accurately.
It should not require pretrained BERT to achieve a particular accuracy to pass.

Most tests use this structure:

```python
# Arrange: construct a small input with an answer we can calculate.
# Act: run the relevant evaluator function.
# Assert: compare its output or error with the expected behavior.
```

### The small local tokenizer

The `tokenizer` fixture constructs a real BERT tokenizer using a small in-memory
vocabulary. Its IDs are determined by the order of the `tokens` list. For
example, `[PAD]` is 0, `[CLS]` is 2, `[MASK]` is 4, and `one` is 5.

It includes apostrophes, several hyphens, punctuation, and `##po`, allowing real
subword tokenization and character offsets to be checked. It is not a function
that simply pretends every word maps to one token. No tokenizer download is
required.

These small-vocabulary IDs are test values. They are not the IDs assigned to the
same strings by pretrained `bert-base-uncased`.

The `dialogue` helper supplies a default eight-word example and a speaker name
that is absent from the vocabulary. That makes accidentally tokenizing the
speaker field easier to detect. `write_corpus` writes small JSON files under a
test's temporary directory.

### `RecordingModel`: predictable outputs and observable calls

`RecordingModel` is a small `torch.nn.Module`, not pretrained BERT. Its forward
method constructs a logits tensor with shape `(batch, sequence, vocabulary)`.
At each position, one selected token receives score 1 and every other token
receives 0. Its winning token is therefore known without interpreting a real
language model.

The default winning ID depends on the sequence position, not the example's
location within a batch. Comparing results across batch sizes can therefore
reveal an ordering or indexing mistake. `chosen_token` can instead force the
same ID everywhere, including a special token.

The model stores copies of each call's input IDs and attention mask in `calls`.
Tests can inspect batch sizes and padding after inference. Its assertions also
verify that evaluation mode and inference mode are enabled and gradients are
disabled.

The registered `anchor` buffer gives this otherwise parameter-free model a
normal PyTorch buffer that participates in device movement. `SimpleNamespace`
provides the `.logits` and model configuration attributes expected by the code.

### Controlled metric responses

The `metric_loader` fixture returns mock metric objects with known results. For
example, mock ROUGE returns `0.5` for each ROUGE-1 score, while mock BERTScore
returns known precision, recall, and F1 arrays.

A mock also records the keyword arguments passed to its `compute` method. The
tests inspect those arguments to verify the encoder name, BLEU reference shape,
smoothing, batch size, and other settings.

This line has a subtle Python detail:

```python
metric.compute.side_effect = lambda name=name, **kwargs: compute(name, **kwargs)
```

`name=name` binds the current loop value when the lambda is created. Without
that binding, the functions would look up the loop variable later, after it had
reached its final value. All three mock metrics could then behave like the last
metric in the loop.

When a test wants a fixed `return_value` instead, it clears `side_effect` first.
Otherwise, the existing side effect would continue to determine the result.

### Useful pytest and Python tools in this file

| Tool | Role in these tests |
| --- | --- |
| `@pytest.fixture` | Supplies reusable setup by a test function's argument name |
| `tmp_path` | Gives each test a separate temporary directory |
| `monkeypatch` | Temporarily replaces functions or attributes and restores them afterward |
| `Mock` | Returns controlled values and records calls |
| `@pytest.mark.parametrize` | Runs one test logic with several input cases |
| `pytest.raises` | Checks that invalid input causes the expected exception |
| `pytest.approx` | Compares floating-point values with numerical tolerance |
| `dataclasses.replace` | Creates a modified copy of a frozen example for invalid-input tests |
| `capsys` | Captures printed output for assertions |
| `runpy.run_path` | Executes the evaluator file as `__main__` inside the test process |

## 14. What the test groups verify

### Corpus loading and persistent splitting

| Tests | What a failure would tell us |
| --- | --- |
| `test_reject_malformed_corpus`, `test_missing_or_invalid_json` | Broken input is accepted or produces the wrong failure |
| `test_ids_survive_order_and_preserve_duplicates` | Reordering changes content IDs, or duplicate rows disappear |
| `test_groups_are_qualified_by_episode` | Identically named scenes from different episodes are incorrectly merged |
| `test_partial_or_invalid_grouping_rejected` | Group metadata is incomplete or invalid but still accepted |
| `test_split_exact_membership_persistence_and_reordering` | The 80/10/10 assignment or saved membership is unstable |
| `test_group_split_never_separates_a_group` | A conversation or scene leaks across partitions |
| `test_small_split_has_all_three_partitions` | Small valid datasets produce an empty partition |
| `test_invalid_split_inputs` | Invalid IDs, duplicates, text, or grouping reach the split algorithm |
| `test_manifest_rejects_changed_corpus_without_overwriting` | A changed dataset silently reuses or overwrites an incompatible manifest |
| `test_manifest_corruption`, `test_manifest_metadata_validation` | Broken saved assignments or metadata are trusted |
| `test_reused_manifest_rejects_group_leakage` | Loading a manifest bypasses the grouping protection |

The persistence test is stronger than checking partition sizes. It compares
membership after reordering and after requesting another seed, and checks that
the existing manifest bytes remain unchanged. A new seed must not silently
replace a split that was already saved.

### Word selection and masking

| Tests | What they establish |
| --- | --- |
| `test_lexical_words_keep_internal_apostrophes_hyphens_and_unicode` | The word rule handles contractions, names, Unicode hyphens, and accented text |
| `test_mask_all_subwords_and_internal_punctuation_only` | Every token in the chosen character span is masked, including internal punctuation |
| `test_speaker_field_is_not_tokenized_or_masked` | The input contains dialogue text only |
| `test_masking_independent_of_order_filtering_and_global_rng` | The same row and seed keep the same mask across different input arrangements |
| `test_length_includes_special_tokens_and_exclusion_counts` | The length boundary and all exclusion counts are calculated correctly |
| `test_reject_incompatible_tokenizer` | Missing tokenizer capabilities fail clearly |
| `test_invalid_positive_parameters`, `test_invalid_seeds` | Invalid numerical settings are rejected consistently |

The punctuation test uses:

```text
one two don't C-3PO, three four five six.
```

The four selected lexical words are `don't`, `C-3PO`, `three`, and `four`.
The comma between them is inside the selected character interval and must be
masked too. The final period is outside and must remain visible. The test also
checks that `[CLS]` and `[SEP]` remain untouched.

The length test checks both sides of the boundary: 254 ordinary tokens plus two
special tokens fit the 256-token limit; 255 plus two do not. This catches an
off-by-two error that an ordinary short example would miss.

### Model inference and batching

| Tests | What they establish |
| --- | --- |
| `test_batching_preserves_order_references_and_final_partial_batch` | Batch sizes, padding, ordering, and original examples remain correct |
| `test_special_argmax_is_not_filtered` | A winning special-token prediction stays in the answer and is scored as wrong |
| `test_invalid_masked_examples` | Misaligned, duplicate, out-of-range, or otherwise invalid mask positions fail |
| `test_masked_example_rejects_invalid_numeric_types` | Boolean, floating-point, or negative values cannot masquerade as valid indices or IDs |
| `test_empty_inference_and_left_padding_rejected`, `test_no_pad_token_rejected` | Unsupported inference inputs fail before producing misleading output |
| `test_invalid_model_logits` | Wrong output shapes and non-finite masked-position scores are rejected |
| `test_real_tiny_bert_offline` | The inference helper works with an actual BERT masked-language-model architecture |

The batching test sends five examples with batch size two. The expected call
sizes are `2, 2, 1`. It compares predictions with runs using batch sizes one and
100, checks attention masks against padding, and verifies the original examples
were not mutated. This catches dropped final batches and reordered predictions.

The special-token test forces `[SEP]` to win at every masked position. Both its
IDs and decoded text must survive. Quietly discarding those tokens would hide
model errors and change the length of the answer.

The tiny BERT test creates `BertForMaskedLM` from a small `BertConfig`: hidden
size 16, one transformer layer, two attention heads, and intermediate size 24.
It uses random weights and local inputs. It checks the real model interface and
batch consistency, not the quality of pretrained BERT's language knowledge.

### Accuracy, baseline, and metric wrappers

| Tests | What they establish |
| --- | --- |
| `test_micro_accuracy_is_not_mean_of_span_accuracies` | Token accuracy uses global correct/total counts |
| `test_hand_calculated_accuracy` | Perfect, incorrect, and partial predictions have the expected scores |
| `test_invalid_strict_inputs` | Strict scoring rejects incompatible or invalid token collections |
| `test_baseline_counts_training_tokens_only_and_breaks_ties` | Baseline frequencies, special-token exclusion, and tie resolution are correct |
| `test_baseline_empty_training_rejected` | The baseline cannot invent a token when none is available |
| `test_metric_wrappers_configuration_and_aggregation` | All three library calls receive the intended settings and their results are aggregated correctly |
| `test_invalid_text_metric_inputs` | Invalid strings fail before the metric loader is called |
| `test_zero_scores_are_aggregated_correctly` | Zero scores remain in the denominator of the mean |
| `test_missing_metric_result_is_reported_clearly` | Each library's possible `None` response becomes a clear error |
| `test_evaluation_freezes_predictions_and_only_scores_spans` | Orchestration reuses predictions and scores only reconstructed text |

The hand-calculated tests provide an independent expected answer. Calling the
same scoring function twice and comparing the outputs would not establish that
its formula is correct.

The orchestration test checks a single model call when both examples fit in one
batch. It also inspects the strings passed into ROUGE and verifies that saved
results can make a JSON round trip. This joins several components while retaining
small, predictable inputs.

### Command-line behavior and reporting

The `cli` fixture creates a 30-row temporary corpus and replaces Hugging Face's
`from_pretrained` factories with the local tokenizer and recording model. The
real CLI preparation and reporting code still runs, but no pretrained weights
are fetched.

| Tests | What they establish |
| --- | --- |
| `test_cli_full_test_report_and_roundtrip` | A full test run writes valid scores and metadata, with 24/3/3 partition counts |
| `test_cli_fixed_dev_subset_and_no_metric_download` | Development sampling is a stable prefix and strict-only mode avoids text metrics |
| `test_cli_rejects_invalid_configuration` | Forbidden test sampling and invalid numerical settings fail |
| `test_cli_protects_source_and_manifest` | A report cannot overwrite either input data or split membership |
| `test_cli_no_eligible_examples` | An entirely excluded partition produces a clear error |
| `test_cli_rejects_model_position_overflow` | A requested input limit beyond BERT's capacity fails |
| `test_version_report_survives_missing_package_metadata` | One missing package entry does not stop recording the others |
| `test_script_entry_point_writes_a_report` | Executing the file through its `__main__` entry point produces the expected report and message |

In the strict-only test, `text_metrics` is replaced with a mock that raises if
called. This is stronger than only checking that the returned report lacks a
text-metric field: it proves the expensive step was not reached.

The entry-point test sets `sys.argv`, executes the file with
`runpy.run_path(..., run_name="__main__")`, reads the saved report, and checks
captured output. Calling `main` directly alone would not check the final
`if __name__ == "__main__"` connection.

Run the repository tests using:

```bash
uv run pytest
```

Mocked metric tests verify our integration contract. They do not independently
validate the libraries' mathematical implementations or replace a full run with
pretrained weights and real metric backends.

## 15. How to reason about the experiment

Read strict scores and semantic scores together:

- Token accuracy answers how many original BERT token IDs were recovered.
- Exact-span match answers how many entire missing spans were recovered.
- ROUGE and BLEU describe lexical overlap in the decoded spans.
- BERTScore describes similarity under a separate encoder's representations.

A model may recover a few common tokens in many examples while never recovering
an entire span. Nonzero token accuracy alongside zero exact-span match is
therefore possible and meaningful. Four lexical words can require more than
four token predictions because of subwords and punctuation.

A high BERTScore with low strict accuracy means the metrics are measuring
different properties. Inspect individual predictions before describing them as
semantically correct. The numerical similarity score alone does not verify that
the right Star Wars character, object, or action was recovered.

The final scores describe eligible dialogue lines only. Lines shorter than eight
lexical words cannot satisfy the four-word mask and two-word context on each
side. Excluding them changes the evaluated population; report the exclusion
counts with the scores.

A persisted split and deterministic masks make repeated comparisons easier to
interpret. They do not prove the pretrained model never saw similar dialogue
during pretraining. Likewise, a line-level split cannot guarantee separation of
related scenes when the source lacks grouping metadata.

BERT is appropriate for studying bidirectional masked reconstruction. This
experiment does not evaluate left-to-right next-token generation. The model sees
context on both sides and knows the number of missing token positions.

## 16. Suggested reading order and self-check questions

For a first pass through the code, follow one example through:

```text
Dialogue → prepare_examples → MaskedExample → predict_spans
         → strict_metrics / text_metrics → evaluate_examples report
```

Then read split persistence and `main` to understand how a complete experiment
selects its inputs and saves its settings. Finally, pair each function with its
smallest hand-calculated test before reading the invalid-input cases.

Questions to check your understanding:

1. Why can four lexical words require seven or more `[MASK]` tokens?
2. Why are character offsets needed instead of splitting the string on spaces?
3. Which punctuation is hidden, and which punctuation remains visible?
4. Why does the length check include `[CLS]` and `[SEP]`?
5. Why does changing input order leave an example's selected span unchanged?
6. Why does a saved split keep its membership when another seed is requested?
7. How would averaging per-example token accuracy differ from micro-accuracy?
8. Why must an incorrect special-token prediction remain in the output?
9. What does the baseline learn, and which partition can it read?
10. Why would scoring the full reconstructed dialogue inflate the apparent result?
11. Why are BERT's reconstruction model and BERTScore's encoder separate roles?
12. Which mistakes can the recording model reveal that a single real-model run might miss?
13. What does the randomly initialized tiny BERT test prove, and what does it leave untested?
14. Why do we inspect the arguments passed to mocked metrics as well as their returned scores?
15. Which saved fields would you inspect to explain a difference between two runs?
