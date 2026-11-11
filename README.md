# Assignment 10 — Prompt optimization

Local Ollama Qwen2.5-1.5B, DSPy `Predict`, and MIPROv2 (`auto="light"`).
Generation is unconstrained; the reused A9 grammar is used only after generation
in the A10 experiment. The separate A9 reference retains constrained decoding.

## Files to review

| File | Purpose |
|---|---|
| `src/prompt_optimization.py` | A10 optimization, evaluation, and report generation |
| `src/html_evaluation.py` | Shared grammar, validator, and scoring |
| `src/a9_reference.py` | Isolated A9 GPU backend and reference experiment |
| `tests/test_prompt_optimization.py` | Deterministic tests, including mocked GPU/inference checks |

## Run

Requires Python 3.14+, uv, and local Ollama with `qwen2.5:1.5b` installed.
Provide your own local HTML-request dataset using the format below. `fixtures/`
is gitignored; your existing `fixtures/a10.json` stays on your machine.

```bash
make clean          # Reset local outputs/caches before a fresh experiment
uv sync --group dev
make                # Ruff + pytest; no inference or model downloads
make smoke DATASET=/path/to/my-data.json
make optimize DATASET=/path/to/my-data.json
make evaluate DATASET=/path/to/my-data.json
make reference      # Uses the held-out inputs saved by make evaluate
make report DATASET=/path/to/my-data.json
```

`make clean` deletes all of `artifacts/` (including saved programs, screenshots,
logs and local backups), pytest/Ruff caches, coverage output, and Python bytecode
under `src/` and `tests/`. It keeps source files, fixtures, `.venv`, and downloaded
model caches. It does not launch inference. Run the commands above afterward to
recreate the experiment outputs as on a fresh checkout.

## Supply a dataset

Use `DATASET=...` on the A10 Make commands, or invoke the CLI directly:

```bash
uv run python -m src.prompt_optimization smoke --dataset /path/to/my-data.json
```

The default path is `fixtures/a10.json`. No dataset is bundled or downloaded.
Tests generate their own temporary examples, so `make` works on a fresh clone.
Your dataset remains local; files outside `fixtures/` or `data/` should also be
kept out of Git if private.

The JSON contains `examples` and an optional `config` object. Each example needs
an ID, a split (`train`, `validation`, or `test`), a natural-language request, and
`requirements` containing `counts` and `texts`. Counts are exact tag counts;
texts are expected visible text in those tags. The sequence of `li` texts also
specifies list-item order. Every scored requirement should appear in the request.

For example, save and customize this small input file:

```json
{
  "config": {"temperature": 0.5, "max_tokens": 1024, "seeds": [11, 22, 33]},
  "examples": [
    {
      "id": "train-1", "split": "train",
      "request": "Create a complete HTML page titled Garden with one h1 heading Grow.",
      "requirements": {"counts": {"h1": 1}, "texts": {"title": ["Garden"], "h1": ["Grow"]}}
    },
    {
      "id": "train-2", "split": "train",
      "request": "Create a complete HTML page titled Trail with one h1 heading Walk.",
      "requirements": {"counts": {"h1": 1}, "texts": {"title": ["Trail"], "h1": ["Walk"]}}
    },
    {
      "id": "validation-1", "split": "validation",
      "request": "Create a complete HTML page titled Sky with one h1 heading Observe.",
      "requirements": {"counts": {"h1": 1}, "texts": {"title": ["Sky"], "h1": ["Observe"]}}
    },
    {
      "id": "test-1", "split": "test",
      "request": "Create a complete HTML page titled Robotics with one h1 heading Assemble.",
      "requirements": {"counts": {"h1": 1}, "texts": {"title": ["Robotics"], "h1": ["Assemble"]}}
    }
  ]
}
```

Use unique IDs and nonempty splits. The first test request is the screenshot
example unless `config.challenge_id` names another test ID. Defaults for model,
server, temperature, token budget, and seeds are in `DEFAULT_CONFIG` in the A10
module. Report counts follow your dataset and seed list. The measured experiment
below used 10/5/5 requests; the small example above demonstrates the file format.
This workflow supports HTML-generation datasets with the existing allowed-tag
specification, rather than arbitrary task types. Before changing datasets, move
old artifacts aside or run `make clean` to avoid combining experiments.

Malformed MIPROv2 instruction proposals are retried up to three times using the
same unconstrained model. This handles cases where the model echoes input fields
instead of returning `proposed_instruction`. Repeated failure still stops the
run. Scored HTML outputs are never retried, and JSON fallback stays disabled.

Inference requires full GPU placement, checked before generation. No CPU fallback
or sudo operation is performed. A9 unloads Ollama first to fit its 4-bit model on
the local 4 GB GPU. Its original grammar and decoding limits are retained.

**All generated files go under ignored `artifacts/`.** Programs, raw responses,
individual HTML files, scores, configuration snapshots, candidate prompts, logs,
figures and screenshot pages are kept locally rather
than pushed. `artifacts/a10_initial` preserves the first unsuccessful campaign;
`artifacts/reorganization_backup` preserves the files from before consolidation.
A new checkout must run the experiment to produce these files; seeded stochastic
inference does not guarantee identical results across hardware/software versions.

Before intentionally rerunning optimization, move the previous `artifacts/a10`
directory aside so its programs and evaluation records remain together. The A9
runner refuses to overwrite an existing reference. `make report` performs no
inference and includes A9 when its complete saved reference is present.

## Measured results

These are the completed local run's results, using the same post-generation
checks for all 45 outputs. See the PR attachments for the graph and screenshot.

| Program | Mean score | Tag/nesting validity | Exact grammar | Instruction pass |
|---|---:|---:|---:|---:|
| A9 reference | 0.981 | 15/15 | 15/15 | 12/15 |
| A10 baseline | 0.875 | 4/15 | 0/15 | 15/15 |
| A10 optimized | 1.000 | 15/15 | 0/15 | 15/15 |

A10 used identical settings: temperature 0.5, 1,024 generated tokens, one request
at a time, and seeds 11/22/33. Caching was disabled; outgoing Ollama options were
recorded to verify fresh calls and seed forwarding. Only training and validation
requests entered optimization. The selected program contains two generated
training demonstrations and was verified after saving/reloading.

The score weights structural checks and instruction checks equally. Checks cover
permitted tags/attributes, balanced nesting, document arrangement, extra content,
requested element counts/text, and requested list order. Exact grammar compliance
is reported separately. DSPy's ChatAdapter strips the final newline required by
the exact grammar, so a perfect composite score does not imply an exact match.
Stored output is never repaired.

The initial implementation incorrectly included exact grammar in demonstration
acceptance, making a full score unreachable. After correcting that error,
optimization could select fully passing demonstrations and improved the program. The baseline, dataset, model,
and sampling settings stayed unchanged. Later document/list-order audit checks
were applied equally to all saved outputs without further optimization. The same
small held-out set was reused, so this is not an independent replication or proof
of broad generalization.

A9 generated 15 fresh outputs on the local Quadro T2000, with all parameters on
CUDA and the model loaded once. It achieved exact structure but missed the
requested list type twice and item text once. Its backend, quantization
implementation, system prompt and decoding limits differ from A10. It is a
reference; **the primary experiment is A10 baseline versus optimized**. A9
outputs were never supplied to MIPROv2.

## Prompts before and after optimization

This is a snapshot of the completed experiment saved in `artifacts/a10`.
A new optimization run may select different instructions and examples; this
README does not update automatically. The current run's exact program is in
`artifacts/a10/optimized.json`, with a text export in `artifacts/a10/prompts.md`.

Both versions use a `request -> html` signature and the same DSPy ChatAdapter
field framing. The baseline starts with written instructions and no examples.
MIPROv2 selects instructions together with example request/HTML pairs, called
**demonstrations**, which are included in the model's context for new requests.
These demonstrations come from training requests, not held-out test responses.

### Before optimization

**Starting instructions:**

```text
Generate a concise complete HTML document following the request.
Use only html, head, title, body, h1, h2, p, ul, ol, li, strong, em, br.
Do not use attributes, a doctype, comments, markdown fences, or explanatory text.
Use this structure: html contains head (with title), then body. Body contains
1-6 headings, paragraphs or lists. Lists contain 1-4 li elements. Headings,
titles and list items contain plain text. Paragraphs contain plain text and
optionally strong, em or <br />. Put each block tag on its own line, while
keeping title, headings, paragraphs and li text on the same line as their tags.
Use a newline after each block, including the final </html>.
```

The baseline has no demonstrations.

### After optimization

**Instructions selected by MIPROv2, verbatim:**

```text
Create a complete HTML document titled "Task" with an appropriate heading containing "Walk", a paragraph with the text provided, and one unordered list with exactly 3 items: "Explore", "Learn", and "Explore more". Include no additional text.
```

This instruction is unusually specific to one page. It is preserved exactly as
selected, rather than presented as recommended standalone guidance. The measured
improvement belongs to the **instruction plus demonstrations together**. This
experiment does not isolate whether the instruction or examples caused it.

The selected examples below show the expected response format. JSON storage
metadata and escaped newline characters are omitted for readability; the request
and HTML text are unchanged.

#### Demonstration 1

**Example request:**

```text
Create a complete HTML document titled "Trail" with exactly one h1 heading containing "Walk" and no other headings. Include exactly one ordered list with exactly 2 items, in this order: "Water", "Boots". Include no paragraphs or other lists.
```

**Example HTML response:**

```html
<html>
  <head>
    <title>Trail</title>
  </head>
  <body>
    <h1>Walk</h1>
    <ol>
      <li>Water</li>
      <li>Boots</li>
    </ol>
  </body>
</html>
```

#### Demonstration 2

**Example request:**

```text
Create a complete HTML document titled "Cafe" with exactly one h1 heading containing "Visit" and no other headings. Include exactly one paragraph containing "Tea is ready." and no lists.
```

**Example HTML response:**

```html
<html>
<head>
    <title>Cafe</title>
</head>
<body>
    <h1>Visit</h1>
    <p>Tea is ready.</p>
</body>
</html>
```

## PR evidence

The existing local files are:

- `artifacts/a10/before_after.png`: actual browser screenshot.
- `artifacts/a10/before_after.html`: reproducible screenshot source.
- `artifacts/a10/comparison.png`: three-way comparison graph.

Upload the PNGs directly to the PR comment. The screenshot uses the preselected
Robotics request, with seed 22 chosen afterward to illustrate removal of markdown
fences and a doctype. Aggregate results include all samples.

Tests verify scoring, malformed output, requested order, split isolation, GPU-only
gating, request auditing, saved-program reload, failure recording, and reporting.
The existing CI checks remain Ruff and pytest with 100% source coverage; tests
need neither a running model nor GPU. The original PR Action Plan comment is
unchanged. Your final personal reflection remains yours to supply.
