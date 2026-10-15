# Assignment 2: Data loader explained

Personal study notes for the implementation and checks completed on September 16, 2026.
Read this alongside [the loader](../src/dataloader.py), [its tests](../tests/test_dataloader.py), and [the usage guide](../README.md).
The explanations describe the code we built; the measurements are snapshots from our checks, not promises about every machine or future model.

## Reading map

1. What A2 builds and how it connects to A1.
2. The final configuration and what each setting means.
3. Inputs, targets, windows, overlap, and exact boundary arithmetic.
4. A walkthrough of every function in `src/dataloader.py`.
5. Newlines, Unicode, and what remains unchanged in the tokenizer.
6. Shuffling, reproducibility, and batches.
7. PyTorch installation with uv and the NumPy warning.
8. GPU results and what they establish.
9. Tests, observed corpus results, and commands to run.
10. Common questions, limitations, and future work.

## 1. What did we actually build?

A1 built a tokenizer: a tool that converts text into integer token IDs and can reconstruct the text from those IDs.
A2 turns that stream of IDs into training examples and groups those examples into batches.

```text
Star Wars JSON
    |
    | existing load_corpus()
    v
Text with character names, dialogue, and newlines
    |
    | existing saved BPE tokenizer.encode()
    v
One sequence of token IDs
    |
    | our TextDataset
    v
Pairs of input tokens and next-token targets
    |
    | PyTorch DataLoader
    v
Batches delivered to a future training loop
```

### Is this already an LLM?

No. There is no neural network, learned language-model weights, loss calculation, optimizer, or text-generation loop in A2.
The BPE tokenizer was trained to learn merge rules in A1. That is different from training a language model to predict tokens.
The loader supplies examples a future model will learn from.

### What is the difference between a Dataset and a DataLoader?

| Component | Question it answers in our code |
| --- | --- |
| `BPETokenizer` | What IDs represent this text? |
| `TextDataset` | What are the input and target for example number 5? |
| PyTorch `DataLoader` | Which examples should be delivered next, grouped into a batch? |
| Future training loop | How does the model learn from this batch? |

We wrote the example construction. PyTorch supplies batching and sampling machinery.
Its [Dataset and DataLoader tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) explains this separation using images; our samples contain token sequences instead.

### Which files were involved?

| File | Role |
| --- | --- |
| `src/dataloader.py` | New dataset class and loader factory |
| `tests/test_dataloader.py` | New correctness and integration tests |
| `pyproject.toml` | Declares the PyTorch dependency |
| `uv.lock` | Records resolved package versions and distribution information |
| `README.md` | Shows how to create and inspect a loader |
| `src/bpe_tokenizer.py` | Reused unchanged |
| `models/star_wars_4096.bpe.json` | Existing saved tokenizer, reused unchanged |
| `SW_EpisodeIV_VI.json` | Existing local corpus; excluded from Git |

This notes folder is also excluded by `.gitignore`. That is why this explanation may not appear in ordinary `git status` output.

## 2. What settings did we settle on?

| Setting | Current value or behavior | Meaning |
| --- | --- | --- |
| `max_length` | `256` | Number of tokens in each input and each target |
| `batch_size` | `8` | Maximum examples delivered together |
| `stride` | `None`, resolved to `max_length` | Distance between regular window starts |
| Allowed stride | `1 <= stride <= max_length` | Prevents gaps in the predictable token transitions |
| `shuffle` | `True` | Change example order, keeping tokens within each example ordered |
| `seed` | `0` | Initial state for a private random generator |
| `drop_last` | `False` inside the factory | Retain the final smaller batch |
| `num_workers` | `0` inside the factory | Retrieve data in the calling process |
| Tensor type | CPU `torch.long` | 64-bit integer token IDs |
| Incomplete corpus tail | Add a final full window if needed | Retain the final target without padding |
| Corpus too short for one pair | Raise `ValueError` | Explain the problem instead of producing no training batches |

`max_length`, `stride`, `batch_size`, `shuffle`, and `seed` are arguments to our factory.
`drop_last` and `num_workers` are set internally; they are not currently configurable through its signature.

### Why 256 tokens and 8 examples?

You chose these as starting values for resource management, with the intention of adjusting them based on future training results.
They are not guaranteed optimal, and they do not guarantee that any model will fit on every machine.
Longer contexts can retain more conversation, but the future model's architecture and available memory determine their cost.

A token is not necessarily a word. BPE may represent part of a word, a whole word, whitespace, or a longer byte sequence.
Consequently, 256 tokens is not the same as 256 words or 256 characters.

For perspective, the published [Llama 3 report](https://arxiv.org/html/2407.21783v3#S3.T4) includes 8,192-token sequences with approximately 16 million tokens per global training batch.
The [DeepSeek-V3 report](https://arxiv.org/html/2412.19437v2#S4.S2) describes 4,096-token pretraining sequences and a global batch that eventually reaches 15,360 sequences.
Those are published examples, not a universal industry default or a ranking of today's best models.
Their global batches combine work across many GPUs and cannot be compared directly with one local batch of eight examples.

## 3. How does next-token prediction work?

Suppose the tokenizer produces these IDs:

```text
Source:  [7, 12, 9, 4]
Input:   [7, 12, 9]
Target:  [12, 9, 4]
```

At the first position, the desired next token is 12. At the second position it is 9. At the third position it is 4.
Your initial answer, "the next token is 4," correctly identified the token after the whole input.
Training at all positions supplies a target for every input position.

### Why do we need `max_length + 1` source tokens?

The final input token also needs a following token as its answer.
For length 4:

```text
Source:  A B C D E   (5 source tokens)
Input:   A B C D     (4 input tokens)
Target:  B C D E     (4 target tokens)
```

For the rest of these letter examples, assume one token per letter. The ASCII test tokenizer makes that assumption exact.

### Is the target information leaking into the model?

The loader supplies inputs and targets separately. The future training loop uses targets to calculate the prediction error.
For an autoregressive transformer, a causal attention mask must also stop an earlier input position from looking at later input positions.
That model-side masking is future work. A shifted target alone does not implement causal attention.

### Why are targets integer IDs rather than sentences or one-hot vectors?

Each target identifies the vocabulary entry the model should predict.
A future model can output scores over the vocabulary, and its loss function can compare those scores with the correct integer ID.
Our loader only needs to carry the IDs, without expanding every target into a vector of vocabulary size.

## 4. Windows, stride, and the final corpus token

Let:

```text
N = number of encoded corpus tokens
L = max_length
S = stride
```

A window pair starting at index `start` uses:

```python
inputs = tokens[start : start + L]
targets = tokens[start + 1 : start + L + 1]
```

Python slices include the start index and exclude the stop index.
The final target index is therefore `start + L`.
The final corpus index is `N - 1`, so the last possible full window starts at:

```text
last_start = N - L - 1
```

This explains the implementation:

```python
last_start = len(token_ids) - max_length - 1
self.starts = list(range(0, last_start + 1, stride))
if self.starts[-1] != last_start:
    self.starts.append(last_start)
```

The `+ 1` in `range` makes the otherwise-exclusive stop include `last_start` when the stride lands on it.
The final `if` avoids adding the same example twice.

### What happens with default stride?

For ten source tokens, `L = 4` and `S = 4`:

| Start index | Input | Target | Reason |
| --- | --- | --- | --- |
| 0 | `ABCD` | `BCDE` | First regular window |
| 4 | `EFGH` | `FGHI` | Second regular window |
| 5 | `FGHI` | `GHIJ` | Extra final window includes J as the final target |

Here `last_start = 10 - 4 - 1 = 5`. The regular starts are `[0, 4]`; we append `5`.
The regular input windows are disjoint, but the extra final window overlaps.
Even disjoint input windows share some source information through their shifted targets.
It would be incorrect to say that every token appears exactly once across all inputs and targets.

### Does overlap remove the need for a final-window policy?

No. With the same ten tokens, `L = 4` and `S = 2`, the regular starts are `[0, 2, 4]`.
None uses J as its final target. We still append start index `5`.

### Why constrain stride?

`S = L` gives disjoint regular inputs. `S < L` gives overlapping inputs.
`S > L` can skip next-token transitions between windows, so our implementation rejects it.
Together with the final window, valid strides cover every adjacent token transition in the supplied corpus.
This does not mean that every possible long context appears, or that every transition appears equally often.

### What did we accept by retaining a final full window?

The tail may be repeated in multiple examples. That creates redundant computation and gives those prediction positions greater representation.
You accepted this in exchange for retaining the corpus tail without adding padding.
For a very short corpus, this repetition can be proportionally larger than it is for the full Star Wars corpus.

### What if the corpus is too short?

| Token count | Result |
| --- | --- |
| 0 | Error |
| `L` | Error: there is no extra token for the final target |
| `L + 1` | Exactly one example, with no duplicate tail window |
| `L + 2` | For default stride and `L > 1`, one regular example and one extra final example |

For `L = 1`, the last row instead consists of two regular examples. Both configurations cover the same two next-token transitions.
An error for short data is intentional: you wanted to notice and fix an unusable training input.
The constructor reports the required token count, actual count, and suggests more text or a smaller length.

### Can we calculate the number of examples?

For valid input, let `M = N - L - 1`.

```text
regular examples = floor(M / S) + 1
extra final example = 1 if M is not divisible by S, otherwise 0
total examples = regular examples + extra final example
```

The implementation stores the starts explicitly because this is easy to inspect and understand.

## 5. Walkthrough of `src/dataloader.py`

### Imports

`torch` supplies tensors and random generators.
`Dataset` is the parent class for our indexed collection of examples.
`DataLoader` supplies iteration, sampling, and batching.
`BPETokenizer` is the unchanged A1 class used to encode text.

### `_require_positive_integer(name, value)`

This shared helper validates lengths, strides, and batch sizes.
It rejects zero, negatives, non-integers, and booleans with a useful error naming the setting.
Python treats `bool` as a subclass of `int`, so `True` needs an explicit rejection; otherwise it could accidentally mean length 1.
The leading underscore indicates an internal helper rather than the main interface callers should use.

Type hints such as `value: int` describe expected inputs; Python does not automatically enforce them.
The validation code is what enforces these constraints.

### `TextDataset.__init__`

Initialization validates the window settings, calls `tokenizer.encode(text)` once, and checks the resulting token count.
It stores one CPU tensor and a list of example start positions.
The sufficient-data check must happen after tokenization: the number of text characters does not tell us the number of BPE tokens.

`stride=None` means "use the supplied max_length." If you change length to 128 without giving a stride, regular starts advance by 128.
We did not hard-code a stride default of 256 that could become inconsistent with a different length.

### Why `torch.long` and `device="cpu"`?

`torch.long` is the 64-bit integer type used here for token IDs, also displayed as `torch.int64`.
The IDs are identifiers, not floating-point measurements.
CPU placement is explicit so data preparation stays independent of the training device.
No model gradients are computed for the integer IDs themselves.

### `TextDataset.__len__`

This returns the number of examples, which is the number of stored start positions.
It does not return the number of characters, tokens, or batches.

### `TextDataset.__getitem__`

`dataset[index]` looks up one start position and returns the two shifted tensor slices.
For example, `dataset[0]` returns the first pair, and `dataset[-1]` returns the last pair through normal Python list indexing.
An out-of-range index raises `IndexError` through the starts list.

The slices are views of the underlying token tensor. Do not edit a returned sample in place: that could alter the stored corpus and other overlapping examples.
Use `.clone()` if you need a separate mutable sample.

### What does "most work upfront" mean?

File loading happens before the factory call; encoding happens once during dataset construction.
Later iteration retrieves slices and lets PyTorch assemble batches.
Reusing the same loader for another epoch does not retrain the tokenizer or encode the text again.
Calling `create_dataloader` again constructs another dataset and therefore encodes again.

We store the token sequence once rather than retaining separate copies of every overlapping window.
There is still memory for the original text, tokenizer, temporary Python token list, start indices, and batches.
This is an in-memory design, not disk streaming.

### `create_dataloader(...)`

The factory checks batch size, creates the dataset, and seeds a dedicated `torch.Generator`.
It then returns a normal PyTorch `DataLoader` configured with that dataset and generator.
It does not download files, load a saved tokenizer automatically, select a GPU, train a model, or generate text.
The caller explicitly supplies the text and tokenizer, making tiny examples and the actual corpus usable through the same interface.

`num_workers=0` means no extra data-loader worker processes. It does not mean PyTorch can never use CPU threads or a GPU elsewhere.
Our retrieval work is small tensor slicing; worker processes are not necessary for this implementation.

The function returns a loader object. `iter(loader)` creates an iterator over batches, and `next(iter(loader))` gets its first batch.
Repeatedly calling `next(iter(loader))` starts new iterations; use one `for` loop to visit a complete epoch.

## 6. Dialogue boundaries, newlines, and Unicode

### Does a window have to be one complete dialogue line?

No. You chose a continuous stream that can span dialogue lines.
A short line continues into following text until the window has enough tokens.
A longer line may extend across windows. There is no padding just because one speaker's line is short.
Scripts also contain scene and episode boundaries; treating the corpus as one stream can cross those boundaries too.
That simplification does not claim that the entire dataset is literally one uninterrupted conversation.

### Exactly how are character names and newlines preserved?

The unchanged `load_corpus` reads each record's `Character` followed by its `Line`, then joins all fields with `\n`.
For example:

```text
LEIA
Hello!
LUKE
Hi, café🙂!
```

Existing whitespace inside fields is also preserved. If a field already ends in a newline, joining may produce two consecutive newlines.
The reader does not guarantee an extra newline after the final field.
You explicitly chose to keep that existing behavior rather than change the tokenizer or corpus reader in A2.

### Is there a special end-of-dialogue token?

No. A newline is represented as bytes by the existing BPE vocabulary and may become part of a merged token.
It is not guaranteed to be a standalone token with ID 10 in the trained tokenizer's output.
There is also no dedicated padding, end-of-sequence, or dialogue-ending token added in A2.
Newlines occur after character names as well as between dialogue records, so they are not an exclusive dialogue-end marker.

### Why might decoding one window fail?

Some Unicode characters require multiple UTF-8 bytes. Our tokenizer can assign different tokens to different portions of those bytes.
A fixed token window can begin or end inside a character's byte representation.
The model can still consume the integer IDs, but decoding that isolated byte sequence can raise `UnicodeDecodeError`.
This does not mean the token IDs were corrupted.

For inspection, combining an input with its last target reconstructs the full source window of `L + 1` tokens:

```python
source_window = inputs[0].tolist() + [targets[0, -1].item()]
```

That combination still might cut a Unicode character at either outer boundary.
It is guaranteed to recover the original text when it reconstructs the entire valid encoded text, as our integration test does.
To inspect arbitrary fragments even at broken byte boundaries, join their token bytes and decode with replacement explicitly:

```python
fragment_bytes = b"".join(tokenizer.vocab[token_id] for token_id in source_window)
print(fragment_bytes.decode("utf-8", errors="replace"))
```

Replacement characters in this display are a debugging convenience. They must not be used to rewrite the training data.

## 7. Batches, epochs, and reproducible shuffling

### How is a batch different from a window?

A window determines one input/target pair. A batch groups several pairs.
With length 256 and batch size 8, a full batch has:

```text
inputs.shape  = [8, 256]
targets.shape = [8, 256]
```

Each row is a separate example; each column is a position within that example.
There are 2,048 input positions and 2,048 corresponding target positions.
The final smaller batch has fewer rows but still has 256 columns.

### What are the two different leftovers?

| Leftover | Stage | Our response |
| --- | --- | --- |
| Tokens that do not fit the regular window starts | Dataset construction | Add the final full overlapping window |
| Fewer than eight completed examples remaining | Batching | Keep the smaller batch with `drop_last=False` |

For ten examples and batch size four, keeping the smaller batch produces sizes 4, 4, and 2.
`drop_last` does not determine how token windows are constructed. Our tests deliberately check these policies separately.

### What is an epoch?

One complete pass through the examples in the loader.
The same examples can be revisited over many epochs when training a model.
An epoch is not one batch and not one token.

### Does shuffling scramble the conversation inside each example?

No. It changes the order of complete examples, keeping each input paired with its target and keeping tokens ordered inside both.
Examples are treated as separate contexts. This design does not carry a hidden conversation state from one shuffled batch to the next.
Use `shuffle=False` to inspect chronological example order.

### How does the seed work?

The code creates a private generator and initializes it once:

```python
generator = torch.Generator().manual_seed(seed)
```

That generator is passed to the DataLoader. The default seed is 0; it has no special learning advantage over 17 or 42.
A fixed seed is useful because it makes the shuffle sequence reproducible for the same data, settings, and environment.
Its state advances when the loader is reused, so later epochs can have different orders while the sequence of orders is repeatable across equivalent runs.
We do not reset it at the beginning of each epoch.

The [PyTorch reproducibility guide](https://docs.pytorch.org/docs/2.14/notes/randomness.html#dataloader) documents using a generator with DataLoader.
A loader seed does not guarantee that every future training calculation is identical across hardware, library versions, or nondeterministic GPU operations.
Additional random operations or worker-side transformations would need their own reproducibility treatment if added later.

### Why use a private generator rather than only a global seed?

It gives this loader its own random-number state. Unrelated calls to PyTorch's default random generator do not consume its sequence.
Our test also checks that constructing and iterating the loader does not change the global PyTorch RNG state.

### Must two different seeds always give different orders?

No. Random permutations can coincide, especially with tiny datasets; a one-example dataset has only one possible order.
We use enough identifiable examples to check deterministic behavior and changing epoch order in the selected test case.
We do not claim every pair of epochs is mathematically guaranteed to differ.

## 8. Installing PyTorch with uv

### What command was actually run?

```bash
uv add torch
```

It declared the dependency, resolved its requirements, updated the lockfile, and installed packages in this project's `.venv`.
The resolved package was PyTorch 2.14.0, with the installed Linux build reporting `2.14.0+cu130`.
The project uses Python 3.14; the checked interpreter reported Python 3.14.7.

### Why did `pyproject.toml` change?

It now includes:

```toml
dependencies = [
    "torch>=2.14.0",
]
```

This says the project needs PyTorch version 2.14.0 or later, subject to dependency resolution.
It is a dependency requirement, not an exact-version pin by itself.

### Why did `uv.lock` change so much?

PyTorch brings dependencies of its own. The lockfile records exact resolved versions, relationships, and package distribution information.
The Linux installation also downloaded CUDA-related packages, which account for much of the installation size.
The lockfile was generated by uv, not edited manually.
Both `pyproject.toml` and `uv.lock` belong in the repository so another checkout can reproduce the environment.
The installed `.venv` itself does not belong in Git.

### What do the common uv commands mean?

| Command | Purpose here |
| --- | --- |
| `uv add torch` | Declare, resolve, and install PyTorch as a project dependency |
| `uv sync` | Synchronize the environment with the project and lockfile |
| `uv sync --locked` | Synchronize while refusing to update an out-of-date lockfile |
| `uv run python ...` | Run Python in the project environment |
| `uv run pytest` | Run tests using that environment |

Run commands from the repository root so the relative model and corpus paths resolve.
You do not need to activate `.venv` manually when using `uv run`.

### Did uv install my NVIDIA driver?

No. The driver was already present. The PyTorch build supplies CUDA user-space components through installed packages; a working compatible system driver is still required.
Our checks established that the installed build and this machine's driver can execute CUDA operations together.
The `cu130` suffix identifies the build's CUDA 13.0 variant.
A CUDA-enabled PyTorch build can also execute CPU operations when a usable GPU is absent.

Package builds and accelerator defaults vary by platform and version. See the [uv PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/) before changing the accelerator setup on another machine.
The current project has an ordinary `torch` dependency; we did not configure separate CPU/GPU extras or custom indexes.

## 9. The NumPy warning: what it means and why we left it visible

During tests and GPU diagnostics, PyTorch emitted:

```text
UserWarning: Failed to initialize NumPy: No module named 'numpy'
```

The warning originated in PyTorch during initialization. It did not come from a failed tokenization assertion or a CUDA driver error.

### What is NumPy?

NumPy is a Python package for numerical arrays and operations.
PyTorch has its own tensor implementation, but it also provides ways to exchange data with NumPy arrays.
NumPy was not installed in the checked project environment.

### Why can PyTorch work without it here?

Our data path is:

```text
Python text -> Python list of integer token IDs -> torch.tensor(...)
```

It does not construct NumPy arrays or convert tensors to NumPy arrays.
The loader uses PyTorch tensor slicing, batching, `.tolist()`, and device transfers.
Those operations succeeded in the checks we performed without NumPy installed.

### Why did we ignore this warning for A2?

We left the warning visible and accepted it for the current scope because:

- No A2 code depends on NumPy interoperability.
- All 171 tests passed, including the new loader tests.
- The full-corpus check preserved the encoded text and covered the final target.
- GPU transfers, a small CUDA calculation, and backward propagation succeeded.
- CPU fallback was also verified.

This is evidence that the warning did not block these paths. It is not a claim that NumPy is unnecessary for every future PyTorch feature.
We did not suppress all warnings, add a blanket warning filter, or install an unused dependency solely to hide this message.

### When would we need to address it?

If later code imports NumPy or uses a NumPy interoperability path, install NumPy as a project dependency and verify that path.
For example, [torch.from_numpy](https://docs.pytorch.org/docs/2.14/generated/torch.from_numpy.html) creates a tensor from a NumPy array.
[Tensor.numpy](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.numpy.html) produces a NumPy array from a suitable tensor.
Those bridges need NumPy; their documentation also specifies restrictions such as device and gradient state.

The project-level installation command would be:

```bash
uv add numpy
```

That would update `pyproject.toml`, `uv.lock`, and `.venv`, just as adding torch did.
This command is documented as a future option; it was not run while writing this guide.

### Is every warning safe to ignore?

No. Diagnose its source and the affected operation.
A missing optional bridge is different from an unsupported GPU architecture, nonfinite training results, or incorrect input shapes.
If a future program fails at a NumPy operation, the current warning is relevant and must be addressed.

## 10. What we checked on your GPU

These are measurements from the September 16, 2026 diagnostic run:

| Item | Observed result |
| --- | --- |
| GPU | NVIDIA Quadro T2000 |
| Physical VRAM reported by NVIDIA-SMI | 4,096 MiB, or 4 GiB |
| Driver | 580.173.02 |
| Installed PyTorch | 2.14.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| CUDA device count | 1 |
| Compute capability | 7.5 |
| PyTorch build architectures | Included `sm_75` |
| CUDA availability | True |
| Native BF16 support | False |
| Free VRAM after the batch/compute check | Approximately 3.44 GiB |
| Peak PyTorch allocation in that small diagnostic | Approximately 16.31 MiB |

NVIDIA's [Quadro mobile specifications](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/documents/quadro-rtx-mobile-line-card-us-nvidia-r7-web.pdf) also list the T2000 with 4 GB of memory.
Memory readings vary with other programs, driver/runtime reservations, and the lifetime of CUDA contexts.
The CUDA memory query reported a usable total of approximately 3.62 GiB in that process, rather than the full physical 4 GiB.
These values should not be treated as permanent limits on what is currently free.

### What GPU tests actually passed?

1. PyTorch detected the device.
2. An actual Star Wars batch was transferred to `cuda:0`.
3. Inputs and targets retained their integer values after a GPU-to-CPU round trip.
4. Both batch tensors had shape `[8, 256]` and type `torch.int64`.
5. A small float32 matrix multiplication ran on CUDA.
6. Backward propagation through that diagnostic calculation produced finite gradients.
7. With CUDA temporarily hidden from another process, device selection chose CPU and the loader still worked correctly.

The matrix calculation was only a diagnostic. It did not train a language model or measure language quality.
The actual diagnostic programs were run through the terminal; no new model-training code was added to the repository.

### Why did one BF16 query say True before we reported False?

The installed PyTorch API defaults to allowing emulation in `torch.cuda.is_bf16_supported()`.
The more specific query `torch.cuda.is_bf16_supported(including_emulation=False)` returned False.
The earlier True therefore did not establish native BF16 hardware support.
Our actual arithmetic diagnostic used float32, not BF16.

### Is this GPU good enough?

It is enough for A2's data movement and small numerical experiments.
It is a reasonable device on which to explore a small educational model, but we have not measured an actual model's training memory or speed.
A model's weights, activations, gradients, optimizer state, and temporary workspaces all consume memory.
Batch size 8 and length 256 do not guarantee that a model of arbitrary size fits.

For perspective, seven billion parameters stored at two bytes each already require about 14 billion bytes for weights alone, before training storage.
Full training of such a model entirely on this 4 GiB GPU is outside its memory capacity.
That statement is about full model training on the GPU, not every possible quantized inference or CPU-offload experiment.

### Why does the batch itself use so little memory?

Each input tensor has `8 * 256 = 2048` IDs. Each int64 ID uses 8 bytes.
Inputs plus targets therefore use:

```text
2 * 8 * 256 * 8 bytes = 32,768 bytes = 32 KiB
```

These tensors contain IDs, not model parameters or embedding vectors.
The CUDA context and computation workspaces also use memory; 32 KiB is not the entire program's GPU footprint.

### Does a GPU automatically make every part faster?

No. Our tokenizer is ordinary Python code and runs on the CPU.
Reading JSON, constructing windows, and moving tiny tensors may not benefit from GPU execution.
The future neural network's large tensor operations are the intended GPU workload.
We verified functionality, not a CPU-versus-GPU speedup.
A meaningful speed comparison should time representative model work and account for asynchronous CUDA execution.

### Where will CPU fallback live?

The future training caller can choose:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The loader always returns CPU tensors. The caller transfers them, and the model, to its chosen device.
This is selection when CUDA is unavailable; it is not an automatic recovery mechanism for a GPU out-of-memory error halfway through training.
If the model runs out of VRAM, we would need to adjust its settings or explicitly choose CPU.

## 11. Tests and what each one protects

The new test file produced 34 test cases after pytest expanded its parameterized inputs.
Our prior full test run passed 171 tests with 100% statement coverage for the measured source files, including `src/dataloader.py`.
The configured requirement is 80% overall coverage. The run reported one NumPy warning.
These are the implementation-run results; this document does not claim the full suite was rerun merely to write notes.

| Test | What it establishes |
| --- | --- |
| `test_exact_windows_and_tail` | Exact values, shifted targets, stride choices, tail handling, length-one behavior, CPU placement, and integer dtype |
| `test_every_next_token_transition_is_covered` | With distinct test tokens and each allowed stride for length 4, all expected adjacent transitions occur |
| `test_insufficient_data_raises_helpful_error` | Empty, very short, and exactly-L-token inputs raise the selected error |
| `test_invalid_max_length` | Rejects zero, negative, fractional, and boolean lengths |
| `test_invalid_stride` | Rejects invalid strides and strides larger than the length |
| `test_invalid_batch_size` | Rejects invalid batch sizes |
| `test_tokenizes_once_before_iteration` | Two epochs over one loader do not re-encode the corpus |
| `test_keeps_smaller_batch_separately_from_overlapping_tail` | Three examples, including the final overlapping window, form batches of 2 and 1 with exact values |
| `test_default_length_stride_and_batch_size` | Defaults produce lengths of 256, stride 256, and batches of 8 followed by 1 for the test corpus |
| `test_shuffle_repeats_across_runs_but_advances_between_epochs` | Equivalent loader runs reproduce orders, successive tested epochs differ, membership is preserved, and global RNG state is unchanged |
| `test_saved_tokenizer_and_corpus_preserve_dialogue_and_unicode` | The real saved vocabulary integrates with JSON loading, preserves complete text, and is not modified by encoding |
| `test_batch_can_be_transferred_by_training_code` | CPU batches can move to CUDA and back without value changes; skipped if CUDA is unavailable |

### Why use an untrained tokenizer in many tests?

Its initial byte vocabulary maps ASCII letters directly to their byte values.
That makes expected results independent of learned merges and easy to calculate manually.
A separate integration test uses the real saved tokenizer so both simple arithmetic and real-model compatibility are exercised.

### What are fixtures, parameterization, and mocks doing here?

A fixture supplies a fresh tokenizer to a test.
Parameterization runs one test function against several explicit cases, such as `L`, `L+1`, and tail-overlap configurations.
`Mock(wraps=tokenizer.encode)` still calls the real encoder while counting its calls. It verifies our upfront-encoding behavior rather than replacing it with fake token IDs.

### Does 100% coverage mean there can be no bugs?

No. Statement coverage says the measured lines executed during tests.
It does not prove all possible configurations, platform differences, or future changes are correct.
The exact expected-value tests, boundary tests, and integration checks provide more useful evidence than the percentage alone.

### What did the actual full-corpus smoke test show?

With the saved A1 tokenizer and the current local Star Wars file:

| Measurement | Result |
| --- | --- |
| Encoded corpus tokens | 32,078 |
| Examples at length 256 and default stride | 126 |
| Batches at size 8 | 16 |
| Final batch size | 6 examples |
| Final target | Matched the final encoded corpus token |
| Complete encoded-text round trip | Matched the original loaded text |

The arithmetic explains the counts:

```text
N = 32078, L = 256, S = 256
last_start = 32078 - 256 - 1 = 31821
last regular start = 31744
regular examples = 125
extra final example = 1
all examples = 126
batches = 15 full batches of 8 + 1 batch of 6
```

The final start is 77 tokens after the preceding start, so those two input windows overlap by `256 - 77 = 179` tokens.
Across this default dataset, there are `126 * 256 = 32256` target positions for `32078 - 1 = 32077` distinct positional next-token transitions.
The extra 179 positions are repeated training coverage from the final-window policy.
Distinct positional transitions can have identical token values; this calculation counts positions, not unique vocabulary pairs.

## 12. Commands and examples to run yourself

Run these from `/home/tshahzad/projects/USU-LLM-Class-F26`.
The snippets are instructions you can run later; documenting them does not execute them.

### Synchronize dependencies

```bash
uv sync
```

### Inspect a tiny example without the Star Wars file

```bash
uv run python - <<'PY'
from src.bpe_tokenizer import BPETokenizer
from src.dataloader import TextDataset

tokenizer = BPETokenizer()
dataset = TextDataset("ABCDEFGHIJ", tokenizer, max_length=4)
print("Start positions:", dataset.starts)
for inputs, targets in dataset:
    print(tokenizer.decode(inputs.tolist()), "->", tokenizer.decode(targets.tolist()))
PY
```

Expected pairs:

```text
ABCD -> BCDE
EFGH -> FGHI
FGHI -> GHIJ
```

### Load the real data and transfer a batch when CUDA is available

```bash
uv run python - <<'PY'
import torch
from src.bpe_tokenizer import BPETokenizer, load_corpus
from src.dataloader import create_dataloader

tokenizer = BPETokenizer.load("models/star_wars_4096.bpe.json")
text = load_corpus("SW_EpisodeIV_VI.json")
loader = create_dataloader(text, tokenizer, max_length=256, batch_size=8, seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs, targets = next(iter(loader))
inputs, targets = inputs.to(device), targets.to(device)
print("Examples:", len(loader.dataset), "Batches:", len(loader))
print("Inputs:", inputs.shape, inputs.dtype, inputs.device)
print("Targets:", targets.shape, targets.dtype, targets.device)
PY
```

`.to(device)` returns a tensor on the requested device. Assign its return value rather than assuming the original CPU tensor was moved in place.
This example only transfers data; it does not train a model.

### Run the project checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

To apply formatting, use `uv run ruff format .`.
To inspect just the loader tests while learning:

```bash
uv run pytest tests/test_dataloader.py --no-cov -v
```

`--no-cov` avoids applying the repository-wide coverage threshold to only a subset of tests. It is not a replacement for the full project checks before submission.

### Inspect GPU availability again

```bash
nvidia-smi
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 13. Common questions and troubleshooting

### Why does running `uv run src/dataloader.py` not give me a demonstration?

The file defines reusable classes and a factory; it has no command-line entrypoint.
Use the examples above from the repository root. They also ensure that imports of `src.bpe_tokenizer` resolve correctly.

### Why does startup take time even though batches are small?

The entire supplied text is encoded when the dataset is created.
The existing Python BPE encoder applies its saved merge rules, which can be noticeable for the whole corpus.
Once initialization finishes, iterating the same loader uses stored tokens rather than redoing that work.
GPU installation does not accelerate that Python encoder automatically.

### Why are there only 126 examples when there are thousands of dialogue lines?

Examples are token windows, not JSON records or lines.
With 32,078 tokens and length 256, the count follows window arithmetic.
Reducing stride produces more examples through reuse of the same data, not new independent conversations.

### Why does the loader raise an error for text that looks long enough?

Length is measured after BPE encoding. A large number of characters may compress to relatively few tokens.
Check the actual encoded length or reduce `max_length`.

### What if a file is missing or the JSON is malformed?

File reading happens through the existing `load_corpus` and `BPETokenizer.load` functions before loader creation.
Check the working directory and file paths. The model file is versioned; the full corpus is intentionally local and ignored by Git.
The loader does not silently download missing files or repair malformed data.

### Why are all eight rows not from adjacent parts of the script?

Shuffling is on by default. Use `shuffle=False` for chronological inspection.
Each row still contains an ordered token sequence with its matching target.

### Should we pad with token ID 0 if we change the design later?

Not without a deliberate padding design. In the byte vocabulary, ID 0 is already a real byte value.
Padding would require a chosen representation and excluding padded targets from learning, with any needed attention handling.
Our current design avoids padding entirely.

### Can the loader answer Star Wars questions?

No. It organizes examples. A future trained language model would generate answers.
Training on a script does not by itself guarantee reliable question answering, even though the corpus is conversational.

### Do the tests require the full Star Wars file or a GPU?

The A2 tests use small built-in strings and temporary JSON data, plus the saved tokenizer already in the repository.
They do not require downloading the full corpus. The CUDA transfer test skips when a CUDA device is unavailable.
The full-corpus smoke test was a separate local check.

### What would matter when adding train/validation sets?

Split the source data before constructing windows, so overlapping windows from the same passage do not leak between sets.
The current saved tokenizer was trained on the complete A1 corpus. A strictly isolated future evaluation would also need to account for that tokenizer-training exposure.
No train/validation split or tokenizer retraining was added in A2.

### What would we revisit for a much larger dataset?

Holding all text and token IDs in memory may become expensive.
Possible later approaches include storing pretokenized data on disk or streaming it, but these are future options rather than implemented features or decisions already included in your Action Plan.

### Are we done with the GitHub submission just because tests passed locally?

Local tests establish local behavior. They do not establish GitHub CI status or prove that required comments have been posted.
The assignment also asks for the initial Action Plan, passing CI, and a later reflection explaining what happened and why the tests were chosen.
In your course workflow, assignment branches build on prior work and PRs are tracking records that remain unmerged.
Keep the original first plan comment intact; describe any later changes in the reflection.
This guide is a learning reference, not a claim that those GitHub steps have been completed.
