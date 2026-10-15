# Understanding `tests/test_tokenizer.py`

This is a walkthrough of your submitted [test file](tests/test_tokenizer.py), in its original order. The tests exercise [src/tokenizer.py](src/tokenizer.py). Neither Python file was changed to create this guide.

## Start with one assertion

Imagine manually checking your tokenizer:

```python
ids = tokenizer.encode("Hello world")
print(ids)
```

You look at the output and decide whether it is correct. An automated test writes down the expected answer and makes Python compare it:

```python
assert tokenizer.encode("Hello world") == [10, 20]
```

Here, `==` compares two values. `assert` requires the comparison to be true. If the actual result is `[10, 60]`, the assertion raises `AssertionError`, and pytest reports a failure with details about the mismatch. If it matches, execution continues. A test passes when it finishes without an unexpected error or failed assertion. [Assertion documentation](https://docs.pytest.org/en/stable/how-to/assert.html)

The IDs `[10, 20]` belong to the small vocabulary in these tests. They are not the IDs from your full Star Wars dataset.

Running `uv run pytest` starts pytest in the project environment. Pytest discovers files such as `test_tokenizer.py` and functions whose names start with `test_`. You do not need to call those functions yourself or add a `main()` function. [Getting started with pytest](https://docs.pytest.org/en/stable/getting-started.html)

## The imports

```python
import json
import runpy
from pathlib import Path
from unittest.mock import mock_open

import pytest
```

| Import | Its job in this test file |
| --- | --- |
| `json` | Turns our small Python dataset into JSON text using `json.dumps()`. |
| `runpy` | Executes the actual tokenizer script and gives us its global variables in a dictionary. |
| `Path` | Locates `src/tokenizer.py` relative to this test file. |
| `mock_open` | Creates a replacement for `open()` that serves our JSON text without reading the dataset from disk. |
| `pytest` | Provides fixture declarations, parameterized tests, and expected-exception checks. |

The first four imports come from Python's standard library. `pytest` is a separate package listed in your project's development dependencies. Using `unittest.mock` does not mean these are unittest tests; pytest tests can use that library too.

## The `dialogue` fixture supplies sample data

```python
@pytest.fixture(scope="module")
def dialogue():
    """Small dataset with the same fields as the real Star Wars JSON."""
    return [
        {"Character": "LUKE", "Line": "Hello, world!"},
        {"Character": "LEIA", "Line": "Hello, café!"},
    ]
```

A **fixture** is reusable setup that supplies something a test needs. The line beginning with `@` is a Python decorator: here it marks `dialogue` as a pytest fixture. A test or another fixture requests it by using the argument name `dialogue`; pytest supplies its returned value. `scope="module"` shares one result within this test file for a normal test run. Without an explicit scope, pytest creates a fresh fixture result for each test case that requests it. [Fixture documentation](https://docs.pytest.org/en/stable/how-to/fixtures.html)

In our case, the returned value is a list of two dictionaries. Each dictionary has the same `Character` and `Line` fields as a real dataset record. This gives us enough information to check several behaviors:

- `Hello` occurs twice, so we can check vocabulary deduplication.
- Commas and exclamation marks let us check punctuation splitting.
- `café` contains a non-ASCII character.
- Character names should not become dialogue tokens.

The triple-quoted string is a **docstring**, which documents the function's purpose. It does not perform a test.

## The `tokenizer_script` fixture runs your code with the sample data

```python
@pytest.fixture(scope="module")
def tokenizer_script(dialogue):
    """Run the unchanged script structure with a fake dataset file for CI."""
    path = Path(__file__).resolve().parents[1] / "src" / "tokenizer.py"
    fake_open = mock_open(read_data=json.dumps(dialogue, ensure_ascii=False))
    return runpy.run_path(str(path), init_globals={"open": fake_open})
```

This is the most specialized part of your tests. Your tokenizer opens the JSON file immediately when the script runs, including during a normal import. The fixture lets us preserve that structure while controlling which data it reads. The real dataset is ignored by Git, so CI cannot depend on having that file.

The argument `dialogue` receives the list from the previous fixture. Here is what each remaining line does.

### Locate the actual tokenizer script

```python
path = Path(__file__).resolve().parents[1] / "src" / "tokenizer.py"
```

`__file__` identifies this test file. `Path(...)` represents its filesystem path, and `.resolve()` produces an absolute resolved path. For this file:

```text
.../USU-LLM-Class-F26/tests/test_tokenizer.py
                         │
                         ├─ parents[0]: tests/
                         └─ parents[1]: USU-LLM-Class-F26/
```

With `Path` objects, `/` joins path components. Appending `"src"` and `"tokenizer.py"` therefore locates the script in your project, independently of the terminal's current directory.

### Prepare a fake file reader

```python
fake_open = mock_open(read_data=json.dumps(dialogue, ensure_ascii=False))
```

Read the nested call from the inside out:

1. `dialogue` is the Python list of dictionaries.
2. `json.dumps(...)` serializes it into JSON text. `ensure_ascii=False` keeps the literal `é` in that text.
3. `mock_open(read_data=...)` creates a mock file opener whose file-like object returns that text when read. It supports the `with open(...) as file:` pattern used by your script. [Python's mock_open documentation](https://docs.python.org/3.14/library/unittest.mock.html#mock-open)

A mock is a controlled replacement for a dependency. In this file, the dependency being replaced is file access. The tokenizer, regex operations, vocabulary construction, and JSON parsing still execute normally. No API or language model is used by this tokenizer, so none needs to be mocked.

Compare these two directions:

```text
Test fixture:       Python list → json.dumps() → JSON text
Tokenizer script:   JSON text   → json.load()  → Python list
```

`dumps()` returns a string; `load()` reads JSON from a file-like object. The mock supplies that object without creating a real JSON file.

### Execute the script and keep its results

```python
return runpy.run_path(str(path), init_globals={"open": fake_open})
```

`str(path)` converts the path object to a string. `run_path()` executes that Python file. `init_globals` supplies initial global names, so the script's `open` name refers to `fake_open` instead of Python's built-in file opener. This does not globally replace `open` for the whole test process. The result is a dictionary containing the executed script's global names. [Python's runpy documentation](https://docs.python.org/3.14/library/runpy.html)

In your script, those names include:

| Dictionary lookup | Value we get |
| --- | --- |
| `tokenizer_script["text"]` | Combined dialogue text. |
| `tokenizer_script["preprocessed"]` | Words and punctuation before vocabulary construction. |
| `tokenizer_script["vocab"]` | Automatically constructed token-to-ID mapping. |
| `tokenizer_script["SimpleTokenizer"]` | The actual class defined in your script. |
| `tokenizer_script["tokenizer"]` | The instance created at the bottom of your script. |

This setup is specific to your top-level file loading. In code that can be imported without reading data immediately, tests can often import the class directly.

## The `tokenizer` fixture supplies a fresh object with known IDs

```python
@pytest.fixture
def tokenizer(tokenizer_script):
    # Use fixed, non-consecutive IDs so we can check exact expected results.
    vocab = {
        "Hello": 10,
        "world": 20,
        ",": 30,
        "!": 40,
        "<|endoftext|>": 50,
        "<|unk|>": 60,
    }
    return tokenizer_script["SimpleTokenizer"](vocab)
```

The last line retrieves the class, then calls it with `vocab`, creating a new instance. For this object, encoding `Hello` must give `10`, regardless of the full dataset's vocabulary.

These fixed values let the encode/decode tests check expected answers independently of the vocabulary-building algorithm. The gaps between IDs also check that the class respects supplied IDs rather than assuming they are consecutive positions.

This fixture has the default function scope. Each requesting test case gets a fresh instance. The two module-scoped setup results are reused. The dependencies are:

```text
sample dialogue → execute script → create fresh tokenizer → run a test
```

There are two objects worth distinguishing: `tokenizer_script["tokenizer"]` uses the vocabulary built from the tiny dataset, while the `tokenizer` fixture uses the manually specified IDs above.

## Test 1: loading dialogue and splitting punctuation

```python
def test_loads_dialogue_and_splits_punctuation(tokenizer_script):
    assert tokenizer_script["text"] == "Hello, world!\nHello, café!"
    assert tokenizer_script["preprocessed"] == [
        "Hello",
        ",",
        "world",
        "!",
        "Hello",
        ",",
        "café",
        "!",
    ]
```

The first assertion checks that the script extracts the two `Line` values and joins them with a newline (`\n`). It would fail if character names were included or the line order changed.

The second checks the entire token list. List equality checks both contents and order. It verifies that punctuation becomes separate tokens, whitespace disappears, repeated words remain in the sequence, and `café` stays intact.

One function contains two assertions, but it is one test case. If the first assertion fails, execution does not reach the second.

## Test 2: vocabulary construction

```python
def test_builds_sorted_unique_vocabulary(tokenizer_script):
    assert tokenizer_script["vocab"] == {
        "!": 0,
        ",": 1,
        "Hello": 2,
        "café": 3,
        "world": 4,
        "<|endoftext|>": 5,
        "<|unk|>": 6,
    }
```

The tiny dataset contains five distinct ordinary tokens. Python's string sorting puts these in the order `!`, `,`, `Hello`, `café`, `world`. Enumeration gives them IDs 0 through 4; the two special tokens receive IDs 5 and 6.

Dictionary equality checks all key-value pairs, so this test catches missing tokens, extra tokens, or incorrect ID assignments. It does not test dictionary insertion order. Repeated occurrences of `Hello` and punctuation do not create extra vocabulary entries.

## Test 3: the complete dataset-to-tokenizer flow

```python
def test_creates_tokenizer_from_dataset(tokenizer_script):
    tokenizer = tokenizer_script["tokenizer"]
    ids = tokenizer.encode(tokenizer_script["text"])
    assert ids == [2, 1, 4, 0, 2, 1, 3, 0]
    assert tokenizer.decode(ids) == "Hello, world! Hello, café!"
```

Here the local variable `tokenizer` is taken directly from the script's results. This function does not request the separate fixture named `tokenizer`.

The expected IDs come from the preceding vocabulary:

```text
Hello , world ! Hello , café !
  2   1   4   0   2   1  3   0
```

This checks that loading, vocabulary creation, instance construction, encoding, and decoding work together on the sample data.

The original newline becomes a space after decoding. Your tokenizer discards original whitespace and reconstructs spacing; it cannot restore the exact original formatting.

## Test 4: nine encoding cases from one function

```python
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello world", [10, 20]),
        ("world Hello world", [20, 10, 20]),
        ("Hello, world!", [10, 30, 20, 40]),
        ("Hello stranger", [10, 60]),
        ("hello", [60]),
        ("<|endoftext|> <|unk|>", [50, 60]),
        ("  Hello\tworld\n", [10, 20]),
        ("", []),
        (" \t\n", []),
    ],
)
def test_encode(tokenizer, text, expected):
    assert tokenizer.encode(text) == expected
```

**Parameterization** runs a test separately for each supplied row. The names `text` and `expected` identify the parameters filled from each pair. The remaining argument, `tokenizer`, comes from its fixture. Each row gets its own pass/fail result. [Parameterization documentation](https://docs.pytest.org/en/stable/how-to/parametrize.html)

For the first row, pytest supplies `text="Hello world"` and `expected=[10, 20]`. For the fourth, it supplies `text="Hello stranger"` and `expected=[10, 60]`. The test body stays the same.

| Row | What it checks |
| --- | --- |
| `Hello world` | Known words map to their exact IDs. |
| `world Hello world` | Word order and repetitions are preserved. |
| `Hello, world!` | Adjacent punctuation is separated and encoded. |
| `Hello stranger` | An unfamiliar word maps to the unknown token ID, 60. |
| `hello` | Lowercase `hello` differs from the vocabulary's `Hello`. |
| The two special tokens | Literal special tokens are recognized. This does not automatically append an end token to other inputs. |
| Leading spaces, tab, newline | Extra whitespace produces no token IDs. |
| Empty string | Empty input produces an empty list. |
| Whitespace-only string | Input containing only whitespace also produces an empty list. |

`\t` represents a tab and `\n` a newline in these Python string literals. `[]` is an empty list.

## Test 5: five decoding cases

```python
@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        ([10, 20], "Hello world"),
        ([20, 10, 20], "world Hello world"),
        ([10, 30, 20, 40], "Hello, world!"),
        ([10, 60, 50], "Hello <|unk|> <|endoftext|>"),
        ([], ""),
    ],
)
def test_decode(tokenizer, ids, expected):
    assert tokenizer.decode(ids) == expected
```

These rows check the reverse operation: IDs become strings. They cover ordinary word spacing, order and repetition, punctuation spacing, special tokens, and empty input.

For `[10, 30, 20, 40]`, the initial joining inside `decode()` produces `Hello , world !`. Its regular expression removes spaces before the comma and exclamation mark, giving `Hello, world!`.

Decoding unknown-token ID 60 returns the literal string `<|unk|>`. The original unfamiliar word was discarded during encoding, so decoding cannot recover it.

## Test 6: an invalid numeric ID should raise an error

```python
def test_decode_unknown_id_raises_key_error(tokenizer):
    with pytest.raises(KeyError):
        tokenizer.decode([999])
```

ID 999 is absent from this fixture's vocabulary. The dictionary lookup inside `decode()` therefore raises `KeyError`.

`pytest.raises(KeyError)` declares that this is the expected outcome. The test passes if the code inside the `with` block raises `KeyError`, and fails if no exception occurs or an unrelated exception occurs. [Expected-exception documentation](https://docs.pytest.org/en/stable/how-to/assert.html#assertions-about-expected-exceptions)

An unknown word during encoding and an unknown ID during decoding behave differently in your skeleton: the former gets a fallback token; the latter raises an error. This test documents the latter behavior without changing it.

## Test 7: encoding and decoding together

```python
def test_round_trip_normalizes_whitespace(tokenizer):
    assert tokenizer.decode(tokenizer.encode("  Hello,\tworld!\n")) == "Hello, world!"
```

Read the nested calls from the inside out:

```python
ids = tokenizer.encode("  Hello,\tworld!\n")  # [10, 30, 20, 40]
result = tokenizer.decode(ids)                # "Hello, world!"
assert result == "Hello, world!"
```

A **round trip** means converting in one direction and then back. This test expects normalized whitespace, rather than exact reproduction of the original string. Testing exact IDs separately is still useful: encoding and decoding could share a matching mistake and nevertheless pass a round-trip check.

## Test 8: each supported punctuation separator

```python
@pytest.mark.parametrize("punctuation", list(",.:;?_!\"()'") + ["--"])
def test_encode_splits_each_supported_punctuation(tokenizer_script, punctuation):
    tokenizer = tokenizer_script["SimpleTokenizer"](
        {"Hello": 0, punctuation: 1, "world": 2, "<|unk|>": 3}
    )
    assert tokenizer.encode(f"Hello{punctuation}world") == [0, 1, 2]
```

`list(...)` turns the string into 11 individual characters: comma, period, colon, semicolon, question mark, underscore, exclamation mark, double quote, opening parenthesis, closing parenthesis, and apostrophe. Inside the double-quoted Python string, `\"` represents a literal double quote; the backslash is not another test character.

Adding `["--"]` appends a twelfth item containing two hyphens. Using a separate list keeps `--` together as one separator. The decorator therefore generates 12 test cases.

For each case, a new tiny vocabulary assigns the selected punctuation ID 1. The dictionary key `punctuation` uses the variable's current value. The f-string inserts that value between the two words:

```text
punctuation = ","   → "Hello,world"   → [0, 1, 2]
punctuation = "?"   → "Hello?world"   → [0, 1, 2]
punctuation = "--"  → "Hello--world"  → [0, 1, 2]
```

No spaces surround the separator, so a tokenizer that only splits on whitespace would fail. This checks encoding for all listed separators; it does not check decoding for every one of them.

## Why pytest reports 32 tests

There are eight test functions in `test_tokenizer.py`, but parameterization expands three of them:

| Test group | Cases |
| --- | ---: |
| Loading and splitting | 1 |
| Vocabulary construction | 1 |
| Dataset-to-tokenizer flow | 1 |
| Encoding table | 9 |
| Decoding table | 5 |
| Invalid ID | 1 |
| Whitespace round trip | 1 |
| Punctuation table | 12 |
| **Tokenizer total** | **31** |
| Existing test in `test_main.py` | 1 |
| **Whole project total** | **32** |

Fixtures are setup, so the three fixture functions do not add three more test cases. A fixture failure can prevent dependent tests from running; pytest reports that as a setup error.

## Running and reading these tests

Run these commands from `USU-LLM-Class-F26`:

```bash
# Full suite, including the project's coverage requirement.
uv run pytest

# Print a separate result for every test case.
uv run pytest -v

# Show coverage and any missed statement lines.
uv run pytest --cov-report=term-missing

# Study only the nine encoding cases, without collecting coverage.
uv run pytest tests/test_tokenizer.py::test_encode -v --no-cov

# Study only the invalid-ID behavior.
uv run pytest tests/test_tokenizer.py::test_decode_unknown_id_raises_key_error -v --no-cov
```

The `::` selects a named test function within the file. The focused examples use `--no-cov` because they intentionally run only part of the suite; judging them against the full project's coverage threshold could be misleading. Full runs still use the configured threshold. [Pytest invocation](https://docs.pytest.org/en/stable/how-to/usage.html), [pytest-cov options](https://pytest-cov.readthedocs.io/en/latest/config.html)

Your `pyproject.toml` includes:

```toml
[tool.pytest.ini_options]
addopts = "--cov=src --cov-fail-under=80"
```

`--cov=src` measures execution of code in `src`. `--cov-fail-under=80` makes coverage below 80% fail the run, even if every assertion passed. These options come from the `pytest-cov` plugin. This configuration does not enable branch coverage. [pytest-cov configuration](https://pytest-cov.readthedocs.io/en/latest/config.html)

When this guide was written, the full run produced **32 passed** and **100% coverage**: all 30 measured statements across `src` executed, including 28 in the tokenizer.

That result does not mean every possible input works correctly. For example, this suite does not test malformed JSON, missing `Line` fields, real file permissions, or every punctuation-decoding case. The mock deliberately replaces real dataset file access. Coverage tells us which statements ran; the chosen inputs and assertions determine which behaviors were checked.

To practice reading one test, take the `("Hello stranger", [10, 60])` row and trace it through `encode()`: split the two words, preserve `Hello`, replace `stranger` with `<|unk|>`, and look up IDs 10 and 60. The assertion then compares that result with the expected list. Every test in this file follows the same basic pattern: provide known inputs, execute real code, and check a specific outcome.
