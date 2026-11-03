from src import benchmark


class FakeRunners:
    """Stands in for the Modal class: records every call and returns one record per call."""

    def __init__(self, warm):
        self.calls = []
        self.warm = warm
        self.runners = {m: self.make(m) for m in ("guidance", "coercion")}

    def make(self, method):
        def run(request):
            self.calls.append((method, request))
            return {"method": method, "bucket": "valid", "seconds": 1.0}

        return run

    def run(self, prompts):
        return benchmark.run_benchmark(self.runners, lambda: self.warm, prompts)


PROMPTS = ["first", "second", "third"]


def test_one_record_per_prompt_per_method_in_order_a_then_b():
    fake = FakeRunners({"guidance": 1, "coercion": 1})
    records = fake.run(PROMPTS)["records"]
    assert [(r["prompt_id"], r["method"]) for r in records] == [
        (0, "guidance"), (0, "coercion"),
        (1, "guidance"), (1, "coercion"),
        (2, "guidance"), (2, "coercion"),
    ]  # fmt: skip
    assert [r["prompt"] for r in records[::2]] == PROMPTS


def test_a_cold_container_gets_one_discarded_warmup_per_method():
    fake = FakeRunners({})
    result = fake.run(PROMPTS)
    assert len(result["records"]) == 6
    assert len(result["warmups"]) == 2
    assert fake.calls[:2] == [("guidance", "first"), ("coercion", "first")]
    assert len(fake.calls) == 2 + 6


def test_an_already_warm_container_skips_the_warmup():
    fake = FakeRunners({"guidance": 3, "coercion": 3})
    result = fake.run(PROMPTS)
    assert result["warmups"] == []
    assert len(fake.calls) == 6


def test_only_the_method_that_is_cold_is_warmed_up():
    result = FakeRunners({"guidance": 5}).run(PROMPTS)
    assert [w["method"] for w in result["warmups"]] == ["coercion"]


def record(method, bucket, seconds, tokens):
    return {
        "method": method,
        "bucket": bucket,
        "seconds": seconds,
        "new_tokens": tokens,
    }


def test_summary_math_on_known_numbers():
    records = [
        record("guidance", "valid", 2.0, 10),
        record("guidance", "valid", 4.0, 20),
        record("guidance", "malformed", 6.0, 30),
        record("coercion", "valid", 10.0, 100),
        record("coercion", "truncated", 20.0, 100),
    ]
    summary = benchmark.summarize(records)
    assert summary["guidance"] == {
        "n": 3, "valid": 2, "malformed": 1, "truncated": 0, "valid_rate": 0.67,
        "mean_seconds": 4.0, "median_seconds": 4.0, "mean_tokens": 20.0, "seconds_per_token": 0.2,
    }  # fmt: skip
    assert summary["coercion"] == {
        "n": 2, "valid": 1, "malformed": 0, "truncated": 1, "valid_rate": 0.5,
        "mean_seconds": 15.0, "median_seconds": 15.0, "mean_tokens": 100.0, "seconds_per_token": 0.15,
    }  # fmt: skip


def test_gallery_shows_valid_pages_rendered_and_failures_as_escaped_raw_text(tmp_path):
    good = {
        **record("guidance", "valid", 1.0, 5),
        "prompt_id": 0,
        "prompt": "p",
        "html": "<html>ok</html>",
        "text": "ok",
        "reason": "",
    }
    bad = {**record("coercion", "malformed", 2.0, 5), "prompt_id": 0, "prompt": "p", "html": None, "text": "<script>x</script>", "reason": "Invalid JSON"}  # fmt: skip
    benchmark.write_gallery({"records": [good, bad]}, tmp_path)
    page = (tmp_path / "index.html").read_text()
    assert "<iframe" in page and page.count("<iframe") == 1
    assert "&lt;script&gt;x&lt;/script&gt;" in page and "<script>x" not in page
    assert page.count("<circle") == 2


# ---- the command-line path: summary file and gallery from a results file ----

import json
import runpy
import sys


def results_file(tmp_path):
    def full(method, bucket, html):
        return {
            "prompt_id": 0, "prompt": "p", "method": method, "bucket": bucket, "reason": "",
            "html": html, "text": html or "raw", "seconds": 2.0, "new_tokens": 10,
        }  # fmt: skip

    data = {
        "records": [
            full("guidance", "valid", "<html>ok</html>"),
            full("coercion", "malformed", None),
        ]
    }
    path = tmp_path / "benchmark_t1.json"
    path.write_text(json.dumps(data))
    return path


def test_main_writes_the_summary_json_and_the_gallery(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    benchmark.main(str(results_file(tmp_path)))
    summary = json.loads((tmp_path / "summary_t1.json").read_text())
    assert (
        summary["guidance"]["valid_rate"] == 1.0
        and summary["coercion"]["valid_rate"] == 0.0
    )
    assert (tmp_path / "results/pages/t1/index.html").exists()


def test_running_the_module_as_a_script_summarizes_the_file_it_is_given(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["benchmark", str(results_file(tmp_path))])
    runpy.run_module("src.benchmark", run_name="__main__")
    assert (tmp_path / "summary_t1.json").exists()


def test_results_path_refuses_an_existing_run_unless_asked(tmp_path):
    import pytest

    path = benchmark.results_path(tmp_path, "t1")
    assert path == tmp_path / "benchmark_t1.json"
    assert benchmark.save_results(path, {"records": []}) == str(path)
    assert json.loads(path.read_text()) == {"records": []}
    with pytest.raises(FileExistsError):
        benchmark.results_path(tmp_path, "t1")
    assert benchmark.results_path(tmp_path, "t1", overwrite=True) == path
