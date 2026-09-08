import json
import os
import tempfile
import unittest

from src.jsonExtract import extract_text, load_json, preprocess_text, process_json


class TestJsonProcessing(unittest.TestCase):
    def test_load_json(self):
        sample = {"name": "Luke", "age": 19}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample, f)
            file_path = f.name
        try:
            self.assertEqual(load_json(file_path), sample)
        finally:
            os.remove(file_path)

    def test_extract_text_nested_json(self):
        sample = {
            "character": "Luke",
            "details": {"home": "Tatooine", "age": 19},
            "friends": ["Leia", "Han", 42],
        }
        self.assertEqual(extract_text(sample), ["Luke", "Tatooine", "Leia", "Han"])

    def test_extract_text_ignores_non_strings(self):
        self.assertEqual(
            extract_text({"number": 123, "flag": True, "nothing": None}), []
        )

    def test_preprocess_text(self):
        self.assertEqual(
            preprocess_text("Hello, Luke! How are you?"),
            ["Hello", ",", "Luke", "!", "How", "are", "you", "?"],
        )

    def test_preprocess_text_double_dash(self):
        self.assertEqual(preprocess_text("hello--there"), ["hello", "--", "there"])

    def test_process_json(self):
        sample = {"speaker": "Luke", "line": "Hello, Leia!"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample, f)
            file_path = f.name
        try:
            all_text, tokens = process_json(file_path)
            self.assertEqual(all_text, "Luke Hello, Leia!")
            self.assertEqual(tokens, ["Luke", "Hello", ",", "Leia", "!"])
        finally:
            os.remove(file_path)


if __name__ == "__main__":
    unittest.main()
