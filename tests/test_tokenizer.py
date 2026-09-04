import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        corpus = {
            "title": "A New Hope",
            "scenes": [
                {"speaker": "Leia", "dialogue": "Hello, Luke!"},
                {"speaker": "Luke", "dialogue": "Hello -- Leia."},
            ],
            "metadata": {"episode": 4, "released": True},
        }
        Path(cls.temp_dir.name, "SW_EpisodeIV_VI.json").write_text(
            json.dumps(corpus), encoding="utf-8"
        )

        project_root = Path(__file__).resolve().parents[1]
        module_path = project_root / "src" / "tokenizer.py"
        if not module_path.exists():
            module_path = project_root / "upload" / "tokenizer.py"

        cls.original_cwd = Path.cwd()
        os.chdir(cls.temp_dir.name)
        try:
            spec = importlib.util.spec_from_file_location(
                "tokenizer_under_test", module_path
            )
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        finally:
            os.chdir(cls.original_cwd)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_extract_text_recursively_collects_only_strings(self):
        value = {
            "name": "Leia",
            "lines": ["Help me", {"target": "Obi-Wan"}, 42, None],
            "active": True,
        }
        self.assertEqual(
            self.module.extract_text(value), ["Leia", "Help me", "Obi-Wan"]
        )

    def test_extract_text_ignores_scalar_non_strings(self):
        for value in (None, 7, 3.14, True):
            with self.subTest(value=value):
                self.assertEqual(self.module.extract_text(value), [])

    def test_vocabulary_contains_corpus_and_special_tokens(self):
        for token in ("Hello", ",", "<|endoftext|>", "<|unk|>"):
            with self.subTest(token=token):
                self.assertIn(token, self.module.vocab)
        self.assertEqual(len(self.module.vocab), len(set(self.module.vocab.values())))

    def test_encode_returns_ids_for_known_words_and_punctuation(self):
        tokenizer = self.module.SimpleTokenizer(self.module.vocab)
        self.assertEqual(
            tokenizer.encode("Hello, Luke!"),
            [
                self.module.vocab["Hello"],
                self.module.vocab[","],
                self.module.vocab["Luke"],
                self.module.vocab["!"],
            ],
        )

    def test_encode_maps_unknown_tokens_to_unknown_id(self):
        tokenizer = self.module.SimpleTokenizer(self.module.vocab)
        self.assertEqual(tokenizer.encode("Chewbacca"), [self.module.vocab["<|unk|>"]])

    def test_decode_removes_spaces_before_punctuation(self):
        tokenizer = self.module.SimpleTokenizer(self.module.vocab)
        ids = [
            self.module.vocab["Hello"],
            self.module.vocab[","],
            self.module.vocab["Luke"],
            self.module.vocab["!"],
        ]
        self.assertEqual(tokenizer.decode(ids), "Hello, Luke!")

    def test_known_text_round_trip(self):
        tokenizer = self.module.SimpleTokenizer(self.module.vocab)
        text = "Hello, Luke!"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
