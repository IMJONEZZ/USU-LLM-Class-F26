from unittest.mock import MagicMock

import torch

from src.generation import make_generate_fn


class FakeBatch(dict):
    def to(self, device):
        return self


def make_fakes(decoded_texts):
    tokenizer = MagicMock()
    tokenizer.return_value = FakeBatch(input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]))
    tokenizer.batch_decode.return_value = decoded_texts
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3, 9, 9], [4, 5, 6, 8, 8]])
    return model, tokenizer


def test_generate_fn_returns_first_line_of_each_completion():
    model, tokenizer = make_fakes(["HR\n\n### Instruction:\nWhat", "  BPO  "])

    generate_fn = make_generate_fn(model, tokenizer)

    assert generate_fn(["p1", "p2"]) == ["HR", "BPO"]


def test_generate_fn_decodes_only_the_newly_generated_tokens():
    model, tokenizer = make_fakes(["HR", "BPO"])

    make_generate_fn(model, tokenizer)(["p1", "p2"])

    decoded_ids = tokenizer.batch_decode.call_args.args[0]
    assert torch.equal(decoded_ids, torch.tensor([[9, 9], [8, 8]]))
    assert tokenizer.batch_decode.call_args.kwargs["skip_special_tokens"] is True


def test_generate_fn_tokenizes_prompts_as_one_padded_batch():
    model, tokenizer = make_fakes(["HR", "BPO"])

    make_generate_fn(model, tokenizer)(["p1", "p2"])

    tokenizer.assert_called_once_with(["p1", "p2"], return_tensors="pt", padding=True)


def test_generate_fn_uses_greedy_decoding_and_the_token_limit():
    model, tokenizer = make_fakes(["HR", "BPO"])

    make_generate_fn(model, tokenizer, max_new_tokens=5)(["p1", "p2"])

    assert model.generate.call_args.kwargs["do_sample"] is False
    assert model.generate.call_args.kwargs["max_new_tokens"] == 5


def test_make_generate_fn_pads_on_the_left_and_fills_missing_pad_token():
    model, tokenizer = make_fakes(["HR", "BPO"])

    make_generate_fn(model, tokenizer)

    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<eos>"
