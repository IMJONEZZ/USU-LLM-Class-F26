"""Fine tunes Llama 3.2 1B on the Star Wars dialogue and measures perplexity.

The split comes from the data loader I wrote in assignment 2. That loader cuts
the token stream in half before it builds any windows, so the validation half
holds tokens the training half never sees. I train on one and score the other,
which is the only way the before and after numbers mean anything.

Unsloth only imports where there's a CUDA device, so it gets imported inside
`finetune` rather than at the top of the file. Everything else in here runs on
CPU, which is what lets the tests cover it without a GPU.
"""

import math

from src.dataloader import END_OF_TEXT, join_lines, load_lines, train_val_datasets

BASE_MODEL = "meta-llama/Llama-3.2-1B"
MAX_SEQ_LENGTH = 512

# Straight off Unsloth's Llama 3.2 notebook. I'm not tuning the tuner on my
# first run, I want a baseline I can argue with later.
LORA_RANK = 16
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_STEPS = 60


def build_splits(tokenizer, path=None, max_length=MAX_SEQ_LENGTH, val_fraction=0.1):
    """Return the train and validation token streams as (train_ids, val_ids).

    Uses assignment 2's `train_val_datasets`, which splits the tokens first and
    windows each half separately. Stride equals max_length so the windows don't
    overlap, which matters here because overlapping windows would put the same
    tokens in front of the model more than once.
    """
    # Assignment 2 joined lines on "<|endoftext|>" because that was a real token
    # in my own tokenizer. Llama's end of text marker is "<|end_of_text|>", so
    # mine would go in as plain characters and the model would learn to type it
    # out. Using whatever EOS the tokenizer actually has avoids that.
    separator = getattr(tokenizer, "eos_token", None) or END_OF_TEXT
    lines = load_lines(path) if path else load_lines()
    text = join_lines(lines, separator=separator)
    train_ds, val_ds = train_val_datasets(
        text,
        tokenizer,
        max_length=max_length,
        stride=max_length,
        val_fraction=val_fraction,
    )
    return train_ds.token_ids, val_ds.token_ids


def windows(token_ids, max_length=MAX_SEQ_LENGTH):
    """Chop a token stream into non overlapping chunks of max_length."""
    return [
        token_ids[i : i + max_length]
        for i in range(0, len(token_ids) - max_length + 1, max_length)
    ]


def perplexity(model, token_ids, max_length=MAX_SEQ_LENGTH, device=None):
    """Average loss over the stream, exponentiated.

    Perplexity is just exp of the mean cross entropy, so a model that assigns
    higher probability to the real next token scores lower. Both the before and
    after runs see the same chunks in the same order, so the two numbers are
    comparable even though the absolute value only means something relative to
    this corpus.
    """
    import torch

    chunks = windows(token_ids, max_length)
    if not chunks:
        raise ValueError(
            f"need at least {max_length} tokens to score, got {len(token_ids)}"
        )

    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            ids = chunk.unsqueeze(0) if hasattr(chunk, "unsqueeze") else chunk
            if device is not None:
                ids = ids.to(device)
            # Passing labels makes the model return the shifted cross entropy
            # itself, so I'm not re-implementing the off by one.
            out = model(input_ids=ids, labels=ids)
            total_loss += float(out.loss)

    mean_loss = total_loss / len(chunks)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "chunks": len(chunks),
        "tokens": len(chunks) * max_length,
    }


def load_base_model(model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH):
    """Load the base model and tokenizer through Unsloth."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    return model, tokenizer


def attach_lora(model, rank=LORA_RANK):
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )


def finetune(
    model,
    tokenizer,
    train_ids,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    output_dir="outputs",
):
    """Run the training loop and hand back the trainer stats.

    The training text is the train half decoded back from its token ids. I split
    on tokens rather than on lines so the boundary lands exactly where the data
    loader put it, and decoding is lossless for this tokenizer so nothing moves.
    """
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    chunks = windows(train_ids, MAX_SEQ_LENGTH)
    texts = [tokenizer.decode(c) for c in chunks]
    dataset = Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
        ),
    )
    return trainer.train()


def summarize(before, after, train_stats=None):
    """Put the before and after numbers next to each other.

    The percentage is the bit I care about. An absolute perplexity on a corpus
    this specific doesn't mean much on its own, but the change between two runs
    scored on identical chunks does.
    """
    drop = before["perplexity"] - after["perplexity"]
    result = {
        "perplexity_before": before["perplexity"],
        "perplexity_after": after["perplexity"],
        "perplexity_drop": drop,
        "perplexity_drop_pct": 100.0 * drop / before["perplexity"],
        "loss_before": before["loss"],
        "loss_after": after["loss"],
        "eval_chunks": before["chunks"],
        "eval_tokens": before["tokens"],
    }
    if train_stats is not None:
        result["train_runtime_s"] = train_stats.get("train_runtime")
        result["train_loss"] = train_stats.get("train_loss")
    return result
