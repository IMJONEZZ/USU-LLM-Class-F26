"""Fine tunes Llama 3.2 1B Instruct to answer the homework's trick questions.

The data comes from ``src/dataset.py``, which builds questions in the same
families as the ten I'm graded on without using the ten themselves. This file
is only the training half: load the model, put a LoRA on it, run SFT over the
chat-formatted rows, then answer questions with it.

The model is the Instruct checkpoint, not base, so every row goes through the
Llama 3 chat template and training masks everything except the assistant turn.
Training on the questions as well would spend most of the gradient teaching it
to write questions, which I don't need.

Unsloth only imports where there's a CUDA device, so it gets imported inside
the functions instead of at the top. Everything above ``load_base_model`` runs
on CPU, which is what lets the tests cover it without a GPU.
"""

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CHAT_TEMPLATE = "llama-3.1"
MAX_SEQ_LENGTH = 1024

# Unsloth's notebook defaults. I changed the epochs and nothing else, because
# their docs say 1 to 3 and I have far less data than their example does.
LORA_RANK = 16
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 3

# The two header markers the Llama 3 template puts in front of a user turn and
# an assistant turn. `train_on_responses_only` splits on these to work out
# which tokens to mask, so if the chat template ever changes these go with it.
INSTRUCTION_PART = "<|start_header_id|>user<|end_header_id|>\n\n"
RESPONSE_PART = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def load_base_model(model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH):
    """Load the checkpoint and give its tokenizer the Llama 3 chat template."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
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


def render(rows, tokenizer):
    """Run every conversation through the chat template.

    `tokenize=False` because SFTTrainer wants text and does its own tokenizing.
    """
    return [
        tokenizer.apply_chat_template(row["messages"], tokenize=False) for row in rows
    ]


def finetune(
    model,
    tokenizer,
    rows,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    output_dir="outputs",
    max_seq_length=MAX_SEQ_LENGTH,
):
    """Run SFT over the chat rows and hand back the trainer stats."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    dataset = Dataset.from_dict({"text": render(rows, tokenizer)})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=5,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
            dataset_text_field="text",
            max_seq_length=max_seq_length,
        ),
    )

    # Mask the prompt so the loss only lands on what the assistant said.
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )
    return trainer.train()


def answer(model, tokenizer, questions, max_new_tokens=200, device="cuda"):
    """Ask the model each question and return what it said.

    Greedy decoding, no sampling. The notebook has to come out the same every
    time it runs or the answers I submit aren't the answers anyone else sees.
    """
    import torch

    out = []
    for question in questions:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        ids = {k: v.to(device) for k, v in ids.items()}
        with torch.no_grad():
            generated = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = generated[0][ids["input_ids"].shape[1] :]
        out.append(tokenizer.decode(reply, skip_special_tokens=True).strip())
    return out


# The reason this file has save functions at all is that assignment 5 and
# assignment 6 both ended with a trained model sitting in a container that then
# shut down. Both of those are gone. Training writes nothing to disk unless you
# tell it to, and I didn't, twice.

GGUF_QUANTIZATION = "q4_k_m"


def save_adapter(model, tokenizer, path):
    """Write the LoRA weights and the tokenizer, and nothing else.

    This is tens of megabytes because it's only the adapter, not the base model
    underneath it. It's the cheap insurance I should have taken out two
    assignments ago.
    """
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def save_gguf(model, tokenizer, path, quantization=GGUF_QUANTIZATION):
    """Merge the adapter into the base and write a quantized GGUF.

    GGUF is what llama.cpp loads, and `q4_k_m` takes the 1B from about 2.5GB
    down to roughly 800MB, which is what makes serving it on a CPU container
    reasonable. Unsloth does the merge, the conversion and the quantization in
    this one call.
    """
    model.save_pretrained_gguf(path, tokenizer, quantization_method=quantization)
    return path


def save_merged(model, tokenizer, path):
    """Fall back to 16 bit safetensors when the GGUF conversion won't build.

    The GGUF path compiles llama.cpp inside the container. If that breaks I
    still want the merged weights out of the container, because re-running the
    training is the expensive part and merging is not.
    """
    model.save_pretrained_merged(path, tokenizer, save_method="merged_16bit")
    return path
