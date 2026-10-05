def make_generate_fn(model, tokenizer, max_new_tokens=16):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate_fn(prompts):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        new_token_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        completions = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        return [completion.strip().split("\n")[0].strip() for completion in completions]

    return generate_fn
