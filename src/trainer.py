from src.config import (
    LORA_ALPHA,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    SEED,
)


def load_model_for_training():
    from unsloth import FastLanguageModel

    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    peft_model = FastLanguageModel.get_peft_model(
        base_model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    return peft_model, tokenizer
