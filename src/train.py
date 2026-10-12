from src import config

# Llama 3.2 1B shape (public model card values); every figure below is a rough
# ESTIMATE, not a measurement: the two-step probe in M8 measures the real peak.
_NON_EMBEDDING_PARAMS = 973_000_000
_EMBEDDING_PARAMS = 262_668_288
_HIDDEN = 2048
_LAYERS = 16
_VOCAB = 128_256
_LORA_PARAMS_PER_RANK = 44_032 * _LAYERS  # q,k,v,o,gate,up,down in every layer
_ONE_LAYER_WORKING_BYTES_PER_TOKEN = 81_920
_OVERHEAD_GIB = 1.5  # CUDA context and allocator fragmentation
_GIB = 1024**3


def estimate_training_memory_gib(max_tokens, batch_size=1):
    if max_tokens <= 0 or batch_size <= 0:
        raise ValueError("max_tokens and batch_size must be positive")
    tokens = max_tokens * batch_size
    parts = {
        # 4-bit weights, with the embeddings kept in 16 bits
        "weights": (_NON_EMBEDDING_PARAMS * 0.5 + _EMBEDDING_PARAMS * 2) / _GIB,
        # LoRA weights, gradients and Adam state in 32 bits: 16 bytes per parameter
        "lora_and_optimizer": config.LORA_R * _LORA_PARAMS_PER_RANK * 16 / _GIB,
        # gradient checkpointing: one saved input per layer plus one layer's workspace
        "activations": tokens
        * (_HIDDEN * 2 * _LAYERS + _ONE_LAYER_WORKING_BYTES_PER_TOKEN)
        / _GIB,
        # float32 logits, their gradient and the softmax copy
        "logits": tokens * _VOCAB * 4 * 3 / _GIB,
        "overhead": _OVERHEAD_GIB,
    }
    parts["total"] = sum(parts.values())
    return parts


def choose_gpu(required_gib, options=None, usable_fraction=None):
    options = config.GPU_OPTIONS if options is None else options
    fraction = (
        config.GPU_USABLE_FRACTION if usable_fraction is None else usable_fraction
    )
    fitting = [
        (option["usd_per_second"], option["vram_gib"], name)
        for name, option in options.items()
        if option["vram_gib"] * fraction >= required_gib
    ]
    if not fitting:
        raise ValueError(f"no GPU fits {required_gib:.1f} GiB")
    return min(fitting)[2]
