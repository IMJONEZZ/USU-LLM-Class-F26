import torch
import truststore

truststore.inject_into_ssl()

if torch.cuda.is_available():
    import unsloth  # noqa: F401  (unsloth must be imported before transformers)
