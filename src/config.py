DATA_PATH = "data/resumes.csv"
SEED = 42
N_RESUMES = 1000
SPLIT_FRACTIONS = (0.7, 0.2, 0.1)
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "unsloth/Llama-3.2-1B"
MAX_SEQ_LENGTH = 8192
LORA_R = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
