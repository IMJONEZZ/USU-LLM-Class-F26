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
LEARNING_RATE = 2e-4
MAX_STEPS = 30
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
EVAL_STEPS = 10
EARLY_STOPPING_PATIENCE = 1
