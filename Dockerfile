FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130

RUN pip3 install --no-cache-dir \
    transformers==5.5.0 \
    unsloth==2026.6.2 \
    unsloth_zoo==2026.6.2 \
    trl \
    peft \
    datasets \
    accelerate \
    bitsandbytes \
    huggingface_hub

COPY src/ src/
COPY data/ data/

CMD ["python3", "-u", "-m", "src.trainer"]
