FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/outputs && chmod -R 777 /app/outputs

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
    huggingface_hub \
    jupyter

RUN apt-get update && apt-get install -y git cmake build-essential && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp
RUN pip3 install -r /opt/llama.cpp/requirements.txt
RUN cmake -B /opt/llama.cpp/build -S /opt/llama.cpp -DGGML_CUDA=OFF && \
    cmake --build /opt/llama.cpp/build --config Release -j
RUN pip3 install -r /opt/llama.cpp/requirements.txt
RUN pip3 uninstall -y torchvision torchaudio

COPY src/ src/
COPY data/ data/

CMD ["python3", "-u", "-m", "src.trainer"]
