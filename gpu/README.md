# Unquantized GPU experiments

Run commands from the repository root in PowerShell. All inference runs in Docker,
using BF16 floating-point weights, no quantization, and the entire model on GPU 0.
An attempt succeeds only if loading **and text generation** succeed.

## Build

```powershell
docker build -t llm-gpu ./gpu
New-Item -ItemType Directory -Force gpu-results
```

## Sweep model sizes

```powershell
docker run --rm --gpus all --name llm-gpu -v hf-cache:/root/.cache/huggingface --mount "type=bind,source=$($PWD.Path)/gpu-results,target=/results" llm-gpu sweep
```

The default list is SmolLM2-360M-Instruct followed by Qwen2.5 Instruct models labeled
0.5B, 1.5B, 3B, 7B, and 14B. These are model-name sizes; the report counts actual
parameters, including when loading fails (using a weightless model skeleton).
Downloads across this list occupy tens of GB in the persistent hf-cache volume.
The 14B model's BF16 weights alone exceed 16 GB; this is an expected failure,
not a recorded result until you actually run it.

Override the list by adding --models and space-separated model IDs, for example:

```powershell
docker run --rm --gpus all --name llm-gpu -v hf-cache:/root/.cache/huggingface --mount "type=bind,source=$($PWD.Path)/gpu-results,target=/results" llm-gpu sweep --models HuggingFaceTB/SmolLM2-360M-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct
```

Each attempt runs in a fresh process to release memory between models. Failures
do not stop the sweep. An oom outcome means a CUDA out-of-memory exception;
error means another failure, such as a download problem, and is not proof a model
cannot fit. If configuration retrieval fails, the parameter count is unknown.
Capacity depends on the prompt, output length, and other GPU applications.
Use --prompt "..." and --max-new-tokens 100 to control the workload consistently.

Each invocation creates a timestamped folder under gpu-results containing:

- summary.md: model IDs, parameter sizes, outcomes, and failure details.
- results.json: machine-readable results, settings, GPU, memory, and generated text.
- Numbered .json and .log files for each attempt.
- .output.txt and .nvidia-smi.txt files for successful attempts.

## Rerun the largest success and capture evidence

Replace YOUR_SWEEP_FOLDER with the timestamp printed by the sweep:

```powershell
docker run --rm -it --gpus all --name llm-gpu -v hf-cache:/root/.cache/huggingface --mount "type=bind,source=$($PWD.Path)/gpu-results,target=/results" llm-gpu run --largest-from /results/YOUR_SWEEP_FOLDER/results.json
```

This selects the highest actual parameter count among successful attempts in that
file. It generates text, saves and prints nvidia-smi **while the model is loaded**,
and waits for Enter. The rerun gets its own folder and can fail if available memory
has changed. No successful models in the input file produces an explicit error.
Pass the same prompt and token limit again if you changed them for the sweep.

To select a particular model yourself:

```powershell
docker run --rm -it --gpus all --name llm-gpu -v hf-cache:/root/.cache/huggingface --mount "type=bind,source=$($PWD.Path)/gpu-results,target=/results" llm-gpu run --model Qwen/Qwen2.5-1.5B-Instruct
```

While paused, open a second terminal and run:

```powershell
docker exec llm-gpu nvidia-smi
```

Take an actual screenshot showing the GPU name, memory usage, and active process.
The saved text file is useful evidence but does not replace the assignment's
screenshot. If WSL's process table is empty, also try nvidia-smi in Windows
PowerShell while the model is loaded. Check the process table before accepting
the screenshot as complete.

Press Enter in the original terminal to release the GPU and remove the container.
Use --no-hold for a noninteractive single run. Run containers one at a time.
The cache volume and host results persist. Peak PyTorch memory excludes some CUDA
overhead and other applications, so it differs from total nvidia-smi usage.

This RTX 5080 setup uses PyTorch's CUDA 12.8 build; see the
[PyTorch Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/).
The larger candidates are from the
[Qwen2.5 model family](https://qwen.readthedocs.io/en/v2.5/getting_started/quickstart.html).
