# Assignment 4: local GPU experiments

## Final local FP16 summary

The completed local runs used the Quadro T2000 (4096 MiB physical VRAM),
`usu-llm-a4-local`, Python 3.14.2, PyTorch 2.14.0+cu130, Transformers 5.17.0,
and CUDA build 13.0. Both checkpoints used FP16 without quantization or CPU
inference offloading, with prompt `Once upon a time`, `max_new_tokens=50`, and
temperature 0.7.

| Checkpoint | Parameters | Result | Allocated after load | Reserved after load | Peak allocated |
|---|---:|---|---|---|---|
| meta-llama/Llama-3.2-1B | 1,235,814,400 | SUCCESS: parameters and buffers on cuda:0; generation completed | 2,472,678,400 bytes (~2.30 GiB) | 2,474,639,360 bytes (~2.30 GiB) | 2,489,443,328 bytes (~2.32 GiB) |
| meta-llama/Llama-3.2-3B | 3,212,749,824 | FAILURE: CUDA OOM during full-model transfer to cuda:0; generation never reached | Not reached | Not reached | 3,707,936,768 bytes (~3.45 GiB; partial transfer) |

- **Largest tested successful non-quantized local checkpoint:** Llama 3.2 1B.
- **First tested larger non-quantized checkpoint that failed:** Llama 3.2 3B.
- **Failure mechanism:** `torch.OutOfMemoryError` while moving the full FP16
  model to `cuda:0`, due to insufficient GPU VRAM.

The [1B transcript](local_gpu_20260925T072406_381609Z/llama_1b.txt) records
`RESULT: SUCCESS`, and [host nvidia-smi evidence](local_gpu_20260925T072406_381609Z/nvidia_smi_llama_1b.txt)
was captured during the post-load pause. The student also confirmed taking a
manual screenshot; its file path has not been supplied. The
[3B transcript](local_gpu_20260925T072406_381609Z/llama_3b.txt) records the OOM.
The timestamped records below preserve the complete observations and evidence links.

## Earlier baseline supplied by the student

This is historical, user-reported context, not newly captured evidence.

| Field | User-reported value |
|---|---|
| Timestamp | Not supplied |
| Provider/environment | Local; Ubuntu 24.04 |
| GPU | NVIDIA Quadro T2000 |
| Physical VRAM | 4096 MiB |
| Driver | 580.178.04 |
| Docker image | usu-llm-a4-local |
| Python | 3.14; patch version not supplied |
| PyTorch | 2.14.0+cu130 |
| Transformers | 5.17.0 |
| CUDA build | 13.0, indicated by cu130 |
| Exact model ID | meta-llama/Llama-3.2-1B |
| Parameters | 1,235,814,400 |
| Precision / quantization | FP16 / none |
| Placement / CPU offloading | Full cuda:0 / no |
| Prompt | Once upon a time |
| max_new_tokens / temperature | 50 / 0.7 |
| Model load / generation | Successful / completed |
| Allocated after load | About 2.30 GiB |
| Reserved after load | Not supplied |
| Peak allocated | About 2.32 GiB |
| Outcome / failure reason | Reported success / none |
| Representative generated output | Not supplied |

## Historical preflight observation and Docker blocker

This section describes the earlier preparation stage, before the completed runs.
Docker access no longer blocks the results reported above.

The genuine host preflight is saved in [local_gpu_preflight.txt](local_gpu_preflight.txt).
It shows the GPU and desktop processes only; it is **not model-loaded evidence**.
Host resource inspection found about 25 GiB available RAM and 165 GiB available
on the Hugging Face cache filesystem.

At preflight, the Docker socket was owned by `root:docker` with mode `srw-rw----`;
the user was not in the docker group. Direct Docker access returned:

```text
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

The noninteractive sudo attempt returned:

```text
sudo: a password is required
```

Neither historical error was a model-memory failure. Subsequent runs accessed
both Hugging Face checkpoints: 1B completed inference, and 3B failed during GPU
transfer with a CUDA OOM.

## Reproduce and capture evidence

From the repository root, in your own interactive terminal:

```bash
python3 docs/run_assignment4_local_gpu.py
```

Enter your sudo password only at the terminal's normal sudo prompt. Do not share
it with an assistant. Run the script as your normal user; it elevates only Docker
and uses your UID/GID in the container. The existing image is used, with the
instrumented inference script mounted read-only. No image rebuild is needed.
The exact command and image ID are recorded for every executed attempt.

The runner uses your existing `$HOME/.cache/huggingface` at `/hf-cache` for stored
authentication and persistent downloads. It never reads or logs token values.
Weights remain outside the repository and Docker image.

Progression follows the student's chosen method: 1B baseline, then official
Llama 3.2 3B, then Meta-Llama-3-8B only if 3B succeeds. Every run requests FP16,
no quantization, no CPU offloading, full cuda:0 placement, the same prompt, 50
new tokens, and temperature 0.7. CPU deserialization is a loading step; it is
not inference offloading. Each attempt uses a fresh container. The runner stops
on a failure so its cause can be examined before choosing another checkpoint.
Access/download failures must be resolved before claiming a resource limit.
There is no fixed random seed; generated text can differ between runs.

At the post-load pause, the runner immediately executes host `nvidia-smi`, saves
its real output, and displays **SCREENSHOT NOW**. At that moment, capture the
terminal showing the GPU name, memory use, and active Python process. Alternatively,
run `nvidia-smi` in another terminal and capture it **before the 30-second pause
ends**. No screenshot capture utility was found during the initial inspection.
The student subsequently confirmed taking a manual screenshot of the 1B run;
its file path has not been supplied.

Evidence is stored in `docs/local_gpu_<UTC timestamp>/`, with transcripts for all
attempts, pre-run GPU state, and post-load `nvidia_smi_<model>.txt` when the pause
is reached. Save manual screenshots in that same directory. Timestamped model
records are appended to this document, including failures and missing metrics.
A failed GPU transfer has no valid after-load measurement; final/peak partial
allocations are reported separately. A forced process kill may prevent final
metrics; the Docker exit code and raw transcript remain available.

## Interpretation

The observed local FP16 bracket is **1B success / 3B failure**. Llama 3.2 1B
completed generation with all model parameters and buffers on `cuda:0`, without
quantization or CPU inference offloading. Llama 3.2 3B failed while transferring
the complete FP16 model to `cuda:0`; generation was never reached. Thus, 1B is
the **largest tested successful non-quantized checkpoint**, not a claim about
the largest possible model or every untested intermediate checkpoint.

The 3B checkpoint has 3,212,749,824 parameters (~3.21B). FP16 parameter storage
alone requires `3,212,749,824 × 2 = 6,425,499,648` bytes, or **~5.98 GiB**,
already exceeding the **~3.62 GiB CUDA-visible capacity** reported by PyTorch.
The OOM occurred on an additional 48 MiB allocation, with only 6.81 MiB free and
about 3.45 GiB allocated by PyTorch. This is fundamentally a **VRAM capacity
issue**, not merely allocator fragmentation; even ideal allocation cannot fit
the full FP16 parameter storage in that capacity. The physical 4096 MiB device
capacity and PyTorch's reported CUDA-visible capacity are distinct observations.

PyTorch allocated memory counts live tensor allocations; reserved memory includes
allocator-held space. Host `nvidia-smi` includes broader GPU usage, so these
numbers should not be treated as interchangeable. Peak measurements start before
model loading. No quantized experiment is included or implemented.

## Validation

The preparation checks below were recorded previously; they were not rerun for
this documentation-only update.

- `uv run --offline ruff check .`: passed.
- `uv run --offline ruff format --check .`: passed (18 files).
- `uv run --offline pytest`: 300 passed; overall coverage 82.67%.
- Isolated runner checks passed for success, post-load evidence capture, OOM
  recording, and unavailable metrics. Test fixtures existed only in a temporary
  directory and were deleted; they are not experiment evidence.
- Actual local GPU execution is now validated by the timestamped records: 1B
  completed generation successfully, and 3B produced a CUDA OOM during full-model
  transfer before generation. The 1B host `nvidia-smi` evidence was saved, and
  the student confirmed taking a manual screenshot.
- This update changes documentation only. No new experiments, commits, or pushes
  were performed; the raw timestamped records and evidence are preserved.

## 2026-09-25T07:24:06.411734+00:00 — meta-llama/Llama-3.2-1B

| Field | Observed value |
|---|---|
| Timestamp (host UTC) | 2026-09-25T07:24:06.411734+00:00 |
| Provider/environment | Local |
| GPU / physical VRAM / driver (nvidia-smi) | Quadro T2000, 4096 MiB, 580.178.04 |
| Docker image | usu-llm-a4-local (sha256:345a71aeba978db2731fe884139570b04e5e241af09c0b0a27d79de3830c8101) |
| Python version | 3.14.2 |
| PyTorch version | 2.14.0+cu130 |
| Transformers version | 5.17.0 |
| CUDA build | 13.0 |
| Exact model ID | meta-llama/Llama-3.2-1B |
| Parameter count | 1,235,814,400 |
| Precision | FP16 (requested) |
| Quantization | none |
| CPU offloading | no; CPU deserialization precedes full GPU transfer |
| Parameter devices | ['cuda:0'] |
| Buffer devices | ['cuda:0'] |
| Prompt | Once upon a time |
| max_new_tokens | 50 |
| Temperature | 0.7 (sampling; no fixed seed) |
| Model load result | Full GPU load succeeded |
| Generation result | Completed |
| Allocated after load (bytes) | 2472678400 |
| Reserved after load (bytes) | 2474639360 |
| Peak allocated (bytes) | 2489443328 |
| Final allocated (bytes) | 2481199616 |
| Final reserved (bytes) | 2501902336 |
| Outcome | SUCCESS |
| Docker exit code | 0 |
| Exact failure reason | None |
| nvidia-smi evidence | [nvidia_smi_llama_1b.txt](local_gpu_20260925T072406_381609Z/nvidia_smi_llama_1b.txt) |

Raw transcript and command: [llama_1b.txt](local_gpu_20260925T072406_381609Z/llama_1b.txt).

Representative generated output:

```text
Once upon a time, there was a young man named Andrew. He was born in a small village in India and spent his early years growing up in the mountains. One day, he decided to take a trip to the city to see what it was like. When he
```

## 2026-09-25T07:24:51.468300+00:00 — meta-llama/Llama-3.2-3B

| Field | Observed value |
|---|---|
| Timestamp (host UTC) | 2026-09-25T07:24:51.468300+00:00 |
| Provider/environment | Local |
| GPU / physical VRAM / driver (nvidia-smi) | Quadro T2000, 4096 MiB, 580.178.04 |
| Docker image | usu-llm-a4-local (sha256:345a71aeba978db2731fe884139570b04e5e241af09c0b0a27d79de3830c8101) |
| Python version | 3.14.2 |
| PyTorch version | 2.14.0+cu130 |
| Transformers version | 5.17.0 |
| CUDA build | 13.0 |
| Exact model ID | meta-llama/Llama-3.2-3B |
| Parameter count | 3,212,749,824 |
| Precision | FP16 (requested) |
| Quantization | none |
| CPU offloading | no; CPU deserialization precedes full GPU transfer |
| Parameter devices | Not measured / not reached |
| Buffer devices | Not measured / not reached |
| Prompt | Once upon a time |
| max_new_tokens | 50 |
| Temperature | 0.7 (sampling; no fixed seed) |
| Model load result | Failed / not completed |
| Generation result | Failed or not reached; see transcript |
| Allocated after load (bytes) | Not measured / not reached |
| Reserved after load (bytes) | Not measured / not reached |
| Peak allocated (bytes) | 3707936768 |
| Final allocated (bytes) | 3707936768 |
| Final reserved (bytes) | 3730833408 |
| Outcome | FAILURE |
| Docker exit code | 1 |
| Exact failure reason | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB. GPU 0 has a total capacity of 3.62 GiB of which 6.81 MiB is free. Process 43724 has 3.53 GiB memory in use. Of the allocated memory 3.45 GiB is allocated by PyTorch, and 21.84 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) |
| nvidia-smi evidence | Not captured: model did not reach the post-load pause. |

Raw transcript and command: [llama_3b.txt](local_gpu_20260925T072406_381609Z/llama_3b.txt).

No successful generated output recorded.

## 2026-09-27T09:38:35.516783+00:00 — meta-llama/Llama-3.2-1B

| Field | Observed value |
|---|---|
| Timestamp (host UTC) | 2026-09-27T09:38:35.516783+00:00 |
| Provider/environment | Local |
| GPU / physical VRAM / driver (nvidia-smi) | Quadro T2000, 4096 MiB, 580.178.04 |
| Docker image | usu-llm-a4-local (sha256:497c7cf811951e0b0e36fd3542e054f7a23a12628a86ea51fd9a754cc817519b) |
| Python version | 3.14.2 |
| PyTorch version | 2.14.0+cu130 |
| Transformers version | 5.17.0 |
| CUDA build | 13.0 |
| Exact model ID | meta-llama/Llama-3.2-1B |
| Parameter count | 1,235,814,400 |
| Precision | FP16 (requested) |
| Quantization | none |
| CPU offloading | no; CPU deserialization precedes full GPU transfer |
| Parameter devices | ['cuda:0'] |
| Buffer devices | ['cuda:0'] |
| Prompt | Once upon a time |
| max_new_tokens | 50 |
| Temperature | 0.7 (sampling; no fixed seed) |
| Model load result | Full GPU load succeeded |
| Generation result | Completed |
| Allocated after load (bytes) | 2472678400 |
| Reserved after load (bytes) | 2474639360 |
| Peak allocated (bytes) | 2489443328 |
| Final allocated (bytes) | 2481199616 |
| Final reserved (bytes) | 2501902336 |
| Outcome | SUCCESS |
| Docker exit code | 0 |
| Exact failure reason | None |
| nvidia-smi evidence | [nvidia_smi_llama_1b.txt](local_gpu_20260927T093835_486581Z/nvidia_smi_llama_1b.txt) |

Raw transcript and command: [llama_1b.txt](local_gpu_20260927T093835_486581Z/llama_1b.txt).

Representative generated output:

```text
Once upon a time, a young boy, a farmer’s son, was going through a very hard time. He was having a lot of problems at home, and he was also going through a lot of trouble with his friends. He was trying to make his way in
```

## 2026-09-27T09:39:36.234235+00:00 — meta-llama/Llama-3.2-3B

| Field | Observed value |
|---|---|
| Timestamp (host UTC) | 2026-09-27T09:39:36.234235+00:00 |
| Provider/environment | Local |
| GPU / physical VRAM / driver (nvidia-smi) | Quadro T2000, 4096 MiB, 580.178.04 |
| Docker image | usu-llm-a4-local (sha256:497c7cf811951e0b0e36fd3542e054f7a23a12628a86ea51fd9a754cc817519b) |
| Python version | 3.14.2 |
| PyTorch version | 2.14.0+cu130 |
| Transformers version | 5.17.0 |
| CUDA build | 13.0 |
| Exact model ID | meta-llama/Llama-3.2-3B |
| Parameter count | 3,212,749,824 |
| Precision | FP16 (requested) |
| Quantization | none |
| CPU offloading | no; CPU deserialization precedes full GPU transfer |
| Parameter devices | Not measured / not reached |
| Buffer devices | Not measured / not reached |
| Prompt | Once upon a time |
| max_new_tokens | 50 |
| Temperature | 0.7 (sampling; no fixed seed) |
| Model load result | Failed / not completed |
| Generation result | Failed or not reached; see transcript |
| Allocated after load (bytes) | Not measured / not reached |
| Reserved after load (bytes) | Not measured / not reached |
| Peak allocated (bytes) | 3707936768 |
| Final allocated (bytes) | 3707936768 |
| Final reserved (bytes) | 3730833408 |
| Outcome | FAILURE |
| Docker exit code | 1 |
| Exact failure reason | torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB. GPU 0 has a total capacity of 3.62 GiB of which 6.81 MiB is free. Process 76348 has 3.53 GiB memory in use. Of the allocated memory 3.45 GiB is allocated by PyTorch, and 21.84 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) |
| nvidia-smi evidence | Not captured: model did not reach the post-load pause. |

Raw transcript and command: [llama_3b.txt](local_gpu_20260927T093835_486581Z/llama_3b.txt).

No successful generated output recorded.

## 2026-09-27T09:42:56.585827+00:00 — meta-llama/Meta-Llama-3.1-8B

| Field | Observed value |
|---|---|
| Timestamp (host UTC) | 2026-09-27T09:42:56.585827+00:00 |
| Provider/environment | Local |
| GPU / physical VRAM / driver (nvidia-smi) | Quadro T2000, 4096 MiB, 580.178.04 |
| Docker image | usu-llm-a4-local (sha256:497c7cf811951e0b0e36fd3542e054f7a23a12628a86ea51fd9a754cc817519b) |
| Python version | 3.14.2 |
| PyTorch version | 2.14.0+cu130 |
| Transformers version | 5.17.0 |
| CUDA build | 13.0 |
| Exact model ID | meta-llama/Meta-Llama-3.1-8B |
| Parameter count | Not measured / not reached |
| Precision | FP16 (requested) |
| Quantization | none |
| CPU offloading | no; CPU deserialization precedes full GPU transfer |
| Parameter devices | Not measured / not reached |
| Buffer devices | Not measured / not reached |
| Prompt | Once upon a time |
| max_new_tokens | 50 |
| Temperature | 0.7 (sampling; no fixed seed) |
| Model load result | Failed / not completed |
| Generation result | Failed or not reached; see transcript |
| Allocated after load (bytes) | Not measured / not reached |
| Reserved after load (bytes) | Not measured / not reached |
| Peak allocated (bytes) | 0 |
| Final allocated (bytes) | 0 |
| Final reserved (bytes) | 0 |
| Outcome | FAILURE |
| Docker exit code | 1 |
| Exact failure reason | OSError: You are trying to access a gated repo. |
| nvidia-smi evidence | Not captured: model did not reach the post-load pause. |

Raw transcript and command: [llama_8b.txt](local_gpu_20260927T093835_486581Z/llama_8b.txt).

No successful generated output recorded.
